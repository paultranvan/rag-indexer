import asyncio
import signal
import sys
import time

import aio_pika
import aiohttp
import structlog
from aiormq import AMQPConnectionError
from pydantic import ValidationError
from structlog.contextvars import clear_contextvars, bind_contextvars

from rag_indexer.config import RABBITMQ_URL, HTTP_TIMEOUT, CONCURRENCY, HEALTH_PORT
from rag_indexer.errors import TransientError, FatalError
from rag_indexer.logging import setup_logging
from rag_indexer.metrics import MESSAGES_TOTAL, PROCESSING_DURATION
from rag_indexer.processing import process_message, get_retry_count, next_retry_queue
from rag_indexer.transport import (
    declare_topology,
    publish_to_retry,
    publish_to_dlq,
    start_health_server,
    shutdown_event,
    request_shutdown,
)


async def main():
    setup_logging()
    log = structlog.get_logger()

    try:
        connection = await aio_pika.connect_robust(RABBITMQ_URL)
    except AMQPConnectionError:
        log.error("rabbitmq_connection_failed")
        sys.exit(1)

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, request_shutdown)
    loop.add_signal_handler(signal.SIGINT, request_shutdown)

    async with connection:
        log.info("rabbitmq_connected")
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=CONCURRENCY)

        queue = await declare_topology(channel)

        health_runner = await start_health_server(connection, port=HEALTH_PORT)

        session = aiohttp.ClientSession(timeout=HTTP_TIMEOUT)

        semaphore = asyncio.Semaphore(CONCURRENCY)
        in_flight: set[asyncio.Task] = set()

        async def handle_message(message: aio_pika.IncomingMessage) -> None:
            """Process a single message with full error isolation."""
            async with semaphore:
                try:
                    clear_contextvars()
                    headers = message.headers or {}
                    action = headers.get("action", "unknown")
                    partition = headers.get("partition", "unknown")
                    bind_contextvars(
                        file_id=headers.get("file_id", "unknown"),
                        partition=partition,
                        action=action,
                    )
                    log.info("message_received")

                    async with message.process(ignore_processed=True, requeue=False):
                        retry_count = get_retry_count(message)
                        start_time = time.monotonic()

                        try:
                            await process_message(message, session)
                            # ACK implicit via context manager on success
                            duration = time.monotonic() - start_time
                            MESSAGES_TOTAL.labels(action=action, status="success", partition=partition).inc()
                            PROCESSING_DURATION.labels(action=action).observe(duration)
                            log.info("message_processed", status="success", duration_s=round(duration, 3))

                        except ValidationError as ve:
                            duration = time.monotonic() - start_time
                            MESSAGES_TOTAL.labels(action=action, status="dlq", partition=partition).inc()
                            PROCESSING_DURATION.labels(action=action).observe(duration)
                            log.error("message_invalid_payload", error=str(ve))
                            await publish_to_dlq(channel, message)

                        except TransientError as te:
                            duration = time.monotonic() - start_time
                            PROCESSING_DURATION.labels(action=action).observe(duration)
                            queue_name = next_retry_queue(retry_count)
                            if queue_name:
                                MESSAGES_TOTAL.labels(action=action, status="retry", partition=partition).inc()
                                log.warning("message_retry", attempt=retry_count + 1, error=str(te), queue=queue_name)
                                await publish_to_retry(channel, message, queue_name)
                            else:
                                MESSAGES_TOTAL.labels(action=action, status="dlq", partition=partition).inc()
                                log.warning("message_dlq_exhausted", attempts=retry_count)
                                await publish_to_dlq(channel, message)

                        except FatalError as fe:
                            duration = time.monotonic() - start_time
                            MESSAGES_TOTAL.labels(action=action, status="dlq", partition=partition).inc()
                            PROCESSING_DURATION.labels(action=action).observe(duration)
                            log.error("message_fatal", error=str(fe))
                            await publish_to_dlq(channel, message)

                except Exception:
                    # Final safety net -- no exception may escape handle_message
                    log.exception("message_unhandled_error")
                    try:
                        await message.nack(requeue=False)
                    except Exception:
                        pass

        async def on_message(message: aio_pika.IncomingMessage) -> None:
            task = asyncio.create_task(handle_message(message))
            in_flight.add(task)
            task.add_done_callback(in_flight.discard)

        consumer_tag = None
        try:
            consumer_tag = await queue.consume(on_message)
            log.info("consumer_started", concurrency=CONCURRENCY)
            await shutdown_event.wait()
        finally:
            # --- Shutdown drain ---
            log.warning("shutdown_drain_start", in_flight=len(in_flight))

            # Cancel consumer -- stop receiving new messages
            if consumer_tag is not None:
                try:
                    await queue.cancel(consumer_tag)
                except Exception:
                    pass  # Channel may already be closed

            # Drain in-flight tasks with 10s timeout
            if in_flight:
                log.info("draining_tasks", count=len(in_flight))
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*in_flight, return_exceptions=True),
                        timeout=10,
                    )
                    log.info("drain_complete")
                except asyncio.TimeoutError:
                    remaining = len([t for t in in_flight if not t.done()])
                    log.warning("drain_timeout", remaining=remaining)
                    for task in list(in_flight):
                        if not task.done():
                            task.cancel()
                    # Wait briefly for cancellation to propagate
                    await asyncio.gather(*in_flight, return_exceptions=True)
                    log.info("drain_forced_complete")
            else:
                log.info("drain_complete", detail="no in-flight tasks")

            await session.close()
            await health_runner.cleanup()
            log.info("shutdown_complete")
