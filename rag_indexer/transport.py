import asyncio

import aio_pika
import structlog
from aio_pika import ExchangeType, Message, DeliveryMode
from aiohttp import web
from prometheus_client.aiohttp import make_aiohttp_handler

from rag_indexer.config import EXCHANGE_NAME, ROUTING_KEY, QUEUE_NAME, RETRY_QUEUES, DLQ_NAME

log = structlog.get_logger()


# ---------- Shutdown coordination ----------
shutdown_event = asyncio.Event()


def request_shutdown():
    """Signal handler -- plain function, NOT async. Safe for loop.add_signal_handler()."""
    log.warning("sigterm_received", detail="initiating graceful shutdown")
    shutdown_event.set()


# ---------- Health endpoint ----------
async def health_handler(request):
    """Minimal health check -- returns 200 if RabbitMQ connection is alive."""
    connection = request.app["rmq_connection"]
    if connection.is_closed:
        return web.json_response({"status": "unhealthy"}, status=503)
    return web.json_response({"status": "healthy"}, status=200)


async def start_health_server(connection, host="0.0.0.0", port=8080):
    """Start a non-blocking HTTP health server alongside the consumer loop."""
    app = web.Application()
    app["rmq_connection"] = connection
    app.router.add_get("/health", health_handler)
    app.router.add_get("/metrics", make_aiohttp_handler())
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    return runner


# ---------- Retry / DLQ publishing ----------
def _file_metadata(metadata: dict) -> dict:
    """File metadata echoed to the cozy callback, mirroring rag_client.build_metadata."""
    result = {
        "version": metadata.get("version") or metadata.get("md5sum") or "",
        "datetime": metadata.get("datetime") or "",
        "doctype": metadata.get("doctype") or "",
    }
    app_metadata = metadata.get("app_metadata")
    if isinstance(app_metadata, dict):
        result |= app_metadata
    return result


async def publish_to_retry(
    channel: aio_pika.Channel,
    original_msg: aio_pika.IncomingMessage,
    retry_queue_name: str,
) -> None:
    """Publish message copy to a retry delay queue via the default exchange."""
    await channel.default_exchange.publish(
        Message(
            original_msg.body,
            content_type=original_msg.content_type,
            headers={**(original_msg.headers or {})},
            delivery_mode=DeliveryMode.PERSISTENT,
        ),
        routing_key=retry_queue_name,
    )


async def publish_to_dlq(
    channel: aio_pika.Channel,
    original_msg: aio_pika.IncomingMessage,
) -> None:
    """Publish message copy to the dead letter queue."""
    await channel.default_exchange.publish(
        Message(
            original_msg.body,
            content_type=original_msg.content_type,
            headers={**(original_msg.headers or {})},
            delivery_mode=DeliveryMode.PERSISTENT,
        ),
        routing_key=DLQ_NAME,
    )


# ---------- Topology declaration ----------
async def declare_topology(channel: aio_pika.Channel) -> aio_pika.Queue:
    """Declare exchanges, queues, and bindings. Returns the main queue."""
    # Main exchange (topic, durable)
    main_ex = await channel.declare_exchange(
        EXCHANGE_NAME, ExchangeType.TOPIC, durable=True
    )

    # Main queue (QUORUM -- Raft replication for data safety)
    main_q = await channel.declare_queue(
        QUEUE_NAME,
        durable=True,
        arguments={"x-queue-type": "quorum"},
    )
    await main_q.bind(main_ex, routing_key=ROUTING_KEY)
    # Also receive messages returning from retry delay queues
    await main_q.bind(main_ex, routing_key="rag.index.retry")

    # Retry delay queues (CLASSIC -- no consumers, TTL only)
    for qname, ttl in RETRY_QUEUES:
        await channel.declare_queue(
            qname,
            durable=True,
            arguments={
                "x-message-ttl": ttl,
                "x-dead-letter-exchange": EXCHANGE_NAME,
                "x-dead-letter-routing-key": "rag.index.retry",
            },
        )

    # DLQ (QUORUM -- dead letters must never be lost)
    await channel.declare_queue(
        DLQ_NAME,
        durable=True,
        arguments={"x-queue-type": "quorum"},
    )

    return main_q
