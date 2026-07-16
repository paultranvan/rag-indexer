#!/usr/bin/env python3
"""DLQ inspection and replay CLI tool.

List and replay dead-lettered messages without using the RabbitMQ management UI.
Uses the same RABBITMQ_URL env var as the consumer.

Usage:
    python scripts/dlq_replay.py list
    python scripts/dlq_replay.py replay <file_id>
    python scripts/dlq_replay.py replay --all
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so rag_indexer is importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import aio_pika
from aio_pika import DeliveryMode

from rag_indexer.config import RABBITMQ_URL, DLQ_NAME, EXCHANGE_NAME


# ---------- Header serialization ----------

def _serialize_value(value):
    """Recursively convert non-JSON-serializable values."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def _serialize_headers(headers: dict) -> dict:
    """Serialize message headers to JSON-safe dict."""
    return {str(k): _serialize_value(v) for k, v in headers.items()}


# ---------- Replay helper ----------

async def _replay_one(msg, channel, exchange):
    """Republish a single message to the main exchange with x-death headers stripped."""
    headers = dict(msg.headers or {})
    for key in ("x-death", "x-first-death-exchange", "x-first-death-queue", "x-first-death-reason"):
        headers.pop(key, None)
    await exchange.publish(
        aio_pika.Message(
            body=msg.body,
            content_type=msg.content_type,
            headers=headers,
            delivery_mode=DeliveryMode.PERSISTENT,
        ),
        routing_key="rag.index.file",
    )
    await msg.ack()


# ---------- Subcommands ----------

async def cmd_list(args: argparse.Namespace) -> None:
    """List all messages in the DLQ as JSON lines."""
    connection = await aio_pika.connect(RABBITMQ_URL)
    async with connection:
        channel = await connection.channel()
        queue = await channel.declare_queue(DLQ_NAME, passive=True)

        total = queue.declaration_result.message_count
        count = 0
        for _ in range(total):
            msg = await queue.get(no_ack=False, fail=False)
            if msg is None:
                break
            headers = _serialize_headers(msg.headers or {})
            info = {
                "message_id": msg.message_id,
                "headers": headers,
                "body_size": len(msg.body),
                "content_type": msg.content_type,
                "routing_key": msg.routing_key,
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
            }
            print(json.dumps(info))
            count += 1
        print(json.dumps({"total": count}), file=sys.stderr)


async def cmd_replay(args: argparse.Namespace) -> None:
    """Replay messages from the DLQ."""
    if not args.file_id and not args.all:
        print("Error: provide <file_id> or --all", file=sys.stderr)
        sys.exit(1)
    if args.file_id and args.all:
        print("Error: provide <file_id> or --all, not both", file=sys.stderr)
        sys.exit(1)

    connection = await aio_pika.connect(RABBITMQ_URL)
    async with connection:
        channel = await connection.channel()
        queue = await channel.declare_queue(DLQ_NAME, passive=True)
        exchange = await channel.declare_exchange(EXCHANGE_NAME, passive=True)

        total = queue.declaration_result.message_count

        if args.all:
            if total == 0:
                print("No messages in DLQ", file=sys.stderr)
                return

            answer = input(f"Replay all {total} messages? [y/N] ").strip().lower()
            if answer != "y":
                print("Aborted", file=sys.stderr)
                return

            replayed = 0
            for _ in range(total):
                msg = await queue.get(no_ack=False, fail=False)
                if msg is None:
                    break

                headers = _serialize_headers(msg.headers or {})
                info = {
                    "replayed": True,
                    "message_id": msg.message_id,
                    "headers": headers,
                    "body_size": len(msg.body),
                }
                await _replay_one(msg, channel, exchange)
                print(json.dumps(info))
                replayed += 1

            print(json.dumps({"total_replayed": replayed}), file=sys.stderr)

        else:
            found = False
            checked = 0
            for _ in range(total):
                msg = await queue.get(no_ack=False, fail=False)
                if msg is None:
                    break

                msg_headers = msg.headers or {}
                if msg_headers.get("file_id") == args.file_id:
                    headers = _serialize_headers(msg_headers)
                    info = {
                        "replayed": True,
                        "message_id": msg.message_id,
                        "file_id": args.file_id,
                        "headers": headers,
                        "body_size": len(msg.body),
                    }
                    await _replay_one(msg, channel, exchange)
                    print(json.dumps(info))
                    found = True
                    break
                checked += 1

            if not found:
                print(f"No message found with file_id={args.file_id} (checked {checked} messages)", file=sys.stderr)
                sys.exit(1)


# ---------- CLI ----------

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DLQ inspection and replay tool for the RAG indexer."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List all DLQ messages as JSON lines (messages remain in queue)")

    replay_p = sub.add_parser("replay", help="Replay DLQ message(s) to the main exchange")
    replay_p.add_argument("file_id", nargs="?", default=None, help="file_id of the message to replay")
    replay_p.add_argument("--all", action="store_true", help="Replay all messages (prompts for confirmation)")

    return p


def main():
    parser = make_parser()
    args = parser.parse_args()

    if args.cmd == "list":
        asyncio.run(cmd_list(args))
    elif args.cmd == "replay":
        asyncio.run(cmd_replay(args))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
