import os

import aiohttp
from dotenv import load_dotenv

load_dotenv()

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
EXCHANGE_NAME = os.getenv("EXCHANGE_NAME", "rag.index.topic")
ROUTING_KEY = os.getenv("ROUTING_KEY", "rag.index.*")
QUEUE_NAME = os.getenv("QUEUE_NAME", "rag.index.q")


def parse_retry_intervals() -> list[int]:
    """Parse RETRY_INTERVALS env var (comma-separated milliseconds).

    Returns default [30000, 300000, 3600000] if not set or empty.
    Raises ValueError if any value is not a positive integer.
    """
    raw = os.getenv("RETRY_INTERVALS", "").strip()
    if not raw:
        return [30_000, 300_000, 3_600_000]
    intervals = []
    for part in raw.split(","):
        val = int(part.strip())
        if val <= 0:
            raise ValueError(f"Retry interval must be positive, got {val}")
        intervals.append(val)
    return intervals


def build_retry_queues(intervals: list[int]) -> list[tuple[str, int]]:
    """Convert interval list into (queue_name, ttl_ms) tuples.

    Queue names embed TTL as human-readable label:
    - 3600000+ and divisible by 3600000 -> "Xh"
    - 60000+ and divisible by 60000 -> "Xm"
    - Otherwise -> "Xs" using ttl_ms // 1000

    Names must stay unique: retry counting reads distinct x-death entries
    keyed by queue name, so two intervals collapsing to the same label
    (e.g. 30000 and 30500 both -> "30s") would break it. Only colliding
    names get a numeric suffix; non-colliding names are unchanged.
    """
    queues = []
    seen = set()
    for ttl_ms in intervals:
        if ttl_ms >= 3_600_000 and ttl_ms % 3_600_000 == 0:
            label = f"{ttl_ms // 3_600_000}h"
        elif ttl_ms >= 60_000 and ttl_ms % 60_000 == 0:
            label = f"{ttl_ms // 60_000}m"
        else:
            label = f"{ttl_ms // 1000}s"
        name = f"rag.index.retry.{label}.q"
        i = 2
        while name in seen:
            name = f"rag.index.retry.{label}-{i}.q"
            i += 1
        seen.add(name)
        queues.append((name, ttl_ms))
    return queues


RETRY_INTERVALS = parse_retry_intervals()
RETRY_QUEUES = build_retry_queues(RETRY_INTERVALS)
MAX_RETRIES = len(RETRY_QUEUES)
DLQ_NAME = os.getenv("DLQ_NAME", "rag.index.dlq")

CONCURRENCY = int(os.getenv("CONCURRENCY", "1"))

_HTTP_TIMEOUT_SECONDS = int(os.getenv("HTTP_TIMEOUT", "60"))
HTTP_TIMEOUT = aiohttp.ClientTimeout(total=_HTTP_TIMEOUT_SECONDS)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

HEALTH_PORT = int(os.getenv("HEALTH_PORT", "8080"))
