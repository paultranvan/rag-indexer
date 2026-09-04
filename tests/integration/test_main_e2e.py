"""E2E tests: launch main() for real and verify the full pipeline.

Messages are published via the producer's publish_message() — same code path
as production. The RAG API is mocked at the HTTP level using aiohttp TestServer
(no monkeypatching of internal functions). main() runs as a real asyncio task.
Tests are skipped when the broker is not reachable.
"""

import asyncio
import contextlib
import sys
from pathlib import Path

import aio_pika
import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer

import rag_indexer.config as config_mod
import rag_indexer.main as main_mod
import rag_indexer.processing as processing_mod
import rag_indexer.transport as transport_mod

from tests.integration.conftest import (
    TEST_EXCHANGE,
    TEST_QUEUE,
    TEST_DLQ,
    TEST_ROUTING_KEY,
    TEST_RETRY_QUEUES,
)
from tests.integration.test_retry_cycle import _consume_one

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import producer as producer_mod  # noqa: E402

pytestmark = pytest.mark.usefixtures("require_broker")

_PUBLISH_ROUTING_KEY = "test.retry.index"

_DOCUMENT_MARKDOWN = (
    b"# Guide d'utilisation\n\n"
    b"Ce document decrit les fonctionnalites principales du systeme.\n\n"
    b"## Installation\n\n"
    b"Suivez les etapes ci-dessous pour installer le composant.\n"
)


# ---------- Polling helpers ----------

async def _poll(cond, *, timeout: float = 15.0, interval: float = 0.2) -> None:
    """Poll a sync or async callable until it returns True, or raise TimeoutError."""
    async with asyncio.timeout(timeout):
        while True:
            result = cond()
            if asyncio.iscoroutine(result):
                result = await result
            if result:
                return
            await asyncio.sleep(interval)


async def _wait_consumer_ready(rabbitmq_url: str, timeout: float = 10.0) -> None:
    """Poll until the main queue has at least one registered consumer.

    Opens a fresh connection+channel on every iteration: a passive declare on a
    non-existent queue triggers an AMQP 404 that closes the channel, so re-using
    the same channel across iterations would leave it in a closed state.
    """
    async def check():
        try:
            conn = await aio_pika.connect(rabbitmq_url)
            async with conn:
                ch = await conn.channel()
                q = await ch.declare_queue(TEST_QUEUE, passive=True)
                return q.declaration_result.consumer_count > 0
        except Exception:
            return False

    await _poll(check, timeout=timeout)


# ---------- Fixtures ----------

class _FakeRunner:
    async def cleanup(self):
        pass


@pytest_asyncio.fixture
async def rag_stub():
    """In-process HTTP server simulating the RAG API.

    Yields (server, call_log, response_rules).
    - call_log: list of dicts recording each request received
    - response_rules: dict (method, path) -> (status, payload) | callable(request)
      Default response when no rule matches: 200 {"status": "ok"}
    """
    call_log = []
    response_rules = {}

    async def handler(request: web.Request) -> web.Response:
        body = await request.read()
        call_log.append({
            "method": request.method,
            "path": request.path,
            "query": dict(request.rel_url.query),
            "headers": dict(request.headers),
            "body": body,
        })
        key = (request.method, request.path)
        if key in response_rules:
            rule = response_rules[key]
            status, payload = rule(request) if callable(rule) else rule
        else:
            status, payload = 200, {"status": "ok"}

        if isinstance(payload, bytes):
            return web.Response(status=status, body=payload)
        if isinstance(payload, str):
            return web.Response(status=status, text=payload)
        return web.json_response(payload, status=status)

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", handler)
    server = TestServer(app)
    await server.start_server()
    yield server, call_log, response_rules
    await server.close()


@pytest_asyncio.fixture
async def main_consumer(rabbitmq_url, monkeypatch):
    """Launch main() as a background task and guarantee shutdown on teardown.

    Patches all config so main() targets the test broker and test topology.
    Uses a fresh asyncio.Event per test to avoid event-loop binding errors
    across tests (pytest-asyncio creates a new loop for each test function).
    """
    new_shutdown_event = asyncio.Event()
    monkeypatch.setattr(transport_mod, "shutdown_event", new_shutdown_event)
    monkeypatch.setattr(main_mod, "shutdown_event", new_shutdown_event)

    for mod in (config_mod, transport_mod):
        monkeypatch.setattr(mod, "EXCHANGE_NAME", TEST_EXCHANGE)
        monkeypatch.setattr(mod, "QUEUE_NAME", TEST_QUEUE)
        monkeypatch.setattr(mod, "DLQ_NAME", TEST_DLQ)
        monkeypatch.setattr(mod, "ROUTING_KEY", TEST_ROUTING_KEY)
        monkeypatch.setattr(mod, "RETRY_QUEUES", TEST_RETRY_QUEUES)
    monkeypatch.setattr(processing_mod, "RETRY_QUEUES", TEST_RETRY_QUEUES)
    monkeypatch.setattr(main_mod, "RABBITMQ_URL", rabbitmq_url)
    monkeypatch.setattr(producer_mod, "RABBITMQ_URL", rabbitmq_url)
    monkeypatch.setattr(producer_mod, "EXCHANGE_NAME", TEST_EXCHANGE)

    async def fake_health_server(connection, port):
        return _FakeRunner()
    monkeypatch.setattr(transport_mod, "start_health_server", fake_health_server)

    task = asyncio.create_task(main_mod.main())
    await _wait_consumer_ready(rabbitmq_url)

    try:
        yield task
    finally:
        new_shutdown_event.set()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), timeout=15.0)
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


# ---------- Publish helper ----------

async def _publish(
    *,
    file_id: str,
    body: bytes,
    name: str,
    rag_base_url: str,
    content_type: str = "text/markdown",
    action: str = "upsert",
) -> None:
    """Publish a message via the producer — same code path as production."""
    headers = producer_mod.build_headers(
        action=action,
        partition="test-partition",
        file_id=file_id,
        rag_base_url=rag_base_url,
        rag_api_key="",
        content_type=content_type,
        name=name,
    )
    await producer_mod.publish_message(
        routing_key=_PUBLISH_ROUTING_KEY,
        headers=headers,
        body=body,
    )


# ---------- Tests ----------

async def test_upsert_happy_path(rabbitmq_url, monkeypatch, rag_stub, rmq_channel, main_consumer):
    """main() upserts a new document: GET 404 → POST 201. Stub records the upload."""
    _, _, dlq = rmq_channel
    server, call_log, response_rules = rag_stub
    rag_base_url = str(server.make_url("")).rstrip("/")

    response_rules[("GET", "/partition/test-partition/file/e2e-upsert-001")] = (404, {})
    response_rules[("POST", "/indexer/partition/test-partition/file/e2e-upsert-001")] = (201, {"id": "e2e-upsert-001"})

    await _publish(file_id="e2e-upsert-001", body=_DOCUMENT_MARKDOWN, name="guide.md", rag_base_url=rag_base_url)
    await _poll(lambda: any(c["method"] == "POST" for c in call_log))

    post_calls = [c for c in call_log if c["method"] == "POST"]
    assert len(post_calls) == 1
    assert "test-partition" in post_calls[0]["path"]
    assert b"guide.md" in post_calls[0]["body"]
    assert await dlq.get(fail=False) is None


async def test_delete_happy_path(rabbitmq_url, monkeypatch, rag_stub, rmq_channel, main_consumer):
    """main() deletes a document: DELETE 200. Stub records the correct endpoint."""
    _, _, dlq = rmq_channel
    server, call_log, response_rules = rag_stub
    rag_base_url = str(server.make_url("")).rstrip("/")

    response_rules[("DELETE", "/indexer/partition/test-partition/file/e2e-delete-001")] = (200, {})

    await _publish(file_id="e2e-delete-001", body=b"", name="doc.md", rag_base_url=rag_base_url, action="delete")
    await _poll(lambda: any(c["method"] == "DELETE" for c in call_log))

    delete_calls = [c for c in call_log if c["method"] == "DELETE"]
    assert len(delete_calls) == 1
    assert delete_calls[0]["path"] == "/indexer/partition/test-partition/file/e2e-delete-001"
    assert await dlq.get(fail=False) is None


async def test_fatal_error_goes_to_dlq_immediately(rabbitmq_url, monkeypatch, rag_stub, rmq_channel, main_consumer):
    """RAG 400 → FatalError → message in DLQ immediately, no retries."""
    _, _, dlq = rmq_channel
    server, call_log, response_rules = rag_stub
    rag_base_url = str(server.make_url("")).rstrip("/")

    response_rules[("GET", "/partition/test-partition/file/e2e-fatal-001")] = (400, "Bad request")

    await _publish(file_id="e2e-fatal-001", body=_DOCUMENT_MARKDOWN, name="fatal.md", rag_base_url=rag_base_url)

    dlq_msg = await _consume_one(dlq, timeout=10.0)
    assert dlq_msg.body == _DOCUMENT_MARKDOWN
    assert len([c for c in call_log if c["method"] == "GET"]) == 1
    await dlq_msg.ack()


async def test_transient_error_then_success_on_retry(rabbitmq_url, monkeypatch, rag_stub, rmq_channel, main_consumer):
    """GET 503 on first attempt → main() retries → GET 404 + POST 201. DLQ stays empty."""
    _, _, dlq = rmq_channel
    server, call_log, response_rules = rag_stub
    rag_base_url = str(server.make_url("")).rstrip("/")

    get_count = {"n": 0}

    def flaky_get(_req):
        get_count["n"] += 1
        return (503, "RAG temporarily down") if get_count["n"] == 1 else (404, {})

    response_rules[("GET", "/partition/test-partition/file/e2e-flaky-001")] = flaky_get
    response_rules[("POST", "/indexer/partition/test-partition/file/e2e-flaky-001")] = (201, {"id": "e2e-flaky-001"})

    await _publish(file_id="e2e-flaky-001", body=_DOCUMENT_MARKDOWN, name="flaky.md", rag_base_url=rag_base_url)

    ttl_ms = TEST_RETRY_QUEUES[0][1]
    await _poll(lambda: get_count["n"] >= 2, timeout=ttl_ms / 1000.0 + 5.0)
    await _poll(lambda: any(c["method"] == "POST" for c in call_log))

    assert get_count["n"] == 2
    assert len([c for c in call_log if c["method"] == "POST"]) == 1
    assert await dlq.get(fail=False) is None


async def test_retry_exhaustion_to_dlq(rabbitmq_url, monkeypatch, rag_stub, rmq_channel, main_consumer):
    """RAG always 503 → main() retries x3 → message ends up in DLQ."""
    _, _, dlq = rmq_channel
    server, call_log, response_rules = rag_stub
    rag_base_url = str(server.make_url("")).rstrip("/")

    response_rules[("GET", "/partition/test-partition/file/e2e-exhausted-001")] = (503, "RAG down")

    await _publish(file_id="e2e-exhausted-001", body=_DOCUMENT_MARKDOWN, name="exhausted.md", rag_base_url=rag_base_url)

    ttl_total = sum(ttl for _, ttl in TEST_RETRY_QUEUES) / 1000.0
    dlq_msg = await _consume_one(dlq, timeout=ttl_total + 5.0)
    assert dlq_msg.body == _DOCUMENT_MARKDOWN
    assert len([c for c in call_log if c["method"] == "GET"]) == len(TEST_RETRY_QUEUES) + 1
    await dlq_msg.ack()
