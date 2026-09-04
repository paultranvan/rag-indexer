import pytest
from aiormq import AMQPConnectionError

from rag_indexer.main import main
from rag_indexer.transport import _file_metadata


def test_file_metadata_maps_fields_and_excludes_secrets():
    meta = {
        "md5sum": "abc123",
        "datetime": "2026-01-15T12:00:00Z",
        "doctype": "io.cozy.files",
        "app_metadata": {"custom": "value"},
        # sensitive fields that must never be echoed back:
        "callback_url": "https://cozy.example/cb",
        "rag": {"api_key": "secret"},
    }

    result = _file_metadata(meta)

    assert result == {
        "version": "abc123",  # md5sum surfaced as version
        "datetime": "2026-01-15T12:00:00Z",
        "doctype": "io.cozy.files",
        "custom": "value",  # app_metadata merged
    }


def test_file_metadata_prefers_explicit_version_over_md5sum():
    result = _file_metadata({"version": "v2", "md5sum": "abc123"})
    assert result["version"] == "v2"


# ------------------------
# BUGF-03: Exit code on connection failure
# ------------------------
@pytest.mark.asyncio
async def test_main_exits_with_nonzero_on_connection_failure(monkeypatch):
    async def fake_connect_robust(url):
        raise AMQPConnectionError("Connection refused")

    monkeypatch.setattr("aio_pika.connect_robust", fake_connect_robust)

    with pytest.raises(SystemExit) as exc_info:
        await main()
    assert exc_info.value.code == 1
