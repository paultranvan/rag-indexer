import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict

import aiohttp
import structlog
from aiohttp import FormData

from rag_indexer.config import HTTP_TIMEOUT
from rag_indexer.models import IndexMessage, RagConn, ContentSpec
from rag_indexer.errors import TransientError, FatalError

log = structlog.get_logger()


@dataclass
class RagResponse:
    status: int
    text: str = ""
    json_data: dict[str, Any] | None = None


async def rag_get_file(
    session: aiohttp.ClientSession, rag: RagConn, partition: str, file_id: str
) -> RagResponse:
    log.debug("rag_api_call", method="GET", endpoint="file")
    url = f"{rag.base_url}/partition/{partition}/file/{file_id}"
    async with session.get(
        url, headers={"Authorization": f"Bearer {rag.api_key}"}, timeout=HTTP_TIMEOUT
    ) as resp:
        text = await resp.text()
        json_data = None
        if resp.status == 200:
            try:
                json_data = json.loads(text)
            except ValueError:
                pass
        return RagResponse(status=resp.status, text=text, json_data=json_data)


async def rag_delete(
    session: aiohttp.ClientSession, rag: RagConn, partition: str, file_id: str
) -> RagResponse:
    log.debug("rag_api_call", method="DELETE", endpoint="file")
    url = f"{rag.base_url}/indexer/partition/{partition}/file/{file_id}"
    async with session.delete(
        url, headers={"Authorization": f"Bearer {rag.api_key}"}, timeout=HTTP_TIMEOUT
    ) as resp:
        text = await resp.text()
        return RagResponse(status=resp.status, text=text)


async def get_producer_file(session: aiohttp.ClientSession, msg: IndexMessage) -> bytes:
    headers = {}
    if msg.content.file_bearer:
        headers["Authorization"] = f"Bearer {msg.content.file_bearer}"

    log.debug("file_download", source="file_url")
    try:
        async with session.get(msg.content.file_url, headers=headers, timeout=HTTP_TIMEOUT) as resp:
            if resp.status == 429 or resp.status >= 500:
                raise TransientError(f"Failed to fetch file_url ({resp.status})")
            if resp.status >= 400:
                raise FatalError(f"Failed to fetch file_url ({resp.status})")
            file = await resp.read()
        return file
    except TransientError:
        raise
    except FatalError:
        raise
    except asyncio.TimeoutError as e:
        raise TransientError(f"Timeout fetching file_url: {e}") from e
    except (aiohttp.ClientConnectorError, aiohttp.ServerDisconnectedError) as e:
        raise TransientError(f"Network error fetching file_url: {e}") from e


def build_metadata(msg: IndexMessage) -> Dict[str, Any]:
    meta = {
        "version": msg.version or msg.md5sum or "",
        "datetime": msg.datetime or "",
        "doctype": msg.doctype or "",
    }
    if msg.app_metadata is not None and isinstance(msg.app_metadata, dict):
        # Custom metadata from app
        app_meta = msg.app_metadata
        meta |= app_meta
    return meta


async def rag_upsert(
    session: aiohttp.ClientSession,
    msg: IndexMessage,
    file: bytes,
    is_new: bool
) -> None:

    form = FormData()
    rag = msg.rag

    # Content source: note_markdown OR file_url download
    filename = msg.name or f"{msg.file_id}.bin"

    if file is not None:
        data_bytes = file
    elif msg.content and msg.content.file_url:
        data_bytes = await get_producer_file(session, msg)
    else:
        raise FatalError("No content provided")

    content_type = msg.content_type or "application/octet-stream"
    form.add_field("file", data_bytes, filename=filename, content_type=content_type)

    meta = build_metadata(msg)

    form.add_field("metadata", json.dumps(meta))

    # query params. TODO: check it is useful, should not
    params = {}
    if msg.dir_id:
        params["parent_id"] = msg.dir_id
    if msg.name:
        params["name"] = msg.name
    if msg.md5sum:
        params["md5sum"] = msg.md5sum

    # POST (new) vs PUT (update) is decided after the existence GET above
    url_base = f"{rag.base_url}/indexer/partition/{msg.partition}/file/{msg.file_id}"

    method = "POST" if is_new else "PUT"
    headers = {"Authorization": f"Bearer {msg.rag.api_key}", "Origin": rag.base_url}

    log.debug("rag_api_call", method=method, endpoint="indexer")
    async with session.request(
        method,
        url_base,
        data=form,
        params=params,
        headers=headers,
        timeout=HTTP_TIMEOUT,
    ) as resp:
        # read body to drain the connection
        resp_text = await resp.text()
        if resp.status == 429 or resp.status >= 500:
            raise TransientError(f"RAG {method} {resp.status}: {resp_text}")
        if resp.status == 409:
            return
        if resp.status >= 400:
            raise FatalError(f"RAG {method} {resp.status}: {resp_text}")
        return
