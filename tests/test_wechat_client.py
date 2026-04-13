"""Tests for iLink client, auth helpers, poller parser, and sender."""
from __future__ import annotations

import asyncio
import base64
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.wechat.client import ILinkClient, _random_uin_header
from src.wechat.auth import TokenStore, _TOKEN_TTL
from src.wechat.poller import MessagePoller, WechatMessage, _parse_message
from src.wechat.sender import MessageSender


# ── client ────────────────────────────────────────────────────────────────────

def test_random_uin_header_is_base64_of_numeric_string():
    h = _random_uin_header()
    decoded = base64.b64decode(h).decode()
    assert decoded.isdigit()


def test_random_uin_header_varies():
    headers = {_random_uin_header() for _ in range(50)}
    assert len(headers) > 1


@pytest.mark.asyncio
async def test_client_get_sets_required_headers():
    client = ILinkClient("https://example.com")
    await client.start()
    client.bot_token = "test_token"

    captured = {}

    async def fake_get(path, **kwargs):
        captured["headers"] = kwargs.get("headers", {})
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value={"ret": 0})
        return resp

    client._http.get = fake_get
    await client.get("/test")

    h = captured["headers"]
    assert h["Content-Type"] == "application/json"
    assert h["AuthorizationType"] == "ilink_bot_token"
    assert "X-WECHAT-UIN" in h
    assert h["Authorization"] == "Bearer test_token"
    await client.close()


@pytest.mark.asyncio
async def test_client_raises_without_start():
    client = ILinkClient("https://example.com")
    with pytest.raises(RuntimeError, match="not started"):
        await client.get("/test")


# ── token store ───────────────────────────────────────────────────────────────

def test_token_store_save_and_load(tmp_path):
    store = TokenStore(str(tmp_path / "data" / "bot_token.json"))
    store.save("mytoken", "https://example.com")
    loaded = store.load()
    assert loaded is not None
    assert loaded["bot_token"] == "mytoken"
    assert loaded["baseurl"] == "https://example.com"
    assert loaded["expire_at"] > time.time()


def test_token_store_load_expired(tmp_path):
    path = tmp_path / "bot_token.json"
    path.write_text(
        json.dumps({"bot_token": "old", "baseurl": "", "expire_at": time.time() - 1}),
        encoding="utf-8",
    )
    store = TokenStore(str(path))
    assert store.load() is None


def test_token_store_load_missing(tmp_path):
    store = TokenStore(str(tmp_path / "missing.json"))
    assert store.load() is None


# ── message parser ─────────────────────────────────────────────────────────────

def _make_raw(
    from_user="u1@im.wechat",
    to_user="bot@im.bot",
    context_token="ctx123",
    msg_type=1,
    content="hello",
) -> dict:
    return {
        "msg_id": "m1",
        "from_user": from_user,
        "to_user": to_user,
        "context_token": context_token,
        "type": msg_type,
        "item_list": [{"type": msg_type, "content": content}],
    }


def test_parse_text_message():
    msg = _parse_message(_make_raw())
    assert msg is not None
    assert msg.from_user == "u1@im.wechat"
    assert msg.content == "hello"
    assert msg.is_text


def test_parse_voice_message_uses_transcription():
    raw = {
        "msg_id": "m2",
        "from_user": "u1@im.wechat",
        "to_user": "bot@im.bot",
        "context_token": "ctx",
        "type": 3,
        "item_list": [{"type": 3, "text": "voice transcription", "content": ""}],
    }
    msg = _parse_message(raw)
    assert msg is not None
    assert msg.content == "voice transcription"
    assert msg.is_voice


def test_parse_message_missing_context_token_returns_none():
    raw = _make_raw()
    raw["context_token"] = ""
    assert _parse_message(raw) is None


def test_parse_message_missing_from_user_returns_none():
    raw = _make_raw()
    raw["from_user"] = ""
    assert _parse_message(raw) is None


# ── poller ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_poller_calls_handler_for_each_message():
    client = MagicMock()
    responses = [
        {
            "ret": 0,
            "get_updates_buf": "cursor1",
            "msgs": [_make_raw(content="msg1"), _make_raw(content="msg2")],
        },
    ]

    async def fake_post(path, body):
        if responses:
            return responses.pop(0)
        poller.stop()
        return {"ret": 0, "get_updates_buf": "cursor2", "msgs": []}

    client.post = fake_post
    poller = MessagePoller(client)

    received: list[str] = []

    async def handler(msg: WechatMessage) -> None:
        received.append(msg.content)

    await asyncio.wait_for(poller.run(handler), timeout=2.0)

    assert received == ["msg1", "msg2"]


@pytest.mark.asyncio
async def test_poller_updates_cursor():
    client = MagicMock()
    cursor_seen: list[str] = []

    call_n = 0

    async def fake_post(path, body):
        nonlocal call_n
        call_n += 1
        cursor_seen.append(body.get("get_updates_buf", ""))
        if call_n == 1:
            return {"ret": 0, "get_updates_buf": "cur1", "msgs": []}
        poller.stop()
        return {"ret": 0, "get_updates_buf": "cur2", "msgs": []}

    client.post = fake_post
    poller = MessagePoller(client)

    async def noop(msg): pass

    await asyncio.wait_for(poller.run(noop), timeout=2.0)

    assert cursor_seen[0] == ""   # initial
    assert cursor_seen[1] == "cur1"  # updated after first response


# ── sender ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sender_send_text_posts_correct_body():
    client = MagicMock()
    posted: list[dict] = []

    async def fake_post(path, body):
        posted.append((path, body))
        return {"ret": 0}

    client.post = fake_post
    sender = MessageSender(client)

    ok = await sender.send_text("u1@im.wechat", "hello", "ctx_abc")
    assert ok
    path, body = posted[0]
    assert path == "/ilink/bot/sendmessage"
    msg = body["msg"]
    assert msg["to_user_id"] == "u1@im.wechat"
    assert msg["from_user_id"] == ""
    assert msg["context_token"] == "ctx_abc"
    assert msg["message_type"] == 2
    assert msg["message_state"] == 2
    assert msg["item_list"][0]["text_item"]["text"] == "hello"
    assert "client_id" in msg


@pytest.mark.asyncio
async def test_sender_returns_false_on_nonzero_ret():
    client = MagicMock()

    async def fake_post(path, body):
        return {"ret": -1, "errmsg": "fail"}

    client.post = fake_post
    sender = MessageSender(client)
    ok = await sender.send_text("u@im.wechat", "hi", "ctx")
    assert not ok


@pytest.mark.asyncio
async def test_sender_send_typing_skips_if_no_ticket():
    client = MagicMock()
    typed: list = []

    async def fake_post(path, body):
        typed.append(path)
        if path == "/ilink/bot/getconfig":
            return {}  # no typing_ticket
        return {"ret": 0}

    client.post = fake_post
    sender = MessageSender(client)
    # Should not raise; just skip sendtyping
    await sender.send_typing("u@im.wechat", "ctx")
    assert "/ilink/bot/sendtyping" not in typed
