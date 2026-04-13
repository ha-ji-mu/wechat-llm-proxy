"""
Long-polling message receiver.

Protocol:
  POST /ilink/bot/getupdates
  body: { get_updates_buf: <cursor>, base_info: { channel_version: "1.0.2" } }
  response: { ret: 0, msgs: [...], get_updates_buf: <new_cursor>, longpolling_timeout_ms: 35000 }

- Server holds connection up to 35 s then returns (possibly with empty msgs).
- Client must immediately re-issue with the new cursor.
- cursor starts as "" (empty string).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from loguru import logger

from src.wechat.client import ILinkClient


@dataclass
class WechatMessage:
    """Parsed representation of a single iLink message."""

    msg_id: str
    from_user: str        # e.g. "abc123@im.wechat"
    to_user: str          # e.g. "botxxx@im.bot"
    content: str          # text content (or transcribed voice)
    msg_type: int         # 1=text, 2=image, 3=voice, 4=file, 5=video
    context_token: str    # must be echoed back when replying
    image_url: str = ""           # msg_type==2: CDN download URL
    image_aes_key: str = ""       # msg_type==2: AES-128 hex key
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def is_text(self) -> bool:
        return self.msg_type == 1

    @property
    def is_image(self) -> bool:
        # message_type is always 1 from iLink; detect image by item content
        return bool(self.image_url)

    @property
    def is_voice(self) -> bool:
        return self.msg_type == 3


MessageHandler = Callable[[WechatMessage], Awaitable[None]]


def _extract_text_from_item(item: dict[str, Any]) -> str:
    """Extract text content from a single item_list entry."""
    item_type = item.get("type", 0)
    if item_type == 1:
        # text_item.text is the actual field; fall back to legacy content
        text_item = item.get("text_item", {})
        return text_item.get("text", "") or item.get("content", "")
    if item_type == 2:
        # handled separately in _parse_message; text content is empty
        return ""
    if item_type == 3:
        # voice: prefer transcription in text_item or text field
        text_item = item.get("text_item", {})
        return text_item.get("text", "") or item.get("text", "") or "[语音消息，暂不支持转文字]"
    return item.get("content", "")


def _parse_message(raw: dict[str, Any]) -> WechatMessage | None:
    """Extract fields from an iLink message dict. Returns None if unparseable."""
    try:
        msg_id = str(raw.get("message_id", raw.get("msg_id", "")))
        from_user = raw.get("from_user_id", raw.get("from_user", ""))
        to_user = raw.get("to_user_id", raw.get("to_user", ""))
        context_token = raw.get("context_token", "")
        msg_type = int(raw.get("message_type", raw.get("type", 1)))

        content = ""
        image_url = ""
        image_aes_key = ""
        item_list = raw.get("item_list", [])
        if item_list:
            for item in item_list:
                itype = item.get("type", 0)
                if itype == 2:
                    img = item.get("image_item", {})
                    media = img.get("media", {})
                    image_url = media.get("full_url", "")
                    image_aes_key = img.get("aeskey", "")
                else:
                    text = _extract_text_from_item(item)
                    if text:
                        content = text
        else:
            content = raw.get("content", raw.get("text", ""))

        if not from_user or not context_token:
            logger.warning(f"Dropping malformed message (missing from_user/context_token): {raw}")
            return None

        return WechatMessage(
            msg_id=msg_id,
            from_user=from_user,
            to_user=to_user,
            content=content,
            msg_type=msg_type,
            context_token=context_token,
            image_url=image_url,
            image_aes_key=image_aes_key,
            raw=raw,
        )
    except Exception as exc:
        logger.error(f"Failed to parse message: {exc} | raw={raw}")
        return None


class MessagePoller:
    def __init__(
        self,
        client: ILinkClient,
        retry_delay: float = 3.0,
    ) -> None:
        self._client = client
        self._retry_delay = retry_delay
        self._cursor: str = ""
        self._running = False

    async def run(self, handler: MessageHandler) -> None:
        """
        Run the long-poll loop indefinitely, calling handler(msg) for each
        incoming message. Handles network errors with exponential back-off.
        """
        self._running = True
        backoff = self._retry_delay
        logger.info("Message poller started")

        while self._running:
            try:
                data = await self._client.post(
                    "/ilink/bot/getupdates",
                    {"get_updates_buf": self._cursor},
                )
                backoff = self._retry_delay  # reset on success

                ret = data.get("ret")
                if isinstance(ret, int) and ret != 0:
                    logger.debug(f"getupdates ret={ret} (non-zero, still processing msgs)")

                new_cursor = data.get("get_updates_buf", self._cursor)
                if new_cursor:
                    self._cursor = new_cursor

                msgs: list[dict] = data.get("msgs", []) or []
                for raw_msg in msgs:
                    msg = _parse_message(raw_msg)
                    if msg is None:
                        continue
                    logger.info(
                        f"← [{msg.msg_type}] {msg.from_user}: "
                        f"{msg.content[:60]!r}{'…' if len(msg.content) > 60 else ''}"
                    )
                    try:
                        await handler(msg)
                    except Exception as exc:
                        logger.exception(f"Handler error for msg from {msg.from_user}: {exc}")

            except Exception as exc:
                logger.error(f"Poll error: {exc} — retry in {backoff:.1f}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    def stop(self) -> None:
        self._running = False
        logger.info("Message poller stopping")
