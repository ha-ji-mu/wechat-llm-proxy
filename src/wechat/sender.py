"""
iLink message sender.

Correct body formats (from wechatbot-sdk source):
- sendmessage:  {"msg": {<message_dict>}, "base_info": {...}}
  message_dict: from_user_id="", to_user_id, client_id=uuid4(),
                message_type=2, message_state=2, context_token,
                item_list=[{"type":1, "text_item":{"text":"..."}}]
- getconfig:    {"ilink_user_id": ..., "context_token": ..., "base_info": {...}}
- sendtyping:   {"ilink_user_id": ..., "typing_ticket": ..., "status": 1|2, "base_info": {...}}
"""
from __future__ import annotations

import time
from uuid import uuid4
from typing import Any

from loguru import logger

from src.wechat.client import ILinkClient

# message_type=2 → BOT outbound; message_state=2 → complete/FINISH
_MSG_TYPE_BOT = 2
_MSG_STATE_FINISH = 2


class MessageSender:
    def __init__(self, client: ILinkClient) -> None:
        self._client = client
        self._typing_ticket: str = ""
        self._ticket_fetched_at: float = 0.0
        self._ticket_ttl = 3600.0

    # ------------------------------------------------------------------
    # typing_ticket
    # ------------------------------------------------------------------

    async def _get_typing_ticket(self, user_id: str, context_token: str) -> str:
        now = time.monotonic()
        if self._typing_ticket and (now - self._ticket_fetched_at) < self._ticket_ttl:
            return self._typing_ticket
        try:
            resp = await self._client.post(
                "/ilink/bot/getconfig",
                {"ilink_user_id": user_id, "context_token": context_token},
            )
            self._typing_ticket = resp.get("typing_ticket", "")
            self._ticket_fetched_at = now
            logger.debug("Refreshed typing_ticket")
        except Exception as exc:
            logger.warning(f"Failed to fetch typing_ticket: {exc}")
        return self._typing_ticket

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    async def send_text(self, to_user: str, text: str, context_token: str) -> bool:
        """Send a text message. Returns True on success."""
        msg: dict[str, Any] = {
            "from_user_id": "",
            "to_user_id": to_user,
            "client_id": str(uuid4()),
            "message_type": _MSG_TYPE_BOT,
            "message_state": _MSG_STATE_FINISH,
            "context_token": context_token,
            "item_list": [{"type": 1, "text_item": {"text": text}}],
        }
        try:
            resp = await self._client.post("/ilink/bot/sendmessage", {"msg": msg})
            ret = resp.get("ret")
            if isinstance(ret, int) and ret != 0:
                logger.error(f"sendmessage failed ret={ret}: {resp}")
                return False
            logger.info(f"→ {to_user}: {text[:60]!r}{'…' if len(text) > 60 else ''}")
            return True
        except Exception as exc:
            logger.error(f"sendmessage exception: {exc}")
            return False

    async def send_typing(self, to_user: str, context_token: str, active: bool = True) -> None:
        """Send or cancel the typing indicator."""
        ticket = await self._get_typing_ticket(to_user, context_token)
        if not ticket:
            return
        try:
            await self._client.post(
                "/ilink/bot/sendtyping",
                {
                    "ilink_user_id": to_user,
                    "typing_ticket": ticket,
                    "status": 1 if active else 2,
                },
            )
        except Exception as exc:
            logger.warning(f"sendtyping error: {exc}")

    async def send_text_with_typing(self, to_user: str, text: str, context_token: str) -> bool:
        """Send typing indicator, then message, then cancel typing."""
        await self.send_typing(to_user, context_token, active=True)
        try:
            return await self.send_text(to_user, text, context_token)
        finally:
            await self.send_typing(to_user, context_token, active=False)
