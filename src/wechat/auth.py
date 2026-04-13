"""
iLink authentication: QR-code login and token persistence.

Flow:
  1. GET /ilink/bot/get_bot_qrcode?bot_type=3
     → { qrcode, qrcode_img_content }
  2. Display QR in terminal; user scans with WeChat
  3. Poll GET /ilink/bot/get_qrcode_status?qrcode=<token>
     status values (strings): "confirmed", "scanned", "expired"
  4. On status=="confirmed": response contains { bot_token, baseurl, ilink_bot_id }
  5. Persist { bot_token, baseurl, bot_id, expire_at } to token_file
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Callable, Awaitable

import qrcode
from loguru import logger

from src.wechat.client import ILinkClient
from src.utils.logger import mask


# iLink expiry is 24 h; refresh 5 min early
_TOKEN_TTL = 24 * 3600
_REFRESH_BEFORE = 5 * 60


class TokenStore:
    def __init__(self, token_file: str) -> None:
        self._path = Path(token_file)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict | None:
        if not self._path.exists():
            return None
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if data.get("expire_at", 0) > time.time() + _REFRESH_BEFORE:
                return data
            logger.info("Stored token expired or near expiry, will re-login")
        except Exception as exc:
            logger.warning(f"Failed to read token file: {exc}")
        return None

    def save(self, bot_token: str, baseurl: str, bot_id: str = "") -> None:
        data = {
            "bot_token": bot_token,
            "baseurl": baseurl,
            "bot_id": bot_id,
            "expire_at": time.time() + _TOKEN_TTL,
        }
        self._path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Token saved to {self._path}")


def _print_qr_to_terminal(url: str) -> None:
    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    qr.make(fit=True)
    qr.print_ascii(invert=True)
    print(f"\n请用微信扫码登录（URL: {url[:60]}...）\n")


async def login(
    client: ILinkClient,
    bot_type: int = 3,
    poll_interval: float = 2.0,
    on_token: Callable[[str, str, str], Awaitable[None]] | None = None,
) -> tuple[str, str]:
    """
    Interactive QR-code login.
    Returns (bot_token, baseurl).
    Calls on_token(bot_token, baseurl) if provided.
    """
    logger.info("Fetching login QR code…")
    resp = await client.get("/ilink/bot/get_bot_qrcode", bot_type=bot_type)
    qrcode_token: str = resp["qrcode"]
    qrcode_url: str = resp.get("qrcode_img_content") or resp.get("qrcode_url") or ""

    _print_qr_to_terminal(qrcode_url or qrcode_token)

    logger.info("Waiting for scan…")
    while True:
        status_resp = await client.get(
            "/ilink/bot/get_qrcode_status", qrcode=qrcode_token
        )
        status = status_resp.get("status", "")

        if status in ("confirmed", 2):
            bot_token: str = status_resp["bot_token"]
            baseurl: str = status_resp.get("baseurl", client.base_url)
            bot_id: str = status_resp.get("ilink_bot_id", "")
            logger.info(f"Login confirmed. bot_id={bot_id} bot_token={mask(bot_token)} baseurl={baseurl}")
            if on_token:
                await on_token(bot_token, baseurl, bot_id)
            return bot_token, baseurl
        elif status in ("scanned", 1):
            logger.info("QR code scanned, waiting for confirmation…")
        elif status in ("expired", -1):
            raise RuntimeError("QR code expired. Restart to try again.")
        else:
            logger.debug(f"qrcode_status: {status_resp}")

        await asyncio.sleep(poll_interval)


async def ensure_logged_in(
    client: ILinkClient,
    token_file: str,
    bot_type: int = 3,
    force: bool = False,
) -> None:
    """
    Load cached token or trigger interactive QR-code login.
    Pass force=True to ignore the cache and re-scan (useful for testing).
    """
    store = TokenStore(token_file)
    cached = None if force else store.load()

    if cached:
        logger.info(f"Using cached token (expires ~{int((cached['expire_at'] - time.time()) / 3600)}h from now)")
        client.bot_token = cached["bot_token"]
        client.bot_id = cached.get("bot_id", "")
        if cached.get("baseurl"):
            await client.switch_baseurl(cached["baseurl"])
        return

    async def _save(bot_token: str, baseurl: str, bot_id: str = "") -> None:
        store.save(bot_token, baseurl, bot_id)
        client.bot_token = bot_token
        client.bot_id = bot_id
        if baseurl:
            await client.switch_baseurl(baseurl)

    await login(client, bot_type=bot_type, on_token=_save)
