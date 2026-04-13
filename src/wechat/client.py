"""
Low-level iLink HTTP client.

Every request carries:
  - Content-Type: application/json
  - AuthorizationType: ilink_bot_token
  - X-WECHAT-UIN: base64(str(random_uint32))   ← anti-replay, new each call
  - Authorization: Bearer <bot_token>           ← omitted before login
"""
from __future__ import annotations

import base64
import random
import time
from typing import Any

import httpx
from loguru import logger

from src.utils.logger import mask


def _random_uin_header() -> str:
    uid = random.randint(0, 0xFFFFFFFF)
    return base64.b64encode(str(uid).encode()).decode()


CHANNEL_VERSION = "2.0.0"
ILINK_APP_ID = "bot"
# 0x00MMNNPP encoding of channel version 2.0.0
ILINK_APP_CLIENT_VERSION = str((2 << 16) | (0 << 8) | 0)  # "131072"


class ILinkClient:
    def __init__(self, base_url: str, timeout: float = 40.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._bot_token: str = ""
        self.bot_id: str = ""   # e.g. "2cbb77c93b27@im.bot"
        self._http: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._http = httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=True,
        )

    async def switch_baseurl(self, baseurl: str) -> None:
        """Switch to a server-assigned baseurl after login."""
        baseurl = baseurl.rstrip("/")
        if baseurl == self.base_url:
            return
        logger.info(f"Switching baseurl: {self.base_url} → {baseurl}")
        self.base_url = baseurl

    async def close(self) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None

    async def __aenter__(self) -> "ILinkClient":
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # token management
    # ------------------------------------------------------------------

    @property
    def bot_token(self) -> str:
        return self._bot_token

    @bot_token.setter
    def bot_token(self, value: str) -> None:
        self._bot_token = value
        if value:
            logger.debug(f"bot_token set: {mask(value)}")

    # ------------------------------------------------------------------
    # internal request helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "AuthorizationType": "ilink_bot_token",
            "X-WECHAT-UIN": _random_uin_header(),
            "iLink-App-Id": ILINK_APP_ID,
            "iLink-App-ClientVersion": ILINK_APP_CLIENT_VERSION,
        }
        if self._bot_token:
            h["Authorization"] = f"Bearer {self._bot_token}"
        return h

    def _ensure_started(self) -> httpx.AsyncClient:
        if self._http is None:
            raise RuntimeError("ILinkClient not started — call await client.start() first")
        return self._http

    async def get(self, path: str, **params: Any) -> dict[str, Any]:
        http = self._ensure_started()
        url = self.base_url + path
        logger.debug(f"GET {url} params={params}")
        resp = await http.get(url, params=params, headers=self._headers())
        resp.raise_for_status()
        data = resp.json()
        logger.debug(f"GET {url} → {data}")
        return data

    async def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        http = self._ensure_started()
        url = self.base_url + path
        # All POST bodies must include base_info per protocol spec
        payload = {"base_info": {"channel_version": CHANNEL_VERSION}, **body}
        logger.debug(f"POST {url}")
        resp = await http.post(url, json=payload, headers=self._headers())
        resp.raise_for_status()
        data = resp.json()
        logger.debug(f"POST {url} → {data}")
        return data
