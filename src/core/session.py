"""
Per-user session management.

Each session holds:
- OpenAI-format message history (system + alternating user/assistant)
- Selected model alias (e.g. "haiku", "opus")
- Last active timestamp for TTL eviction
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class Session:
    user_id: str
    messages: list[dict] = field(default_factory=list)
    model_key: str = ""          # alias key; empty = use default
    last_active: float = field(default_factory=time.time)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    # Pending image waiting for a text instruction (option B flow)
    pending_image_b64: str = ""
    pending_image_media_type: str = ""

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        self.last_active = time.time()

    def add_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})
        self.last_active = time.time()

    def add_usage(self, input_tokens: int, output_tokens: int) -> None:
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def set_pending_image(self, b64: str, media_type: str) -> None:
        self.pending_image_b64 = b64
        self.pending_image_media_type = media_type
        self.last_active = time.time()

    def pop_pending_image(self) -> tuple[str, str] | None:
        """Return and clear the pending image, or None if there isn't one."""
        if not self.pending_image_b64:
            return None
        b64, mt = self.pending_image_b64, self.pending_image_media_type
        self.pending_image_b64 = ""
        self.pending_image_media_type = ""
        return b64, mt

    def clear(self) -> None:
        self.messages.clear()
        self.pending_image_b64 = ""
        self.pending_image_media_type = ""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        logger.info(f"Session cleared for {self.user_id}")


class SessionManager:
    def __init__(self, timeout_minutes: int = 30, max_history: int = 20) -> None:
        self._timeout = timeout_minutes * 60
        self._max_history = max_history
        self._sessions: dict[str, Session] = {}
        self._cleanup_task: asyncio.Task | None = None

    def get_or_create(self, user_id: str) -> Session:
        if user_id not in self._sessions:
            self._sessions[user_id] = Session(user_id=user_id)
            logger.info(f"New session for {user_id}")
        session = self._sessions[user_id]
        # Trim history to prevent runaway growth
        if len(session.messages) > self._max_history * 2:
            # Keep system message if present, drop oldest pairs
            session.messages = session.messages[-self._max_history * 2:]
        return session

    def start_cleanup(self) -> None:
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    def stop_cleanup(self) -> None:
        if self._cleanup_task:
            self._cleanup_task.cancel()

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(300)  # check every 5 min
            now = time.time()
            expired = [
                uid for uid, s in self._sessions.items()
                if now - s.last_active > self._timeout
            ]
            for uid in expired:
                del self._sessions[uid]
                logger.info(f"Session expired and removed: {uid}")
