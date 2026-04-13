"""
Streaming text chunker.

Buffers LLM output and sends it in natural chunks via WeChat:
- Waits for sentence/paragraph boundaries before sending
- Holds code blocks until the closing ``` is received
- Respects min/max char limits and time intervals
"""
from __future__ import annotations

import asyncio
import re
import time
from typing import Callable, Awaitable

from loguru import logger

from src.config import ChunkerConfig


# Sentence-ending punctuation (Chinese + Latin)
_SENTENCE_END = re.compile(r'[。！？!?\n]{1}|[\.\?!]\s')
_CODE_FENCE = re.compile(r'```')


class StreamChunker:
    def __init__(
        self,
        on_send: Callable[[str], Awaitable[None]],
        cfg: ChunkerConfig,
    ) -> None:
        self._on_send = on_send
        self._cfg = cfg
        self._buffer = ""
        self._code_fence_count = 0   # open fences, even=closed, odd=open
        self._last_send_time = time.monotonic()

    def _in_code_block(self) -> bool:
        return self._code_fence_count % 2 == 1

    def _update_fence_count(self, text: str) -> None:
        self._code_fence_count += len(_CODE_FENCE.findall(text))

    async def feed(self, chunk: str) -> None:
        """Feed a new chunk from the LLM stream."""
        self._update_fence_count(chunk)
        self._buffer += chunk

        if self._in_code_block():
            return  # never split inside a code block

        now = time.monotonic()
        elapsed = now - self._last_send_time

        should_send = (
            len(self._buffer) >= self._cfg.max_chars
            or (
                len(self._buffer) >= self._cfg.min_chars
                and elapsed >= self._cfg.interval_seconds
                and _SENTENCE_END.search(self._buffer)
            )
        )

        if should_send:
            await self._flush()

    async def finalize(self) -> None:
        """Called after LLM stream ends — flush whatever remains."""
        if self._buffer.strip():
            await self._flush(force=True)

    async def _flush(self, force: bool = False) -> None:
        if not self._buffer.strip():
            self._buffer = ""
            return

        if force or self._in_code_block():
            text, self._buffer = self._buffer, ""
        else:
            text, self._buffer = self._split_at_boundary(self._buffer)

        if text.strip():
            logger.debug(f"Sending chunk ({len(text)} chars)")
            await self._on_send(text.strip())
            self._last_send_time = time.monotonic()

    def _split_at_boundary(self, text: str) -> tuple[str, str]:
        """Split text at the last good sentence boundary within max_chars."""
        if len(text) <= self._cfg.max_chars:
            # Try to split at last sentence-end
            for m in reversed(list(_SENTENCE_END.finditer(text))):
                end = m.end()
                if end >= self._cfg.min_chars:
                    return text[:end], text[end:]
            # No boundary found — send it all
            return text, ""
        else:
            # Must split; find boundary before max_chars
            window = text[: self._cfg.max_chars]
            for m in reversed(list(_SENTENCE_END.finditer(window))):
                end = m.end()
                if end >= self._cfg.min_chars:
                    return text[:end], text[end:]
            # Hard split at max_chars
            return text[: self._cfg.max_chars], text[self._cfg.max_chars:]
