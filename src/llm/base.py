from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Awaitable


class LLMAdapter(ABC):
    @abstractmethod
    async def stream_chat(
        self,
        messages: list[dict],
        on_chunk: Callable[[str], Awaitable[None]],
        model: str | None = None,
        on_usage: Callable[[int, int], None] | None = None,
    ) -> str:
        """
        Stream a chat completion.
        Calls on_chunk(text) for each piece of generated text.
        Returns the full concatenated response.
        """
