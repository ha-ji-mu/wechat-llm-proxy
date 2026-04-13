"""
OpenAI-compatible streaming adapter.

Works with any endpoint that implements POST /chat/completions with SSE,
including Anthropic's API (https://api.anthropic.com/v1),
OpenRouter, DeepSeek, Gemini OpenAI-compat, etc.
"""
from __future__ import annotations

import json
from typing import Callable, Awaitable

import httpx
from loguru import logger

from src.llm.base import LLMAdapter


class OpenAICompatAdapter(LLMAdapter):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        default_model: str,
        max_tokens: int = 4096,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._default_model = default_model
        self._max_tokens = max_tokens
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            # Anthropic's OpenAI-compat layer accepts this header silently
            "anthropic-version": "2023-06-01",
        }

    async def stream_chat(
        self,
        messages: list[dict],
        on_chunk: Callable[[str], Awaitable[None]],
        model: str | None = None,
        on_usage: Callable[[int, int], None] | None = None,
    ) -> str:
        model = model or self._default_model
        url = f"{self._base_url}/chat/completions"
        body = {
            "model": model,
            "messages": messages,
            "stream": True,
            "max_tokens": self._max_tokens,
        }

        full_text = ""
        logger.debug(f"LLM request: model={model} messages={len(messages)}")

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST", url, json=body, headers=self._headers()
            ) as resp:
                if resp.status_code != 200:
                    body_text = await resp.aread()
                    raise RuntimeError(
                        f"LLM API error {resp.status_code}: {body_text.decode()[:200]}"
                    )

                async for raw_line in resp.aiter_lines():
                    line = raw_line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    # usage chunk (last SSE before [DONE])
                    usage = chunk.get("usage")
                    if usage:
                        in_tok = usage.get("input_tokens") or usage.get("prompt_tokens", 0)
                        out_tok = usage.get("output_tokens") or usage.get("completion_tokens", 0)
                        logger.info(f"[usage] model={model} in={in_tok} out={out_tok} total={in_tok+out_tok}")
                        if on_usage:
                            on_usage(in_tok, out_tok)

                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    text = delta.get("content") or ""
                    if text:
                        full_text += text
                        await on_chunk(text)

        logger.debug(f"LLM response: {len(full_text)} chars")
        return full_text
