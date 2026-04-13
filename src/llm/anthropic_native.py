"""
Anthropic-native streaming adapter using POST /v1/messages.

Supports:
- Streaming text with SSE event parsing
- Anthropic-hosted web_search_20250305 (transparent, no client round-trip)
- Client-side tool use (get_datetime) via agent loop
- Extended thinking block passthrough (configurable)
"""
from __future__ import annotations

import json
from typing import Callable, Awaitable

import httpx
from loguru import logger

from src.llm.base import LLMAdapter
from src.llm.tools import TOOL_DEFINITIONS, SERVER_SIDE_TOOLS, execute_tool

# Safety limit to prevent infinite tool loops
_MAX_TOOL_ROUNDS = 6


class AnthropicNativeAdapter(LLMAdapter):
    def __init__(
        self,
        api_key: str,
        default_model: str,
        base_url: str = "https://api.anthropic.com",
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
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "web-search-2025-03-05",
            "Content-Type": "application/json",
        }

    async def _do_stream(
        self,
        messages: list[dict],
        model: str,
        on_chunk: Callable[[str], Awaitable[None]],
    ) -> tuple[list[dict], str, int, int]:
        """
        Run one streaming request.

        Returns:
            (assistant_content, stop_reason, input_tokens, output_tokens)
        """
        url = f"{self._base_url}/v1/messages"
        body = {
            "model": model,
            "messages": messages,
            "stream": True,
            "max_tokens": self._max_tokens,
            "tools": TOOL_DEFINITIONS,
        }

        content_blocks: dict[int, dict] = {}  # index -> block being accumulated
        stop_reason = "end_turn"
        input_tokens = 0
        output_tokens = 0
        event_type: str | None = None

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST", url, json=body, headers=self._headers()
            ) as resp:
                if resp.status_code != 200:
                    err = await resp.aread()
                    raise RuntimeError(
                        f"Anthropic API error {resp.status_code}: {err.decode()[:300]}"
                    )

                async for raw_line in resp.aiter_lines():
                    line = raw_line.strip()
                    if not line:
                        event_type = None
                        continue

                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                        continue

                    if not line.startswith("data:"):
                        continue

                    data_str = line[5:].strip()
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if event_type == "message_start":
                        usage = data.get("message", {}).get("usage", {})
                        input_tokens = usage.get("input_tokens", 0)

                    elif event_type == "content_block_start":
                        idx = data["index"]
                        block = data["content_block"]
                        btype = block.get("type", "")
                        if btype == "text":
                            content_blocks[idx] = {"type": "text", "text": ""}
                        elif btype == "thinking":
                            content_blocks[idx] = {"type": "thinking", "thinking": ""}
                        elif btype == "tool_use":
                            content_blocks[idx] = {
                                "type": "tool_use",
                                "id": block["id"],
                                "name": block["name"],
                                "_input_buf": "",
                            }
                        elif btype == "server_tool_use":
                            # Anthropic-hosted tool (web_search) — just track it
                            content_blocks[idx] = {
                                "type": "server_tool_use",
                                "id": block.get("id", ""),
                                "name": block.get("name", ""),
                                "_input_buf": "",
                            }
                        else:
                            content_blocks[idx] = {"type": btype}

                    elif event_type == "content_block_delta":
                        idx = data["index"]
                        delta = data.get("delta", {})
                        dtype = delta.get("type", "")

                        if dtype == "text_delta":
                            text = delta.get("text", "")
                            if text and idx in content_blocks:
                                content_blocks[idx]["text"] = (
                                    content_blocks[idx].get("text", "") + text
                                )
                                await on_chunk(text)

                        elif dtype == "thinking_delta":
                            if idx in content_blocks:
                                content_blocks[idx]["thinking"] = (
                                    content_blocks[idx].get("thinking", "")
                                    + delta.get("thinking", "")
                                )

                        elif dtype == "input_json_delta":
                            if idx in content_blocks:
                                content_blocks[idx]["_input_buf"] = (
                                    content_blocks[idx].get("_input_buf", "")
                                    + delta.get("partial_json", "")
                                )

                    elif event_type == "message_delta":
                        stop_reason = data.get("delta", {}).get("stop_reason", "end_turn")
                        usage = data.get("usage", {})
                        output_tokens = usage.get("output_tokens", 0)

        # Build clean assistant content list
        assistant_content: list[dict] = []
        for idx in sorted(content_blocks.keys()):
            b = content_blocks[idx]
            btype = b["type"]
            if btype == "text":
                if b.get("text"):
                    assistant_content.append({"type": "text", "text": b["text"]})
            elif btype == "tool_use":
                try:
                    tool_input = json.loads(b["_input_buf"]) if b.get("_input_buf") else {}
                except json.JSONDecodeError:
                    tool_input = {}
                assistant_content.append({
                    "type": "tool_use",
                    "id": b["id"],
                    "name": b["name"],
                    "input": tool_input,
                })
            elif btype == "server_tool_use":
                try:
                    tool_input = json.loads(b["_input_buf"]) if b.get("_input_buf") else {}
                except json.JSONDecodeError:
                    tool_input = {}
                assistant_content.append({
                    "type": "server_tool_use",
                    "id": b["id"],
                    "name": b["name"],
                    "input": tool_input,
                })
            # thinking blocks: keep internally but don't surface to user

        return assistant_content, stop_reason, input_tokens, output_tokens

    async def stream_chat(
        self,
        messages: list[dict],
        on_chunk: Callable[[str], Awaitable[None]],
        model: str | None = None,
        on_usage: Callable[[int, int], None] | None = None,
    ) -> str:
        model = model or self._default_model
        logger.debug(f"LLM request: model={model} messages={len(messages)}")

        current_messages = list(messages)
        full_text_parts: list[str] = []
        total_in = 0
        total_out = 0

        # Wrap on_chunk to also accumulate full text
        async def _on_chunk(text: str) -> None:
            full_text_parts.append(text)
            await on_chunk(text)

        for _round in range(_MAX_TOOL_ROUNDS):
            assistant_content, stop_reason, in_tok, out_tok = await self._do_stream(
                current_messages, model, _on_chunk
            )
            total_in += in_tok
            total_out += out_tok

            current_messages = current_messages + [
                {"role": "assistant", "content": assistant_content}
            ]

            if stop_reason != "tool_use":
                break

            # Execute client-side tools; skip server-side ones
            tool_results: list[dict] = []
            for block in assistant_content:
                if block["type"] != "tool_use":
                    continue
                name = block["name"]
                if name in SERVER_SIDE_TOOLS:
                    # Should not happen (server-side tools don't generate stop_reason=tool_use)
                    logger.warning(f"Unexpected server-side tool in tool_use round: {name}")
                    continue
                logger.info(f"Tool call: {name}({block['input']})")
                try:
                    result = await execute_tool(name, block["input"])
                except Exception as exc:
                    result = f"Error: {exc}"
                    logger.warning(f"Tool {name} failed: {exc}")
                logger.debug(f"Tool result: {name} → {str(result)[:120]}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": str(result),
                })

            if not tool_results:
                break

            current_messages = current_messages + [
                {"role": "user", "content": tool_results}
            ]
        else:
            logger.warning("Reached max tool rounds, stopping agent loop")

        logger.info(
            f"[usage] model={model} in={total_in} out={total_out} total={total_in + total_out}"
        )
        if on_usage:
            on_usage(total_in, total_out)

        full_text = "".join(full_text_parts).strip()
        logger.debug(f"LLM response: {len(full_text)} chars")
        return full_text
