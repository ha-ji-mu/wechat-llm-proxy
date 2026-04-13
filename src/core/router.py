"""
Message routing: whitelist, command parsing, model selection.

Commands:
  !model:<alias>  — switch model (fuzzy match against configured aliases)
  !clear          — clear conversation history
  !help           — show available commands and models
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from loguru import logger

from src.config import AppConfig


@dataclass
class RouteResult:
    text: str                  # cleaned message text to send to LLM
    adapter_name: str          # which adapter to use
    model: str                 # exact model string
    command: str = ""          # "clear" | "help" | "" (normal message)
    reject: bool = False       # True if message should be silently dropped/rejected
    reject_message: str = ""   # reply to send if rejected


class MessageRouter:
    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg

    def route(self, user_id: str, text: str, current_model_key: str = "") -> RouteResult:
        # ── whitelist ─────────────────────────────────────────────────
        if self._cfg.whitelist.enabled and self._cfg.whitelist.users:
            if user_id not in self._cfg.whitelist.users:
                return RouteResult(
                    text="",
                    adapter_name="",
                    model="",
                    reject=True,
                    reject_message=self._cfg.whitelist.reject_message,
                )

        stripped = text.strip()

        # ── !!! stats ─────────────────────────────────────────────────
        if stripped == "!!!":
            return RouteResult(text="", adapter_name="", model="", command="stats")

        # ── !! new conversation ───────────────────────────────────────
        if stripped == "!!":
            return RouteResult(text="", adapter_name="", model="", command="clear")

        # ── !clear / !help (keep for discoverability) ─────────────────
        if stripped.lower() == "!clear":
            return RouteResult(text="", adapter_name="", model="", command="clear")

        if stripped.lower() == "!help":
            return RouteResult(text="", adapter_name="", model="", command="help")

        # ── !model:<alias> ────────────────────────────────────────────
        model_match = re.match(r"^!model:(\S+)\s*", stripped, re.IGNORECASE)
        if model_match:
            alias_query = model_match.group(1).lower()
            remainder = stripped[model_match.end():].strip()
            matched = self._match_alias(alias_query)
            if matched:
                alias_key, alias_cfg = matched
                logger.info(f"Model switch: {alias_query!r} → {alias_key} ({alias_cfg.model})")
                return RouteResult(
                    text=remainder,
                    adapter_name=alias_cfg.adapter,
                    model=alias_cfg.model,
                    command=f"model:{alias_key}",
                )
            else:
                # Unknown alias — treat as normal message with default model
                logger.warning(f"Unknown model alias: {alias_query!r}")

        # ── normal message ────────────────────────────────────────────
        adapter_name, model = self._resolve_model(current_model_key)
        return RouteResult(text=stripped, adapter_name=adapter_name, model=model)

    def _match_alias(self, query: str) -> tuple[str, object] | None:
        aliases = self._cfg.models.aliases
        # Exact match first
        if query in aliases:
            return query, aliases[query]
        # Prefix match
        matches = [(k, v) for k, v in aliases.items() if k.startswith(query)]
        if len(matches) == 1:
            return matches[0]
        # Substring match
        matches = [(k, v) for k, v in aliases.items() if query in k]
        if len(matches) == 1:
            return matches[0]
        return None

    def _resolve_model(self, model_key: str) -> tuple[str, str]:
        if model_key and model_key in self._cfg.models.aliases:
            alias = self._cfg.models.aliases[model_key]
            return alias.adapter, alias.model
        # Fall back to default adapter
        default = self._cfg.models.default
        adapter_cfg = self._cfg.adapters.get(default)
        if adapter_cfg:
            return default, adapter_cfg.default_model
        return default, ""

    def help_text(self) -> str:
        lines = ["**可用命令**", ""]
        lines.append("`!!` — 开始新对话（清空上下文）")
        lines.append("`!!!` — 查看可用模型和本次 token 用量")
        lines.append("`!model:<名称>` — 切换模型，例如 `!model:sonnet`")
        lines.append("`!help` — 显示此帮助")
        lines.append("")
        lines.append("**可用模型**")
        for key, alias in self._cfg.models.aliases.items():
            lines.append(f"`{key}` — {alias.model}")
        return "\n".join(lines)
