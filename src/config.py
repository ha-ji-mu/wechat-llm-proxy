from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

_ENV_PATTERN_START = "${"
_ENV_PATTERN_END = "}"


def _expand_env(value: Any) -> Any:
    """
    Recursively expand ${VARNAME} in loaded YAML values.
    - If env var is missing, expands to empty string.
    - Only expands when the whole string is exactly "${VARNAME}".
    """
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, str):
        s = value.strip()
        if s.startswith(_ENV_PATTERN_START) and s.endswith(_ENV_PATTERN_END) and " " not in s:
            key = s[len(_ENV_PATTERN_START) : -len(_ENV_PATTERN_END)]
            return os.getenv(key, "")
        return value
    return value


class WechatConfig(BaseModel):
    base_url: str = "https://ilinkai.weixin.qq.com"
    bot_type: int = 3
    channel_version: str = "1.0.2"
    token_file: str = "./data/bot_token.json"


class WhitelistConfig(BaseModel):
    enabled: bool = True
    users: list[str] = Field(default_factory=list)
    reject_message: str = "暂未开放，请联系管理员"


class ModelAlias(BaseModel):
    adapter: str
    model: str


class ModelsConfig(BaseModel):
    default: str = "openai_compat"
    fallback_order: list[str] = Field(default_factory=list)
    aliases: dict[str, ModelAlias] = Field(default_factory=dict)


class AdapterConfig(BaseModel):
    type: str
    # openai_compat
    base_url: str = ""
    api_key: str = ""
    default_model: str = ""
    # web adapters
    cookie_file: str = ""
    organization_id: str = ""

    model_config = {"extra": "allow"}


class SessionConfig(BaseModel):
    timeout_minutes: int = 30
    max_history_messages: int = 20


class ChunkerConfig(BaseModel):
    min_chars: int = 50
    max_chars: int = 500
    interval_seconds: float = 2.5
    code_block_whole: bool = True
    show_thinking: bool = False


class AppConfig(BaseModel):
    wechat: WechatConfig = Field(default_factory=WechatConfig)
    whitelist: WhitelistConfig = Field(default_factory=WhitelistConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    adapters: dict[str, AdapterConfig] = Field(default_factory=dict)
    session: SessionConfig = Field(default_factory=SessionConfig)
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)


def load_config(path: str | Path = "config/config.yaml") -> AppConfig:
    path = Path(path)
    if not path.exists():
        example = path.parent / "config.example.yaml"
        if example.exists():
            raise FileNotFoundError(
                f"Config file not found: {path}\n"
                f"Copy {example} to {path} and fill in your settings."
            )
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    raw = _expand_env(raw)
    return AppConfig.model_validate(raw)
