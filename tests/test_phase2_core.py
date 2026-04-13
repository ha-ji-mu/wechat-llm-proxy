from __future__ import annotations

import os

from src.config import load_config
from src.core.router import MessageRouter
from src.core.session import SessionManager


def test_config_expands_env_vars(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
adapters:
  openai_compat:
    type: "openai_compat"
    api_key: "${OPENAI_COMPAT_API_KEY}"
    base_url: "https://example.com"
    default_model: "m"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "secret")
    cfg = load_config(cfg_path)
    assert cfg.adapters["openai_compat"].api_key == "secret"


def test_router_model_switch_and_remainder(tmp_path):
    # Minimal config to drive router
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
whitelist:
  enabled: false
models:
  default: "openai_compat"
  aliases:
    haiku: { adapter: "openai_compat", model: "claude-haiku-4-5" }
adapters:
  openai_compat:
    type: "openai_compat"
    base_url: "https://example.com"
    api_key: "x"
    default_model: "m"
""".strip(),
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    router = MessageRouter(cfg)
    rr = router.route("u", "!model:hai 你好", "")
    assert rr.command == "model:haiku"
    assert rr.text == "你好"
    assert rr.adapter_name == "openai_compat"
    assert rr.model == "claude-haiku-4-5"


def test_session_manager_trims_history():
    sm = SessionManager(timeout_minutes=30, max_history=2)
    s = sm.get_or_create("u")
    # add 10 messages; trim to last 4
    for i in range(10):
        s.add_user(f"u{i}")
    s2 = sm.get_or_create("u")
    assert len(s2.messages) <= 4

