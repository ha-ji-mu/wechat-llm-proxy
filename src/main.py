"""
wechat-llm-proxy — Phase 2 entry point.

Usage:
    python -m src.main [--config config/config.yaml] [--debug]
"""
from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path

from loguru import logger

from src.config import load_config
from src.core.chunker import StreamChunker
from src.core.router import MessageRouter
from src.core.session import SessionManager
from src.llm.anthropic_native import AnthropicNativeAdapter
from src.llm.openai_compat import OpenAICompatAdapter
from src.utils.logger import setup_logger
from src.utils.seen_users import SeenUsers
from src.wechat.auth import ensure_logged_in
from src.wechat.client import ILinkClient
from src.wechat.media import download_image_as_base64
from src.wechat.poller import MessagePoller, WechatMessage
from src.wechat.sender import MessageSender


def _build_adapters(cfg) -> dict:
    """Instantiate all configured LLM adapters. Returns {name: adapter}."""
    adapters = {}

    a = cfg.adapters.get("anthropic_native")
    if a:
        if not a.api_key:
            logger.warning("anthropic_native: missing api_key, skipping")
        else:
            adapters["anthropic_native"] = AnthropicNativeAdapter(
                api_key=a.api_key,
                default_model=a.default_model,
                base_url=a.base_url or "https://api.anthropic.com",
            )

    a = cfg.adapters.get("openai_compat")
    if a:
        if not a.api_key:
            logger.warning("openai_compat: missing api_key, skipping")
        else:
            adapters["openai_compat"] = OpenAICompatAdapter(
                base_url=a.base_url,
                api_key=a.api_key,
                default_model=a.default_model,
            )

    return adapters


async def run(config_path: str, force_login: bool = False) -> None:
    cfg = load_config(config_path)
    setup_logger()

    seen = SeenUsers()
    seen.print_summary()

    async with ILinkClient(cfg.wechat.base_url) as client:
        # ── auth ──────────────────────────────────────────────────────
        await ensure_logged_in(
            client,
            cfg.wechat.token_file,
            cfg.wechat.bot_type,
            force=force_login,
        )

        sender = MessageSender(client)

        # ── Phase 2: router/session/llm/chunker ───────────────────────
        router = MessageRouter(cfg)
        sessions = SessionManager(
            timeout_minutes=cfg.session.timeout_minutes,
            max_history=cfg.session.max_history_messages,
        )
        sessions.start_cleanup()

        llm_adapters = _build_adapters(cfg)
        if not llm_adapters:
            raise RuntimeError("No LLM adapters configured. Check config.yaml.")

        async def handle(msg: WechatMessage) -> None:
            seen.record(msg.from_user)

            if not msg.is_image and (not msg.content or not msg.content.strip()):
                return

            session = sessions.get_or_create(msg.from_user)
            rr = router.route(
                user_id=msg.from_user,
                text=msg.content,
                current_model_key=session.model_key,
            )

            if rr.reject:
                if rr.reject_message:
                    await sender.send_text_with_typing(
                        to_user=msg.from_user,
                        text=rr.reject_message,
                        context_token=msg.context_token,
                    )
                return

            if rr.command == "help":
                await sender.send_text_with_typing(
                    to_user=msg.from_user,
                    text=router.help_text(),
                    context_token=msg.context_token,
                )
                return

            if rr.command == "clear":
                session.clear()
                await sender.send_text_with_typing(
                    to_user=msg.from_user,
                    text="已开启新对话。",
                    context_token=msg.context_token,
                )
                return

            if rr.command == "stats":
                adapter_name, model_str = router._resolve_model(session.model_key)
                lines = ["**当前状态**", ""]
                lines.append(f"模型：`{session.model_key or '默认'}` → {model_str}")
                lines.append(f"本轮 token：in={session.total_input_tokens} out={session.total_output_tokens}")
                lines.append(f"上下文轮数：{len(session.messages) // 2}")
                lines.append("")
                lines.append("**可用模型**")
                for key, alias in cfg.models.aliases.items():
                    lines.append(f"`{key}` — {alias.model}")
                await sender.send_text_with_typing(
                    to_user=msg.from_user,
                    text="\n".join(lines),
                    context_token=msg.context_token,
                )
                return

            if rr.command.startswith("model:"):
                # switch model alias for subsequent messages
                session.model_key = rr.command.split(":", 1)[1]
                if not rr.text:
                    await sender.send_text_with_typing(
                        to_user=msg.from_user,
                        text=f"已切换模型：`{session.model_key}`（{rr.model}）",
                        context_token=msg.context_token,
                    )
                    return
                # fall through: rr.text will be sent using new model

            # Build user message content — option B image flow
            if msg.is_image:
                # Step 1: download image and park it; ask user what they want
                try:
                    b64, media_type = await download_image_as_base64(
                        msg.image_url, msg.image_aes_key
                    )
                except Exception as exc:
                    logger.exception(f"Image download failed: {exc}")
                    await sender.send_text_with_typing(
                        to_user=msg.from_user,
                        text=f"图片下载失败：{exc}",
                        context_token=msg.context_token,
                    )
                    return

                session.set_pending_image(b64, media_type)
                await sender.send_text_with_typing(
                    to_user=msg.from_user,
                    text="收到图片，请告诉我你想做什么？（例如：提取文字、描述内容、翻译…）",
                    context_token=msg.context_token,
                )
                return

            elif rr.text:
                # Step 2: if there's a pending image, combine it with this text
                pending = session.pop_pending_image()
                if pending:
                    b64, media_type = pending
                    session.messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": b64,
                                },
                            },
                            {"type": "text", "text": rr.text},
                        ],
                    })
                else:
                    session.add_user(rr.text)
            else:
                return

            # Stream LLM and send in buffered chunks
            full_reply_parts: list[str] = []

            async def _send_piece(text: str) -> None:
                await sender.send_text_with_typing(
                    to_user=msg.from_user,
                    text=text,
                    context_token=msg.context_token,
                )

            chunker = StreamChunker(on_send=_send_piece, cfg=cfg.chunker)

            async def _on_llm_chunk(t: str) -> None:
                full_reply_parts.append(t)
                await chunker.feed(t)

            try:
                adapter = llm_adapters.get(rr.adapter_name)
                if not adapter:
                    raise RuntimeError(
                        f"Adapter {rr.adapter_name!r} not available. "
                        f"Configured: {list(llm_adapters)}"
                    )
                await adapter.stream_chat(
                    messages=session.messages,
                    on_chunk=_on_llm_chunk,
                    model=rr.model,
                    on_usage=session.add_usage,
                )
            except Exception as exc:
                logger.exception(f"LLM error: {exc}")
                await chunker.finalize()  # flush any partial content before error message
                await sender.send_text_with_typing(
                    to_user=msg.from_user,
                    text=f"LLM 调用失败：{exc}",
                    context_token=msg.context_token,
                )
                return

            await chunker.finalize()

            full_reply = "".join(full_reply_parts).strip()
            if full_reply:
                session.add_assistant(full_reply)

        # ── polling loop ──────────────────────────────────────────────
        poller = MessagePoller(client)

        poll_task = asyncio.create_task(poller.run(handle))

        def _shutdown(*_: object) -> None:
            logger.info("Shutdown signal received")
            poll_task.cancel()

        loop = asyncio.get_running_loop()
        if sys.platform != "win32":
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, _shutdown)

        try:
            await poll_task
        except asyncio.CancelledError:
            pass
        finally:
            sessions.stop_cleanup()

    logger.info("Bye.")


def main() -> None:
    parser = argparse.ArgumentParser(description="wechat-llm-proxy")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    parser.add_argument("--force-login", action="store_true", help="Ignore cached token and re-scan QR code")
    args = parser.parse_args()

    setup_logger("DEBUG" if args.debug else "INFO")
    logger.info("Starting wechat-llm-proxy (Phase 2)")

    try:
        asyncio.run(run(args.config, force_login=args.force_login))
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
