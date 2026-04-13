"""
Track every user_id that has ever messaged the bot.
Persisted to data/seen_users.json for whitelist population.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from loguru import logger


class SeenUsers:
    def __init__(self, path: str = "./data/seen_users.json") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, float] = self._load()

    def _load(self) -> dict[str, float]:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save(self) -> None:
        self._path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def record(self, user_id: str) -> bool:
        """
        Record a user_id. Returns True if this is the first time we've seen them.
        Logs a prominent notice on first contact.
        """
        is_new = user_id not in self._data
        self._data[user_id] = time.time()
        self._save()
        if is_new:
            logger.info(
                f"[NEW USER] {user_id!r} — "
                f"add to whitelist: users:\n  - \"{user_id}\""
            )
        return is_new

    def all_users(self) -> list[str]:
        return list(self._data.keys())

    def print_summary(self) -> None:
        if not self._data:
            logger.info("No users have contacted the bot yet.")
            return
        logger.info(f"Known users ({len(self._data)}) — copy to whitelist:")
        for uid in self._data:
            logger.info(f"  - \"{uid}\"")
