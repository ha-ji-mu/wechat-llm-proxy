"""
Tool definitions for Anthropic native tool use.

web_search_20250305 is Anthropic-hosted (server-side) — no local execution needed.
get_datetime is a simple client-side tool.
"""
from __future__ import annotations

import datetime
from typing import Any


# Included in every /v1/messages request
TOOL_DEFINITIONS: list[dict] = [
    # Anthropic-hosted: Anthropic executes the search, streams results back transparently
    {
        "type": "web_search_20250305",
        "name": "web_search",
    },
    # Client-side: we execute this ourselves
    {
        "name": "get_datetime",
        "description": "Return the current local date and time in ISO 8601 format.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

# Names of tools executed server-side by Anthropic (no local round-trip needed)
SERVER_SIDE_TOOLS: frozenset[str] = frozenset({"web_search"})


async def execute_tool(name: str, tool_input: dict[str, Any]) -> str:
    """Execute a client-side tool and return the result as a string."""
    if name == "get_datetime":
        return datetime.datetime.now().isoformat(timespec="seconds")
    raise ValueError(f"Unknown client-side tool: {name!r}")
