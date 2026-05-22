"""Prompt-only loader for online RL (GRPO / PPO).

Consumes ``rl_prompts.jsonl`` produced by
``llmharness-distill rl-prompts``. Each row::

    {
      "phase": "extractor" | "auditor",
      "sample_id": "...",
      "source_case_id": "...",
      "firing_index": ...,
      "input": {"system": "...", "user": "..."},
      "meta": {...}
    }

Returns a flat list of dicts (no tokenization here) plus a helper that
converts one row into chat-template-ready ``[system, user]`` messages.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

Phase = Literal["extractor", "auditor"]


def load_rl_prompts(
    jsonl_path: str,
    *,
    phase: Phase | None = None,
) -> list[dict[str, Any]]:
    """Load rl_prompts.jsonl. Filters by ``phase`` when set."""
    rows: list[dict[str, Any]] = []
    p = Path(jsonl_path)
    if not p.exists():
        raise FileNotFoundError(f"rl_prompts file not found: {jsonl_path}")
    with p.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"rl_prompts row {line_no} not valid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"rl_prompts row {line_no} must be an object")
            _validate_row(row, line_no)
            if phase is not None and row.get("phase") != phase:
                continue
            rows.append(row)
    return rows


def _validate_row(row: dict[str, Any], line_no: int) -> None:
    for key in ("phase", "sample_id", "input"):
        if key not in row:
            raise ValueError(f"rl_prompts row {line_no} missing key '{key}'")
    inp = row["input"]
    if not isinstance(inp, dict) or "system" not in inp or "user" not in inp:
        raise ValueError(
            f"rl_prompts row {line_no} 'input' must carry both 'system' and 'user'"
        )


def to_chat_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """Convert one rl_prompt row into ``[system, user]`` chat messages."""
    inp = row.get("input")
    if not isinstance(inp, dict):
        raise ValueError("rl_prompt row missing 'input'")
    system = str(inp.get("system", ""))
    user = str(inp.get("user", ""))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


__all__ = ["Phase", "load_rl_prompts", "to_chat_messages"]
