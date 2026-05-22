"""Prompt-only loader for online RL (GRPO / PPO).

Consumes ``rl_prompts.jsonl`` produced by
``llmharness-distill rl-prompts``. Each row is a
**stripped-ReplayRecord** dict — i.e. ``ReplayRecord.to_dict()`` minus
the teacher-output fields (``output``, ``status``, ``error``,
``latency_ms``, ``raw_assistant_messages``). The runtime side
re-hydrates the record via ``ReplayRecord.from_dict(row)`` and replays
it through llmharness's ``replay_extractor_record`` /
``replay_auditor_record`` helpers.

Concretely, each row carries at least::

    {
      "phase": "extractor" | "auditor",
      "root_session_id": "...",
      "turn_index": ...,
      "ts_ns": ...,
      "compose_kwargs": {...},   # composer args (carries prompts)
      "payload": {...},          # the dict sent to child.prompt(...)
      "provider": [...] | null,
      ...optional: sample_id, source_case_id, firing_index, meta...
    }

This loader does **not** inspect ``compose_kwargs`` / ``payload``; it
just validates that the row has the keys the runtime needs and returns
the dicts as-is.
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
    # The runtime path needs payload + compose_kwargs to rebuild a
    # ReplayRecord; everything else is best-effort metadata.
    for key in ("phase", "payload", "compose_kwargs"):
        if key not in row:
            raise ValueError(
                f"rl_prompts row {line_no} missing key '{key}' "
                "(expected stripped-ReplayRecord shape)"
            )


def to_chat_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """Best-effort chat-template view for tokenization-only callers.

    Real RL drivers should pass the full row to the runtime instead —
    this helper is only useful for SFT-style consumers that want to
    tokenize a flat ``[system, user]`` pair. The system prompt is
    pulled from ``compose_kwargs.base_prompt`` (or
    ``compose_kwargs.prompt_override`` for legacy sidecars); the user
    side is the JSON-serialized payload (matching what
    ``child.prompt(json.dumps(payload))`` sends in the live path).
    """
    ck = row.get("compose_kwargs") or {}
    payload = row.get("payload") or {}
    if not isinstance(ck, dict) or not isinstance(payload, dict):
        raise ValueError("rl_prompt row 'compose_kwargs' / 'payload' must be dicts")
    system = str(ck.get("base_prompt") or ck.get("prompt_override") or "")
    user = json.dumps(payload, ensure_ascii=False)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


__all__ = ["Phase", "load_rl_prompts", "to_chat_messages"]
