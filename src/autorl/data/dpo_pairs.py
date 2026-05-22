"""DPO pair loader (offline preference pairs).

Consumes ``dpo_pairs.jsonl`` produced by the upcoming llmharness
fork-and-continue pipeline. Each row::

    {
      "phase": "extractor" | "auditor",
      "pair_id": "<case>:<firing>:<i>vs<j>",
      "source_case_id": "...",
      "firing_index": ...,
      "prompt": {"system": "...", "user": "..."},
      "chosen":   {"messages": [...]},
      "rejected": {"messages": [...]},
      "chosen_score": 0.0..1.0,
      "rejected_score": 0.0..1.0,
      "meta": {...}
    }

The tokenization helper applies a chat template separately to
``prompt`` and to ``prompt + chosen/rejected``, returning input-id
lists ready for a DPO loss.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

Phase = Literal["extractor", "auditor"]


def load_dpo_pairs(
    jsonl_path: str,
    *,
    phase: Phase | None = None,
) -> list[dict[str, Any]]:
    """Load dpo_pairs.jsonl. Filters by ``phase`` when set."""
    rows: list[dict[str, Any]] = []
    p = Path(jsonl_path)
    if not p.exists():
        raise FileNotFoundError(f"dpo_pairs file not found: {jsonl_path}")
    with p.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"dpo_pairs row {line_no} not valid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"dpo_pairs row {line_no} must be an object")
            _validate_row(row, line_no)
            if phase is not None and row.get("phase") != phase:
                continue
            rows.append(row)
    return rows


def _validate_row(row: dict[str, Any], line_no: int) -> None:
    for key in ("phase", "pair_id", "prompt", "chosen", "rejected"):
        if key not in row:
            raise ValueError(f"dpo_pairs row {line_no} missing key '{key}'")
    prompt = row["prompt"]
    if (
        not isinstance(prompt, dict)
        or "system" not in prompt
        or "user" not in prompt
    ):
        raise ValueError(
            f"dpo_pairs row {line_no} 'prompt' must carry 'system' and 'user'"
        )
    for side in ("chosen", "rejected"):
        side_val = row[side]
        if not isinstance(side_val, dict) or not isinstance(
            side_val.get("messages"), list
        ):
            raise ValueError(
                f"dpo_pairs row {line_no} '{side}.messages' must be a list"
            )


def pair_to_dpo_inputs(
    row: dict[str, Any],
    tokenizer: Any,
    *,
    max_length: int | None = None,
) -> dict[str, Any]:
    """Tokenize one pair row into ``{prompt, chosen, rejected}_input_ids``.

    The chat template is applied to ``prompt`` (with
    ``add_generation_prompt=True``) and to ``prompt + chosen`` /
    ``prompt + rejected`` (with ``add_generation_prompt=False``). The
    chosen/rejected portion is the contrast span DPO loss operates on.
    """
    prompt = row["prompt"]
    prompt_msgs: list[dict[str, Any]] = [
        {"role": "system", "content": str(prompt.get("system", ""))},
        {"role": "user", "content": str(prompt.get("user", ""))},
    ]
    chosen_msgs = list(row["chosen"]["messages"])
    rejected_msgs = list(row["rejected"]["messages"])

    prompt_ids = _apply_chat_template(
        tokenizer, prompt_msgs, add_generation_prompt=True
    )
    chosen_ids = _apply_chat_template(
        tokenizer,
        [*prompt_msgs, *chosen_msgs],
        add_generation_prompt=False,
    )
    rejected_ids = _apply_chat_template(
        tokenizer,
        [*prompt_msgs, *rejected_msgs],
        add_generation_prompt=False,
    )

    if max_length is not None:
        prompt_ids = prompt_ids[:max_length]
        chosen_ids = chosen_ids[:max_length]
        rejected_ids = rejected_ids[:max_length]

    return {
        "prompt_input_ids": prompt_ids,
        "chosen_input_ids": chosen_ids,
        "rejected_input_ids": rejected_ids,
        "chosen_score": float(row.get("chosen_score", 1.0)),
        "rejected_score": float(row.get("rejected_score", 0.0)),
    }


def _apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError(
            "tokenizer must support apply_chat_template (Qwen/GLM thinking models do)"
        )
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if hasattr(token_ids, "get") and "input_ids" in token_ids:
        token_ids = token_ids["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(tid) for tid in token_ids]


__all__ = ["Phase", "load_dpo_pairs", "pair_to_dpo_inputs"]
