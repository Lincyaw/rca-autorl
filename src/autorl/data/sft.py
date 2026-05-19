"""SFT dataset builder for llmharness/distill outputs.

Consumes ``extractor.jsonl`` / ``auditor.jsonl`` produced by
``llmharness-distill export``.  Each row has the shape::

    {
      "phase": "extractor" | "auditor",
      "sample_id": "...",
      "root_session_id": "...",
      "turn_index": 0,
      "input":  {"system": "...", "user": "..."},
      "target": {"messages": [{"role": "assistant",
                               "content": "<think>...</think>\\n\\n",
                               "tool_calls": [{"type": "function",
                                               "function": {"name": "...",
                                                            "arguments": "..."}}]}]},
      "meta":   {...}
    }

The loader applies the tokenizer's chat template to the
``input`` system+user pair (prompt) and to the same pair concatenated
with ``target.messages`` (full sequence), and emits a ``loss_mask``
that only covers the assistant segment.  This keeps the ``<think>``
block inside the supervised range so the student learns to emit
reasoning before the tool call.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from datasets import Dataset

from .samples import load_manifest_samples


def build_sft_dataset_from_manifest(
    manifest_path: str,
    tokenizer: Any,
    *,
    max_length: int | None = None,
) -> Dataset:
    """Build a ``datasets.Dataset`` of ``{input_ids, loss_mask}`` rows."""
    samples = load_manifest_samples(manifest_path)
    rows: list[dict[str, Any]] = []
    for sample in samples:
        row = _convert_sample(sample, tokenizer=tokenizer)
        if max_length is None or len(row["input_ids"]) <= max_length:
            rows.append(row)
    if not rows:
        raise ValueError(
            f"SFT manifest produced no usable rows (max_length={max_length}): {manifest_path}"
        )
    return Dataset.from_list(rows)


def _convert_sample(sample: Mapping[str, Any], *, tokenizer: Any) -> dict[str, Any]:
    input_payload = sample.get("input")
    target_payload = sample.get("target")
    if not isinstance(input_payload, Mapping) or not isinstance(target_payload, Mapping):
        raise ValueError(
            "SFT row must carry 'input' and 'target' objects (llmharness/distill shape)"
        )

    system_text = str(input_payload.get("system", "")).rstrip()
    user_text = str(input_payload.get("user", "")).rstrip()
    if not system_text or not user_text:
        raise ValueError("SFT row 'input' requires both 'system' and 'user' text")

    target_messages = target_payload.get("messages")
    if not isinstance(target_messages, list) or not target_messages:
        raise ValueError("SFT row 'target.messages' must be a non-empty list")

    prompt_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    full_messages: list[dict[str, Any]] = [*prompt_messages, *target_messages]

    prompt_ids = _apply_chat_template(tokenizer, prompt_messages, add_generation_prompt=True)
    full_ids = _apply_chat_template(tokenizer, full_messages, add_generation_prompt=False)
    if len(full_ids) < len(prompt_ids):
        raise ValueError("full SFT sequence is shorter than prompt sequence")

    loss_mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
    return {
        "input_ids": full_ids,
        "loss_mask": loss_mask,
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


def coerce_target_to_string(sample: Mapping[str, Any]) -> str:
    """Legacy assistant-string fallback for non-thinking baselines.

    Some experiments train on plain assistant text instead of
    ``<think>`` + tool_calls — e.g. a non-thinking base model run on
    the same case set. This helper concatenates the assistant
    ``content`` with the tool_calls JSON payload so the row can still
    be consumed by a text-only SFT path.
    """
    target_messages = (sample.get("target") or {}).get("messages") or []
    if not target_messages:
        return ""
    asst = target_messages[0]
    parts: list[str] = []
    content = asst.get("content")
    if content:
        parts.append(str(content).rstrip())
    for tc in asst.get("tool_calls") or []:
        fn = (tc.get("function") or {})
        args = fn.get("arguments") or ""
        parts.append(f"<tool_call>{json.dumps({'name': fn.get('name'), 'arguments': args})}</tool_call>")
    return "\n".join(parts)
