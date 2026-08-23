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
                                                            "arguments": "..."}}]},
                              {"role": "tool", "content": "..."},
                              {"role": "assistant", ...},
                              ...]},
      "meta":   {...}
    }

The loader applies the tokenizer's chat template to the ``input``
system+user pair (prompt, with ``add_generation_prompt=True``) and
then **incrementally** to the prompt concatenated with one additional
``target.messages`` entry at a time. The length delta between
consecutive tokenizations gives the per-message token span; only
spans belonging to **assistant** roles receive ``loss_mask=1``.

This matters under the v19 extractor flow: ``target.messages`` is
a multi-turn ``[assistant, tool, assistant, tool, ..., assistant]``
sequence. Tool messages are deterministic witness output that the
student must NOT learn to generate — supervising tool tokens is wrong
signal. The ``<think>`` block stays inside the supervised range
(it sits inside each assistant message) so the student still learns
to emit reasoning before each tool call.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
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
    num_proc = _tokenize_num_proc()
    if num_proc > 1:
        rows = _convert_samples_parallel(
            samples, tokenizer, max_length=max_length, num_proc=num_proc
        )
    else:
        rows = []
        for sample in samples:
            row = _convert_sample(sample, tokenizer=tokenizer)
            if max_length is None or len(row["input_ids"]) <= max_length:
                rows.append(row)
    if not rows:
        raise ValueError(
            f"SFT manifest produced no usable rows (max_length={max_length}): {manifest_path}"
        )
    return Dataset.from_list(rows)


_WORKER_TOKENIZER: Any | None = None
_WORKER_MAX_LENGTH: int | None = None


def _tokenize_num_proc() -> int:
    raw = os.environ.get("SFT_TOKENIZE_NUM_PROC", "1")
    try:
        value = int(raw)
    except ValueError:
        value = 1
    return max(1, value)


def _convert_samples_parallel(
    samples: list[Mapping[str, Any]],
    tokenizer: Any,
    *,
    max_length: int | None,
    num_proc: int,
) -> list[dict[str, Any]]:
    if os.name != "posix":
        raise ValueError("SFT_TOKENIZE_NUM_PROC>1 requires POSIX fork semantics")

    global _WORKER_TOKENIZER, _WORKER_MAX_LENGTH
    _WORKER_TOKENIZER = tokenizer
    _WORKER_MAX_LENGTH = max_length

    chunksize = max(1, int(os.environ.get("SFT_TOKENIZE_CHUNKSIZE", "8")))
    with mp.get_context("fork").Pool(processes=num_proc) as pool:
        converted = pool.imap(_convert_sample_worker, samples, chunksize=chunksize)
        return [row for row in converted if row is not None]


def _convert_sample_worker(sample: Mapping[str, Any]) -> dict[str, Any] | None:
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("SFT tokenizer worker was not initialized")
    row = _convert_sample(sample, tokenizer=_WORKER_TOKENIZER)
    if _WORKER_MAX_LENGTH is not None and len(row["input_ids"]) > _WORKER_MAX_LENGTH:
        return None
    return row


def _convert_sample(sample: Mapping[str, Any], *, tokenizer: Any) -> dict[str, Any]:
    input_payload = sample.get("input")
    target_payload = sample.get("target")
    if not isinstance(input_payload, Mapping) or not isinstance(
        target_payload, Mapping
    ):
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

    # The prompt's add_generation_prompt=True suffix (e.g. Qwen's
    # ``<|im_start|>assistant\n<think>\n``) is rendered template-side; we
    # treat those tokens as part of the prompt range (mask=0) so the
    # student only learns the assistant *body*, not the role header it
    # never had to emit. Subsequent assistant turns inside a multi-turn
    # target (after a tool message) carry their own header tokens, which
    # the student DOES need to emit — those go into the supervised span.
    prompt_ids = _apply_chat_template(
        tokenizer, prompt_messages, add_generation_prompt=True
    )

    if _has_tool_turns(target_messages):
        full_ids, loss_mask = _convert_with_role_offsets(
            tokenizer,
            prompt_messages=prompt_messages,
            target_messages=target_messages,
        )
    else:
        try:
            full_ids, loss_mask = _convert_with_prefix_deltas(
                tokenizer,
                prompt_messages=prompt_messages,
                target_messages=target_messages,
                prompt_ids=prompt_ids,
            )
        except ValueError as exc:
            if "prefix-stable" not in str(exc):
                raise
            full_ids, loss_mask = _convert_with_role_offsets(
                tokenizer,
                prompt_messages=prompt_messages,
                target_messages=target_messages,
            )

    if len(loss_mask) != len(full_ids):
        raise ValueError(
            "loss_mask length drifted from input_ids length "
            f"({len(loss_mask)} vs {len(full_ids)}) — internal bug"
        )
    return {
        "input_ids": full_ids,
        "loss_mask": loss_mask,
    }


def _has_tool_turns(messages: list[dict[str, Any]]) -> bool:
    return any(
        str(msg.get("role") or "").lower() == "tool" or bool(msg.get("tool_calls"))
        for msg in messages
    )


def _convert_with_prefix_deltas(
    tokenizer: Any,
    *,
    prompt_messages: list[dict[str, Any]],
    target_messages: list[dict[str, Any]],
    prompt_ids: list[int],
) -> tuple[list[int], list[int]]:
    # Incrementally tokenize prompt + first k target messages, k = 1..N.
    # Length delta between consecutive tokenizations is the token span
    # of message k. Supervise only spans whose role is "assistant".
    cur_ids = prompt_ids
    loss_mask: list[int] = [0] * len(prompt_ids)
    full_ids: list[int] = list(prompt_ids)
    for k, msg in enumerate(target_messages, start=1):
        next_ids = _apply_chat_template(
            tokenizer,
            [*prompt_messages, *target_messages[:k]],
            add_generation_prompt=False,
        )
        if len(next_ids) < len(cur_ids):
            raise ValueError(
                "incremental tokenization shrank the SFT sequence at "
                f"target message {k}; tokenizer chat template is non-monotonic"
            )
        # Cross-check that the previous tokenization is a strict prefix.
        # If a tokenizer rewrites earlier tokens when later messages are
        # appended, the per-message mask attribution would be wrong.
        if next_ids[: len(cur_ids)] != cur_ids:
            raise ValueError(
                "tokenizer chat template is not prefix-stable across "
                f"turns; cannot derive per-message loss mask (turn {k})"
            )
        span = next_ids[len(cur_ids) :]
        role = str(msg.get("role") or "").lower()
        mask_bit = 1 if role == "assistant" else 0
        loss_mask.extend([mask_bit] * len(span))
        full_ids = next_ids
        cur_ids = next_ids
    return full_ids, loss_mask


def _convert_with_role_offsets(
    tokenizer: Any,
    *,
    prompt_messages: list[dict[str, Any]],
    target_messages: list[dict[str, Any]],
) -> tuple[list[int], list[int]]:
    """Fallback for Qwen tool templates that rewrite prior assistant turns.

    Qwen3 renders an assistant tool-call differently once the following
    tool response is present, so token-prefix deltas are not stable. The
    final rendered chat still has explicit role delimiters; use character
    offsets to mask assistant message bodies and leave user/tool messages
    unsupervised.
    """

    full_text = _apply_chat_template_text(
        tokenizer,
        [*prompt_messages, *target_messages],
        add_generation_prompt=False,
    )
    prompt_text = _apply_chat_template_text(
        tokenizer,
        prompt_messages,
        add_generation_prompt=True,
    )
    if not full_text.startswith(prompt_text):
        raise ValueError(
            "tokenizer chat template is not prefix-stable and rendered prompt "
            "is not a string prefix of the full chat; cannot derive loss mask"
        )

    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    full_ids = encoded["input_ids"]
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise ValueError(
            "tokenizer must return offset_mapping when chat template is not prefix-stable"
        )
    if hasattr(full_ids, "tolist"):
        full_ids = full_ids.tolist()

    assistant_spans = _qwen_assistant_spans(full_text, start_at=len(prompt_text))
    loss_mask: list[int] = []
    span_idx = 0
    for start, end in offsets:
        while span_idx < len(assistant_spans) and assistant_spans[span_idx][1] <= start:
            span_idx += 1
        supervised = (
            span_idx < len(assistant_spans)
            and end > assistant_spans[span_idx][0]
            and start < assistant_spans[span_idx][1]
        )
        loss_mask.append(1 if supervised else 0)

    return [int(tid) for tid in full_ids], loss_mask


def _qwen_assistant_spans(text: str, *, start_at: int) -> list[tuple[int, int]]:
    marker = "<|im_start|>assistant\n"
    end_marker = "<|im_end|>"
    spans: list[tuple[int, int]] = []
    pos = start_at
    preceding_role_start = text.rfind(marker, 0, start_at)
    if preceding_role_start >= 0:
        preceding_body_start = preceding_role_start + len(marker)
        preceding_body_end = text.find(end_marker, preceding_body_start)
        if preceding_body_start <= start_at < preceding_body_end:
            spans.append((start_at, preceding_body_end))
            pos = preceding_body_end + len(end_marker)
    while True:
        role_start = text.find(marker, pos)
        if role_start < 0:
            return spans
        body_start = role_start + len(marker)
        body_end = text.find(end_marker, body_start)
        if body_end < 0:
            raise ValueError("assistant message missing <|im_end|> in rendered chat")
        spans.append((body_start, body_end))
        pos = body_end + len(end_marker)


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


def _apply_chat_template_text(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> str:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    if not isinstance(text, str):
        raise ValueError("tokenizer chat template did not return text")
    return text


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
        fn = tc.get("function") or {}
        args = fn.get("arguments") or ""
        parts.append(
            f"<tool_call>{json.dumps({'name': fn.get('name'), 'arguments': args})}</tool_call>"
        )
    return "\n".join(parts)
