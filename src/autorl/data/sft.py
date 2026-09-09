"""SFT dataset builder for exported DeepSeek Harness sessions.

Consumes the rows `autorl.data.export` writes — one conversation per session,
with the tool schemas the teacher was offered::

    {"phase": "rca" | "checkpoint", "sample_id": ..., "root_session_id": ...,
     "messages": [system, user, assistant, tool, user, assistant, ...],
     "tools": [...], "first_supervised": k, "meta": {...}}

and produces AReaL's SFT rows, `{"input_ids", "loss_mask"}`.

**One row per assistant turn from `first_supervised` on.** Messages before
that index were written on an earlier context and are supervised there; here
they are the prompt only. For turn *k* the prompt is
`apply_chat_template(messages[:k], tools=..., add_generation_prompt=True)` and
the sequence is `apply_chat_template(messages[:k+1], ...)`; the mask is 0 over
the prompt and 1 over the rest. That is the same shape AReaL's own `gsm8k` SFT
path uses — prompt versus prompt-plus-answer — applied per turn.

Per turn rather than once per trajectory, because a thinking model's template
renders a turn's reasoning only when the turn is after the conversation's last
`user`-role message (Qwen3's `ns.last_query_index` gate). Rendered whole, a
trajectory keeps `<think>` on its last turns and silently drops it from every
earlier one — 44 turns down to 2 on the longest episode collected so far. Per
turn, each supervised turn is the last message, so its reasoning always
renders, and the context before it renders exactly as the rollout will render
it: history reasoning stripped where the template strips it, the harness's own
`user` turns present, tools declared. Training and inference see one rendering,
not two.

The mask boundary is a character offset, not a token count: the template is
free to merge the token at the prompt/turn seam, and on the sessions collected
so far it does for 7 turns in 190. `offset_mapping` puts the boundary where the
template put it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from datasets import Dataset

from .samples import read_jsonl


def build_sft_dataset_from_manifest(
    manifest_path: str,
    tokenizer: Any,
    *,
    max_length: int | None = None,
) -> Dataset:
    """Build a ``datasets.Dataset`` of ``{input_ids, loss_mask}`` rows."""
    rows = [
        row
        for sample in read_jsonl(manifest_path)
        for row in convert_sample(sample, tokenizer=tokenizer, max_length=max_length)
    ]
    if not rows:
        raise TypeError(
            f"SFT manifest produced no usable rows (max_length={max_length}): {manifest_path}"
        )
    return Dataset.from_list(rows)


def convert_sample(
    sample: Mapping[str, Any],
    *,
    tokenizer: Any,
    max_length: int | None = None,
) -> list[dict[str, Any]]:
    """Every supervisable assistant turn of one conversation, as AReaL rows."""
    messages = sample.get("messages")
    if not isinstance(messages, list) or not messages:
        raise TypeError("SFT row must carry a non-empty 'messages' list")
    tools = sample.get("tools") or None
    first_supervised = int(sample.get("first_supervised", 0))

    rows: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if index < first_supervised:
            continue
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        row = _turn_row(tokenizer, messages, index, tools)
        if max_length is None or len(row["input_ids"]) <= max_length:
            rows.append(row)
    return rows


def _turn_row(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    index: int,
    tools: Any,
) -> dict[str, Any]:
    prompt = _render(tokenizer, messages[:index], tools=tools, add_generation_prompt=True)
    full = _render(tokenizer, messages[: index + 1], tools=tools, add_generation_prompt=False)
    if not full.startswith(prompt):
        raise ValueError(
            f"chat template did not extend the turn-{index} prompt; "
            "the generation prompt is not a prefix of the rendered turn"
        )

    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = encoded["input_ids"]
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise ValueError("tokenizer must return offset_mapping to place the loss boundary")
    boundary = len(prompt)
    return {
        "input_ids": [int(token) for token in input_ids],
        "loss_mask": [1 if start >= boundary else 0 for start, _ in offsets],
    }


def _render(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Any,
    add_generation_prompt: bool,
) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("tokenizer must support apply_chat_template (Qwen/GLM thinking models do)")
    text = tokenizer.apply_chat_template(
        list(messages),
        tools=tools,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    if not isinstance(text, str):
        raise TypeError("tokenizer chat template did not return text")
    return text


__all__ = ["build_sft_dataset_from_manifest", "convert_sample"]
