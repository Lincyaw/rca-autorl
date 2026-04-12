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
    samples = load_manifest_samples(manifest_path)
    rows: list[dict[str, Any]] = []
    for sample in samples:
        row = _convert_sample(sample, tokenizer=tokenizer)
        if max_length is None or len(row["input_ids"]) <= max_length:
            rows.append(row)
    if not rows:
        raise ValueError(f"SFT manifest produced no usable rows: {manifest_path}")
    return Dataset.from_list(rows)


def _convert_sample(sample: Mapping[str, Any], *, tokenizer: Any) -> dict[str, Any]:
    messages = sample.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("SFT manifest sample requires non-empty 'messages'")

    assistant_response = sample.get("assistant_response")
    if assistant_response is None:
        if "answer" in sample:
            assistant_response = json.dumps(sample["answer"], ensure_ascii=False, sort_keys=True)
        else:
            raise ValueError("SFT manifest sample requires 'assistant_response' or 'answer'")
    assistant_text = str(assistant_response)

    prompt_ids = _apply_chat_template(tokenizer, messages, add_generation_prompt=True)
    full_ids = _apply_chat_template(
        tokenizer,
        [*messages, {"role": "assistant", "content": assistant_text}],
        add_generation_prompt=False,
    )
    if len(full_ids) < len(prompt_ids):
        raise ValueError("full SFT sequence is shorter than prompt sequence")

    loss_mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
    return {
        "input_ids": full_ids,
        "loss_mask": loss_mask,
    }


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> list[int]:
    if hasattr(tokenizer, "apply_chat_template"):
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
        return list(token_ids)

    text = "\n".join(f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in messages)
    if add_generation_prompt:
        text += "\nassistant:"
    encoded = tokenizer.encode(text)
    return list(encoded)
