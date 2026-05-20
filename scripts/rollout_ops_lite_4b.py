#!/usr/bin/env python3
"""Roll out an ops-lite RCA trajectory SFT checkpoint on one jsonl row."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--data",
        default=Path(".runs/data/ops-lite-trajectories/extractor.jsonl"),
        type=Path,
    )
    parser.add_argument("--row", default=62, type=int)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--max-new-tokens", default=1800, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repetition-penalty", default=1.0, type=float)
    parser.add_argument("--no-repeat-ngram-size", default=0, type=int)
    return parser.parse_args()


def load_row(path: Path, row_id: int) -> dict:
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx == row_id:
                return json.loads(line)
    raise IndexError(f"row {row_id} not found in {path}")


def analyze(text: str) -> dict:
    match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)
    info = {
        "closes_think": "</think>" in text,
        "has_tool_call": "<tool_call>" in text,
        "has_submit_events": "submit_events" in text,
        "valid_json": False,
        "event_count": None,
        "json_error": None,
    }
    if not match:
        return info
    try:
        payload = json.loads(match.group(1).strip())
        events = payload.get("arguments", {}).get("events")
        info["valid_json"] = True
        info["event_count"] = len(events) if isinstance(events, list) else None
    except Exception as exc:  # noqa: BLE001 - report parser error for debugging rollout quality.
        info["json_error"] = str(exc)
    return info


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    row = load_row(args.data, args.row)
    messages = [
        {"role": "system", "content": row["input"]["system"].rstrip()},
        {"role": "user", "content": row["input"]["user"].rstrip()},
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_path = args.outdir / f"row{args.row}_prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    if row.get("target"):
        (args.outdir / f"row{args.row}_target.txt").write_text(
            row["target"]["messages"][0]["content"], encoding="utf-8"
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        dtype=torch.bfloat16,
        device_map={"": args.device},
        trust_remote_code=True,
    )
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = output[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    rollout_path = args.outdir / f"row{args.row}_rollout.txt"
    rollout_path.write_text(text, encoding="utf-8")

    summary = analyze(text)
    summary["new_tokens"] = int(gen_ids.shape[0])
    summary["prompt_tokens"] = int(inputs["input_ids"].shape[1])
    summary["rollout_path"] = str(rollout_path)
    (args.outdir / f"row{args.row}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
