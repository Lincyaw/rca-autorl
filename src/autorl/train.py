"""AReaL GRPO/PPO launcher for the DeepSeek Harness RCA workflow."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from datasets import Dataset

from autorl.algorithm import validate_advantage_config
from autorl.config import RCAPPOConfig


def load_rca_dataset(path: str) -> Dataset:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"dataset row {line_number} must be a JSON object")
            if not any(row.get(key) for key in ("incident", "question", "prompt")):
                raise ValueError(f"dataset row {line_number} has no incident/question")
            records.append(row)
    if not records:
        raise ValueError(f"dataset is empty: {path}")
    return Dataset.from_list(records)


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, RCAPPOConfig)
    validate_advantage_config(config)
    train_dataset = load_rca_dataset(config.train_dataset.path)
    valid_dataset = (
        load_rca_dataset(config.valid_dataset.path) if config.valid_dataset is not None else None
    )
    # The served model name and the generation limits are AReaL's to declare,
    # not ours to duplicate: rollout.model is the name the gateway routes on,
    # gconfig.max_new_tokens is what the harness sends as the request's
    # `max_tokens`, and sglang.context_length is the window compaction must
    # trigger below. Passing them through keeps one number per concept.
    econfig: dict[str, Any] = {
        **asdict(config.econfig),
        "model": str(config.rollout.model),
        "max_tokens": int(config.gconfig.max_new_tokens),
        "context_window": int(config.sglang.context_length or config.gconfig.max_tokens),
        # The harness sends no sampling parameters of its own; this is how the
        # rollout comes to sample at the temperature the actor is trained at.
        "temperature": float(config.gconfig.temperature),
        # The workflow returns differences between turn values; AReaL turns
        # them back into values with this discount. One number, declared once.
        "turn_discount": float(config.rollout.agent.turn_discount),
    }
    workflow_kwargs = {"econfig": econfig}

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="autorl.agent.DshWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="autorl.agent.DshWorkflow" if valid_dataset is not None else None,
            eval_workflow_kwargs=workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
