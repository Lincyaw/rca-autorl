"""AReaL GRPO/PPO launcher for the AgentM RCA workflow."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from areal import PPOTrainer
from areal.api.cli_args import PPOConfig, load_expr_config
from datasets import Dataset


@dataclass
class AgentMConfig:
    scenario: str = "rca"
    model: str = "default"
    dataset_root: str = ""
    max_turns: int = 128
    timeout: float = 1800.0


@dataclass
class RCAPPOConfig(PPOConfig):
    econfig: AgentMConfig = field(default_factory=AgentMConfig)


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
    train_dataset = load_rca_dataset(config.train_dataset.path)
    valid_dataset = (
        load_rca_dataset(config.valid_dataset.path)
        if config.valid_dataset is not None
        else None
    )
    workflow_kwargs = {"econfig": asdict(config.econfig)}

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="autorl.agent.AgentMWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="autorl.agent.AgentMWorkflow"
            if valid_dataset is not None
            else None,
            eval_workflow_kwargs=workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
