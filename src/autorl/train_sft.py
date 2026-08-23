"""AReaL SFT launcher for distilled AgentM trajectories."""

from __future__ import annotations

import sys
from typing import Any

from areal import SFTTrainer
from areal.api.cli_args import SFTConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer

from autorl.data.sft import build_sft_dataset_from_manifest


def _build_dataset(config: SFTConfig, *, split: str, train: bool, tokenizer: Any) -> Any | None:
    dataset_config = config.train_dataset if train else config.valid_dataset
    if dataset_config is None:
        return None
    path = getattr(dataset_config, "path", None)
    if path and str(path).endswith((".json", ".jsonl")):
        return build_sft_dataset_from_manifest(
            str(path),
            tokenizer,
            max_length=getattr(dataset_config, "max_length", None),
        )
    return get_custom_dataset(
        split=split,
        dataset_config=dataset_config,
        tokenizer=tokenizer,
    )


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, SFTConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    train_dataset = _build_dataset(config, split="train", train=True, tokenizer=tokenizer)
    if train_dataset is None:
        raise ValueError("train_dataset must be configured")
    valid_dataset = _build_dataset(config, split="test", train=False, tokenizer=tokenizer)
    with SFTTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])
