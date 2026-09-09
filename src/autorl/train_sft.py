"""AReaL SFT launcher for distilled AgentM trajectories."""

from __future__ import annotations

import sys

from areal import SFTTrainer
from areal.api.cli_args import SFTConfig, load_expr_config
from areal.utils.hf_utils import load_hf_tokenizer

from autorl.data.sft import build_sft_dataset_from_manifest


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, SFTConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    train_dataset = build_sft_dataset_from_manifest(
        config.train_dataset.path, tokenizer, max_length=config.train_dataset.max_length
    )
    valid_dataset = (
        build_sft_dataset_from_manifest(
            config.valid_dataset.path, tokenizer, max_length=config.valid_dataset.max_length
        )
        if config.valid_dataset is not None
        else None
    )
    with SFTTrainer(config, train_dataset=train_dataset, valid_dataset=valid_dataset) as trainer:
        trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])
