from __future__ import annotations

import sys

from areal import SFTTrainer
from areal.api.cli_args import SFTConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer

from autorl.data.sft import build_sft_dataset_from_manifest


def _is_manifest_path(path: str | None) -> bool:
    return bool(path) and str(path).endswith((".json", ".jsonl"))


def _build_dataset(config: SFTConfig, *, split: str, train: bool, tokenizer):
    dataset_cfg = config.train_dataset if train else config.valid_dataset
    if dataset_cfg is None:
        return None

    dataset_path = getattr(dataset_cfg, "path", None)
    max_length = getattr(dataset_cfg, "max_length", None)
    if _is_manifest_path(dataset_path):
        return build_sft_dataset_from_manifest(
            str(dataset_path),
            tokenizer,
            max_length=max_length,
        )

    return get_custom_dataset(
        split=split,
        dataset_config=dataset_cfg,
        tokenizer=tokenizer,
    )


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, SFTConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = _build_dataset(config, split="train", train=True, tokenizer=tokenizer)
    if train_dataset is None:
        raise ValueError("train_dataset must be configured for SFT training")

    valid_dataset = None
    if getattr(config, "valid_dataset", None) is not None:
        valid_dataset = _build_dataset(config, split="test", train=False, tokenizer=tokenizer)

    with SFTTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])
