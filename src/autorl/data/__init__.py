"""Dataset materialization helpers."""

from __future__ import annotations

from typing import Any

from .samples import load_manifest_samples


def build_train_dataset(config: Any):
    from .materialize import build_train_dataset as _build_train_dataset

    return _build_train_dataset(config)


def build_valid_dataset(config: Any):
    from .materialize import build_valid_dataset as _build_valid_dataset

    return _build_valid_dataset(config)


def build_sft_dataset_from_manifest(*args: Any, **kwargs: Any):
    from .sft import build_sft_dataset_from_manifest as _build_sft_dataset_from_manifest

    return _build_sft_dataset_from_manifest(*args, **kwargs)


__all__ = [
    "build_sft_dataset_from_manifest",
    "build_train_dataset",
    "build_valid_dataset",
    "load_manifest_samples",
]
