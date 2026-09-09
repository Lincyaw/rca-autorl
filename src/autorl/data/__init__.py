"""Dataset helpers used by the SFT entrypoint."""

from .samples import read_jsonl
from .sft import build_sft_dataset_from_manifest

__all__ = ["build_sft_dataset_from_manifest", "read_jsonl"]
