from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download

from areal.infra.data_service.rdataset import RDataset

from .samples import load_manifest_samples


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _resolve_json_data_file(path: str, *, revision: str | None = None) -> str:
    """Resolve JSON/JSONL dataset file path.

    Supports local files and Hugging Face dataset shorthand like
    `owner__repo/filename.jsonl`.
    """

    candidate = Path(path)
    if candidate.exists():
        return str(candidate)

    if "__" in path and "/" in path:
        repo_shorthand, filename = path.split("/", 1)
        repo_id = repo_shorthand.replace("__", "/")
        try:
            return hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                revision=revision,
            )
        except Exception as err:  # pragma: no cover
            raise FileNotFoundError(
                f"Dataset file '{path}' is not local and download from '{repo_id}/{filename}' failed: {err}"
            ) from err

    return path


def _as_json_dataset(path: str, *, revision: str | None = None) -> Dataset:
    data_file = _resolve_json_data_file(path, revision=revision)
    return load_dataset(path="json", data_files=data_file, split="train")


def _load_dataset_from_cfg(
    dataset_cfg: Any,
    *,
    default_split: str,
    dataset_name: str,
    manifest_path_override: str | None = None,
    revision_override: str | None = None,
) -> Dataset | RDataset:
    manifest_path = manifest_path_override or _cfg_get(dataset_cfg, "manifest_path")
    revision = revision_override if revision_override is not None else _cfg_get(dataset_cfg, "revision")
    split = _cfg_get(dataset_cfg, "split", default_split)

    if manifest_path:
        samples = load_manifest_samples(manifest_path)
        return Dataset.from_list(samples)

    dataset_path = _cfg_get(dataset_cfg, "path")
    if not dataset_path:
        raise ValueError(f"{dataset_name}.path is required when manifest_path is not set")

    if isinstance(dataset_path, str) and dataset_path.endswith((".json", ".jsonl")):
        return _as_json_dataset(dataset_path, revision=revision)
    return load_dataset(path=dataset_path, split=split, revision=revision)


def build_train_dataset(config: Any) -> Dataset | RDataset:
    dataset_cfg = config.train_dataset
    return _load_dataset_from_cfg(
        dataset_cfg,
        default_split="train",
        dataset_name="train_dataset",
        manifest_path_override=getattr(config, "train_manifest_path", None),
        revision_override=getattr(config, "train_dataset_revision", None),
    )


def build_valid_dataset(config: Any) -> Dataset | RDataset | None:
    valid_cfg = getattr(config, "valid_dataset", None)
    if valid_cfg is None and not getattr(config, "valid_manifest_path", None):
        return None

    return _load_dataset_from_cfg(
        valid_cfg,
        default_split="test",
        dataset_name="valid_dataset",
        manifest_path_override=getattr(config, "valid_manifest_path", None),
        revision_override=getattr(config, "valid_dataset_revision", None),
    )
