from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _validate_sample(sample: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    if not isinstance(sample, Mapping):
        raise TypeError(f"manifest sample #{index} must be a JSON object")
    return dict(sample)


def load_manifest_samples(manifest_path: str | Path) -> list[dict[str, Any]]:
    """Load manifest samples from JSON/JSONL without task-specific validation."""

    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"manifest file not found: {path}")

    suffix = path.suffix.lower()
    samples: list[dict[str, Any]] = []

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                samples.append(_validate_sample(obj, index=idx))
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            obj = json.load(handle)
        rows: Any = obj.get("samples") if isinstance(obj, Mapping) else obj
        if not isinstance(rows, list):
            raise ValueError("JSON manifest must be a list or an object with key 'samples'")
        for idx, row in enumerate(rows, start=1):
            samples.append(_validate_sample(row, index=idx))
    else:
        raise ValueError(f"unsupported manifest format: {path.suffix}; use .json or .jsonl")

    if not samples:
        raise ValueError(f"manifest contains no samples: {path}")

    return samples
