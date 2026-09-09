from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """The objects of a JSONL file: a manifest, a session log, an export."""
    path = Path(path)
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{index} must be a JSON object")
            samples.append(row)
    if not samples:
        raise ValueError(f"empty: {path}")
    return samples
