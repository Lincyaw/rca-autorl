from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonlTraceSink:
    """Best-effort JSONL sink for canonical runtime artifacts."""

    def __init__(self, output_dir: str | Path | None):
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def emit(self, *, name: str, payload: Any) -> str | None:
        if self.output_dir is None:
            return None
        target = self.output_dir / f"{name}.jsonl"
        serializable = _to_jsonable(payload)
        record = {
            "emitted_at": datetime.now(timezone.utc).isoformat(),
            "payload": serializable,
        }
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
            handle.write("\n")
        return str(target)


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    return value
