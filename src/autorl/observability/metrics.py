from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class MetricSetup:
    run_id: str
    run_name: str
    phase: str
    metric_names: dict[str, str]
    metadata_path: str | None


def setup_metrics(
    *,
    run_name: str | None = None,
    phase: str = "train",
    metadata_dir: str | Path | None = "artifacts/metadata",
    extra_tags: dict[str, Any] | None = None,
) -> MetricSetup:
    """Build a minimal project metric contract and persist run metadata.

    The helper keeps observability lightweight: it standardizes metric names and
    writes a tiny metadata record that can be consumed by scripts or dashboards.
    """

    run_id = uuid4().hex[:12]
    phase_slug = phase.strip().lower() or "train"
    resolved_run_name = run_name or f"{phase_slug}-{run_id}"

    metric_names = {
        "reward": f"autorl.{phase_slug}.reward",
        "turns": f"autorl.{phase_slug}.turns",
        "num_search": f"autorl.{phase_slug}.num_search",
        "num_access": f"autorl.{phase_slug}.num_access",
    }

    metadata_path: str | None = None
    if metadata_dir:
        out_dir = Path(metadata_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run_id,
            "run_name": resolved_run_name,
            "phase": phase_slug,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metric_names": metric_names,
            "tags": extra_tags or {},
        }
        target = out_dir / f"{resolved_run_name}.json"
        target.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        metadata_path = str(target)

    return MetricSetup(
        run_id=run_id,
        run_name=resolved_run_name,
        phase=phase_slug,
        metric_names=metric_names,
        metadata_path=metadata_path,
    )
