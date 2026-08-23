"""RCA reward computation shared by training and evaluation."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def verify_rca(
    sample: Mapping[str, Any],
    prediction: Any,
    *,
    data_dir: str | Path,
    has_submission: bool,
) -> dict[str, float]:
    """Score an AgentM RCA result, preferring the canonical FPG grader."""
    graph_path = Path(data_dir) / "causal_graph_verified.json"
    if graph_path.is_file():
        score = _fpg_score(prediction, graph_path)
        return {
            "reward": score,
            "fpg_score": score,
            "has_submission": float(has_submission),
        }

    service_hit, fault_kind_hit = _legacy_score(sample, prediction)
    return {
        "reward": 0.7 * service_hit + 0.3 * fault_kind_hit,
        "service_hit": service_hit,
        "fault_kind_hit": fault_kind_hit,
        "has_submission": float(has_submission),
    }


def _fpg_score(prediction: Any, graph_path: Path) -> float:
    try:
        from fpg import ModelRCAOutput, Scenario, compare_model_to_ground_truth

        output = ModelRCAOutput.model_validate(prediction)
        scenario = Scenario.model_validate_json(graph_path.read_text(encoding="utf-8"))
        comparison = compare_model_to_ground_truth(output, scenario)
        return float(comparison.score)
    except Exception:
        # Invalid agent output is a failed episode, not a failed training job.
        return 0.0


def _legacy_score(sample: Mapping[str, Any], prediction: Any) -> tuple[float, float]:
    expected_services = sample.get("expected_services") or sample.get("ground_truth") or []
    if not isinstance(expected_services, list):
        expected_services = []
    expected_fault = sample.get("fault_kind") or sample.get("fault_type") or ""

    blob = _normalize(json.dumps(prediction, ensure_ascii=False, default=str))
    service_hit = float(
        any(_normalize(service) in blob for service in expected_services if service)
    )
    fault = _normalize(expected_fault)
    fault_kind_hit = float(bool(fault) and fault in blob)
    return service_hit, fault_kind_hit


def _normalize(value: Any) -> str:
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(value or ""))
    return " ".join(text.lower().replace("_", " ").split())


__all__ = ["verify_rca"]
