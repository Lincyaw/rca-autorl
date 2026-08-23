"""RCA reward computation shared by training and evaluation."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from autorl.algorithm import EpisodeSignals, RCARewardConfig, compute_episode_reward


def verify_rca(
    sample: Mapping[str, Any],
    prediction: Any,
    *,
    data_dir: str | Path,
    has_submission: bool,
    reward_config: RCARewardConfig | None = None,
    tool_calls: int = 0,
    tokens: int = 0,
    elapsed_seconds: float = 0.0,
    redundant_actions: int = 0,
    invalid_actions: int = 0,
    declared_objects: int = 0,
    violation_count: int = 0,
) -> dict[str, float]:
    """Verify an AgentM result and compute the configured episode return."""
    graph_path = Path(data_dir) / "causal_graph_verified.json"
    if graph_path.is_file():
        cause_correct, attribution_score = _fpg_signals(prediction, graph_path)
    else:
        cause_correct, attribution_score = _legacy_signals(sample, prediction)

    signals = EpisodeSignals(
        cause_correct=cause_correct,
        attribution_score=attribution_score,
        violation_count=violation_count,
        tool_calls=tool_calls,
        tokens=tokens,
        elapsed_seconds=elapsed_seconds,
        redundant_actions=redundant_actions,
        invalid_actions=invalid_actions,
        declared_objects=declared_objects,
        has_submission=has_submission,
    )
    return compute_episode_reward(signals, reward_config or RCARewardConfig())


def _fpg_signals(prediction: Any, graph_path: Path) -> tuple[bool, float]:
    try:
        from fpg import ModelRCAOutput, Scenario, compare_model_to_ground_truth

        output = ModelRCAOutput.model_validate(prediction)
        scenario = Scenario.model_validate_json(graph_path.read_text(encoding="utf-8"))
        comparison = compare_model_to_ground_truth(output, scenario)
        root = comparison.root_subjects
        cause_correct = root.precision == 1.0 and root.recall == 1.0
        attribution_score = (
            float(comparison.subjects.recall) + float(comparison.soft_subject_edges.recall)
        ) / 2.0
        return cause_correct, attribution_score
    except Exception:
        # Invalid agent output is a failed episode, not a failed training job.
        return False, 0.0


def _legacy_signals(sample: Mapping[str, Any], prediction: Any) -> tuple[bool, float]:
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
    cause_correct = bool(service_hit and (fault_kind_hit or not fault))
    return cause_correct, 0.0


def _normalize(value: Any) -> str:
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(value or ""))
    return " ".join(text.lower().replace("_", " ").split())


__all__ = ["verify_rca"]
