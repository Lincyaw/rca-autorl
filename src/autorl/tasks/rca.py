from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from autorl.rca_utils import normalize_root_causes, root_cause_key_set
from autorl.contracts import AgentInput, TaskOutcome, TaskSample, Trajectory

from .base import TaskAdapter

_ID_KEYS = ("sample_id", "case_id", "qid", "query_id", "id")
_INCIDENT_KEYS = ("incident", "question", "prompt", "task_description", "task")
_DATA_DIR_KEYS = ("data_dir", "case_dir", "observability_dir")
_REFERENCE_KEYS = ("answer", "root_causes", "ground_truth", "reference")
_RESERVED_KEYS = set(_ID_KEYS + _INCIDENT_KEYS + _DATA_DIR_KEYS + _REFERENCE_KEYS + ("messages",))


class RCATaskAdapter(TaskAdapter):
    """Normalize RCA samples for AgentM-backed investigations."""

    task_type = "rca"

    def validate_sample(self, raw_sample: Mapping[str, Any]) -> TaskSample:
        incident = _get_required_text(raw_sample, _INCIDENT_KEYS, "incident")
        data_dir = _get_required_text(raw_sample, _DATA_DIR_KEYS, "data_dir")
        sample_id = _get_sample_id(raw_sample)
        reference = normalize_root_causes(_extract_reference(raw_sample))

        metadata = {
            key: value
            for key, value in raw_sample.items()
            if key not in _RESERVED_KEYS
        }

        target = {"root_causes": reference} if reference else None
        return TaskSample(
            sample_id=sample_id,
            task_type=self.task_type,
            input={"incident": incident, "data_dir": data_dir},
            target=target,
            reference=target,
            metadata=metadata,
            raw=dict(raw_sample),
        )

    def to_agent_input(self, sample: TaskSample) -> AgentInput:
        incident = str(sample.input.get("incident", ""))
        data_dir = str(sample.input.get("data_dir", ""))
        messages_raw = sample.raw.get("messages")
        if isinstance(messages_raw, list) and messages_raw:
            messages = list(messages_raw)
        else:
            messages = [{"role": "user", "content": incident}]

        return AgentInput(
            sample_id=sample.sample_id,
            task_type=sample.task_type,
            instruction="Investigate the RCA case and return a causal graph.",
            context={"incident": incident, "data_dir": data_dir},
            messages=messages,
            available_tools=[],
            constraints={},
            metadata=dict(sample.metadata),
            raw_sample=dict(sample.raw),
        )

    def to_task_outcome(self, sample: TaskSample, trajectory: Trajectory) -> TaskOutcome:
        prediction = normalize_root_causes(trajectory.final_output.get("prediction"))
        reference = normalize_root_causes(sample.reference)
        prediction_keys = root_cause_key_set(prediction)
        reference_keys = root_cause_key_set(reference)
        success = bool(prediction_keys) and bool(reference_keys) and prediction_keys == reference_keys

        metrics = dict(trajectory.summary_stats)
        metrics["predicted_root_causes"] = float(len(prediction_keys))
        metrics["reference_root_causes"] = float(len(reference_keys))
        metrics["matched_root_causes"] = float(len(prediction_keys & reference_keys))

        return TaskOutcome(
            sample_id=sample.sample_id,
            task_type=sample.task_type,
            prediction=prediction,
            reference=reference,
            success=success if reference else None,
            termination_reason=str(trajectory.final_output.get("termination", trajectory.status.value)),
            metrics=metrics,
            metadata=dict(sample.metadata),
        )


def _get_sample_id(raw_sample: Mapping[str, Any]) -> str:
    for key in _ID_KEYS:
        value = raw_sample.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return uuid4().hex


def _get_required_text(
    raw_sample: Mapping[str, Any],
    keys: tuple[str, ...],
    label: str,
) -> str:
    for key in keys:
        value = raw_sample.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError(f"rca sample missing required key: {label}")


def _extract_reference(raw_sample: Mapping[str, Any]) -> Any:
    for key in _REFERENCE_KEYS:
        if key in raw_sample:
            return raw_sample[key]
    return None
