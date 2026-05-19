"""TaskAdapter for RCA cases that target the new AgentM ``rca:harness.sync``
scenario.

Sample shape (one JSONL row per case)::

    {
      "id":                "ts9-...",
      "incident":          "<natural-language incident description>",
      "data_dir":          "/mnt/.../rcabench/<case>",
      "expected_services": ["ts-order-service", ...],
      "fault_kind":        "cpu_stress",
      "root_causes":       [...]            # optional richer GT
    }

The adapter is intentionally minimal: it carries ``incident`` /
``data_dir`` into :class:`AgentInput.context` (the AgentMRuntime reads
them from there) and stashes the ground-truth labels on the
``reference`` field for the reward strategy to consume. The prediction
shape comes back from :class:`AgentMRuntime` as an
``AgentRCAOutput``-shaped dict in ``trajectory.final_output['prediction']``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from autorl.contracts import AgentInput, TaskOutcome, TaskSample, Trajectory

from .base import TaskAdapter

_ID_KEYS = ("sample_id", "case_id", "qid", "query_id", "id")
_INCIDENT_KEYS = ("incident", "question", "prompt", "task_description", "task")
_DATA_DIR_KEYS = ("data_dir", "case_dir", "observability_dir")
_RESERVED = set(
    _ID_KEYS
    + _INCIDENT_KEYS
    + _DATA_DIR_KEYS
    + ("expected_services", "fault_kind", "root_causes", "ground_truth", "messages")
)


class RCATaskAdapter(TaskAdapter):
    """Normalize rcabench-shaped samples for AgentM-backed RCA rollouts."""

    task_type = "rca"

    def validate_sample(self, raw_sample: Mapping[str, Any]) -> TaskSample:
        incident = _required_text(raw_sample, _INCIDENT_KEYS, "incident")
        data_dir = _required_text(raw_sample, _DATA_DIR_KEYS, "data_dir")
        sample_id = _pick_id(raw_sample)

        reference = _extract_reference(raw_sample)
        metadata = {k: v for k, v in raw_sample.items() if k not in _RESERVED}

        return TaskSample(
            sample_id=sample_id,
            task_type=self.task_type,
            input={"incident": incident, "data_dir": data_dir},
            target=reference,
            reference=reference,
            metadata=metadata,
            raw=dict(raw_sample),
        )

    def to_agent_input(self, sample: TaskSample) -> AgentInput:
        incident = str(sample.input.get("incident", ""))
        data_dir = str(sample.input.get("data_dir", ""))
        return AgentInput(
            sample_id=sample.sample_id,
            task_type=sample.task_type,
            instruction="Investigate the RCA case and submit a root-cause verdict.",
            context={"incident": incident, "data_dir": data_dir},
            messages=[{"role": "user", "content": incident}],
            available_tools=[],
            constraints={},
            metadata=dict(sample.metadata),
            raw_sample=dict(sample.raw),
        )

    def to_task_outcome(self, sample: TaskSample, trajectory: Trajectory) -> TaskOutcome:
        prediction = trajectory.final_output.get("prediction") or {}
        reference = sample.reference or {}
        services_hit, fault_kind_hit = _grade_against_reference(prediction, reference)

        metrics = dict(trajectory.summary_stats)
        metrics["service_hit"] = float(services_hit)
        metrics["fault_kind_hit"] = float(fault_kind_hit)
        metrics["score"] = 0.7 * services_hit + 0.3 * fault_kind_hit

        success: bool | None
        if reference:
            success = bool(services_hit) and bool(fault_kind_hit)
        else:
            success = None

        return TaskOutcome(
            sample_id=sample.sample_id,
            task_type=sample.task_type,
            prediction=prediction,
            reference=reference,
            success=success,
            termination_reason=str(
                trajectory.final_output.get("termination", trajectory.status.value)
            ),
            metrics=metrics,
            metadata=dict(sample.metadata),
        )


def _pick_id(raw: Mapping[str, Any]) -> str:
    for key in _ID_KEYS:
        value = raw.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return uuid4().hex


def _required_text(raw: Mapping[str, Any], keys: tuple[str, ...], label: str) -> str:
    for key in keys:
        value = raw.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError(f"rca sample missing required key: {label}")


def _extract_reference(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Collect ground-truth fields the reward strategy needs."""
    reference: dict[str, Any] = {}
    services = raw.get("expected_services")
    if isinstance(services, list):
        reference["expected_services"] = [str(s) for s in services if s]
    fault_kind = raw.get("fault_kind")
    if fault_kind is not None and str(fault_kind).strip():
        reference["fault_kind"] = str(fault_kind).strip()
    root_causes = raw.get("root_causes")
    if isinstance(root_causes, list):
        reference["root_causes"] = list(root_causes)
    ground_truth = raw.get("ground_truth")
    if isinstance(ground_truth, dict):
        reference["ground_truth"] = dict(ground_truth)
    return reference


def _grade_against_reference(
    prediction: Any,
    reference: Mapping[str, Any],
) -> tuple[float, float]:
    """Service-hit + fault-kind-hit shape, matching agentm's baseline grader.

    ``prediction`` is the AgentRCAOutput-shaped dict returned by
    :class:`AgentMRuntime`:

        {"root_causes": [{"service": "...", "fault_kind": "..."}, ...],
         "propagation": [...]}

    Reference may carry ``expected_services`` (list of service names)
    and ``fault_kind`` (substring matched case-insensitively).
    """
    services_hit = 0.0
    fault_kind_hit = 0.0

    expected_services = [
        _normalize_kind(s)
        for s in reference.get("expected_services") or []
        if isinstance(s, str)
    ]
    # rcabench injection.json uses snake_case fault_kind (``cpu_stress``,
    # ``pod_kill``); LLM verdicts tend toward natural language ("CPU
    # stress"). Normalize both ends by lowercasing and collapsing ``_``
    # / multi-space to a single space before substring matching.
    expected_fault = _normalize_kind(reference.get("fault_kind") or "")

    if not expected_services and not expected_fault:
        return 0.0, 0.0

    pred_services: list[str] = []
    pred_fault_kinds: list[str] = []
    if isinstance(prediction, Mapping):
        for rc in prediction.get("root_causes") or []:
            if isinstance(rc, Mapping):
                svc = rc.get("service")
                kind = rc.get("fault_kind")
                if isinstance(svc, str):
                    pred_services.append(_normalize_kind(svc))
                if isinstance(kind, str):
                    pred_fault_kinds.append(_normalize_kind(kind))

    raw_blob = " ".join(pred_services + pred_fault_kinds)
    if expected_services and any(
        any(exp in svc or exp == svc for svc in pred_services + [raw_blob])
        for exp in expected_services
    ):
        services_hit = 1.0
    if expected_fault and any(
        expected_fault in (k or "") for k in pred_fault_kinds + [raw_blob]
    ):
        fault_kind_hit = 1.0

    return services_hit, fault_kind_hit


def _normalize_kind(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", " ")
    return " ".join(text.split())
