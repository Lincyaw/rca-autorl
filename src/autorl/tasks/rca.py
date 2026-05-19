"""TaskAdapter for RCA cases that target the AgentM ``rca:harness.sync``
scenario.

The adapter accepts the canonical sample shape produced by AgentM's
``agentm_rca.eval.seed_dataset`` pipeline — i.e. the rows in
``$AGENTM_RCA_DATASET_ROOT/data.jsonl``::

    {"id":            5,
     "source":        "ts0-mysql-corrupt-kwx8n5",
     "question":      "<natural-language SLO/incident description>",
     "answer":        "mysql,ts-station-service",
     "ground_truth":  ["mysql", "ts-station-service"],
     "fault_type":    "NetworkCorrupt",
     "datapack_name": "ts0-mysql-corrupt-kwx8n5",
     ...}

``data_dir`` is recovered as ``$AGENTM_RCA_DATASET_ROOT / datapack_name``
so SFT and RL trajectories land on the same on-disk case directory
that ``AgentMAgent`` consumes via ``AGENTM_RCA_DATA_DIR``.

The pre-AgentM shape (``incident`` / ``data_dir`` / ``expected_services``
/ ``fault_kind``) is still accepted as a fallback for one-off RL
manifests that don't go through ``seed_dataset``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from autorl.contracts import AgentInput, TaskOutcome, TaskSample, Trajectory

from .base import TaskAdapter

_ID_KEYS = ("sample_id", "case_id", "qid", "query_id", "source", "id")
_INCIDENT_KEYS = ("incident", "question", "prompt", "task_description", "task")
_DATA_DIR_KEYS = ("data_dir", "case_dir", "observability_dir")
_DATAPACK_KEYS = ("datapack_name", "data_pack_name", "case_name")
_RESERVED = (
    set(_ID_KEYS)
    | set(_INCIDENT_KEYS)
    | set(_DATA_DIR_KEYS)
    | set(_DATAPACK_KEYS)
    | {
        "expected_services",
        "fault_kind",
        "fault_type",
        "root_causes",
        "ground_truth",
        "answer",
        "messages",
        "tags",
    }
)

_DATASET_ROOT_ENV = "AGENTM_RCA_DATASET_ROOT"


class RCATaskAdapter(TaskAdapter):
    """Bind AgentM-shaped RCA rows to the autorl runtime contracts."""

    task_type = "rca"

    def validate_sample(self, raw_sample: Mapping[str, Any]) -> TaskSample:
        incident = _required_text(raw_sample, _INCIDENT_KEYS, "incident or question")
        data_dir = _resolve_data_dir(raw_sample)
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


def _resolve_data_dir(raw: Mapping[str, Any]) -> str:
    """Find ``data_dir`` directly or rebuild it from ``datapack_name`` + env root."""
    for key in _DATA_DIR_KEYS:
        value = raw.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    for key in _DATAPACK_KEYS:
        value = raw.get(key)
        if value is None or not str(value).strip():
            continue
        root = os.environ.get(_DATASET_ROOT_ENV, "").strip()
        if not root:
            raise ValueError(
                f"rca sample carries {key!r} but {_DATASET_ROOT_ENV} is not set — "
                f"point it at the directory that holds the case folders "
                f"(e.g. /home/ddq/AoyangSpace/dataset/rca)"
            )
        return str(Path(root).expanduser() / str(value).strip())
    raise ValueError(
        "rca sample missing required key: data_dir or datapack_name (+ "
        f"{_DATASET_ROOT_ENV})"
    )


def _extract_reference(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Collect ground-truth fields the reward strategy needs.

    Accepts both shapes:
      * AgentM seed_dataset row: ``ground_truth=[...services...]``,
        ``fault_type="NetworkCorrupt"``;
      * pre-AgentM hand-written manifest: ``expected_services=[...]``,
        ``fault_kind="cpu_stress"``.
    """
    reference: dict[str, Any] = {}

    services: list[str] = []
    explicit = raw.get("expected_services")
    if isinstance(explicit, list):
        services = [str(s) for s in explicit if isinstance(s, str) and s]
    elif isinstance(raw.get("ground_truth"), list):
        services = [
            str(s) for s in raw["ground_truth"] if isinstance(s, str) and s
        ]
    if services:
        reference["expected_services"] = services

    fault = raw.get("fault_kind")
    if fault is None or not str(fault).strip():
        fault = raw.get("fault_type")
    if fault is not None and str(fault).strip():
        reference["fault_kind"] = str(fault).strip()

    if isinstance(raw.get("root_causes"), list):
        reference["root_causes"] = list(raw["root_causes"])
    if isinstance(raw.get("ground_truth"), dict):
        reference["ground_truth"] = dict(raw["ground_truth"])
    elif isinstance(raw.get("ground_truth"), list):
        reference.setdefault("ground_truth", list(raw["ground_truth"]))
    return reference


def _grade_against_reference(
    prediction: Any,
    reference: Mapping[str, Any],
) -> tuple[float, float]:
    """Service-hit + fault-kind-hit shape, matching agentm's baseline grader.

    ``prediction`` is the AgentRCAOutput-shaped dict returned by
    :class:`AgentMRuntime`::

        {"root_causes": [{"service": "...", "fault_kind": "..."}, ...],
         "propagation": [...]}

    Reference carries ``expected_services`` (list of service names) and
    ``fault_kind`` (substring matched case-insensitively). The
    ``_normalize_kind`` helper folds the rcabench snake_case shape
    (``cpu_stress``, ``NetworkDelay``) into the same space-separated
    lowercase form LLM verdicts use ("CPU stress", "network delay")
    so substring matching survives the train/serve mismatch.
    """
    services_hit = 0.0
    fault_kind_hit = 0.0

    expected_services = [
        _normalize_kind(s)
        for s in reference.get("expected_services") or []
        if isinstance(s, str)
    ]
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
    """Fold ``cpu_stress``, ``NetworkDelay``, ``CPU stress`` to one form.

    rcabench's ``injection.json`` uses snake_case OR CamelCase fault
    labels; LLM verdicts tend toward natural English. Insert spaces at
    camelCase boundaries, replace underscores with spaces, lowercase,
    collapse whitespace. All call sites can then do substring match
    safely.
    """
    raw = str(value or "")
    spaced = _CAMEL_RE.sub(" ", raw)
    return " ".join(spaced.lower().replace("_", " ").split())


_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")
