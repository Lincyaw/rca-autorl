"""TaskAdapters for llmharness firing-context samples.

A firing-context sample is one row of ``rl_prompts.jsonl`` (see
``autorl.data.rl_prompts``). The new on-disk shape is a
**stripped-ReplayRecord** dict: it carries every field
``ReplayRecord.to_dict()`` produces except the teacher-output fields
(``output``, ``status``, ``error``, ``latency_ms``,
``raw_assistant_messages``). The runtime rehydrates it via
``ReplayRecord.from_dict(row)`` and replays it through llmharness's
``replay_extractor_record`` / ``replay_auditor_record`` helpers.

The adapters here treat the row as opaque payload — we only inspect
``phase``, ``sample_id``, ``root_session_id``, and ``turn_index`` so we
can produce a stable ``TaskSample`` shell. The full row goes through
the pipeline on ``raw_sample``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from autorl.contracts import AgentInput, TaskOutcome, TaskSample, Trajectory

from .base import TaskAdapter


def _firing_metadata(raw: Mapping[str, Any]) -> dict[str, Any]:
    meta_in = raw.get("meta")
    meta = dict(meta_in) if isinstance(meta_in, Mapping) else {}
    # Surface the ReplayRecord identifying fields onto metadata so
    # downstream code (reward strategies, dashboards) can group by case
    # without re-parsing the raw row.
    for key in (
        "phase",
        "source_case_id",
        "firing_index",
        "root_session_id",
        "turn_index",
    ):
        if key in raw and key not in meta:
            meta[key] = raw[key]
    return meta


def _derive_sample_id(raw: Mapping[str, Any]) -> str:
    explicit = raw.get("sample_id")
    if explicit:
        return str(explicit)
    rsid = raw.get("root_session_id")
    turn = raw.get("turn_index")
    if rsid is not None and turn is not None:
        return f"{rsid}:turn-{turn}"
    raise ValueError(
        "firing row needs 'sample_id' or both 'root_session_id' and 'turn_index'"
    )


class _LlmharnessFiringTaskAdapter(TaskAdapter):
    """Shared base for extractor / auditor firing adapters.

    Subclasses set ``task_type`` and ``expected_phase`` (``"extractor"``
    or ``"auditor"``).
    """

    expected_phase: str = ""

    def validate_sample(self, raw_sample: Mapping[str, Any]) -> TaskSample:
        phase = raw_sample.get("phase")
        if not phase:
            raise ValueError("firing row missing key 'phase'")
        if self.expected_phase and phase != self.expected_phase:
            raise ValueError(
                f"adapter expects phase={self.expected_phase!r}, got {phase!r}"
            )
        # ReplayRecord-shaped rows always carry payload + compose_kwargs.
        for key in ("payload", "compose_kwargs"):
            if key not in raw_sample:
                raise ValueError(f"firing row missing key '{key}'")
        sample_id = _derive_sample_id(raw_sample)
        return TaskSample(
            sample_id=sample_id,
            task_type=self.task_type,
            input=dict(raw_sample),
            target=None,
            reference=None,
            metadata=_firing_metadata(raw_sample),
            raw=dict(raw_sample),
        )

    def to_agent_input(self, sample: TaskSample) -> AgentInput:
        # The runtime rehydrates the full ReplayRecord from raw_sample;
        # we deliberately don't try to extract system/user from the
        # payload here — the prompts live inside ``compose_kwargs`` and
        # the live composer reconstructs them at replay time.
        return AgentInput(
            sample_id=sample.sample_id,
            task_type=sample.task_type,
            instruction=None,
            context={"phase": sample.metadata.get("phase")},
            messages=[],
            available_tools=[],
            constraints={},
            metadata=dict(sample.metadata),
            raw_sample=dict(sample.raw),
        )

    def to_task_outcome(
        self, sample: TaskSample, trajectory: Trajectory
    ) -> TaskOutcome:
        # Reward strategies compute the actual reward; this adapter
        # only surfaces what was recorded in the trajectory.
        return TaskOutcome(
            sample_id=sample.sample_id,
            task_type=sample.task_type,
            prediction=trajectory.final_output,
            reference=sample.reference,
            success=None,
            termination_reason=str(
                trajectory.final_output.get("termination", "")
            )
            or None,
            metrics=dict(trajectory.summary_stats),
            metadata=dict(sample.metadata),
        )


class LlmharnessExtractorFiringTaskAdapter(_LlmharnessFiringTaskAdapter):
    """Firing-context adapter for extractor child rollouts."""

    task_type = "llmharness_extractor_firing"
    expected_phase = "extractor"


class LlmharnessAuditorFiringTaskAdapter(_LlmharnessFiringTaskAdapter):
    """Firing-context adapter for auditor child rollouts."""

    task_type = "llmharness_auditor_firing"
    expected_phase = "auditor"


__all__ = [
    "LlmharnessAuditorFiringTaskAdapter",
    "LlmharnessExtractorFiringTaskAdapter",
]
