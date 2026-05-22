"""TaskAdapters for llmharness firing-context samples.

A firing-context sample is one row of ``rl_prompts.jsonl`` (see
``autorl.data.rl_prompts``): an extractor or auditor child agent
prompt re-rendered from a recorded case, ready to be fed back to the
model under RL.

Both adapters share validation / projection logic; they differ only
in the canonical ``task_type`` they advertise.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from autorl.contracts import AgentInput, TaskOutcome, TaskSample, Trajectory

from .base import TaskAdapter


def _validate_firing_row(raw: Mapping[str, Any]) -> None:
    for key in ("sample_id", "input"):
        if key not in raw:
            raise ValueError(f"firing row missing key '{key}'")
    inp = raw["input"]
    if not isinstance(inp, Mapping) or "system" not in inp or "user" not in inp:
        raise ValueError("firing row 'input' must carry 'system' and 'user'")


def _firing_metadata(raw: Mapping[str, Any]) -> dict[str, Any]:
    meta_in = raw.get("meta")
    meta = dict(meta_in) if isinstance(meta_in, Mapping) else {}
    for key in ("source_case_id", "firing_index", "phase"):
        if key in raw and key not in meta:
            meta[key] = raw[key]
    return meta


class _LlmharnessFiringTaskAdapter(TaskAdapter):
    """Shared base for extractor / auditor firing adapters."""

    def validate_sample(self, raw_sample: Mapping[str, Any]) -> TaskSample:
        _validate_firing_row(raw_sample)
        sample_id = str(raw_sample["sample_id"])
        inp = raw_sample["input"]
        return TaskSample(
            sample_id=sample_id,
            task_type=self.task_type,
            input={
                "system": str(inp.get("system", "")),
                "user": str(inp.get("user", "")),
            },
            target=None,
            reference=None,
            metadata=_firing_metadata(raw_sample),
            raw=dict(raw_sample),
        )

    def to_agent_input(self, sample: TaskSample) -> AgentInput:
        system = str(sample.input.get("system", ""))
        user = str(sample.input.get("user", ""))
        return AgentInput(
            sample_id=sample.sample_id,
            task_type=sample.task_type,
            instruction=user,
            context={"phase": sample.metadata.get("phase")},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
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


class LlmharnessAuditorFiringTaskAdapter(_LlmharnessFiringTaskAdapter):
    """Firing-context adapter for auditor child rollouts."""

    task_type = "llmharness_auditor_firing"


__all__ = [
    "LlmharnessAuditorFiringTaskAdapter",
    "LlmharnessExtractorFiringTaskAdapter",
]
