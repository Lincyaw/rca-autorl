"""Outcome-level reward for auditor rollouts.

Mirrors ``llmharness_extractor_outcome``: passthrough of the
case_outcome ``composite_score`` for auditor-phase rollouts.
"""

from __future__ import annotations

from autorl.contracts import RuntimeContext, TaskOutcome, TaskSample, Trajectory

from .base import RewardStrategy
from .llmharness_extractor_outcome import _extract_case_outcome
from .registry import register_reward


class LlmharnessAuditorOutcomeRewardStrategy(RewardStrategy):
    """Composite-score passthrough reward for auditor outcome pairs."""

    async def compute(
        self,
        sample: TaskSample,
        trajectory: Trajectory,
        outcome: TaskOutcome,
        runtime_context: RuntimeContext,
    ) -> dict[str, float]:
        case_outcome = _extract_case_outcome(sample)
        if case_outcome is None:
            return {"reward": 0.0, "no_outcome_label": 1.0}
        score = float(case_outcome.get("composite_score", 0.0))
        return {
            "reward": score,
            "service_hit": float(case_outcome.get("service_hit", 0.0)),
            "fault_kind_hit": float(case_outcome.get("fault_kind_hit", 0.0)),
        }


register_reward(
    "llmharness_auditor_outcome", LlmharnessAuditorOutcomeRewardStrategy
)


__all__ = ["LlmharnessAuditorOutcomeRewardStrategy"]
