"""Outcome-level reward for extractor rollouts.

Reads the case_outcome annotation produced by
``llmharness-distill annotate-case-outcome`` and returns its
``composite_score`` as the reward. Pure dict access — no LLM call.

If the annotation isn't attached to the sample (``sample.reference``
or ``sample.metadata`` lacks ``case_outcome``), returns
``{"reward": 0.0, "no_outcome_label": 1.0}`` so the trainer can drop
the rollout cleanly.
"""

from __future__ import annotations

from typing import Any

from autorl.contracts import RuntimeContext, TaskOutcome, TaskSample, Trajectory

from .base import RewardStrategy
from .registry import register_reward


def _extract_case_outcome(sample: TaskSample) -> dict[str, Any] | None:
    for source in (sample.reference, sample.metadata):
        if isinstance(source, dict):
            co = source.get("case_outcome")
            if isinstance(co, dict):
                return co
    return None


class LlmharnessExtractorOutcomeRewardStrategy(RewardStrategy):
    """Composite-score passthrough reward for extractor outcome pairs."""

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
    "llmharness_extractor_outcome", LlmharnessExtractorOutcomeRewardStrategy
)


__all__ = ["LlmharnessExtractorOutcomeRewardStrategy"]
