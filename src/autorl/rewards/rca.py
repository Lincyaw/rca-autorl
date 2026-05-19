"""Reward strategy for RCA rollouts.

Pulls precomputed ``service_hit`` / ``fault_kind_hit`` / ``score`` off
``TaskOutcome.metrics`` (populated by :class:`RCATaskAdapter`) and
returns the composite score as the scalar reward. The same dimensions
are surfaced in the dict return so AReaL's ``stats_tracker`` can log
them per-episode.

The grading shape (``0.7 * service_hit + 0.3 * fault_kind_hit``)
mirrors AgentM's own ``contrib/scenarios/rca/eval/baseline/grader.py``
so SFT-time supervision (from llmharness/distill labels) and RL-time
reward stay on one rubric.
"""

from __future__ import annotations

from autorl.contracts import RuntimeContext, TaskOutcome, TaskSample, Trajectory

from .base import RewardStrategy


class RCABaselineRewardStrategy(RewardStrategy):
    """Service-hit + fault-kind-hit composite reward."""

    async def compute(
        self,
        sample: TaskSample,
        trajectory: Trajectory,
        outcome: TaskOutcome,
        runtime_context: RuntimeContext,
    ) -> dict[str, float]:
        metrics = outcome.metrics or {}
        service_hit = float(metrics.get("service_hit", 0.0))
        fault_kind_hit = float(metrics.get("fault_kind_hit", 0.0))
        score = float(metrics.get("score", 0.7 * service_hit + 0.3 * fault_kind_hit))
        has_submission = float(metrics.get("has_submission", 0.0))

        return {
            "reward": score,
            "service_hit": service_hit,
            "fault_kind_hit": fault_kind_hit,
            "has_submission": has_submission,
        }
