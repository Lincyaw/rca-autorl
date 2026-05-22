"""Reward strategy for llmharness auditor child rollouts.

Mirrors ``llmharness_extractor`` for the auditor phase: delegates the
process-reward math to ``llmharness.train_signals.auditor_process_reward``
when available, otherwise uses a local fallback with identical semantics
(witness/finalize/efficiency, where ``finalize`` for the auditor is
``submit_verdict``).
"""

from __future__ import annotations

from autorl.contracts import RuntimeContext, TaskOutcome, TaskSample, Trajectory

from .base import RewardStrategy
from .llmharness_extractor import ToolEvent, _trajectory_to_tool_events
from .registry import register_reward

_FINALIZE_TOOL_NAME = "submit_verdict"
_DEFAULT_MAX_STEPS = 32


def _local_auditor_process_reward(
    tool_events: list[ToolEvent],
    max_steps_budget: int,
) -> dict[str, float]:
    """Fallback implementation if llmharness.train_signals is unavailable.

    TODO(llmharness): swap to upstream once
    ``llmharness.train_signals.auditor_process_reward`` is exposed.
    """
    if not tool_events:
        return {
            "reward": 0.0,
            "witness_pass_rate": 0.0,
            "finalize_success": 0.0,
            "efficiency_penalty": 0.0,
        }

    pass_count = sum(1 for ev in tool_events if not ev.get("is_error", False))
    witness_pass_rate = pass_count / len(tool_events)

    last = tool_events[-1]
    finalize_success = (
        1.0
        if last.get("name") == _FINALIZE_TOOL_NAME and not last.get("is_error", False)
        else 0.0
    )

    budget = max_steps_budget if max_steps_budget > 0 else _DEFAULT_MAX_STEPS
    efficiency_penalty = min(1.0, len(tool_events) / float(budget))

    composite = (
        0.5 * finalize_success
        + 0.3 * witness_pass_rate
        - 0.2 * efficiency_penalty
    )
    return {
        "reward": composite,
        "witness_pass_rate": witness_pass_rate,
        "finalize_success": finalize_success,
        "efficiency_penalty": efficiency_penalty,
    }


def _call_auditor_process_reward(
    tool_events: list[ToolEvent], max_steps_budget: int
) -> dict[str, float]:
    try:
        from llmharness.train_signals import (  # type: ignore[import-not-found]
            auditor_process_reward,
        )
    except ImportError:
        return _local_auditor_process_reward(tool_events, max_steps_budget)
    result = auditor_process_reward(tool_events, max_steps_budget)
    return {str(k): float(v) for k, v in result.items()}


class LlmharnessAuditorProcessRewardStrategy(RewardStrategy):
    """Witness / submit_verdict / efficiency composite for auditor rollouts."""

    async def compute(
        self,
        sample: TaskSample,
        trajectory: Trajectory,
        outcome: TaskOutcome,
        runtime_context: RuntimeContext,
    ) -> dict[str, float]:
        tool_events = _trajectory_to_tool_events(trajectory)
        budget = (
            runtime_context.limits.max_steps
            if runtime_context.limits.max_steps and runtime_context.limits.max_steps > 0
            else _DEFAULT_MAX_STEPS
        )
        return _call_auditor_process_reward(tool_events, budget)


register_reward(
    "llmharness_auditor_process", LlmharnessAuditorProcessRewardStrategy
)


__all__ = ["LlmharnessAuditorProcessRewardStrategy"]
