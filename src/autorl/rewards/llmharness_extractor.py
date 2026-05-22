"""Reward strategy for llmharness extractor child rollouts.

Delegates the witness/finalize/efficiency math to
``llmharness.train_signals.extractor_process_reward`` so the SDK side
and the RL side share a single source of truth. If that symbol is not
yet available (parallel-track race), this module falls back to a
local copy with identical semantics.

Composite: ``0.5 * finalize + 0.3 * witness - 0.2 * efficiency``.
``trajectory.steps`` is assumed to be populated by an upstream runtime
adapter that emits paired TOOL_CALL / TOOL_RESULT events.
"""

from __future__ import annotations

from typing import Any, TypedDict

from autorl.contracts import RuntimeContext, TaskOutcome, TaskSample, Trajectory
from autorl.contracts.trajectory import TrajectoryStep, TrajectoryStepType

from .base import RewardStrategy
from .registry import register_reward

_FINALIZE_TOOL_NAME = "finalize_extraction"
_DEFAULT_MAX_STEPS = 32


class ToolEvent(TypedDict, total=False):
    """Upstream tool-event shape (matches llmharness.train_signals contract).

    A ToolEvent represents one tool_call ↔ tool_result pair extracted
    from a trajectory. ``name`` is the tool name; ``is_error`` is a
    bool flag (True iff the call failed). ``call_id`` is informational.
    """

    name: str
    is_error: bool
    call_id: str


def _tool_name(step: TrajectoryStep) -> str:
    raw: Any = step.input.get("name") or step.input.get("tool_name")
    if not raw:
        fn = step.input.get("function")
        if isinstance(fn, dict):
            raw = fn.get("name")
    return str(raw or "")


def _is_error(step: TrajectoryStep) -> bool:
    out = step.output
    if "is_error" in out:
        return bool(out["is_error"])
    if "error" in out and out["error"]:
        return True
    return False


def _pair_calls_with_results(
    steps: list[TrajectoryStep],
) -> list[tuple[TrajectoryStep, TrajectoryStep | None]]:
    """Pair TOOL_CALL ↔ TOOL_RESULT by ``call_id`` (fallback: position)."""
    calls = [s for s in steps if s.step_type == TrajectoryStepType.TOOL_CALL]
    results = [s for s in steps if s.step_type == TrajectoryStepType.TOOL_RESULT]
    by_call_id: dict[str, TrajectoryStep] = {
        r.call_id: r for r in results if r.call_id
    }

    pairs: list[tuple[TrajectoryStep, TrajectoryStep | None]] = []
    result_cursor = 0
    for call in calls:
        match: TrajectoryStep | None = None
        if call.call_id and call.call_id in by_call_id:
            match = by_call_id[call.call_id]
        else:
            call_idx = steps.index(call)
            while result_cursor < len(results):
                r = results[result_cursor]
                if steps.index(r) > call_idx:
                    match = r
                    result_cursor += 1
                    break
                result_cursor += 1
        pairs.append((call, match))
    return pairs


def _trajectory_to_tool_events(trajectory: Trajectory) -> list[ToolEvent]:
    """Project a Trajectory's tool_call/tool_result steps into ToolEvent list."""
    pairs = _pair_calls_with_results(list(trajectory.steps))
    events: list[ToolEvent] = []
    for call, result in pairs:
        ev: ToolEvent = {
            "name": _tool_name(call),
            "is_error": _is_error(result) if result is not None else True,
        }
        if call.call_id:
            ev["call_id"] = call.call_id
        events.append(ev)
    return events


def _local_extractor_process_reward(
    tool_events: list[ToolEvent],
    max_steps_budget: int,
) -> dict[str, float]:
    """Fallback implementation if llmharness.train_signals is unavailable.

    TODO(llmharness): swap to upstream once
    ``llmharness.train_signals.extractor_process_reward`` is exposed.
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
        1.0 if last.get("name") == _FINALIZE_TOOL_NAME and not last.get("is_error", False) else 0.0
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


def _call_extractor_process_reward(
    tool_events: list[ToolEvent], max_steps_budget: int
) -> dict[str, float]:
    """Delegate to llmharness.train_signals.extractor_process_reward if available."""
    try:
        from llmharness.train_signals import (  # type: ignore[import-not-found]
            extractor_process_reward,
        )
    except ImportError:
        # TODO(llmharness): swap to upstream once exposed.
        return _local_extractor_process_reward(tool_events, max_steps_budget)
    result = extractor_process_reward(tool_events, max_steps_budget)
    return {str(k): float(v) for k, v in result.items()}


class LlmharnessExtractorRewardStrategy(RewardStrategy):
    """Witness / finalize / efficiency composite for extractor rollouts."""

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
        return _call_extractor_process_reward(tool_events, budget)


# Canonical name + backward-compat alias.
register_reward("llmharness_extractor_process", LlmharnessExtractorRewardStrategy)
register_reward("llmharness_extractor", LlmharnessExtractorRewardStrategy)


__all__ = ["LlmharnessExtractorRewardStrategy", "ToolEvent"]
