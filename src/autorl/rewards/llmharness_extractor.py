"""Reward strategy for llmharness extractor child rollouts.

The extractor child agent (see
``contrib/extensions/llmharness/src/llmharness/audit/extractor/``)
drives an incremental graph build via the v19 tool surface:
``upsert_node`` / ``delete_node`` / ``upsert_edge`` / ``delete_edge``
/ ``reset_extraction`` / ``finalize_extraction``. Each edit gets
narrow witness-validation feedback per tool call; the child terminates
by calling ``finalize_extraction`` with no payload.

This reward is deterministic — no LLM judge. It reads the trajectory's
tool_call / tool_result events and grades three dimensions:

* ``witness_pass_rate`` — fraction of tool results that came back
  non-error. Captures how cleanly the model navigates the per-edit
  validation feedback.
* ``finalize_success`` — 1 iff the last tool_call was
  ``finalize_extraction`` and its result was non-error. Without
  finalize, downstream consumers see an unterminated firing.
* ``efficiency_penalty`` — ``min(1, steps_used / max_steps_budget)``;
  ``max_steps_budget`` comes from ``runtime_context.limits.max_steps``
  (default 32).

Composite: ``0.5 * finalize + 0.3 * witness - 0.2 * efficiency``.
``trajectory.steps`` is assumed to be populated by an upstream runtime
adapter; the runtime work itself lives in a separate PR. If
``trajectory.steps`` has no tool events the reward is zero across the
board (defensive — empty trajectories should never crash the rollout
loop).
"""

from __future__ import annotations

from typing import Any

from autorl.contracts import RuntimeContext, TaskOutcome, TaskSample, Trajectory
from autorl.contracts.trajectory import TrajectoryStep, TrajectoryStepType

from .base import RewardStrategy
from .registry import register_reward

_FINALIZE_TOOL_NAME = "finalize_extraction"
_DEFAULT_MAX_STEPS = 32


def _tool_name(step: TrajectoryStep) -> str:
    """Extract the tool name from a TOOL_CALL step's ``input`` payload."""
    raw: Any = step.input.get("name") or step.input.get("tool_name")
    if not raw:
        # Some adapters nest the call under ``function`` (OpenAI tool-call shape).
        fn = step.input.get("function")
        if isinstance(fn, dict):
            raw = fn.get("name")
    return str(raw or "")


def _is_error(step: TrajectoryStep) -> bool:
    """Tool-result success/failure flag.

    Accepts both ``is_error`` (Anthropic-style) and ``error`` (truthy
    object). A result with neither key is treated as success — adapters
    that don't surface failure explicitly shouldn't penalize the model.
    """
    out = step.output
    if "is_error" in out:
        return bool(out["is_error"])
    if "error" in out and out["error"]:
        return True
    return False


def _pair_calls_with_results(
    steps: list[TrajectoryStep],
) -> list[tuple[TrajectoryStep, TrajectoryStep | None]]:
    """Pair each TOOL_CALL with its matching TOOL_RESULT.

    Pair by ``call_id`` when both sides carry it; otherwise fall back
    to position (each call's nearest following result). Results without
    a preceding call are dropped — they don't belong to any pairing.
    """
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
            # Positional fallback: the next result strictly after this call
            # in trajectory order. Use a moving cursor so each result is
            # consumed at most once.
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


class LlmharnessExtractorRewardStrategy(RewardStrategy):
    """Witness / finalize / efficiency composite for extractor rollouts."""

    async def compute(
        self,
        sample: TaskSample,
        trajectory: Trajectory,
        outcome: TaskOutcome,
        runtime_context: RuntimeContext,
    ) -> dict[str, float]:
        steps: list[TrajectoryStep] = list(trajectory.steps)
        pairs = _pair_calls_with_results(steps)

        if not pairs:
            return {
                "reward": 0.0,
                "witness_pass_rate": 0.0,
                "finalize_success": 0.0,
                "efficiency_penalty": 0.0,
            }

        # witness_pass_rate: among results we observed, fraction non-error.
        observed_results = [result for _call, result in pairs if result is not None]
        if observed_results:
            pass_count = sum(1 for r in observed_results if not _is_error(r))
            witness_pass_rate = pass_count / len(observed_results)
        else:
            witness_pass_rate = 0.0

        # finalize_success: last call is ``finalize_extraction`` and its result is non-error.
        last_call, last_result = pairs[-1]
        finalize_success = (
            1.0
            if _tool_name(last_call) == _FINALIZE_TOOL_NAME
            and last_result is not None
            and not _is_error(last_result)
            else 0.0
        )

        # efficiency_penalty: steps_used / budget, clipped to [0, 1].
        budget: int = (
            runtime_context.limits.max_steps
            if runtime_context.limits.max_steps and runtime_context.limits.max_steps > 0
            else _DEFAULT_MAX_STEPS
        )
        steps_used = len(pairs)
        efficiency_penalty = min(1.0, steps_used / float(budget))

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


register_reward("llmharness_extractor", LlmharnessExtractorRewardStrategy)


__all__ = ["LlmharnessExtractorRewardStrategy"]
