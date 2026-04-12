from __future__ import annotations

import math
from typing import Any

from autorl.contracts import TaskOutcome, Trajectory, TrajectoryStatus, TrajectoryStepType


def build_rollout_metrics(
    *,
    trajectory: Trajectory,
    outcome: TaskOutcome,
    reward: float | dict[str, float],
) -> dict[str, float]:
    """Extract scalar episode metrics for AReaL rollout logging."""

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    llm_requests = 0
    llm_responses = 0
    tool_calls = 0
    tool_results = 0
    env_actions = 0
    env_observations = 0

    for step in trajectory.steps:
        if step.step_type == TrajectoryStepType.LLM_REQUEST:
            llm_requests += 1
        elif step.step_type == TrajectoryStepType.LLM_RESPONSE:
            llm_responses += 1
        elif step.step_type == TrajectoryStepType.TOOL_CALL:
            tool_calls += 1
        elif step.step_type == TrajectoryStepType.TOOL_RESULT:
            tool_results += 1
        elif step.step_type == TrajectoryStepType.ENV_ACTION:
            env_actions += 1
        elif step.step_type == TrajectoryStepType.ENV_OBSERVATION:
            env_observations += 1

        if step.usage is not None:
            prompt_tokens += int(step.usage.prompt_tokens)
            completion_tokens += int(step.usage.completion_tokens)
            total_tokens += int(step.usage.total_tokens)

    metrics: dict[str, float] = {
        "episode_reward": _reward_total(reward),
        "success": _bool_to_float(outcome.success),
        "trajectory_steps": float(len(trajectory.steps)),
        "turns": _as_float(trajectory.summary_stats.get("turns"), fallback=float(llm_responses)),
        "num_search": _as_float(trajectory.summary_stats.get("num_search"), fallback=0.0),
        "num_access": _as_float(trajectory.summary_stats.get("num_access"), fallback=0.0),
        "llm_requests": float(llm_requests),
        "llm_responses": float(llm_responses),
        "tool_calls": float(tool_calls),
        "tool_results": float(tool_results),
        "env_actions": float(env_actions),
        "env_observations": float(env_observations),
        "prompt_tokens": float(prompt_tokens),
        "completion_tokens": float(completion_tokens),
        "total_tokens": float(total_tokens),
        "status_completed": float(trajectory.status == TrajectoryStatus.COMPLETED),
        "status_failed": float(trajectory.status == TrajectoryStatus.FAILED),
        "status_truncated": float(trajectory.status == TrajectoryStatus.TRUNCATED),
        "status_rejected": float(trajectory.status == TrajectoryStatus.REJECTED),
        "term_answer": float(outcome.termination_reason == "answer"),
        "term_llm_call_limit": float(outcome.termination_reason == "llm_call_limit"),
        "term_token_limit": float(outcome.termination_reason == "token_limit"),
        "term_answer_not_found": float(outcome.termination_reason == "answer_not_found"),
    }

    if isinstance(reward, dict):
        for key, value in reward.items():
            slug = _metric_slug(key)
            scalar = _as_float(value, fallback=None)
            if scalar is not None:
                metrics[f"reward_component_{slug}"] = scalar

    return {key: value for key, value in metrics.items() if value is not None and math.isfinite(value)}


def log_rollout_metrics(metrics: dict[str, float]) -> None:
    """Best-effort logging into AReaL's rollout stats pipeline."""

    try:
        from areal import workflow_context
        from areal.utils import stats_tracker

        stats_tracker.get(workflow_context.stat_scope()).scalar(**metrics)
    except Exception:
        # Keep repo-level runtime usable even when AReaL metrics plumbing is unavailable.
        return


def build_rollout_metric_record(
    *,
    sample_id: str,
    task_type: str,
    trajectory: Trajectory,
    outcome: TaskOutcome,
    reward: float | dict[str, float],
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "task_type": task_type,
        "trajectory_id": trajectory.trajectory_id,
        "trajectory_status": trajectory.status.value,
        "termination_reason": outcome.termination_reason,
        "metrics": metrics,
        "reward": reward,
    }


def _reward_total(reward: float | dict[str, float]) -> float:
    if isinstance(reward, dict):
        total = 0.0
        for value in reward.values():
            scalar = _as_float(value, fallback=0.0)
            total += scalar or 0.0
        return total
    return _as_float(reward, fallback=0.0) or 0.0


def _metric_slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name.strip().lower()).strip("_") or "value"


def _bool_to_float(value: bool | None) -> float:
    if value is None:
        return 0.0
    return 1.0 if value else 0.0


def _as_float(value: Any, fallback: float | None) -> float | None:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return fallback
