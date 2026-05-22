"""Real runtime for a single llmharness auditor child firing.

Auditor counterpart of :class:`LlmharnessExtractorRuntime`. Wraps
:func:`llmharness.replay_auditor_record`. The auditor's output shape is
a verdict dict rather than the extractor's events/edges/dropped_edges
trio, so the summary stats differ accordingly.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from uuid import uuid4

from autorl.contracts import (
    AgentInput,
    AgentRunResult,
    RuntimeContext,
    TaskOutcome,
    Trajectory,
    TrajectoryStatus,
    TrajectoryStep,
    TrajectoryStepType,
)

from .base import AgentRuntime


class LlmharnessAuditorRuntime(AgentRuntime):
    """Drive one auditor child session via ``replay_auditor_record``."""

    def __init__(self, max_steps: int = 32, timeout: float = 600.0) -> None:
        self.max_steps = max_steps
        self.timeout = timeout

    async def run(
        self,
        agent_input: AgentInput,
        runtime_context: RuntimeContext,
    ) -> AgentRunResult:
        # Lazy import — keep this module importable without llmharness.
        from llmharness import (  # type: ignore[import-not-found]
            ReplayRecord,
            replay_auditor_record,
        )

        try:
            from llmharness import (  # type: ignore[import-not-found,attr-defined]
                tool_events_from_phase_result,
            )
        except ImportError:  # pragma: no cover - transitional fallback
            from llmharness.train_signals import (  # type: ignore[import-not-found]
                tool_events_from_phase_result,
            )

        record = ReplayRecord.from_dict(agent_input.raw_sample)
        provider_override = _build_provider_override(runtime_context)

        with _isolated_cwd() as cwd:
            started_ms = _now_ms()
            phase_result = await replay_auditor_record(
                record,
                cwd=cwd,
                provider_override=provider_override,
            )
            finished_ms = _now_ms()

        tool_events = list(tool_events_from_phase_result(phase_result))
        steps = _trajectory_steps_from_tool_events(
            tool_events, base_ts_ms=started_ms
        )

        out: dict[str, Any] = phase_result.output or {}
        verdict = out.get("verdict") if isinstance(out, dict) else None
        has_verdict = isinstance(verdict, dict) and bool(verdict)
        summary: dict[str, float] = {
            "has_verdict": 1.0 if has_verdict else 0.0,
            "latency_ms": float(getattr(phase_result, "latency_ms", 0) or 0),
            "tool_calls": float(len(tool_events)),
            "wall_ms": float(finished_ms - started_ms),
        }
        # Surface a verdict count if the auditor returns findings.
        findings = out.get("findings") if isinstance(out, dict) else None
        if isinstance(findings, list):
            summary["findings_count"] = float(len(findings))

        status_str = str(getattr(phase_result, "status", "unknown"))
        traj_status = (
            TrajectoryStatus.COMPLETED
            if status_str == "ok"
            else TrajectoryStatus.FAILED
        )

        trajectory = Trajectory(
            trajectory_id=uuid4().hex,
            sample_id=agent_input.sample_id,
            task_type=agent_input.task_type,
            status=traj_status,
            final_output={
                "phase_output": out,
                "phase_status": status_str,
                "phase_error": getattr(phase_result, "error", None),
                "termination": status_str,
            },
            steps=steps,
            summary_stats=summary,
            metadata={
                "runtime": "llmharness_auditor",
                **agent_input.metadata,
            },
        )
        outcome = TaskOutcome(
            sample_id=agent_input.sample_id,
            task_type=agent_input.task_type,
            prediction=out,
            reference=None,
            success=(status_str == "ok"),
            termination_reason=status_str,
            metrics=dict(summary),
            metadata={
                "runtime": "llmharness_auditor",
                **agent_input.metadata,
            },
        )
        return AgentRunResult(trajectory=trajectory, outcome=outcome)


# ---------------------------------------------------------------- helpers


def _build_provider_override(
    runtime_context: RuntimeContext,
) -> tuple[str, dict[str, Any]] | None:
    endpoint = runtime_context.model_endpoint
    api_key = runtime_context.api_key
    if not endpoint or not api_key:
        return None
    meta = runtime_context.metadata or {}
    config: dict[str, Any] = {
        "base_url": str(endpoint),
        "api_key": str(api_key),
        "verify_ssl": False,
        "model": str(meta.get("agentm_model") or "gpt-4o-mini"),
    }
    return ("agentm.extensions.builtin.llm_openai", config)


@contextmanager
def _isolated_cwd() -> Iterator[str]:
    with tempfile.TemporaryDirectory(prefix="llmharness-replay-") as tmp:
        try:
            subprocess.run(
                ["git", "init", "-q"],
                cwd=tmp,
                check=False,
                capture_output=True,
            )
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.email=replay@local",
                    "-c",
                    "user.name=replay",
                    "commit",
                    "--allow-empty",
                    "-m",
                    "init",
                    "-q",
                ],
                cwd=tmp,
                check=False,
                capture_output=True,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            )
        except FileNotFoundError:
            pass
        yield tmp


def _trajectory_steps_from_tool_events(
    tool_events: list[dict[str, Any]],
    *,
    base_ts_ms: int,
) -> list[TrajectoryStep]:
    steps: list[TrajectoryStep] = []
    for i, ev in enumerate(tool_events):
        tool_name = str(ev.get("tool_name") or "")
        args_in = ev.get("args")
        args: dict[str, Any] = (
            dict(args_in) if isinstance(args_in, dict) else {}
        )
        is_error = bool(ev.get("is_error", False))
        error_text = ev.get("error_text")
        call_id = f"tool-{i}"
        call_ts = base_ts_ms + i * 2
        result_ts = call_ts + 1
        steps.append(
            TrajectoryStep(
                step_id=f"{call_id}-call",
                step_type=TrajectoryStepType.TOOL_CALL,
                timestamp_ms=call_ts,
                input={"tool_name": tool_name, "args": args},
                call_id=call_id,
                metadata={"index": i},
            )
        )
        steps.append(
            TrajectoryStep(
                step_id=f"{call_id}-result",
                step_type=TrajectoryStepType.TOOL_RESULT,
                timestamp_ms=result_ts,
                output={
                    "tool_name": tool_name,
                    "is_error": is_error,
                    "error_text": error_text,
                },
                call_id=call_id,
                metadata={"index": i},
            )
        )
    return steps


def _now_ms() -> int:
    return int(time.time() * 1000)


__all__ = ["LlmharnessAuditorRuntime"]
