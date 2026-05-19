"""AgentRuntime adapter for the new AgentM ``rca:harness.sync`` scenario.

Wraps :class:`agentm_rca.eval.agent.AgentMAgent` (the rcabench-platform
BaseAgent) so AReaL's rollout workflow can drive an in-process AgentM
RCA investigation. The session emits the standard
``submit_final_report`` shape and writes a ``.agentm/audit_replay``
sidecar pair when ``scenario == "rca:harness.sync"``; the sidecar is
the same artifact ``llmharness-distill`` consumes, so SFT and RL
trajectories share one ground-truth shape.

Boundary: this module is a host-side driver. It uses the AgentM public
``rcabench-platform`` entry point (``AgentMAgent``) — no
``agentm.core.runtime.*`` imports — so it stays compatible with the
SDK/scenario separation rule in AgentM's ``CLAUDE.md``.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from typing import Any, Iterator
from uuid import uuid4

from autorl.contracts import (
    AgentInput,
    AgentRunResult,
    RuntimeContext,
    TaskOutcome,
    TokenUsage,
    Trajectory,
    TrajectoryStatus,
    TrajectoryStep,
    TrajectoryStepType,
)

from .base import AgentRuntime


class AgentMRuntime(AgentRuntime):
    """Run one RCA episode through :class:`AgentMAgent`."""

    def __init__(
        self,
        scenario: str = "rca:harness.sync",
        model: str | None = None,
        provider: str | None = None,
        max_steps: int = 128,
        timeout: float = 600.0,
        exp_id: str | None = None,
    ) -> None:
        self.scenario = scenario
        self.model = model
        self.provider = provider
        self.max_steps = max_steps
        self.timeout = timeout
        self.exp_id = exp_id

    async def run(
        self,
        agent_input: AgentInput,
        runtime_context: RuntimeContext,
    ) -> AgentRunResult:
        # Import inside ``run`` so that constructing the runtime in
        # config-loading paths doesn't drag the agentm dependency in.
        from agentm_rca.eval.agent import AgentMAgent

        incident = _extract_required_text(
            agent_input, "incident", ("incident", "question", "prompt")
        )
        data_dir = _extract_required_text(
            agent_input, "data_dir", ("data_dir", "case_dir", "observability_dir")
        )

        started_at = _now_ms()
        max_turns = min(
            self.max_steps,
            runtime_context.limits.max_steps or self.max_steps,
        )
        agent = AgentMAgent(
            scenario=self.scenario,
            model=self.model,
            provider=self.provider,
            exp_id=self.exp_id or _coerce_optional_text(agent_input.metadata.get("exp_id")),
            max_turns=max_turns,
        )

        with _agentm_env_overrides(runtime_context):
            result = await agent.run(incident=incident, data_dir=data_dir)

        prediction = _loads_or_text(result.response)
        finished_at = _now_ms()

        metadata: dict[str, Any] = {
            "incident": incident,
            "data_dir": data_dir,
            "scenario": self.scenario,
            "model": self.model,
            "provider": self.provider,
            "agentm_trace_id": result.trace_id,
            "agentm_metadata": dict(result.metadata or {}),
            "used_areal_proxy": bool(runtime_context.model_endpoint),
            **agent_input.metadata,
        }

        steps = [
            TrajectoryStep(
                step_id="agentm-dispatch",
                step_type=TrajectoryStepType.CONTROL,
                timestamp_ms=started_at,
                input={
                    "incident": incident,
                    "data_dir": data_dir,
                    "scenario": self.scenario,
                },
                metadata={"runtime": "agentm"},
            ),
            TrajectoryStep(
                step_id="agentm-final",
                step_type=TrajectoryStepType.FINAL,
                timestamp_ms=finished_at,
                output={"prediction": prediction, "trace_id": result.trace_id},
                usage=TokenUsage(),
                metadata={"runtime": "agentm"},
            ),
        ]

        submission_seen = bool((result.metadata or {}).get("submit_final_report_seen"))
        termination = "agentm_complete" if submission_seen else "agentm_no_submission"

        trajectory = Trajectory(
            trajectory_id=uuid4().hex,
            sample_id=agent_input.sample_id,
            task_type=agent_input.task_type,
            status=TrajectoryStatus.COMPLETED if submission_seen else TrajectoryStatus.TRUNCATED,
            final_output={
                "prediction": prediction,
                "trace_id": result.trace_id,
                "termination": termination,
            },
            steps=steps,
            summary_stats={
                "has_submission": 1.0 if submission_seen else 0.0,
                "response_bytes": float(len(result.response or "")),
            },
            artifacts={},
            metadata=metadata,
        )
        outcome = TaskOutcome(
            sample_id=agent_input.sample_id,
            task_type=agent_input.task_type,
            prediction=prediction,
            reference=None,
            success=None,
            termination_reason=termination,
            metrics=dict(trajectory.summary_stats),
            metadata=metadata,
        )
        return AgentRunResult(trajectory=trajectory, outcome=outcome)


@contextmanager
def _agentm_env_overrides(runtime_context: RuntimeContext) -> Iterator[None]:
    """Route AgentM's LLM calls through AReaL's proxy.

    The new AgentMAgent honors ``AGENTM_PROVIDER`` / ``AGENTM_MODEL``
    plus ``OPENAI_BASE_URL`` / ``ANTHROPIC_BASE_URL`` / ``OPENAI_API_KEY``
    / ``ANTHROPIC_API_KEY``. AReaL injects an OpenAI-compatible proxy
    via ``RuntimeContext.model_endpoint`` + ``api_key``; we forward
    those to the AgentM-facing env so rollouts never bypass the
    training proxy.
    """
    updates: dict[str, str] = {}
    if runtime_context.model_endpoint:
        # AReaL's proxy is OpenAI-compatible; prefer the openai provider.
        updates["AGENTM_PROVIDER"] = str(
            runtime_context.metadata.get("agentm_provider", "openai")
        )
        updates["OPENAI_BASE_URL"] = str(runtime_context.model_endpoint)
        updates["OPENAI_API_KEY"] = runtime_context.api_key or "EMPTY"
        model = runtime_context.metadata.get("agentm_model")
        if model:
            updates["AGENTM_MODEL"] = str(model)
    if runtime_context.metadata.get("agentm_log_level"):
        updates["AGENTM_LOG_LEVEL"] = str(runtime_context.metadata["agentm_log_level"])

    old_values = {key: os.environ.get(key) for key in updates}
    for key, value in updates.items():
        os.environ[key] = value
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _extract_required_text(
    agent_input: AgentInput,
    label: str,
    keys: tuple[str, ...],
) -> str:
    for source in (agent_input.context, agent_input.metadata, agent_input.raw_sample):
        for key in keys:
            value = source.get(key)
            text = _coerce_optional_text(value)
            if text:
                return text
    raise ValueError(f"AgentMRuntime requires `{label}` in sample context or metadata")


def _coerce_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _loads_or_text(payload: str | None) -> Any:
    if not payload:
        return {}
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return payload


def _now_ms() -> int:
    return int(time.time() * 1000)
