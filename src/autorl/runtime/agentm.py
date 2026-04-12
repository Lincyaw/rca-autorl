from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
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

_AGENTM_RUN_LOCK: asyncio.Lock | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


class AgentMRuntime(AgentRuntime):
    """Run RCA episodes through the vendored AgentM headless runner."""

    def __init__(
        self,
        scenario_dir: str | None = None,
        config_path: str | None = None,
        max_steps: int = 100,
        timeout: float = 600.0,
        exp_id: str | None = None,
    ) -> None:
        repo_root = _repo_root()
        self.scenario_dir = str(
            Path(scenario_dir)
            if scenario_dir is not None
            else repo_root / "third_party" / "agentm" / "config" / "scenarios" / "rca_hypothesis"
        )
        self.config_path = str(
            Path(config_path)
            if config_path is not None
            else repo_root / "third_party" / "agentm" / "config" / "system.yaml"
        )
        self.max_steps = max_steps
        self.timeout = timeout
        self.exp_id = exp_id

    async def run(
        self,
        agent_input: AgentInput,
        runtime_context: RuntimeContext,
    ) -> AgentRunResult:
        from agentm.cli.run import run_investigation_headless

        incident = _extract_required_text(agent_input, "incident", ("incident", "question", "prompt"))
        data_dir = _extract_required_text(agent_input, "data_dir", ("data_dir", "case_dir", "observability_dir"))
        started_at = _now_ms()

        # Vendored AgentM still reads per-request config from process env,
        # so serialize runs while those overrides are in place.
        async with _get_agentm_run_lock():
            with _agentm_env_overrides(runtime_context):
                structured_response_json, trajectory_json, run_id, trajectory_file_path = await run_investigation_headless(
                    data_dir=data_dir,
                    incident=incident,
                    scenario_dir=self.scenario_dir,
                    config_path=self.config_path,
                    max_steps=min(self.max_steps, runtime_context.limits.max_steps or self.max_steps),
                    timeout=runtime_context.limits.timeout_seconds or self.timeout,
                    exp_id=self.exp_id or _coerce_optional_text(agent_input.metadata.get("exp_id")),
                )

        prediction = _loads_or_text(structured_response_json)
        metadata = {
            "incident": incident,
            "data_dir": data_dir,
            "scenario_dir": self.scenario_dir,
            "config_path": self.config_path,
            "agentm_run_id": run_id,
            "agentm_trajectory_file": trajectory_file_path,
            "used_areal_proxy": bool(runtime_context.model_endpoint),
            **agent_input.metadata,
        }
        final_output = {
            "prediction": prediction,
            "structured_response_json": structured_response_json,
            "run_id": run_id,
            "trajectory_file": trajectory_file_path,
            "termination": "agentm_complete",
        }
        steps = [
            TrajectoryStep(
                step_id="agentm-dispatch",
                step_type=TrajectoryStepType.CONTROL,
                timestamp_ms=started_at,
                input={
                    "incident": incident,
                    "data_dir": data_dir,
                    "scenario_dir": self.scenario_dir,
                },
                metadata={"runtime": "agentm"},
            ),
            TrajectoryStep(
                step_id="agentm-final",
                step_type=TrajectoryStepType.FINAL,
                timestamp_ms=_now_ms(),
                output={"prediction": prediction, "run_id": run_id},
                usage=TokenUsage(),
                metadata={"runtime": "agentm"},
            ),
        ]
        trajectory = Trajectory(
            trajectory_id=uuid4().hex,
            sample_id=agent_input.sample_id,
            task_type=agent_input.task_type,
            status=TrajectoryStatus.COMPLETED,
            final_output=final_output,
            steps=steps,
            summary_stats={
                "has_agentm_trajectory": 1.0 if trajectory_json else 0.0,
                "structured_response_bytes": float(len(structured_response_json or "")),
            },
            artifacts={"agentm_trajectory_json": trajectory_json} if trajectory_json else {},
            metadata=metadata,
        )
        outcome = TaskOutcome(
            sample_id=agent_input.sample_id,
            task_type=agent_input.task_type,
            prediction=prediction,
            reference=None,
            success=None,
            termination_reason="agentm_complete",
            metrics=dict(trajectory.summary_stats),
            metadata=metadata,
        )
        return AgentRunResult(trajectory=trajectory, outcome=outcome)


def _get_agentm_run_lock() -> asyncio.Lock:
    global _AGENTM_RUN_LOCK
    if _AGENTM_RUN_LOCK is None:
        _AGENTM_RUN_LOCK = asyncio.Lock()
    return _AGENTM_RUN_LOCK


@contextmanager
def _agentm_env_overrides(runtime_context: RuntimeContext) -> Iterator[None]:
    updates: dict[str, str] = {}
    if runtime_context.model_endpoint:
        updates["AGENTM_API_BASE_URL"] = str(runtime_context.model_endpoint)
        updates["AGENTM_API_KEY"] = runtime_context.api_key or "EMPTY"
        updates["AGENTM_ORCHESTRATOR_MODEL"] = str(
            runtime_context.metadata.get("agentm_orchestrator_model", "default")
        )
        updates["AGENTM_WORKER_MODEL"] = str(
            runtime_context.metadata.get("agentm_worker_model", "default")
        )
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
