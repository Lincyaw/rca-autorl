"""Direct AgentM workflow for AReaL proxy-mode rollouts."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agentm import (
    AgentSession,
    AgentSessionConfig,
    LoopConfig,
    ScenarioSpec,
    load_scenario_manifest,
)
from areal.infra import workflow_context
from areal.utils import logging, stats_tracker

from autorl.algorithm import RCARewardConfig
from autorl.verifier import verify_rca

logger = logging.getLogger("AgentM-RCA")


class AgentMWorkflow:
    """Run one RCA case with AgentM and return its verifier reward."""

    def __init__(self, econfig: dict[str, Any] | None = None) -> None:
        config = econfig or {}
        self.scenario = str(config.get("scenario") or "rca")
        self.model = str(config.get("model") or "default")
        self.max_turns = int(config.get("max_turns") or 128)
        self.timeout = float(config.get("timeout") or 1800.0)
        self.dataset_root = str(
            config.get("dataset_root") or os.getenv("AGENTM_RCA_DATASET_ROOT") or ""
        )
        reward_config = config.get("reward")
        self.reward_config = RCARewardConfig.from_mapping(
            reward_config if isinstance(reward_config, Mapping) else None
        )

    async def run(self, data: dict[str, Any], **extra_kwargs: Any) -> float:
        base_url = extra_kwargs.get("base_url")
        if not base_url:
            raise ValueError("AReaL did not provide a rollout proxy base_url")
        api_key = str(extra_kwargs.get("api_key") or "EMPTY")

        incident = _required_text(data, ("incident", "question", "prompt"))
        data_dir = resolve_data_dir(data, self.dataset_root)
        provider = (
            "agentm.extensions.builtin.llm_openai",
            {
                "name": "areal-rollout",
                "model": self.model,
                "base_url": str(base_url),
                "api_key": api_key,
            },
        )

        session = await AgentSession.create(
            AgentSessionConfig(
                cwd=data_dir,
                scenario=self.scenario,
                scenario_loader=_load_scenario,
                provider=provider,
                loop_config=LoopConfig(
                    max_turns=self.max_turns,
                    max_tool_calls=self.max_turns * 20,
                ),
            )
        )
        started_at = time.monotonic()
        try:
            await asyncio.wait_for(session.run(incident), timeout=self.timeout)
            final_result = session.final_result()
            turns = session.get_turns()
        finally:
            await session.shutdown()
        elapsed_seconds = time.monotonic() - started_at
        response = final_result.text if final_result is not None else ""
        prediction = _parse_prediction(response)
        has_submission = bool(
            final_result is not None and final_result.reason == "structured_output:submitted"
        )
        usage = _session_usage(turns)
        metrics = verify_rca(
            data,
            prediction,
            data_dir=data_dir,
            has_submission=has_submission,
            reward_config=self.reward_config,
            tool_calls=usage["tool_calls"],
            tokens=usage["tokens"],
            elapsed_seconds=elapsed_seconds,
            invalid_actions=usage["invalid_actions"],
        )
        case_id = data.get("id") or data.get("source") or data.get("datapack_name")
        logger.info(
            f"Finished RCA episode: case={case_id} "
            f"reward={metrics['reward']:.4f} submitted={has_submission}"
        )
        _log_metrics(metrics)
        # AReaL v2 interprets a dict as {completion_id: reward}; diagnostic
        # metrics must be logged separately and the workflow must return a scalar.
        return metrics["reward"]


def resolve_data_dir(sample: Mapping[str, Any], dataset_root: str = "") -> str:
    for key in ("data_dir", "case_dir", "observability_dir"):
        value = sample.get(key)
        if value and str(value).strip():
            return str(Path(str(value)).expanduser())

    datapack = sample.get("datapack_name") or sample.get("data_pack_name")
    if datapack and dataset_root:
        return str(Path(dataset_root).expanduser() / str(datapack))
    raise ValueError(
        "RCA sample needs data_dir, or datapack_name with econfig.dataset_root/"
        "AGENTM_RCA_DATASET_ROOT"
    )


def _required_text(sample: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = sample.get(key)
        if value and str(value).strip():
            return str(value).strip()
    raise ValueError(f"RCA sample needs one of: {', '.join(keys)}")


def _parse_prediction(response: str | None) -> Any:
    if not response:
        return {}
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return response


def _load_scenario(name: str) -> ScenarioSpec:
    manifest = Path(__file__).parents[2] / "contrib" / "scenarios" / name / "manifest.yaml"
    return load_scenario_manifest(manifest, requested_name=name)


def _session_usage(turns: list[Any]) -> dict[str, int]:
    tool_calls = 0
    tokens = 0
    invalid_actions = 0
    for turn in turns:
        response = getattr(turn, "response", None)
        usage = getattr(response, "usage", None)
        if usage is not None:
            tokens += int(usage.input_tokens) + int(usage.output_tokens)
        records = getattr(turn, "tool_results", ())
        tool_calls += len(records)
        invalid_actions += sum(bool(record.result.is_error) for record in records)
    return {
        "tool_calls": tool_calls,
        "tokens": tokens,
        "invalid_actions": invalid_actions,
    }


def _log_metrics(metrics: Mapping[str, float]) -> None:
    stats_tracker.get(workflow_context.stat_scope()).scalar(**metrics)


__all__ = ["AgentMWorkflow", "resolve_data_dir"]
