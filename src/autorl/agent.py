"""Direct AgentM workflow for AReaL proxy-mode rollouts."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from areal.utils import logging
except ImportError:  # Keep data/verifier helpers importable without AReaL installed.
    import logging

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

        # AgentM's public RCA adapter owns the SDK session, tools and trajectory.
        from rca_eval.agent import AgentMAgent

        agent = AgentMAgent(
            scenario=self.scenario,
            provider_tuple=provider,
            max_turns=self.max_turns,
        )
        started_at = time.monotonic()
        result = await asyncio.wait_for(
            agent.run(incident=incident, data_dir=data_dir),
            timeout=self.timeout,
        )
        elapsed_seconds = time.monotonic() - started_at
        prediction = _parse_prediction(result.response)
        metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
        has_submission = bool(metadata.get("submit_final_report_seen"))
        metrics = verify_rca(
            data,
            prediction,
            data_dir=data_dir,
            has_submission=has_submission,
            reward_config=self.reward_config,
            tool_calls=_count_tool_calls(result.trajectory),
            tokens=_metadata_int(metadata, "total_tokens"),
            elapsed_seconds=elapsed_seconds,
            redundant_actions=_metadata_int(metadata, "redundant_actions"),
            invalid_actions=_metadata_int(metadata, "invalid_actions"),
            declared_objects=_metadata_int(metadata, "declared_objects"),
            violation_count=_metadata_int(metadata, "violation_count"),
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


def _count_tool_calls(trajectory: Any) -> int:
    if hasattr(trajectory, "model_dump"):
        trajectory = trajectory.model_dump(mode="python")
    if not isinstance(trajectory, Mapping):
        return 0
    count = 0
    for agent_trajectory in trajectory.get("agent_trajectories") or []:
        if not isinstance(agent_trajectory, Mapping):
            continue
        for turn in agent_trajectory.get("turns") or []:
            if not isinstance(turn, Mapping):
                continue
            for message in turn.get("messages") or []:
                if isinstance(message, Mapping):
                    count += len(message.get("tool_calls") or [])
    return count


def _metadata_int(metadata: Mapping[str, Any], key: str) -> int:
    value = metadata.get(key, 0)
    return int(value) if isinstance(value, int | float) else 0


def _log_metrics(metrics: Mapping[str, float]) -> None:
    try:
        from areal.infra import workflow_context
        from areal.utils import stats_tracker
    except ImportError:
        return
    stats_tracker.get(workflow_context.stat_scope()).scalar(**metrics)


__all__ = ["AgentMWorkflow", "resolve_data_dir"]
