"""Direct AgentM workflow for AReaL proxy-mode rollouts."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from areal.utils import logging
except ImportError:  # Keep data/verifier helpers importable without AReaL installed.
    import logging

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

    async def run(self, data: dict[str, Any], **extra_kwargs: Any) -> dict[str, float]:
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

        # AgentM owns the agent loop, tools, trajectory persistence and result schema.
        from rca_eval.agent import AgentMAgent

        agent = AgentMAgent(
            scenario=self.scenario,
            provider_tuple=provider,
            max_turns=self.max_turns,
        )
        result = await asyncio.wait_for(
            agent.run(incident=incident, data_dir=data_dir),
            timeout=self.timeout,
        )
        prediction = _parse_prediction(result.response)
        has_submission = bool((result.metadata or {}).get("submit_final_report_seen"))
        reward = verify_rca(
            data,
            prediction,
            data_dir=data_dir,
            has_submission=has_submission,
        )
        case_id = data.get("id") or data.get("source") or data.get("datapack_name")
        logger.info(
            f"Finished RCA episode: case={case_id} "
            f"reward={reward['reward']:.4f} submitted={has_submission}"
        )
        return reward


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


__all__ = ["AgentMWorkflow", "resolve_data_dir"]
