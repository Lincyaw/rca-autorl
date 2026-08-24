"""Direct AgentM workflow for AReaL proxy-mode rollouts."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Unpack, cast

from agentm import (
    AgentSession,
    AgentSessionConfig,
    LoopConfig,
    ScenarioLoader,
    ScenarioSpec,
    load_scenario_manifest,
)
from areal.utils import logging

from autorl.interfaces import (
    AgentMWorkflowConfig,
    AReaLAgentWorkflow,
    AReaLRunOptions,
    RCASample,
)

logger = logging.getLogger("AgentM-RCA")


class AgentMWorkflow(AReaLAgentWorkflow):
    """Run one RCA case with AgentM and return the placeholder reward."""

    def __init__(self, econfig: AgentMWorkflowConfig | None = None) -> None:
        config = econfig or {}
        self.scenario = str(config.get("scenario") or "rca")
        self.model = str(config.get("model") or "default")
        self.max_turns = int(config.get("max_turns") or 128)
        self.timeout = float(config.get("timeout") or 1800.0)
        self.dataset_root = str(
            config.get("dataset_root") or os.getenv("AGENTM_RCA_DATASET_ROOT") or ""
        )

    async def run(
        self,
        data: RCASample,
        **extra_kwargs: Unpack[AReaLRunOptions],
    ) -> float:
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
                scenario_loader=cast(ScenarioLoader, _load_scenario),
                provider=provider,
                loop_config=LoopConfig(
                    max_turns=self.max_turns,
                    max_tool_calls=self.max_turns * 20,
                ),
            )
        )
        try:
            await asyncio.wait_for(session.run(incident), timeout=self.timeout)
            final_result = session.final_result()
        finally:
            await session.shutdown()
        has_submission = bool(
            final_result is not None and final_result.reason == "structured_output:submitted"
        )
        case_id = data.get("id") or data.get("source") or data.get("datapack_name")
        logger.info(f"Finished RCA episode: case={case_id} submitted={has_submission}")
        return 0.0


def resolve_data_dir(sample: RCASample, dataset_root: str = "") -> str:
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


def _required_text(sample: RCASample, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = sample.get(key)
        if value and str(value).strip():
            return str(value).strip()
    raise ValueError(f"RCA sample needs one of: {', '.join(keys)}")


def _load_scenario(name: str) -> ScenarioSpec:
    manifest = Path(__file__).parents[2] / "contrib" / "scenarios" / name / "manifest.yaml"
    return load_scenario_manifest(manifest, requested_name=name)


__all__ = ["AgentMWorkflow", "resolve_data_dir"]
