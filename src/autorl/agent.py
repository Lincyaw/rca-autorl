"""Direct DeepSeek Harness workflow for AReaL proxy-mode rollouts."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Unpack

from areal.utils import logging
from deepseek_harness import DeepSeekHarness, RunResult

from autorl.harness import PROFILE, require_bundle, scenario_patch
from autorl.interfaces import (
    AReaLAgentWorkflow,
    AReaLRunOptions,
    DshWorkflowConfig,
    JsonValue,
    RCASample,
)

logger = logging.getLogger("Dsh-RCA")

SUBMIT_TOOL = "submit_result"


class DshWorkflow(AReaLAgentWorkflow):
    """Run one RCA case with DeepSeek Harness and return the placeholder reward."""

    def __init__(self, econfig: DshWorkflowConfig | None = None) -> None:
        config = econfig or {}
        self.scenario = str(config.get("scenario") or "rca")
        self.model = str(config.get("model") or "default")
        self.max_tokens = int(config.get("max_tokens") or 4096)
        self.context_window = int(config.get("context_window") or 0)
        self.timeout = float(config.get("timeout") or 1800.0)
        self.dataset_root = str(config.get("dataset_root") or os.getenv("RCA_DATASET_ROOT") or "")
        self.dsh_home = (
            Path(str(config.get("dsh_home") or os.getenv("DSH_HOME") or ".runs/dsh-home"))
            .expanduser()
            .resolve()
        )
        self.patch = scenario_patch(self.scenario)
        require_bundle(self.dsh_home)

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
        result = await asyncio.to_thread(
            self._run_episode, incident, data_dir, str(base_url), api_key
        )
        case_id = data.get("id") or data.get("source") or data.get("datapack_name")
        submission = submitted_result(result)
        logger.info(
            f"Finished RCA episode: case={case_id} finish_reason={result.finish_reason} "
            f"submitted={submission is not None}"
        )
        return 0.0

    def _run_episode(self, incident: str, data_dir: str, base_url: str, api_key: str) -> RunResult:
        with DeepSeekHarness(
            dsh_home=str(self.dsh_home),
            profile=PROFILE,
            patches=(self.patch,),
            cwd=data_dir,
            model=self.model,
            max_tokens=self.max_tokens,
            base_url=base_url,
            api_key=api_key,
            # Compaction triggers at a fraction of the model's context window,
            # which the harness assumes is 1M unless told the serving limit.
            env=({"DSH_CONTEXT_WINDOW": str(self.context_window)} if self.context_window else {}),
            request_timeout_seconds=self.timeout,
        ) as harness:
            return harness.run(incident, session_id=f"rca-{uuid.uuid4().hex}")


def resolve_data_dir(sample: RCASample, dataset_root: str = "") -> str:
    for key in ("data_dir", "case_dir", "observability_dir"):
        value = sample.get(key)
        if value and str(value).strip():
            return str(Path(str(value)).expanduser())

    datapack = sample.get("datapack_name") or sample.get("data_pack_name")
    if datapack and dataset_root:
        return str(Path(dataset_root).expanduser() / str(datapack))
    raise ValueError(
        "RCA sample needs data_dir, or datapack_name with econfig.dataset_root/RCA_DATASET_ROOT"
    )


def submitted_result(result: RunResult) -> dict[str, JsonValue] | None:
    """The arguments of the episode's `submit_result` call, or None when it never came.

    The harness bundle's tool is terminal, so at most one call reaches the log.
    This is where the verifier will read the fault propagation graph from.
    """
    for event in result.events:
        if event.get("type") != "tool/call":
            continue
        data = event.get("data")
        if not isinstance(data, dict) or data.get("name") != SUBMIT_TOOL:
            continue
        arguments = data.get("arguments")
        if not isinstance(arguments, str):
            continue
        parsed = json.loads(arguments)
        if isinstance(parsed, dict):
            return parsed
    return None


def _required_text(sample: RCASample, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = sample.get(key)
        if value and str(value).strip():
            return str(value).strip()
    raise ValueError(f"RCA sample needs one of: {', '.join(keys)}")


__all__ = ["DshWorkflow", "resolve_data_dir", "submitted_result"]
