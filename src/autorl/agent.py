"""Direct DeepSeek Harness workflow for AReaL proxy-mode rollouts."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Unpack

from areal.utils import logging
from deepseek_harness import RunResult

from autorl.harness import model_route, require_bundle, run_episode, scenario_patch
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
        self.max_tokens = int(config.get("max_tokens") or 8192)
        self.context_window = int(config.get("context_window") or 0)
        self.timeout = float(config.get("timeout") or 1800.0)
        self.dataset_root = str(config.get("dataset_root") or os.getenv("RCA_DATASET_ROOT") or "")
        self.dsh_home = (
            Path(str(config.get("dsh_home") or os.getenv("DSH_HOME") or ".runs/dsh-home"))
            .expanduser()
            .resolve()
        )
        scenario_patch(self.scenario)  # fail at construction, not mid-rollout
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
        # A row whose id is 0 is a row, so this is not an `or` chain.
        case_id = next(
            (data[key] for key in ("id", "source", "datapack_name") if data.get(key) is not None),
            None,
        )
        submission = submitted_result(result)
        logger.info(
            f"Finished RCA episode: case={case_id} finish_reason={result.finish_reason} "
            f"submitted={submission is not None}"
        )
        return 0.0

    def _run_episode(self, incident: str, data_dir: str, base_url: str, api_key: str) -> RunResult:
        # AReaL's proxy is reached as a declared route, the same way the SFT
        # collector reaches a teacher endpoint; `model_route` says why a
        # `base_url` override is not enough. The route also carries
        # DSH_CONTEXT_WINDOW, which is what compaction triggers below and which
        # the harness otherwise assumes is 1M.
        route = model_route(
            scenario=self.scenario,
            model=self.model,
            base_url=base_url,
            api_key=api_key,
            context_window=self.context_window,
        )
        return run_episode(
            dsh_home=self.dsh_home,
            route=route,
            cwd=data_dir,
            prompt=incident,
            session_id=f"rca-{uuid.uuid4().hex}",
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )


def resolve_data_dir(sample: RCASample, dataset_root: str = "") -> str:
    for key in ("data_dir", "case_dir", "observability_dir"):
        value = sample.get(key)
        if value and str(value).strip():
            return str(Path(str(value)).expanduser())

    datapack = sample.get("datapack_name") or sample.get("data_pack_name")
    if datapack and dataset_root:
        root = Path(dataset_root).expanduser()
        # `datapacks/ops-lite` keeps its snapshots one level down, beside the
        # manifest they are listed in, so the dataset root a caller names is the
        # corpus and not the case directory.
        for candidate in (root / str(datapack), root / "cases" / str(datapack)):
            if candidate.is_dir():
                return str(candidate)
        return str(root / str(datapack))
    raise ValueError(
        "RCA sample needs data_dir, or datapack_name with econfig.dataset_root/RCA_DATASET_ROOT"
    )


def accepted_call_ids(events: Sequence[dict[str, Any]]) -> set[str]:
    """Call ids whose `tool/result` is not an error."""
    accepted = set()
    for event in events:
        if event.get("type") != "tool/result":
            continue
        block = event.get("data", {}).get("message", {}).get("content", [{}])[0]
        call_id = block.get("toolCallId")
        if isinstance(call_id, str) and not block.get("isError", False):
            accepted.add(call_id)
    return accepted


def submitted_result(result: RunResult) -> dict[str, JsonValue] | None:
    """The arguments of the episode's accepted `submit_result` call, or None.

    Only a call the tool actually executed counts. `submit_result` is terminal —
    `execute` ends the turn and arms a monotonic guard — but argument-schema
    validation runs *before* `execute`, so a call the registry rejects neither
    ends the turn nor arms the guard, and the model retries inside the same
    turn. A `tool/call` event is written before execution either way, so the log
    can hold several, and the rejected one comes first.

    Reading that first call is not a near miss: a submission rejected for
    missing `edges` and `root_causes` is exactly the shape a verifier scores
    zero, which would punish an episode for recovering rather than for failing.
    On the first ten collected episodes three took that path. Hence the pairing
    by call id against the result the tool returned.
    """
    accepted = accepted_call_ids(result.events)
    for event in result.events:
        if event.get("type") != "tool/call":
            continue
        data = event.get("data")
        if not isinstance(data, dict) or data.get("name") != SUBMIT_TOOL:
            continue
        if data.get("callId") not in accepted:
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


__all__ = ["DshWorkflow", "accepted_call_ids", "resolve_data_dir", "submitted_result"]
