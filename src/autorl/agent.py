"""Direct DeepSeek Harness workflow for AReaL proxy-mode rollouts."""

from __future__ import annotations

import asyncio
import json
import os
import random
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Unpack

from areal.utils import logging
from deepseek_harness import RunResult

from autorl.difficulty import AXES, Answer, element_weights, weighted_score
from autorl.fork import fork_prefix, steps
from autorl.fpg import parse_submission
from autorl.harness import model_route, require_bundle, run_episode, scenario_patch
from autorl.interfaces import (
    AReaLAgentWorkflow,
    AReaLRunOptions,
    DshWorkflowConfig,
    JsonValue,
    RCASample,
)
from autorl.reward import episode_reward, read_completions, step_completions, truth_for_case

logger = logging.getLogger("Dsh-RCA")

SUBMIT_TOOL = "submit_result"


@dataclass
class Sibling:
    """One response sampled at a fork point, and the continuations run from it."""

    first: str | None  # the completion that is the response itself
    answers: list[Answer]  # one per continuation, the last of which is the response's own


@dataclass
class Sample:
    """What `rescore_group` needs from one rollout after it has run."""

    answer: Answer
    outcome: float
    child_ids: set[str] = field(default_factory=set)  # every completion a fork produced
    forked_at: str | None = None  # the parent's own completion at the fork step
    siblings: list[Sibling] = field(default_factory=list)


class DshWorkflow(AReaLAgentWorkflow):
    """Run one RCA case with DeepSeek Harness and return its per-completion rewards."""

    def __init__(self, econfig: DshWorkflowConfig | None = None) -> None:
        config = econfig or {}
        self.scenario = str(config.get("scenario") or "rca")
        self.model = str(config.get("model") or "default")
        self.max_tokens = int(config.get("max_tokens") or 8192)
        self.context_window = int(config.get("context_window") or 0)
        self.timeout = float(config.get("timeout") or 1800.0)
        # The same number `apply_reward_discount` uses. The rewards this
        # workflow returns are differences between turn values, and the
        # differencing only inverts the accumulation if both sides agree —
        # so it is read from the config rather than assumed here.
        self.turn_discount = float(str(config.get("turn_discount") or 1.0))
        self.difficulty = bool(config.get("difficulty", True))
        self.centring = str(config.get("centring") or "rloo")
        if self.centring not in CENTRINGS:
            raise ValueError(f"econfig.centring must be one of {sorted(CENTRINGS)}")
        self.temperature = config.get("temperature")
        self.fork_every = int(str(config.get("fork_every") or 0))
        self.fork_siblings = int(str(config.get("fork_siblings") or 4))
        self.fork_continuations = int(str(config.get("fork_continuations") or 2))
        self.dataset_root = str(config.get("dataset_root") or os.getenv("RCA_DATASET_ROOT") or "")
        self.dsh_home = (
            Path(str(config.get("dsh_home") or os.getenv("DSH_HOME") or ".runs/dsh-home"))
            .expanduser()
            .resolve()
        )
        # One workflow instance serves every sample of a group, which is what
        # lets `rescore_group` see them together.
        self._samples: dict[int, Sample] = {}
        scenario_patch(self.scenario)  # fail at construction, not mid-rollout
        require_bundle(self.dsh_home)

    async def run(
        self,
        data: RCASample,
        **extra_kwargs: Unpack[AReaLRunOptions],
    ) -> dict[str, float]:
        base_url = extra_kwargs.get("base_url")
        if not base_url:
            raise ValueError("AReaL did not provide a rollout proxy base_url")
        api_key = str(extra_kwargs.get("api_key") or "EMPTY")

        incident = _required_text(data, ("incident", "question", "prompt"))
        data_dir = resolve_data_dir(data, self.dataset_root)
        session_id = f"rca-{uuid.uuid4().hex}"
        sample = _sample_index()
        # Under ReMax the group's first sample is the greedy baseline.
        temperature = 0.0 if self.centring == "remax" and sample == 0 else self.temperature
        result = await asyncio.to_thread(
            self._run_episode, incident, data_dir, str(base_url), api_key, session_id, temperature
        )
        # A row whose id is 0 is a row, so this is not an `or` chain.
        case_id = next(
            (data[key] for key in ("id", "source", "datapack_name") if data.get(key) is not None),
            None,
        )
        submission = submitted_result(result)
        truth = truth_for_case(Path(data_dir))
        answer = answer_of(submission, truth)
        # The flat graph score. `rescore_group` replaces it with the advantage
        # once the siblings are in.
        outcome = weighted_score(answer, {})
        completions = read_completions(self.dsh_home, session_id)
        episode = episode_reward(
            events=result.events,
            completions=completions,
            outcome=outcome,
            turn_discount=self.turn_discount,
        )
        logger.info(
            f"Finished RCA episode: case={case_id} finish_reason={result.finish_reason} "
            f"submitted={submission is not None} outcome={outcome:.3f} "
            f"rewarded={len(episode.shaped)} unmapped={episode.unmapped}"
        )
        record = Sample(answer, outcome)
        rewards = dict(episode.shaped)
        if self._forks(sample):
            fork = await self._fork(
                result, completions, incident, data_dir, str(base_url), api_key, session_id, truth
            )
            if fork is not None:
                record.forked_at, record.siblings, record.child_ids = fork
                # The children's rows are placed here and valued in
                # `rescore_group`; zero keeps the parent's accumulation intact.
                rewards.update(dict.fromkeys(record.child_ids, 0.0))
        # Keyed by sample index, which AReaL sets before the episode runs.
        self._samples[sample] = record
        # A dict addresses turns by the completion the proxy cached; a float
        # would land the whole episode on its last one.
        return rewards

    async def rescore_group(self, results: list[Any]) -> list[Any] | None:
        """Turn a prompt's scores into advantages, sample against sample.

        The group is visible nowhere else, so this is where §3 happens: the
        sibling difficulty weighting, the centring, and the fork advantages.
        The rows arrive already accumulated backward, so what is written here
        is each row's final value: every row of a trajectory carries its
        advantage, a fork sibling's own response carries the fork advantage,
        and everything a fork produced beyond that carries nothing.
        """
        samples = self._samples
        self._samples = {}
        if len(samples) != len(results) or any(r is None for r in results):
            # A sample was rejected, or ran without recording. Weights read off
            # an incomplete group would call its missing parts hard.
            return None
        ordered = [samples[index] for index in sorted(samples)]
        answers = [record.answer for record in ordered]
        weights = (
            {axis: element_weights(answers, axis) for axis, _ in AXES} if self.difficulty else {}
        )
        scores = [weighted_score(answer, weights) for answer in answers]
        advantages = centre(scores, self.centring)
        for result, record, advantage in zip(results, ordered, advantages, strict=True):
            for completion_id in result:
                result[completion_id].reward = (
                    0.0 if completion_id in record.child_ids else advantage
                )
            if not record.siblings:
                continue
            values = [
                mean(weighted_score(a, weights) for a in sibling.answers)
                for sibling in record.siblings
            ]
            for index, sibling in enumerate(record.siblings):
                others = values[:index] + values[index + 1 :]
                fork_advantage = values[index] - mean(others) if others else 0.0
                if sibling.first in result:
                    # Spec §5: the siblings replace the parent's turn at 1/K_b each.
                    result[sibling.first].reward = fork_advantage / len(record.siblings)
            if record.forked_at in result:
                result[record.forked_at].reward = 0.0
        return results

    def _forks(self, sample: int) -> bool:
        """Whether this sample is the one its group forks (spec §4, §6 budget)."""
        if not self.fork_every:
            return False
        designated = 1 if self.centring == "remax" else 0
        return sample == designated and hash(_task_id()) % self.fork_every == 0

    async def _fork(
        self,
        parent: RunResult,
        completions: list[dict[str, Any]],
        incident: str,
        data_dir: str,
        base_url: str,
        api_key: str,
        session_id: str,
        truth: Any,
    ) -> tuple[str | None, list[Sibling], set[str]] | None:
        """Sample K_b responses at one step of the parent and run each to the end.

        The step is drawn uniformly, which is the random-point control of
        Ablation 2; the draft-disagreement rule needs drafts this does not
        sample yet. Continuations beyond the first fork the child at its own
        second step, so they share the response and differ after it.
        """
        candidates = steps(parent.events)
        if len(candidates) < 2:
            return None
        step = random.Random(session_id).choice(candidates[1:])
        prefix = fork_prefix(parent.events, step)
        forked_at = step_completions(parent.events, completions).get(step)

        async def continue_from(prefix: dict[str, Any], child_id: str) -> RunResult:
            path = self.dsh_home / "rca-forks" / f"{child_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(prefix), encoding="utf-8")
            return await asyncio.to_thread(
                self._run_episode,
                incident,
                data_dir,
                base_url,
                api_key,
                child_id,
                self.temperature,
                str(path),
            )

        async def sibling(index: int) -> tuple[Sibling, set[str]]:
            child_id = f"{session_id}-s{step}b{index}"
            child = await continue_from(prefix, child_id)
            rows = read_completions(self.dsh_home, child_id)
            first = next((str(c["responseId"]) for c in rows if c.get("purpose") == "agent"), None)
            ids = {str(c["responseId"]) for c in rows if c.get("responseId")}
            answers = [answer_of(submitted_result(child), truth)]
            # The response is the child's first step; later continuations
            # branch from its second, when it has one.
            for extra in range(1, self.fork_continuations):
                if len(steps(child.events)) < 2:
                    break
                grandchild_id = f"{child_id}m{extra}"
                grandchild = await continue_from(
                    fork_prefix(child.events, 2, base=prefix), grandchild_id
                )
                ids |= {
                    str(c["responseId"])
                    for c in read_completions(self.dsh_home, grandchild_id)
                    if c.get("responseId")
                }
                answers.append(answer_of(submitted_result(grandchild), truth))
            return Sibling(first, answers), ids

        outcomes = await asyncio.gather(*(sibling(b) for b in range(self.fork_siblings)))
        siblings = [s for s, _ in outcomes]
        child_ids = set().union(*(ids for _, ids in outcomes))
        logger.info(
            f"Forked {session_id} at step {step}: {len(siblings)} siblings, "
            f"{len(child_ids)} completions"
        )
        return forked_at, siblings, child_ids

    def _run_episode(
        self,
        incident: str,
        data_dir: str,
        base_url: str,
        api_key: str,
        session_id: str,
        temperature: float | None,
        fork: str | None = None,
    ) -> RunResult:
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
            temperature=temperature,
            fork=fork,
        )
        return run_episode(
            dsh_home=self.dsh_home,
            route=route,
            cwd=data_dir,
            prompt=incident,
            session_id=session_id,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )


CENTRINGS = {"rloo", "grpo", "remax"}


def centre(scores: list[float], method: str) -> list[float]:
    """The advantages of a group's scores, by the centring of spec §3.2."""
    if method == "remax":
        return [0.0] + [score - scores[0] for score in scores[1:]]
    if method == "grpo":
        mu = mean(scores)
        sd = pstdev(scores)
        return [(score - mu) / sd if sd > 0 else 0.0 for score in scores]
    return [
        score - mean(scores[:i] + scores[i + 1 :]) if len(scores) > 1 else 0.0
        for i, score in enumerate(scores)
    ]


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


def _sample_index() -> int:
    """Which sample of its group this episode is; 0 when nothing says."""
    try:
        from areal.infra import workflow_context

        return int(getattr(workflow_context.get(), "sample_idx", 0) or 0)
    except Exception:
        return 0


def _task_id() -> str:
    """The prompt this episode belongs to, as AReaL names it; "" when nothing says."""
    try:
        from areal.infra import workflow_context

        return str(getattr(workflow_context.get(), "task_id", "") or "")
    except Exception:
        return ""


def answer_of(submission: dict[str, JsonValue] | None, truth: Any) -> Answer:
    """One sample's claimed and correct elements, as the sets difficulty reads.

    Three axes, all on subjects: the root causes, every node, and every edge as
    an ordered subject pair. Predicates are left out, the way `fpg`'s own
    scalar leaves them out.
    """
    truths = _elements(
        truth.graph.nodes, truth.graph.edges, [n.id for n in truth.graph.root_causes]
    )
    if submission is None:
        claimed = {axis: frozenset[str]() for axis in truths}
    else:
        answer = parse_submission(submission)
        claimed = _elements(answer.nodes, answer.edges, answer.root_causes)
    found = {axis: claimed[axis] & truths[axis] for axis in truths}
    return Answer(found=found, truth=truths, claimed=claimed)


def _elements(nodes: Any, edges: Any, root_ids: Any) -> dict[str, frozenset[str]]:
    subject = {node.id: str(node.subject) for node in nodes}
    return {
        "roots": frozenset(subject[i] for i in root_ids if i in subject),
        "subjects": frozenset(subject.values()),
        "edges": frozenset(f"{subject[e.src]}->{subject[e.dst]}" for e in edges),
    }
