"""Direct DeepSeek Harness workflow for AReaL proxy-mode rollouts."""

from __future__ import annotations

import asyncio
import json
import os
import random
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from areal.infra import workflow_context
from areal.utils import logging, stats_tracker
from deepseek_harness import RunResult

from autorl.difficulty import AXES, Answer, element_weights, weighted_score
from autorl.fork import fork_prefix, steps
from autorl.harness import model_route, require_bundle, run_episode, scenario_patch
from autorl.reward import answer_of, step_ids, submitted_result, truth_for_case
from autorl.trajectory import episode_metrics

logger = logging.getLogger("Dsh-RCA")

# An episode holds a thread for as long as the harness runs, up to `timeout`.
# asyncio's default pool stops at a few dozen, which would queue episodes
# silently once a batch and its forks run concurrently.
EPISODES = ThreadPoolExecutor(max_workers=256, thread_name_prefix="rca-episode")


@dataclass
class Sibling:
    """One response sampled at a fork point, and the continuations run from it."""

    first: str | None  # the completion that is the response itself
    answers: list[Answer]  # one per continuation, the first of which is the response's own


@dataclass
class Sample:
    """What `rescore_group` needs from one rollout after it has run."""

    answer: Answer
    own: set[str] = field(default_factory=set)  # the completions behind the trajectory's steps
    forked_at: str | None = None  # the parent's own completion at the fork step
    siblings: list[Sibling] = field(default_factory=list)


class DshWorkflow:
    """Run one RCA case with DeepSeek Harness and return its outcome."""

    def __init__(self, econfig: dict[str, Any] | None = None) -> None:
        config = econfig or {}
        self.scenario = str(config.get("scenario") or "rca")
        self.model = str(config.get("model") or "default")
        self.max_tokens = int(config.get("max_tokens") or 8192)
        self.context_window = int(config.get("context_window") or 0)
        self.timeout = float(config.get("timeout") or 1800.0)
        self.difficulty = bool(config.get("difficulty", True))
        self.centring = str(config.get("centring") or "rloo")
        if self.centring not in CENTRINGS:
            raise ValueError(f"econfig.centring must be one of {sorted(CENTRINGS)}")
        # Under ReMax the group's first sample is decoded greedily and is the
        # baseline (`centre`); the forked sample, when any, is the next one.
        self.greedy: int | None = 0 if self.centring == "remax" else None
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
        # One instance serves a whole batch, so the groups in flight are told
        # apart by the task id AReaL sets, and their samples by index.
        self._groups: dict[int | None, dict[int, Sample]] = {}
        scenario_patch(self.scenario)  # fail at construction, not mid-rollout
        require_bundle(self.dsh_home)

    async def run(self, data: dict[str, Any], **extra_kwargs: Any) -> float:
        base_url = extra_kwargs.get("base_url")
        if not base_url:
            raise ValueError("AReaL did not provide a rollout proxy base_url")
        api_key = str(extra_kwargs.get("api_key") or "EMPTY")

        incident = str(data.get("question") or "").strip()
        if not incident:
            raise ValueError("RCA sample has no question")
        data_dir = resolve_data_dir(data, self.dataset_root)
        session_id = f"rca-{uuid.uuid4().hex}"
        context = workflow_context.get()
        sample = int(context.sample_idx or 0)
        temperature = 0.0 if sample == self.greedy else self.temperature
        result = await self._episode(
            incident, data_dir, str(base_url), api_key, session_id, temperature
        )
        submission = submitted_result(result.events)
        truth = truth_for_case(Path(data_dir))
        answer = answer_of(submission, truth)
        # The flat graph score. `rescore_group` replaces it with the advantage
        # once the siblings are in.
        outcome = weighted_score(answer, {})
        logger.info(
            f"Finished RCA episode: case={data.get('id')} finish_reason={result.finish_reason} "
            f"submitted={submission is not None} outcome={outcome:.3f}"
        )
        # Everything below is only in the text log otherwise. An episode that
        # never submitted and one that submitted a wrong graph both score zero,
        # and the difference — where it stopped, how close to the window it came,
        # whether compaction held — is what says which of the two to fix.
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            score=outcome,
            submitted=float(submission is not None),
            **episode_metrics(result.events),
        )
        record = Sample(answer, set(step_ids(result.events).values()))
        if self._forks(sample, data):
            await self._fork(
                record, result, incident, data_dir, str(base_url), api_key, session_id, truth
            )
        self._groups.setdefault(context.task_id, {})[sample] = record
        # The proxy lands this on the session's last completion and carries it
        # back over the others; the values that train are written below.
        return outcome

    async def rescore_group(self, results: list[Any]) -> list[Any] | None:
        """Turn a prompt's scores into advantages, sample against sample.

        The group is visible nowhere else, so this is where §3 happens: the
        sibling difficulty weighting, the centring, and the fork advantages.
        The rows arrive already accumulated, so what is written here is each
        row's final value: every step of a trajectory carries its advantage, a
        fork sibling's own response carries the fork advantage, and every other
        row the session produced — a continuation, a compaction summary —
        carries nothing.
        """
        samples = self._groups.pop(workflow_context.get().task_id, {})
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
        # The group is visible nowhere else, so this is the only place the
        # measurement of §3.2 can be taken: a group whose weighted scores are
        # equal hands every sibling a zero advantage, and a batch of those trains
        # on nothing. `group_score_std` is that quantity — the first number the
        # method's design depends on being non-zero.
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            group_score_std=pstdev(scores) if len(scores) > 1 else 0.0,
            group_score_spread=max(scores) - min(scores),
            advantage_abs_mean=mean(abs(a) for a in advantages),
        )
        for result, record, advantage in zip(results, ordered, advantages, strict=True):
            for completion_id in result:
                result[completion_id].reward = advantage if completion_id in record.own else 0.0
            if not record.siblings:
                continue
            values = [
                mean(weighted_score(a, weights) for a in sibling.answers)
                for sibling in record.siblings
            ]
            for sibling, fork_advantage in zip(
                record.siblings, centre(values, "rloo"), strict=True
            ):
                if sibling.first in result:
                    # Spec §5: the siblings replace the parent's turn at 1/K_b each.
                    result[sibling.first].reward = fork_advantage / len(record.siblings)
            if record.forked_at in result:
                result[record.forked_at].reward = 0.0
        return results

    def _forks(self, sample: int, data: dict[str, Any]) -> bool:
        """Whether this sample is the one its group forks (spec §4, §6 budget).

        One sample of every `fork_every`-th case, by the case's own manifest
        index, so the same cases fork on every run.
        """
        if not self.fork_every:
            return False
        designated = 0 if self.greedy is None else self.greedy + 1
        return sample == designated and int(data.get("id", 0)) % self.fork_every == 0

    async def _fork(
        self,
        record: Sample,
        parent: RunResult,
        incident: str,
        data_dir: str,
        base_url: str,
        api_key: str,
        session_id: str,
        truth: Any,
    ) -> None:
        """Sample K_b responses at one step of the parent and run each to the end.

        The step is drawn uniformly, which is the random-point control of
        Ablation 2; the draft-disagreement rule needs drafts this does not
        sample yet. Continuations beyond the first fork the child at its own
        second step, so they share the response and differ after it.
        """
        candidates = steps(parent.events)
        if len(candidates) < 2:
            return
        step = random.Random(session_id).choice(candidates[1:])
        prefix = fork_prefix(parent.events, step)
        record.forked_at = step_ids(parent.events).get(step)

        async def run_children(
            prefix: dict[str, Any], name: str, count: int
        ) -> list[tuple[RunResult, Answer]]:
            """`count` episodes from one restored state, side by side."""
            path = self.dsh_home / "rca-forks" / f"{name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(prefix), encoding="utf-8")

            async def one(index: int) -> tuple[RunResult, Answer]:
                result = await self._episode(
                    incident, data_dir, base_url, api_key, f"{name}c{index}", self.temperature, path
                )
                return result, answer_of(submitted_result(result.events), truth)

            try:
                return list(await asyncio.gather(*(one(i) for i in range(count))))
            finally:
                path.unlink(missing_ok=True)

        siblings = await run_children(prefix, f"{session_id}-s{step}", self.fork_siblings)
        # The response is the child's first step; later continuations branch
        # from its second, when it has one. Every sibling's continuations run
        # at once.
        branching = [
            index
            for index, (child, _) in enumerate(siblings)
            if self.fork_continuations > 1 and len(steps(child.events)) > 1
        ]
        more = dict(
            zip(
                branching,
                await asyncio.gather(
                    *(
                        run_children(
                            fork_prefix(siblings[index][0].events, 2, base=prefix),
                            f"{session_id}-s{step}c{index}m",
                            self.fork_continuations - 1,
                        )
                        for index in branching
                    )
                ),
                strict=True,
            )
        )
        for index, (child, answer) in enumerate(siblings):
            first = next(iter(step_ids(child.events).values()), None)
            answers = [answer, *(a for _, a in more.get(index, []))]
            record.siblings.append(Sibling(first, answers))
        logger.info(f"Forked {session_id} at step {step}: {len(record.siblings)} siblings")

    async def _episode(
        self,
        incident: str,
        data_dir: str,
        base_url: str,
        api_key: str,
        session_id: str,
        temperature: float | None,
        fork: Path | None = None,
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
            fork=None if fork is None else str(fork),
        )
        return await asyncio.get_running_loop().run_in_executor(
            EPISODES,
            lambda: run_episode(
                dsh_home=self.dsh_home,
                route=route,
                cwd=data_dir,
                prompt=incident,
                session_id=session_id,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            ),
        )


CENTRINGS = {"rloo", "grpo", "remax"}


def centre(scores: list[float], method: str) -> list[float]:
    """The advantages of a group's scores, by the centring of spec §3.2.

    Computed here rather than by the framework's group normalization, which
    writes one value to every row of a trajectory and would overwrite the
    per-row values a fork needs (§4).
    """
    if len(scores) < 2:
        return [0.0] * len(scores)
    if method == "remax":
        return [0.0] + [score - scores[0] for score in scores[1:]]
    if method == "grpo":
        mu = mean(scores)
        sd = pstdev(scores)
        return [(score - mu) / sd if sd > 0 else 0.0 for score in scores]
    return [score - mean(scores[:i] + scores[i + 1 :]) for i, score in enumerate(scores)]


def resolve_data_dir(sample: dict[str, Any], dataset_root: str = "") -> str:
    """The snapshot directory: `data_dir` as given, or the case under the corpus root."""
    if sample.get("data_dir"):
        return str(Path(str(sample["data_dir"])).expanduser())
    datapack = sample.get("datapack_name")
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


__all__ = ["DshWorkflow", "Sample", "Sibling", "centre", "resolve_data_dir"]
