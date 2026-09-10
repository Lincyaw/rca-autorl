"""One DeepSeek Harness session, read as the shape a reviewer wants to see.

The harness already records everything an audit needs — every request, every
tool call with its arguments, every compaction pass, and each assistant turn's
`usage` — so nothing here collects anything. This is a projection: the same
event stream `autorl.data.export` turns into SFT rows, turned instead into one
`Episode` per session for `autorl.web` to serve.

What the projection is for, concretely. A rollout that never submits looks
identical in the logs to one that submits and scores zero, and the reason it
stopped is spread over four event types: the last `turn/end`'s reason, the
`compaction/end` errors before it, the `usage.inputTokens` climbing toward the
serving window, and the tool results that came back as errors. Reading that by
hand took a dozen ad-hoc greps per episode.

The readers are separate — `read_steps`, `read_compactions`, `read_finish` — so
the training workflow can take the few numbers it reports without a session file
or a score, and the dashboard can assemble the whole `Episode` from the same
three.

The one number the harness does not record is a failed compaction request's own
prompt size: `compaction.js` throws before `assembler.usage` exists. Until that
gap is closed, `Compaction.shadowed_tokens` on a failed pass is the closest
available proxy — the region the pass was trying to replace.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from autorl.difficulty import Answer, weighted_score
from autorl.reward import answer_of, submitted_result, tool_results, truth_for_case

SESSION_FILE = "session.jsonl"
#: A tool result in full is up to `maxChars` (4000) of TSV; the head is what a
#: reviewer reads before deciding to open the raw log.
RESULT_HEAD = 400


@dataclass
class Step:
    """One agent step: what it thought, what it called, what came back."""

    index: int
    reasoning: str = ""
    text: str = ""
    tool: str = ""
    arguments: str = ""
    result_head: str = ""
    result_error: bool = False
    # The context size at this step, as the serving side counted it. This is the
    # curve that runs into the window; a step with no assistant message (a
    # compaction pass, an aborted request) carries 0 and is skipped when plotted.
    input_tokens: int = 0


@dataclass
class Compaction:
    """One compaction pass, whether or not it produced a checkpoint."""

    at_step: int
    shadowed_tokens: int = 0
    summary: str = ""
    error: str = ""

    @property
    def landed(self) -> bool:
        return not self.error and bool(self.summary)


@dataclass
class Episode:
    """One session, scored the way the trainer scores it."""

    session_id: str
    case: str
    finish: str = ""
    error: str = ""
    steps: list[Step] = field(default_factory=list)
    compactions: list[Compaction] = field(default_factory=list)
    submission: dict[str, Any] | None = None
    score: float = 0.0
    answer: Answer | None = None

    @property
    def peak_input_tokens(self) -> int:
        return peak_input_tokens(self.steps)

    @property
    def tool_counts(self) -> dict[str, int]:
        return tool_counts(self.steps)

    def summary(self) -> dict[str, Any]:
        """The one row a list view shows, without the per-step detail."""
        failed = sum(1 for c in self.compactions if c.error)
        return {
            "session_id": self.session_id,
            "case": self.case,
            "finish": self.finish,
            "error": self.error,
            "score": round(self.score, 4),
            "submitted": self.submission is not None,
            "steps": len(self.steps),
            "tools": self.tool_counts,
            "peak_input_tokens": self.peak_input_tokens,
            "compactions": len(self.compactions),
            "compaction_failures": failed,
        }


def peak_input_tokens(steps: Sequence[Step]) -> int:
    """The largest context any of these steps was served on."""
    return max((s.input_tokens for s in steps), default=0)


def tool_counts(steps: Sequence[Step]) -> dict[str, int]:
    """How many times each tool was called over these steps."""
    counts: dict[str, int] = {}
    for step in steps:
        if step.tool:
            counts[step.tool] = counts.get(step.tool, 0) + 1
    return counts


def read_events(path: Path) -> list[dict[str, Any]]:
    """A session log's events, skipping lines a run in flight left half-written.

    Streamed rather than `read_text().splitlines()`: a session log runs to a
    couple of megabytes, and materializing the whole string and then a list of
    every line triples the transient footprint of a file that is being read
    several at a time.
    """
    events = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def session_paths(home: Path) -> Iterator[Path]:
    """Every session log under a Harness home, in whatever order the glob gives.

    The directory a session sits in is the snapshot path with separators
    replaced, which is where `case_of` reads the corpus case from. Counting them,
    or taking the newest mtime, does not need them ordered.
    """
    yield from (home / "sessions").glob(f"*/*/{SESSION_FILE}")


def find_sessions(home: Path) -> Iterator[Path]:
    """Every session log under a Harness home, newest first."""
    yield from sorted(session_paths(home), key=lambda p: p.stat().st_mtime, reverse=True)


def case_of(path: Path) -> str:
    """The corpus case a session ran on, from the mangled snapshot directory.

    The harness names the directory after the snapshot path with every separator
    replaced by `-`, so `/home/.../cases/batch-01KQ...` arrives as
    `--home-...-cases-batch-01KQ...--`. The case is the trailing segment, which
    is `batch-<id>` — two `-`-separated parts, so this cannot split on `-`.
    """
    name = path.parent.parent.name.strip("-")
    marker = "-batch-"
    if marker in name:
        return name[name.rindex(marker) + 1 :]
    return name


def case_dir_for(case: str, root: Path) -> Path | None:
    """The snapshot a case ran on under `root`, or None when it is not here.

    `datapacks/ops-lite` keeps its snapshots in a `cases/` child beside the
    manifest, so a root may name either level — the same tolerance
    `agent.resolve_data_dir` gives the rollout, without importing the trainer.
    """
    for candidate in (root / case, root / "cases" / case):
        if candidate.is_dir():
            return candidate
    return None


@cache
def _truth(case_dir: str) -> Any:
    """The annotation for one case, read and validated once.

    A group of K siblings shares a case, so the uncached path validates the same
    `Scenario` K times: 360 reads for the 45 annotations of a K=8 sweep.
    """
    return truth_for_case(Path(case_dir))


def load_episode(path: Path, *, case_dir: Path | None = None) -> Episode:
    """One session as an `Episode`, scored when a case directory is at hand.

    `case_dir` is where the ground truth lives. It is optional because a list
    view wants the shape of 360 sessions without validating 360 annotations,
    and because a session whose snapshot has since moved is still worth reading.
    """
    events = read_events(path)
    finish, error = read_finish(events)
    episode = Episode(
        session_id=path.parent.name,
        case=case_of(path),
        finish=finish,
        error=error,
        steps=read_steps(events),
        compactions=read_compactions(events),
    )
    episode.submission = submitted_result(events)
    if case_dir is not None:
        # Also when nothing was submitted: `answer_of(None, truth)` claims
        # nothing, so it scores zero and still carries the truth. That is what
        # makes "found none of these six nodes" reviewable, rather than
        # indistinguishable from a case with nothing to find.
        episode.answer = answer_of(episode.submission, _truth(str(case_dir)))
        # Flat weights: the difficulty weighting of the method spec is defined
        # against a sibling group, and one episode is not a group.
        episode.score = weighted_score(episode.answer, {})
    return episode


def read_steps(events: Sequence[dict[str, Any]]) -> list[Step]:
    """One `Step` per `step/start`, filled from the events that follow it.

    Keyed by the step number the harness writes rather than by arrival order: a
    compaction pass and a retry both emit events between a step's start and its
    assistant message.
    """
    steps: dict[int, Step] = {}

    def at(data: dict[str, Any]) -> Step | None:
        index = data.get("step")
        if not isinstance(index, int):
            return None
        return steps.setdefault(index, Step(index=index))

    results = tool_results(events)
    for event in events:
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        kind = event.get("type")
        if kind == "step/start":
            at(data)
        elif kind == "assistant/message":
            step = at(data)
            if step is None:
                continue
            for block in (data.get("message") or {}).get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "reasoning":
                    step.reasoning += str(block.get("text") or "")
                elif block.get("type") == "text":
                    step.text += str(block.get("text") or "")
            step.input_tokens = int((data.get("usage") or {}).get("inputTokens") or 0)
        elif kind == "tool/call":
            step = at(data)
            if step is None:
                continue
            step.tool = str(data.get("name") or "")
            step.arguments = str(data.get("arguments") or "")
            result = results.get(str(data.get("callId")))
            if result is not None:
                step.result_head, step.result_error = _result_text(result)

    return [steps[i] for i in sorted(steps)]


def _result_text(event: dict[str, Any]) -> tuple[str, bool]:
    """A tool result's leading text and whether the tool refused the call."""
    block = event["data"]["message"]["content"][0]
    error = bool(block.get("isError", False))
    parts = [
        str(inner.get("text") or "")
        for inner in block.get("content") or []
        if isinstance(inner, dict) and inner.get("type") == "text"
    ]
    return "".join(parts)[:RESULT_HEAD], error


def read_compactions(events: Sequence[dict[str, Any]]) -> list[Compaction]:
    """One `Compaction` per pass, pairing each start with its summary and end.

    A pass writes `compaction/start`, then `compaction/summary` only if the
    model produced a checkpoint, then `compaction/end` carrying the error if it
    did not. They are paired by order rather than by `compactionId`, because a
    pass that fails before summarizing writes no id to pair on.
    """
    passes: list[Compaction] = []
    last_step = 0
    for event in events:
        data = event.get("data")
        if not isinstance(data, dict):
            data = {}
        kind = event.get("type")
        if isinstance(data.get("step"), int):
            last_step = int(data["step"])
        if kind == "compaction/start":
            passes.append(Compaction(at_step=last_step))
        elif passes and kind == "compaction/summary":
            passes[-1].shadowed_tokens = int(data.get("shadowedTokenCount") or 0)
            passes[-1].summary = _summary_text(data)
        elif passes and kind == "compaction/end" and data.get("error"):
            passes[-1].error = str(data["error"])
    return passes


def _summary_text(data: dict[str, Any]) -> str:
    return "".join(
        str(block.get("text") or "")
        for block in data.get("summary") or []
        if isinstance(block, dict) and block.get("type") == "text"
    )


def read_finish(events: Sequence[dict[str, Any]]) -> tuple[str, str]:
    """How the episode ended and why, from the last `turn/end` carrying a reason.

    The reason is a dict on some harness versions and its repr on others, so the
    kind is read structurally where possible and by substring where not.
    """
    for event in reversed(events):
        if event.get("type") != "turn/end":
            continue
        reason = (event.get("data") or {}).get("reason")
        if reason is None:
            continue
        if isinstance(reason, dict):
            failure = reason.get("error")
            message = str(failure.get("message") or "") if isinstance(failure, dict) else ""
            return str(reason.get("kind") or ""), message
        text = str(reason)
        if "'error'" in text:
            return "error", text
        return text[:60], ""
    return "", ""


def episode_metrics(events: Sequence[dict[str, Any]]) -> dict[str, float]:
    """What a rollout is worth logging, straight off the events it produced.

    The workflow calls this rather than loading an `Episode`, because it has no
    session file yet and needs no scoring — the reward it already computed is the
    outcome. Everything here is a fact about how the episode ran: how far it got,
    how close to the window it came, and whether compaction held.
    """
    steps = read_steps(events)
    passes = read_compactions(events)
    tools = tool_counts(steps)
    return {
        "steps": float(len(steps)),
        "sql_calls": float(tools.get("sql", 0)),
        "note_calls": float(tools.get("take_note", 0)),
        "peak_input_tokens": float(peak_input_tokens(steps)),
        "compactions": float(len(passes)),
        "compaction_failures": float(sum(1 for c in passes if c.error)),
        "completed": float(read_finish(events)[0] == "completed"),
    }


def group_stats(episodes: Sequence[Episode]) -> dict[str, Any]:
    """What a set of episodes says about whether RL has anything to optimise.

    Same quantities `scripts/pass_at_k.py` reports, over episodes read off disk
    rather than run live: pass@1 is a case's mean score, pass@k its best, and
    the fraction of cases whose samples disagree at all is the share of an RLOO
    batch that can carry a gradient (method spec §3.2).
    """
    by_case: dict[str, list[Episode]] = {}
    for episode in episodes:
        by_case.setdefault(episode.case, []).append(episode)
    if not by_case:
        return {"cases": 0, "episodes": 0}

    cases = [_case_stats(case, group) for case, group in sorted(by_case.items())]
    total = len(episodes)
    varied = [c for c in cases if c.spread > 0]
    return {
        "cases": len(cases),
        "episodes": total,
        "submission_rate": sum(1 for e in episodes if e.submission is not None) / total,
        "pass_1": sum(c.pass_1 for c in cases) / len(cases),
        "pass_k": sum(c.pass_k for c in cases) / len(cases),
        "varied_fraction": len(varied) / len(cases),
        "per_case": [asdict(c) for c in sorted(cases, key=lambda c: -c.spread)],
    }


@dataclass
class CaseStats:
    """One case's K samples, reduced to what decides whether it can teach."""

    case: str
    k: int
    pass_1: float
    pass_k: float
    spread: float
    submitted: int


def _case_stats(case: str, group: list[Episode]) -> CaseStats:
    scores = [e.score for e in group]
    return CaseStats(
        case=case,
        k=len(group),
        pass_1=sum(scores) / len(scores),
        pass_k=max(scores),
        spread=max(scores) - min(scores),
        submitted=sum(1 for e in group if e.submission is not None),
    )


__all__ = [
    "Compaction",
    "Episode",
    "Step",
    "case_dir_for",
    "case_of",
    "episode_metrics",
    "find_sessions",
    "group_stats",
    "load_episode",
    "peak_input_tokens",
    "read_compactions",
    "read_events",
    "read_finish",
    "read_steps",
    "session_paths",
    "tool_counts",
]
