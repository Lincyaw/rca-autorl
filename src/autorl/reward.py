"""What an episode earned, and which of its turns earned it.

Two signals, deliberately different in kind.

The **outcome** is `fpg.compare_model_to_ground_truth`: the submitted graph
against the annotation, scored predicate-agnostically on root subjects, all
subjects, and contracted edges. It is the only term that says whether the
episode was right.

The **process** is what makes that outcome attributable to steps rather than to
positions. An episode runs 50-200 tool calls and submits once; with a single
terminal number every turn is credited by where it sat, not by what it did. The
`take_note` policy already cuts the trajectory into blocks — a run of queries
followed by the finding the model commits to — and a block whose queries
interrogate entities that are actually on the true propagation path is a block
that moved. Measured across the first fifty collected episodes, that per-block
hit rate correlates 0.41 with the final score, where the per-query variants
(hops-to-root, how early a root is first queried) correlate 0.06 and -0.06.

Placement is what turns those numbers into credit. AReaL accumulates rewards
backward — `reward[i] += reward[i+1] * turn_discount` — so a reward left on the
turn that closed a block reaches every turn inside it and every turn before it,
and nothing after. Within a block that is uniform, which is the intent: the
queries of a block are one investigative move. Across blocks it is not, which is
also the intent. The positional component that survives is removed by the group
baseline, since every sample of a prompt shares it.

Compaction completions get nothing. The summarizer is a request the proxy caches
like any other, but no policy chose it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fpg import compare_model_to_ground_truth

from autorl.fpg import parse_submission, schema

SQL_TOOL = "sql"
NOTE_TOOL = "take_note"
SUBMIT_TOOL = "submit_result"

# Values a `WHERE` compares against. Entity names reach a query as literals, and
# reading the literals is what separates a query that interrogates a service
# from one that merely prints its name among two hundred rows.
LITERAL = re.compile(r"'([^']{2,120})'")


@dataclass(frozen=True)
class Block:
    """One investigative move: the queries since the last note, and their turns."""

    steps: tuple[int, ...]
    statements: tuple[str, ...]
    closing_step: int

    def hit_rate(self, entities: frozenset[str]) -> float:
        """Share of this block's queries that interrogate an entity on the true path."""
        if not self.statements:
            return 0.0
        hits = sum(1 for s in self.statements if any(lit in entities for lit in LITERAL.findall(s)))
        return hits / len(self.statements)


@dataclass
class Episode:
    """An episode's reward and the parts it was built from, for logging."""

    outcome: float
    shaped: dict[str, float] = field(default_factory=dict)
    blocks: int = 0
    progress: float = 0.0
    unmapped: int = 0


TRUTH_FILE = "causal_graph_verified.json"


def truth_for_case(case_dir: Path) -> Any:
    """The annotation beside the snapshot the episode read, bound to our vocabulary.

    Beside it, not looked up by name: the episode already resolved which
    directory it ran in, and asking the corpus again is a second chance to
    disagree with it.
    """
    path = Path(case_dir) / TRUTH_FILE
    if not path.is_file():
        raise FileNotFoundError(f"no {TRUTH_FILE} beside the snapshot: {case_dir}")
    return schema().Scenario.model_validate_json(path.read_text(encoding="utf-8"))


def load_truth(dataset_root: Path, datapack: str) -> Any:
    """The same, addressed by corpus root and case name."""
    for candidate in (dataset_root / datapack, dataset_root / "cases" / datapack):
        if (candidate / TRUTH_FILE).is_file():
            return truth_for_case(candidate)
    raise FileNotFoundError(f"no {TRUTH_FILE} for {datapack} under {dataset_root}")


def true_entities(truth: Any) -> frozenset[str]:
    """Every entity on the true propagation path, without its kind prefix.

    Without the prefix because a query filters on the name: `WHERE service_name =
    'ts-station-service'`, never on `svc:ts-station-service`.
    """
    return frozenset(node.subject.split(":", 1)[1] for node in truth.graph.nodes)


def blocks_of(events: Sequence[Mapping[str, Any]]) -> list[Block]:
    """Segment the trajectory at `take_note`, keeping each block's queries and turns."""
    found: list[Block] = []
    steps: list[int] = []
    statements: list[str] = []
    for event in events:
        if event.get("type") != "tool/call":
            continue
        data = event.get("data")
        if not isinstance(data, Mapping):
            continue
        step = data.get("step")
        if not isinstance(step, int):
            continue
        if data.get("name") == SQL_TOOL:
            steps.append(step)
            statements.append(str(_arguments(data).get("statement", "")))
        elif data.get("name") == NOTE_TOOL and statements:
            found.append(Block(tuple(steps), tuple(statements), step))
            steps, statements = [], []
    if statements:
        # A trailing run the model never noted still ends somewhere: the step it
        # stopped querying on is the turn that closed it.
        found.append(Block(tuple(steps), tuple(statements), steps[-1]))
    return found


def step_completions(
    events: Sequence[Mapping[str, Any]], completions: Sequence[Mapping[str, Any]]
) -> dict[int, str]:
    """Which completion id produced each agent step.

    The session log numbers steps and the sidecar numbers requests, and neither
    knows about the other. They align by order and by kind: the sidecar's
    `agent` records are the model's own turns in the order it took them, and the
    log's `assistant/message` events are those same turns. A mismatch in count
    means something inserted a request we cannot attribute, so the mapping is
    reported empty rather than silently shifted by one.
    """
    steps = [
        int(e["data"]["step"])
        for e in events
        if e.get("type") == "assistant/message" and isinstance(e.get("data"), Mapping)
    ]
    ids = [
        str(c["responseId"])
        for c in sorted(completions, key=lambda c: int(c.get("ordinal", 0)))
        if c.get("purpose") == "agent" and c.get("responseId")
    ]
    if len(steps) != len(ids):
        return {}
    return dict(zip(steps, ids, strict=True))


def read_completions(dsh_home: Path, session_id: str) -> list[dict[str, Any]]:
    """The sidecar the `rca-completions` row wrote for one session."""
    path = dsh_home / "rca-completions" / f"{session_id}.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def outcome_score(submission: Mapping[str, Any] | None, truth: Any) -> float:
    """The submitted graph against the annotation, or zero when nothing was submitted."""
    if submission is None:
        return 0.0
    return float(compare_model_to_ground_truth(parse_submission(submission), truth).score)


def episode_reward(
    *,
    events: Sequence[Mapping[str, Any]],
    completions: Sequence[Mapping[str, Any]],
    submission: Mapping[str, Any] | None,
    truth: Any,
    shaping: float = 0.2,
    turn_discount: float = 1.0,
) -> Episode:
    """Per-completion rewards for one episode, in the form the advantage reads.

    Each turn's value is

        r(turn) = outcome + shaping * (block rate - the episode's mean block rate)

    which is a redistribution, not a bonus. The shaping term is zero-mean over
    the episode's turns, so the mean of a trajectory's turn values is its
    outcome and no amount of querying can raise it. That matters: the model
    cannot see the true entity set, so the only way to raise a hit rate it does
    not understand is to name more services per query — `WHERE service_name IN
    ('a', ..., 'z')` scores as well as a considered filter. Addition would pay
    for that; centering cannot.

    The two halves stay readable downstream because they live on different
    axes: `autorl.advantage` takes a trajectory's mean as its outcome and each
    turn's deviation from that mean as its credit.

    What is returned is what AReaL must *add* at each turn, not the value
    itself. It accumulates backward (`reward[i] += reward[i+1] * discount`), so
    the own-reward is the difference between neighbouring values.
    """
    outcome = outcome_score(submission, truth)
    episode = Episode(outcome=outcome)
    by_step = step_completions(events, completions)
    if not by_step:
        last = next(
            (str(c["responseId"]) for c in reversed(list(completions)) if c.get("responseId")), ""
        )
        episode.unmapped = len(completions)
        episode.shaped = {last: outcome} if last else {}
        return episode

    entities = true_entities(truth)
    found = blocks_of(events)
    episode.blocks = len(found)
    rates = [block.hit_rate(entities) for block in found]
    episode.progress = sum(rates) / len(rates) if rates else 0.0

    # Every turn's value, in the order the episode took them. Centring on the
    # mean *per turn* rather than the mean per block is what makes the shaping
    # sum to zero: blocks hold different numbers of queries, so the average of
    # the rates is not the average a turn sees.
    steps = sorted(by_step)
    credit = dict.fromkeys(steps, 0.0)
    for block, rate in zip(found, rates, strict=True):
        for step in (*block.steps, block.closing_step):
            if step in credit:
                credit[step] = shaping * rate
    level = sum(credit.values()) / len(credit) if credit else 0.0
    values = [outcome + credit[step] - level for step in steps]

    # Backward accumulation turns own-rewards into values, so invert it.
    own = [
        value - turn_discount * (values[i + 1] if i + 1 < len(values) else 0.0)
        for i, value in enumerate(values)
    ]
    episode.shaped = {by_step[step]: own[i] for i, step in enumerate(steps)}
    return episode


def _arguments(data: Mapping[str, Any]) -> dict[str, Any]:
    raw = data.get("arguments")
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


__all__ = [
    "Block",
    "Episode",
    "blocks_of",
    "episode_reward",
    "load_truth",
    "outcome_score",
    "read_completions",
    "step_completions",
    "true_entities",
    "truth_for_case",
]
