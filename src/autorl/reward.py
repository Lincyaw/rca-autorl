"""What an episode earned, and what each of its turns is therefore worth.

Every turn of an episode carries the same value: the episode's outcome, scored
by `autorl.difficulty` against the siblings that answered the same case. There
is no turn-level credit, and that is a decision rather than an omission.

The candidate was a per-block hit rate — the share of a block's queries that
filtered on an entity in the true graph, blocks being the runs of queries the
`take_note` policy already separates. It correlated 0.35 with the final score
across the fifty collected episodes, which sounds usable until it is decomposed:
0.41 for entities the incident text already names, 0.24 for root causes, 0.14
for the middle of the chain. Its strongest component was querying the service
the model was handed. Within a chaos family the correlation ran 0.07 to 0.67
over six to fifteen episodes, which is noise at that sample size. Nothing here
establishes that the signal is real, and a term that cannot be explained after a
training run makes the run unexplainable in both directions. See
`.doc/designs/rl-reward.md`.

What this returns is what AReaL must *add* at each turn, not the value itself:
it accumulates backward (`reward[i] += reward[i+1] * discount`), so the
own-reward is the difference between neighbouring values — zero everywhere but
the last turn when the discount is 1, and not zero when it is not.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fpg import compare_model_to_ground_truth

from autorl.fpg import parse_submission, schema

SUBMIT_TOOL = "submit_result"


@dataclass
class Episode:
    """An episode's reward and the parts it was built from, for logging."""

    outcome: float
    shaped: dict[str, float] = field(default_factory=dict)
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
    turn_discount: float = 1.0,
) -> Episode:
    """Per-completion rewards for one episode.

    Every row the proxy cached gets the episode's outcome, the compaction
    summarizer included — `export_style: individual` trains on that row too, so
    leaving it out would not exclude it, it would let it accumulate whatever its
    neighbour carries.
    """
    outcome = outcome_score(submission, truth)
    episode = Episode(outcome=outcome)
    ordered = [
        c
        for c in sorted(completions, key=lambda c: int(c.get("ordinal", 0)))
        if c.get("responseId")
    ]
    if not ordered:
        return episode
    by_step = step_completions(events, completions)
    if not by_step:
        # Nothing to attribute to. The outcome still has to land somewhere, and
        # the last turn the *model* took is where the submission happened — not
        # simply the last completion, which may be the compaction summarizer.
        agent = [c for c in ordered if c.get("purpose") == "agent"] or ordered
        episode.unmapped = len(completions)
        episode.shaped = {str(agent[-1]["responseId"]): outcome}
        return episode

    values = [outcome] * len(ordered)
    own = [
        value - turn_discount * (values[i + 1] if i + 1 < len(values) else 0.0)
        for i, value in enumerate(values)
    ]
    episode.shaped = {str(c["responseId"]): own[i] for i, c in enumerate(ordered)}
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
    "Episode",
    "episode_reward",
    "load_truth",
    "outcome_score",
    "read_completions",
    "step_completions",
    "truth_for_case",
]
