"""What an episode earned, and what each of its turns is therefore worth.

The terminal reward is the graph score of `autorl.difficulty`, weighted by
sibling difficulty in `DshWorkflow.rescore_group` once the group is in. Every
turn of an episode carries the same value: there is no turn-level credit. The
candidate for one was measured and withdrawn; see the Notes log entry
`2026-09-08-rl-reward-and-credit-assignment`.

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

from autorl.fpg import schema

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


def read_completions(dsh_home: Path, session_id: str) -> list[dict[str, Any]]:
    """The sidecar the `rca-completions` row wrote for one session."""
    path = dsh_home / "rca-completions" / f"{session_id}.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def episode_reward(
    *,
    events: Sequence[Mapping[str, Any]],
    completions: Sequence[Mapping[str, Any]],
    outcome: float,
    turn_discount: float = 1.0,
) -> Episode:
    """Per-completion rewards for one episode.

    Every row the proxy cached gets the episode's outcome, the compaction
    summarizer included — `export_style: individual` trains on that row too, so
    leaving it out would not exclude it, it would let it accumulate whatever its
    neighbour carries.
    """
    episode = Episode(outcome=outcome)
    ordered = [
        c
        for c in sorted(completions, key=lambda c: int(c.get("ordinal", 0)))
        if c.get("responseId")
    ]
    if not ordered:
        return episode
    if not step_completions(events, completions):
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


__all__ = [
    "Episode",
    "accepted_call_ids",
    "episode_reward",
    "load_truth",
    "read_completions",
    "step_completions",
    "truth_for_case",
]
