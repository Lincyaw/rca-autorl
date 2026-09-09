"""Restore a trajectory at a step, so that a fresh episode continues from there.

Spec §4 forks at a turn boundary: siblings sampled from the same history and
the same snapshot. The harness has no session resume, so the state is rebuilt
instead. The model-visible part is the parent's surface up to that step, folded
the way the harness folds it (compaction included), and the rest of the episode
state is what the bundle keeps in memory: the notebook, and how many results
the last note has not covered. The snapshot itself is deterministic.

The prefix is written to a file the harness reads through `RCA_FORK_PREFIX`:
`agent/rca-harness/src/sampling.js` splices the messages after the child's own
incident prompt on every request, and `index.js` seeds the notebook and the
ledger from it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from autorl.data.export import fold_surface
from autorl.reward import accepted_call_ids

SURFACE = {"assistant/message", "tool/result", "user/message"}


def fork_prefix(
    events: Sequence[dict[str, Any]], step: int, base: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The parent's state at the start of `step`, or a ValueError if there is none.

    `base` is the prefix the parent itself was forked from: the walk starts
    from its history, notes and note debt.
    """
    cut = next(
        (
            i
            for i, e in enumerate(events)
            if e.get("type") == "step/start" and int(e["data"]["step"]) == step
        ),
        None,
    )
    if cut is None:
        raise ValueError(f"no step {step} in the parent episode")
    before = list(events[:cut])
    messages = [
        e["data"]["message"] if e["type"] != "user/message" else e["data"]
        for e in fold_surface(before)
        if e.get("type") in SURFACE
    ]
    # The first user message is the incident prompt; the child receives its own.
    prompt = next((i for i, m in enumerate(messages) if m.get("role") == "user"), None)
    if prompt is None:
        raise ValueError("the parent episode has no incident prompt before the fork")
    del messages[prompt]

    accepted = accepted_call_ids(before)
    notes: list[str] = list(base["notes"]) if base else []
    unnoted = int(base["unnoted"]) if base else 0
    for e in before:
        if e.get("type") != "tool/call" or e["data"].get("callId") not in accepted:
            continue
        if e["data"]["name"] == "take_note":
            notes.append(str(json.loads(e["data"]["arguments"]).get("content", "")).strip())
            unnoted = 0
        elif e["data"]["name"] == "sql":
            unnoted += 1
    if base is not None:
        messages = [*base["messages"], *messages]
    return {"step": step, "messages": messages, "notes": notes, "unnoted": unnoted}


def steps(events: Sequence[dict[str, Any]]) -> list[int]:
    """The step indices an episode ran, in order; fork points are drawn from these."""
    return [int(e["data"]["step"]) for e in events if e.get("type") == "step/start"]


__all__ = ["fork_prefix", "steps"]
