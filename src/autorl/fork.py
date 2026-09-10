"""Restore a trajectory at a step, so that a fresh episode continues from there.

Spec §4 forks at a turn boundary: siblings sampled from the same history and
the same snapshot. The harness has no session resume, so the state is rebuilt
instead. The model-visible part is the parent's surface up to that step, folded
the way the harness folds it (compaction included), and the rest of the episode
state is what the bundle keeps in memory: the notebook, and how many results
the last note has not covered. The snapshot itself is deterministic.

The prefix is written to a file the harness reads through `RCA_FORK_PREFIX`:
`agent/rca-harness/src/index.js` splices the messages in after the child's own
incident prompt on every request, and seeds the notebook and the ledger from
it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from autorl.data.export import fold_surface, is_incident
from autorl.reward import accepted_calls

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
    # The incident prompt is left out: the child receives its own. After a
    # compaction it is already gone from the surface.
    messages = [
        e["data"]["message"] if e["type"] != "user/message" else e["data"]
        for e in fold_surface(before)
        if e.get("type") in SURFACE and not (e["type"] == "user/message" and is_incident(e))
    ]

    notes: list[dict[str, str]] = list(base["notes"]) if base else []
    unnoted = int(base["unnoted"]) if base else 0
    for name, arguments in accepted_calls(before):
        # A `take_note` with no content is a read, not a write (`notebook.js`):
        # appending it would put a blank note in the child's notebook under a
        # name derived from nothing, and clearing the debt would hand the child a
        # free pass through `noteLimit`.
        if name == "take_note" and str(arguments.get("content", "")).strip():
            notes.append(
                {
                    "id": str(arguments.get("id", "")).strip(),
                    "content": str(arguments["content"]).strip(),
                }
            )
            unnoted = 0
        elif name == "sql":
            unnoted += 1
    if base is not None:
        messages = [*base["messages"], *messages]
    return {"step": step, "messages": messages, "notes": notes, "unnoted": unnoted}


def steps(events: Sequence[dict[str, Any]]) -> list[int]:
    """The step indices an episode ran, in order; fork points are drawn from these."""
    return [int(e["data"]["step"]) for e in events if e.get("type") == "step/start"]


__all__ = ["fork_prefix", "steps"]
