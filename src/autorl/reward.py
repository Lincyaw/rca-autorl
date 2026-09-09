"""From an episode's session log to what it earned.

The terminal reward is the graph score of `autorl.difficulty`, weighted by
sibling difficulty in `DshWorkflow.rescore_group` once the group is in. Every
turn of an episode carries the same value: there is no turn-level credit. The
candidate for one was measured and withdrawn; see the Notes log entry
`2026-09-08-rl-reward-and-credit-assignment`.

This module reads the log: the ground truth beside the snapshot, the accepted
`submit_result` call, the `Answer` the two make together, and the id of the
completion behind each step, which is how a value reaches a row.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from autorl.difficulty import Answer
from autorl.fpg import parse_submission, schema

SUBMIT_TOOL = "submit_result"
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


def step_ids(events: Sequence[dict[str, Any]]) -> dict[int, str]:
    """The proxy's completion id behind each agent step, in step order.

    The `llm-pi-ai` adapter keeps the provider's response id in the message's
    replay state, and the loop logs that with the `assistant/message`; AReaL
    keys its interaction cache by the same id.
    """
    ids: dict[int, str] = {}
    for event in events:
        if event.get("type") != "assistant/message":
            continue
        source = event["data"]["message"].get("source") or {}
        response_id = ((source.get("replayState") or {}).get("response") or {}).get("responseId")
        if response_id:
            ids[int(event["data"]["step"])] = str(response_id)
    return ids


def tool_results(events: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Each `tool/result` event by the call id its first block answers."""
    results: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("type") != "tool/result":
            continue
        call_id = event["data"]["message"]["content"][0].get("toolCallId")
        if isinstance(call_id, str):
            results[call_id] = event
    return results


def accepted_calls(events: Sequence[dict[str, Any]]) -> Iterator[tuple[str, Any]]:
    """The name and parsed arguments of every call whose result is not an error."""
    accepted = {
        call_id
        for call_id, event in tool_results(events).items()
        if not event["data"]["message"]["content"][0].get("isError", False)
    }
    for event in events:
        data = event.get("data")
        if event.get("type") != "tool/call" or not isinstance(data, dict):
            continue
        if data.get("callId") in accepted and isinstance(data.get("arguments"), str):
            yield str(data["name"]), json.loads(data["arguments"])


def submitted_result(events: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
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
    for name, arguments in accepted_calls(events):
        if name == SUBMIT_TOOL and isinstance(arguments, dict):
            return arguments
    return None


def answer_of(submission: dict[str, Any] | None, truth: Any) -> Answer:
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


__all__ = [
    "accepted_calls",
    "answer_of",
    "step_ids",
    "submitted_result",
    "tool_results",
    "truth_for_case",
]
