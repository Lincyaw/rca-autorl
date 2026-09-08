"""Turn DeepSeek Harness session logs into SFT rows.

One session becomes one row: the conversation the teacher actually held, as the
harness composed it, plus the tool schemas it was offered::

    {"phase": "rca", "sample_id": ..., "root_session_id": ...,
     "messages": [system, user, assistant, tool, user, assistant, ...],
     "tools": [...],
     "meta": {...}}

Nothing here decides what is supervised. `autorl.data.sft` renders these
messages through the tokenizer's own chat template, once per assistant turn and
exactly as the rollout would, and the mask falls out of that. Keeping the split
here would mean guessing at the template's rules — which turns keep their
reasoning, how a tool call renders once its response exists — and every such
guess is a place for training to drift from inference.

Two things the row is careful about:

*The conversation is the session's final surface, not its raw log.* A
compaction pass replaces events in place, and the replacement is what the model
had in context by the end; exporting the superseded originals would supervise a
student on context its teacher never saw. `fold_surface` is the harness's own
fold, transcribed.

*The incident comes from the raw log.* A compaction checkpoint shadows the
original question away, so on the surface it is simply gone. Read from the
surface alone, the row would ask the student to continue an investigation it
was never given.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SESSION_FILE = "session.jsonl"


def fold_surface(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Events in surface order, folded the way the harness itself folds them.

    `surfaceOp.start` and `surfaceOp.end` are seqs that must both be *on the
    current surface*, and the replacement shadows the nodes between their
    **positions** — not the seqs numerically between them. The two readings
    agree only until the first replacement: a replacement splices its own,
    higher, seq in at the position of the range it shadowed, so from then on
    surface order and seq order differ. Read numerically, a later replace then
    shadows the wrong nodes — keeping events the model no longer had and
    dropping ones it did — which breaks call/result pairing and truncates the
    trajectory to nothing without saying so.
    """
    by_seq: dict[int, dict[str, Any]] = {}
    nodes: list[int] = []
    for event in events:
        op = event.get("surfaceOp")
        if op is None:
            continue
        seq = int(event["seq"])
        by_seq[seq] = event
        if op == "append":
            nodes.append(seq)
            continue
        start, end = int(op["start"]), int(op["end"])
        if start not in nodes or end not in nodes:
            raise ValueError(f"replace at seq {seq}: [{start}..{end}] is not on the surface")
        first, last = nodes.index(start), nodes.index(end)
        if first > last:
            raise ValueError(f"replace at seq {seq}: start {start} is after end {end}")
        nodes[first : last + 1] = [seq]
    return [by_seq[seq] for seq in nodes]


def assistant_message(event: dict[str, Any]) -> dict[str, Any]:
    """One assistant turn: reasoning as a `<think>` block, then text and tool calls."""
    blocks = event["data"]["message"]["content"]
    reasoning = "".join(b.get("text", "") for b in blocks if b.get("type") == "reasoning")
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    content = f"<think>{reasoning}</think>\n\n{text}" if reasoning else text
    calls = [
        {
            "type": "function",
            "function": {"name": b["name"], "arguments": b["arguments"]},
        }
        for b in blocks
        if b.get("type") == "tool-call"
    ]
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    return message


def tool_message(event: dict[str, Any]) -> dict[str, Any]:
    result = event["data"]["message"]["content"][0]
    text = "".join(b.get("text", "") for b in result.get("content", []) if b.get("type") == "text")
    return {"role": "tool", "content": text}


def user_message(event: dict[str, Any]) -> dict[str, Any]:
    text = "".join(b.get("text", "") for b in event["data"]["content"] if b.get("type") == "text")
    return {"role": "user", "content": text}


def is_incident(event: dict[str, Any]) -> bool:
    """The episode's own question, as opposed to a turn the harness injected."""
    return (event["data"].get("source") or {}).get("kind") == "user"


def build_messages(surface: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The surface as a chat sequence, tool results paired by call id.

    A chat template needs exactly one tool response per declared tool call, and
    surface order does not guarantee that: a compaction pass can shadow a result
    away, and an episode can end between a call and its result. Pairing by id
    makes the gap visible, and the trajectory is truncated at the first
    unanswered call rather than emitting a sequence the template cannot render.
    The final assistant turn keeps unanswered calls, which is a valid ending.
    """
    results: dict[str, dict[str, Any]] = {}
    for event in surface:
        if event.get("type") != "tool/result":
            continue
        block = event["data"]["message"]["content"][0]
        call_id = block.get("toolCallId")
        if isinstance(call_id, str):
            results[call_id] = event

    remaining = sum(1 for event in surface if event.get("type") == "assistant/message")
    messages: list[dict[str, Any]] = []
    for event in surface:
        kind = event.get("type")
        if kind == "user/message":
            messages.append(user_message(event))
            continue
        if kind != "assistant/message":
            continue
        remaining -= 1
        call_ids = [
            block["id"]
            for block in event["data"]["message"]["content"]
            if block.get("type") == "tool-call"
        ]
        answered = [call_id for call_id in call_ids if call_id in results]
        if len(answered) != len(call_ids) and remaining > 0:
            break
        messages.append(assistant_message(event))
        messages.extend(tool_message(results[call_id]) for call_id in answered)
    return messages


def _provenance(session_path: Path) -> str:
    """Which session this row came from: its directory and file, nothing above.

    These rows are a checked-in artifact. The absolute path names the machine
    that collected them, and the directory above the session is `dsh`'s slug of
    the episode's working directory, which names that machine's layout. The
    session id identifies the session on its own, and `sample_id` already
    carries which case it was.
    """
    return f"{session_path.parent.name}/{session_path.name}"


def export_session(session_path: Path, sample_id: str | None = None) -> dict[str, Any] | None:
    """One SFT row for one session, or None when it carries no assistant turn."""
    events = [
        json.loads(line)
        for line in session_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    headers = (e["data"]["header"] for e in events if e.get("type") == "request/header")
    header: dict[str, Any] = next(headers, {})
    incident = next((e for e in events if e.get("type") == "user/message" and is_incident(e)), None)

    messages: list[dict[str, Any]] = []
    system = str(header.get("system", ""))
    if system:
        messages.append({"role": "system", "content": system})
    if incident is not None:
        messages.append(user_message(incident))
    messages.extend(build_messages(fold_surface(events)))

    if not any(m["role"] == "assistant" for m in messages):
        return None
    return {
        "phase": "rca",
        "sample_id": sample_id or events[0]["id"],
        "root_session_id": events[0]["id"],
        "messages": messages,
        "tools": header.get("tools") or [],
        # Relative to the Harness home: an absolute path would carry the
        # machine that collected the trajectories into a checked-in artifact.
        "meta": {"source": _provenance(session_path)},
    }


def export_home(dsh_home: Path) -> tuple[list[dict[str, Any]], list[tuple[Path, str]]]:
    """Every session under a Harness home, oldest first, and the ones with no row.

    A session that exports to nothing is reported rather than skipped: it means
    the trajectory was lost, not that the episode was empty, and losing it
    silently hides exactly the long compacted episodes that are hardest to
    collect.
    """
    rows, skipped = [], []
    for path in sorted(dsh_home.glob(f"sessions/*/*/{SESSION_FILE}")):
        try:
            row = export_session(path)
        except ValueError as error:
            skipped.append((path, str(error)))
            continue
        if row is None:
            skipped.append((path, "no assistant turn survived pairing"))
        else:
            rows.append(row)
    return rows, skipped


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        raise SystemExit("usage: python -m autorl.data.export <dsh-home> <out.jsonl>")
    rows, skipped = export_home(Path(argv[0]).expanduser().resolve())
    out = Path(argv[1]).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"exported {len(rows)} session(s) to {out}")
    for path, reason in skipped:
        print(f"  no row from {path.parent.name}: {reason}")


if __name__ == "__main__":
    main(sys.argv[1:])
