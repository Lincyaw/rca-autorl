"""Turn DeepSeek Harness session logs into SFT rows.

`autorl.data.sft` consumes the row shape AgentM's `llmharness-distill export`
produced. Nothing produced it once the agent loop moved to `dsh`, so this module
is the replacement: it reads a session's append-only JSONL and emits the same
shape, which keeps the SFT tokenizer, the loss mask, and its tests unchanged.

One session becomes one row::

    {"phase": "rca", "sample_id": ..., "root_session_id": ..., "turn_index": 0,
     "input":  {"system": ..., "user": ...},
     "target": {"messages": [assistant, tool, assistant, tool, ...]},
     "meta":   {...}}

The exported trajectory is the session's **final surface**, not its raw log. A
compaction pass replaces events in place (`surfaceOp: {op: "replace"}`), and the
replacement is what the model actually had in context by the end; exporting the
superseded originals would supervise a student on context its teacher never saw.
Ranges the compactor shadowed away are dropped for the same reason.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SESSION_FILE = "session.jsonl"


def read_surface(session_path: Path) -> list[dict[str, Any]]:
    """Events in surface order after applying every append and replace."""
    surface: list[dict[str, Any]] = []
    by_seq: dict[int, int] = {}
    for line in session_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        op = event.get("surfaceOp")
        if op == "append":
            by_seq[int(event["seq"])] = len(surface)
            surface.append(event)
        elif isinstance(op, dict) and op.get("op") == "replace":
            start, end = int(op["start"]), int(op["end"])
            positions = sorted(index for seq, index in by_seq.items() if start <= seq <= end)
            if not positions:
                continue
            surface[positions[0]] = event
            for index in positions[1:]:
                surface[index] = {}
            by_seq = {
                seq: index
                for seq, index in by_seq.items()
                if not (start <= seq <= end) or index == positions[0]
            }
            by_seq[int(event["seq"])] = positions[0]
    return [event for event in surface if event]


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


def build_messages(surface: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The `[assistant, tool, ...]` sequence, paired by call id rather than by order.

    A chat template needs exactly one tool response per declared tool call, and
    log order does not guarantee that: a compaction pass can shadow a result
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

    assistants = [event for event in surface if event.get("type") == "assistant/message"]
    messages: list[dict[str, Any]] = []
    for index, event in enumerate(assistants):
        message = assistant_message(event)
        call_ids = [
            block["id"]
            for block in event["data"]["message"]["content"]
            if block.get("type") == "tool-call"
        ]
        answered = [call_id for call_id in call_ids if call_id in results]
        if len(answered) != len(call_ids) and index < len(assistants) - 1:
            break
        messages.append(message)
        messages.extend(tool_message(results[call_id]) for call_id in answered)
    return messages


def export_session(session_path: Path, sample_id: str | None = None) -> dict[str, Any] | None:
    """One SFT row for one session, or None when it carries no assistant turn."""
    surface = read_surface(session_path)
    header = next(
        (
            json.loads(line)["data"]["header"]
            for line in session_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line).get("type") == "request/header"
        ),
        None,
    )
    system = str(header.get("system", "")) if header else ""

    user = ""
    for event in surface:
        if (
            event.get("type") == "user/message"
            and event["data"].get("source", {}).get("kind") == "user"
        ):
            user = "".join(
                b.get("text", "") for b in event["data"]["content"] if b.get("type") == "text"
            )
            break

    messages = build_messages(surface)

    if not any(m["role"] == "assistant" for m in messages):
        return None
    session_id = json.loads(session_path.read_text(encoding="utf-8").splitlines()[0])["id"]
    return {
        "phase": "rca",
        "sample_id": sample_id or session_id,
        "root_session_id": session_id,
        "turn_index": 0,
        "input": {"system": system, "user": user},
        "target": {"messages": messages},
        "meta": {"source": str(session_path)},
    }


def export_home(dsh_home: Path) -> list[dict[str, Any]]:
    """Every session under a Harness home, oldest first."""
    rows = []
    for path in sorted(dsh_home.glob(f"sessions/*/*/{SESSION_FILE}")):
        row = export_session(path)
        if row is not None:
            rows.append(row)
    return rows


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        raise SystemExit("usage: python -m autorl.data.export <dsh-home> <out.jsonl>")
    rows = export_home(Path(argv[0]).expanduser().resolve())
    out = Path(argv[1]).expanduser()
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"exported {len(rows)} session(s) to {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
