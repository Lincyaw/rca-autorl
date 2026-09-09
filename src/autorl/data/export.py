"""Turn DeepSeek Harness session logs into SFT rows.

A session becomes one row per stretch of conversation the teacher held on one
context, plus one row per checkpoint it wrote::

    {"phase": "rca" | "checkpoint", "sample_id": ..., "root_session_id": ...,
     "messages": [system, user, assistant, tool, user, assistant, ...],
     "tools": [...], "first_supervised": k, "meta": {...}}

Nothing here decides what is supervised. `autorl.data.sft` renders these
messages through the tokenizer's own chat template, once per assistant turn and
exactly as the rollout would, and the mask falls out of that. Keeping the split
here would mean guessing at the template's rules — which turns keep their
reasoning, how a tool call renders once its response exists — and every such
guess is a place for training to drift from inference.

*A compaction pass splits the session into epochs.* It replaces a span of the
surface with a checkpoint, and from then on the model works on a different
context. Every turn is supervised on the surface it was generated on: the
turns before a pass on the surface as it stood when the pass began, the turns
after it on the surface with the checkpoint in place. Exporting only the final
surface, as this once did, kept one epoch in five and dropped the rest. A
pruned tool result is a replace as well, of one node by its shorter self, and
splits an epoch the same way; a run of replaces with no turn between them
yields epochs with nothing to supervise, which are not exported.

*A checkpoint is a turn the model wrote.* The harness writes it with one more
LLM call — the shadowed span, then the checkpoint instruction — and the same
model answers that call at rollout time. `compaction/summary` keeps the raw
output, so the call is exported as a row of its own: the span, the instruction,
and what the teacher wrote back.

*The tail of an epoch is context in the next one.* A pass leaves the surface's
most recent nodes in place, so they sit on both the surface they were written
on and the surface the checkpoint heads. `first_supervised` marks where an
epoch's own turns begin; before it, messages are the prompt only.

*A checkpoint shadows the incident.* After the first pass the question is
gone from the surface, and the model continues from the checkpoint's own
`Incident` section, at rollout as at export. Nothing is put back.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from autorl.data.samples import read_jsonl
from autorl.reward import tool_results

SESSION_FILE = "session.jsonl"
PROMPTS_JS = Path(__file__).resolve().parents[3] / "agent" / "rca-harness" / "src" / "prompts.js"


@dataclass
class Epoch:
    """The surface as it stood when a compaction pass began, or when the episode ended."""

    surface: list[dict[str, Any]]
    first_own: int  # position of the first node appended during this epoch
    shadowed: list[dict[str, Any]]  # the span the pass replaced; empty for the last epoch
    summary: dict[str, Any] | None  # the `compaction/summary` event of that pass


def replay_surface(events: list[dict[str, Any]]) -> list[Epoch]:
    """The surface at every compaction pass, folded the way the harness folds it.

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
    summaries = {
        e["data"]["compactionId"]: e for e in events if e.get("type") == "compaction/summary"
    }
    by_seq: dict[int, dict[str, Any]] = {}
    nodes: list[int] = []
    epochs: list[Epoch] = []
    first_own = 0
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
        surface = [by_seq[s] for s in nodes]
        compaction_id = (event["data"].get("source") or {}).get("compactionId")
        epochs.append(
            Epoch(surface, first_own, surface[first : last + 1], summaries.get(compaction_id))
        )
        nodes[first : last + 1] = [seq]
        first_own = len(nodes)
    epochs.append(Epoch([by_seq[s] for s in nodes], first_own, [], None))
    return epochs


def fold_surface(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Events in surface order at the end of the log."""
    return replay_surface(events)[-1].surface


def assistant_message(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """One assistant turn: reasoning as a `<think>` block, then text and tool calls."""
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


def build_messages(
    surface: list[dict[str, Any]], first_own: int = 0
) -> tuple[list[dict[str, Any]], int]:
    """The surface as a chat sequence, tool results paired by call id.

    Returns the messages and the index of the first one produced from a node
    at or past `first_own`, which is where supervision may start.

    A chat template needs exactly one tool response per declared tool call, and
    surface order does not guarantee that: a compaction pass can shadow a result
    away, and an episode can end between a call and its result. Pairing by id
    makes the gap visible, and the trajectory is truncated at the first
    unanswered call rather than emitting a sequence the template cannot render.
    The final assistant turn keeps unanswered calls, which is a valid ending.
    """
    results = tool_results(surface)
    remaining = sum(1 for event in surface if event.get("type") == "assistant/message")
    messages: list[dict[str, Any]] = []
    first_supervised: int | None = None
    for position, event in enumerate(surface):
        if position >= first_own and first_supervised is None:
            first_supervised = len(messages)
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
        messages.append(assistant_message(event["data"]["message"]["content"]))
        messages.extend(tool_message(results[call_id]) for call_id in answered)
    return messages, len(messages) if first_supervised is None else first_supervised


@cache
def checkpoint_instruction() -> str:
    """`RCA_INSTRUCTION` as the bundle exports it: the one string the summarizer appends."""
    script = (
        f"import('{PROMPTS_JS.as_uri()}').then(m => console.log(JSON.stringify(m.RCA_INSTRUCTION)))"
    )
    output = subprocess.run(
        ["node", "--input-type=module", "-e", script], capture_output=True, text=True, check=True
    ).stdout
    text = json.loads(output)
    if not isinstance(text, str) or not text:
        raise ValueError(f"RCA_INSTRUCTION is not a string in {PROMPTS_JS}")
    return text


def _provenance(session_path: Path) -> str:
    """Which session this row came from: its directory and file, nothing above.

    These rows are a checked-in artifact. The absolute path names the machine
    that collected them, and the directory above the session is `dsh`'s slug of
    the episode's working directory, which names that machine's layout. The
    session id identifies the session on its own, and `sample_id` already
    carries which case it was.
    """
    return f"{session_path.parent.name}/{session_path.name}"


def export_session(session_path: Path, sample_id: str | None = None) -> list[dict[str, Any]]:
    """The SFT rows of one session: one per epoch with a turn of its own, one per checkpoint."""
    events = read_jsonl(session_path)
    headers = (e["data"]["header"] for e in events if e.get("type") == "request/header")
    header: dict[str, Any] = next(headers, {})
    system = str(header.get("system", ""))
    prefix = [{"role": "system", "content": system}] if system else []

    def row(
        phase: str, messages: list[dict[str, Any]], first_supervised: int, **meta: Any
    ) -> dict[str, Any]:
        return {
            "phase": phase,
            "sample_id": sample_id or events[0]["id"],
            "root_session_id": events[0]["id"],
            "messages": prefix + messages,
            "tools": header.get("tools") or [],
            "first_supervised": len(prefix) + first_supervised,
            # Relative to the Harness home: an absolute path would carry the
            # machine that collected the trajectories into a checked-in artifact.
            "meta": {"source": _provenance(session_path), **meta},
        }

    rows: list[dict[str, Any]] = []
    for index, epoch in enumerate(replay_surface(events)):
        messages, first_supervised = build_messages(epoch.surface, epoch.first_own)
        if any(m["role"] == "assistant" for m in messages[first_supervised:]):
            rows.append(row("rca", messages, first_supervised, epoch=index))
        if epoch.summary is None or "rawOutput" not in epoch.summary["data"]:
            continue
        # The summarizer's call: the shadowed span, the instruction, the checkpoint.
        # The harness keeps only the text of the answer; a teacher that also
        # called a tool broke the instruction, and is not imitated.
        output = epoch.summary["data"]["rawOutput"]
        if any(block.get("type") == "tool-call" for block in output):
            continue
        span, _ = build_messages(epoch.shadowed)
        span.append({"role": "user", "content": checkpoint_instruction()})
        span.append(assistant_message(output))
        rows.append(
            row(
                "checkpoint",
                span,
                len(span) - 1,
                epoch=index,
                compaction=epoch.summary["data"]["compactionId"],
            ),
        )
    return rows


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
            session_rows = export_session(path)
        except ValueError as error:
            skipped.append((path, str(error)))
            continue
        if not session_rows:
            skipped.append((path, "no assistant turn survived pairing"))
        rows.extend(session_rows)
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
    sessions = len({row["root_session_id"] for row in rows})
    checkpoints = sum(1 for row in rows if row["phase"] == "checkpoint")
    print(
        f"exported {sessions} session(s) as {len(rows)} row(s) ({checkpoints} checkpoints) to {out}"
    )
    for path, reason in skipped:
        print(f"  no row from {path.parent.name}: {reason}")


if __name__ == "__main__":
    main(sys.argv[1:])
