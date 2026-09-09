"""A session exports one row per context the teacher worked on, and one per checkpoint."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from autorl.data.export import checkpoint_instruction, export_session, replay_surface

SEQ = 0


def ev(kind: str, data: dict[str, Any], surface: Any = "append") -> dict[str, Any]:
    global SEQ
    SEQ += 1
    event: dict[str, Any] = {"id": "case1-abc", "seq": SEQ, "type": kind, "data": data}
    if surface is not None:
        event["surfaceOp"] = surface
    return event


def user(text: str, source: dict[str, Any], surface: Any = "append") -> dict[str, Any]:
    return ev(
        "user/message",
        {"role": "user", "content": [{"type": "text", "text": text}], "source": source},
        surface,
    )


def assistant(step: int, call_id: str) -> dict[str, Any]:
    content = [
        {"type": "reasoning", "text": f"think {step}"},
        {"type": "tool-call", "id": call_id, "name": "sql", "arguments": '{"statement": "1"}'},
    ]
    return ev(
        "assistant/message", {"step": step, "message": {"role": "assistant", "content": content}}
    )


def result(call_id: str, text: str, surface: Any = "append") -> dict[str, Any]:
    block = {
        "type": "tool-result",
        "toolCallId": call_id,
        "content": [{"type": "text", "text": text}],
    }
    return ev("tool/result", {"message": {"role": "user", "content": [block]}}, surface)


def episode() -> list[dict[str, Any]]:
    """Two steps, a checkpoint over the first, a third step, a prune, a fourth step."""
    global SEQ
    SEQ = 0
    events = [
        ev("session", {}, None),
        ev("request/header", {"header": {"system": "SYS", "tools": [{"name": "sql"}]}}, None),
        user("incident", {"kind": "user"}),
    ]
    incident, a1, r1 = events[-1], assistant(1, "c1"), result("c1", "rows 1")
    a2, r2 = assistant(2, "c2"), result("c2", "rows 2")
    events += [a1, r1, a2, r2]
    raw = [{"type": "reasoning", "text": "condense"}, {"type": "text", "text": "## Incident\n- x"}]
    events.append(ev("compaction/summary", {"compactionId": "k1", "rawOutput": raw}, None))
    events.append(
        user(
            "<compacted-summary>...",
            {"kind": "plugin", "plugin": "compact", "compactionId": "k1"},
            {"op": "replace", "start": incident["seq"], "end": r1["seq"]},
        )
    )
    a3, r3 = assistant(3, "c3"), result("c3", "rows 3")
    events += [a3, r3]
    events.append(
        result("c2", "rows 2 (pruned)", {"op": "replace", "start": r2["seq"], "end": r2["seq"]})
    )
    a4, r4 = assistant(4, "c4"), result("c4", "rows 4")
    events += [a4, r4]
    return events


def roles(row: dict[str, Any]) -> list[str]:
    return [m["role"] for m in row["messages"]]


def supervised(row: dict[str, Any]) -> list[str]:
    return [
        m["content"] for m in row["messages"][row["first_supervised"] :] if m["role"] == "assistant"
    ]


class ReplayTest(unittest.TestCase):
    def test_every_replace_starts_an_epoch(self) -> None:
        epochs = replay_surface(episode())
        self.assertEqual(len(epochs), 3)
        self.assertEqual([len(e.surface) for e in epochs], [5, 5, 7])
        self.assertEqual([e.first_own for e in epochs], [0, 3, 5])
        self.assertEqual([len(e.shadowed) for e in epochs], [3, 1, 0])
        self.assertEqual(epochs[0].summary["data"]["compactionId"], "k1")
        self.assertIsNone(epochs[1].summary)  # a prune writes no checkpoint

    def test_prune_shows_on_the_later_surface(self) -> None:
        last = replay_surface(episode())[-1].surface
        texts = [
            e["data"]["message"]["content"][0]["content"][0]["text"]
            for e in last
            if e["type"] == "tool/result"
        ]
        self.assertEqual(texts, ["rows 2 (pruned)", "rows 3", "rows 4"])


class ExportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.home)
        path = self.home / "case1-abc" / "session.jsonl"
        path.parent.mkdir()
        path.write_text("".join(json.dumps(e) + "\n" for e in episode()), encoding="utf-8")
        with mock.patch("autorl.data.export.checkpoint_instruction", return_value="INSTRUCTION"):
            self.rows = export_session(path)

    def test_one_row_per_epoch_and_per_checkpoint(self) -> None:
        self.assertEqual([r["phase"] for r in self.rows], ["rca", "checkpoint", "rca", "rca"])
        self.assertTrue(
            all(r["messages"][0] == {"role": "system", "content": "SYS"} for r in self.rows)
        )
        self.assertTrue(all(r["tools"] == [{"name": "sql"}] for r in self.rows))

    def test_each_turn_is_supervised_once_on_its_own_context(self) -> None:
        before, _, after, pruned = self.rows
        self.assertEqual(
            roles(before), ["system", "user", "assistant", "tool", "assistant", "tool"]
        )
        self.assertEqual(before["messages"][1]["content"], "incident")
        self.assertEqual(
            supervised(before), ["<think>think 1</think>\n\n", "<think>think 2</think>\n\n"]
        )
        # After the checkpoint the incident is gone; turn 2 is context, turn 3 is supervised.
        self.assertEqual(roles(after), ["system", "user", "assistant", "tool", "assistant", "tool"])
        self.assertEqual(after["messages"][1]["content"], "<compacted-summary>...")
        self.assertEqual(supervised(after), ["<think>think 3</think>\n\n"])
        # After the prune, turn 4 sees the pruned result and is the only new turn.
        self.assertEqual(pruned["messages"][3]["content"], "rows 2 (pruned)")
        self.assertEqual(supervised(pruned), ["<think>think 4</think>\n\n"])

    def test_checkpoint_row_is_the_summarizer_call(self) -> None:
        row = self.rows[1]
        self.assertEqual(roles(row), ["system", "user", "assistant", "tool", "user", "assistant"])
        self.assertEqual(row["messages"][1]["content"], "incident")
        self.assertEqual(row["messages"][3]["content"], "rows 1")  # the shadowed span only
        self.assertEqual(row["messages"][4]["content"], "INSTRUCTION")
        self.assertEqual(
            row["messages"][5],
            {"role": "assistant", "content": "<think>condense</think>\n\n## Incident\n- x"},
        )
        self.assertEqual(row["first_supervised"], 5)
        self.assertEqual(row["meta"]["compaction"], "k1")

    def test_checkpoint_that_called_a_tool_is_not_exported(self) -> None:
        events = episode()
        summary = next(e for e in events if e["type"] == "compaction/summary")
        summary["data"]["rawOutput"].append(
            {"type": "tool-call", "id": "x", "name": "sql", "arguments": "{}"}
        )
        path = self.home / "case1-abc" / "session.jsonl"
        path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
        with mock.patch("autorl.data.export.checkpoint_instruction", return_value="INSTRUCTION"):
            rows = export_session(path)
        self.assertEqual([r["phase"] for r in rows], ["rca", "rca", "rca"])


class InstructionTest(unittest.TestCase):
    def test_reads_the_bundle_string(self) -> None:
        if shutil.which("node") is None:
            self.skipTest("node is not installed")
        self.assertTrue(
            checkpoint_instruction().startswith("You are now acting as a compaction engine")
        )


if __name__ == "__main__":
    unittest.main()
