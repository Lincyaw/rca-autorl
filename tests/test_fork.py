"""A fork prefix is the parent's state at a step: its surface, its notes, its debt."""

from __future__ import annotations

import json
import unittest
from typing import Any

from autorl.fork import fork_prefix, steps

SEQ = 0


def ev(kind: str, data: dict[str, Any], surface: Any = "append") -> dict[str, Any]:
    global SEQ
    SEQ += 1
    event: dict[str, Any] = {"seq": SEQ, "type": kind, "data": data}
    if surface is not None:
        event["surfaceOp"] = surface
    return event


def call(step: int, call_id: str, name: str, **args: Any) -> list[dict[str, Any]]:
    return [
        ev(
            "tool/call",
            {"step": step, "callId": call_id, "name": name, "arguments": json.dumps(args)},
            None,
        ),
        ev(
            "tool/result",
            {
                "step": step,
                "message": {
                    "role": "user",
                    "content": [{"type": "tool-result", "toolCallId": call_id, "isError": False}],
                },
            },
        ),
    ]


def assistant(step: int, text: str) -> dict[str, Any]:
    return ev(
        "assistant/message",
        {
            "step": step,
            "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
        },
    )


def episode() -> list[dict[str, Any]]:
    global SEQ
    SEQ = 0
    events = [
        ev("step/start", {"step": 1}, None),
        ev(
            "user/message",
            {
                "role": "user",
                "content": [{"type": "text", "text": "incident"}],
                "source": {"kind": "user"},
            },
        ),
    ]
    events += [assistant(1, "a1"), *call(1, "c1", "sql", statement="SELECT 1")]
    events += [
        ev("step/start", {"step": 2}, None),
        assistant(2, "a2"),
        *call(2, "c2", "sql", statement="SELECT 2"),
    ]
    events += [
        ev("step/start", {"step": 3}, None),
        assistant(3, "a3"),
        *call(3, "c3", "take_note", content="geo is down"),
    ]
    events += [
        ev("step/start", {"step": 4}, None),
        assistant(4, "a4"),
        *call(4, "c4", "sql", statement="SELECT 4"),
    ]
    events += [ev("step/start", {"step": 5}, None)]
    return events


class ForkPrefixTest(unittest.TestCase):
    def test_the_prefix_is_the_surface_before_the_step_minus_the_prompt(self) -> None:
        prefix = fork_prefix(episode(), 3)
        self.assertEqual([m["role"] for m in prefix["messages"]], ["assistant", "user"] * 2)
        self.assertEqual(prefix["messages"][0]["content"][0]["text"], "a1")

    def test_notes_and_debt_follow_the_accepted_calls(self) -> None:
        self.assertEqual(fork_prefix(episode(), 3)["unnoted"], 2)
        at_four = fork_prefix(episode(), 4)
        self.assertEqual(at_four["notes"], ["geo is down"])
        self.assertEqual(at_four["unnoted"], 0)
        self.assertEqual(fork_prefix(episode(), 5)["unnoted"], 1)

    def test_a_notebook_read_is_neither_a_note_nor_a_payment(self) -> None:
        """`take_note` with no content reads the notebook (`notebook.js`).

        Counting it would put a blank note in the child's notebook, shifting
        every later note id away from the ids the spliced history cites, and
        clearing the debt would let the child past `noteLimit` for free.
        """
        events = episode()
        events += [
            ev("step/start", {"step": 6}, None),
            assistant(6, "a6"),
            *call(6, "c6", "take_note"),
        ]
        events += [ev("step/start", {"step": 7}, None)]
        at_seven = fork_prefix(events, 7)
        self.assertEqual(at_seven["notes"], ["geo is down"])
        # Step 4's query is still unpaid; the read did not settle it.
        self.assertEqual(at_seven["unnoted"], 1)

    def test_a_pruned_result_is_forked_as_the_model_saw_it(self) -> None:
        events = episode()
        pruned = next(e for e in events if e["type"] == "tool/result")
        replacement = ev(
            "tool/result",
            {
                "step": 1,
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool-result",
                            "toolCallId": "c1",
                            "isError": False,
                            "pruned": True,
                        }
                    ],
                },
            },
            {"start": pruned["seq"], "end": pruned["seq"]},
        )
        events.insert(events.index(pruned) + 1, replacement)
        prefix = fork_prefix(events, 2)
        self.assertTrue(prefix["messages"][1]["content"][0].get("pruned"))
        self.assertEqual(len(prefix["messages"]), 2)

    def test_an_unknown_step_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            fork_prefix(episode(), 9)
        self.assertEqual(steps(episode()), [1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
