"""SFT loss-mask correctness against a real Qwen3 Thinking tokenizer.

`autorl.data.sft` turns one exported conversation into one row per assistant
turn: the prompt is the conversation up to that turn with a generation prompt,
the supervised span is the turn itself. This pins the two properties the
trainer depends on:

* every assistant turn produces a row, and its supervised tokens decode to that
  turn only — its `<think>` block and its tool call — while everything before
  it, system and user and tool alike, stays unsupervised;
* a turn's reasoning is present in its own supervised span even when a `user`
  turn follows it later in the conversation. Rendered whole, the template drops
  reasoning from every turn before the conversation's last `user` message; per
  turn, the supervised turn is always last, so it never loses it.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
import unittest


def _ensure_stubs() -> None:
    if "datasets" not in sys.modules:
        datasets_module = types.ModuleType("datasets")
        datasets_module.__spec__ = importlib.machinery.ModuleSpec("datasets", loader=None)

        class _StubDataset:
            @classmethod
            def from_list(cls, rows):  # type: ignore[no-untyped-def]
                return rows

        datasets_module.Dataset = _StubDataset  # type: ignore[attr-defined]
        sys.modules["datasets"] = datasets_module


_ensure_stubs()


from autorl.data.sft import convert_sample  # noqa: E402

_QWEN3_MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"


def _load_qwen3_tokenizer():  # type: ignore[no-untyped-def]
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment guard
        raise unittest.SkipTest(f"transformers not installed: {exc}") from None
    try:
        return AutoTokenizer.from_pretrained(_QWEN3_MODEL_ID, local_files_only=True)
    except Exception as exc:  # pragma: no cover - optional cache
        raise unittest.SkipTest(f"cannot load {_QWEN3_MODEL_ID}: {exc}") from None


MESSAGES = [
    {"role": "system", "content": "you are an rca agent"},
    {"role": "user", "content": "endpoint /login is failing"},
    {
        "role": "assistant",
        "content": "<think>look at the tables first</think>\n\n",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "sql", "arguments": '{"statement":"SHOW TABLES"}'},
            }
        ],
    },
    {"role": "tool", "content": "abnormal_logs\nnormal_logs"},
    # A harness turn mid-trajectory: this is what strips reasoning from every
    # earlier assistant turn when the conversation is rendered in one pass.
    {"role": "user", "content": "Note reminder: 10 queries have run since your last note."},
    {
        "role": "assistant",
        "content": "<think>record the finding</think>\n\n",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "take_note", "arguments": '{"note":"db slow"}'},
            }
        ],
    },
    {"role": "tool", "content": "notebook: db slow"},
]

TOOLS = [
    {
        "name": "sql",
        "description": "Query the incident snapshot",
        "parameters": {"type": "object", "properties": {"statement": {"type": "string"}}},
    },
    {
        "name": "take_note",
        "description": "Record a finding",
        "parameters": {"type": "object", "properties": {"note": {"type": "string"}}},
    },
]


class Qwen3LossMaskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = _load_qwen3_tokenizer()
        self.rows = convert_sample({"messages": MESSAGES, "tools": TOOLS}, tokenizer=self.tokenizer)

    def _decode(self, row, bit):  # type: ignore[no-untyped-def]
        return self.tokenizer.decode(
            [tid for tid, m in zip(row["input_ids"], row["loss_mask"], strict=True) if m == bit]
        )

    def test_one_row_per_assistant_turn(self) -> None:
        self.assertEqual(len(self.rows), 2)
        for row in self.rows:
            self.assertEqual(len(row["input_ids"]), len(row["loss_mask"]))
            self.assertTrue(any(row["loss_mask"]), "no tokens supervised")
            self.assertFalse(all(row["loss_mask"]), "no tokens unsupervised")

    def test_turns_before_first_supervised_are_prompt_only(self) -> None:
        rows = convert_sample(
            {"messages": MESSAGES, "tools": TOOLS, "first_supervised": 3},
            tokenizer=self.tokenizer,
        )
        self.assertEqual(len(rows), 1)
        self.assertIn("record the finding", self._decode(rows[0], 1))
        self.assertIn("SHOW TABLES", self._decode(rows[0], 0))
        self.assertNotIn("SHOW TABLES", self._decode(rows[0], 1))

    def test_supervised_span_is_the_turn_alone(self) -> None:
        first, second = self.rows
        self.assertIn("look at the tables first", self._decode(first, 1))
        self.assertIn("SHOW TABLES", self._decode(first, 1))
        self.assertNotIn("record the finding", self._decode(first, 1))

        self.assertIn("record the finding", self._decode(second, 1))
        self.assertIn("take_note", self._decode(second, 1))
        self.assertNotIn("look at the tables first", self._decode(second, 1))

    def test_prompt_side_is_never_supervised(self) -> None:
        for row in self.rows:
            unsupervised = self._decode(row, 0)
            self.assertIn("you are an rca agent", unsupervised)
            self.assertIn("endpoint /login is failing", unsupervised)
        # The tool response and the harness reminder precede the second turn.
        second = self._decode(self.rows[1], 0)
        self.assertIn("abnormal_logs", second)
        self.assertIn("Note reminder", second)

    def test_reasoning_survives_a_later_user_turn(self) -> None:
        """The whole point of one row per turn: turn 1 keeps its own reasoning."""
        whole = self.tokenizer.apply_chat_template(
            MESSAGES, tools=TOOLS, tokenize=False, add_generation_prompt=False
        )
        self.assertNotIn("look at the tables first", whole)
        self.assertIn("look at the tables first", self._decode(self.rows[0], 1))


if __name__ == "__main__":
    unittest.main()
