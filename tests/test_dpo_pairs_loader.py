"""Tests for the dpo_pairs loader."""

from __future__ import annotations

import importlib.machinery
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


def _ensure_stubs() -> None:
    if "datasets" not in sys.modules:
        m = types.ModuleType("datasets")
        m.__spec__ = importlib.machinery.ModuleSpec("datasets", loader=None)

        class _StubDataset:
            @classmethod
            def from_list(cls, rows):  # type: ignore[no-untyped-def]
                return rows

        m.Dataset = _StubDataset  # type: ignore[attr-defined]
        sys.modules["datasets"] = m


_ensure_stubs()
src_path = Path(__file__).resolve().parents[1] / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from autorl.data.dpo_pairs import load_dpo_pairs, pair_to_dpo_inputs  # noqa: E402


class _StubTokenizer:
    """Char-level tokenizer that supports apply_chat_template.

    Each token = one character of the concatenated message contents
    plus a fixed-length per-message ``<role>`` header. Deterministic
    and prefix-stable — sufficient to test pair_to_dpo_inputs.
    """

    def apply_chat_template(
        self,
        messages,  # type: ignore[no-untyped-def]
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ):
        text = ""
        for m in messages:
            text += f"<{m['role']}>" + str(m.get("content", ""))
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(c) for c in text]


_ROWS = [
    {
        "phase": "extractor",
        "pair_id": "case-1:0:0vs1",
        "source_case_id": "case-1",
        "firing_index": 0,
        "prompt": {"system": "sys", "user": "u"},
        "chosen": {"messages": [{"role": "assistant", "content": "GOOD"}]},
        "rejected": {"messages": [{"role": "assistant", "content": "X"}]},
        "chosen_score": 0.9,
        "rejected_score": 0.1,
        "meta": {},
    },
    {
        "phase": "auditor",
        "pair_id": "case-2:0:0vs1",
        "source_case_id": "case-2",
        "firing_index": 0,
        "prompt": {"system": "sys", "user": "u"},
        "chosen": {"messages": [{"role": "assistant", "content": "AAA"}]},
        "rejected": {"messages": [{"role": "assistant", "content": "B"}]},
        "chosen_score": 1.0,
        "rejected_score": 0.0,
        "meta": {},
    },
]


class DpoPairsLoaderTests(unittest.TestCase):
    def _write(self, rows):  # type: ignore[no-untyped-def]
        tmp = tempfile.NamedTemporaryFile(
            "w", delete=False, suffix=".jsonl", encoding="utf-8"
        )
        for r in rows:
            tmp.write(json.dumps(r) + "\n")
        tmp.close()
        return tmp.name

    def test_loads_and_filters(self) -> None:
        path = self._write(_ROWS)
        self.assertEqual(len(load_dpo_pairs(path)), 2)
        self.assertEqual(len(load_dpo_pairs(path, phase="extractor")), 1)
        self.assertEqual(len(load_dpo_pairs(path, phase="auditor")), 1)

    def test_pair_to_dpo_inputs_lengths(self) -> None:
        out = pair_to_dpo_inputs(_ROWS[0], _StubTokenizer())
        self.assertGreater(
            len(out["chosen_input_ids"]), len(out["prompt_input_ids"])
        )
        self.assertGreater(
            len(out["rejected_input_ids"]), len(out["prompt_input_ids"])
        )
        # Chosen "GOOD" is longer than rejected "X" so chosen ids span larger
        self.assertGreater(
            len(out["chosen_input_ids"]), len(out["rejected_input_ids"])
        )
        self.assertEqual(out["chosen_score"], 0.9)
        self.assertEqual(out["rejected_score"], 0.1)

    def test_missing_keys_raise(self) -> None:
        bad = self._write([{"phase": "extractor", "pair_id": "x"}])
        with self.assertRaises(ValueError):
            load_dpo_pairs(bad)


if __name__ == "__main__":
    unittest.main()
