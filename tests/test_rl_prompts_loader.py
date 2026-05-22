"""Tests for the rl_prompts loader."""

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

from autorl.data.rl_prompts import load_rl_prompts, to_chat_messages  # noqa: E402


_ROWS = [
    {
        "phase": "extractor",
        "sample_id": "case-1:firing-0",
        "source_case_id": "case-1",
        "firing_index": 0,
        "input": {"system": "you are extractor", "user": "extract from log A"},
        "meta": {},
    },
    {
        "phase": "extractor",
        "sample_id": "case-1:firing-1",
        "source_case_id": "case-1",
        "firing_index": 1,
        "input": {"system": "you are extractor", "user": "extract from log B"},
        "meta": {},
    },
    {
        "phase": "auditor",
        "sample_id": "case-2:firing-0",
        "source_case_id": "case-2",
        "firing_index": 0,
        "input": {"system": "you are auditor", "user": "audit graph X"},
        "meta": {},
    },
]


class RlPromptsLoaderTests(unittest.TestCase):
    def _write(self, rows):  # type: ignore[no-untyped-def]
        tmp = tempfile.NamedTemporaryFile(
            "w", delete=False, suffix=".jsonl", encoding="utf-8"
        )
        for r in rows:
            tmp.write(json.dumps(r) + "\n")
        tmp.close()
        return tmp.name

    def test_loads_all_rows_when_phase_unset(self) -> None:
        path = self._write(_ROWS)
        rows = load_rl_prompts(path)
        self.assertEqual(len(rows), 3)

    def test_filters_by_phase(self) -> None:
        path = self._write(_ROWS)
        extractor = load_rl_prompts(path, phase="extractor")
        self.assertEqual(len(extractor), 2)
        auditor = load_rl_prompts(path, phase="auditor")
        self.assertEqual(len(auditor), 1)
        self.assertEqual(auditor[0]["source_case_id"], "case-2")

    def test_to_chat_messages_returns_system_user_pair(self) -> None:
        msgs = to_chat_messages(_ROWS[0])
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "user")
        self.assertIn("extractor", msgs[0]["content"])
        self.assertIn("log A", msgs[1]["content"])

    def test_missing_keys_raise(self) -> None:
        bad = self._write([{"phase": "extractor", "sample_id": "x"}])
        with self.assertRaises(ValueError):
            load_rl_prompts(bad)


if __name__ == "__main__":
    unittest.main()
