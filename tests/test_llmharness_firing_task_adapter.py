"""Tests for the llmharness firing TaskAdapters (new stripped-ReplayRecord shape)."""

from __future__ import annotations

import importlib.machinery
import sys
import types
from pathlib import Path
import unittest


def _ensure_areal_stub() -> None:
    if "areal.utils.dynamic_import" in sys.modules:
        return
    areal_module = sys.modules.setdefault("areal", types.ModuleType("areal"))
    utils_module = sys.modules.setdefault(
        "areal.utils", types.ModuleType("areal.utils")
    )
    dynamic_import_module = types.ModuleType("areal.utils.dynamic_import")
    hf_utils_module = types.ModuleType("areal.utils.hf_utils")
    dynamic_import_module.import_from_string = lambda dotted_path: dotted_path
    hf_utils_module.load_hf_tokenizer = lambda name: name
    sys.modules["areal.utils.dynamic_import"] = dynamic_import_module
    sys.modules["areal.utils.hf_utils"] = hf_utils_module
    areal_module.utils = utils_module
    utils_module.dynamic_import = dynamic_import_module
    utils_module.hf_utils = hf_utils_module


def _ensure_misc_stubs() -> None:
    for name in ("json5", "openai", "transformers"):
        if name not in sys.modules:
            try:
                __import__(name)
            except ImportError:
                m = types.ModuleType(name)
                m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
                sys.modules[name] = m
    if "openai" in sys.modules and not hasattr(sys.modules["openai"], "AsyncOpenAI"):
        class _StubAsyncOpenAI:
            def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                pass

        sys.modules["openai"].AsyncOpenAI = _StubAsyncOpenAI  # type: ignore[attr-defined]


def _ensure_stubs() -> None:
    _ensure_areal_stub()
    _ensure_misc_stubs()
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


from autorl.tasks.llmharness_firing import (  # noqa: E402
    LlmharnessAuditorFiringTaskAdapter,
    LlmharnessExtractorFiringTaskAdapter,
)


# A stripped-ReplayRecord-shaped row (no output/status/error/latency_ms/
# raw_assistant_messages — those are the teacher-output fields the
# distill CLI strips).
_ROW: dict = {
    "phase": "extractor",
    "sample_id": "case-1:firing-0",
    "source_case_id": "case-1",
    "firing_index": 0,
    "root_session_id": "sess-abc",
    "turn_index": 0,
    "ts_ns": 1_700_000_000_000_000_000,
    "compose_kwargs": {
        "base_prompt": "you are an extractor",
        "cards_tools_config": None,
        "observability_config": None,
    },
    "payload": {"recent_graph": [], "next_event_id": 1, "user": "extract from log"},
    "provider": None,
    "meta": {"datapack_name": "ts0-mysql-corrupt-kwx8n5"},
}


class FiringAdapterTests(unittest.TestCase):
    def test_extractor_adapter_validate(self) -> None:
        adapter = LlmharnessExtractorFiringTaskAdapter()
        sample = adapter.validate_sample(_ROW)
        self.assertEqual(sample.sample_id, "case-1:firing-0")
        self.assertEqual(sample.task_type, "llmharness_extractor_firing")
        self.assertEqual(sample.metadata.get("source_case_id"), "case-1")
        self.assertEqual(sample.metadata.get("firing_index"), 0)
        self.assertEqual(sample.metadata.get("phase"), "extractor")
        self.assertEqual(sample.metadata.get("root_session_id"), "sess-abc")
        self.assertEqual(
            sample.metadata.get("datapack_name"), "ts0-mysql-corrupt-kwx8n5"
        )

    def test_extractor_adapter_to_agent_input_passes_full_row(self) -> None:
        adapter = LlmharnessExtractorFiringTaskAdapter()
        sample = adapter.validate_sample(_ROW)
        agent_input = adapter.to_agent_input(sample)
        # The runtime rehydrates the full ReplayRecord — so raw_sample
        # must round-trip every key the row carried.
        for key in ("phase", "payload", "compose_kwargs", "root_session_id"):
            self.assertEqual(agent_input.raw_sample[key], _ROW[key])
        self.assertEqual(agent_input.context.get("phase"), "extractor")

    def test_auditor_adapter_task_type(self) -> None:
        row = {**_ROW, "phase": "auditor"}
        adapter = LlmharnessAuditorFiringTaskAdapter()
        sample = adapter.validate_sample(row)
        self.assertEqual(sample.task_type, "llmharness_auditor_firing")

    def test_extractor_adapter_rejects_wrong_phase(self) -> None:
        adapter = LlmharnessExtractorFiringTaskAdapter()
        with self.assertRaises(ValueError):
            adapter.validate_sample({**_ROW, "phase": "auditor"})

    def test_missing_phase_raises(self) -> None:
        adapter = LlmharnessExtractorFiringTaskAdapter()
        with self.assertRaises(ValueError):
            adapter.validate_sample({"sample_id": "x", "payload": {}, "compose_kwargs": {}})

    def test_missing_payload_raises(self) -> None:
        adapter = LlmharnessExtractorFiringTaskAdapter()
        bad = {k: v for k, v in _ROW.items() if k != "payload"}
        with self.assertRaises(ValueError):
            adapter.validate_sample(bad)

    def test_sample_id_derived_when_absent(self) -> None:
        adapter = LlmharnessExtractorFiringTaskAdapter()
        bad = {k: v for k, v in _ROW.items() if k != "sample_id"}
        sample = adapter.validate_sample(bad)
        self.assertEqual(sample.sample_id, "sess-abc:turn-0")


if __name__ == "__main__":
    unittest.main()
