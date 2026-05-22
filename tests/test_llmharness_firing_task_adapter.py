"""Tests for the llmharness firing TaskAdapters + stub runtimes."""

from __future__ import annotations

import asyncio
import importlib.machinery
import sys
import types
import unittest
from pathlib import Path


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

from autorl.contracts import RuntimeContext  # noqa: E402
from autorl.contracts.runtime import RuntimeLimits  # noqa: E402

# Load the runtime stub modules directly from file to avoid triggering
# ``autorl.runtime.__init__`` (which transitively imports heavy deps
# like qwen_agent / openai / sglang that aren't installed for unit
# tests). The stubs themselves only depend on autorl.contracts.
import importlib.util as _ilu  # noqa: E402

def _load_module(dotted: str, path: Path):  # type: ignore[no-untyped-def]
    spec = _ilu.spec_from_file_location(dotted, path)
    assert spec and spec.loader
    mod = _ilu.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


# Ensure parent package object exists without executing its __init__.
if "autorl.runtime" not in sys.modules:
    pkg = types.ModuleType("autorl.runtime")
    pkg.__path__ = [str(src_path / "autorl" / "runtime")]  # type: ignore[attr-defined]
    sys.modules["autorl.runtime"] = pkg

# Load base + stub runtime modules from file.
_load_module("autorl.runtime.base", src_path / "autorl" / "runtime" / "base.py")
_extractor_mod = _load_module(
    "autorl.runtime.llmharness_extractor",
    src_path / "autorl" / "runtime" / "llmharness_extractor.py",
)
_auditor_mod = _load_module(
    "autorl.runtime.llmharness_auditor",
    src_path / "autorl" / "runtime" / "llmharness_auditor.py",
)
LlmharnessExtractorRuntime = _extractor_mod.LlmharnessExtractorRuntime
LlmharnessAuditorRuntime = _auditor_mod.LlmharnessAuditorRuntime

from autorl.tasks.llmharness_firing import (  # noqa: E402
    LlmharnessAuditorFiringTaskAdapter,
    LlmharnessExtractorFiringTaskAdapter,
)


_ROW = {
    "phase": "extractor",
    "sample_id": "case-1:firing-0",
    "source_case_id": "case-1",
    "firing_index": 0,
    "input": {
        "system": "you are an extractor",
        "user": "extract from log fragment X",
    },
    "meta": {"datapack_name": "ts0-mysql-corrupt-kwx8n5"},
}


class FiringAdapterTests(unittest.TestCase):
    def test_extractor_adapter_validate(self) -> None:
        adapter = LlmharnessExtractorFiringTaskAdapter()
        sample = adapter.validate_sample(_ROW)
        self.assertEqual(sample.sample_id, "case-1:firing-0")
        self.assertEqual(sample.task_type, "llmharness_extractor_firing")
        self.assertEqual(sample.input["system"], "you are an extractor")
        self.assertEqual(sample.metadata.get("source_case_id"), "case-1")
        self.assertEqual(sample.metadata.get("firing_index"), 0)
        self.assertEqual(
            sample.metadata.get("datapack_name"), "ts0-mysql-corrupt-kwx8n5"
        )

    def test_extractor_adapter_to_agent_input(self) -> None:
        adapter = LlmharnessExtractorFiringTaskAdapter()
        sample = adapter.validate_sample(_ROW)
        agent_input = adapter.to_agent_input(sample)
        self.assertEqual(agent_input.sample_id, "case-1:firing-0")
        self.assertEqual(len(agent_input.messages), 2)
        self.assertEqual(agent_input.messages[0]["role"], "system")
        self.assertEqual(agent_input.messages[1]["role"], "user")
        self.assertIn("extract", agent_input.messages[1]["content"])

    def test_auditor_adapter_task_type(self) -> None:
        row = {**_ROW, "phase": "auditor"}
        adapter = LlmharnessAuditorFiringTaskAdapter()
        sample = adapter.validate_sample(row)
        self.assertEqual(sample.task_type, "llmharness_auditor_firing")

    def test_missing_input_raises(self) -> None:
        adapter = LlmharnessExtractorFiringTaskAdapter()
        with self.assertRaises(ValueError):
            adapter.validate_sample({"sample_id": "x"})

    def test_extractor_stub_runtime_produces_non_empty_trajectory(self) -> None:
        adapter = LlmharnessExtractorFiringTaskAdapter()
        sample = adapter.validate_sample(_ROW)
        agent_input = adapter.to_agent_input(sample)
        ctx = RuntimeContext(mode="eval", limits=RuntimeLimits(max_steps=32))
        result = asyncio.run(LlmharnessExtractorRuntime().run(agent_input, ctx))
        self.assertGreater(len(result.trajectory.steps), 0)
        self.assertEqual(result.trajectory.sample_id, "case-1:firing-0")
        # to_task_outcome bridges trajectory → outcome
        outcome = adapter.to_task_outcome(sample, result.trajectory)
        self.assertEqual(outcome.sample_id, "case-1:firing-0")

    def test_auditor_stub_runtime_produces_non_empty_trajectory(self) -> None:
        adapter = LlmharnessAuditorFiringTaskAdapter()
        sample = adapter.validate_sample({**_ROW, "phase": "auditor"})
        agent_input = adapter.to_agent_input(sample)
        ctx = RuntimeContext(mode="eval", limits=RuntimeLimits(max_steps=32))
        result = asyncio.run(LlmharnessAuditorRuntime().run(agent_input, ctx))
        self.assertGreater(len(result.trajectory.steps), 0)


if __name__ == "__main__":
    unittest.main()
