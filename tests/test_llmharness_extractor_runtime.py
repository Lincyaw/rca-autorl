"""Tests for the real LlmharnessExtractorRuntime.

The runtime wraps ``llmharness.replay_extractor_record``. We stub
llmharness's public surface entirely so the test doesn't require a
live LLM, AgentM, or the llmharness package itself to be installed.
"""

from __future__ import annotations

import asyncio
import importlib.machinery
import importlib.util as _ilu
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _ensure_stubs() -> None:
    # Minimal areal stub (other tests use this).
    if "areal.utils.dynamic_import" not in sys.modules:
        areal_module = sys.modules.setdefault("areal", types.ModuleType("areal"))
        utils_module = sys.modules.setdefault(
            "areal.utils", types.ModuleType("areal.utils")
        )
        dyn = types.ModuleType("areal.utils.dynamic_import")
        hf = types.ModuleType("areal.utils.hf_utils")
        dyn.import_from_string = lambda p: p  # type: ignore[attr-defined]
        hf.load_hf_tokenizer = lambda n: n  # type: ignore[attr-defined]
        sys.modules["areal.utils.dynamic_import"] = dyn
        sys.modules["areal.utils.hf_utils"] = hf
        areal_module.utils = utils_module  # type: ignore[attr-defined]
        utils_module.dynamic_import = dyn  # type: ignore[attr-defined]
        utils_module.hf_utils = hf  # type: ignore[attr-defined]
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


# ---- llmharness stub ----------------------------------------------------

_REPLAY_CAPTURED: dict[str, Any] = {}


@dataclass
class _StubReplayRecord:
    phase: str
    payload: dict
    compose_kwargs: dict
    root_session_id: str = "sess"
    turn_index: int = 0

    @classmethod
    def from_dict(cls, d):  # type: ignore[no-untyped-def]
        return cls(
            phase=d["phase"],
            payload=dict(d.get("payload") or {}),
            compose_kwargs=dict(d.get("compose_kwargs") or {}),
            root_session_id=str(d.get("root_session_id") or "sess"),
            turn_index=int(d.get("turn_index") or 0),
        )


@dataclass
class _StubPhaseResult:
    output: dict | None
    status: str
    error: str | None
    latency_ms: int
    messages: list


_TOOL_EVENTS_FIXTURE: list[dict[str, Any]] = []
_PHASE_RESULT_FIXTURE: _StubPhaseResult = _StubPhaseResult(
    output={}, status="ok", error=None, latency_ms=0, messages=[]
)


async def _stub_replay_extractor_record(record, *, cwd, provider_override=None, prompt_override=None):  # type: ignore[no-untyped-def]
    _REPLAY_CAPTURED["record"] = record
    _REPLAY_CAPTURED["cwd"] = cwd
    _REPLAY_CAPTURED["provider_override"] = provider_override
    return _PHASE_RESULT_FIXTURE


async def _stub_replay_auditor_record(record, *, cwd, provider_override=None, prompt_override=None):  # type: ignore[no-untyped-def]
    _REPLAY_CAPTURED["record"] = record
    _REPLAY_CAPTURED["cwd"] = cwd
    _REPLAY_CAPTURED["provider_override"] = provider_override
    return _PHASE_RESULT_FIXTURE


def _stub_tool_events_from_phase_result(result):  # type: ignore[no-untyped-def]
    return list(_TOOL_EVENTS_FIXTURE)


def _install_llmharness_stub() -> None:
    mod = types.ModuleType("llmharness")
    mod.__spec__ = importlib.machinery.ModuleSpec("llmharness", loader=None)
    mod.ReplayRecord = _StubReplayRecord  # type: ignore[attr-defined]
    mod.PhaseResult = _StubPhaseResult  # type: ignore[attr-defined]
    mod.replay_extractor_record = _stub_replay_extractor_record  # type: ignore[attr-defined]
    mod.replay_auditor_record = _stub_replay_auditor_record  # type: ignore[attr-defined]
    mod.tool_events_from_phase_result = _stub_tool_events_from_phase_result  # type: ignore[attr-defined]
    sys.modules["llmharness"] = mod


_install_llmharness_stub()


# ---- load the runtime module without importing autorl.runtime.__init__ ---

def _load_module(dotted: str, path: Path):  # type: ignore[no-untyped-def]
    spec = _ilu.spec_from_file_location(dotted, path)
    assert spec and spec.loader
    m = _ilu.module_from_spec(spec)
    sys.modules[dotted] = m
    spec.loader.exec_module(m)
    return m


if "autorl.runtime" not in sys.modules:
    pkg = types.ModuleType("autorl.runtime")
    pkg.__path__ = [str(src_path / "autorl" / "runtime")]  # type: ignore[attr-defined]
    sys.modules["autorl.runtime"] = pkg

# Only load each runtime submodule once — overwriting sys.modules with a
# fresh instance would break ``isinstance`` checks in sibling tests
# that have already cached the class object.
if "autorl.runtime.base" not in sys.modules:
    _load_module("autorl.runtime.base", src_path / "autorl" / "runtime" / "base.py")
if "autorl.runtime.llmharness_extractor" not in sys.modules:
    _load_module(
        "autorl.runtime.llmharness_extractor",
        src_path / "autorl" / "runtime" / "llmharness_extractor.py",
    )
_extractor_mod = sys.modules["autorl.runtime.llmharness_extractor"]
LlmharnessExtractorRuntime = _extractor_mod.LlmharnessExtractorRuntime


from autorl.contracts import AgentInput, RuntimeContext  # noqa: E402
from autorl.contracts.runtime import RuntimeLimits  # noqa: E402
from autorl.contracts.trajectory import TrajectoryStepType  # noqa: E402


_RAW_ROW: dict[str, Any] = {
    "phase": "extractor",
    "sample_id": "case-1:firing-0",
    "root_session_id": "sess-abc",
    "turn_index": 0,
    "compose_kwargs": {"base_prompt": "you are extractor"},
    "payload": {"recent_graph": [], "next_event_id": 1},
    "provider": None,
}


def _make_agent_input(raw: dict | None = None) -> AgentInput:
    raw = raw or _RAW_ROW
    return AgentInput(
        sample_id=str(raw["sample_id"]),
        task_type="llmharness_extractor_firing",
        raw_sample=dict(raw),
    )


class LlmharnessExtractorRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        global _TOOL_EVENTS_FIXTURE, _PHASE_RESULT_FIXTURE
        _REPLAY_CAPTURED.clear()
        _TOOL_EVENTS_FIXTURE = []
        _PHASE_RESULT_FIXTURE = _StubPhaseResult(
            output={}, status="ok", error=None, latency_ms=0, messages=[]
        )
        # Re-install the stub: a sibling test module may have replaced
        # ``sys.modules["llmharness"]`` with its own version between
        # collection and execution.
        _install_llmharness_stub()

    def _run(self, *, ctx: RuntimeContext | None = None):  # type: ignore[no-untyped-def]
        ctx = ctx or RuntimeContext(mode="eval", limits=RuntimeLimits(max_steps=32))
        return asyncio.run(LlmharnessExtractorRuntime().run(_make_agent_input(), ctx))

    def test_success_translates_tool_events_into_paired_steps(self) -> None:
        global _TOOL_EVENTS_FIXTURE, _PHASE_RESULT_FIXTURE
        _TOOL_EVENTS_FIXTURE = [
            {"tool_name": "upsert_node", "args": {"id": "n1"}, "is_error": False, "error_text": None},
            {"tool_name": "upsert_edge", "args": {"src": "n1", "dst": "n2"}, "is_error": True, "error_text": "boom"},
            {"tool_name": "finalize_extraction", "args": {}, "is_error": False, "error_text": None},
        ]
        _PHASE_RESULT_FIXTURE = _StubPhaseResult(
            output={"events": [{"id": 1}, {"id": 2}], "edges": [{}], "dropped_edges": []},
            status="ok",
            error=None,
            latency_ms=42,
            messages=[],
        )

        result = self._run()

        traj = result.trajectory
        # 3 tool events → 6 steps (call+result each).
        self.assertEqual(len(traj.steps), 6)
        # Order: TOOL_CALL, TOOL_RESULT, TOOL_CALL, TOOL_RESULT, ...
        for i, step in enumerate(traj.steps):
            expected = (
                TrajectoryStepType.TOOL_CALL
                if i % 2 == 0
                else TrajectoryStepType.TOOL_RESULT
            )
            self.assertEqual(step.step_type, expected)
        # Paired call_ids — call_id repeats for each call/result pair.
        self.assertEqual(traj.steps[0].call_id, traj.steps[1].call_id)
        self.assertEqual(traj.steps[2].call_id, traj.steps[3].call_id)
        self.assertNotEqual(traj.steps[0].call_id, traj.steps[2].call_id)
        # First step carries the tool args.
        self.assertEqual(traj.steps[0].input["tool_name"], "upsert_node")
        self.assertEqual(traj.steps[0].input["args"], {"id": "n1"})
        # Second step carries the error flag.
        self.assertFalse(traj.steps[1].output["is_error"])
        self.assertTrue(traj.steps[3].output["is_error"])
        self.assertEqual(traj.steps[3].output["error_text"], "boom")
        # Summary stats reflect PhaseResult.output and tool count.
        self.assertEqual(traj.summary_stats["events_count"], 2.0)
        self.assertEqual(traj.summary_stats["edges_count"], 1.0)
        self.assertEqual(traj.summary_stats["dropped_count"], 0.0)
        self.assertEqual(traj.summary_stats["tool_calls"], 3.0)
        self.assertEqual(traj.summary_stats["latency_ms"], 42.0)
        # Status maps "ok" → COMPLETED.
        from autorl.contracts import TrajectoryStatus
        self.assertEqual(traj.status, TrajectoryStatus.COMPLETED)
        # Outcome mirrors phase status.
        self.assertTrue(result.outcome.success)
        self.assertEqual(result.outcome.termination_reason, "ok")

    def test_error_status_marks_trajectory_failed(self) -> None:
        global _PHASE_RESULT_FIXTURE
        _PHASE_RESULT_FIXTURE = _StubPhaseResult(
            output=None, status="spawn_error", error="provider unavailable",
            latency_ms=10, messages=[],
        )
        result = self._run()
        from autorl.contracts import TrajectoryStatus
        self.assertEqual(result.trajectory.status, TrajectoryStatus.FAILED)
        self.assertFalse(result.outcome.success)
        self.assertEqual(result.outcome.termination_reason, "spawn_error")
        # Empty output → zeroed summary buckets.
        self.assertEqual(result.trajectory.summary_stats["events_count"], 0.0)
        # No tool events → no steps.
        self.assertEqual(len(result.trajectory.steps), 0)

    def test_empty_messages_produces_no_steps(self) -> None:
        global _TOOL_EVENTS_FIXTURE
        _TOOL_EVENTS_FIXTURE = []
        result = self._run()
        self.assertEqual(len(result.trajectory.steps), 0)
        self.assertEqual(result.trajectory.summary_stats["tool_calls"], 0.0)

    def test_provider_override_built_from_runtime_context(self) -> None:
        ctx = RuntimeContext(
            mode="train",
            model_endpoint="http://areal-proxy:8080/v1",
            api_key="secret-xyz",
            limits=RuntimeLimits(max_steps=8),
            metadata={"agentm_model": "qwen-coder-7b"},
        )
        self._run(ctx=ctx)
        override = _REPLAY_CAPTURED["provider_override"]
        self.assertIsNotNone(override)
        module, config = override
        self.assertEqual(module, "agentm.extensions.builtin.llm_openai")
        self.assertEqual(config["base_url"], "http://areal-proxy:8080/v1")
        self.assertEqual(config["api_key"], "secret-xyz")
        self.assertEqual(config["model"], "qwen-coder-7b")

    def test_no_provider_override_when_endpoint_absent(self) -> None:
        self._run()
        self.assertIsNone(_REPLAY_CAPTURED["provider_override"])


if __name__ == "__main__":
    unittest.main()
