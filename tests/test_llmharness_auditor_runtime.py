"""Tests for the real LlmharnessAuditorRuntime.

Mirrors the extractor-runtime test — stubs out llmharness entirely.
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


_CAPTURED: dict[str, Any] = {}


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
    _CAPTURED["record"] = record
    _CAPTURED["cwd"] = cwd
    _CAPTURED["provider_override"] = provider_override
    return _PHASE_RESULT_FIXTURE


async def _stub_replay_auditor_record(record, *, cwd, provider_override=None, prompt_override=None):  # type: ignore[no-untyped-def]
    _CAPTURED["record"] = record
    _CAPTURED["cwd"] = cwd
    _CAPTURED["provider_override"] = provider_override
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

if "autorl.runtime.base" not in sys.modules:
    _load_module("autorl.runtime.base", src_path / "autorl" / "runtime" / "base.py")
if "autorl.runtime.llmharness_auditor" not in sys.modules:
    _load_module(
        "autorl.runtime.llmharness_auditor",
        src_path / "autorl" / "runtime" / "llmharness_auditor.py",
    )
_auditor_mod = sys.modules["autorl.runtime.llmharness_auditor"]
LlmharnessAuditorRuntime = _auditor_mod.LlmharnessAuditorRuntime


from autorl.contracts import AgentInput, RuntimeContext, TrajectoryStatus  # noqa: E402
from autorl.contracts.runtime import RuntimeLimits  # noqa: E402
from autorl.contracts.trajectory import TrajectoryStepType  # noqa: E402


_RAW_ROW: dict[str, Any] = {
    "phase": "auditor",
    "sample_id": "case-1:firing-0",
    "root_session_id": "sess-abc",
    "turn_index": 0,
    "compose_kwargs": {"base_prompt": "you are auditor"},
    "payload": {"graph": []},
    "provider": None,
}


def _make_agent_input() -> AgentInput:
    return AgentInput(
        sample_id="case-1:firing-0",
        task_type="llmharness_auditor_firing",
        raw_sample=dict(_RAW_ROW),
    )


class LlmharnessAuditorRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        global _TOOL_EVENTS_FIXTURE, _PHASE_RESULT_FIXTURE
        _CAPTURED.clear()
        _TOOL_EVENTS_FIXTURE = []
        _PHASE_RESULT_FIXTURE = _StubPhaseResult(
            output={}, status="ok", error=None, latency_ms=0, messages=[]
        )
        _install_llmharness_stub()

    def _run(self, *, ctx: RuntimeContext | None = None):  # type: ignore[no-untyped-def]
        ctx = ctx or RuntimeContext(mode="eval", limits=RuntimeLimits(max_steps=32))
        return asyncio.run(LlmharnessAuditorRuntime().run(_make_agent_input(), ctx))

    def test_success_with_verdict_translates_steps_and_summary(self) -> None:
        global _TOOL_EVENTS_FIXTURE, _PHASE_RESULT_FIXTURE
        _TOOL_EVENTS_FIXTURE = [
            {"tool_name": "get_event_detail", "args": {"id": 1}, "is_error": False, "error_text": None},
            {"tool_name": "submit_verdict", "args": {"verdict": {"label": "ok"}}, "is_error": False, "error_text": None},
        ]
        _PHASE_RESULT_FIXTURE = _StubPhaseResult(
            output={"verdict": {"label": "ok"}, "findings": [{"id": "f1"}]},
            status="ok",
            error=None,
            latency_ms=33,
            messages=[],
        )

        result = self._run()

        traj = result.trajectory
        self.assertEqual(len(traj.steps), 4)
        self.assertEqual(traj.steps[0].step_type, TrajectoryStepType.TOOL_CALL)
        self.assertEqual(traj.steps[1].step_type, TrajectoryStepType.TOOL_RESULT)
        self.assertEqual(traj.steps[0].call_id, traj.steps[1].call_id)
        self.assertEqual(traj.summary_stats["has_verdict"], 1.0)
        self.assertEqual(traj.summary_stats["tool_calls"], 2.0)
        self.assertEqual(traj.summary_stats["findings_count"], 1.0)
        self.assertEqual(traj.status, TrajectoryStatus.COMPLETED)
        self.assertTrue(result.outcome.success)

    def test_error_status_marks_failed(self) -> None:
        global _PHASE_RESULT_FIXTURE
        _PHASE_RESULT_FIXTURE = _StubPhaseResult(
            output=None, status="no_call", error="model did not call submit_verdict",
            latency_ms=5, messages=[],
        )
        result = self._run()
        self.assertEqual(result.trajectory.status, TrajectoryStatus.FAILED)
        self.assertEqual(result.outcome.termination_reason, "no_call")
        self.assertEqual(result.trajectory.summary_stats["has_verdict"], 0.0)

    def test_empty_messages_no_steps(self) -> None:
        self._run()
        # The default fixture has 0 tool events.
        result = self._run()
        self.assertEqual(len(result.trajectory.steps), 0)

    def test_provider_override(self) -> None:
        ctx = RuntimeContext(
            mode="train",
            model_endpoint="http://proxy/v1",
            api_key="K",
            limits=RuntimeLimits(),
            metadata={"agentm_model": "qwen-7b"},
        )
        self._run(ctx=ctx)
        override = _CAPTURED["provider_override"]
        assert override is not None
        module, config = override
        self.assertEqual(module, "agentm.extensions.builtin.llm_openai")
        self.assertEqual(config["base_url"], "http://proxy/v1")
        self.assertEqual(config["model"], "qwen-7b")


if __name__ == "__main__":
    unittest.main()
