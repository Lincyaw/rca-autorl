"""Tests for LlmharnessAuditorProcessRewardStrategy.

Mirrors the extractor reward tests, but the finalize tool is
``submit_verdict``.
"""

from __future__ import annotations

import asyncio
import importlib.machinery
import sys
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

from autorl.contracts import (  # noqa: E402
    RuntimeContext,
    TaskOutcome,
    TaskSample,
    Trajectory,
    TrajectoryStatus,
)
from autorl.contracts.runtime import RuntimeLimits  # noqa: E402
from autorl.contracts.trajectory import TrajectoryStep, TrajectoryStepType  # noqa: E402
from autorl.rewards.llmharness_auditor import (  # noqa: E402
    LlmharnessAuditorProcessRewardStrategy,
)
from autorl.rewards.registry import get_reward_fn  # noqa: E402


def _call(idx: int, name: str) -> TrajectoryStep:
    return TrajectoryStep(
        step_id=f"call-{idx}",
        step_type=TrajectoryStepType.TOOL_CALL,
        timestamp_ms=idx * 10,
        input={"name": name},
        call_id=f"c-{idx}",
    )


def _result(idx: int, *, is_error: bool = False) -> TrajectoryStep:
    return TrajectoryStep(
        step_id=f"res-{idx}",
        step_type=TrajectoryStepType.TOOL_RESULT,
        timestamp_ms=idx * 10 + 1,
        output={"is_error": is_error},
        call_id=f"c-{idx}",
    )


def _sample() -> TaskSample:
    return TaskSample(
        sample_id="case", task_type="llmharness_auditor", input={}, reference={}
    )


def _outcome() -> TaskOutcome:
    return TaskOutcome(sample_id="case", task_type="llmharness_auditor")


def _ctx(max_steps: int = 32) -> RuntimeContext:
    return RuntimeContext(mode="eval", limits=RuntimeLimits(max_steps=max_steps))


def _traj(steps: list[TrajectoryStep]) -> Trajectory:
    return Trajectory(
        trajectory_id="t",
        sample_id="case",
        task_type="llmharness_auditor",
        status=TrajectoryStatus.COMPLETED,
        steps=steps,
    )


def _run(strategy, **kw):  # type: ignore[no-untyped-def]
    return asyncio.run(strategy.compute(**kw))


class AuditorProcessRewardTests(unittest.TestCase):
    def test_registered_under_process_name(self) -> None:
        self.assertIs(
            get_reward_fn("llmharness_auditor_process"),
            LlmharnessAuditorProcessRewardStrategy,
        )

    def test_happy_path_ends_with_submit_verdict(self) -> None:
        plan = ["inspect_node", "inspect_edge", "inspect_edge", "submit_verdict"]
        steps: list[TrajectoryStep] = []
        for i, n in enumerate(plan):
            steps.append(_call(i, n))
            steps.append(_result(i))
        reward = _run(
            LlmharnessAuditorProcessRewardStrategy(),
            sample=_sample(),
            trajectory=_traj(steps),
            outcome=_outcome(),
            runtime_context=_ctx(max_steps=32),
        )
        self.assertEqual(reward["witness_pass_rate"], 1.0)
        self.assertEqual(reward["finalize_success"], 1.0)
        self.assertAlmostEqual(reward["efficiency_penalty"], 4 / 32)
        self.assertAlmostEqual(
            reward["reward"], 0.5 + 0.3 - 0.2 * (4 / 32)
        )

    def test_no_submit_verdict_zeros_finalize(self) -> None:
        plan = ["inspect_node", "inspect_edge"]
        steps: list[TrajectoryStep] = []
        for i, n in enumerate(plan):
            steps.append(_call(i, n))
            steps.append(_result(i))
        reward = _run(
            LlmharnessAuditorProcessRewardStrategy(),
            sample=_sample(),
            trajectory=_traj(steps),
            outcome=_outcome(),
            runtime_context=_ctx(max_steps=32),
        )
        self.assertEqual(reward["finalize_success"], 0.0)

    def test_empty_trajectory_returns_zero(self) -> None:
        reward = _run(
            LlmharnessAuditorProcessRewardStrategy(),
            sample=_sample(),
            trajectory=_traj([]),
            outcome=_outcome(),
            runtime_context=_ctx(),
        )
        self.assertEqual(reward["reward"], 0.0)


if __name__ == "__main__":
    unittest.main()
