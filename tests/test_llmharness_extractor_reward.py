"""Tests for LlmharnessExtractorRewardStrategy.

Pin the four shape-defining cases the brief calls out:

1. Happy path — 6 tool_calls, all succeed, last is ``finalize_extraction``.
2. Mid-flight failure — one tool_result is_error=True; finalize still succeeds.
3. Never finalized — trajectory ends without a finalize call.
4. Empty trajectory — no tool events. Reward zero, no crash.
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
from autorl.rewards.llmharness_extractor import (  # noqa: E402
    LlmharnessExtractorRewardStrategy,
)
from autorl.rewards.registry import get_reward_fn  # noqa: E402


def _call(idx: int, name: str, args: dict | None = None) -> TrajectoryStep:
    return TrajectoryStep(
        step_id=f"call-{idx}",
        step_type=TrajectoryStepType.TOOL_CALL,
        timestamp_ms=idx * 10,
        input={"name": name, "arguments": args or {}},
        call_id=f"c-{idx}",
    )


def _result(idx: int, *, is_error: bool = False, body: str = "ok") -> TrajectoryStep:
    return TrajectoryStep(
        step_id=f"res-{idx}",
        step_type=TrajectoryStepType.TOOL_RESULT,
        timestamp_ms=idx * 10 + 1,
        output={"is_error": is_error, "body": body},
        call_id=f"c-{idx}",
    )


def _sample() -> TaskSample:
    return TaskSample(
        sample_id="case", task_type="llmharness_extractor", input={}, reference={}
    )


def _outcome() -> TaskOutcome:
    return TaskOutcome(sample_id="case", task_type="llmharness_extractor")


def _ctx(max_steps: int = 32) -> RuntimeContext:
    return RuntimeContext(mode="eval", limits=RuntimeLimits(max_steps=max_steps))


def _traj(steps: list[TrajectoryStep]) -> Trajectory:
    return Trajectory(
        trajectory_id="t",
        sample_id="case",
        task_type="llmharness_extractor",
        status=TrajectoryStatus.COMPLETED,
        steps=steps,
    )


def _run(strategy, **kw):  # type: ignore[no-untyped-def]
    return asyncio.run(strategy.compute(**kw))


class ExtractorRewardTests(unittest.TestCase):
    def test_registered_under_canonical_name(self) -> None:
        self.assertIs(
            get_reward_fn("llmharness_extractor"),
            LlmharnessExtractorRewardStrategy,
        )

    def test_happy_path_all_success_with_finalize(self) -> None:
        steps: list[TrajectoryStep] = []
        names = [
            "upsert_node",
            "upsert_node",
            "upsert_edge",
            "upsert_edge",
            "upsert_edge",
            "finalize_extraction",
        ]
        for i, n in enumerate(names):
            steps.append(_call(i, n))
            steps.append(_result(i, is_error=False))
        reward = _run(
            LlmharnessExtractorRewardStrategy(),
            sample=_sample(),
            trajectory=_traj(steps),
            outcome=_outcome(),
            runtime_context=_ctx(max_steps=32),
        )
        self.assertEqual(reward["witness_pass_rate"], 1.0)
        self.assertEqual(reward["finalize_success"], 1.0)
        # 6 / 32 = 0.1875
        self.assertAlmostEqual(reward["efficiency_penalty"], 6 / 32)
        # 0.5 + 0.3 - 0.2 * 0.1875 = 0.7625
        self.assertAlmostEqual(reward["reward"], 0.5 + 0.3 - 0.2 * (6 / 32))

    def test_mid_flight_failure_does_not_block_finalize(self) -> None:
        steps: list[TrajectoryStep] = []
        plan = [
            ("upsert_node", False),
            ("upsert_node", False),
            ("upsert_node", False),
            ("upsert_edge", True),  # the witness-validation failure
            ("upsert_edge", False),
            ("finalize_extraction", False),
        ]
        for i, (n, err) in enumerate(plan):
            steps.append(_call(i, n))
            steps.append(_result(i, is_error=err))
        reward = _run(
            LlmharnessExtractorRewardStrategy(),
            sample=_sample(),
            trajectory=_traj(steps),
            outcome=_outcome(),
            runtime_context=_ctx(max_steps=32),
        )
        self.assertAlmostEqual(reward["witness_pass_rate"], 5 / 6)
        self.assertEqual(reward["finalize_success"], 1.0)
        self.assertAlmostEqual(reward["efficiency_penalty"], 6 / 32)
        expected = 0.5 * 1.0 + 0.3 * (5 / 6) - 0.2 * (6 / 32)
        self.assertAlmostEqual(reward["reward"], expected)

    def test_never_finalized_zeros_finalize_dimension(self) -> None:
        steps: list[TrajectoryStep] = []
        for i in range(5):
            steps.append(_call(i, "upsert_node"))
            steps.append(_result(i, is_error=False))
        reward = _run(
            LlmharnessExtractorRewardStrategy(),
            sample=_sample(),
            trajectory=_traj(steps),
            outcome=_outcome(),
            runtime_context=_ctx(max_steps=32),
        )
        self.assertEqual(reward["finalize_success"], 0.0)
        self.assertEqual(reward["witness_pass_rate"], 1.0)
        self.assertAlmostEqual(reward["efficiency_penalty"], 5 / 32)
        expected = 0.0 + 0.3 * 1.0 - 0.2 * (5 / 32)
        self.assertAlmostEqual(reward["reward"], expected)

    def test_empty_trajectory_returns_zero_without_crash(self) -> None:
        reward = _run(
            LlmharnessExtractorRewardStrategy(),
            sample=_sample(),
            trajectory=_traj([]),
            outcome=_outcome(),
            runtime_context=_ctx(max_steps=32),
        )
        self.assertEqual(reward["reward"], 0.0)
        self.assertEqual(reward["witness_pass_rate"], 0.0)
        self.assertEqual(reward["finalize_success"], 0.0)
        self.assertEqual(reward["efficiency_penalty"], 0.0)


if __name__ == "__main__":
    unittest.main()
