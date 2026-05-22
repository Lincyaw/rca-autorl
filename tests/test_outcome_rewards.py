"""Tests for outcome-passthrough reward strategies.

Covers the presence + absence of ``case_outcome`` for both extractor
and auditor variants.
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
from autorl.rewards.llmharness_auditor_outcome import (  # noqa: E402
    LlmharnessAuditorOutcomeRewardStrategy,
)
from autorl.rewards.llmharness_extractor_outcome import (  # noqa: E402
    LlmharnessExtractorOutcomeRewardStrategy,
)
from autorl.rewards.registry import get_reward_fn  # noqa: E402


def _traj() -> Trajectory:
    return Trajectory(
        trajectory_id="t",
        sample_id="case",
        task_type="x",
        status=TrajectoryStatus.COMPLETED,
    )


def _outcome() -> TaskOutcome:
    return TaskOutcome(sample_id="case", task_type="x")


def _ctx() -> RuntimeContext:
    return RuntimeContext(mode="eval", limits=RuntimeLimits(max_steps=32))


def _run(strategy, sample):  # type: ignore[no-untyped-def]
    return asyncio.run(
        strategy.compute(
            sample=sample,
            trajectory=_traj(),
            outcome=_outcome(),
            runtime_context=_ctx(),
        )
    )


class OutcomeRewardTests(unittest.TestCase):
    def test_extractor_registered(self) -> None:
        self.assertIs(
            get_reward_fn("llmharness_extractor_outcome"),
            LlmharnessExtractorOutcomeRewardStrategy,
        )

    def test_auditor_registered(self) -> None:
        self.assertIs(
            get_reward_fn("llmharness_auditor_outcome"),
            LlmharnessAuditorOutcomeRewardStrategy,
        )

    def test_extractor_reads_case_outcome_from_reference(self) -> None:
        sample = TaskSample(
            sample_id="case",
            task_type="x",
            input={},
            reference={
                "case_outcome": {
                    "composite_score": 0.75,
                    "service_hit": 1.0,
                    "fault_kind_hit": 0.5,
                }
            },
        )
        out = _run(LlmharnessExtractorOutcomeRewardStrategy(), sample)
        self.assertAlmostEqual(out["reward"], 0.75)
        self.assertEqual(out["service_hit"], 1.0)
        self.assertEqual(out["fault_kind_hit"], 0.5)

    def test_auditor_reads_case_outcome_from_metadata(self) -> None:
        sample = TaskSample(
            sample_id="case",
            task_type="x",
            input={},
            metadata={"case_outcome": {"composite_score": 0.2}},
        )
        out = _run(LlmharnessAuditorOutcomeRewardStrategy(), sample)
        self.assertAlmostEqual(out["reward"], 0.2)

    def test_absent_label_returns_zero_with_flag(self) -> None:
        sample = TaskSample(
            sample_id="case", task_type="x", input={}, reference={}, metadata={}
        )
        out = _run(LlmharnessExtractorOutcomeRewardStrategy(), sample)
        self.assertEqual(out["reward"], 0.0)
        self.assertEqual(out["no_outcome_label"], 1.0)

        out2 = _run(LlmharnessAuditorOutcomeRewardStrategy(), sample)
        self.assertEqual(out2["reward"], 0.0)
        self.assertEqual(out2["no_outcome_label"], 1.0)


if __name__ == "__main__":
    unittest.main()
