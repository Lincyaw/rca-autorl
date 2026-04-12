from __future__ import annotations

from typing import Any

from autorl.rca_utils import root_cause_key_set
from autorl.contracts import RuntimeContext, TaskOutcome, TaskSample, Trajectory

from .base import RewardStrategy


def root_cause_f1_reward(
    prediction: Any,
    reference: Any,
) -> float:
    pred = root_cause_key_set(prediction)
    ref = root_cause_key_set(reference)
    if not pred or not ref:
        return 0.0
    true_positive = len(pred & ref)
    if true_positive == 0:
        return 0.0
    precision = true_positive / len(pred)
    recall = true_positive / len(ref)
    return (2.0 * precision * recall) / (precision + recall)


class RootCauseMatchRewardStrategy(RewardStrategy):
    """Reward RCA predictions by root-cause set overlap."""

    async def compute(
        self,
        sample: TaskSample,
        trajectory: Trajectory,
        outcome: TaskOutcome,
        runtime_context: RuntimeContext,
    ) -> float:
        del trajectory, runtime_context
        reference = outcome.reference
        if reference is None:
            reference = sample.reference
        return root_cause_f1_reward(outcome.prediction, reference)
