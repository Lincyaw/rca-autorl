from __future__ import annotations

from autorl.contracts import RuntimeContext, TaskOutcome, TaskSample, Trajectory

from .base import RewardStrategy


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _normalize_answer(answer: str | list[str] | tuple[str, ...] | None) -> str:
    if answer is None:
        return ""
    if isinstance(answer, (list, tuple)):
        answer = " | ".join(str(x) for x in answer)
    return _normalize_text(str(answer))


def reference_match_reward(prediction: str, answer: str | list[str] | None = None) -> float:
    pred = _normalize_text(str(prediction))
    gt = _normalize_answer(answer)
    if not pred or not gt:
        return 0.0
    if pred == gt or pred in gt or gt in pred:
        return 1.0
    return 0.0


class ReferenceMatchRewardStrategy(RewardStrategy):
    """Simple reward strategy that compares prediction against reference text."""

    async def compute(
        self,
        sample: TaskSample,
        trajectory: Trajectory,
        outcome: TaskOutcome,
        runtime_context: RuntimeContext,
    ) -> float:
        reference = outcome.reference
        if reference is None and isinstance(sample.reference, dict):
            reference = sample.reference.get("answer")
        return reference_match_reward(str(outcome.prediction or ""), reference)
