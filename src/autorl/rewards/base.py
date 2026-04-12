from __future__ import annotations

from abc import ABC, abstractmethod

from autorl.contracts import RuntimeContext, TaskOutcome, TaskSample, Trajectory


class RewardStrategy(ABC):
    """Framework-agnostic reward interface consumed by UnifiedAgentWorkflow."""

    @abstractmethod
    async def compute(
        self,
        sample: TaskSample,
        trajectory: Trajectory,
        outcome: TaskOutcome,
        runtime_context: RuntimeContext,
    ) -> float | dict[str, float]:
        raise NotImplementedError
