from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from autorl.contracts import (
    AgentInput,
    TaskOutcome,
    TaskSample,
    TrainingEpisodeView,
    Trajectory,
)


class TaskAdapter(ABC):
    """Task-specific adapter that keeps runtime and dataset schema decoupled."""

    task_type: str
    required_keys: tuple[str, ...] = ()

    @abstractmethod
    def validate_sample(self, raw_sample: Mapping[str, Any]) -> TaskSample:
        """Validate raw payload and map it into TaskSample."""
        raise NotImplementedError

    @abstractmethod
    def to_agent_input(self, sample: TaskSample) -> AgentInput:
        """Project TaskSample into canonical AgentInput."""
        raise NotImplementedError

    @abstractmethod
    def to_task_outcome(
        self,
        sample: TaskSample,
        trajectory: Trajectory,
    ) -> TaskOutcome:
        """Extract task-level outcome from trajectory."""
        raise NotImplementedError

    def to_training_view(
        self,
        sample: TaskSample,
        trajectory: Trajectory,
        outcome: TaskOutcome,
    ) -> TrainingEpisodeView:
        """Default no-op training projection; tasks can override when needed."""

        return TrainingEpisodeView(
            sample_id=sample.sample_id,
            trajectory_id=trajectory.trajectory_id,
            model_interactions=[],
            reward_assignments=[],
            discount_config={},
            aux_metrics=dict(outcome.metrics),
            metadata=dict(sample.metadata),
        )
