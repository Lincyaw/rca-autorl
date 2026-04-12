from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import Metadata
from .trajectory import Trajectory


@dataclass(slots=True)
class TaskOutcome:
    """Task-level evaluation view extracted from a trajectory."""

    sample_id: str
    task_type: str
    prediction: Any = None
    reference: Any = None
    success: bool | None = None
    termination_reason: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class AgentRunResult:
    """Canonical runtime result returned by AgentRuntime."""

    trajectory: Trajectory
    outcome: TaskOutcome
