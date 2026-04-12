"""Framework-agnostic contracts shared by runtime, tasks, and gateways."""

from .common import JsonScalar, JsonValue, Metadata
from .gateway import EnvStepResult, ToolCallResult
from .outcome import AgentRunResult, TaskOutcome
from .runtime import RuntimeContext, RuntimeLimits, RuntimeMode
from .task import AgentInput, TaskSample
from .training import RewardAssignment, TrainingEpisodeView
from .trajectory import (
    TokenUsage,
    Trajectory,
    TrajectoryStatus,
    TrajectoryStep,
    TrajectoryStepType,
)

__all__ = [
    "AgentInput",
    "AgentRunResult",
    "EnvStepResult",
    "JsonScalar",
    "JsonValue",
    "Metadata",
    "RewardAssignment",
    "RuntimeContext",
    "RuntimeLimits",
    "RuntimeMode",
    "TaskOutcome",
    "TaskSample",
    "TokenUsage",
    "ToolCallResult",
    "TrainingEpisodeView",
    "Trajectory",
    "TrajectoryStatus",
    "TrajectoryStep",
    "TrajectoryStepType",
]
