from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .common import Metadata


class TrajectoryStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    TRUNCATED = "truncated"
    REJECTED = "rejected"


class TrajectoryStepType(str, Enum):
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ENV_ACTION = "env_action"
    ENV_OBSERVATION = "env_observation"
    AGENT_THOUGHT = "agent_thought"
    CONTROL = "control"
    FINAL = "final"


@dataclass(slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(slots=True)
class TrajectoryStep:
    """Single event in a runtime trajectory."""

    step_id: str
    step_type: TrajectoryStepType
    timestamp_ms: int
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    usage: TokenUsage | None = None
    call_id: str | None = None
    parent_call_id: str | None = None
    metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class Trajectory:
    """Canonical trajectory format for training and inference pipelines."""

    trajectory_id: str
    sample_id: str
    task_type: str
    status: TrajectoryStatus
    final_output: dict[str, Any] = field(default_factory=dict)
    steps: list[TrajectoryStep] = field(default_factory=list)
    summary_stats: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: Metadata = field(default_factory=dict)
