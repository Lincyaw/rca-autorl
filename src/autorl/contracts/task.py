from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TaskSample:
    """Normalized task envelope used across data, reward, and evaluation."""

    sample_id: str
    task_type: str
    input: dict[str, Any]
    target: Any | None = None
    reference: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentInput:
    """Framework-agnostic runtime input for a single agent episode."""

    sample_id: str
    task_type: str
    instruction: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)
    available_tools: list[dict[str, Any]] = field(default_factory=list)
    available_envs: list[dict[str, Any]] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_sample: dict[str, Any] = field(default_factory=dict)
