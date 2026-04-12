from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import Metadata


@dataclass(slots=True)
class ToolCallResult:
    ok: bool
    name: str
    result: Any = None
    error: str | None = None
    latency_ms: float | None = None
    metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class EnvStepResult:
    ok: bool
    name: str
    observation: Any = None
    reward: float | None = None
    done: bool | None = None
    info: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    latency_ms: float | None = None
    metadata: Metadata = field(default_factory=dict)
