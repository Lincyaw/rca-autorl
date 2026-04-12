from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from .common import Metadata

if TYPE_CHECKING:
    from autorl.gateways.base import EnvGateway, ToolGateway

RuntimeMode = Literal["train", "eval", "infer"]


@dataclass(slots=True)
class RuntimeLimits:
    """Execution guardrails for a single agent episode."""

    max_steps: int | None = None
    max_llm_calls: int | None = None
    max_tokens: int | None = None
    timeout_seconds: float | None = None


@dataclass(slots=True)
class RuntimeContext:
    """Runtime resources injected by rollout or inference host."""

    mode: RuntimeMode
    model_endpoint: str | None = None
    api_key: str | None = None
    http_client: Any | None = None
    tool_gateway: "ToolGateway | None" = None
    env_gateway: "EnvGateway | None" = None
    trace_collector: Any | None = None
    limits: RuntimeLimits = field(default_factory=RuntimeLimits)
    metadata: Metadata = field(default_factory=dict)
