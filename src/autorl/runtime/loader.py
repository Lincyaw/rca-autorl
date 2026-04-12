from __future__ import annotations

from typing import Any

from areal.utils.dynamic_import import import_from_string

from .base import AgentRuntime


def build_agent_runtime(path: str, kwargs: dict[str, Any] | None = None) -> AgentRuntime:
    """Resolve and instantiate an AgentRuntime from import path."""

    obj = import_from_string(path)
    if isinstance(obj, AgentRuntime):
        return obj
    if isinstance(obj, type):
        instance = obj(**(kwargs or {}))
        if not isinstance(instance, AgentRuntime):
            raise TypeError(f"runtime instance from '{path}' is not an AgentRuntime")
        return instance
    raise TypeError(f"runtime path '{path}' does not resolve to an AgentRuntime class/instance")
