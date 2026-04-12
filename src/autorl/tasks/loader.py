from __future__ import annotations

from typing import Any

from areal.utils.dynamic_import import import_from_string

from .base import TaskAdapter


def build_task_adapter(path: str, kwargs: dict[str, Any] | None = None) -> TaskAdapter:
    """Resolve and instantiate a TaskAdapter from import path."""

    obj = import_from_string(path)
    if isinstance(obj, TaskAdapter):
        return obj
    if isinstance(obj, type):
        instance = obj(**(kwargs or {}))
        if not isinstance(instance, TaskAdapter):
            raise TypeError(f"adapter instance from '{path}' is not a TaskAdapter")
        return instance
    raise TypeError(f"adapter path '{path}' does not resolve to a TaskAdapter class/instance")
