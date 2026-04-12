"""Task adapters and example task implementations."""

from .base import TaskAdapter
from .loader import build_task_adapter
from .search import SearchTaskAdapter

__all__ = [
    "SearchTaskAdapter",
    "TaskAdapter",
    "build_task_adapter",
]
