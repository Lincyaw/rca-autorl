"""Task adapters and example task implementations."""

from .base import TaskAdapter
from .loader import build_task_adapter
from .rca import RCATaskAdapter
from .search import SearchTaskAdapter

__all__ = [
    "RCATaskAdapter",
    "SearchTaskAdapter",
    "TaskAdapter",
    "build_task_adapter",
]
