"""Dataset loading is task-agnostic at the repo level.

Task-specific field validation belongs to TaskAdapter implementations under
`autorl.tasks`, not in the generic data materialization layer.
"""

TASK_ADAPTER_OWNS_SCHEMA = True
