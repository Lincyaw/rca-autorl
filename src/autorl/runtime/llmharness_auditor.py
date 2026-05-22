"""Stub runtime for a single llmharness auditor child firing.

Auditor counterpart of ``LlmharnessExtractorRuntime``. Same contract,
different tool surface (read-only graph inspection + ``submit_verdict``).

# TODO(real-impl): wire to the llmharness auditor child loop.
#
# The real implementation needs to:
#   1. Resolve the auditor atom from llmharness
#      (``llmharness.audit.auditor.atom``) and mount its tool list.
#   2. Seed the session with ``agent_input.messages`` (system + user
#      built by the firing TaskAdapter).
#   3. Drive read-only inspection turns; capture each tool_call /
#      tool_result pair into Trajectory steps.
#   4. Stop on ``submit_verdict`` or ``max_steps`` exhaustion.
#   5. Populate ``trajectory.final_output`` with the verdict payload.
"""

from __future__ import annotations

import time
from uuid import uuid4

from autorl.contracts import (
    AgentInput,
    AgentRunResult,
    RuntimeContext,
    TaskOutcome,
    Trajectory,
    TrajectoryStatus,
    TrajectoryStep,
    TrajectoryStepType,
)

from .base import AgentRuntime


class LlmharnessAuditorRuntime(AgentRuntime):
    """Single auditor-child runtime (STUB)."""

    async def run(
        self,
        agent_input: AgentInput,
        runtime_context: RuntimeContext,
    ) -> AgentRunResult:
        now_ms = int(time.time() * 1000)
        step = TrajectoryStep(
            step_id="control-1",
            step_type=TrajectoryStepType.CONTROL,
            timestamp_ms=now_ms,
            input={"runtime": "llmharness_auditor", "stub": True},
            output={
                "message": "stub runtime — see TODO(real-impl) in source",
                "mode": runtime_context.mode,
            },
            metadata={"sample_id": agent_input.sample_id},
        )
        trajectory = Trajectory(
            trajectory_id=uuid4().hex,
            sample_id=agent_input.sample_id,
            task_type=agent_input.task_type,
            status=TrajectoryStatus.COMPLETED,
            final_output={"termination": "stub_auditor_runtime"},
            steps=[step],
            summary_stats={"steps": 1.0},
            metadata=dict(agent_input.metadata),
        )
        outcome = TaskOutcome(
            sample_id=agent_input.sample_id,
            task_type=agent_input.task_type,
            success=None,
            termination_reason="stub_auditor_runtime",
            metadata=dict(agent_input.metadata),
        )
        return AgentRunResult(trajectory=trajectory, outcome=outcome)


__all__ = ["LlmharnessAuditorRuntime"]
