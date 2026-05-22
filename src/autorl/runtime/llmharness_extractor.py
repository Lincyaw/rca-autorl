"""Stub runtime for a single llmharness extractor child firing.

This runtime is the online-RL counterpart of the SFT path: each
rollout corresponds to **one** extractor child session — driven by the
v19 tool surface (``upsert_node`` / ``upsert_edge`` /
``finalize_extraction``) — rather than a full RCA case (that's
``AgentMRuntime``'s job).

The reward signal comes from the deterministic process reward in
``autorl.rewards.llmharness_extractor``; this runtime just has to
produce a Trajectory whose ``steps`` carry paired TOOL_CALL /
TOOL_RESULT events so the reward strategy can score it.

# TODO(real-impl): wire to the llmharness extractor child loop.
#
# The real implementation needs to:
#   1. Resolve the extractor atom from llmharness
#      (``llmharness.audit.extractor.atom``) and mount its tool list
#      via a minimal AgentM scenario OR a direct ExtensionAPI stub.
#   2. Seed the session with ``agent_input.messages`` (system + user
#      built by the firing TaskAdapter).
#   3. Drive turns: ask the model for the next tool call, route it
#      through the mounted tools, capture witness validation feedback,
#      emit TrajectoryStep entries for each call/result.
#   4. Stop when the model emits ``finalize_extraction`` or the
#      ``max_steps`` budget runs out.
#   5. Populate ``trajectory.final_output`` with the resulting graph
#      JSON so downstream consumers can audit it.
#
# Until that's in place this class returns a one-step CONTROL
# trajectory marked "stub runtime" so trainer configs can still
# reference the class path.
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


class LlmharnessExtractorRuntime(AgentRuntime):
    """Single extractor-child runtime (STUB).

    Produces a placeholder Trajectory so downstream code (rewards,
    workflow) doesn't crash when the trainer wiring references this
    runtime path before the real implementation lands.
    """

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
            input={"runtime": "llmharness_extractor", "stub": True},
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
            final_output={"termination": "stub_extractor_runtime"},
            steps=[step],
            summary_stats={"steps": 1.0},
            metadata=dict(agent_input.metadata),
        )
        outcome = TaskOutcome(
            sample_id=agent_input.sample_id,
            task_type=agent_input.task_type,
            success=None,
            termination_reason="stub_extractor_runtime",
            metadata=dict(agent_input.metadata),
        )
        return AgentRunResult(trajectory=trajectory, outcome=outcome)


__all__ = ["LlmharnessExtractorRuntime"]
