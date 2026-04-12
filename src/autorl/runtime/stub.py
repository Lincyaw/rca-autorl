from __future__ import annotations

import time
from uuid import uuid4

from openai import AsyncOpenAI

from autorl.contracts import (
    AgentInput,
    AgentRunResult,
    RuntimeContext,
    TaskOutcome,
    TokenUsage,
    Trajectory,
    TrajectoryStatus,
    TrajectoryStep,
    TrajectoryStepType,
)

from .base import AgentRuntime


class StubAgentRuntime(AgentRuntime):
    """Minimal runtime implementation for integration smoke checks.

    This runtime makes a single model call through the injected OpenAI-compatible
    endpoint so AReaL still records trainable interactions on the smoke path.
    """

    async def run(
        self,
        agent_input: AgentInput,
        runtime_context: RuntimeContext,
    ) -> AgentRunResult:
        if not runtime_context.model_endpoint:
            raise ValueError("StubAgentRuntime requires RuntimeContext.model_endpoint")

        now_ms = int(time.time() * 1000)
        max_tokens = int(runtime_context.metadata.get("max_tokens_per_turn", 64))
        client = AsyncOpenAI(
            base_url=runtime_context.model_endpoint,
            api_key=runtime_context.api_key or "EMPTY",
            http_client=runtime_context.http_client,
            max_retries=0,
        )
        completion = await client.chat.completions.create(
            model="default",
            messages=agent_input.messages,
            temperature=0.0,
            max_completion_tokens=max_tokens,
        )
        message = completion.choices[0].message
        prediction = message.content or ""
        usage = TokenUsage(
            prompt_tokens=getattr(completion.usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(completion.usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(completion.usage, "total_tokens", 0) or 0,
        )

        trajectory = Trajectory(
            trajectory_id=uuid4().hex,
            sample_id=agent_input.sample_id,
            task_type=agent_input.task_type,
            status=TrajectoryStatus.COMPLETED,
            final_output={
                "prediction": prediction,
                "completion_id": completion.id,
                "termination": "stub_single_turn",
            },
            steps=[
                TrajectoryStep(
                    step_id="control-1",
                    step_type=TrajectoryStepType.CONTROL,
                    timestamp_ms=now_ms,
                    output={"mode": runtime_context.mode},
                ),
                TrajectoryStep(
                    step_id="llm-response-1",
                    step_type=TrajectoryStepType.LLM_RESPONSE,
                    timestamp_ms=int(time.time() * 1000),
                    output={"prediction": prediction},
                    usage=usage,
                    call_id=completion.id,
                ),
                TrajectoryStep(
                    step_id="final-1",
                    step_type=TrajectoryStepType.FINAL,
                    timestamp_ms=int(time.time() * 1000),
                    output={"prediction": prediction},
                ),
            ],
            summary_stats={"steps": 3.0},
            metadata=dict(agent_input.metadata),
        )
        outcome = TaskOutcome(
            sample_id=agent_input.sample_id,
            task_type=agent_input.task_type,
            prediction=prediction,
            reference=None,
            success=None,
            termination_reason="stub_single_turn",
            metadata=dict(agent_input.metadata),
        )
        return AgentRunResult(trajectory=trajectory, outcome=outcome)
