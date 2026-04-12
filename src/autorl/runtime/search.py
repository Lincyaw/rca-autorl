from __future__ import annotations

import time
from uuid import uuid4

from openai import AsyncOpenAI
from transformers import PreTrainedTokenizerFast

from areal.utils.hf_utils import load_hf_tokenizer

from autorl.agents import AgentRunner
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
from autorl.rewards.local import reference_match_reward
from autorl.tool_env.client import ToolEnvClient

from .base import AgentRuntime


def _step_type_for(role: str, content: str) -> TrajectoryStepType:
    if "<tool_call>" in content:
        return TrajectoryStepType.TOOL_CALL
    if "<tool_response>" in content:
        return TrajectoryStepType.TOOL_RESULT
    if role == "assistant":
        return TrajectoryStepType.LLM_RESPONSE
    return TrajectoryStepType.CONTROL


class SearchAgentRuntime(AgentRuntime):
    """Example runtime backed by the existing multi-turn search agent loop."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast | str,
        max_tokens_per_turn: int = 4096,
        max_llm_calls_per_run: int = 100,
        max_total_tokens: int = 32768,
        tool_env_base_url: str | None = None,
        judge_engine_addr: str | None = None,
    ) -> None:
        if isinstance(tokenizer, str):
            tokenizer = load_hf_tokenizer(tokenizer)
        self.tokenizer = tokenizer
        self.max_tokens_per_turn = max_tokens_per_turn
        self.max_llm_calls_per_run = max_llm_calls_per_run
        self.max_total_tokens = max_total_tokens
        self.tool_env_base_url = tool_env_base_url
        self.judge_engine_addr = judge_engine_addr

    async def run(self, agent_input: AgentInput, runtime_context: RuntimeContext) -> AgentRunResult:
        question = str(agent_input.context.get("question", ""))
        reference = agent_input.context.get("answer")
        payload = {
            "qid": agent_input.sample_id,
            "question": question,
            "answer": reference,
        }

        tool_env_base_url = str(
            runtime_context.metadata.get("tool_env_base_url") if runtime_context.metadata else ""
        ) or self.tool_env_base_url
        judge_base_url = str(
            runtime_context.metadata.get("judge_base_url") if runtime_context.metadata else ""
        ) or self.judge_engine_addr

        tool_client = ToolEnvClient(base_url=tool_env_base_url)

        async with AsyncOpenAI(
            base_url=runtime_context.model_endpoint,
            api_key=runtime_context.api_key or "EMPTY",
            http_client=runtime_context.http_client,
            max_retries=0,
        ) as actor_client:
            judge_client = None
            if judge_base_url:
                judge_client = AsyncOpenAI(
                    base_url=judge_base_url,
                    api_key="EMPTY",
                    http_client=runtime_context.http_client,
                    max_retries=0,
                )
            try:
                agent = AgentRunner(
                    tokenizer=self.tokenizer,
                    max_tokens_per_turn=self.max_tokens_per_turn,
                    max_llm_calls_per_run=self.max_llm_calls_per_run,
                    max_total_tokens=self.max_total_tokens,
                    judge_client=judge_client,
                    tool_client=tool_client,
                )
                result = await agent.run_agent(data=payload, client=actor_client)
            finally:
                if judge_client is not None:
                    await judge_client.close()

        prediction = str(result.get("prediction", ""))
        success = bool(reference_match_reward(prediction=prediction, answer=reference)) if reference is not None else None
        now_ms = int(time.time() * 1000)

        trajectory = Trajectory(
            trajectory_id=uuid4().hex,
            sample_id=agent_input.sample_id,
            task_type=agent_input.task_type,
            status=TrajectoryStatus.COMPLETED,
            final_output={"prediction": prediction},
            summary_stats={
                key: float(value)
                for key, value in (result.get("stats", {}) or {}).items()
                if isinstance(value, (int, float))
            },
            artifacts={"messages": result.get("messages", [])},
            metadata={"termination": str(result.get("termination", ""))},
        )

        for idx, message in enumerate(result.get("messages", []) or []):
            role = str(message.get("role", "")) if isinstance(message, dict) else ""
            content = str(message.get("content", "")) if isinstance(message, dict) else str(message)
            trajectory.steps.append(
                TrajectoryStep(
                    step_id=f"s{idx}",
                    step_type=_step_type_for(role, content),
                    timestamp_ms=now_ms + idx,
                    output={"role": role, "content": content},
                )
            )

        trajectory.steps.append(
            TrajectoryStep(
                step_id="final",
                step_type=TrajectoryStepType.FINAL,
                timestamp_ms=now_ms + len(trajectory.steps) + 1,
                output={"prediction": prediction},
            )
        )

        outcome = TaskOutcome(
            sample_id=agent_input.sample_id,
            task_type=agent_input.task_type,
            prediction=prediction,
            reference=reference,
            success=success,
            termination_reason=str(result.get("termination", "")),
            metrics=dict(trajectory.summary_stats),
            metadata=dict(agent_input.metadata),
        )
        return AgentRunResult(trajectory=trajectory, outcome=outcome)
