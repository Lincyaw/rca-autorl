from __future__ import annotations

import datetime
import time
from typing import Any
from uuid import uuid4

import json5
from openai import AsyncOpenAI

from autorl.agents.prompt import SYSTEM_PROMPT
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
from autorl.runtime.base import AgentRuntime

_TOOL_STOP_TOKENS = ["\n<tool_response>", "<tool_response>"]


class SearchAgentRuntime(AgentRuntime):
    """Example search-oriented runtime built on the canonical contracts."""

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    async def run(
        self,
        agent_input: AgentInput,
        runtime_context: RuntimeContext,
    ) -> AgentRunResult:
        if not runtime_context.model_endpoint:
            raise ValueError("SearchAgentRuntime requires RuntimeContext.model_endpoint")
        client = AsyncOpenAI(
            base_url=runtime_context.model_endpoint,
            api_key=runtime_context.api_key or "EMPTY",
            http_client=runtime_context.http_client,
            max_retries=0,
        )
        tool_gateway = runtime_context.tool_gateway
        if tool_gateway is None:
            raise ValueError("SearchAgentRuntime requires a tool gateway")

        question = str(agent_input.context.get("question", ""))
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT + datetime.date.today().strftime("%Y-%m-%d"),
            },
            *agent_input.messages,
        ]
        max_llm_calls = runtime_context.limits.max_llm_calls or 100
        max_tokens_per_turn = int(runtime_context.metadata.get("max_tokens_per_turn", 1024))
        max_total_tokens = runtime_context.limits.max_tokens or 32768
        total_tokens = 0
        completion_ids: list[str] = []
        steps: list[TrajectoryStep] = []
        stats = {"turns": 0.0, "num_search": 0.0, "num_access": 0.0}
        final_content = ""
        termination = "answer_not_found"
        status = TrajectoryStatus.COMPLETED

        for turn in range(1, max_llm_calls + 1):
            stats["turns"] += 1.0
            request_step_id = f"llm-request-{turn}"
            response_step_id = f"llm-response-{turn}"
            steps.append(
                TrajectoryStep(
                    step_id=request_step_id,
                    step_type=TrajectoryStepType.LLM_REQUEST,
                    timestamp_ms=_now_ms(),
                    input={"messages": messages[-4:]},
                    metadata={"turn": turn},
                )
            )
            completion = await client.chat.completions.create(
                model="default",
                messages=messages,
                temperature=self.temperature,
                stop=_TOOL_STOP_TOKENS,
                max_completion_tokens=max_tokens_per_turn,
            )
            usage = TokenUsage(
                prompt_tokens=getattr(completion.usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(completion.usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(completion.usage, "total_tokens", 0) or 0,
            )
            total_tokens += usage.total_tokens
            message = completion.choices[0].message
            content = message.content or ""
            final_content = content
            completion_ids.append(completion.id)
            steps.append(
                TrajectoryStep(
                    step_id=response_step_id,
                    step_type=TrajectoryStepType.LLM_RESPONSE,
                    timestamp_ms=_now_ms(),
                    output={"content": content},
                    usage=usage,
                    call_id=completion.id,
                    parent_call_id=request_step_id,
                    metadata={"turn": turn},
                )
            )
            messages.append(message.model_dump(exclude_none=True))

            tool_call = _extract_tool_call(content)
            if tool_call is not None:
                tool_name = str(tool_call.get("name", ""))
                tool_args = tool_call.get("arguments", {})
                tool_call_id = f"tool-call-{turn}"
                steps.append(
                    TrajectoryStep(
                        step_id=tool_call_id,
                        step_type=TrajectoryStepType.TOOL_CALL,
                        timestamp_ms=_now_ms(),
                        input={"name": tool_name, "arguments": tool_args},
                        parent_call_id=completion.id,
                        metadata={"turn": turn},
                    )
                )
                tool_result = await tool_gateway.execute(tool_name, tool_args if isinstance(tool_args, dict) else {})
                if tool_name == "search":
                    stats["num_search"] += 1.0
                if tool_name == "visit":
                    stats["num_access"] += 1.0
                tool_text = str(tool_result.result if tool_result.ok else f"Error: {tool_result.error or 'tool execution failed'}")
                steps.append(
                    TrajectoryStep(
                        step_id=f"tool-result-{turn}",
                        step_type=TrajectoryStepType.TOOL_RESULT,
                        timestamp_ms=_now_ms(),
                        output={"name": tool_name, "result": tool_text},
                        parent_call_id=tool_call_id,
                        metadata={"turn": turn, "ok": tool_result.ok},
                    )
                )
                messages.append({"role": "user", "content": f"<tool_response>\n{tool_text}\n</tool_response>"})

            if _has_answer(content):
                termination = "answer"
                break

            if total_tokens >= max_total_tokens or turn == max_llm_calls:
                termination = "token_limit" if total_tokens >= max_total_tokens else "llm_call_limit"
                status = TrajectoryStatus.TRUNCATED
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Stop tool use and provide the best final answer in the format "
                            "<answer>...</answer>."
                        ),
                    }
                )
                forced_completion = await client.chat.completions.create(
                    model="default",
                    messages=messages,
                    temperature=self.temperature,
                    max_completion_tokens=max_tokens_per_turn,
                )
                forced_usage = TokenUsage(
                    prompt_tokens=getattr(forced_completion.usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(forced_completion.usage, "completion_tokens", 0)
                    or 0,
                    total_tokens=getattr(forced_completion.usage, "total_tokens", 0) or 0,
                )
                forced_message = forced_completion.choices[0].message
                final_content = forced_message.content or final_content
                completion_ids.append(forced_completion.id)
                steps.append(
                    TrajectoryStep(
                        step_id=f"llm-response-{turn}-forced",
                        step_type=TrajectoryStepType.LLM_RESPONSE,
                        timestamp_ms=_now_ms(),
                        output={"content": final_content, "forced": True},
                        usage=forced_usage,
                        call_id=forced_completion.id,
                        parent_call_id=request_step_id,
                        metadata={"turn": turn, "forced": True},
                    )
                )
                break

        prediction = _extract_answer(final_content) or final_content
        if termination != "answer" and status == TrajectoryStatus.COMPLETED:
            status = TrajectoryStatus.FAILED
        trajectory = Trajectory(
            trajectory_id=uuid4().hex,
            sample_id=agent_input.sample_id,
            task_type=agent_input.task_type,
            status=status,
            final_output={
                "prediction": prediction,
                "raw_content": final_content,
                "termination": termination,
                "completion_ids": completion_ids,
            },
            steps=steps,
            summary_stats=stats,
            metadata={"question": question, **agent_input.metadata},
        )
        outcome = TaskOutcome(
            sample_id=agent_input.sample_id,
            task_type=agent_input.task_type,
            prediction=prediction,
            reference=None,
            success=None,
            termination_reason=termination,
            metrics=stats,
            metadata=dict(agent_input.metadata),
        )
        return AgentRunResult(trajectory=trajectory, outcome=outcome)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _extract_tool_call(content: str) -> dict[str, Any] | None:
    if "<tool_call>" not in content or "</tool_call>" not in content:
        return None
    raw = content.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0]
    try:
        parsed = json5.loads(raw)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _has_answer(content: str) -> bool:
    return "<answer>" in content and "</answer>" in content


def _extract_answer(content: str) -> str:
    if not _has_answer(content):
        return ""
    return content.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
