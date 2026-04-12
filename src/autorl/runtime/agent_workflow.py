from __future__ import annotations

from typing import Any

from autorl.observability import (
    build_rollout_metric_record,
    build_rollout_metrics,
    log_rollout_metrics,
)

from .factory import (
    build_agent_runtime,
    build_reward_strategy,
    build_runtime_context,
    build_task_adapter,
)


class UnifiedAgentWorkflow:
    """Canonical AReaL-native agent workflow entrypoint for this repo."""

    def __init__(
        self,
        task_adapter_path: str,
        agent_runtime_path: str,
        reward_strategy_path: str,
        execution_mode: str = "train",
        tool_gateway_mode: str = "local",
        tool_gateway_base_url: str | None = None,
        env_gateway_mode: str = "none",
        env_gateway_base_url: str | None = None,
        trace_dir: str | None = None,
        max_llm_calls_per_run: int = 100,
        max_tokens_per_trajectory: int = 32768,
        max_tokens_per_turn: int | None = None,
        judge_engine_addr: str | None = None,
    ) -> None:
        self.task_adapter = build_task_adapter(task_adapter_path)
        self.agent_runtime = build_agent_runtime(agent_runtime_path)
        self.reward_strategy = build_reward_strategy(reward_strategy_path)
        self.execution_mode = execution_mode
        self.tool_gateway_mode = tool_gateway_mode
        self.tool_gateway_base_url = tool_gateway_base_url
        self.env_gateway_mode = env_gateway_mode
        self.env_gateway_base_url = env_gateway_base_url
        self.trace_dir = trace_dir
        self.max_llm_calls_per_run = max_llm_calls_per_run
        self.max_tokens_per_trajectory = max_tokens_per_trajectory
        self.max_tokens_per_turn = max_tokens_per_turn
        self.judge_engine_addr = judge_engine_addr

    async def run(self, data: dict[str, Any], **extra_kwargs: Any) -> float | dict[str, float]:
        sample = self.task_adapter.validate_sample(data)
        agent_input = self.task_adapter.to_agent_input(sample)
        runtime_context = build_runtime_context(
            execution_mode=self.execution_mode,
            base_url=extra_kwargs.get("base_url"),
            api_key=extra_kwargs.get("api_key"),
            http_client=extra_kwargs.get("http_client"),
            tool_gateway_mode=self.tool_gateway_mode,
            tool_gateway_base_url=self.tool_gateway_base_url,
            env_gateway_mode=self.env_gateway_mode,
            env_gateway_base_url=self.env_gateway_base_url,
            trace_dir=self.trace_dir,
            max_llm_calls_per_run=self.max_llm_calls_per_run,
            max_tokens_per_trajectory=self.max_tokens_per_trajectory,
            max_tokens_per_turn=self.max_tokens_per_turn,
            judge_base_url=self.judge_engine_addr,
            metadata={
                "sample_id": sample.sample_id,
                "task_type": sample.task_type,
            },
        )
        result = await self.agent_runtime.run(agent_input, runtime_context)
        outcome = self.task_adapter.to_task_outcome(sample, result.trajectory)
        reward = await self.reward_strategy.compute(
            sample,
            result.trajectory,
            outcome,
            runtime_context,
        )
        rollout_metrics = build_rollout_metrics(
            trajectory=result.trajectory,
            outcome=outcome,
            reward=reward,
        )
        log_rollout_metrics(rollout_metrics)
        trace_collector = runtime_context.trace_collector
        if trace_collector is not None:
            trace_collector.emit(name="trajectory", payload=result.trajectory)
            trace_collector.emit(name="outcome", payload=outcome)
            trace_collector.emit(name="reward", payload={"sample_id": sample.sample_id, "reward": reward})
            trace_collector.emit(
                name="episode_metrics",
                payload=build_rollout_metric_record(
                    sample_id=sample.sample_id,
                    task_type=sample.task_type,
                    trajectory=result.trajectory,
                    outcome=outcome,
                    reward=reward,
                    metrics=rollout_metrics,
                ),
            )
        return reward
