from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from autorl.contracts import RuntimeContext, RuntimeLimits
from autorl.gateways import NullEnvGateway, ToolEnvGateway
from autorl.rewards.base import RewardStrategy
from autorl.runtime.trace_sink import JsonlTraceSink
from autorl.tasks.base import TaskAdapter

from .base import AgentRuntime


def import_from_path(path: str) -> Any:
    module_name, _, attr_name = path.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(f"invalid import path: {path!r}")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr_name)
    except AttributeError as err:
        raise ValueError(f"{path!r} does not resolve to an attribute") from err


def instantiate_from_path(path: str, **kwargs: Any) -> Any:
    obj = import_from_path(path)
    if isinstance(obj, type):
        return obj(**kwargs)
    if kwargs:
        raise ValueError(f"cannot pass kwargs to non-class import path: {path!r}")
    return obj


def build_task_adapter(path: str) -> TaskAdapter:
    adapter = instantiate_from_path(path)
    if not isinstance(adapter, TaskAdapter):
        raise TypeError(f"task adapter must implement TaskAdapter, got {type(adapter).__name__}")
    return adapter


def build_agent_runtime(path: str, **kwargs: Any) -> AgentRuntime:
    runtime = instantiate_from_path(path, **kwargs)
    if not isinstance(runtime, AgentRuntime):
        raise TypeError(f"agent runtime must implement AgentRuntime, got {type(runtime).__name__}")
    return runtime


def build_reward_strategy(path: str) -> RewardStrategy:
    strategy = instantiate_from_path(path)
    if not isinstance(strategy, RewardStrategy):
        raise TypeError(f"reward strategy must implement RewardStrategy, got {type(strategy).__name__}")
    return strategy


def build_tool_gateway(mode: str, base_url: str | None):
    if mode not in {"local", "http"}:
        raise ValueError(f"unsupported tool gateway mode: {mode}")
    return ToolEnvGateway(base_url=base_url if mode == "http" else None)


def build_env_gateway(mode: str, base_url: str | None):
    if mode == "none":
        return NullEnvGateway()
    if mode in {"local", "http"}:
        metadata = {"base_url": base_url} if base_url else {}
        gateway = NullEnvGateway(name=f"{mode}_env")
        gateway._last_observation = metadata or None  # noqa: SLF001
        return gateway
    raise ValueError(f"unsupported env gateway mode: {mode}")


def build_trace_sink(trace_dir: str | Path | None) -> JsonlTraceSink | None:
    if trace_dir in (None, ""):
        return None
    return JsonlTraceSink(trace_dir)


def build_runtime_context(
    *,
    execution_mode: str,
    base_url: str | None,
    api_key: str | None,
    http_client: Any | None,
    tool_gateway_mode: str,
    tool_gateway_base_url: str | None,
    env_gateway_mode: str,
    env_gateway_base_url: str | None,
    trace_dir: str | Path | None,
    max_llm_calls_per_run: int,
    max_tokens_per_trajectory: int,
    max_tokens_per_turn: int | None,
    judge_base_url: str | None,
    metadata: dict[str, Any] | None = None,
) -> RuntimeContext:
    limits = RuntimeLimits(
        max_steps=max_llm_calls_per_run,
        max_llm_calls=max_llm_calls_per_run,
        max_tokens=max_tokens_per_trajectory,
        timeout_seconds=None,
    )
    runtime_metadata = dict(metadata or {})
    if max_tokens_per_turn is not None:
        runtime_metadata["max_tokens_per_turn"] = max_tokens_per_turn
    if judge_base_url:
        runtime_metadata["judge_base_url"] = judge_base_url
    return RuntimeContext(
        mode=execution_mode,
        model_endpoint=base_url,
        api_key=api_key,
        http_client=http_client,
        tool_gateway=build_tool_gateway(tool_gateway_mode, tool_gateway_base_url),
        env_gateway=build_env_gateway(env_gateway_mode, env_gateway_base_url),
        trace_collector=build_trace_sink(trace_dir),
        limits=limits,
        metadata=runtime_metadata,
    )
