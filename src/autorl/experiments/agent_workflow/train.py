from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from areal import PPOTrainer  # noqa: E402
from areal.api import WorkflowLike  # noqa: E402
from areal.api.cli_args import load_expr_config  # noqa: E402

from autorl.data import build_train_dataset, build_valid_dataset  # noqa: E402
from autorl.observability import setup_metrics  # noqa: E402

from .configs import (  # noqa: E402
    DEFAULT_EVAL_WORKFLOW,
    DEFAULT_TRAIN_WORKFLOW,
    EXPR_CONFIG_CLS,
)


def build_trace_dir(config, phase: str) -> str | None:
    configured = getattr(config, "trace_dir", None)
    if configured:
        return str(Path(configured) / phase)
    fileroot = getattr(getattr(config, "cluster", None), "fileroot", None)
    if not fileroot:
        return str(Path("artifacts/trajectories") / phase)
    return str(Path(fileroot) / "autorl" / "trajectories" / phase)


def assert_agent_scheduler_supported(config) -> None:
    scheduler_type = getattr(getattr(config, "scheduler", None), "type", None)
    if scheduler_type == "ray":
        raise ValueError(
            "AReaL proxy-backed agent workflows are only supported on local/slurm schedulers; "
            "set scheduler.type=local or scheduler.type=slurm."
        )


def build_workflow_kwargs(
    config,
    *,
    execution_mode: str,
    judge_engine_addr: str | None = None,
) -> dict:
    resolved_judge_addr = judge_engine_addr or getattr(config, "judge_engine_addr", None)
    return {
        "task_adapter_path": getattr(config, "task_adapter_path"),
        "agent_runtime_path": getattr(config, "agent_runtime_path"),
        "reward_strategy_path": getattr(config, "reward_strategy_path"),
        "execution_mode": execution_mode,
        "tool_gateway_mode": getattr(config, "tool_gateway_mode", "local"),
        "tool_gateway_base_url": getattr(config, "tool_gateway_base_url", None),
        "env_gateway_mode": getattr(config, "env_gateway_mode", "none"),
        "env_gateway_base_url": getattr(config, "env_gateway_base_url", None),
        "trace_dir": build_trace_dir(config, execution_mode),
        "max_llm_calls_per_run": getattr(config, "max_llm_calls_per_run", 100),
        "max_tokens_per_trajectory": getattr(config, "max_tokens_per_trajectory", 32768),
        "max_tokens_per_turn": getattr(config, "max_tokens_per_turn", None),
        "judge_engine_addr": resolved_judge_addr,
    }


@contextmanager
def maybe_launch_judge_engine(config, trainer):
    """Launch optional judge SGLang engine when config.judge_engine is provided."""

    judge_engine_cfg = getattr(config, "judge_engine", None)
    if judge_engine_cfg is None:
        yield getattr(config, "judge_engine_addr", None)
        return

    judge_backend = getattr(judge_engine_cfg, "backend", "")
    if not judge_backend:
        yield getattr(config, "judge_engine_addr", None)
        return

    from areal.api import ModelAllocation
    from areal.api.cli_args import SGLangConfig
    from areal.engine.sglang_remote import RemoteSGLangEngine

    judge_alloc = ModelAllocation.from_str(judge_backend)
    if judge_alloc.backend != "sglang":
        raise ValueError(f"judge_engine backend must be sglang, got {judge_alloc.backend}")

    server_args = SGLangConfig.build_args(
        sglang_config=config.sglang,
        tp_size=judge_alloc.parallel.tp_size,
        base_gpu_id=0,
    )
    judge_engine_cfg.max_head_offpolicyness = int(1e12)

    controller = RemoteSGLangEngine.as_controller(judge_engine_cfg, trainer.scheduler)
    try:
        controller.initialize(role="judge_engine", server_args=server_args)
        controller.start_proxy()
        controller.start_proxy_gateway()
        yield controller.proxy_gateway_addr
    finally:
        controller.destroy()


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, EXPR_CONFIG_CLS)
    assert_agent_scheduler_supported(config)

    train_dataset = build_train_dataset(config)
    valid_dataset = build_valid_dataset(config)

    train_workflow: WorkflowLike = getattr(config, "workflow", None) or DEFAULT_TRAIN_WORKFLOW
    eval_workflow = getattr(config, "eval_workflow", None) or DEFAULT_EVAL_WORKFLOW

    fileroot = getattr(getattr(config, "cluster", None), "fileroot", None)
    metadata_dir = Path(fileroot) / "autorl" / "metadata" if fileroot else Path("artifacts/metadata")
    try:
        setup_metrics(
            run_name=(
                f"{getattr(config, 'experiment_name', 'autorl')}-"
                f"{getattr(config, 'trial_name', 'trial')}-train"
            ),
            phase="train",
            metadata_dir=metadata_dir,
            extra_tags={
                "experiment_name": str(getattr(config, "experiment_name", "")),
                "trial_name": str(getattr(config, "trial_name", "")),
                "workflow": str(train_workflow),
                "eval_workflow": str(eval_workflow),
                "task_adapter": str(getattr(config, "task_adapter_path", "")),
                "agent_runtime": str(getattr(config, "agent_runtime_path", "")),
                "reward_strategy": str(getattr(config, "reward_strategy_path", "")),
                "train_dataset": str(getattr(config.train_dataset, "path", "")),
                "valid_dataset": str(getattr(getattr(config, "valid_dataset", None), "path", "")),
            },
        )
    except Exception:
        pass

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        with maybe_launch_judge_engine(config, trainer) as judge_engine_addr:
            trainer.train(
                workflow=train_workflow,
                workflow_kwargs=build_workflow_kwargs(
                    config,
                    execution_mode="train",
                    judge_engine_addr=judge_engine_addr,
                ),
                eval_workflow=eval_workflow,
                eval_workflow_kwargs=build_workflow_kwargs(
                    config,
                    execution_mode="eval",
                    judge_engine_addr=judge_engine_addr,
                ),
            )


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
