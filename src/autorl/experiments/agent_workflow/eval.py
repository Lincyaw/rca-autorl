from __future__ import annotations

from pathlib import Path

from areal.api import ModelAllocation  # noqa: E402
from areal.api.cli_args import SGLangConfig, load_expr_config, vLLMConfig  # noqa: E402
from areal.engine import RemoteSGLangEngine, RemotevLLMEngine  # noqa: E402
from areal.infra import LocalScheduler, RayScheduler, SlurmScheduler  # noqa: E402
from areal.utils import logging, seeding  # noqa: E402
from areal.utils.dataloader import create_dataloader  # noqa: E402
from areal.utils.printing import tabulate_stats  # noqa: E402

from autorl.data import build_train_dataset, build_valid_dataset  # noqa: E402
from autorl.observability import setup_metrics  # noqa: E402

from .configs import (  # noqa: E402
    DEFAULT_EVAL_WORKFLOW,
    DEFAULT_TRAIN_WORKFLOW,
    EXPR_CONFIG_CLS,
)
from .train import assert_agent_scheduler_supported, build_workflow_kwargs  # noqa: E402

logger = logging.getLogger("autorl-agent-eval")


def _make_scheduler(config):
    scheduler_type = config.scheduler.type
    if scheduler_type == "local":
        return LocalScheduler(exp_config=config)
    if scheduler_type == "ray":
        return RayScheduler(exp_config=config)
    if scheduler_type == "slurm":
        return SlurmScheduler(exp_config=config)
    raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def _build_eval_dataloader(config):
    valid_dataset = build_valid_dataset(config)
    dataset_cfg = config.valid_dataset

    if valid_dataset is None or dataset_cfg is None:
        if not bool(getattr(config, "allow_train_fallback_for_eval", False)):
            raise ValueError(
                "valid_dataset must be configured for eval; "
                "set allow_train_fallback_for_eval=true to explicitly allow fallback"
            )
        valid_dataset = build_train_dataset(config)
        dataset_cfg = config.train_dataset

    return create_dataloader(
        valid_dataset,
        rank=0,
        world_size=1,
        dataset_config=dataset_cfg,
    )


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, EXPR_CONFIG_CLS)
    assert_agent_scheduler_supported(config)
    logging.setup_file_logging(f"{config.cluster.fileroot}/eval.log")
    seeding.set_random_seed(config.seed, key="eval")

    scheduler = _make_scheduler(config)
    rollout_alloc = ModelAllocation.from_str(config.rollout.backend, name="rollout")

    config.rollout.max_head_offpolicyness = int(1e12)
    if rollout_alloc.backend == "sglang":
        controller = RemoteSGLangEngine.as_controller(config.rollout, scheduler)
        server_args = SGLangConfig.build_args(
            sglang_config=config.sglang,
            tp_size=rollout_alloc.parallel.tp_size,
            base_gpu_id=0,
        )
    elif rollout_alloc.backend == "vllm":
        controller = RemotevLLMEngine.as_controller(config.rollout, scheduler)
        server_args = vLLMConfig.build_args(
            vllm_config=config.vllm,
            tp_size=rollout_alloc.parallel.tp_size,
            pp_size=rollout_alloc.parallel.pp_size,
        )
    else:
        raise ValueError(f"Unsupported rollout backend: {rollout_alloc.backend}")

    dataloader = _build_eval_dataloader(config)
    workflow = (
        getattr(config, "eval_workflow", None)
        or DEFAULT_EVAL_WORKFLOW
        or getattr(config, "workflow", DEFAULT_TRAIN_WORKFLOW)
    )
    workflow_kwargs = build_workflow_kwargs(config, execution_mode="eval")

    fileroot = getattr(getattr(config, "cluster", None), "fileroot", None)
    metadata_dir = Path(fileroot) / "autorl" / "metadata" if fileroot else Path("artifacts/metadata")
    try:
        setup_metrics(
            run_name=(
                f"{getattr(config, 'experiment_name', 'autorl')}-"
                f"{getattr(config, 'trial_name', 'trial')}-eval"
            ),
            phase="eval",
            metadata_dir=metadata_dir,
            extra_tags={
                "experiment_name": str(getattr(config, "experiment_name", "")),
                "trial_name": str(getattr(config, "trial_name", "")),
                "workflow": str(workflow),
                "task_adapter": str(getattr(config, "task_adapter_path", "")),
                "agent_runtime": str(getattr(config, "agent_runtime_path", "")),
                "reward_strategy": str(getattr(config, "reward_strategy_path", "")),
                "valid_dataset": str(getattr(getattr(config, "valid_dataset", None), "path", "")),
                "allow_train_fallback_for_eval": str(
                    bool(getattr(config, "allow_train_fallback_for_eval", False))
                ),
            },
        )
    except Exception as err:
        logger.warning("Failed to write eval metrics metadata: %s", err)

    submitted = 0
    try:
        controller.initialize(role="eval-rollout", server_args=server_args)
        controller.start_proxy()
        for batch in dataloader:
            for item in batch:
                controller.submit(
                    item,
                    workflow=workflow,
                    workflow_kwargs=workflow_kwargs,
                    group_size=config.gconfig.n_samples,
                    is_eval=True,
                )
                submitted += 1

        controller.wait(submitted, timeout=None)
        eval_stats = controller.export_stats()
        logger.info("Evaluation Results: %s", tabulate_stats(eval_stats))
    finally:
        controller.destroy()


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
