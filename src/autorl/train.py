"""AReaL GRPO/PPO launcher for the DeepSeek Harness RCA workflow."""

from __future__ import annotations

import sys
from dataclasses import asdict
from typing import Any

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from datasets import Dataset

from autorl.config import RCAPPOConfig
from autorl.data.samples import read_jsonl


def validate_advantage_config(config: RCAPPOConfig) -> None:
    """Fail fast unless AReaL v2 leaves the advantage to the workflow.

    The centring of spec §3.2 (RLOO, GRPO, ReMax) is computed per trajectory in
    `DshWorkflow.rescore_group`, together with the fork advantages of §4, and
    written to every row of every complete group. So the framework must add no
    baseline of its own, must export one row per request, and must drop a
    group the hook refused rather than train it on raw outcomes.
    """
    errors: list[str] = []
    if int(config.gconfig.n_samples) < 2:
        errors.append("gconfig.n_samples must be >= 2")
    if config.gconfig.reward_normalization:
        errors.append("gconfig.reward_normalization must be false")
    if not config.gconfig.drop_incomplete_group:
        errors.append("gconfig.drop_incomplete_group must be true: a partial group is not rescored")
    if config.rollout.agent.export_style != "individual":
        errors.append("rollout.agent.export_style must be individual: one row per completion")
    if config.actor.reward_norm is not None:
        errors.append("actor.reward_norm must be null: the workflow centres the group itself")
    if config.actor.adv_norm is not None:
        errors.append("actor.adv_norm must be null")
    if float(config.actor.discount) != 1.0 or float(config.actor.gae_lambda) != 1.0:
        errors.append("actor.discount and actor.gae_lambda must both equal 1")
    if config.critic is not None:
        errors.append("critic must be null for critic-free training")
    if errors:
        raise ValueError("invalid RCA advantage config: " + "; ".join(errors))


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, RCAPPOConfig)
    validate_advantage_config(config)
    train_dataset = Dataset.from_list(read_jsonl(config.train_dataset.path))
    valid_dataset = (
        Dataset.from_list(read_jsonl(config.valid_dataset.path))
        if config.valid_dataset is not None
        else None
    )
    # The served model name and the generation limits are AReaL's to declare,
    # not ours to duplicate: rollout.model is the name the gateway routes on,
    # gconfig.max_new_tokens is what the harness sends as the request's
    # `max_tokens`, sglang.context_length is the window compaction must
    # trigger below, and gconfig.temperature is what the harness samples at.
    econfig: dict[str, Any] = {
        **asdict(config.econfig),
        "model": str(config.rollout.model),
        "max_tokens": int(config.gconfig.max_new_tokens),
        "context_window": int(config.sglang.context_length or config.gconfig.max_tokens),
        "temperature": float(config.gconfig.temperature),
    }
    workflow_kwargs = {"econfig": econfig}

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="autorl.agent.DshWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="autorl.agent.DshWorkflow" if valid_dataset is not None else None,
            eval_workflow_kwargs=workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
