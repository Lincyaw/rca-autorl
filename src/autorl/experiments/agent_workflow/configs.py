from __future__ import annotations

from dataclasses import dataclass, field

from areal.api.cli_args import InferenceEngineConfig, PPOConfig


@dataclass
class AgentWorkflowConfig(PPOConfig):
    """PPO config extension for framework-agnostic agent workflows."""

    workflow: str | None = field(
        default=None,
        metadata={"help": "Optional workflow override. Defaults to the repo canonical workflow."},
    )
    eval_workflow: str | None = field(
        default=None,
        metadata={"help": "Optional eval workflow override."},
    )
    infer_workflow: str | None = field(
        default=None,
        metadata={"help": "Optional inference workflow override."},
    )
    task_adapter_path: str = field(
        default="autorl.tasks.search.SearchTaskAdapter",
        metadata={"help": "Import path for the task adapter."},
    )
    agent_runtime_path: str = field(
        default="autorl.tasks.search_runtime.SearchAgentRuntime",
        metadata={"help": "Import path for the agent runtime implementation."},
    )
    reward_strategy_path: str = field(
        default="autorl.tasks.search_reward.SearchRewardStrategy",
        metadata={"help": "Import path for the reward strategy."},
    )
    tool_gateway_mode: str = field(
        default="local",
        metadata={"help": "Tool gateway mode: local or http."},
    )
    tool_gateway_base_url: str | None = field(
        default=None,
        metadata={"help": "Optional HTTP base URL for remote tool execution."},
    )
    env_gateway_mode: str = field(
        default="none",
        metadata={"help": "Environment gateway mode: none, local, or http."},
    )
    env_gateway_base_url: str | None = field(
        default=None,
        metadata={"help": "Optional HTTP base URL for a remote environment gateway."},
    )
    trace_dir: str | None = field(
        default=None,
        metadata={"help": "Optional directory for canonical trajectory JSONL output."},
    )
    train_manifest_path: str | None = field(
        default=None,
        metadata={"help": "Optional manifest path for train dataset (.json/.jsonl)."},
    )
    valid_manifest_path: str | None = field(
        default=None,
        metadata={"help": "Optional manifest path for valid dataset (.json/.jsonl)."},
    )
    train_dataset_revision: str | None = field(
        default=None,
        metadata={"help": "Optional HuggingFace revision for train dataset/path shorthand."},
    )
    valid_dataset_revision: str | None = field(
        default=None,
        metadata={"help": "Optional HuggingFace revision for valid dataset/path shorthand."},
    )
    max_llm_calls_per_run: int = field(
        default=100,
        metadata={"help": "Maximum number of model calls per trajectory."},
    )
    max_tokens_per_trajectory: int = field(
        default=32768,
        metadata={"help": "Maximum total tokens budget per trajectory."},
    )
    max_tokens_per_turn: int | None = field(
        default=None,
        metadata={"help": "Optional maximum completion tokens per model call."},
    )
    judge_engine: InferenceEngineConfig | None = field(
        default=None,
        metadata={"help": "Optional judge engine config."},
    )
    judge_engine_addr: str | None = field(
        default=None,
        metadata={"help": "Optional external judge engine URL."},
    )
    infer_max_items: int = field(
        default=1,
        metadata={"help": "Maximum items for infer entrypoint."},
    )
    allow_train_fallback_for_eval: bool = field(
        default=False,
        metadata={
            "help": (
                "Allow eval.py to reuse train_dataset when valid_dataset is not set. "
                "Default is false to avoid train/eval leakage."
            )
        },
    )
    allow_train_fallback_for_infer: bool = field(
        default=False,
        metadata={
            "help": (
                "Allow infer.py to reuse train_dataset when valid_dataset is not set. "
                "Default is false to avoid accidental train-data inference reports."
            )
        },
    )


EXPR_CONFIG_CLS = AgentWorkflowConfig

DEFAULT_TRAIN_WORKFLOW = "autorl.workflows.agent.AReaLNativeAgentWorkflow"
DEFAULT_EVAL_WORKFLOW = DEFAULT_TRAIN_WORKFLOW
