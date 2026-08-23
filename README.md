# rca-autorl

Thin AReaL integration for training the AgentM RCA agent.

The repository intentionally follows AReaL's SWE example: AgentM owns the agent loop,
tools, trajectory persistence, and output schema; this repository only connects that
agent to AReaL's rollout proxy and trainer.

```text
AReaL PPO/GRPO trainer
        │ OpenAI-compatible rollout proxy
        ▼
AgentMWorkflow ──▶ AgentMAgent ──▶ RCA tools and investigation
        ▲                              │
        └──────── verifier reward ◀────┘
```

## What lives here

- `src/autorl/agent.py`: direct AReaL workflow around `rca_eval.AgentMAgent`.
- `src/autorl/verifier.py`: canonical FPG reward, with a legacy label fallback.
- `src/autorl/train.py`: JSONL loading plus `PPOTrainer` launch.
- `src/autorl/train_sft.py`: `SFTTrainer` launch for distilled AgentM trajectories.
- `src/autorl/data/sft.py`: chat-template rendering and assistant-only loss masks.
- `configs/train/agentm_rca_smoke.yaml`: one-GPU RCA RL smoke config.
- `configs/sft/`: SFT configs for distilled trajectories.

Generic task/runtime/gateway contracts are deliberately absent. If another agent is
trained, add another direct workflow like AReaL does for SWE instead of introducing a
framework inside this repository.

## Setup

The checkout uses AReaL as a submodule and AgentM's published SDK package:

```bash
git submodule update --init --recursive
UV_HTTP_TIMEOUT=300 uv sync --python 3.12
```

`agentm[eval]` is consumed as a pinned SDK package. Its RCA scenario supplies the
`rca_eval.AgentMAgent` adapter and FPG grader used by rollout workers. It is not
overridden by a sibling checkout, so local development, CI, and the training host
resolve the same SDK contract.

## RL

The input is AgentM's processed RCA JSONL. Each row needs an incident (`question` or
`incident`) and either an absolute `data_dir` or a `datapack_name` resolvable below the
dataset root.

```bash
export AGENTM_RCA_DATASET_ROOT=/path/to/rca
./scripts/run_smoke.sh -p train_dataset.path=$AGENTM_RCA_DATASET_ROOT/data.jsonl \
  -p valid_dataset.path=$AGENTM_RCA_DATASET_ROOT/data.jsonl
```

For every rollout, AReaL passes a proxy URL and API key to `AgentMWorkflow`. The
workflow builds an explicit AgentM OpenAI provider from those values, runs
`AgentMAgent`, and returns the verifier metrics directly to `PPOTrainer`.

When a case contains `causal_graph_verified.json`, reward comes from
`fpg.compare_model_to_ground_truth`. Older datasets fall back to the previous
`0.7 * service_hit + 0.3 * fault_kind_hit` score.

## SFT

SFT consumes JSON/JSONL exported from AgentM's distillation tooling. Prompt tokens and
tool responses are masked; assistant reasoning and tool calls are supervised.

```bash
./scripts/run_sft_smoke.sh
```

Override the dataset or model with normal AReaL config patches:

```bash
./scripts/run_sft_smoke.sh \
  -p train_dataset.path=/path/to/extractor.jsonl \
  -p actor.path=/path/to/model
```

## Validation

```bash
uv sync --dev
uv run pre-commit install
./scripts/check.sh
```

The commit hook and GitHub Actions both run Ruff linting, Ruff formatting checks, and
mypy. Unit tests remain a separate command because some SFT coverage uses a locally
cached model tokenizer:

```bash
python -m unittest discover -s tests
```

An end-to-end smoke additionally requires a GPU, the RCA dataset, and working AgentM
scenario dependencies.
