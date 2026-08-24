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
- `src/autorl/algorithm.py`: reward scalarization, dynamic group filtering, and
  AReaL v2 RLOO configuration validation.
- `src/autorl/verifier.py`: converts canonical FPG evaluation into reward signals.
- `src/autorl/train.py`: JSONL loading plus `PPOTrainer` launch.
- `src/autorl/train_sft.py`: `SFTTrainer` launch for distilled AgentM trajectories.
- `src/autorl/data/sft.py`: chat-template rendering and assistant-only loss masks.
- `configs/train/agentm_rca_smoke.yaml`: one-GPU RCA RL smoke config.
- `configs/sft/`: SFT configs for distilled trajectories.

Generic task/runtime/gateway contracts are deliberately absent. If another agent is
trained, add another direct workflow like AReaL does for SWE instead of introducing a
framework inside this repository.

## Setup

The checkout uses AReaL as a submodule and AgentM as a pinned SDK package:

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
`AgentMAgent`, and returns one scalar reward in the format expected by AReaL v2.

When a case contains `causal_graph_verified.json`, reward comes from
`fpg.compare_model_to_ground_truth`. The reward follows the Notes specification:

```text
terminal = cause_weight × (+correct_reward or -incorrect_penalty)
         + attribution_weight × attribution_score
         - violation_weight × violation_count
return   = terminal - investigation_cost
```

The default weights satisfy the reward-safety constraint
`cause_weight × incorrect_penalty > attribution_weight`. AReaL v2 performs
trajectory-level RLOO using group leave-one-out reward centering, no standard-deviation
normalization, and undiscounted `gamma=lambda=1`. Tied non-positive total-return groups
are rejected and refilled; tied successful and contrastive groups are retained.

The current FPG verifier uses exact root-subject matching for the binary cause score and
the mean of subject recall and soft subject-edge recall as the attribution proxy. The
algorithm module also defines the exact anomaly-account formula, including false-dismissal
penalties; wiring that path requires the dataset generator to emit verified per-anomaly
outcomes. Tool-call cost is measured today. Token, redundancy, invalid-action,
declaration, and violation costs activate when AgentM exposes their counts in result
metadata.

Selective same-state forking remains out of scope until AgentM exposes exact
snapshot-and-resume at the selected pre-action state.

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
