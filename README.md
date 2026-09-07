# rca-autorl

Thin AReaL integration for training the DeepSeek Harness RCA agent.

The repository intentionally follows AReaL's SWE example: DeepSeek Harness (`dsh`) owns
the agent loop, tools, and trajectory persistence; this repository only connects that
agent to AReaL's rollout proxy and trainer.

```text
AReaL PPO/GRPO trainer
        │ OpenAI-compatible rollout proxy
        ▼
DshWorkflow ──▶ DeepSeekHarness SDK ──▶ dsh runtime subprocess
        ▲                                   │  sdk-minimal + agent/rca-harness
        │                                   │  tools: sql (DuckDB over the snapshot)
        │                                   ▼         submit_result
        └── placeholder 0.0 reward ◀── fault propagation graph
```

## What lives here

- `src/autorl/agent.py`: direct AReaL workflow around the public `DeepSeekHarness` SDK.
- `src/autorl/harness.py`: installs the RCA harness bundle into a `dsh` profile.
- `agent/`: the RCA harness — the `dsh` bundle that replaces the shell with a
  bounded DuckDB `sql` tool over the snapshot, an investigation notebook, and
  the terminal `submit_result`, plus the RCA scenario patch. Its
  [README](agent/README.md) is the design.
- `src/autorl/algorithm.py`: validates the AReaL v2 RLOO configuration.
- `src/autorl/train.py`: JSONL loading plus `PPOTrainer` launch.
- `src/autorl/train_sft.py`: `SFTTrainer` launch for distilled RCA trajectories.
- `src/autorl/data/sft.py`: chat-template rendering and assistant-only loss masks.
- `configs/train/dsh_rca_smoke.yaml`: one-GPU RCA RL smoke config.
- `configs/sft/`: SFT configs for distilled trajectories.

Generic task/runtime/gateway contracts are deliberately absent. If another agent is
trained, add another direct workflow like AReaL does for SWE instead of introducing a
framework inside this repository.

## Setup

The checkout uses AReaL as a submodule and DeepSeek Harness as a pinned SDK package:

```bash
git submodule update --init --recursive
UV_HTTP_TIMEOUT=300 uv sync --python 3.12
```

`deepseek-harness-sdk` pulls the matching `deepseek-harness-runtime-bin` wheel, so the
`dsh` runtime ships with the environment and needs no system Node.js. The workflow boots
the shipped `sdk-minimal` profile — the only one that accepts an arbitrary model id,
which is what the rollout proxy serves — with the `agent/rca-harness` bundle installed
into it and `agent/profiles/<scenario>.patch.yml` layered on top. The bundle disables
`sdk-minimal`'s shell and editor rows, so an episode's whole action space is `sql`,
`take_note`, and `submit_result`.

Install the bundle into the profile once, before any rollout starts:

```bash
python -m autorl.harness .runs/dsh-rca-smoke/dsh-home     # --reinstall after editing it
```

`dsh plugin` shells out to `pnpm`; a one-line shim (`exec corepack pnpm "$@"`) on `PATH`
is enough. `scripts/run_smoke.sh` runs this step for you. A rollout whose profile is
missing the bundle fails immediately with the command to run.

Each episode reuses that one `$DSH_HOME` (`econfig.dsh_home`, default `.runs/dsh-home`)
with a fresh session id.

## RL

The input is the processed RCA JSONL. Each row needs an incident (`question` or
`incident`) and either an absolute `data_dir` or a `datapack_name` resolvable below the
dataset root.

```bash
export RCA_DATASET_ROOT=/path/to/rca
./scripts/run_smoke.sh -p train_dataset.path=$RCA_DATASET_ROOT/data.jsonl \
  -p valid_dataset.path=$RCA_DATASET_ROOT/data.jsonl
```

For every rollout, AReaL passes a proxy URL and session API key to `DshWorkflow`. The
workflow launches one `dsh` runtime whose `DEEPSEEK_BASE_URL` and `DEEPSEEK_API_KEY` are
those values, runs the incident as a single session in the case's snapshot directory, and
returns one scalar reward in the format expected by AReaL v2. The runtime posts
OpenAI-compatible `POST {base_url}/chat/completions` requests, so the proxy records the
whole trajectory. Keep `econfig.max_tokens` below `sglang.context_length`: the harness
default per-request cap is 256k and the inference worker rejects it.

Reward is temporarily fixed at `0.0`. The verifier and reward design will be added later;
until then the training entry exercises rollout plumbing but produces no policy-gradient
signal. The episode's answer is already machine-readable:
`autorl.agent.submitted_result` returns the fault propagation graph the model passed to
`submit_result`, read back from the session log, and that is what the verifier will
score.

## SFT

SFT consumes JSON/JSONL exported from the distillation tooling. Prompt tokens and
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

An end-to-end smoke additionally requires a GPU and the RCA dataset.
