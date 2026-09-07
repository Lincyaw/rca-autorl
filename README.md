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
- `src/autorl/harness.py`: composes a launch — bundle install, scenario layer,
  model route, and the one `dsh` runtime construction both paths use.
- `agent/`: the RCA harness — the `dsh` bundle that replaces the shell with a
  bounded DuckDB `sql` tool over the snapshot, an investigation notebook, and
  the terminal `submit_result`, plus the RCA scenario patch and the gateway
  model route. Its
  [README](agent/README.md) is the design.
- `src/autorl/algorithm.py`: validates the AReaL v2 RLOO configuration.
- `src/autorl/train.py`: JSONL loading plus `PPOTrainer` launch.
- `src/autorl/train_sft.py`: `SFTTrainer` launch for distilled RCA trajectories.
- `src/autorl/data/sft.py`: chat-template rendering and assistant-only loss masks.
- `src/autorl/data/export.py`: turns `dsh` session logs into SFT rows.
- `src/autorl/data/collect.py`: runs cases against a teacher endpoint to fill a
  Harness home with the sessions SFT trains on.
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

For every rollout, AReaL passes a proxy URL and session API key to `DshWorkflow`.
The workflow declares that endpoint as a model route, launches one `dsh` runtime for
the incident in the case's snapshot directory, and returns one scalar reward in the
format expected by AReaL v2. The proxy sees OpenAI-compatible
`POST {base_url}/chat/completions` requests, so it records the whole trajectory.

The route matters. `sdk-minimal` registers only the shipped `deepseek-official`
adapter, which carries a streamed tool call's id and name across deltas with an
`!== undefined` guard. sglang writes both as an explicit `null` on every argument
delta (`serving_chat.py`: "Subsequent chunks: null ID and name for argument deltas",
serialized without `exclude_none`) and AReaL's gateway forwards the stream verbatim,
so the guard passes and the id and name the first delta established are erased. The
call then fails as `Error: unknown tool ""` with its arguments intact — and the model
retries, forever: against a replay of that stream the shipped adapter produced 4530
calls, every name empty, not one submission. `autorl.harness.model_route` declares
the endpoint as an `llm-pi-ai` route instead, whose `openai-completions` protocol is
the OpenAI dialect proper; the same replay then yields `sql, sql, submit_result`. SFT
collection composes through the same function, so the two paths cannot drift.

Generation limits stay AReaL's: `rollout.model` is the served name the route declares,
`gconfig.max_new_tokens` becomes the request's `max_tokens`, and `sglang.context_length`
becomes the window compaction triggers below. `train.py` passes all three into the
workflow, so none is declared twice.

Reward is temporarily fixed at `0.0`. The verifier and reward design will be added later;
until then the training entry exercises rollout plumbing but produces no policy-gradient
signal. The episode's answer is already machine-readable:
`autorl.agent.submitted_result` returns the fault propagation graph the model passed to
`submit_result`, read back from the session log, and that is what the verifier will
score.

## SFT

Teacher trajectories come from `autorl.data.collect`, which points the same
harness composition at a served model instead of AReaL's rollout proxy. Export
the Harness home it fills, then train on it.

```bash
export RCA_GATEWAY_BASE_URL=... RCA_GATEWAY_API_KEY=...
python -m autorl.data.collect $RCA_DATASET_ROOT/data.jsonl .runs/sft-collect/dsh-home \
  --limit 10 --concurrency 5
python -m autorl.data.export .runs/sft-collect/dsh-home .runs/sft/rca_sessions.jsonl
./scripts/run_sft_smoke.sh
```

`--base-url` takes the same route the rollout takes — see the RL section for why
an endpoint is declared rather than overridden. Without it the episode runs on
`sdk-minimal`'s own `deepseek-official` route, which reads `DEEPSEEK_API_KEY`.

The exporter writes the conversation the teacher actually held — the session's
final surface, folded the way the harness folds it, plus the tool schemas it was
offered. It decides nothing about supervision: `autorl.data.sft` renders those
messages through the tokenizer's own chat template, once per assistant turn and
exactly as a rollout would, and the mask falls out of that. Per turn rather than
once per trajectory because a thinking model's template renders a turn's
reasoning only when the turn follows the conversation's last `user` message; a
whole trajectory rendered in one pass keeps `<think>` on its last turns and
drops it from every earlier one. Ten episodes become 190 rows.

Override the dataset or model with normal AReaL config patches:

```bash
./scripts/run_sft_smoke.sh \
  -p train_dataset.path=/path/to/rca_sessions.jsonl \
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
