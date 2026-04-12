# rca-autorl

AReaL-first agent training scaffold with framework-agnostic runtime contracts.

## Scope

This repository keeps AReaL as the rollout/training substrate and adds a repo-owned
contract layer for task adapters, agent runtimes, trajectories, and tool/env gateways.

- Implemented: `agent_workflow` train/eval/infer path through AReaL native agent workflow (`async run(data, **extra_kwargs)`)
- Implemented: repo-owned RCABench manifest builder for AgentM RCA smoke / iteration
- Implemented: SFT path for repo-owned JSON/JSONL manifests through AReaL `SFTTrainer`
- Implemented: framework-agnostic contracts under `autorl.contracts`
- Implemented: runtime/task/gateway abstractions under `autorl.runtime`, `autorl.tasks`, `autorl.gateways`
- Example tasks: search-style QA and AgentM-backed RCA

## Repository structure

- `configs/train/`
  - baseline RL/rollout experiment configs (`smoke`, `base`, `agentm_rca_smoke`)
- `configs/sft/`
  - SFT smoke configs for standardized RCA manifests
- `scripts/build_rcabench_dataset.py`
  - convert RCABench case directories into repo-owned `rl.jsonl` / `sft.jsonl`
- `src/autorl/contracts/`
  - canonical data formats (`TaskSample`, `AgentInput`, `Trajectory`, `TaskOutcome`)
- `src/autorl/runtime/`
  - runtime interface, canonical AReaL workflow entrypoint, trace sink, AgentM runtime bridge
- `src/autorl/tasks/`
  - task adapters plus example runtime/reward implementations (`search`, `rca`)
- `src/autorl/gateways/`
  - tool/env boundary interfaces and local wrappers
- `src/autorl/data/`
  - generic loaders plus RCABench manifest conversion and SFT dataset tokenization
- `src/autorl/experiments/agent_workflow/`
  - thin train/eval/infer entrypoints
- `src/autorl/experiments/agent_sft/`
  - repo-owned SFT entrypoint that accepts JSON/JSONL manifests

## Canonical entrypoints

- Workflow: `autorl.runtime.agent_workflow.UnifiedAgentWorkflow`
- Runtime interface: `autorl.runtime.AgentRuntime`
- Task interface: `autorl.tasks.TaskAdapter`
- Tool gateway: `autorl.gateways.ToolGateway`
- AgentM RCA pieces:
  - `autorl.tasks.rca.RCATaskAdapter`
  - `autorl.runtime.agentm.AgentMRuntime`
  - `autorl.rewards.rca.RootCauseMatchRewardStrategy`

## Setup

Initialize submodules first (AReaL is used as a local editable dependency):

```bash
git submodule update --init --recursive
UV_HTTP_TIMEOUT=120 uv sync --python 3.12
```

Core training/runtime dependencies follow `third_party/AReaL/pyproject.toml`. Keep the
root project lean: do not re-pin `torch`, `sglang`, `flash-attn`, `openai`, or similar
packages here unless the repo adds a dependency that AReaL does not already own.

The root `pyproject.toml` depends on `areal[sglang]` and `agentm` as editable local
packages, and the repo-level `tool.uv` settings mirror the minimum AReaL overrides
needed to resolve the vendored SGLang stack from this workspace.

## Baseline runs

Use repo-owned baseline configs:

```bash
./scripts/run_smoke.sh
```

```bash
PATH="$PWD/.venv/bin:$PATH" python3 -m autorl.experiments.agent_workflow.train --config configs/train/base.yaml
```

The explicit `PATH` prefix matters for AReaL local runs because worker processes invoke
`python3` internally; keeping `.venv/bin` first ensures actor and rollout workers reuse
one environment.

## AgentM integration

`third_party/agentm` is wired in as an editable `uv` workspace dependency. The repo now
targets Python 3.12 so `agentm` and `autorl` share one environment.

- RCA adapter: `autorl.tasks.rca.RCATaskAdapter`
- AgentM runtime: `autorl.runtime.agentm.AgentMRuntime`
- RCA reward: `autorl.rewards.rca.RootCauseMatchRewardStrategy`

Sample training record shape:

```json
{"id":"case-1","incident":"checkout latency spikes after deploy","data_dir":"/abs/path/to/case","root_causes":["ts-price-service"]}
```

Config switch example:

```yaml
task_adapter_path: autorl.tasks.rca.RCATaskAdapter
agent_runtime_path: autorl.runtime.agentm.AgentMRuntime
reward_strategy_path: autorl.rewards.rca.RootCauseMatchRewardStrategy
```

`AgentMRuntime` defaults to `third_party/agentm/config/system.yaml` and the
`rca_hypothesis` scenario. Override its LLM endpoint with AgentM's own env vars such as
`AGENTM_API_KEY`, `AGENTM_API_BASE_URL`, and optional model overrides.

When AReaL injects an OpenAI-compatible proxy endpoint into the workflow runtime,
`AgentMRuntime` automatically exports it as:

- `AGENTM_API_BASE_URL=<AReaL proxy base_url>`
- `AGENTM_API_KEY=<AReaL proxy api_key>`
- `AGENTM_ORCHESTRATOR_MODEL=default`
- `AGENTM_WORKER_MODEL=default`

This keeps AgentM rollout calls on the AReaL proxy path instead of bypassing training.

## RCABench dataset builder

Build standardized manifests from a single case or a directory of cases:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_rcabench_dataset.py \
  /mnt/nvme0/rcabench_data/rcabench/ts9-ts-route-plan-service-request-replace-path-9dg8qf \
  --output-dir .runs/data/agentm_rca_smoke
```

The builder writes:

- `rl.jsonl` — for AgentM RCA rollout / RL (`incident`, `data_dir`, `root_causes`, `messages`)
- `sft.jsonl` — for repo-owned SFT (`messages`, `assistant_response`)
- `metadata.json` — conversion summary

The RL manifest also carries eval-friendly `question` / `answer` fields so it can be
reused by RCABench-style tooling.

## Smoke flow

Dataset conversion:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_rcabench_dataset.py \
  /mnt/nvme0/rcabench_data/rcabench/ts9-ts-route-plan-service-request-replace-path-9dg8qf \
  --output-dir .runs/data/agentm_rca_smoke
```

SFT smoke:

```bash
PATH="$PWD/.venv/bin:$PATH" python3 -m autorl.experiments.agent_sft.train \
  --config configs/sft/agentm_rca_sft_smoke.yaml
```

RL / rollout smoke:

```bash
PATH="$PWD/.venv/bin:$PATH" python3 -m autorl.experiments.agent_workflow.train \
  --config configs/train/agentm_rca_smoke.yaml
```

The live AgentM rollout path still requires model credentials or a reachable OpenAI-style
proxy. If you are not running under AReaL proxy injection, set `AGENTM_API_KEY` and
optionally `AGENTM_API_BASE_URL` before invoking the AgentM runtime.

## Safe eval/infer behavior

`eval` and `infer` run against explicit non-train data by default.

- Do not rely on implicit train-set fallback for reportable evaluation claims.
- If you intentionally need train-set fallback for debugging, use explicit flags:
  - `allow_train_fallback_for_eval: true`
  - `allow_train_fallback_for_infer: true`

## Runtime notes

- AReaL proxy injects `base_url` / `api_key` / `http_client` into workflow `run(...)` calls.
- `tool_gateway_mode: local` uses the repo-local `tool_env` implementation.
- `tool_gateway_mode: http` uses `tool_gateway_base_url`.
- AReaL built-in observability is available through `stats_logger` (wandb / swanlab / tensorboard / trackio) and `perf_tracer` / `session_tracer`.
- Repo rollout workflows publish per-episode scalars such as reward, success, turns, tool usage, and token counts through AReaL's `stats_tracker` so they flow into the configured `stats_logger` backend.
- `setup_metrics()` writes run metadata under `<cluster.fileroot>/autorl/metadata` when `cluster.fileroot` is set; fallback is `artifacts/metadata`.
- Canonical trajectories and episode-level metric records can be persisted through `trace_dir` or the derived `<cluster.fileroot>/autorl/trajectories/<phase>` path.

## Current limitations

- No repo-level test suite has been added in this pass.
- The repo-owned RCA builder currently falls back to `injection.json.ground_truth` for
  target graphs; `conclusion.parquet` in the provided case does not expose a ready-made
  causal graph schema.
- End-to-end live rollout still depends on runtime credentials (`AGENTM_API_KEY`) or an
  AReaL proxy endpoint being available at execution time.
