# rca-autorl

AReaL-first agent training scaffold with framework-agnostic runtime contracts.

## Scope

This repository keeps AReaL as the rollout/training substrate and adds a repo-owned
contract layer for task adapters, agent runtimes, trajectories, and tool/env gateways.

- Implemented: `agent_workflow` train/eval/infer path through AReaL native agent workflow (`async run(data, **extra_kwargs)`)
- Implemented: framework-agnostic contracts under `autorl.contracts`
- Implemented: runtime/task/gateway abstractions under `autorl.runtime`, `autorl.tasks`, `autorl.gateways`
- Example task: search-style agent (`SearchTaskAdapter` + `SearchAgentRuntime`)
- Not implemented yet: `agent_sft/train.py` remains a stub

## Repository structure

- `configs/train/`
  - baseline experiment configs (`smoke`, `base`)
- `src/autorl/contracts/`
  - canonical data formats (`TaskSample`, `AgentInput`, `Trajectory`, `TaskOutcome`)
- `src/autorl/runtime/`
  - runtime interface, canonical AReaL workflow entrypoint, trace sink
- `src/autorl/tasks/`
  - task adapters plus example runtime/reward implementations
- `src/autorl/gateways/`
  - tool/env boundary interfaces and local wrappers
- `src/autorl/experiments/agent_workflow/`
  - thin train/eval/infer entrypoints
- `src/autorl/data/`
  - dataset loading/materialization; task adapters own schema validation

## Canonical entrypoints

- Workflow: `autorl.runtime.agent_workflow.UnifiedAgentWorkflow`
- Runtime interface: `autorl.runtime.AgentRuntime`
- Task interface: `autorl.tasks.TaskAdapter`
- Tool gateway: `autorl.gateways.ToolGateway`
- Example task pieces:
  - `autorl.tasks.search.SearchTaskAdapter`
  - `autorl.tasks.search_runtime.SearchAgentRuntime`
  - `autorl.tasks.search_reward.SearchRewardStrategy`

## Setup

Initialize submodules first (AReaL is used as a local editable dependency):

```bash
git submodule update --init --recursive
uv sync
```

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

- SFT path is not implemented yet (`src/autorl/experiments/agent_sft/train.py`).
- No repo-level test suite has been added in this pass.
- The example task is still search-oriented; new tasks should be added through new task adapters/runtimes rather than by editing the shared workflow.
