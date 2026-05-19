# rca-autorl

AReaL-first agent training scaffold with framework-agnostic runtime contracts.

## Scope

This repository keeps AReaL as the rollout/training substrate and adds a repo-owned
contract layer for task adapters, agent runtimes, trajectories, and tool/env gateways.

- Implemented: `agent_workflow` train/eval/infer path through AReaL native agent workflow (`async run(data, **extra_kwargs)`)
- Implemented: repo-owned RCABench manifest builder for AgentM RCA smoke / iteration
- Implemented: SFT path consuming `llmharness/distill` JSONL through AReaL `SFTTrainer`
- Implemented: framework-agnostic contracts under `autorl.contracts`
- Implemented: runtime/task/gateway abstractions under `autorl.runtime`, `autorl.tasks`, `autorl.gateways`
- Example tasks: search-style QA and AgentM-backed RCA

## Repository structure

- `configs/train/`
  - baseline RL/rollout experiment configs (`smoke`, `base`, `agentm_rca_smoke`)
- `configs/sft/`
  - SFT smoke configs that consume `llmharness/distill` output
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
  - `autorl.runtime.agentm.AgentMRuntime` (wraps `agentm_rca.eval.agent.AgentMAgent`)
  - `autorl.rewards.rca.RCABaselineRewardStrategy`

## Setup

Initialize submodules first (AReaL is used as a local editable dependency):

```bash
git submodule update --init --recursive
UV_HTTP_TIMEOUT=120 uv sync --python 3.12
```

Core training/runtime dependencies follow `third_party/AReaL/pyproject.toml`. Keep the
root project lean: do not re-pin `torch`, `sglang`, `flash-attn`, `openai`, or similar
packages here unless the repo adds a dependency that AReaL does not already own.

The root `pyproject.toml` depends on `areal[sglang]` as an editable submodule and on
`agentm` as an editable reference to the sibling `../AgentM` working tree (no
vendored copy). Make sure `../AgentM` exists and is on the branch you want to use
before running `uv sync`.

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

`agentm` is referenced as an editable install pointing at the sibling `../AgentM`
working tree (the legacy `third_party/agentm` submodule was removed during the
migration to the new harness-sync scenario).  The repo targets Python 3.12 so
`agentm` and `autorl` share one environment.

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

## SFT data — llmharness/distill output

SFT data is produced by AgentM's ``llmharness`` extension, not by a
repo-owned builder. After running an ``agentm`` rca:harness.sync eval
pass, ``llmharness-distill export`` emits one JSONL row per
extractor (and auditor) child session with Qwen / GLM ``<think>`` +
tool_calls shape::

    {
      "phase": "extractor",
      "input":  {"system": "...", "user": "..."},
      "target": {"messages": [{"role": "assistant",
                               "content": "<think>...</think>",
                               "tool_calls": [...]}]},
      ...
    }

Point ``configs/sft/agentm_rca_sft_smoke.yaml``'s
``train_dataset.path`` at the resulting ``extractor.jsonl`` and the
trainer will tokenize through the Qwen3-Thinking chat template,
masking the prompt and supervising the assistant ``<think>`` + tool
call. See ``../AgentM/contrib/extensions/llmharness/runs/`` for a
sample bundle.

## End-to-end run order

1. **Bootstrap once** (submodule + venv):

   ```bash
   git submodule update --init --depth 1 third_party/AReaL
   UV_HTTP_TIMEOUT=300 uv sync --python 3.12
   ```

2. **SFT** — consumes the llmharness/distill bundle (no other data prep needed):

   ```bash
   ./scripts/run_sft_smoke.sh
   ```

   Equivalent to:

   ```bash
   PATH="$PWD/.venv/bin:$PATH" python3 -m autorl.experiments.agent_sft.train \
     --config configs/sft/agentm_rca_sft_smoke.yaml
   ```

   No GPU available? Run `scripts/sft_cpu_smoke.py` first — it loads
   the same distill bundle on Qwen3-0.6B (CPU, ~7 min) and prints
   per-step loss so you can confirm the chat template / loss_mask
   wiring is healthy before booking GPUs. See its docstring for env
   overrides.

3. **RL** — consumes AgentM's processed RCA dataset directly. Set the
   dataset root once so `RCATaskAdapter` can resolve each
   `datapack_name` to an absolute case directory:

   ```bash
   export AGENTM_RCA_DATASET_ROOT=/home/ddq/AoyangSpace/dataset/rca

   PATH="$PWD/.venv/bin:$PATH" python3 -m autorl.experiments.agent_workflow.train \
     --config configs/train/agentm_rca_smoke.yaml
   ```

   If you have a different processed dataset, point `train_dataset.path`
   at its `data.jsonl` (with override `-p train_dataset.path=...`).

   The live AgentM rollout still needs LLM credentials: either let
   AReaL inject its proxy, or set `OPENAI_API_KEY` / `OPENAI_BASE_URL`
   (matched by `AGENTM_PROVIDER=openai`) before launch. Anthropic-style
   providers are equally supported — see `agentm_rca.eval.agent` for
   the env-var convention.

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
