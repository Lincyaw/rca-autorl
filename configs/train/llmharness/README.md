# llmharness training strategies

Six pluggable training-strategy configs over the llmharness extractor
and auditor child agents. Each strategy plugs in via a YAML config;
all six share the same eval pipeline (firing TaskAdapter + reward
strategy + child runtime).

| Config                          | Phase     | Strategy           | Role     |
|---------------------------------|-----------|--------------------|----------|
| extractor_sft_baseline.yaml     | extractor | pure SFT           | baseline |
| extractor_grpo_process.yaml     | extractor | SFT + GRPO process | method   |
| extractor_dpo_outcome.yaml      | extractor | SFT + DPO outcome  | method   |
| auditor_sft_baseline.yaml       | auditor   | pure SFT           | baseline |
| auditor_grpo_process.yaml       | auditor   | SFT + GRPO process | method   |
| auditor_dpo_outcome.yaml        | auditor   | SFT + DPO outcome  | method   |

## Plug points

Every non-SFT config references three importable class paths:

- `task_adapter_path`     — `autorl.tasks.llmharness_firing.{Extractor,Auditor}FiringTaskAdapter`
- `agent_runtime_path`    — `autorl.runtime.llmharness_{extractor,auditor}.Llmharness{Extractor,Auditor}Runtime` (stubs in this PR)
- `reward_strategy_path`  — `autorl.rewards.llmharness_{extractor,auditor}{,_outcome}.Llmharness{...}RewardStrategy`

The class-path test (`tests/test_configs_class_paths.py`) imports each
path on every CI run to catch typos.

## Data layout

The configs reference `${LLMHARNESS_RUNS_DIR}/sft-XX/...` artifacts:

- `extractor.jsonl` / `auditor.jsonl` — SFT distill rows (already produced).
- `rl_prompts.jsonl` — firing prompts, one row per child firing
  (produced by `llmharness-distill rl-prompts`, parallel-track work).
- `dpo_pairs.jsonl` — preference pairs over child firings (produced
  by `llmharness-distill dpo-pairs`, parallel-track work).
- `<bundle>/case_outcome.json` — per-case outcome labels (produced by
  `llmharness-distill annotate-case-outcome`).

These configs are not runnable end-to-end this PR — they wait on the
upstream data files. Their class paths are imported by the unit test
suite so refactors that rename a class break loudly.

## DPO note

AReaL does not currently ship a DPO trainer in the submodule. The
`*_dpo_outcome.yaml` configs use TRL `DPOConfig`-shaped defaults under
a top-level `dpo:` block so they're implementable against either an
upcoming AReaL DPO trainer or a TRL bridge.
