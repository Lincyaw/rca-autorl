# Experiment Plan: RCA Baseline (Phase 1)

## Hypothesis

在 RCABench 数据集上，通过 AReaL RL 训练（PPO + root_cause_f1 reward），Qwen2.5-7B-Instruct 能学会使用 AgentM 工具集进行 RCA 调查，达到 F1 > 0.6 的定位准确率。SFT 预训练（warm start）比 cold start RL 收敛更快、最终质量更高。

## Success criteria

- **Proceed**: root_cause_f1 (eval) > 0.6, avg_turns < 20, false_positive_rate < 0.15
- **Refine**: 0.4 < root_cause_f1 ≤ 0.6（reward signal 存在但不够强，需调参或数据增强）
- **Pivot**: root_cause_f1 < 0.4（需检查 reward function、数据质量或 agent 框架是否有根本问题）

## Variables

### Independent (what changes)

| Variable | Values to test | Rationale |
|----------|---------------|-----------|
| init_strategy | cold (RL only), warm (SFT → RL) | SFT warm start 是 AReaL 标准做法，需验证 RCA 场景是否也有效 |
| model_scale | 0.5B, 7B | 0.5B 验证 pipeline 可跑，7B 冲目标 F1 |
| reward_scaling | 1.0, 5.0, 10.0 | 过小则信号弱收敛慢，过大则不稳定；base.yaml 默认 10.0，需找最优 |
| n_samples | 1, 4 | 每 prompt 采样数影响 PPO advantage 估计质量和 GPU 开销 |
| max_new_tokens | 2048, 4096 | 限制推理长度，影响 agent 能走多少步 |
| temperature | 0.7, 1.0 | 探索-利用 tradeoff |

### Dependent (what you measure)

| Metric | Source | Expected direction |
|--------|--------|--------------------|
| reward_mean | metrics.json | ↑ |
| eval_reward_mean | metrics.json | ↑ |
| root_cause_f1 | metrics.json (eval) | ↑ |
| root_cause_precision | metrics.json (eval) | ↑ |
| root_cause_recall | metrics.json (eval) | ↑ |
| exact_match_rate | metrics.json (eval) | ↑ |
| avg_turns | metrics.json | ↓ |
| avg_tool_calls | metrics.json | observe |
| truncation_rate | metrics.json | ↓ |
| training_loss | tensorboard | ↓ |
| training_time_hours | metrics.json | observe |

### Controlled (what stays fixed)

| Parameter | Value | Why fixed |
|-----------|-------|-----------|
| base_model_family | Qwen2.5-Instruct | 当前唯一适配路径 |
| task_adapter | autorl.tasks.rca.RCATaskAdapter | Phase 1 唯一任务 |
| agent_runtime | autorl.runtime.agentm.AgentMRuntime | Phase 1 唯一 runtime |
| reward_strategy | autorl.rewards.rca.RootCauseMatchRewardStrategy | Phase 1 硬匹配 F1 |
| dataset | RCABench (全量) | 数据不变才能对比 |
| optimizer | Adam (lr=5e-6, wd=0.01) | 沿用 base.yaml 默认 |
| kl_ctl | 0.0 | AReaL 默认关闭 KL penalty |
| tool_gateway | local | Phase 1 不走远端 |
| backend_train | fsdp | 不切换训练后端 |
| backend_rollout | sglang | 不切换推理后端 |

## Planned runs

### Stage 0: Pipeline validation (1 run)

验证端到端 pipeline 能跑通，不关注指标。

| Run ID | Config base | Model | Key settings | Purpose |
|--------|------------|-------|-------------|---------|
| 20260413-rca-smoke | agentm_rca_smoke.yaml | 0.5B | steps=2, batch=1 | Pipeline 通过性验证 |

### Stage 1: Baseline establishment (2 runs)

Cold start RL，建立基线。

| Run ID | Config base | Model | Key differences from baseline |
|--------|------------|-------|-------------------------------|
| 2026MMDD-rca-cold-7b | base → rca fork | 7B | reward_scaling=10, n_samples=4, max_new_tokens=4096, temp=1.0 |
| 2026MMDD-rca-cold-0.5b | base → rca fork | 0.5B | 同上，观察小模型是否有 signal |

### Stage 2: SFT warm start (2 runs)

先 SFT 再 RL，对比 cold start。

| Run ID | Config base | Model | Key differences |
|--------|------------|-------|-----------------|
| 2026MMDD-rca-sft-7b | sft config | 7B | SFT on RCABench sft.jsonl, ~3 epochs |
| 2026MMDD-rca-warm-7b | base → rca fork | 7B | init from SFT checkpoint，与 cold-7b 对比 |

### Stage 3: Hyperparameter sensitivity (4 runs, 可按需裁剪)

基于 Stage 1-2 的 best config，单变量消融。

| Run ID | Variable | Value | Baseline value |
|--------|----------|-------|----------------|
| 2026MMDD-rca-rs1 | reward_scaling | 1.0 | 10.0 |
| 2026MMDD-rca-rs5 | reward_scaling | 5.0 | 10.0 |
| 2026MMDD-rca-ns1 | n_samples | 1 | 4 |
| 2026MMDD-rca-t07 | temperature | 0.7 | 1.0 |

### Stage 4: Context window (2 runs, optional)

评估 max_new_tokens 对调查深度的影响。

| Run ID | Variable | Value | Notes |
|--------|----------|-------|-------|
| 2026MMDD-rca-ctx2k | max_new_tokens | 2048 | 是否足够完成调查 |
| 2026MMDD-rca-ctx8k | max_new_tokens | 8192 | 更长推理是否有收益 |

## Baseline

- Config: `configs/train/base.yaml` 的 RCA fork（替换 task_adapter/runtime/reward/dataset 为 RCA 版本）
- Expected metrics: 无先验基线；commercial model (GPT-4o) 在 RCABench 上的 F1 可作为 upper bound 参考（待测）

## Resource estimate

| Stage | Runs | GPU type | GPUs/run | Est. time/run | Total GPU-hours |
|-------|------|----------|----------|---------------|-----------------|
| 0: Smoke | 1 | any 1x | 1 | 5 min | ~0 |
| 1: Baseline | 2 | 8x A100 | 8 | 4-8h | 64-128 |
| 2: SFT+warm | 2 | 8x A100 | 8 | 6-10h | 96-160 |
| 3: HP sensitivity | 4 | 8x A100 | 8 | 4-8h | 128-256 |
| 4: Context (opt) | 2 | 8x A100 | 8 | 4-8h | 64-128 |
| **Total** | **11** | | | | **352-672** |

注：0.5B 的 run 资源大幅减少（1-2 GPU, 1-2h），上表按 7B worst case 估算。

## Data preparation

1. 获取 RCABench 数据：case 目录结构（含 injection.json, env.json, 可选 conclusion.parquet）
2. 构建 manifest：
   ```bash
   python scripts/build_rcabench_dataset.py <rcabench_root> \
     --output-dir .runs/data/rcabench_full/
   ```
3. 产出：`.runs/data/rcabench_full/rl.jsonl`（RL 训练用）、`sft.jsonl`（SFT 用）
4. Eval split：按 case_name 8:2 划分 train/eval，确保同一 case 不跨 split

## Config creation

Phase 1 需要创建以下 config 文件（基于 base.yaml 修改）：

```yaml
# configs/train/rca_baseline.yaml — 7B RL cold start baseline
task_adapter_path: autorl.tasks.rca.RCATaskAdapter
agent_runtime_path: autorl.runtime.agentm.AgentMRuntime
reward_strategy_path: autorl.rewards.rca.RootCauseMatchRewardStrategy
trace_dir: .runs/rca-baseline/trajectories
train_dataset:
  path: .runs/data/rcabench_full/rl.jsonl
valid_dataset:
  path: .runs/data/rcabench_full/rl_eval.jsonl  # eval split
```

## Pre-flight checklist

- [ ] AgentM submodule initialized and importable
- [ ] RCABench data downloaded and accessible
- [ ] Dataset manifests built (rl.jsonl, sft.jsonl) with train/eval split
- [ ] Reward function tested in isolation (`root_cause_f1_reward` unit test)
- [ ] Smoke config (Stage 0) passes end-to-end
- [ ] Eval pipeline tested (eval.py produces metrics)
- [ ] Metric logging configured (tensorboard path, stats_tracker)
- [ ] GPU cluster accessible and quota confirmed
- [ ] Wandb/tensorboard project set up for experiment tracking

## Risks and mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| AgentM submodule broken | Blocks all runs | Stage 0 smoke test catches this early |
| RCABench data too small | Weak training signal | Check case count after manifest build; if < 50, consider data augmentation |
| reward_scaling too aggressive | Training instability (reward explosion) | Stage 3 sweeps reward_scaling; monitor gradient norms |
| Agent generates garbage output | F1 = 0, no learning signal | SFT warm start provides reasonable init; check trajectory quality manually |
| Truncation dominates | Agent runs out of tokens before concluding | Monitor truncation_rate; Stage 4 tests longer context |

## Exit criteria for Phase 1

Phase 1 完成条件（不是每个 run 的 success criteria，而是整个 Phase 的退出条件）：

1. **至少一个 run** 达到 F1 > 0.6 → Phase 1 complete, 进入 Phase 1.5
2. **最佳 run** F1 ∈ (0.4, 0.6] → 记录最优配置，分析瓶颈，制定 Phase 1b 补充 plan
3. **所有 run** F1 < 0.4 → 根本性问题排查（数据、reward function、agent 框架），不进入 Phase 2
