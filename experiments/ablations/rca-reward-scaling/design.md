# Ablation: RCA Reward Scaling & Init Strategy

## Question

Which combination of reward_scaling and initialization strategy (cold RL vs SFT warm start)
produces the best RCA F1 on the eval set?

## Base configuration

7B model, full RCABench dataset, n_samples=4, max_new_tokens=4096, temperature=1.0,
all other parameters from `configs/train/rca_baseline.yaml`.

## Variable matrix

| Run | init_strategy | reward_scaling | n_samples | temperature | Expected effect |
|-----|--------------|----------------|-----------|-------------|-----------------|
| cold-7b | cold RL | 10.0 | 4 | 1.0 | primary baseline |
| warm-7b | SFT → RL | 10.0 | 4 | 1.0 | SFT should help convergence |
| cold-rs1 | cold RL | 1.0 | 4 | 1.0 | weaker signal, slower but more stable? |
| cold-rs5 | cold RL | 5.0 | 4 | 1.0 | middle ground |
| cold-ns1 | cold RL | 10.0 | 1 | 1.0 | lower variance in advantage estimate |
| cold-t07 | cold RL | 10.0 | 4 | 0.7 | less exploration, more exploitation |

## Analysis plan

1. **Primary comparison**: cold-7b vs warm-7b → quantify SFT warm start benefit
2. **Reward scaling sweep**: cold-rs1 vs cold-rs5 vs cold-7b → find optimal scaling
3. **Sampling**: cold-ns1 vs cold-7b → n_samples=1 vs 4 efficiency tradeoff
4. **Temperature**: cold-t07 vs cold-7b → exploration benefit at different stages

## Key metrics for comparison

| Metric | Primary? | Notes |
|--------|----------|-------|
| eval_reward_mean (root_cause_f1) | Yes | main optimization target |
| root_cause_precision | Yes | watch for brute-force hacking |
| root_cause_recall | Yes | watch for conservative predictions |
| avg_turns | No | efficiency indicator, not in reward |
| truncation_rate | No | context sufficiency indicator |
| training_time_hours | No | resource efficiency |
| convergence_step | No | when does reward plateau? |
