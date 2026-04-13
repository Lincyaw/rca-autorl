# Experiment Changelog

Reverse chronological. Each entry records one decision point.

---

## 2026-04-13: Phase 1 experiment plan created

**Decision**: proceed with RCA baseline plan — 11 planned runs across 4 stages
**Plan**: `plans/rca-baseline-plan.md`
**Ablation**: `ablations/rca-reward-scaling/` — reward_scaling × init_strategy matrix
**Config**: `configs/train/rca_baseline.yaml` (7B cold start baseline)
**Key variables**: init_strategy (cold/warm), reward_scaling (1/5/10), n_samples (1/4), temperature (0.7/1.0)
**Next**: complete pre-flight checklist (agentm submodule, data build, Stage 0 smoke)

### Context
Phase 1 north-star: F1 > 0.6, avg_turns < 20, FPR < 0.15. Plan starts with
pipeline validation (Stage 0 smoke), then baseline establishment (Stage 1),
SFT warm start comparison (Stage 2), hyperparameter sensitivity (Stage 3),
and optional context window ablation (Stage 4). Success criteria use
proceed/refine/pivot thresholds at F1 = 0.6/0.4.

---

## 2026-04-12: Project initialization

**Decision**: proceed — scaffold established, ready for first experiment plan
**Evidence**: contracts, runtime, reward, dataset builder all implemented
**Code**: main@68ed974
**Next**: design first ablation study (reward function variants or SFT→RL pipeline)

### Context
rca-autorl scaffold is functional: AReaL workflow, AgentM runtime, RCABench dataset
builder, F1 reward. Smoke config exists but not yet validated end-to-end (agentm
submodule needs initialization).

---
