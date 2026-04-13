# Experiment Management

This directory is the traceability hub for all RL/SFT training experiments.

## Structure

```
experiments/
├── plans/              — experiment designs (hypothesis + variable matrix)
│   └── <plan-name>.md
├── runs/               — per-run results (meta + metrics + notes)
│   └── <run-id>/
│       ├── meta.yaml   — reproducibility record (config, commit, env)
│       ├── metrics.json — quantitative results
│       └── notes.md    — observations, anomalies, interpretation
├── ablations/          — ablation study designs + comparison tables
│   └── <study-name>/
│       ├── design.md   — variable matrix
│       └── results.md  — filled comparison table
├── scripts/            — automation for metrics collection and validation
└── changelog.md        — decision narrative (proceed/pivot/refine/abandon)
```

## Workflow

```
1. Design (plans/)       — define hypothesis, variables, baselines
2. Configure (configs/)  — create YAML config for each run
3. Execute               — run training, collect traces
4. Record (runs/)        — fill meta.yaml + metrics.json + notes.md
5. Compare (ablations/)  — build comparison tables
6. Decide (changelog)    — proceed / pivot / refine / abandon
```

## Naming conventions

- **Run ID**: `{date}-{study}-{variant}` e.g. `20260412-rca-baseline`
- **Plan name**: `{study}-plan.md` e.g. `rca-reward-plan.md`
- **Ablation name**: matches study name e.g. `rca-reward/`
- **Config name**: matches run variant e.g. `configs/train/rca_reward_f1.yaml`

## Key rule

**Every number has a birth certificate.** Any metric in metrics.json must be traceable through:

```
metric → runs/<id>/meta.yaml → code commit + config + environment
```

If you can't trace it, it's not a result.
