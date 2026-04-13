# North-star baseline — 2026-04-12

## Targets

| # | Target | Indicator | Baseline | Goal |
|---|--------|-----------|----------|------|
| 1 | End-to-end smoke | smoke script exit 0 | untested (agentm submodule missing) | exit 0 |
| 2 | Scaffold extensibility | new task = 3 interfaces + config, no workflow diff | holds for rca + search | holds for 3+ task types |
| 3 | Pipeline health | ruff errors + import failures | 1 ruff error | 0 errors |

## Observation mechanisms

| Target | Mechanism | Command |
|--------|-----------|---------|
| E2E smoke | script | `./scripts/run_smoke.sh` |
| Extensibility | agent review | review contracts/ interfaces when adding tasks |
| Pipeline health | script | `python3 -m ruff check src` |

## Progress log

| Date | Commit | Smoke | Extensibility | Ruff errors | Notes |
|------|--------|-------|---------------|-------------|-------|
| 2026-04-12 | 68ed974 | n/a | 2 tasks (rca, search) | 1 | baseline established |
