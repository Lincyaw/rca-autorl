# CLAUDE.md

See `AGENTS.md` for detailed coding style, testing, and commit guidelines — those take precedence over anything here.

## North-star targets

### Scaffold (always active)
1. **Pipeline health** — lint clean + core imports pass (currently: 1 ruff error, imports OK)
   Measure: `python3 -m ruff check src`
2. **Scaffold extensibility** — new agent = TaskAdapter + AgentRuntime + RewardStrategy + config, zero workflow changes
   Measure: agent review against contracts/ interfaces

### Phase 1 — RCA Agent baseline (current)
3. **RCA F1** > 0.6 on RCABench eval (currently: no baseline)
4. **RCA efficiency** — avg_turns < 20, false_positive_rate < 0.15

### Phase 2 — All agents independent (mock dependencies)
5. **World Model** — propagation_edge_f1 > 0.5
6. **Fault Injection** — injection_validity_rate > 0.8
7. **Verification** — rca_judgment_accuracy > 0.8 (商业模型 judge 质量监控)

### Phase 3-4 — Integrated pipeline
8. **Adversarial equilibrium** — FI success rate ∈ [0.35, 0.65]
9. **E2E detection rate** > 0.7
10. **Verification-as-judge** — f1_correlation > 0.85

Full metrics spec: `.doc/designs/agent-roles-spec.md` §5

Secondary criterion: **simplicity** — if two approaches give similar metrics, prefer less code.

<!-- auto-harness:begin -->
## Project conventions

- Package manager: `uv` (not pip). Use `uv sync` to install, `uv run` to execute.
- Lint: `uv run ruff check src`
- Smoke test: `./scripts/run_smoke.sh`
- Python: 3.12, `src/` layout, type hints required
- Third-party deps (AReaL, agentm) are editable submodules under `third_party/`
- Language: discussion in Chinese, code/docs in English

## Active skills

- dev-loop — iteration methodology: implement, test, verify, measure
- north-star — quantifiable optimization targets with observation mechanisms
- long-horizon — autonomous decision-making with escalation ladder
- notify — push iteration reports via email/Feishu/Telegram
- areal-rl — AReaL RL training methodology: reward design, SFT→RL pipeline, async staleness, scaling, pitfalls
- experiment-lab — experiment lifecycle: plan, execute, record, compare, decide
- new-project — spec-driven development, requirements index maintenance
- research-ideation — hypothesis lifecycle: formulate, validate, evolve
- plotting — publication-quality experiment figures
<!-- auto-harness:end -->
