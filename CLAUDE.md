# CLAUDE.md

See `AGENTS.md` for detailed coding style, testing, and commit guidelines — those take precedence over anything here.

## What this repo is

本仓库承载一个完整的科研项目，覆盖从 idea 形成到代码实现、实验运行、再到论文撰写的全流程。**不是单纯的代码仓库，也不是单纯的论文仓库**。

研究主题：**Self-Evolving Microservice RCA via Triadic Self-Play** —— 借鉴 self_play_evolves / GASP / SGS 三篇论文的三元角色框架（Proposer/Solver/Verifier），训练一组围绕微服务故障定位的 LLM agent。当前主线包含 4 个相对独立的角色（RCA / Fault Injection / Verifier / Controller），用 data-level 机制做 strong-to-weak 同步。

## 仓库布局（high level）

```
.doc/
  designs/           — 系统设计文档（agent 角色 spec、训练框架、harness 设计）
  references/        — 参考文献 PDF + 综述笔记
paper/
  ideas/             — research idea 家族（v0 母 idea + v1 子 hypothesis）
  (later)            — 后续会有 sections/、figures/、experiments/ 等
src/autorl/          — RL 训练框架代码（contracts / runtime / tasks / gateways / rewards）
experiments/         — 实验配置、运行脚本、结果记录
third_party/         — AReaL、agentm 等子模块
```

## 当前研究状态

- 主线 idea：`paper/ideas/triadic-self-play/v0-overview.md` 及同目录下 4 个 v1 子 idea；索引 `paper/ideas/INDEX.md`
- 主线高层架构：v0-overview 指明 4 个主线 agent = **RCA / FI / Verifier / Controller**
- 训练流程：`.doc/designs/training-pipeline.md`（Stage 0/1/2/3 训练过程轴，与下方 Phase 北极星正交）
- 详细实现 spec：`.doc/designs/agent-roles-spec.md`（TaskAdapter / Reward / 指标定义；Verifier ≡ 旧 spec 的 Verification Agent）
- 当前进度：Phase 1 RCA 基线（早期）
- 平行研究线（正交于主线，主线通后启动）：
  - **Controller harness** —— 黑盒 wrapper，见 `paper/ideas/controller-harness/v0.md` + `.doc/designs/llm-harness.md`
  - **World Model** —— 三元组任务隐含的因果传播能力，先 probe 后 standalone，见 `paper/ideas/world-model/v0.md`

## 工作流约定

- **新 idea / hypothesis**：先在 `paper/ideas/` 加文档，至少有 motivation + setting + modeling approach。validation 字段可后补
- **新实验**：从某个 v1 idea 派生，进 `experiments/plans/`，测得的指标回写到 idea 文档的 success criteria
- **代码改动**：遵循 `AGENTS.md` 的 code style；新 agent 的接口约定在 `src/autorl/contracts/`
- **论文写作**：进入到产出阶段后用 `paper/sections/` + LaTeX 工程化（晚期再启动）

## North-star targets（按研究阶段）

> Phase = 结果里程碑轴（本节）。Stage = 训练过程轴（见 `.doc/designs/training-pipeline.md`）。两轴正交。

### Scaffold（始终激活）
1. **Pipeline health** — lint clean + core imports pass (`python3 -m ruff check src`)
2. **Scaffold extensibility** — 新 agent = TaskAdapter + AgentRuntime + RewardStrategy + config，零 workflow 改动

### Phase 1 — RCA baseline
3. **RCA F1** > 0.6 on RCABench eval
4. **RCA efficiency** — avg_turns < 20, false_positive_rate < 0.15

### Phase 2 — Each agent independent (mocked dependencies)
5. **Fault Injection** — injection_validity_rate > 0.8
6. **Verifier (judge accuracy)** — rca_judgment_accuracy > 0.8（LLM-as-judge 与硬编码 F1 anchor 的一致性）
7. **Verifier (process score)** — hop 三子分自洽性 ≥ 0.7（同 trajectory 多次采样的一致率）

### Phase 3-4 — Integrated self-play
8. **Adversarial equilibrium** — FI success rate ∈ [0.35, 0.65]
9. **E2E detection rate** > 0.7
10. **Asymmetry health** — `asymmetry_gap` 稳定在 [0.2, 0.7]（见 `paper/ideas/triadic-self-play/asymmetry-ladder/v1.md`）
11. **Verifier-as-judge correlation** — `f1_correlation` > 0.85

### Parallel — 正交研究线（低优先级，主线 Stage 2 末期再启）
12. **Controller drift detection** — precision/recall ≥ 0.6 on annotated trace（per `.doc/designs/llm-harness.md` P2）
13. **World Model probe** —— 主线训练后 RCA / FI / Verifier 内部传播预测准确率显著高于 base baseline；如启 standalone WM，则 `propagation_edge_f1` > 0.5

完整指标定义：`.doc/designs/agent-roles-spec.md` §5（Verifier 部分参考其 §5.4 Verification Agent；World Model 详细 reward / 数据见 §5.2）

Secondary criterion: **simplicity** — 同等指标下选代码更少的方案。

<!-- auto-harness:begin -->
## Project conventions

- Package manager: `uv` (not pip). Use `uv sync` to install, `uv run` to execute.
- Lint: `uv run ruff check src`
- Smoke test: `./scripts/run_smoke.sh`
- Python: 3.12, `src/` layout, type hints required
- Third-party deps (AReaL, agentm) are editable submodules under `third_party/`
- Language: discussion in Chinese, code/docs in English

## Active skills

- research-ideation — hypothesis lifecycle: formulate, validate, evolve
- experiment-lab — experiment lifecycle: plan, execute, record, compare, decide
- paper-writing — narrative structure, terminology consistency, anti-AI-style hygiene
- paper-engineering — LaTeX repo setup, figure pipeline, reproducibility
- dev-loop — iteration methodology: implement, test, verify, measure
- north-star — quantifiable optimization targets with observation mechanisms
- areal-rl — AReaL RL training methodology: reward design, SFT→RL pipeline, async staleness, scaling, pitfalls
- long-horizon — autonomous decision-making with escalation ladder
- plotting — publication-quality experiment figures
- notify — push iteration reports via email/Feishu/Telegram
<!-- auto-harness:end -->
