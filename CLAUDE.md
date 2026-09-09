# CLAUDE.md

See `AGENTS.md` for coding style, testing, and commit guidelines — those take precedence over anything here.

## What this repo is

RCA benchmark loop 三个子项目（Injector / Verifier / RCA agent）中 **RCA agent** 的训练代码。研究主题：**Cost-Sensitive Active Diagnosis** —— 把微服务故障定位建模为部分可观测下的 cost-sensitive active diagnosis，agent 通过只读查询收集证据，提交一张 fault propagation graph，reward 只来自程序可校验的 outcome。

**设计文档不在本仓库。** 全部在 Notes 仓库（`~/AoyangSpace/Notes`）的 `research/ongoing/rcabench/rca/`：

- `CONTEXT.md` —— 词表与阅读顺序，先读这个
- `idea/proposal.md` —— 22-field 项目 brief（状态、成功标准、当前最重要的问题）
- `idea/method.md` —— 训练方法 spec（reward、credit assignment、dynamic sampling、fork、ablation grid）
- `log/` —— 决策记录；`2026-09-08-rl-reward-and-credit-assignment.md` 是当前 reward 的依据
- 上一级 `rcabench/CONTEXT.md` 讲三个子项目怎么组成一个 loop，injector / verifier 各有自己的目录

本仓库只回答"代码做了什么、怎么跑"：`README.md`（入口与流程）和 `agent/README.md`（harness 设计）。设计层面的改动先进 Notes，再改代码；Notes 的规则是新 log entry 要先问过用户。

## 仓库布局

```
src/autorl/          AReaL workflow (agent.py)、reward + difficulty、fpg 绑定、dataset 准备、train / train_sft 入口、data/ (collect / export / sft)
agent/               dsh bundle（sql / take_note / submit_result / compaction / pruner）+ scenario patch
configs/fpg/         答案契约的 vocabulary profile；configs/train、configs/sft 训练配置
data/sft/            45 条蒸馏 teacher episode（git-lfs；只含 train 划分的 case）
datapacks/ops-lite/  500-case 语料，含 fpg Scenario ground truth（不入 git）
tests/               reward、difficulty、contract 一致性、SFT mask、answer-leak
third_party/AReaL    fork Lincyaw/AReaL 分支 rca-autorl（多 rescore_group hook 与 store 修复）
```

答案契约用外部 `fpg` 包（github.com/Lincyaw/fpg-convention，按 commit 锁定）：agent 输出 `ModelRCAOutput`，标注是 `Scenario`。词表 `configs/fpg/microservices.toml` 沿用标注侧的那份。`src/autorl/fpg.py` 绑定两者并生成 `agent/rca-harness/src/vocabulary.js`，harness 在提交时拒收不合契约的答案。

## 当前状态（2026-09-09）

- RL 路径在单卡上端到端跑通过（rollout → reward → rescore_group → 一步 PPO 更新），但还没观察到非零 advantage
- reward = fpg 约定的三轴图 F1（method spec §2）；advantage = sibling 难度加权后的 RLOO（§3，`econfig.difficulty` 关掉即 flat 分数的消融臂）。没有 cost，没有 turn-level credit
- method spec 里的 target design（declare/seal、anomaly set 与 attribution/dismissal、per-turn cost、selective fork、curriculum feedback）**都还没实现**；spec 自己标明了哪些是 target、哪些是 current
- SFT 数据：45 条 teacher episode 已 check in（原 50 条，去掉了 5 条落在 held-out 划分的）；下一步是 SFT 到一个会提交 graph 的 checkpoint，再测 within-group variance

## 工作流约定

- **代码改动**：遵循 `AGENTS.md`；agent loop 留在 DeepSeek Harness，本仓库只负责 data、harness 组合、workflow、reward 与 trainer wiring。不建仓库内通用框架
- **设计改动**：先改 Notes 的 method spec 或加 log entry，再改代码。代码注释引用 Notes 时用文档名（如 `2026-09-08-rl-reward-and-credit-assignment`），不用本机路径
- **新实验**：从 Notes 的 method spec §9 ablation grid 派生；协议进 Notes 的 `experiments/`，代码与配置进本仓库 `configs/`

## North-star targets

以 Notes `idea/proposal.md` 第 17、18 字段为准，这里只列当前阶段要看的数：

### Scaffold（始终激活）
1. **Pipeline health** —— `./scripts/check.sh` 全过（Ruff + mypy + vocabulary.js 新鲜度）；`python -m unittest discover -s tests` 全过
2. **Integration simplicity** —— 同等指标下选代码更少的方案

### 当前阶段：SFT → RL 就绪
3. **SFT 产出会提交的 checkpoint** —— held-out case 上 `submit_result` 被接受的比例 > 0.8
4. **Within-group variance** —— K=8 个 sibling 的 weighted score 在足够比例的 case 上方差非零，RL batch 不会被 dynamic sampling 掏空（这是 method spec 全部设计赖以成立的第一个测量）

### 下一阶段：RL baseline
5. 训练后的 agent 在 frozen held-out 上的 root-cause accuracy 与 path validity 超过两个 baseline：one-shot prompting、flat outcome reward
6. Stopping 是 cost-sensitive 的：既不总是耗尽 budget，也不总是过早停止

## Project conventions

- Package manager: `uv` (not pip). `uv sync` to install, `uv run` to execute
- Static checks: `./scripts/check.sh`；unit tests: `python -m unittest discover -s tests`
- Smoke: `./scripts/run_sft_smoke.sh`（SFT，单卡）、`./scripts/run_smoke.sh`（RL）
- Python 3.12, `src/` layout, type hints required；bundle 是无 build 的 ESM
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
