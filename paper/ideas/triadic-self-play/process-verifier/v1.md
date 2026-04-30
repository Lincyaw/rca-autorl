---
id: idea-v1-process-verifier
date: 2026-04-30
status: active
parent: idea-v0-rca-self-evolves
---

# Hop-Level Process Verification for RCA Reasoning

## Motivation

现有 RCA reward 设计基本只看终态 —— 即根因集合的 F1 分数。这有两个问题：(a) reward signal 稀疏，每个 episode 只在最后给一次分；(b) 模型可能输出"幻觉式正确答案" —— 答案对了但中间推理瞎编 —— 而 outcome-only reward 没法发现。这两个问题让 RCA 的 RL 训练在中后期容易陷入 plateau 或 reward hacking。SGS 论文已经证明，没有过程级质量信号，Conjecturer/Solver 必然朝着 reward 的漏洞收敛（生成 disjunctive 长结论刷分）。

我们想引入**过程级 reward** —— 把 RCA 推理过程拆成可验证的"假设跳"（hypothesis hop），每一跳由 Verifier 单独打分。理论支撑是 self_play_evolves §3.3 主动信息获取：Solver 应该学会主动选择最有诊断价值的证据查询，而不仅仅是猜对终态。

## Setting

输入：RCA Agent 完成一个 case 后产生的完整 trajectory（user task + assistant messages + tool calls + tool results）。
输出：每个 hop 的三个子分（warrant / plan_alignment / conclusion_validity）+ 整条 trajectory 的总过程分。
评估对象：用过程 reward 训练出的 RCA 是否在 (1) F1 不降的前提下 (2) 显著降低幻觉率 (3) 比 outcome-only baseline 更难 hack。

"hop" 的语义定义：一个完整的"形成假设 → 计划证据 → 多次工具调用 → 收尾结论"的语义单元。一个 hop 通常跨 3-10 次工具调用。

## Modeling approach

把 RCA 推理过程建模为 abduction-as-MDP：

```
hop_i = ⟨hypothesis_i, evidence_plan_i, tool_calls_i+, conclusion_i⟩
```

每个 hop 由 Verifier（Phase 1-2 用商业 LLM judge，Phase 3+ 训练专用 verifier）打三个子分：
- **warrant**：该假设在当时已有 evidence 下是否合理（vs 凭空 hallucinate）
- **plan_alignment**：要查的 evidence 是不是真能验证这个假设
- **conclusion_validity**：拿到 evidence 后下的结论有没有越界

最终 reward = `outcome_reward + λ × process_reward`，其中 `λ ≪ 1`（起步 0.2，做 ablation 扫到 0.5）。比 outcome 小是 anti-hacking 设计 —— 防止 RCA 学会"写华丽过程但答案错"。

**Hop 切分** 是这个 idea 内部的一个待定子问题，三种候选（RCA 自报 schema / Verifier 后切 / 两者并存做 ablation），我们倾向第三种作为消融实验。

## Related work

- **SGS** (Bailey et al., 2026)：用 Guide 给 Conjecturer 打过程分（relevance + cleanness），但他们的 Guide 在 Lean4 编译这种确定性场景下工作。我们要把这个思路推到 LLM-judged 软场景
- **PRM (Process Reward Model)** 系列：math/code 域已经有大量过程奖励工作，但都不是 abduction 任务
- **Pag / Recursive Introspection**（Qu et al., 2024 / Jiang et al., 2025）：multi-turn 自纠错，但 reward 仍是 outcome 级

## Differentiator vs 三篇 reference

| 维度 | self_play_evolves | GASP | SGS | 本 idea |
|---|---|---|---|---|
| 域 | code | code | Lean math | **microservice RCA (abduction)** |
| 过程信号 | 无显式 | 无 | Guide rho(x,x̃) 给问题打分 | **给推理 hop 打分** |
| Hop 概念 | 无 | 无 | 无 | 新引入 |
