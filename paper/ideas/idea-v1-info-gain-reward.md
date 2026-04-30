---
id: idea-v1-info-gain-reward
date: 2026-04-30
status: active
parent: idea-v0-rca-self-evolves
---

# Diagnostic Information Gain as Per-Step RCA Reward

## Motivation

RCA 的本质是 abduction：观测 → 候选根因集合 → 选证据缩小集合 → 再观测 → 收敛到根因。这个循环你（用户）已经描述得很清楚 —— "假设从现有信息获得，证伪假设的过程获得新信息，refine 假设"。这正是 self_play_evolves §3.3 主动信息获取（proactive information seeking）在 RCA 域的具体形态。

但现有 reward 设计完全没体现这件事 —— RCA 怎么查证据、查得多准、查得多浪费，全都不进 reward。结果是 RCA 学不会"挑信息量大的工具调用"，只会粗暴遍历或机械跟随 prompt 套路。这不仅效率低（avg_turns 上不去），更重要的是 —— 它学不会**主动诊断推理**这种核心能力，遇到训练分布外的故障会直接退化成猜。

## Setting

输入：RCA 在一个 case 上完成的整条 trajectory（含每次 tool call 前的当前 belief / 候选根因集合）。
输出：每次 tool call 的"诊断价值分" —— 该次调用前后 RCA 对根因的不确定性下降了多少。
评估对象：用 info-gain reward 训练的 RCA vs outcome-only baseline，在 (1) avg_turns（效率）、(2) tool_call_utility（每次工具调用的有效率）、(3) 训练分布外 case 的 F1 三个维度的差异。

## Modeling approach

理论上理想的信号是 **Bayesian information gain**：每次 tool call 前后根因 posterior 的熵下降。但在线计算太重（要维护精确 belief）。我们用一个轻量代理：

`step_info_gain(t) = α × (该 tool call 结果在后续 hop 中被引用) + β × (该结果触发了假设 refine 而非维持)`

两个二元判定都由 Verifier 在 episode 结束后回看 trajectory 给出。每次 tool call 拿到一个 step reward，`info_gain_total = Σ step_info_gain(t)`。最终 reward = `outcome_reward + λ_proc × process_reward + λ_info × info_gain_total`，三者权重做 ablation。

**为什么和 idea-v1-process-verifier 不重复**：
- process-verifier 是给"假设跳的逻辑 soundness"打分（每个 hop 一次）
- info-gain 是给"工具调用的探索价值"打分（每次 tool call 一次，更细粒度）
- 两者粒度不同、信号不同，可以叠加，也可以单独消融

## Related work

- **self_play_evolves §3.3** (Liu et al.)：proactive info seeking 的理论提案，本 idea 是它在 RCA 域的实例化
- **Pag** (Jiang et al., 2025) / **Recursive Introspection** (Qu et al., 2024)：multi-turn 自纠错有相关思想，但他们的 step reward 还是来自答对/答错，不是来自信息论意义上的 gain
- **tool-use efficiency**（agentic RL 文献里若干工作）：avg_turns / tool calls 优化是常见 eval 指标，但很少进 reward

## Differentiator

把 RCA 的 abduction 循环显式建模成 information-seeking MDP，并把 information gain（哪怕是 cheap proxy 版）放进 reward。论文层面的贡献点：在一个 abduction-heavy 真实任务上，第一次定量回答 "process reward + info gain reward 是否真的能让 Solver 学会更好的探索策略" 这个 §3.3 留下的开放问题。
