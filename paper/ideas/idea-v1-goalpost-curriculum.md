---
id: idea-v1-goalpost-curriculum
date: 2026-04-30
status: active
parent: idea-v0-rca-self-evolves
---

# Goalpost-Grounded Lemma-Lift Curriculum for Fault Injection

## Motivation

现有 spec 里 Fault Injection (FI) Agent 的 reward 是 `rca_defeat × validity × blast_bonus`。这个公式在单个 case 上看起来合理，但放到训练流里会出 GASP 论文已经验证的两个塌缩模式：(a) FI 一直生成 RCA 完全做不出的极端注入（`p=0`，没有梯度，Solver 学不动），(b) FI 一直生成 RCA 都能做出的简单注入（`p=1`，没有信息量）。

更深的问题：FI 的 "高价值" 定义如果只用 `rca_defeat`，是 **goal-agnostic** 的 —— FI 会找到 RCA 的某些奇怪盲区然后反复注入这种"难但无聊"的故障，对真实生产场景没有意义。GASP 在 code 域上已经定量证明了这个失败模式，并给出解药：用真实数据中的难 case 作为 goalpost 来 ground 整个生成过程。

## Setting

输入：(a) 一个 goalpost 集合 H = "当前商业模型 RCA 都搞不定且对 SO 有重大影响的故障 case"；(b) 当前 RCA Agent 的实时 pass rate 估计。
输出：FI 生成的注入方案，要求落在 learnability band 内（lemma `p∈[0.3,0.7]`，lift `p∈[0.1,0.5]`）。
评估对象：用这种课程训出的 FI vs 仅用 `rca_defeat` reward 的 FI，在 (1) RCA 训练曲线斜率、(2) goalpost 解决率、(3) 注入多样性（不塌缩）三个维度的差异。

## Modeling approach

完全照搬 GASP 的 lemma-lift 双阶段课程，平移到故障域：

1. **Goalpost 筛选**：从 RCABench 难 case + 真实生产事故里筛 "商业模型 RCA pass@N=0 ∧ economic_loss_proxy 高" 的子集
2. **Lemma 阶段**：FI 看到 goalpost h 后生成更简单变体 ℓ₀（譬如降低传播跳数、减少 confounder、锁定单服务）。Reward = learnability(p_ℓ₀) ∝ [4p(1-p)]^k，峰值 p=0.5
3. **Lift 阶段**：FI **只看 ℓ₀ 不看 h** 生成更难变体 ℓ₁，逼近 RCA 当前 frontier。Reward 峰值 p=0.1（更严苛 band）
4. **Solver 训练阶段**：RCA 在 ℓ₀ ∪ ℓ₁ 上做 RL 更新

故障域的难度双轴（对应 GASP 的 I/O 轴 + f 轴）：
- **拓扑轴**：blast radius / 受影响服务数 / 跨子系统传播
- **propagation pattern 轴**：cascade 深度 / confounder 数量 / silent failure 程度

注入有效性 (Verifier task-1) 作为 SGS 风格的 quality guard —— 任何 invalid 注入直接 reward 归零，相当于 SGS 的 R_solve × R_guide 乘法结构。

## Related work

- **GASP** (Jana et al., 2026)：直接基础，方法几乎照搬
- **SGS** (Bailey et al., 2026)：Guide 防止 Proposer reward hacking，对应我们的 Verifier task-1
- **现有 spec §5.3**：原 reward 公式作为 baseline 对比

## Differentiator vs GASP

| 维度 | GASP | 本 idea |
|---|---|---|
| Goalpost 来源 | LCB 难题（pass@100=0）| **真实生产事故 + RCABench**（更接 production 痛点）|
| 难度双轴 | I/O 轴 + f 轴 | **拓扑轴 + 传播 pattern 轴** |
| Validity guard | code 编译/test pass | **注入观测验证 + 经济损失 proxy** |
| 学生任务 | code generation | **abduction (RCA)** |
