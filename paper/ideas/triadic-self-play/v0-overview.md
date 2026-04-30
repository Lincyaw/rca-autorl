---
id: idea-v0-rca-self-evolves
date: 2026-04-30
status: active
parent: null
---

# Self-Evolving Microservice RCA via Triadic Self-Play

## Motivation

微服务故障根因定位（RCA）当前主要依赖人工运维经验或基于规则/小模型的启发式定位，泛化能力差、对未见过的故障模式无效。本项目想用 self-evolving LLM 范式来解决这件事 —— 借鉴 Liu et al. (2026) 提出的三元角色 Proposer/Solver/Verifier 框架（self_play_evolves.pdf），把"故障注入 vs 故障定位"这对天然非对称任务组织成一个能持续提升 RCA 能力的训练循环。这个场景比论文里的 code/math 更适合论文的理论框架：故障注入侧天然知道 ground truth，验证侧有明确判定标准，定位侧需要复杂跨服务推理 —— 三者的信息不对称是结构性的，不是人为构造的。

## Setting

研究对象是一组围绕微服务 RCA 训练的 agent，由 4 个相对独立的角色组成：

1. **RCA Agent (Solver)**：从 metrics/traces/logs 推理根因，需要可验证的推理过程而非幻觉式猜测
2. **Fault Injection Agent (Proposer)**：生成"高价值"故障 —— 即对系统服务影响大且当前 RCA 定位不到的注入方案
3. **Verifier**：双重职责 —— (a) 验证注入是否真的生效并产生经济损失（防止 FI 刷分），(b) 验证 RCA 推理过程是否 sound（提供过程级 reward）
4. **Controller**（平行研究线）：当 RCA 是不可训练的黑盒商业 LLM 时，训练一个小模型 wrapper 来规范并提升其行为

数据底座：RCABench（已有）+ 后续接入的真实生产事故记录。训练框架：基于 AReaL 的 RL pipeline。

## Modeling approach

**关键设计决定**：strong-to-weak 同步用 **data-level 机制** 显式实现，而不依赖 self_play_evolves 论文里"共享权重自动同步"的隐式假设。原因：4 个 agent 输入输出 schema 异质（trace 推理 / 离散注入 / judge / wrapper），weight-level 同步在异质任务下不一定有效（即便共享 init，分别 RL 后参数会快速分歧）。把同步机制显式做在数据层，无论后续选共享还是独立的权重方案，都有一条可监控、可干预的同步通道：

- **Goalpost 漂移**：FI 的训练集动态维护为"当前 RCA 还搞不定的难 case"
- **共享 replay buffer**：RCA 失败 trace 同时是 FI 正样本和 Verifier 难 judge 样本
- **周期性蒸馏**（可选）：把 RCA-N 的能力蒸馏为 prompt prior 喂给 FI 和 Verifier

整体研究分解为 4 个核心子 hypothesis（见同级子目录 `process-verifier/`、`goalpost-curriculum/`、`asymmetry-ladder/`、`info-gain-reward/`）+ 1 条平行线（`../controller-harness/`）。

## Related work

- **self_play_evolves** (Liu et al., 2026)：理论根基，提供 epiplexity 度量和三元组框架
- **GASP** (Jana et al., 2026)：lemma-lift goalpost-grounded 课程，影响 `goalpost-curriculum/`
- **SGS** (Bailey et al., 2026)：Guide model + REINFORCE^(1/2)，影响 `process-verifier/` 和 RCA 训练目标选择
- **现有 RCA 工作**（RCABench 等）：提供 baseline 和数据底座

## Open questions（待 v1 ideas 收敛后回答）

- Step / hop 的精确切分由谁负责（RCA 自报告 vs Verifier 后切分 vs 两者）—— 当前倾向第三种作消融，见 `process-verifier/v1.md`
- Process reward 相对 outcome reward 的权重起点 —— 起步 0.2，扫到 0.5，见 `process-verifier/v1.md`
- Goalpost 数据源（RCABench 难 case + 真实生产事故），见 `goalpost-curriculum/v1.md`
- 4 agent 是否共享同一个 base checkpoint —— 不预设，留给 stage kickoff 时根据实验决定
