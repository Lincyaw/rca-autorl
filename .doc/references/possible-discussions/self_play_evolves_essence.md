# Self-Play Only Evolves When... — 精华笔记

> **论文**：Liu, Qi, Du, He. *Self-Play Only Evolves When Self-Synthetic Pipeline Ensures Learnable Information Gain.*
> KCL & Alan Turing Institute, arXiv 2603.02218, 2026-02-10. Position paper（不是实验论文）。
>
> **本笔记定位**：精简到一页量级，便于本项目后续随手翻阅。深度分析见同目录 `research_comprehensive_report.md`，引用图谱见 `literature_graph_self_play_evolves.md`。

---

## 一句话核心

Self-evolution loop 能否持续，**唯一**充要条件是：自合成数据流的 *learnable information* 在迭代中**单调上升**。reward shaping、scaling 单独都不够 — 只能保证表面任务指标涨，不保证可学结构涨。

## 框架（Fig. 2）

一个 LLM 在循环中扮演 3 个角色（**Proposer / Solver / Verifier**），共享同一份 base 权重。三者构成 *internal environment*，与 *external environment* 通过 §3.3 proactive seeking 双向交互。

```
External env  ──Introduce──►  Information  ──Transform──►  Learnable Info
                                  ↑                              │
                                  │                              ▼
                          Proposer + Verifier  ◄────►  Solver  (梯度只用 Learnable 部分)
                            (internal env)         strong→weak sync
```

---

## 必须记住的概念

### Learnable information（学得到的结构）

- 用 **epiplexity** $S_{C,T}(X)$ 度量：在参数预算 $C$、推理预算 $T$ 下，bounded observer 能从数据 $X$ 提取的**可压缩结构**量。
- 互补量是 **bounded entropy** $H_{C,T}(X)$ — 在该预算下"看起来仍像噪声"的部分。
- $\mathrm{MDL}_{C,T}(X) = S_{C,T}(X) + H_{C,T}(X)$。
- **Goldilocks zone**（甜点区）：data 既不能太简单（low S, low H — 没东西学），也不能太混沌（low S, high H — 学不动）；要"复杂到非平凡，但结构到可学"。
- 关键性质：epiplexity 是**相对于观察者**的，不是数据的绝对属性 — 同一份 data 对不同 capacity 的 model 表现为不同的结构 / 噪声比例。

### Information asymmetry（论文的"杠杆"）

- 在很多任务里，**propose+verify 比 solve 计算上更便宜**（math/code 是典型；healthcare 是反例 — 诊断比 verify 容易）。
- 形式化：若 $Y = f(X)$ 中 $f$ 多项式可算但反演困难，
  $H_{\text{poly}}(X|Y) - H_{\text{poly}}(Y|X) \geq c \log n$
- 这个 gap 是 learnable information 的"可挖掘体积"。

---

## 论文核心：三个必要的 system-level 设计

### §3.1 Asymmetric Co-Evolution

- 利用 propose / solve 的计算 gap，弱 P/V 监督强 S（**weak→strong**）。
- Solver 进步后必须**反向同步**回 P/V（**strong→weak synchronisation**）。否则 Proposer 漂出 frontier、Verifier 变废 → loop collapse。
- 实施手段：把 synthetic directions 组织成 **asymmetry ladder**（小 gap → 大 gap → 反向 gap），按 Solver frontier 渐进推进。
- "Intelligence Sync" 动作：用强 Solver 反向蒸馏 / 训练弱 P/V（图 4）。

### §3.2 Capacity Growth

- 必须让 $C^{(t)}, T^{(t)}$ **随迭代增长**，否则 learnable information 涨了 observer 装不下，loop 停滞。
- $C$ 增长：渐进堆叠层、加专家、LoRA rank 提升、活跃子集扩张。
- $T$ 增长：推理 token 长度、自适应递归深度。
- 失败模式：固定 $C^{(t)}$ → loss 平台、loop 转向 trivial 任务以适应当前 budget。

### §3.3 Proactive Information Seeking

- 纯闭环必然枯竭（**信息守恒** — Shannon transformation 不产生新信息）。
- 三种实现策略：
  1. **Learn to ask** — Proposer 根据 Solver failures / Verifier disagreement 主动检索 $d$，合成依赖 $d$ 的题目。
  2. **Turn context into asymmetry gaps** — 相同 $d$ 下生成多种难度合成方向，按 curriculum 推进，而不是把检索结果当 hint 直接喂。
  3. **Co-evolve external env** — retrieval / memory 模块也跟 P/V 一起更新，不是固定的。

---

## 实验（小规模诊断性，**不是大规模 SOTA 论文**）

### Exp 1（图 5）：epiplexity vs capacity 矩阵

3 类合成任务（induction / abduction / deduction）× 不同 size 的 Proposer/Solver。

- **强 Proposer ⇒ 合成数据 learnable info 更高**。
- **Solver size 与 epiplexity 是先升后降的"甜点曲线"** — 太小学不到结构，太大直接记忆，跳过结构学习。
- 三类任务难度：**induction > abduction ≈ deduction**（induction 含最多 learnable info）。

### Exp 2（图 6）：纯 RL self-play 没有显式信息增长机制时

- 三类任务上 epiplexity **剧烈震荡，不单调上升**。
- 印证：reward optimization 单独不够，必须叠加 §3.1 / §3.2 / §3.3。

---

## 失败模式总览（→ 本项目要避开）

| 失败模式 | 触发条件 | 对应到本项目的预防 |
|---|---|---|
| Mode collapse | 纯 P+S 闭环无外部锚定 | 必须接 RCABench / 真实事故做 ground info |
| Trivial identity-like tasks | Proposer 无 frontier 压力，退化成 $f(x)=x$ | FI 硬约束："被当前 RCA 定位不到" + "经济损失 proxy > 阈值" |
| Verifier 漂移 | 多 reward RL 训 V 时不稳，自洽信号不保证 V 进步 | Verifier 训练至少部分挂钩 ground truth（FI 是否真生效），不能纯 LLM-as-judge |
| Capacity saturation | $C, T$ 固定 ⇒ loss 平台、loop 转向 trivial | 4 个 agent 各自有训练预算可调（base 选型留待实验决定），推理预算需随阶段抬升 |
| Static context | RAG 接死 ⇒ 早期超 budget、后期无新结构 | Goalpost 课程必须随 RCA 进步动态更新（已写进 `goalpost-curriculum/v1`） |

## Alternative views（论文逐一驳斥）

- **Reward shaping** — 必要不充分；reward 可被 hack 而 learnable info 不涨。
- **Curriculum** — conflates 难度因子，"难度涨" ≠ "信息涨"。
- **Agent–env co-evolution** — 环境复杂化 ≠ learnable structure 增加。
- **Scaling** — capacity 放大可能只是放大记忆，不一定放大可学结构。

## 论文承认的 open problems（→ 本项目的"机会窗口"）

1. **Hard-to-verify 域** — asymmetric co-evolution 目前只在 math/code 验证。RCA 是**部分可验证**（FI 注入有 ground truth，推理链没有）⇒ 我们的 process verifier idea 正打在这个空白上。
2. **Online epiplexity 估计** — 当前 prequential coding 太重，只能离线诊断；轻量在线估计是开放工具型问题。
3. **Proactive info seeking 的可落地实现** — 论文承认 "recognise what model doesn't know" 是公开难题 ⇒ 我们的 info-gain-reward idea 是该方向在 RCA 域的实例化。

---

## 与本项目的映射

| 论文 mechanism | 本项目 idea | 文件 |
|---|---|---|
| Triadic role + 显式 data-level 同步（不依赖共享权重假设） | v0 整体架构 | `paper/ideas/triadic-self-play/v0-overview.md` |
| Asymmetric co-evolution + strong→weak sync | data-level 显式同步机制 | `triadic-self-play/asymmetry-ladder/v1.md` |
| Verifier 提供 process supervision（§3.1 末段） | hop 三子分（warrant / plan / conclusion） | `triadic-self-play/process-verifier/v1.md` |
| Proactive info seeking（§3.3） | per tool-call 信息增益 reward | `triadic-self-play/info-gain-reward/v1.md` |
| Goldilocks zone curriculum（§2.2 + §3.3.2） | GASP 风格 lemma-lift goalpost | `triadic-self-play/goalpost-curriculum/v1.md` |
| Capacity growth（§3.2） | **尚未对应** — 后续研究空白 | 待写 v1 |

> ⚠️ Capacity growth 这条线我们暂时没认领。短期内不必造对应的 idea 文档；中期如果实验观察到 RCA 在中后期出现 plateau，再回头把 §3.2 落到具体 idea。

---

## 项目 elevator pitch 中怎么用这篇论文

> "Self-Play Evolves 是一篇 position paper，提出 triadic role 下三个必要的 system-level 设计（asymmetric co-evolution / capacity growth / proactive info seeking），但只在 math/code 易验证域做了诊断性小实验。本项目在 RCA 这个 abduction-heavy 的**半验证域**首次系统实例化它的全部三个机制，并把论文里'靠共享权重隐式做 strong-to-weak 同步'这个假设替换成 data-level 显式同步机制。"
