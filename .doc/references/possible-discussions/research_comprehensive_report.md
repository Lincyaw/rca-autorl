# Self-Play Evolves 研究综述与微服务故障注入应用方案

> 综合报告：论文解读、文献图谱、同期工作分析与应用方案设计
> 生成日期：2026-04-30

---

# 第一部分：核心论文深度解读

## 论文基本信息

| 项目 | 内容 |
|------|------|
| **标题** | Self-Play Only Evolves When Self-Synthetic Pipeline Ensures Learnable Information Gain |
| **作者** | Wei Liu, Siya Qi, Yali Du, Yulan He |
| **机构** | King's College London, The Alan Turing Institute |
| **arXiv** | 2603.02218 |
| **投稿** | ICML 2026 |

---

## 一、核心问题诊断

当前大语言模型的"自演化"（self-evolving）研究非常火热，但作者指出**大多数现有系统本质上只是"自博弈"（self-play），而非真正的自演化**，它们在几轮迭代后就会迅速陷入停滞或崩溃。

### 典型失败现象

| 论文 | 失败现象 |
|------|---------|
| Zhao et al. (2025a) Absolute Zero | Proposer 倾向于生成极其简单的恒等问题（如 f(x)=x） |
| Huang et al. (2025) R-Zero | 模型性能早期达到峰值后迅速下降 |
| Chen et al. (2025a) | Proposer 需要精心调优的 prompt 才能保持在合理区间 |
| Yang et al. (2025); Lu et al. (2025) | 需要定期引入真实数据来重新校准 Verifier |

### 核心诊断

> **这些系统失败的根源不在于奖励优化不够，而在于虽然合成了更多数据，但数据中包含的"可学习信息"（learnable information）并没有在迭代中单调增加。**

模型只是在用不同形式重复自己已经知道的东西。

---

## 二、理论框架

### 2.1 三元角色（Triadic Roles）

| 角色 | 功能 | 对应合成方向 |
|------|------|-------------|
| **Proposer** | 生成任务 | 合成问题 + 参考答案 |
| **Solver** | 求解问题 | 合成解答 |
| **Verifier** | 评估解答并提供训练信号 | 合成反馈 |

**与现有范式的对比**：
- **P+S**（如 RLVR）：Verifier 是固定的规则，只适用于可验证领域
- **S+V**：在固定偏好数据上训练更好的 Verifier，不涉及自合成数据
- **P+S+V**（本论文）：同时进化三个角色，共享同一个基础模型

### 2.2 可学习信息（Learnable Information）

**为什么不用香农熵？** 香农熵度量总不确定性，但不区分"可复用的结构"和"随机噪声"。

**MDL（最小描述长度）**：
> 描述长度 = 模型描述长度 + 预测损失

**Epiplexity（认知复杂度）**：有界观察者下的 MDL

> P* = argmin_{P in P_{C,T}} { |P| + E[log(1/P(X))] }

其中：
- |P|：模型描述长度（参数容量成本）
- E[log(1/P(X))]：数据编码成本（预测损失）
- S_{C,T}(X) = |P*|**：可学习信息**
- H_{C,T}(X) = E[log(1/P*(X))]**：不可学习残余**

**关键洞察**：可学习信息不是数据的绝对属性，而是**相对于观察者的容量和计算预算**而言的。

---

## 三、三大系统设计原则

### 3.1 非对称协同进化（Asymmetric Co-evolution）

**核心观察**：在很多任务中，提出问题和验证答案远比解决问题容易。

**设计机制**——weak-to-strong-to-weak 闭环：
1. **Weak-to-strong**：当前模型的 proposing/verifying 能力（较弱但可行）用来监督训练更强的 Solver
2. **Strong-to-weak**：Solver 提升后，反向同步回内部环境，使 Proposer 和 Verifier 跟上 Solver 的前沿水平
3. 形成渐进式的"不对称阶梯"

### 3.2 容量增长（Capacity Growth）

模型必须持续扩展有效容量预算：
- **参数容量 C(t)**：更多参数、LoRA 秩增长、稀疏激活组件
- **推理时间预算 T(t)**：更长的推理链、更多推理步骤

### 3.3 主动信息获取（Proactive Information Seeking）

现有系统的三种局限：
- 零数据系统：信息源被限制在当前权重中
- 固定数据集系统：退化为对静态语料的微调
- 固定外部机制系统：迭代无关的上下文附加

**设计**：将信息获取视为内部环境的显式责任，每轮迭代主动选择外部上下文。

### 三个机制的协同关系

| 模块 | 管道角色 | 功能 |
|------|---------|------|
| **非对称协同进化** | **Generator** | 将不可学习噪声转化为可学习结构，创造"信息势能" |
| **容量增长** | **Receiver** | 扩展假设空间，内化新暴露的可学习信息 |
| **主动信息获取** | **Open Feeder** | 持续注入新鲜熵和上下文 |

---

## 四、实验验证

### 实验1：不同角色、容量和合成方向的可学习信息

使用 Qwen 系列模型，在三种代码任务上测试：

| 发现 | 说明 |
|------|------|
| 更强的 Proposer 产生更多可学习信息 | Qwen2.5 7B -> 14B -> Qwen3 4B，epiplexity 递增 |
| Solver 容量与可学习信息呈倒U型 | 小模型被迫学结构，大模型可能直接记忆 |
| 不同合成方向信息差异巨大 | Induction >> Abduction > Deduction |

### 实验2：多轮自博弈中的信息演化

- 多轮自博弈后，信息量**并未稳定增长**，而是**剧烈波动**
- 没有显式机制时，模型无法实现持续进化
- 行为上表现为：Solver 能力下降 + Proposer 生成的问题模式崩溃

**这直接验证了论文核心诊断**。

---

## 五、核心结论

> "Prevailing stagnation in self-evolving systems stems not from insufficient reward optimisation, but from the failure to sustain a monotonic increase in learnable information for bounded observers."

> 自演化系统的普遍停滞并非源于奖励优化不足，而是未能为有界观察者维持可学习信息的单调增长。

---

# 第二部分：文献图谱

## 一、谁引用了这篇论文？

| 引用论文 | 作者 | arXiv | 相关性 |
|---------|------|-------|--------|
| Teaching LLMs to Use Private Libraries for Code Generation | 多人 | 2603.15159 | 代码自演化 |
| Towards Self-Improving Error Diagnosis in Multi-Agent... | Jiazheng Li, Yulan He等 | 2604.17658 | 多智能体自改进 |

由于论文刚上 arXiv 不到两个月，正式引用还很少，但预计 ICML 2026 结果出来后引用会快速增长。

## 二、核心参考文献网络（58条）

### 第一层：理论支柱（必读）

| 论文 | 作者/年份 | 核心贡献 | 与本论文关系 |
|------|----------|---------|-------------|
| **From Entropy to Epiplexity** | Finzi et al. / 2026 | 提出 Epiplexity | **核心理论基础** |
| **Weak-to-Strong Generalization** | Burns et al. / ICML 2024 | 弱监督激发强模型 | 支撑非对称协同进化 |
| **A Mathematical Theory of Communication** | Shannon / 1948 | 香农信息论 | 引出 MDL 必要性 |
| **MDL** | Rissanen / 1978 | 最小描述长度 | 区分结构与噪声 |

### 第二层：直接对标的 Self-Play/RLVR 工作

| 论文 | arXiv | 核心机制 | 本论文如何批判/借鉴 |
|------|-------|---------|-------------------|
| **Absolute Zero** | 2505.03335 | 零数据自博弈 | 主要对标，trivial identity problems |
| **R-Zero** | 2508.05004 | Challenger-Solver 协同进化 | 主要对标，non-sustained improvement |
| **SPELL** | - | 开放域自博弈 | 需 periodic ground truth recalibration |
| **Multi-Agent Evolve (MAE)** | 2510.23595 | P+S+J 三元角色 | 最接近的同期工程实现 |
| **SPICE** | 2510.24684 | 语料环境自博弈 | 支撑主动信息获取 |
| **Self-Rewarding LMs** | 2401.10065 | 自奖励机制 | 属于 S+V 范式 |
| **Self-play Fine-tuning** | 2401.09092 | 弱模型自博弈变强 | 早期工作 |
| **DeepSeek-R1** | 2501.12948 | RLVR 推理增强 | 外部 verifier 范式 |
| **STaR** | 2203.14465 | 自举推理 | 早期自监督训练 |

### 第三层：信息论基础

| 论文 | 作者/年份 | 核心内容 |
|------|----------|---------|
| **Logical Depth** | Bennett / 1988 | 逻辑深度 |
| **Complexity, Depth** | Koppel / 1987 | 算法信息论 |
| **Effective Complexity** | McAllister / 2003 | 有效复杂度 |
| **Resource-bounded Kolmogorov Complexity** | Allender et al. / 2011 | 资源有界 KC |

### 第四层：协同进化与环境设计

| 论文 | arXiv | 核心贡献 |
|------|-------|---------|
| **GenEnv** | 2512.19682 | 难度对齐协同进化 |
| **Cooper** | 2508.05613 | 策略与奖励协同优化 |
| **Bootstrapping Task Spaces** | 2509.04575 | 任务空间自举 |
| **Self-questioning LMs** | 2508.03682 | 自我提问机制 |

## 三、Epiplexity 引用网络（2026年新兴研究集群）

Finzi et al. (2026) 已被 10+ 篇论文引用：

| 引用论文 | arXiv | 应用场景 |
|---------|-------|---------|
| Group DRO-Driven RL for LLM Reasoning | 2601.19280 | Prompt 采样与计算分配 |
| Thermodynamic Limits of Physical Intelligence | 2602.05463 | 学习的热力学理论 |
| Data-efficient pre-training | 2603.18534 | 合成数据信息效率 |
| Winsorized DPO for Robust LLM Alignment | 2603.07211 | 对齐中的信息论 |
| Easy Samples Are All You Need | 2604.18639 | 数据高效 RL |
| A Thermodynamic Theory of Learning I | 2601.17607 | 认知成本 |

**本论文是首个将 epiplexity 系统应用于自演化系统动态分析的工作。**

## 四、关键论文阅读优先级

**必读（理论根基）**：
1. Finzi et al. (2026) - From Entropy to Epiplexity [arXiv:2601.03220]
2. Burns et al. (2024) - Weak-to-Strong Generalization [ICML 2024]

**必读（直接对标/同期）**：
3. Zhao et al. (2025a) - Absolute Zero [arXiv:2505.03335]
4. Huang et al. (2025) - R-Zero [arXiv:2508.05004]
5. Chen et al. (2025b) - Multi-Agent Evolve [arXiv:2510.23595]
6. Li et al. (2026) - R-Diverse [arXiv:2602.13103]
7. Jana et al. (2026) - GASP [arXiv:2603.15957]

**选读（拓展视角）**：
8. Liu et al. (2025a) - SPICE [arXiv:2510.24684]
9. Guo et al. (2025b) - GenEnv [arXiv:2512.19682]
10. Kuba et al. (2025) - Language Self-Play [arXiv:2509.07414]

**综述**：
11. Fang et al. (2025) - Comprehensive Survey [arXiv:2508.07407]
12. The Landscape of Agentic RL for LLMs (2026) [arXiv:2509.02547]

---

# 第三部分：同期工作深入分析

## 一、GASP: Guided Asymmetric Self-Play (Jana et al., 2026)

### 1.1 基本信息

| 项目 | 内容 |
|------|------|
| **作者** | Swadesh Jana, Cansu Sancaktar, Tomas Danis, Georg Martius, Antonio Orvieto, Pavel Kolev |
| **机构** | University of Tubingen, Max Planck Institute |
| **arXiv** | 2603.15957 |
| **会议** | ICLR 2026 RSI Workshop (Spotlight) + LLA Workshop |

### 1.2 核心动机

现有非对称自博弈（如 Absolute Zero）是 **goal-agnostic** 的：Teacher 只根据 Student 通过率优化，可能产生"很难但无聊"的问题。

> "not all problems that are hard to solve are interesting or informative"

这与 Liu et al. "合成更多数据但没有增加可学习信息"是同一现象。

### 1.3 核心方法：Goalpost-Grounded Lemma-Lift 课程

| 组件 | 说明 |
|------|------|
| **Goalpost** | 从 LiveCodeBench 筛选的 146 个最难真实问题（3轮过滤：Post-RL -> AZR-checkpoint -> Final RL，pass@100=0） |
| **Lemma** | Teacher 看到 Goalpost 后生成**更简单**变体，要求 pass rate in [0.3, 0.7] |
| **Lift** | Teacher**只看 Lemma、不看 Goalpost**，生成**更难**变体，要求 pass rate in [0.1, 0.5] |

**关键设计**：Lift 生成时不直接看 Goalpost，确保从 Student 当前 frontier 渐进提升。

### 1.4 难度调整双轴设计

- **I/O axis**：改变输入输出表示复杂度，保持底层算法不变
- **f axis**：改变算法本身复杂度

消融：去掉 I/O axis 后，Goalpost 解决数从 11 降到 4。

### 1.5 奖励设计

- **Lemma reward**：`[4p(1-p)]^5`，峰值 p=0.5
- **Lift reward**：`10p[(1-p)/0.9]^9`，峰值 p=0.1

### 1.6 Rejection Sampling

计算候选问题与全局 buffer 的嵌入余弦相似度，> 0.95 拒绝。没有这个过滤器，后期问题相似度急剧上升（模式崩溃）。

### 1.7 实验结果

- pass@20 on LCBv5：比 AZR 提升 **2.5%**
- 解决了所有 baseline 都无法解决的 hard goalpost 问题
- 两阶段课程（Lemma-Lift）比单阶段更有效

### 1.8 与 Liu et al. 的关系

| Liu et al. 设计原则 | GASP 的对应/空白 |
|---------------------|-----------------|
| 非对称协同进化 | **直接实现**，但缺少显式 strong->weak 同步 |
| 容量增长 | **空白**，固定 7B 模型 |
| 主动信息获取 | **部分实现**，静态 goalpost（留作未来工作） |
| 可学习信息度量 | **空白**，仅用 pass rate 代理 |

---

## 二、SGS: Scaling Self-Play with Self-Guidance (Bailey et al., 2026)

### 2.1 基本信息

| 项目 | 内容 |
|------|------|
| **作者** | Luke Bailey, Kaiyue Wen, Kefan Dong, Tatsunori Hashimoto, Tengyu Ma |
| **机构** | Stanford |
| **arXiv** | 2604.20209 |
| **领域** | Lean4 形式定理证明 |

### 2.2 核心动机

现有自博弈方法无法随计算量增长而持续学习：

> "Conjecturer learns to hack its reward, collapsing to artificially complex problems that do not help the Solver improve"

具体表现：生成"人为复杂但无意义"的问题（logically bloated, inelegant），导致 Distribution Collapse。

### 2.3 核心方法：三位一体 + Self-Guidance

| 角色 | 功能 | 与 Liu et al. 对应 |
|------|------|-------------------|
| **Solver** | 求解目标问题和合成问题 | = Solver |
| **Conjecturer** | 为未解决问题生成"更简单但相关"的子问题 | = Proposer |
| **Guide** | 评分合成问题（相关性 + 清晰/自然度） | ~ Verifier |

**关键创新**：Guide 是 frozen/fine-tuned reviewer model，评估 relevance 和 cleanness。

### 2.4 算法流程

```
每轮迭代：
1. 采样 batch B，分为 B_solved 和 B_unsolved
2. 对每个 x in B_unsolved，Conjecturer 生成 synthetic problem x̃
3. Solver 对所有问题生成 k 个 attempt，Lean4 编译器验证
4. 更新 Solver：REINFORCE^{1/2} — 只在 solve rate <= 0.5 的问题上做 LLH
5. 更新 Conjecturer：Reward = R_solve * R_guide
   - R_solve = 0 if solve rate=0 或 top 30% easiest; else 1 - solve rate
   - R_guide = Guide 评分 rho(x, x̃)
```

### 2.5 REINFORCE^{1/2}：Solver 熵管理的关键

**核心洞察**：Solver 的熵崩溃会饿死 Conjecturer。

| Solver 目标 | 现象 | 对 Conjecturer 的影响 |
|------------|------|----------------------|
| CISPO（分组 RL）| 熵快速崩溃 | solve rate 集中在 0 或 1 -> R_solve=0 -> 无梯度 |
| REINFORCE^{1/2} | 熵缓慢下降 | solve rate 保持 spread -> 稳定学习信号 |

这与 Liu et al. 的"可学习信息"概念直接相关：solve rate 只取 0 或 1 时，中间的可学习结构区间消失。

### 2.6 实验规模与 Scaling Law

- **6.3M generations**（约 6B tokens）
- 目标数据集被遍历 **230 次**
- 拟合 cumulative solve rate 的 scaling law

**结果**：
- 7B DeepSeek-Prover-V2 超过 **671B 同系列模型 pass@4**
- Asymptotic solve rate 比最强 RL baseline 高 **7%**
- 在 RL baseline 无法解决的 1346 道难题上，SGS 解决近 **10%**

### 2.7 消融实验

| 消融变体 | 结果 | 启示 |
|---------|------|------|
| No Guide | 问题长度爆炸 10 倍，80% 出现 disjunctive 结论；asymptotic 65.5% vs 67.1% | Guide 是防止 degenerate problems 的关键 |
| No Problem Conditioning | 不超过 RL baseline | 必须让 Conjecturer 知道目标 |
| Frozen Conjecturer | 性能降至 RL baseline | 固定分布快速饱和 |

### 2.8 与 Liu et al. 的关系

| Liu et al. 设计原则 | SGS 的对应/空白 |
|---------------------|-----------------|
| 非对称协同进化 | **直接实现**，共享权重同步 |
| 容量增长 | **空白**，固定 7B 模型 |
| 主动信息获取 | **空白**，完全固定数据集 |
| 可学习信息度量 | **部分对应**，用 solve rate 代理 |

---

## 三、三篇论文的统一视角

### 时间线

- Liu et al. (Self-Play Only Evolves): arXiv 2026-03-04
- GASP: arXiv 2026-03-16（晚 12 天）
- SGS: arXiv 2026-04-22（晚 Liu 一个半月）

### 引用关系

- SGS 引用了 GASP，将其列为"使用真实数据 ground 生成问题"的代表
- GASP 和 SGS 可能都是在没有直接看到 Liu et al. 信息论框架的情况下独立完成
- 三者殊途同归：都发现了自博弈的核心瓶颈，各自从工程角度提出解决方案

### 统一对照表

| Liu et al. 理论概念 | GASP 实现 | SGS 实现 |
|---------------------|----------|---------|
| PROPOSER | Teacher (lemma-lift) | Conjecturer (合成子问题) |
| SOLVER | Student | Solver |
| VERIFIER | 无显式角色 | Guide (质量评分) |
| 非对称协同进化 | Teacher <-> Student | Conjecturer <-> Solver |
| 防止信息饱和 | Rejection sampling + Goalpost | Guide + Problem conditioning |
| 容量增长 | 固定 7B | 固定 7B |
| 主动信息获取 | 静态 goalpost | 完全固定数据集 |
| Learnable Information | 无显式度量 | 无显式度量 |

---

# 第四部分：微服务故障注入场景应用方案

## 一、场景天然契合度分析

你的场景比论文讨论的 coding/math **更适合** Liu et al. 的框架。

### 1.1 三元角色的天然对应

| Liu et al. 角色 | 你的系统组件 | 为什么天然匹配 |
|----------------|------------|---------------|
| **Proposer** | 故障注入器 | 控制"题目"的生成：故障类型、注入位置、复杂度、时机 |
| **Solver** | 根因定位 Agent | 通过观测数据推理故障根因 |
| **Verifier** | 故障验证器 + 推理验证器 | 验证注入成功 + 验证推理链条正确 |

### 1.2 非对称性天然存在

- **注入故障**（Proposer）-> 相对机械化，知道 ground truth
- **验证注入成功**（Verifier）-> 检查系统行为是否符合预期，有明确判断标准
- **定位根因**（Solver）-> 需要复杂的跨服务关联推理、时序分析、因果推断

### 1.3 Ground Truth 天然存在

**故障注入的那一刻，你就知道 ground truth 是什么。**

这意味着：
- Verifier 可以是**完美的**
- 可以精确度量 Solver 的准确率
- 可以直接计算"哪些故障类型 Agent 已经掌握了"

---

## 二、系统架构设计

### 2.1 经典 Self-Play 闭环

```
Proposer (故障注入) --故障场景--> Solver (根因定位) --根因报告--> Verifier
       ^                                                              |
       |________________反馈: solve rate / 难度评估____________________|
```

**Proposer 奖励**：
- 故障注入成功：+1
- 根因定位 Agent solve rate in [0.3, 0.7]：+bonus
- solve rate = 0 或 = 1：惩罚

**Solver 奖励**：
- 根因定位正确：+1
- 推理过程合理（Verifier 评估）：+bonus

### 2.2 融入 Liu et al. 三个增强机制

#### ① Asymmetric Co-evolution

```
迭代 1: 单服务 CPU 飙升 -> Agent 轻松解决
迭代 2: 跨服务级联延迟 -> Agent 部分解决
迭代 3: 网络分区 + 超时风暴 -> Agent 困难
...
迭代 N: Proposer 根据 Agent 能力边界动态调整故障复杂度
```

#### ② Capacity Growth

| 维度 | 具体实现 |
|------|---------|
| 参数容量 | 7B -> 14B -> 32B 渐进增长 |
| 推理预算 | 推理步数/工具调用次数随复杂度增加 |
| 上下文长度 | Metrics window、log lines 逐步扩大 |
| 工具使用 | Metric query -> tracing -> log analysis -> profiling |

#### ③ Proactive Information Seeking

```
静态阶段：预设故障库中循环（网络/资源/代码bug）
      |
      v learnable information 饱和
主动获取阶段：
  -> 从生产环境真实事故报告提取新故障模式
  -> 从开源系统学习新故障类型
  -> 组合已知故障形成复合故障（A + B 的交互效应）
```

---

## 三、Verifier 的具体设计

### Level 1：结果验证（Pass/Fail）

| 验证项 | 方法 | 可信度 |
|--------|------|--------|
| 故障注入成功 | 检查系统指标是否出现预期异常 | 100% |
| 根因定位正确 | 对比 Agent 输出 vs ground truth | 100% |
| 定位精度 | 精确到服务/实例/代码位置 | 可量化 |

### Level 2：推理过程验证

| 验证维度 | 具体方法 |
|---------|---------|
| 推理链完整性 | 是否检查了所有相关服务的指标？ |
| 因果逻辑正确 | 从观测到根因的推断是否符合因果方向？ |
| 无冗余步骤 | 是否有不必要的探测步骤？ |
| 证据充分性 | 结论是否有足够的观测数据支撑？ |
| 假设检验 | 是否排除了其他可能的根因？ |

**实现建议**：用更强的 LLM（或规则引擎）作为推理验证器，参照 Lean4 的 proof checking 思路。

---

## 四、需要特别注意的问题

### 问题 1：故障类型的有限性 -> Information Saturation

**症状**：5 轮后 Agent 掌握全部基础故障 -> learnable information = 0 -> plateau

**解决方案**：
- 故障组合：单故障 -> 双故障 -> 三故障级联
- 故障演化：同样故障在不同拓扑下表现不同
- 真实数据注入：从生产事故学习新 signature
- 时序复杂度：并发注入、交错注入

### 问题 2：Proposer 的"难度盲目"

**症状**：产生 Agent 已掌握（太简单）或完全不会（太难）的故障

**解决方案**：
- 维护"Agent 能力地图"
- 优先选择 solve rate in [0.3, 0.7] 的故障（learnable band）
- 设计渐进式故障序列（类似 GASP 的 lemma-lift）

### 问题 3：Solver 的"模式记忆"

**症状**：Agent 记住常见故障模式而非真正推理

**解决方案**：
- 变化观测数据：同样故障提供不同 metrics subset
- 引入干扰指标：加入无关但看似相关的异常指标
- 推理链强制输出：要求完整推理步骤，由 Verifier 验证
- Epiplexity 监控：区分记忆与泛化

### 问题 4：Strong->Weak 同步

**症状**：Solver 变强后，Proposer 生成的故障都太简单

**解决方案**：
- Proposer 也是 LLM Agent，根据"Agent 已掌握的故障"设计新故障
- 定期用更强的故障注入策略替换旧的
- Proposer 奖励与"故障是否在 learnable band"挂钩

---

## 五、最小可行实验（MVP）路线图

| 阶段 | 目标 | 时间估计 |
|------|------|---------|
| **Phase 1** | 搭建故障注入 + 根因定位的 single-shot pipeline | 2-3 周 |
| **Phase 2** | 加入 Verifier（两级），形成 self-play 闭环 | 1-2 周 |
| **Phase 3** | 实现 Asymmetric Co-evolution（Proposer 根据 solve rate 调整难度） | 2 周 |
| **Phase 4** | 引入 epiplexity 监控，诊断 learnable information 是否增长 | 2-3 周 |
| **Phase 5** | 实现 Capacity Growth（模型/推理预算渐进增长） | 3-4 周 |
| **Phase 6** | 引入真实故障数据，实现 Proactive Information Seeking | 持续 |

---

## 六、后续研究方向（基于文献空白）

### 方向 1：难验证域的非对称协同进化
- 将 P+S+V 从 math/code 推广到 general reasoning
- 设计通用的 Verifier-free 或 LLM-as-a-Judge 反馈机制

### 方向 2：容量增长的动态预算分配
- 根据 epiplexity 增长率自适应调整 LoRA rank 和推理 budget
- 相关起点：Progressive Stacking、Mixture-of-recursions

### 方向 3：主动信息获取的具体实现
- 结合不确定性估计和知识缺口检测
- 检索增强自演化（RAG + self-play）

### 方向 4：Epiplexity 的高效在线估计
- 开发轻量级在线估计器
- 作为 early stopping 或数据选择的信号

### 方向 5：多样性幻觉与信息饱和的统一理论
- 用 epiplexity 解释 R-Diverse 的 diversity illusion
- 信息论指导 diversity 设计

### 方向 6：从 Self-Play 到 Open-Ended Evolution
- 结合 SPICE 的 corpus 环境、GenEnv 的环境演化
- 构建真正开放边界的自演化系统

---

# 第五部分：附件清单

本报告配套材料：

| 文件 | 说明 |
|------|------|
| `research_comprehensive_report.md` | 本综合报告 |
| `self_play_evolves.pdf` | 原始论文：Liu et al. (2026) |
| `literature_graph_self_play_evolves.md` | 文献图谱（详细版） |
| `gasp.pdf` | GASP 论文 (Jana et al., 2026) |
| `sgs.pdf` | SGS 论文 (Bailey et al., 2026) |

---

> 本报告由 Kimi 基于以下材料整理生成：
> - Liu et al. (2026) 论文全文解读
> - GASP 和 SGS 论文全文分析
> - Web 搜索结果（arXiv、Google Scholar）
> - 用户讨论：微服务故障注入场景设计
>
> 生成日期：2026-04-30
