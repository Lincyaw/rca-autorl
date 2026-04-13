# Multi-Agent RL Training Specification

> 原始需求文档 — 2026-04-12
> 状态：draft

---

## 1. 系统目标

训练一组专精 agent，形成微服务故障分析与注入的完整 pipeline。所有 agent 共享同一个 base model，通过不同的 agent 框架（TaskAdapter + AgentRuntime + RewardStrategy）赋予不同角色能力，独立训练，组合评估。

最终目标：通过 Fault Injection Agent 与 RCA Agent 的对抗训练，持续提升 RCA 能力的覆盖面，同时发现系统中高风险的故障盲区。

---

## 2. Agent 角色定义

### 2.1 RCA Agent（根因定位）

**职责**：给定微服务系统的可观测数据（metrics, traces, logs），定位故障根因，输出因果图。

**输入**：
- 故障场景描述（incident description）
- 可观测数据目录（telemetry data dir）
- 可用工具集（查询 metrics, 读 traces, 搜索 logs 等）

**输出**：
- 因果图（nodes, edges, root_causes, component_to_service）
- 根因列表

**工具调用**：通过 AgentM 框架的工具接口查询可观测数据

**Reward**：root-cause F1（预测根因集合与实际根因集合的 F1 分数）

**当前状态**：已实现（RCATaskAdapter + AgentMRuntime + RootCauseMatchRewardStrategy）

---

### 2.2 World Model Agent（世界模型 / 因果推理器）

**职责**：给定当前系统状态 snapshot（State_t）和一个 action（故障注入动作），综合代码和可观测信息，预测注入后的状态变化（State_{t+1}）——哪些服务受影响、指标如何波动、日志/trace 变化、故障如何传播。

**输入**：
- 注入前的系统状态 snapshot（完整的 metrics, traces, logs）
- 一个 action（故障注入动作描述）
- 代码仓库访问权限
- 服务拓扑信息

**输出**（多粒度预测）：
- L1: 受影响的服务列表
- L2: 具体指标变化方向（哪个 metric 升/降）
- L3: 代码/span 级变化（可选）
- 传播关系：A 的异常导致 B 的异常（因果边）
- 置信度

**工具调用**：
- 代码阅读（读取微服务源代码，理解依赖关系和调用链）
- 可观测数据查询（验证预测与实际观测是否一致）
- 拓扑查询（服务间依赖、网络拓扑等）

**Reward**：基于注入前后 snapshot 自动对比——node_f1（服务级命中）+ metric_direction_accuracy（方向匹配）+ edge_f1（传播边命中），详见 §5.2

**关键特性**：
- 基于离线训练数据（历史注入 case 的前后 snapshot 对）训练
- 不仅是代码静态分析——需要结合运行时可观测数据做动态推理
- 需要建模混杂变量（confounders）和中间态
  - 例：代码注入 bug → 触发 CPU 异常升高 → CPU 升高导致延迟上升
  - CPU 异常升高是中间态 action，也是混杂变量
- 验证通过 Verification Agent 对比预测与实际 snapshot 完成

**当前状态**：未实现

---

### 2.3 Fault Injection Agent（故障注入 / 红队）

**职责**：给定当前系统信息，决定哪些故障注入最有价值——具体来说，寻找**影响面大且 RCA Agent 当前无法定位**的故障。

**输入**：
- 系统拓扑和服务信息
- RCA Agent 的历史表现（哪些故障模式被成功定位，哪些失败）
- 可用的故障注入手段（CPU 注入、网络延迟、代码 bug、资源耗尽等）
- 业务影响模型（哪些服务更关键，影响面更大）

**Action space**：离散选择题——候选项组织为预制的树形结构（故障大类 → 目标服务 → 具体参数），FI Agent 在树中导航选择，选择后附带文本解释（reasoning）。

**输出**：
- 故障注入选择（从树形结构中选取的路径）
- 文本解释（选择原因和预期影响）

**Reward**：对抗性 reward（乘法结构）——
- `rca_defeat × injection_validity × blast_bonus`
- 无效注入 → reward = 0（不管 RCA 是否失败）
- RCA 成功定位 → reward = 0（不管影响面多大）
- 初期 RCA 对手用商业模型（水平稳定，提供有意义的对抗信号），后期替换为训练好的 RCA 模型
- 详见 §5.3

**对抗关系**：Fault Injection Agent 是 RCA Agent 的红队。两者形成对抗训练循环——
- FI Agent 找 RCA 的盲区 → RCA 被迫扩展覆盖面 → FI Agent 被迫找新的盲区

**当前状态**：未实现

---

### 2.4 Verification Agent（验证 / 全链路裁判 / 评估基础设施）

**职责**：作为 pipeline 的评估基础设施，为其他 Agent 提供判定和 reward signal。三重验证——

1. **注入可观测性验证**：注入后系统的实际可观测行为是否符合预期（如注入 CPU 异常后，metrics 上确实看到 CPU spike）
2. **RCA 定位准确性验证**：对比故障注入的 ground truth 与 RCA Agent 的定位结果——通过 LLM-as-judge 语义匹配，支持自由文本输出和格式容忍
3. **World Model 预测验证**：对比 World Model 的预测与实际 snapshot 变化

**输入**：
- 故障注入方案 + 实际注入记录（来自 Fault Injection Agent，作为 ground truth）
- 注入前后的可观测数据 snapshot
- World Model 的预测输出（如有）
- RCA Agent 的定位输出（因果图 + 根因列表）

**输出**：
- 注入验证结果（有效 / 部分有效 / 无效）
- RCA 定位验证结果（正确 / 部分正确 / 错误 + 具体差异分析）
- World Model 预测验证结果（各粒度的命中情况）
- 综合质量评分

**实现方式**：预置规则代码 + Agent 自主判定混合模式。规则覆盖常见故障类型的已知传播模式（确定性、零噪声），Agent 自主判定处理规则未覆盖的长尾。规则可通过正循环持续沉淀（Agent 自主判定 → 人工审核 → 沉淀为规则代码）。

**训练策略**：Phase 1-2 **不训练**，完全由商业模型驱动。Phase 3-4 根据积累的数据和需求决定是否训练自己的 Verification 模型。详见 §5.4。

**作为 reward provider 的角色**：

Phase 1-2 由商业模型驱动，为其他 agent 提供 reward signal：
- 为 RCA Agent 提供 reward：LLM-as-judge 语义匹配（Phase 2 起替代硬编码 F1）
- 为 FI Agent 提供 reward：injection_validity 判定
- 为 World Model 提供 reward：snapshot 观测比对
- 远期（Phase 3-4）：训练好的 Verification 模型可提供 learned reward signal，捕捉硬编码指标无法覆盖的细粒度判定

**当前状态**：未实现（Phase 1 使用硬编码 F1 作为 RCA reward，不依赖 Verification）

---

## 3. Pipeline 架构

### 3.1 训练阶段（独立）

每个 agent 独立训练，各自有自己的 TaskAdapter + AgentRuntime + RewardStrategy：

```
Base Model (e.g. Qwen2.5-7B)
    ├── + RCA Agent Framework      → RL train (F1 reward)
    ├── + World Model Framework    → RL train (prediction accuracy)
    ├── + Fault Injection Framework → RL train (adversarial reward)
    └── + Verification Framework   → RL train (verification accuracy)
```

### 3.1.1 Mock 解耦策略

每个 agent 独立训练时，其依赖的其他 agent 角色用商业模型（GPT-4o / Claude）mock：

```
训练 RCA Agent 时:
  FI Agent        = mock (商业模型生成注入方案，或直接用 RCABench 已有 case)
  World Model     = mock (商业模型预测传播，或跳过)
  Verification    = mock (商业模型做 judge，或用硬编码 F1)

训练 World Model 时:
  FI Agent        = mock (商业模型或 RCABench 提供注入场景)
  RCA Agent       = mock (不需要)
  Verification    = mock (商业模型对比预测 vs 实际)

训练 FI Agent 时:
  RCA Agent       = mock (商业模型做根因定位，作为对手)
  World Model     = mock (商业模型预测传播)
  Verification    = mock (商业模型验证)

训练 Verification Agent 时:
  FI Agent        = mock (RCABench 已有注入 ground truth)
  RCA Agent       = mock (商业模型做根因定位，提供待验证的输出)
  World Model     = mock (商业模型预测)
```

好处：
- 完全解耦——任何 agent 可以随时独立训练，不等其他 agent 先成熟
- 商业模型提供高质量 mock，比随机或规则 mock 更接近真实交互
- 训练好一个 agent 后，可以逐步替换对应位置的 mock

Mock → 真实模型的替换路径：
1. 所有角色都用商业模型 mock（baseline）
2. 某个 agent 训练完成后，替换掉对应 mock
3. 重新评估其他 agent 在新环境下的表现
4. 如果退化，用新环境重新训练

训练顺序建议（可并行，因为已解耦）：
1. **Phase 1**: RCA Agent（已有基础设施，mock = RCABench + 硬编码 F1）
2. **Phase 2**: 其他三个 agent 可按需并行启动，各自 mock 依赖
3. **Phase 3**: 逐步替换 mock 为训练好的模型，重新评估
4. **Phase 4**: 对抗训练循环（FI ↔ RCA，Verification 做裁判）

### 3.2 评估阶段（组合 pipeline）

```
Fault Injection Agent
  │ 选择故障方案
  ▼
World Model Agent
  │ 预测注入后的因果传播
  ▼
[执行故障注入到目标系统]
  │
  ├──▶ RCA Agent
  │      从可观测数据定位根因
  │               │
  │               ▼
  └──▶ Verification Agent（全链路裁判）
         │ 1. 注入有效吗？ (对比预期 vs 实际观测)
         │ 2. RCA 定位对吗？(对比注入 ground truth vs RCA 输出)
         │ 3. FI 找到盲区了吗？(RCA 失败 = FI 成功)
         ▼
    Learned reward signals:
      → FI Agent:  注入有效性 × RCA失败 × 影响面
      → RCA Agent: 定位准确性（比硬编码 F1 更细粒度）
      → World Model: 预测 vs Verification 观测的一致性
```

### 3.3 对抗训练循环

```
Round N:
  FI Agent (v_N) 生成故障方案
  → RCA Agent (v_N) 尝试定位
  → 收集 FI reward (RCA 失败 = FI 成功)
  → 收集 RCA reward (RCA 成功 = RCA 进步)
  → 分别更新 FI Agent → v_{N+1}, RCA Agent → v_{N+1}
  → Round N+1
```

---

## 4. 共享基础设施

### 4.1 需要扩展的 contracts

当前 `autorl.contracts` 需要支持：
- **因果图（CausalGraph）**：nodes, edges, root_causes 的标准格式（已有雏形）
- **故障注入方案（InjectionPlan）**：注入类型、目标服务、参数、预期影响
- **系统状态（SystemState）**：可观测数据的摘要表示
- **传播路径（PropagationPath）**：action → intermediate states → final state
- **验证结果（VerificationResult）**：有效性判定 + 差异分析

### 4.2 需要新增的 TaskAdapter

| Agent | TaskAdapter | 说明 |
|-------|------------|------|
| RCA | `RCATaskAdapter` | ✅ 已有 |
| World Model | `WorldModelTaskAdapter` | 需新建 |
| Fault Injection | `FaultInjectionTaskAdapter` | 需新建 |
| Verification | `VerificationTaskAdapter` | 需新建 |

### 4.3 需要新增的 RewardStrategy

| Agent | Reward | 设计要点 |
|-------|--------|---------|
| RCA | `RootCauseMatchRewardStrategy` | ✅ 已有，Phase 1 F1 硬匹配；Phase 2+ LLM-as-judge 语义匹配 |
| World Model | `SnapshotPredictionRewardStrategy` | node_f1 × w1 + metric_direction_accuracy × w2 + edge_f1 × w3（基于 snapshot 自动对比） |
| Fault Injection | `AdversarialInjectionRewardStrategy` | rca_defeat × injection_validity × blast_bonus（乘法结构，无效注入归零） |
| Verification | — | Phase 1-2 不训练，由商业模型驱动；Phase 3-4 的 reward 待定 |

### 4.4 数据需求

| 数据类型 | 来源 | 用途 |
|---------|------|------|
| RCABench cases | 已有 | RCA 训练、World Model 训练 |
| 故障注入记录 | 需生成 | FI Agent 训练 |
| 代码仓库 | 目标微服务 | World Model 代码阅读 |
| 因果传播标注 | 需标注或从 injection.json 推导 | World Model ground truth |
| RCA Agent 历史表现 | 训练过程产出 | FI Agent 输入 |

---

## 5. 可观测性与优化目标（逐 Agent 详细分析）

### 5.1 RCA Agent

**核心目标**：给定可观测数据，准确定位微服务故障的根因。

**Root cause 粒度**：根因粒度与故障注入粒度绑定——不同故障类型在不同注入点注入，ground truth 形如 `(service, fault_type, injection_point)` 的多级结构。Agent 输出自由文本，由 LLM-as-judge 做语义匹配（见下方 Reward 设计）。

**指标分类**：

指标分为两类——**Reward 指标**（用于 RL 训练的 reward function，要求可精确计算、确定性、抗 hack）和 **Eval 指标**（用于评估和消融分析，允许有主观成分或需要人工 review）。

#### Reward 指标

| 指标 | 定义 | 度量方式 | 目标 |
|------|------|---------|------|
| root_cause_f1 | 预测根因集合 vs 实际根因集合的 F1 | Phase 1: 硬匹配 `root_cause_f1_reward()`；Phase 2+: LLM-as-judge 语义匹配 | ↑ |
| root_cause_precision | 预测根因中正确的比例 | TP / (TP + FP) | ↑ |
| root_cause_recall | 实际根因中被找到的比例 | TP / (TP + FN) | ↑ |

#### Eval 指标

| 维度 | 指标 | 定义 | 度量方式 | 目标 |
|------|------|------|---------|------|
| **定位准确性** | exact_match_rate | 根因集合完全匹配的 case 比例 | bool(pred == ref) 的均值 | ↑ |
| **因果图质量** | graph_node_f1 | 预测因果图节点 vs 标注图的 F1 | 节点集合对比 | ↑ |
| | graph_edge_f1 | 预测因果图边 vs 标注图的 F1 | 边集合对比 | ↑ |
| | causal_direction_accuracy | 因果方向是否正确（A→B 而非 B→A） | 有向边匹配率 | ↑ |
| **调查效率** | avg_turns | 平均调查轮次 | 轨迹中 LLM 交互轮数 | ↓ |
| | avg_tool_calls | 平均工具调用次数 | 轨迹中 tool_call 事件数 | ↓（在准确性不降的前提下） |
| | tool_call_utility | 有效工具调用比例 | 启发式：调用结果被后续轮次引用 = 有用 | ↑ |
| | time_to_first_correct_cause | 首次命中正确根因的轮次 | 最终根因列表中首个正确项出现在哪一轮的输出 | ↓ |
| **鲁棒性** | performance_by_fault_type | 按故障类型分组的 F1 | 分组统计 | 各类型均 ↑ |
| | performance_on_multi_root_cause | 多根因 case 的 F1 | 仅多根因 case 统计 | ↑ |
| | false_positive_rate | 错误定位了不相关服务的比例 | FP / total predictions | ↓ |
| **泛化性** | unseen_fault_f1 | 训练集中未出现的故障类型上的 F1 | 按故障类型 hold-out 评估 | ↑ |
| | cross_topology_f1 | 不同服务拓扑上的 F1 | 按拓扑 hold-out 评估 | ↑ |

**因果图 ground truth 来源**：因果图标注由 Verification Agent 构建（见 §5.4）。早期由商业模型驱动，结合预置的 domain knowledge 规则（`fault_type → expected_propagation` 映射表）做 bootstrap。规则是辅助 Verification Agent 启动的手段，不直接作为 ground truth。因果图指标有标注噪声，因此只做 eval 不做 reward。

**Reward 设计（分阶段）**：

| 阶段 | Reward | 说明 |
|------|--------|------|
| Phase 1 | `root_cause_f1`（硬匹配） | injection.json 直接对比，零噪声，零成本 |
| Phase 2 | LLM-as-judge 评分 | 商业模型驱动的 Verification Agent 做语义匹配，容忍格式差异，支持更细粒度的 partial credit |
| Phase 3 | Learned judge reward | 训练好的 Verification Agent 提供 reward signal |

LLM-as-judge 实现要点：
- temperature=0 + 结构化 prompt（给定 ground truth 列表，逐条判定 agent 输出是否匹配，输出 JSON）保证确定性
- 定期抽样人工 review judge 判定，监控 judge 自身的准确性
- Phase 1 的硬匹配 F1 始终作为 anchor——任何阶段都可回退对比，检测 judge 漂移

**Reward hacking 防线**：

| 风险 | 描述 | 对策 |
|------|------|------|
| Brute-force 猜测 | 输出所有服务名作为根因，靠 recall 拉高 F1 | 监控 `len(pred) / len(ref)` 比值，>3x 时 flag；precision 设下限 |
| 效率惩罚导致放弃探索 | 如果加 turns penalty，agent 第 1 轮直接猜 | 不将 turns 放入 reward；仅作 eval 指标；如需惩罚，用阶梯式（前 N 轮免费） |
| Judge 偏好利用 | Agent 学会 judge 对特定表述方式更宽容 | Phase 1 硬匹配作为 anchor 定期对比；人工抽检 judge 判定 |
| 因果图标注噪声放大 | 有噪声的 ground truth 做 reward 会被 RL exploit | 因果图指标只做 eval，不做 reward |

**关键消融维度**：
- Reward 函数对比：F1 硬匹配 vs LLM-as-judge vs learned judge → 收敛速度和最终质量
- 工具集配置：完整工具 vs 受限工具 → 评估哪些工具对定位最关键
- 上下文长度：incident 描述详略程度对准确性的影响
- 模型规模：0.5B / 7B / 14B 在 F1 和效率上的 tradeoff

---

### 5.2 World Model Agent

**核心目标**：给定 State_t（snapshot: metrics, logs, traces）+ Action（故障注入动作），预测 State_{t+1} 的变化——哪些服务受影响、指标如何波动、日志/trace 变化。

**任务模型**：World Model 基于离线训练数据（历史注入 case 的前后 snapshot 对）进行训练。输入是注入前的 snapshot + 注入动作描述，输出是对注入后 snapshot 变化的预测。验证通过 Verification Agent 对比预测与实际 snapshot 完成。

**预测粒度**：预测在多个粒度上评估，reward 按粒度加权——粗粒度命中给基础分，细粒度命中给额外分：

| 粒度 | 预测内容 | 示例 | 验证方式 |
|------|---------|------|---------|
| L1: service 级 | 哪些服务受影响 | "payment-service 受影响" | 该服务任何 metric 异常即命中 |
| L2: 指标级 | 具体什么指标、什么方向 | "payment-service.latency_p99 上升" | 对应 metric 的异常方向匹配 |
| L3: 代码级 | 具体函数/span 级变化 | "processOrder 函数耗时增加" | trace 中 span 级数据对比 |

**传播关系（边）**：除了节点（什么受影响），还评估边（怎么传播的）——这是 World Model 的核心价值，否则退化为简单的影响面预测器。边的验证 = 两端节点都命中 + 时序一致（A 的异常先于 B）。

**指标分类**：

#### Reward 指标

| 指标 | 定义 | 度量方式 | 目标 |
|------|------|---------|------|
| node_f1 (L1) | service 级受影响节点的 F1 | 预测节点 vs 实际异常服务集合 | ↑ |
| metric_direction_accuracy (L2) | 指标变化方向预测准确率 | 预测方向 vs 实际方向（上升/下降/不变） | ↑ |
| edge_f1 | 传播边的 F1 | 两端节点命中 + 时序一致 = 边命中 | ↑ |

Reward 组合：`node_f1 × w1 + metric_direction_accuracy × w2 + edge_f1 × w3`，权重 w1 > w2 > w3（粗粒度权重高，保证基础信号稳定）。

这三个指标可完全自动化验证（不需要 LLM judge），只需对比 snapshot 的 metric 异常和时序：
- 节点命中：注入后该服务是否有 metric 异常（由预置规则代码判定）
- 方向匹配：异常方向是否一致（binary 判定）
- 边命中：两端节点 + 时序一致性

#### Eval 指标

| 维度 | 指标 | 定义 | 度量方式 | 目标 |
|------|------|------|---------|------|
| **细粒度预测** | metric_magnitude_error | 预测幅度 vs 实际幅度的误差 | 需定义容差范围 | ↓ |
| | code_level_hit_rate (L3) | 代码级预测命中率 | trace span 对比 | ↑ |
| | log_pattern_recall | 预测的日志模式 vs 实际出现的 | 日志模式匹配 | ↑ |
| | trace_anomaly_recall | 预测的 trace 变化 vs 实际 | trace 异常检测 | ↑ |
| **传播建模** | propagation_order_accuracy | 传播顺序是否正确 | rank correlation (Kendall's τ) on 异常时间戳 | ↑ |
| | intermediate_state_recall | 中间 snapshot 的变化预测匹配度 | 多步传播中每步 snapshot 的 node_f1 | ↑ |
| | confounder_identification_rate | 正确识别混杂变量的比例 | Verification Agent 判定（降级为纯 eval） | ↑ |
| **推理质量** | code_evidence_usage | 预测中引用了代码证据的比例 | 输出中包含代码引用的预测 / 总预测 | ↑ |
| | observability_evidence_usage | 预测中引用了可观测数据的比例 | 输出中包含 metric/trace 引用 / 总预测 | ↑ |
| | reasoning_chain_validity | 推理链逻辑一致性 | Verification Agent 判定 | ↑ |
| **校准度** | confidence_calibration | 置信度与准确率的相关性 | ECE (Expected Calibration Error) | ↓ |
| | overconfidence_rate | 高置信度但错误的比例 | P(wrong \| confidence > 0.8) | ↓ |
| **覆盖面** | fault_type_coverage | 能正确预测的故障类型数 | 按故障类型分组统计 | ↑ |
| | cross_service_accuracy | 跨服务传播预测准确率 | 仅涉及多服务的传播路径 | ↑ |

**Verification Agent 在 World Model 评估中的角色**：

Verification Agent 对 World Model 预测做"观测比对"——逐项检查预测与实际 snapshot 是否一致。采用**预置规则 + 自主判定**混合模式：

```
1. 预置规则代码（per fault_type）
   - 确定性执行，可复现，零噪声
   - 覆盖常见故障类型的已知传播模式
   - 示例：cpu_stress → check latency_p99 是否上升 > 10%

2. Agent 自主判定（规则未覆盖的长尾）
   - Verification Agent 读 snapshot 数据自主比对
   - 更灵活，但有噪声

3. 规则沉淀正循环
   - Agent 自主判定稳定后 → 人工审核 → 沉淀为新规则代码
   - 规则覆盖率随时间增长，Agent 只处理长尾
```

**Reward hacking 防线**：

| 风险 | 描述 | 对策 |
|------|------|------|
| 节点 brute-force | 预测所有服务都受影响，recall=1 | precision 约束——F1 自然平衡；监控 pred/ref 比值 |
| 编造传播链 | 预测长链 A→B→C→D，只有部分对 | edge precision 约束——F1 惩罚虚假边 |
| 证据编造 | 引用不存在的代码/metric 来刷 evidence bonus | evidence 指标只做 eval 不做 reward |
| 保守预测 | 只预测最明显的传播（如直接下游），回避不确定的多跳传播 | recall 指标确保覆盖面；消融实验对比传播深度 |

**关键消融维度**：
- 信息源对比：仅代码 vs 仅观测 vs 代码+观测 → 量化各信息源贡献
- 传播深度：直接影响 vs 多跳传播的预测准确率衰减曲线
- 预测粒度：L1 only vs L1+L2 vs L1+L2+L3 作为 reward 的对比
- 边 vs 无边：有/无传播关系（边）对 World Model 预测质量和下游 RCA 效果的影响

---

### 5.3 Fault Injection Agent

**核心目标**：生成高影响、RCA 难以定位的故障注入方案（红队）。

**Action space**：离散选择题。候选项组织为预制的树形结构（故障大类 → 目标服务 → 具体参数），FI Agent 在树中导航选择。选择后附带文本解释（reasoning），解释只做 eval 不进入 reward。

**RCA 对手**：训练初期使用商业模型 Agent 作为 RCA 对手，避免训练中的 RCA 太弱导致 reward signal 无区分度。商业模型 RCA 的水平稳定，提供有意义的对抗信号。后期逐步替换为训练好的 RCA 模型。

**指标分类**：

#### Reward 指标

| 指标 | 定义 | 度量方式 | 目标 |
|------|------|---------|------|
| rca_defeat | 商业模型 RCA 是否无法正确定位 | 1 - rca_f1 on this case | ↑ |
| injection_validity | Verification Agent 判定注入是否生效 | 有效=1, 无效=0 | ↑ |
| blast_radius | 注入后受影响的服务数量 | 注入后 snapshot 中异常服务数 | ↑ |

Reward 组合（乘法结构）：
```
reward = rca_defeat × injection_validity × (1 + 0.1 × (blast_radius - 1))
```

乘法保证：
- 无效注入 → reward = 0（不管 RCA 是否失败——防止无效注入刷分）
- RCA 成功定位 → reward = 0（不管影响面多大）
- 只有"有效注入 + RCA 失败"才有正 reward，blast_radius 做 bonus 调节

#### Eval 指标

| 维度 | 指标 | 定义 | 度量方式 | 目标 |
|------|------|------|---------|------|
| **对抗有效性** | rca_defeat_rate | 批次级：RCA 无法定位的注入比例 | 批次统计 | ↑（有上限，见均衡度） |
| | rca_f1_degradation | FI case 上的 RCA F1 相比基线的下降 | 商业模型 RCA 基线 F1 - FI case F1 | ↑ |
| **多样性** | fault_type_entropy | 选择的故障类型分布的熵 | Shannon entropy on 树的各层选择分布 | ↑ |
| | service_coverage | 覆盖的目标服务数 / 总服务数 | 去重计数 | ↑ |
| | tree_branch_coverage | 树形结构中被选择过的分支比例 | 已访问节点 / 总节点 | ↑ |
| | repeat_rate | 完全重复选择的比例 | 离散 action 直接判重 | ↓ |
| **策略质量** | reasoning_quality | 文本解释的质量 | 人工 review 抽样 | ↑ |
| | defeat_by_fault_type | 按故障类型分组的 defeat rate | 分组统计 | 分布均匀 |

**移除的指标及原因**：

| 原指标 | 移除原因 |
|--------|---------|
| economic_loss_estimate | 无业务影响模型，用 blast_radius 替代 |
| impact_score | 同上，blast_radius 作为影响面代理 |
| injection_distinguishability | 与 injection_validity 重叠——无效注入自然不可区分 |
| exploitation_vs_exploration | 定义模糊，用 fault_type_entropy + tree_branch_coverage 间接反映 |
| plan_specificity | 离散选择题下不适用——选择本身就是完整的 |
| novelty_rate | 离散 action space 下意义有限，用 repeat_rate 的反面替代 |

**Reward hacking 防线**：

| 风险 | 描述 | 对策 |
|------|------|------|
| 无效注入刷 defeat | 生成无意义注入，RCA 自然"定位不了" | 乘法结构：injection_validity = 0 → reward = 0 |
| 模式坍缩 | 只选某一类高 reward 的注入反复使用 | eval 阶段监控 fault_type_entropy 和 repeat_rate；如严重，考虑加多样性 bonus 到 reward |
| 商业模型 RCA 的盲区偏好 | FI 学会商业模型 RCA 的特定弱点而非通用弱点 | 替换对手后重新评估；消融实验对比不同对手 |
| blast_radius 极端化 | 选择影响面最大的注入（如网关故障），忽略精细故障 | blast_radius 系数小（0.1），主驱动力是 defeat × validity |

**关键消融维度**：
- Reward 组成对比：仅 defeat vs defeat × validity vs defeat × validity × blast
- 对手强度：商业模型 RCA vs 训练中的 RCA（不同版本）对 FI 策略的影响
- 信息量：给 FI Agent 看 RCA 历史表现 vs 不看 → 量化信息优势
- 树深度：浅层选择（故障大类）vs 深层选择（完整参数）的策略差异

---

### 5.4 Verification Agent

**核心目标**：作为 pipeline 的基础设施角色，为其他 Agent 提供可靠的评估和 reward signal。

**定位**：Verification Agent 在 Phase 1-2 **不训练**，完全由商业模型驱动。它不是一个需要早期训练的 Agent，而是一个**评估基础设施**——只有 Phase 3-4 一切成熟后才考虑训练自己的 Verification 模型。

**职责**（商业模型驱动期间）：

| 服务对象 | Verification 做什么 | 实现方式 |
|---------|-------------------|---------|
| RCA Agent | LLM-as-judge 语义匹配（root cause 判定） | 结构化 prompt, temperature=0 |
| World Model | Snapshot 观测比对（预测 vs 实际） | 预置规则代码 + 自主判定 |
| FI Agent | 注入有效性判定（injection_validity） | 预置规则代码 + 自主判定 |
| World Model | 因果图标注（传播路径构建） | 结合 domain knowledge 规则 bootstrap |

**Judge 质量监控指标**（当前阶段——确保商业模型 judge 的可靠性）：

| 指标 | 定义 | 度量方式 | 目标 | 自动化程度 |
|------|------|---------|------|-----------|
| injection_detection_accuracy | 注入是否生效的判定准确率 | Verification 判定 vs 预置规则代码自动判定（规则覆盖的 case 上交叉校验） | ↑ | 全自动 |
| rca_judgment_accuracy | RCA 判定 vs ground truth 一致性 | Verification 判定 vs injection.json | ↑ | 全自动 |
| f1_correlation | Verification 评分 vs 硬编码 F1 的相关性 | pearson_r(verif_score, f1_score) | > 0.8 | 全自动 |
| reward_signal_stability | 同一 case 多次评估的方差 | temperature=0 下应为 0 | → 0 | 全自动 |
| wm_judgment_accuracy | World Model 预测判定 vs snapshot 实际数据 | Verification 判定 vs 规则代码自动判定 | ↑ | 全自动 |

**监控机制**：
- 规则代码覆盖的 case 作为 Verification 的"考试题"——自动对比 Verification 判定与规则判定的一致性
- 不一致的 case 自动 flag，定期人工 review
- 如果一致性持续下降，说明商业模型 API 变化或 prompt 漂移，需要排查

**远期训练准备指标**（Phase 3-4 考虑训练 Verification 模型时才启用）：

| 指标 | 定义 | 何时需要 |
|------|------|---------|
| partial_credit_granularity | 区分"部分正确"的细粒度能力 | 训练 Verification 时——需要人工标注的 partial credit 案例 |
| improvement_over_f1 | 比硬编码 F1 更合理的 case 比例 | 评估是否值得从商业模型切换到训练模型 |
| reward_hacking_risk | RCA 利用 Verification 漏洞刷分 | 部署为 learned judge 后——监控 Verification reward ↑ 但人工评估 ↓ 的案例 |
| calibration | 置信度与准确率一致性 (ECE) | 训练 Verification 时 |
| reward_signal_informativeness | Verification reward vs F1 对 RCA 训练的效果对比 | 决定是否切换 reward source 时 |

**Reward 设计**：Phase 1-2 不训练，无 reward。Phase 3-4 的训练 reward 待定——依赖届时积累的数据和对 Verification 能力的需求分析。

**关键消融维度**（远期）：
- 商业模型对比：GPT-4o vs Claude 做 judge 的一致性和准确性差异
- 规则覆盖率影响：更多预置规则 vs 更多 Agent 自主判定对 judge 质量的影响
- Learned judge vs 商业模型 judge：训练好的 Verification 模型是否比商业模型更适合做 reward provider

---

### 5.5 Pipeline 组合指标

| 指标 | 定义 | 度量方式 | 目标 |
|------|------|---------|------|
| **E2E detection rate** | 从故障注入到根因定位全流程的成功率 | FI 生成 case → RCA 定位 → 判定正确的比例 | ↑ |
| **adversarial equilibrium** | FI vs RCA 的对抗均衡度 | FI 成功率（应趋近 50%——两者势均力敌） | → 50% |
| **world model calibration** | World Model 预测 vs 实际传播的一致性 | Verification 观测 vs World Model 预测的 edge F1 | ↑ |
| **injection effectiveness** | 有效注入占比 | Verification 判定有效 / 总注入 | ↑ |
| **blind spot convergence** | 对抗训练每轮后 RCA 新覆盖的故障模式数 | 按故障类型分组，统计新覆盖数 | 初期↑后期→0 |
| **mock→real degradation** | 替换 mock 后的性能退化幅度 | 替换前后各 agent 主指标的 delta | → 0 |
| **system resilience score** | 综合系统韧性评分 | E2E detection rate × (1 - avg impact of undetected faults) | ↑ |

### 5.6 North-star 目标（逐阶段）

**Phase 1 (RCA only)**:
1. `root_cause_f1` on RCABench eval > 0.6
2. `avg_turns` < 20
3. `false_positive_rate` < 0.15

**Phase 2 (all agents independent, mock dependencies)**:
4. `propagation_edge_f1` > 0.5 (World Model)
5. `injection_validity_rate` > 0.8 (FI Agent)
6. `rca_judgment_accuracy` > 0.8 (Verification 商业模型 judge 质量监控——不是训练目标，而是基础设施可靠性要求)

**Phase 3-4 (integrated pipeline)**:
7. `adversarial_equilibrium` ∈ [0.35, 0.65]
8. `e2e_detection_rate` > 0.7
9. `f1_correlation` > 0.85 (Verification as judge 的可信度)

---

## 6. 实施优先级

### Phase 1: RCA Agent 基线（当前阶段）
- 目标：跑通 RCA 训练 pipeline，建立 F1 基线
- 工作：完善 smoke test、修 lint、跑首次 eval
- Mock：FI = RCABench 已有 case，Verification = 硬编码 F1
- 依赖：agentm 子模块初始化、RCABench 数据准备

### Phase 2a: World Model Agent（可与 2b/2c 并行）
- 目标：实现 snapshot-based 因果传播预测训练
- 工作：WorldModelTaskAdapter、SnapshotPredictionRewardStrategy、snapshot 前后对比工具
- Reward：node_f1 + metric_direction_accuracy + edge_f1（全自动 snapshot 对比，无需 LLM judge）
- Verification：商业模型驱动，预置规则代码 + 自主判定混合模式
- 依赖：离线训练数据（历史注入 case 的前后 snapshot 对）、代码阅读工具集成

### Phase 2b: Verification Agent 基础设施（可与 2a/2c 并行）
- 目标：搭建商业模型驱动的 Verification 基础设施（**不训练模型**）
- 工作：预置规则代码库、LLM-as-judge prompt 设计、judge 质量监控 pipeline
- 产出：为 RCA/WM/FI 提供评估服务的基础设施
- 依赖：商业模型 API 接入、注入前后数据对比机制

### Phase 2c: Fault Injection Agent（可与 2a/2b 并行）
- 目标：实现对抗性故障注入训练
- 工作：FaultInjectionTaskAdapter、AdversarialInjectionRewardStrategy、树形 action space 构建
- Reward：rca_defeat × injection_validity × blast_bonus（乘法结构）
- RCA 对手：商业模型 Agent（水平稳定，提供有意义的对抗信号）
- Verification：商业模型驱动（Phase 2b 产出）
- 依赖：树形故障候选项制作、故障注入执行环境

### Phase 3: Mock 替换 + 重新评估
- 目标：逐步用训练好的模型替换各位置的 mock
- 工作：替换后重跑 eval，对比替换前后的指标变化
- 关键判断：替换后其他 agent 是否退化？退化则重新训练

### Phase 4: 对抗训练循环
- 目标：FI ↔ RCA 交替训练，Verification 做裁判
- 工作：对抗训练编排、均衡度监控、自动切换机制
- 依赖：Phase 3 中至少 RCA + FI + Verification 完成替换

---

## 7. 开放问题

### 已解决

1. ~~**World Model 的 ground truth 怎么获取？**~~ → 基于注入前后 snapshot 自动对比。Reward 用 node_f1 + direction_accuracy + edge_f1，全自动化，不需要因果图标注。详见 §5.2。
2. ~~**业务影响模型**~~ → 用 blast_radius（受影响服务数量）替代经济损失估算，数据直接从 snapshot 异常检测获得。详见 §5.3。
3. ~~**Verification 的 ground truth**~~ → 预置规则代码做确定性判定（规则覆盖的 case），Verification Agent 自主判定处理长尾。详见 §5.4。
4. ~~**Verification-as-judge 的 bootstrap 问题**~~ → Phase 1-2 完全由商业模型驱动，不训练 Verification 模型。Phase 1 用硬编码 F1，Phase 2 用商业模型 LLM-as-judge。训练 Verification 是 Phase 3-4 的远期目标。详见 §5.4。

### 仍然开放

5. **对抗训练的稳定性**：FI 和 RCA 交替训练时，如何防止一方过强导致另一方完全崩溃？可能需要 population-based training 或 ELO-style matchmaking。Phase 1-2 通过商业模型做对手缓解此问题，但 Phase 4 对抗训练循环仍需解决。
6. **代码阅读的范围和粒度**：World Model 需要读目标微服务的代码，每个 RCABench case 对应的代码仓库如何组织和提供？
7. **Verification 与硬编码 reward 的一致性校准**：Phase 2 引入 LLM-as-judge 后，需要监控 judge 评分与 F1 硬匹配的相关系数。偏差过大说明 judge 漂移或被 exploit。已设计监控机制（§5.4），但具体阈值和响应策略待定。
8. **树形 action space 的粒度和规模**：FI Agent 的故障候选项树的深度、广度、总节点数对训练效率和策略质量的影响待评估。
9. **商业模型 mock 的质量保障**：不同商业模型做 mock 的行为差异可能导致训练出的 agent 在替换 mock 时退化。需要评估 mock→real degradation 的幅度和可接受范围。

---

## 8. Mock 基础设施需求

每个 agent 训练时需要 mock 其他角色。scaffold 需要支持：

| 组件 | 说明 |
|------|------|
| `MockAgentRuntime` | 包装商业模型 API（OpenAI/Anthropic），实现 AgentRuntime 接口 |
| mock config 字段 | 在 YAML config 中指定哪些角色用 mock：`rca_mock: gpt-4o`、`verification_mock: claude-sonnet` |
| mock ↔ real 切换 | 同一个 config 可以通过改一个字段在 mock 和真实模型间切换 |
| mock 质量监控 | mock 输出的分布应接近真实模型，需要定期对比 |

实现路径：
1. 通用 `MockAgentRuntime`：接受 `model_name` + `system_prompt` + `api_key`，通过 OpenAI-compatible API 调用商业模型
2. 每个 agent 角色的 mock 有各自的 system prompt（定义角色行为）
3. Config 示例：
```yaml
# 训练 RCA Agent，FI 和 Verification 用 mock
task_adapter_path: autorl.tasks.rca.RCATaskAdapter
agent_runtime_path: autorl.runtime.agentm.AgentMRuntime  # 真实训练
mock_roles:
  verification:
    model: gpt-4o
    system_prompt: "You are a verification agent. Given the injection info and RCA output, judge if the RCA result is correct..."
  fault_injection: null  # 不需要 mock，直接用 RCABench case
```

---

## 9. 与现有代码的映射

| 需求 | 现有实现 | 需要做的 |
|------|---------|---------|
| Agent 框架接口 | `autorl.contracts` (TaskSample, AgentInput, Trajectory, TaskOutcome) | 扩展因果图、注入方案等 contract |
| RCA Agent | `tasks.rca` + `runtime.agentm` + `rewards.rca` | 完善 eval pipeline |
| World Model Agent | — | 新建 TaskAdapter + Runtime + Reward |
| Fault Injection Agent | — | 新建 TaskAdapter + Runtime + Reward |
| Verification Agent | — | 新建 TaskAdapter + Runtime + Reward |
| 训练入口 | `experiments/agent_workflow/train.py` | 无需修改（靠 config 切换 agent） |
| 数据 | `data/rcabench.py` | 扩展支持因果传播、注入方案格式 |
| 实验管理 | `experiments/` (刚设置) | 按 agent role 组织实验 |
