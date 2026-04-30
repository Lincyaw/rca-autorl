# Training Pipeline: 4-Stage Cold-Start to Self-Play

> 4 个 agent 从 base checkpoint 到 self-play 闭环的**训练过程轴**视图。
>
> 与本文档正交的是**研究里程碑轴**（RCA F1 > 0.6 / agents 独立 / 集成），见 `CLAUDE.md` 的 North-star Targets。两套划分独立命名以避免混淆：本文档用 **Stage**，CLAUDE.md 用 **Phase**。

## 大图

4 个 agent (FI / RCA / Verifier / Controller) 任务高度耦合。直接进 self-play 会卡在所有人都太弱、彼此互相 produce 噪声数据这种死循环。所以走 **API 当替身 → 解耦冷启 → 各自 RL → 合流** 四步：

```
Stage 0          Stage 1                Stage 2              Stage 3
─────────        ─────────              ─────────            ─────────
API-as-all       4 路 SFT 并行          各 agent AReaL RL    asymmetry
in roles         (从同一 base)          (其他角色继续 mock)   ladder
on RCABench                                                  ↓
                                                             4 角色合流
```

Controller 不在这 4 个 stage 节奏里，自有路径，见 §Controller。

---

## Stage 0 — API-as-all-roles 数据采集

**目标**：用商业 API 模型分别扮演 FI / RCA / Verifier 在 RCABench 上跑出 SFT 数据。

**做的事**：
- 4 套 system prompt 分别让 API 扮演 FI / RCA / Verifier (Controller 暂缓)
- 在 RCABench 上批量跑：API-FI 注入 → API-RCA 推理 → API-Verifier 打分
- 收集对应的 (input, output) 轨迹作各 agent 的 SFT 数据

**质量门** —— **不看 F1 数字，看 trajectory 结构是否可用作 SFT 起点**。人工抽查 ~30 条样本看 tool 调用格式、推理跳的边界、Verifier 打分 schema 一致性。结构烂就改 prompt 重跑，不能直接进 Stage 1。

**Stage 0 内部硬序**：API-RCA 跑完 → API-Verifier 跑（Verifier 标的对象就是 RCA trajectory，必须先稳）。

---

## Stage 1 — 4 路并行 SFT

**目标**：从合适的 base checkpoint 出发，4 个 agent 各自 SFT distill 到 cold-started 小模型。base 选型 (e.g., Qwen2.5-7B / Llama-3.1-8B) 以及"是否 4 个 agent 共享同一个 base" 不在本文档预设，留给具体 stage kickoff 时根据实验决定。

**Agent 间依赖**：
- **FI / RCA 并行启动** —— 数据彼此独立
- **Verifier 必须等 RCA SFT 收敛** —— Verifier 学的是"给 RCA-style trajectory 打分"，RCA 还在改 trajectory 格式时 Verifier SFT 是污染的

---

## Stage 2 — 各自 RL（仍解耦）

**目标**：每个 agent 单独跑 AReaL RL，其他 3 个继续用 SFT 版本（或 API）作 mock environment。

**做的事**（每个 agent 一个 RL 训练 job，互不干扰）：
- **RCA**：outcome reward (RCABench ground truth) + λ × process reward (Verifier-SFT 打分)
- **FI**：reward 用 GASP lemma-lift 公式 + RCA-SFT 的 pass rate 当难度信号
- **Verifier**：reward 用 anchor case 上的判定一致性（见 `paper/ideas/triadic-self-play/asymmetry-ladder/v1.md` 机制 2）

**互不直接相连** —— Stage 2 的 4 路 RL **不读对方训练中权重**，只读固定的 Stage 1 SFT checkpoint 当 environment。

---

## Stage 3 — 合流 + asymmetry ladder

**目标**：进入 self-play 闭环。strong-to-weak 同步通过 `paper/ideas/triadic-self-play/asymmetry-ladder/v1.md` 的 3 个 data-level 机制实现。

**渐进合流**（不一上来 4 个一起转）：
1. RCA + Verifier 配对 RL（process reward 实时回传到 RCA 训练）
2. FI + Verifier 配对 RL（注入 validity guard）
3. 4 角色一起进 ladder

**健康监控**：`asymmetry_gap ∈ [0.2, 0.7]`。出区间触发 goalpost 漂移 / Verifier anchor 校准 / replay buffer 重打分（机制 1/2/3）。

---

## Controller 路径（不同源）

Controller 的 cold-start 数据是 **(state, intervention) pairs**，不是 (input, output) 复刻 —— 它学的是"在什么 transcript / system state 下做什么 wrapper 动作"，API 自己不是 wrapper，没法直接 distill。

实施路径见 `.doc/designs/llm-harness.md`：
- **P0** rule-based detector（已实现）
- **P1** 8B event summarizer
- **P2** drift reasoner（4 类漂移 + few-shot）
- **P3** RL training，event 序列当 PRM 数据源

启动时机：主线 Stage 2 末期跑通后再启 Controller P3。

---

## Stage 间的硬序约束

| 约束 | 原因 |
|---|---|
| Stage 0 内：API-RCA 跑完 → API-Verifier 才跑 | Verifier 标的对象是 RCA trajectory |
| Stage 1：RCA SFT 收敛 → Verifier SFT 启动 | trajectory 格式先稳 |
| Stage 1 全部完成 → 任一 Stage 2 RL 启动 | RL 需要其他角色的 SFT 版作 mock |
| Stage 2：RCA + Verifier 配对通过 → FI 进入合流 | 给 FI 一个稳定 difficulty signal |
| 主线 Stage 2 通过 → Controller P3 启动 | Controller 需要 RCA 真实轨迹做 environment |

---

## 不在本文档的事

- 各 stage 的具体数据量 / case 数：写到对应 stage 启动时的 experiment plan
- API 选型 / 数据采集运行细节：操作层，stage kickoff 时再定
- 评估指标 / north-star 阈值：见 `CLAUDE.md` 和 `.doc/designs/agent-roles-spec.md` §5
