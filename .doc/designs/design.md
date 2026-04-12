# AReaL-first 运行接口设计（当前仓库版）

> 目标：先定义统一格式与接口，再承载真实数据、多任务、多 agent 框架与后续 scale-out。

---

## 0. 当前定位（2026-04）

- 本仓库是 **AReaL-first 的 agent 训练脚手架**，不是特定 search 逻辑的专用仓库。
- 训练主线采用 AReaL 原生 agent workflow 语义：`async run(data, **extra_kwargs)`。
- `search` 只是一个 reference task，不参与定义底层抽象。
- `agent_sft/train.py` 仍是占位实现。

---

## 1. 核心设计原则

1. **框架无关**：agent 框架可替换，但外部 contract 稳定。
2. **训练/推理一致**：train/eval/infer 复用同一 runtime + task adapter 语义。
3. **AReaL 原生优先**：token-level 追踪和 proxy 由 AReaL 负责，仓库只做薄适配。
4. **格式先行**：先统一 `TaskSample` / `AgentInput` / `Trajectory` / `TaskOutcome`。
5. **混合边界**：tool/env 通过统一 gateway 接口接入，本地与远端返回同构。

---

## 2. 模块分层

| 层 | 目录 | 职责 |
|---|---|---|
| Contracts | `src/autorl/contracts/` | 统一格式：`TaskSample` / `AgentInput` / `Trajectory` / `TaskOutcome` |
| Runtime | `src/autorl/runtime/` | `AgentRuntime` 接口、canonical workflow、trace sink |
| Tasks | `src/autorl/tasks/` | `TaskAdapter`，负责样本 -> 输入 -> outcome 映射 |
| Gateways | `src/autorl/gateways/` | `ToolGateway` / `EnvGateway` 边界与本地实现 |
| Workflow Adapter | `src/autorl/workflows/` | 对 runtime 的薄 re-export |
| Experiments | `src/autorl/experiments/agent_workflow/` | train/eval/infer 启动编排 |
| Data | `src/autorl/data/` | 数据加载；task adapter 负责 schema 校验 |

---

## 3. Canonical 入口

- Workflow：`autorl.runtime.agent_workflow.UnifiedAgentWorkflow`
- Runtime 接口：`autorl.runtime.AgentRuntime`
- Task 接口：`autorl.tasks.TaskAdapter`
- Reward 接口：`autorl.rewards.base.RewardStrategy`
- Gateway 接口：`autorl.gateways.ToolGateway` / `autorl.gateways.EnvGateway`
- 示例任务：
  - `autorl.tasks.search.SearchTaskAdapter`
  - `autorl.tasks.search_runtime.SearchAgentRuntime`
  - `autorl.tasks.search_reward.SearchRewardStrategy`

---

## 4. 统一格式（最小集合）

- `TaskSample`：任务样本标准封装。
- `AgentInput`：runtime 执行所需标准输入。
- `RuntimeContext`：AReaL 注入的模型端点、HTTP client、gateway 与限制。
- `Trajectory` + `TrajectoryStep`：统一轨迹事件格式。
- `TaskOutcome`：评估与 reward 消费标准输出。
- `TrainingEpisodeView`：训练期映射视图（当前预留）。

> 原则：任务字段可以不同，但外层结构必须一致。

---

## 5. 运行链路

1. data 层加载样本，不做任务特定字段判断。
2. workflow 在 `run(data, **extra_kwargs)` 中完成：
   - record -> `TaskSample` -> `AgentInput`
   - 构建 `RuntimeContext`
   - 调用 `AgentRuntime.run(...)`
   - 从 `Trajectory` 提取 `TaskOutcome`
   - 通过 `RewardStrategy` 计算 reward
3. AReaL 负责 proxy、interaction capture 与训练导出。
4. repo 可选把 canonical trajectory/outcome/reward 写入 JSONL。

---

## 6. 配置要点

`configs/train/*.yaml` 的关键字段：

- `workflow`
- `task_adapter_path`
- `agent_runtime_path`
- `reward_strategy_path`
- `tool_gateway_mode` / `tool_gateway_base_url`
- `env_gateway_mode` / `env_gateway_base_url`
- `trace_dir`

AReaL backend / scheduler / actor / rollout 配置继续沿用原生字段。

---

## 7. 当前限制

- 默认示例仍是 search task；其他任务需新增自己的 adapter/runtime/reward。
- SFT 主线未实现。
- 当前未新增 repo-level 测试，只做编译与 CLI/导入级验证。
- AReaL proxy-backed agent workflow 仍以 `local` / `slurm` scheduler 为主；`ray` 不作为主支持路径。

---

## 8. 维护约束

- 新任务优先新增 `TaskAdapter` 与 `AgentRuntime`，不要改共享 workflow 逻辑。
- 新工具或环境优先通过 gateway 扩展，不直接耦合进 runtime。
- train/eval/infer 三入口必须持续共享同一 workflow contract。
