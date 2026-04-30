# LLM-as-Harness 设计草案

> 用一个 8B 级小模型作为 Claude Code hook，在主 agent 推理过程中做异步、非阻塞的语义级行为监督。大多数时候保持沉默；检测到任务漂移类 anti-pattern 才注入一条提醒。

---

## 0. 定位

- **不是新的 agent**，是 agent 之外的 monitor / harness。
- **inference-time guardrail**：直接挂 Claude Code hook 监督主 agent 行为。
- **train-time PRM 复用**：产出的 `event sequence + drift label` 同时是 RCA agent RL 训练的 process-level reward 数据源。同一份资产两用。
- 运行时位置：`src/autorl/observability/harness/`（observability 已存在，harness 是 inference-time 子模块）。

---

## 1. 核心思路：对话 → event 序列

主 agent 的 transcript 在历史前缀上不可变。把 transcript 折叠成稳定的 event 序列，所有判断在 event 序列上进行：

```
turn 0:  user task
turn 1:  assistant message + tool calls
turn 2:  tool results
...
              ↓ summarize (incremental, cacheable)
event 0: {kind: task,        summary: "...", refs: []}
event 1: {kind: hypothesis,  summary: "...", refs: [0]}
event 2: {kind: evidence,    summary: "...", refs: [1]}
...
              ↓ reason over full event sequence (prefix-cacheable)
verdict: {drift: false}  |  {drift: true, type: ..., reminder: "..."}
```

每次 hook 触发只对**新增的 turn**做 summarize；reasoner 看完整 event 序列，但前缀几乎全部命中 prefix cache。

---

## 2. Event schema

```json
{
  "turn": 7,
  "kind": "task | hypothesis | evidence | decision | action | reflection | conclusion",
  "summary": "在没有 metric 异常证据下仍坚持 service-A 假设",
  "refs": [3, 5],
  "source_turns": [12, 13]
}
```

约束：
- `kind` 是闭集枚举，强制 8B 只做分类不做创作。
- `summary` ≤ 30 字，自然语言。
- `refs` 指向相关的更早 event id，用于漂移检测时构建简单的推理图。
- `source_turns` 记录该 event 对应的原始 transcript turn 范围，便于回溯。
- 一次 hook 触发可能产出 0~N 个 event（一个 assistant message 内可能同时包含 hypothesis + action）。

---

## 3. 两次 8B 调用

**Call A — Event Summarizer（增量、便宜）**
- 输入：`history_events` 的最后 5~10 条 + `new_turns`（自上次 hook 以来的对话片段）
- 输出：`new_events: List[Event]`
- 特点：上下文小、稳定，可批量；模型只做"读 turn → 输出 event 列表"。

**Call B — Drift Reasoner（前缀稳定、cache 命中率高）**
- 输入：`full_event_sequence`（含刚生成的 new_events）
- 输出：`{drift: bool, type?: str, confidence: float, reminder?: str}`
- 特点：前缀只增不改，prefix cache 命中近 100%；只读最后几条 event 决定是否漂移。
- 沉默优先：默认 `drift=false`，仅当 `confidence ≥ τ` 时才输出 reminder。τ 起步设 0.7。

两次调用看似贵，实际 input cost 比单次"summarize+reason 合一"还低，且两步可独立换模型/独立迭代。

---

## 4. 漂移类型最小集合（先做 4 类）

| 类型 | 触发信号 |
|---|---|
| `task_drift` | 当前 action / hypothesis 的语义距离 turn 0 任务过远 |
| `evidence_ignored` | decision 之后出现矛盾的 evidence event，但没有对应 reflection |
| `premature_conclusion` | conclusion 类 event 出现时，支撑它的 evidence event 数量 < 阈值 |
| `stuck_loop` | 连续 N 个同类 action，期间无 evidence/reflection 间隔 |

这 4 类对 8B 友好，prompt 中给 few-shot 即可，不需要训练。后续可加入更细的 RCA 领域专属漂移（e.g. "未验证假设就修改代码"）。

---

## 5. Hook 接线（Claude Code）

```
PostToolUse / Stop hook  →  inbox/<session>.jsonl   （写 transcript delta，立即返回）
                                       ↓
                               background worker
                                       ↓ (Call A → Call B)
                            events/<session>.jsonl
                            pending_reminders/<session>.json
                                       ↓
UserPromptSubmit hook    ←  pending_reminders/<session>.json  →  inject 一条 system reminder，清空
```

要点：
- PostToolUse hook **零阻塞**：只 append 到 inbox，不等 worker。
- worker 是独立进程（cron 或长驻 daemon），消费 inbox → 产出 event 与可选 reminder。
- 提醒最多滞后 1 turn 才被主 agent 看到，对"防止持续偏移"足够。
- UserPromptSubmit hook 投递后立即清空 pending，避免重复打扰。
- 频率限制：同一 session 内 ≤ 1 reminder / N turns（N 起步设 5）。

---

## 6. 文件协议

```
.harness/
  inbox/<session_id>.jsonl           # transcript delta，append-only
  events/<session_id>.jsonl          # 已生成的 event 序列，append-only
  pending_reminders/<session_id>.json # 待投递提醒，单条；投递后清空
  cursor/<session_id>.json            # 上次处理到的 turn index
```

所有文件 JSONL/JSON，便于离线消费（训练数据导出、漂移分析、case study）。

---

## 7. Reminder 注入格式

注入到主 agent 的 system reminder 内容：

```
[harness] possible drift detected — type: task_drift
context: 最近 3 个 action 都在调试 service-A，但任务原始目标是定位 service-B 的 latency 异常
suggestion: 回看 turn 0 的任务陈述，确认当前路径是否仍服务于该目标
```

约束：≤ 80 字，**只描述观察 + 提出反思**，不替主 agent 决策。

---

## 8. 实现路径

| 阶段 | 内容 | 验收 |
|---|---|---|
| P0 | Hook + 文件协议 + 规则版 drift detector（无 8B） | 跑通注入回环；在 RCA agent 已有 trace 上人工 review |
| P1 | 8B event summarizer（Call A） | event 序列质量人工 review；缓存命中率验证 |
| P2 | 8B drift reasoner（Call B），4 类漂移 + few-shot | 在标注好的 RCA trace 上 precision/recall ≥ 0.6 |
| P3 | RL 训练复用：event 序列作为 PRM 训练数据源 | 端到端 RCA agent 用 PRM reward 训练跑通 |

**P0 已完成**（不依赖 LLM）：

- 文件协议 / store / 规则 summarizer / 规则 detector — `src/autorl/observability/harness/`
- Claude Code hook 适配器 — `claude_code.py`：`parse_hook_payload` + `read_transcript_turns` 从 `transcript_path` 派生稳定 index 的 `Turn`，幂等
- CLI `ingest --from-hook` / `inject --from-hook` — stdin 读 hook payload，自动取 `session_id`
- post-delivery 速率限制 — cursor 增加 `last_reminder_at_index`，N=5 turn 内不再发
- 跨进程并发安全 — `HarnessStore.session_lock(sid)` + `fcntl.flock`，cron worker 与长驻 daemon 可共存
- 一键安装 — `scripts/harness/install.sh` 写 `.claude/settings.local.json` + 起后台 worker；`settings.example.json` 为静态模板

---

## 9. 与项目复用

- inference-time guardrail：本仓库 RCA agent 推理时直接受益。
- train-time PRM：event 序列 + 漂移标注 → 训练 process reward model → 喂给 AReaL 做 dense reward。
- 跨 agent 复用：本仓库其他 agent（FI / Verifier，以及正交研究线的 World Model）可共享同一套 harness 与 event schema。

---

## 10. 待决问题

- ~~session_id 如何稳定标识？~~ **已解决**：Claude Code hook payload stdin 直接含 `session_id` 与 `transcript_path`，不依赖 env var。
- inbox → worker 的触发：当前选 cron 风格的轮询（`install.sh` 起 `while sleep 5` daemon），inotify/hook 同步起 worker 推迟到性能成为瓶颈再切。
- 8B 选型：候选 Qwen2.5-7B-Instruct / Llama-3.1-8B-Instruct / 自蒸馏。先用 Qwen 起步。
- 漂移阈值 τ 与 N（频率限制）需要在真实 trace 上调一次。N 当前默认 5。
- 对 PRM 训练，event 是否需要附加"该 event 是否对最终 outcome 有帮助"的回溯标签？这是 P3 的问题，P0/P1/P2 可以推迟。
