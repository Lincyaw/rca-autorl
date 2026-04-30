# Idea Index

本仓库的研究 idea 按"研究线 → 子 idea topic"分层组织。每个 topic 一个目录，目录内 `v0/v1/v2.md` 表示该 idea 的演化历史，prototype 代码或验证脚本可一并放在该目录下。

## 主研究线：Triadic Self-Play for RCA

围绕 self_play_evolves 三元角色（Proposer/Solver/Verifier）框架，针对 RCA / FI / Verifier / Controller 4 个角色组织训练，用 data-level 机制做 strong-to-weak 同步。

| Topic | 文件 | Status | 一句话 |
|---|---|---|---|
| Master | [`triadic-self-play/v0-overview.md`](triadic-self-play/v0-overview.md) | active | 项目母 idea：4 角色 + 信息不对称的整体架构 |
| Process verifier | [`triadic-self-play/process-verifier/v1.md`](triadic-self-play/process-verifier/v1.md) | active | 把 RCA 推理拆成"假设跳"，每跳由 Verifier 给三子分（warrant / plan / conclusion） |
| Goalpost curriculum | [`triadic-self-play/goalpost-curriculum/v1.md`](triadic-self-play/goalpost-curriculum/v1.md) | active | GASP 风格 lemma-lift 课程，用真实生产事故做 goalpost ground FI |
| Asymmetry ladder | [`triadic-self-play/asymmetry-ladder/v1.md`](triadic-self-play/asymmetry-ladder/v1.md) | active | 用 data-level 机制把论文里隐式的 strong-to-weak 同步显式化 |
| Info-gain reward | [`triadic-self-play/info-gain-reward/v1.md`](triadic-self-play/info-gain-reward/v1.md) | active | 把 RCA 的 abduction 循环建模为 info-seeking MDP，每步信息增益进 reward |

## 平行研究线

正交于主线三元组的独立研究方向，主线训练通后再启动。

| Topic | 文件 | Status | 一句话 |
|---|---|---|---|
| Controller harness | [`controller-harness/v0.md`](controller-harness/v0.md) | active (低优先级) | 给不可训练黑盒 LLM 训一个 wrapper，在 inference time 规范行为并提升性能 |
| World Model | [`world-model/v0.md`](world-model/v0.md) | active (低优先级) | 三元组任务隐含的因果传播能力，研究是否要显式训一个 WM；先 probe 后 standalone |

## 约定

- **添加新 idea**：在合适的研究线下新建 topic 目录，从 `v0.md` 开始；如果是某个 v1 的子分支，可直接 `cd <parent-topic>/` 加 `v2.md` 并设 frontmatter `parent: <parent id>`
- **idea 演化**：不要原地编辑 v1，新建 v2 并在 frontmatter 标 `parent: <v1 id>`，让 git 历史保留思路演变
- **frontmatter 字段稳定**：`id` 是 idea 的稳定标识，可以独立于文件路径，因此跨 idea 引用建议用 ID（不脆于路径）
