# rca-autorl

Thin AReaL integration for training the DeepSeek Harness RCA agent.

The repository intentionally follows AReaL's SWE example: DeepSeek Harness (`dsh`) owns
the agent loop, tools, and trajectory persistence; this repository only connects that
agent to AReaL's rollout proxy and trainer.

```text
AReaL PPO/GRPO trainer
        │ OpenAI-compatible rollout proxy
        ▼
DshWorkflow ──▶ DeepSeekHarness SDK ──▶ dsh runtime subprocess
        ▲                                   │  sdk-minimal + agent/rca-harness
        │                                   │  tools: sql (DuckDB over the snapshot)
        │                                   ▼         submit_result
        └── placeholder 0.0 reward ◀── fault propagation graph
```

## What lives here

- `src/autorl/agent.py`: direct AReaL workflow around the public `DeepSeekHarness` SDK.
- `src/autorl/harness.py`: composes a launch — bundle install, scenario layer,
  model route, and the one `dsh` runtime construction both paths use.
- `src/autorl/fpg.py`: binds the fault propagation graph schema to this
  testbed's vocabulary, and projects it into the harness bundle.
- `configs/fpg/microservices.toml`: that vocabulary — the entity kinds, failure
  modes, and propagation mechanisms an answer and its ground truth may use.
- `src/autorl/dataset.py`: prepares `datapacks/ops-lite` — the incident manifest
  and the ground truth the verifier will read.
- `agent/`: the RCA harness — the `dsh` bundle that replaces the shell with a
  bounded DuckDB `sql` tool over the snapshot, an investigation notebook, and
  the terminal `submit_result`, plus the RCA scenario patch and the gateway
  model route. Its
  [README](agent/README.md) is the design.
- `src/autorl/algorithm.py`: validates the AReaL v2 RLOO configuration.
- `src/autorl/train.py`: JSONL loading plus `PPOTrainer` launch.
- `src/autorl/train_sft.py`: `SFTTrainer` launch for distilled RCA trajectories.
- `src/autorl/data/sft.py`: chat-template rendering and assistant-only loss masks.
- `src/autorl/data/export.py`: turns `dsh` session logs into SFT rows.
- `src/autorl/data/collect.py`: runs cases against a teacher endpoint to fill a
  Harness home with the sessions SFT trains on.
- `configs/train/dsh_rca_smoke.yaml`: one-GPU RCA RL smoke config.
- `configs/sft/`: SFT configs for distilled trajectories.

Generic task/runtime/gateway contracts are deliberately absent. If another agent is
trained, add another direct workflow like AReaL does for SWE instead of introducing a
framework inside this repository.

## The answer and the corpus

What an episode submits and what it is scored against are the same schema. That
schema is `fpg` ([fpg-convention](https://github.com/Lincyaw/fpg-convention)), a
pinned dependency owning the fault propagation graph's structure — nodes as
time-anchored `(entity, failure mode, window, evidence)` statements, edges as
existence and direction, root causes stated explicitly. `ModelRCAOutput` is what
`submit_result` accepts; `Scenario` is what the annotation writes.

`configs/fpg/microservices.toml` owns the half that is testbed-specific: the
entity kinds a subject may name, the failure modes a predicate may assert, the
channels an edge may travel. It is adopted, not authored — `datapacks/ops-lite`
is already annotated in exactly these terms, and a vocabulary that existed on
only one side would be a vocabulary nothing could be scored on.

`datapacks/ops-lite` is our copy of a 500-case corpus over three testbeds
(train-ticket 310, DeathStarBench hotel reservation 152, OpenTelemetry Demo 38).
Each case carries the abnormal and normal telemetry as Parquet plus
`causal_graph_verified.json`, an `fpg.Scenario` ground truth. Two things it does
not ship are derived by `python -m autorl.dataset`: the incident prompt, from
the SLO violations in `conclusion.parquet`, and a `vocab_version` restamped from
the corpus's `microservices-0.4.0` to the 0.5.0 this repository maintains — an
additive migration, since 0.5.0 only adds three predicates, one mechanism, and
the `link` entity type. It writes `data.jsonl` (433 trainable cases) and
`data.excluded.jsonl`, which records why the other 67 are out: 51 show no SLO
violation at all (the injected fault never reached an endpoint — 32 of the 38
otel-demo cases), 15 carry the annotation pipeline's `.invalid` marker, and 3
have an empty verified graph.

`datapacks/` is not versioned — 7.4G of telemetry does not belong in a git
history. It is reproduced by copying the release into `datapacks/ops-lite` and
running the preparation once, which is idempotent, so a second run on an
already-prepared corpus changes nothing.

```bash
python -m autorl.dataset datapacks/ops-lite        # idempotent; --check to dry-run
```

Both halves of the contract are enforced at the moment the model answers.
`src/autorl/fpg.py` binds them into pydantic models and generates
`agent/rca-harness/src/vocabulary.js` so the harness tool rejects an
out-of-contract submission before it becomes a training example, and
`tests/test_submit_result_contract.py` holds the two enforcements to the same
verdicts.

## Setup

The checkout uses AReaL as a submodule and DeepSeek Harness as a pinned SDK package:

```bash
git submodule update --init --recursive
UV_HTTP_TIMEOUT=300 uv sync --python 3.12
```

`deepseek-harness-sdk` pulls the matching `deepseek-harness-runtime-bin` wheel, so the
`dsh` runtime ships with the environment and needs no system Node.js. The workflow boots
the shipped `sdk-minimal` profile — the only one that accepts an arbitrary model id,
which is what the rollout proxy serves — with the `agent/rca-harness` bundle installed
into it and `agent/profiles/<scenario>.patch.yml` layered on top. The bundle disables
`sdk-minimal`'s shell and editor rows, so an episode's whole action space is `sql`,
`take_note`, and `submit_result`.

Install the bundle into the profile once, before any rollout starts:

```bash
python -m autorl.harness .runs/dsh-rca-smoke/dsh-home     # --reinstall after editing it
```

`dsh plugin` shells out to `pnpm`. A machine with Node but no global pnpm needs no
setup: the install writes a corepack shim for the subprocess and leaves a real
`pnpm` alone. `scripts/run_smoke.sh` runs this step for you, and a rollout whose
profile is missing the bundle fails immediately with the command to run.

Each episode reuses that one `$DSH_HOME` (`econfig.dsh_home`, default `.runs/dsh-home`)
with a fresh session id.

## RL

The input is the processed RCA JSONL. Each row needs an incident (`question` or
`incident`) and either an absolute `data_dir` or a `datapack_name` resolvable below the
dataset root.

```bash
export RCA_DATASET_ROOT=$PWD/datapacks/ops-lite
./scripts/run_smoke.sh train_dataset.path=$RCA_DATASET_ROOT/data.jsonl \
  valid_dataset.path=$RCA_DATASET_ROOT/data.jsonl
```

For every rollout, AReaL passes a proxy URL and session API key to `DshWorkflow`.
The workflow declares that endpoint as a model route, launches one `dsh` runtime for
the incident in the case's snapshot directory, and returns one scalar reward in the
format expected by AReaL v2. The proxy sees OpenAI-compatible
`POST {base_url}/chat/completions` requests, so it records the whole trajectory.

The route matters. `sdk-minimal` registers only the shipped `deepseek-official`
adapter, which carries a streamed tool call's id and name across deltas with an
`!== undefined` guard. sglang writes both as an explicit `null` on every argument
delta (`serving_chat.py`: "Subsequent chunks: null ID and name for argument deltas",
serialized without `exclude_none`) and AReaL's gateway forwards the stream verbatim,
so the guard passes and the id and name the first delta established are erased. The
call then fails as `Error: unknown tool ""` with its arguments intact — and the model
retries, forever: against a replay of that stream the shipped adapter produced 4530
calls, every name empty, not one submission. `autorl.harness.model_route` declares
the endpoint as an `llm-pi-ai` route instead, whose `openai-completions` protocol is
the OpenAI dialect proper; the same replay then yields `sql, sql, submit_result`. SFT
collection composes through the same function, so the two paths cannot drift.

Reward is a redistribution, and AReaL's group normalization reads it on two axes.

The outcome is `fpg.compare_model_to_ground_truth`: the submitted graph against
the case's `causal_graph_verified.json`. The process term exists because an
episode runs 50-200 tool calls and submits once, so one terminal number credits
every turn by where it sat. The `take_note` policy already cuts the trajectory
into blocks — a run of queries and the finding the model commits to — and a
block whose queries interrogate entities on the true propagation path is a block
that moved; that per-block hit rate correlates 0.35 with the final score across
the fifty collected episodes, against 0.06 for hops-to-root and -0.06 for how
early a root is first queried.

A turn's value is `outcome + shaping * (its block's rate - the per-turn mean
rate)`. The second term is centred, so a trajectory's mean turn value is exactly
its outcome and no amount of querying can raise it. That is not fastidiousness:
the model cannot see the true entity set, so the only way to raise a hit rate it
does not understand is to name more services per filter, and a bonus would pay
for `WHERE service_name IN ('a', ..., 'z')`.

Both axes then fall out of the normalization AReaL already applies.
`GroupedRolloutWorkflow` merges the `n_samples` samples of a prompt into one
trajectory, so `concat_batch` reports that prompt's whole group as one entry and
`reward_norm(mean_level="group", mean_leave1out=true)` centres each turn against
every turn of every sibling sample. Since the shaping is zero-mean inside each
episode, that baseline is the cross-sample mean outcome, and a turn's advantage
comes out as `outcome - mean sibling outcome + its block's deviation`. Nothing
custom is needed, which is why nothing custom is here.

Addressing a turn needs the id AReaL keys its cache by, and the session log does
not carry it. The `rca-completions` row reads it off `finish`'s `replayState`
— documented as "response-level adapter-private metadata (ids, native stop
reason)", where `llm-pi-ai` puts the provider's `responseId` — and writes one
line per request to `$DSH_HOME/rca-completions/<session>.jsonl`. `purpose`
separates the agent's turns from the compaction summarizer. `DshWorkflow.run`
returns `dict[completion_id, reward]`, and since AReaL accumulates backward,
what it returns is the difference between neighbouring turn values, not the
values. If the sidecar's agent-request count does not match the log's, the
mapping is refused whole and the outcome falls back to the last turn: crediting
the wrong turn is worse than crediting none.

Generation limits stay AReaL's: `rollout.model` is the served name the route declares,
`gconfig.max_new_tokens` becomes the request's `max_tokens`, and `sglang.context_length`
becomes the window compaction triggers below. `train.py` passes all three into the
workflow, so none is declared twice.

Reward has two terms and lands per turn, not per episode.

The outcome is `fpg.compare_model_to_ground_truth`: the submitted graph against
the case's `causal_graph_verified.json`, scored on root subjects, all subjects,
and contracted edges. The process term is what makes that attributable. An
episode runs 50-200 tool calls and submits once, so a single terminal number
credits every turn by where it sat. The `take_note` policy already cuts the
trajectory into blocks — a run of queries and the finding the model commits to —
and a block whose queries interrogate entities on the true propagation path is a
block that moved. Across the first fifty episodes that per-block hit rate
correlates 0.35 with the final score, against 0.06 for hops-to-root and -0.06
for how early a root is first queried; it is weighted at 0.2 (`econfig.shaping`)
because it tracks being right without defining it.

Placement does the rest. AReaL accumulates rewards backward
(`reward[i] += reward[i+1] * turn_discount`), so a block's reward left on the
turn that closed it reaches every turn inside and before, which is uniform
within a block and not across them. `DshWorkflow.run` therefore returns
`dict[completion_id, reward]`.

Addressing a turn needs the id AReaL keys its cache by, and the session log does
not carry it. The `rca-completions` row reads it off `finish`'s `replayState`
— documented as "response-level adapter-private metadata (ids, native stop
reason)", where `llm-pi-ai` puts the provider's `responseId` — and writes one
line per request to `$DSH_HOME/rca-completions/<session>.jsonl`. `purpose`
separates the agent's turns from the compaction summarizer, which the proxy
caches too and which no policy chose. If the sidecar's agent-request count does
not match the log's, the mapping is refused whole and the outcome falls back to
the last turn: crediting the wrong turn is worse than crediting none.

## SFT

Teacher trajectories come from `autorl.data.collect`, which points the same
harness composition at a served model instead of AReaL's rollout proxy. Export
the Harness home it fills, then train on it.

```bash
export RCA_GATEWAY_BASE_URL=... RCA_GATEWAY_API_KEY=...
python -m autorl.data.collect $RCA_DATASET_ROOT/data.jsonl .runs/sft-collect/dsh-home \
  --limit 10 --concurrency 5
python -m autorl.data.export .runs/sft-collect/dsh-home .runs/sft/rca_sessions.jsonl
./scripts/run_sft_smoke.sh
```

`data/sft/rca_sessions.jsonl` is the distillation the SFT configs point at by
default, checked in through git-lfs: 50 teacher episodes over a stratified slice
of the corpus, which `autorl.data.sft` expands into 947 training rows. Clone with
`git lfs pull` to get it; a fresh export goes to `.runs/` and is selected with
`train_dataset.path=...`.

`--base-url` takes the same route the rollout takes — see the RL section for why
an endpoint is declared rather than overridden. Without it the episode runs on
`sdk-minimal`'s own `deepseek-official` route, which reads `DEEPSEEK_API_KEY`.
Each episode's submission is re-validated against the bound schema and reported
as `in_contract`; the harness tool already enforced it, so anything other than
all of them means the two sides have drifted.

The exporter writes the conversation the teacher actually held — the session's
final surface, folded the way the harness folds it, plus the tool schemas it was
offered. It decides nothing about supervision: `autorl.data.sft` renders those
messages through the tokenizer's own chat template, once per assistant turn and
exactly as a rollout would, and the mask falls out of that. Per turn rather than
once per trajectory because a thinking model's template renders a turn's
reasoning only when the turn follows the conversation's last `user` message; a
whole trajectory rendered in one pass keeps `<think>` on its last turns and
drops it from every earlier one. Ten episodes become 190 rows.

`actor.mb_spec.max_tokens_per_mb` has to be at least the longest training row —
AReaL's micro-batch packer refuses a row it cannot fit (`Values [8190] is larger
than capacity 4096`) rather than splitting it. The checked-in distillation runs
to 32051 tokens (p50 9890, p90 15442), which is why the config says 32768. On a
card that cannot hold that alongside the optimizer state, cut the tail with
`train_dataset.max_length` instead: 16384 keeps 93% of the rows, 12288 keeps 74%.

Override the dataset or model with normal AReaL config patches:

```bash
./scripts/run_sft_smoke.sh \
  train_dataset.path=/path/to/rca_sessions.jsonl \
  actor.path=/path/to/model
```

## Validation

```bash
uv sync --dev
uv run pre-commit install
./scripts/check.sh
```

The commit hook and GitHub Actions both run Ruff linting, Ruff formatting checks, and
mypy. Unit tests remain a separate command because some SFT coverage uses a locally
cached model tokenizer:

```bash
python -m unittest discover -s tests
```

An end-to-end smoke additionally requires a GPU and the RCA dataset.
