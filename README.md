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

The submodule points at `Lincyaw/AReaL`, branch `rca-autorl`, and not at upstream.
It carries two changes upstream does not have, both of which a rollout needs:
`RolloutWorkflow.rescore_group`, the hook the group weighting is computed in, and
a fix for a client `store: false` being read as "do not cache this interaction",
which silently cost a 279-episode run every one of its training samples. The
submodule's `upstream` remote is the original, so rebasing onto a newer AReaL
stays a normal fetch.

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

Reward is the outcome and nothing else, scored against the siblings that
answered the same case. Every turn of an episode carries the same value; there
is no turn-level credit, because the one candidate measured for it correlated
0.35 with the outcome and 0.41 of that was querying the service the incident
text already names. `.doc/designs/rl-reward.md` works one real case through
every number below, and section 9 has that measurement.

`fpg.compare_model_to_ground_truth` counts every element of the true graph
alike, and that is not what separates a good answer from a lucky one. The
service the incident text already names is a node of the graph too, so an
episode that filters on the endpoint it was handed scores a hit for restating
the prompt while the injected service it had to dig for counts the same.

`n_samples` rollouts of one prompt are enough to tell those apart with no extra
label and no extra cost. `autorl.difficulty` builds the matrix of which sample
found which element and reads the columns: an element every sibling found was
free and is weighted to nothing, one a single sibling found was the case and is
weighted to almost one, one nobody found keeps full weight — the hardest part of
a case is not its least relevant part. Recall is weighted that way; precision is
not, because difficulty is a property of the truth and a claim outside the graph
is simply wrong wherever it lands. On the collected episodes a sibling that
names only the service the prompt named scores 0.00 where the flat comparison
gave it partial credit.

The group is visible in exactly one place, so AReaL grew a hook for it:
`RolloutWorkflow.rescore_group` is called by `GroupedRolloutWorkflow` once every
sample of a prompt has run and before their interactions are merged
(`third_party/AReaL`, three additive changes; `OpenAIProxyWorkflow` forwards it
to the agent it wraps). `DshWorkflow` implements it, and leaves an incomplete
group alone: weights read off a partial group would call its missing parts hard.

Addressing a turn needs the id AReaL keys its cache by, and the session log does
not carry it. The `rca-completions` row reads it off `finish`'s `replayState`
— documented as "response-level adapter-private metadata (ids, native stop
reason)", where `llm-pi-ai` puts the provider's `responseId` — and writes one
line per request to `$DSH_HOME/rca-completions/<session>.jsonl`. `purpose`
separates the agent's turns from the compaction summarizer. Every row the proxy
cached is addressed, the summarizer included: `individual` exports and trains on
it, so leaving it out would let it accumulate its neighbour's value.

Generation limits stay AReaL's: `rollout.model` is the served name the route declares,
`gconfig.max_new_tokens` becomes the request's `max_tokens`, and `sglang.context_length`
becomes the window compaction triggers below. `train.py` passes all three into the
workflow, so none is declared twice.

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

## Running it elsewhere

A checkout is not a runnable environment. Three things the history does not
carry have to arrive first:

```bash
# Behind a firewall, before anything else. git, uv, huggingface and pnpm each
# reach a different host and each reads the environment rather than git's own
# proxy setting, so exporting is what covers all four.
export https_proxy=http://127.0.0.1:7890 http_proxy=$https_proxy all_proxy=$https_proxy
export no_proxy=localhost,127.0.0.1

# A clone made before the submodule was repointed keeps the old url in
# .git/config, where `update` reads it; `sync` is what copies the new one over.
git pull && git submodule sync --recursive
git submodule update --init --recursive          # AReaL
git lfs pull                                      # data/sft/rca_sessions.jsonl
UV_HTTP_TIMEOUT=300 uv sync --python 3.12
# The corpus, not versioned. Copy a working copy into place, then derive the
# two files the trainers read. Idempotent.
rsync -a --info=progress2 \
  --include='*/' --include='cases/*/*.parquet' \
  --include='cases/*/causal_graph_verified.json' --include='cases/*/.invalid' \
  --include='manifest.jsonl' --exclude='*' \
  <host>:<path>/datapacks/ops-lite/ datapacks/ops-lite/
python -m autorl.dataset datapacks/ops-lite       # --check to dry-run
```

The filter is not an optimization, it is the whole set: 3.4G of the case
directory's 7.4G. `result.json` alone is 2.1G and nothing reads it, and
`injection.json`, `label.txt` and `causal_graph.json` are the answer stated
plainly — the `sql` tool materializes `*.parquet` and nothing else, so a case
that does not carry them cannot leak them either.

The published release (`anon-ops/ops-lite` on the Hub) is not a substitute. It
ships `causal_graph.json` and `conclusion.parquet` but not
`causal_graph_verified.json`, which is the `fpg.Scenario` the reward is computed
against — an episode on those cases raises rather than scoring, and the whole
run stops. Transfer a copy that has it, and check before training:

```bash
ls datapacks/ops-lite/cases/*/causal_graph_verified.json | wc -l   # expect 500
```

`python -m autorl.dataset datapacks/ops-lite --check` and `./scripts/check.sh`
both pass without a GPU, so a machine that fails either is not yet ready to
train.

SFT comes before RL, and not as a preference. RL grades a fault propagation
graph, and a base model does not emit one: it never reaches `submit_result`, so
every sibling in the group scores zero, the leave-one-out baseline is zero, and
the advantage is identically zero. There is nothing to learn from until a
checkpoint answers in the contract.

```bash
./scripts/run_sft_smoke.sh total_train_epochs=3 total_train_steps=null
```

SFT needs none of the above except the checkout, `uv sync`, `git lfs pull` and
the base model: it trains on `data/sft/rca_sessions.jsonl`, which already
carries the teacher's conversations. No corpus, no `RCA_DATASET_ROOT`, no
harness bundle, no `pnpm`. The 50 episodes expand to 947 rows, the longest
32051 tokens, which is why `max_tokens_per_mb` is 32768 there too.

```bash
# Confirms the whole SFT input path on a machine with nothing else on it.
python -c "
from areal.utils.hf_utils import load_hf_tokenizer
from autorl.data.sft import build_sft_dataset_from_manifest
t = load_hf_tokenizer('Qwen/Qwen3-4B-Thinking-2507')
d = build_sft_dataset_from_manifest('data/sft/rca_sessions.jsonl', t, max_length=32768)
print(len(d), max(len(r['input_ids']) for r in d))"
```

The saver writes an HF model plus tokenizer under
`<fileroot>/checkpoints/<user>/<experiment_name>/<trial_name>/default/epoch<E>epochstep<S>globalstep<G>`,
which is what RL then loads:

```bash
SFT_ROOT=.runs/dsh-rca-sft-smoke/checkpoints/$USER/autorl-dsh-rca-sft-smoke
CKPT=$(realpath "$(ls -dt $SFT_ROOT/local-smoke/default/epoch* | head -1)")

export RCA_DATASET_ROOT=$PWD/datapacks/ops-lite
export AREAL_ADMIN_KEY=$(openssl rand -hex 16)
./scripts/run_smoke.sh actor.path="$CKPT" total_train_steps=200
```

`run_smoke.sh` is the multi-GPU path; `run_rl_smoke_1gpu.sh` exists only because
a single consumer card needs a dozen overrides that a datacenter node does not.
What has to move with the hardware is the parallelism (`cluster.n_gpus_per_node`
and the `d*p*t*` suffixes of `actor.backend`, `ref.backend`, `rollout.backend`),
the batch (`train_dataset.batch_size`, which `rollout.consumer_batch_size`
follows), and `rollout.max_concurrent_rollouts` — one episode is a `dsh`
subprocess doing DuckDB queries between generations, so a rollout worker spends
most of its wall clock off the GPU and the smoke value of 1 wastes the node.
`gconfig.n_samples` is not a throughput knob: it is the group `rescore_group`
reads difficulty off, and shrinking it makes that weighting noisier.

Two settings are sized so that a real episode does not crash the run, and are
worth knowing before they are overridden. `actor.mb_spec.max_tokens_per_mb`
tracks `sglang.context_length` because AReaL's packer refuses a row it cannot
fit rather than splitting it, and with `export_style: individual` a row is one
whole request. `rollout.agent.admin_api_key` is read from `AREAL_ADMIN_KEY`
because the rollout proxy binds the node's routable address, where AReaL refuses
to serve admin endpoints under the key its own source documents.

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
