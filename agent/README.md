# The RCA harness

DeepSeek Harness (`dsh`) owns the agent loop. This directory owns what an RCA
episode needs on top of it: the action space, the answer channel, and the bound
on how long an episode may take.

```text
agent/
  rca-harness/       the bundle @rca-autorl/dsh-rca-harness — the mechanism
    cordis.patch.yml   the bundle layer: disable the shell, mount our row
    src/index.js       plugin entry: config validation, episode state
    src/sql-tool.js        the sql tool — the whole evidence surface
    src/notebook.js        the take_note investigation notebook
    src/submit-result.js   the terminal submit_result tool
    src/contract.js        the answer schema and the rules the schema DSL cannot state
    src/vocabulary.js      GENERATED from configs/fpg/microservices.toml
    src/compaction.js      the RCA checkpoint template
    src/note-policy.js     the note reminder and its backstop
    src/note-ledger.js     unnoted-result count, shared by policy and pruner
    src/pruner.js          the note-aware tool-result pruner
    src/debug.js           the env-gated trace
  profiles/
    rca.patch.yml    the RCA scenario layer — the persona
    openai-gateway.patch.yml  an OpenAI-compatible gateway as a model route
```

The model sees exactly three tools: `sql`, `take_note`, and `submit_result`.

## Why a bundle at all

The composed tree starts from the shipped `sdk-minimal` profile, which is the
only profile that accepts an arbitrary model id — the AReaL rollout proxy serves
the model being trained, not a catalog entry. What `sdk-minimal` gives an
episode is a general coding agent: persistent bash, `str_replace_editor`, local
execution, a JSONL session store. An RCA episode needs almost none of that, and needs what it does not have.

A bundle rather than a pile of patch rows because a bundle carries its own patch
layer: `dsh plugin add` records it in the profile's `dsh.profile.bundles`, and
the rows appear in every session that profile starts. Patches alone cannot add
behavior that no shipped package implements.

## The action space is one SQL tool

An RCABench datapack is telemetry in Parquet — normal-window and abnormal-window
logs, metrics, and traces — so the honest action space is a query interface, not
a machine. `sql` opens one in-process DuckDB per episode, materializes every
`*.parquet` in the snapshot as a table named after the file stem, and answers
statements against it. Schema discovery is SQL too (`SHOW TABLES`,
`DESCRIBE abnormal_logs`), which is why one tool is enough and there is no
second tool for listing or describing anything.

Results are bounded three ways — `maxRows`, a total `maxChars`, and a per-cell
`maxCellChars` — because the episode's context is the resource an RCA agent
actually spends. One unfiltered `SELECT` over trace spans is otherwise a single
15 000-character tool result. A bounded result is not a dead end: the status
line reports how many rows matched, `offset` pages through them, and the same
line says outright that aggregating beats paging. The table itself is TSV, not
JSON — the same rows for a fraction of the tokens. The canonical value stays
structured JSON (`columns`, `rows`, `offset`, `matched_rows`, `truncated`) for
anything reading the log programmatically.

Dropping the shell is not a hardening afterthought, it is the design:

- **The dataset is not reachable.** A datapack ships its own answer beside its
  evidence: `injection.json` names the injected fault's source and target
  service, `causal_graph.json` is the ground-truth propagation graph,
  `result.json` carries the labeled paths. They are not registered as tables,
  and `enable_external_access` is switched off once the tables exist, so
  DuckDB's own `read_parquet` / `read_json_auto` / `read_csv` answer
  `Permission Error` instead of handing over the label. With a shell in the
  roster this would be a sandbox to keep honest forever; without one there is
  nothing to confine.
- **Episodes stay comparable.** `avg_turns` and `false_positive_rate` mean
  something when every episode moves through the same query interface. They
  mean much less when one episode greps and another writes Python.
- **No wandering.** The first shell-based episode we ran spent its wall clock on
  `find /` sweeps of the whole machine after the snapshot had already answered
  the question in four calls. A tool that can only reach registered tables
  cannot do that.

## The investigation notebook

`take_note` records a finding in a persistent in-memory notebook. Each call
returns the full notebook in the tool result, so the model's accumulated
findings stay visible even after older messages are compacted. The notebook is
ephemeral to the episode — it resets between episodes.

The tool exists for context management: SQL results can be large and fill the
context window. By noting the key finding from each query, the model preserves
what matters. When older tool results are compacted, the most recent
`take_note` result still contains the entire notebook.

Notes are not terminal and are not rationed: the model takes as many as it
wants.

## The answer channel

`submit_result` is the only thing the trainer reads. An RL episode needs an
answer a verifier can score, not prose to parse, so the tool's parameter schema
*is* the fault propagation graph contract — and that contract is not ours to
invent. It is `ModelRCAOutput` from the [`fpg`](https://github.com/Lincyaw/fpg-convention)
package, bound to the vocabulary in `configs/fpg/microservices.toml`: nodes are
time-anchored statements (`subject` entity, `predicate` failure mode, window,
re-executable evidence), edges carry existence and direction only, and
`root_causes` are stated rather than read off the graph. The ground truth under
`datapacks/ops-lite` is annotated in the same schema and the same vocabulary,
which is the point — a submission and the answer it is scored against differ in
content, never in shape.

Structure is one thing and value space another, and the second is what actually
decides whether an answer can be scored. Before the schema was bound, `subject`
and `predicate` were free text, and all 98 nodes of the first ten collected
episodes were out of contract: subjects like `ts-station-service HikariCP
connection pool` mixed an entity with a component, predicates ran to a median
115 characters of prose, and two episodes wrote `+01` for a timezone offset. No
set-matching verifier can score that, and no reward built on it would mean
anything.

Enforcement is split by what each half can express. The registry validates the
model's arguments against the parameter schema before `execute` runs, which
covers types, required keys, and the two closed vocabularies that fit the
enforced JSON Schema subset as `enum` — the predicate and the evidence query
language. The subset has no `pattern`, no `minItems`, and no cross-field rules,
so `contract.js` hand-checks the rest: the entity-reference format, ISO 8601
timestamps with a real offset, `start` not after `end`, evidence unless the node
is a hypothesis, unique node ids, no self-loops, and every edge endpoint and root
cause resolving to a declared id. A violation becomes an ordinary error result,
so the model retries inside the same turn.

That leaves one schema enforced in two languages, which is a new way to drift —
the old failure was a shape copied by hand with none of its rules, and copying
the rules by hand instead would only move the problem. So neither side is
written: `src/vocabulary.js` is generated from that profile by
`python -m autorl.fpg`, and the entity-reference regex in it is read off the
bound pydantic model rather than restated. What cannot be generated is tested.
`tests/test_submit_result_contract.py` puts nineteen fixtures through
`autorl.fpg.parse_submission` and the seventeen that `validate` owns through
`contract.js` under Node, and requires the same verdict from each.

The call is terminal *once it executes*: `exec.concludeTurn()` stops the loop
after the step, and a monotonic `ctx.tools.guard()` denies every later call in
the same response. Schema validation runs before `execute`, so a rejected call
ends nothing and arms nothing, and the model retries inside the same turn — an
episode can leave several `tool/call` events, the rejected ones first. Three of
the first ten collected episodes did. `autorl.agent.submitted_result` therefore
pairs each call with its `tool/result` by call id and reads the one the tool
accepted; taking the first would hand the verifier a submission with no
`root_causes` and score a recovered episode zero. Nothing else needs to persist
the answer: model-visible input is already reconstructable from that log.

## Context management

`sdk-minimal` mounts none: no token meter, no compaction, so an investigation
runs until the serving window rejects the request. The bundle patch adds the
base bundle's own rows rather than inheriting them, which keeps the roster an
allowlist — a later base-bundle addition cannot change what an episode can do.

Two thresholds have to be told the truth or the whole thing is inert.
`compaction-basic` compacts at `thresholdRatio` (0.8) of the *model's* context
window, which `llm-deepseek` reads from `$DSH_CONTEXT_WINDOW` and otherwise
assumes is 1M; `train.py` passes `sglang.context_length` through to it, so the
serving window is declared once, by AReaL. The
`tool-result-pruner` threshold sits below the `sql` tool's `maxChars`, so a past
result shrinks to a head plus its `(saved to qN.tsv)` tail while the notebook
keeps the finding.

### Why the checkpoint template is ours

The shipped compaction instruction opens with "You are now acting as a
compaction engine for this AI coding assistant" and fixes eight sections around
*Files and Code*, *Errors and Fixes*, and "preserve exact file paths, commands,
function signatures". An RCA episode has none of those: half the sections
compact to "(none)", and what must actually survive has nowhere to go — which
SQL already ran, what each result showed, the causal chain so far, which
hypotheses are still open.

The instruction is not configurable; `BasicCompactionConfig` exposes ratios,
retention, and the summarization route, never the prompt. But the backend
documents `summarize()` as the sole hook to override "for a template or remote
summarizer", and `BasicCompactionEngine` is exported, so `src/compaction.js`
subclasses it and replaces only the instruction and the one-shot call. Region
selection, token accounting, checkpoint landing, and the
`<compacted-summary>` framing all stay with the shipped engine.

The RCA template keeps verbatim only the SQL that grounds a finding — a
submitted node must cite the statement behind it — and records every other
query as one line without its text, which is enough to stop a re-run. Scoping
it that way is not tidiness: asked for every query verbatim, the checkpoint
grew with the span it was replacing, and across the first ten collected
episodes 45% of all compaction time went to summaries discarded for
overrunning the token cap or for not being smaller than what they shadowed
(875 queries run, 148 cited). The template also keeps service, metric, table,
and column names exact, and forbids promoting a hypothesis to a finding across
a checkpoint.

## Mechanism here, scenario there

`agent/rca-harness/cordis.patch.yml` mounts the row with its default config.
`agent/profiles/rca.patch.yml` is the per-launch layer `DshWorkflow` passes, and
it owns the persona. The split is what lets the Fault Injection, Verifier, and
Controller agents reuse the same submission and budget machinery under their own
persona and row config; a patch replaces a row's whole config, so a scenario
layer that changes one field restates the rest.

The persona is part of the mechanism, not decoration. It states that the
snapshot is a self-contained export and that a gap in it is a finding to record
as a hypothesis — a belief about the task that no guard can install.

## Debugging a composition

Set `RCA_HARNESS_LOG` to a file path and the bundle appends one JSON line per
decision: the row's config at mount, every pressure check with the context
window and measurement it decided on, every prune pass with how much it
protected and removed, and every checkpoint. It is inert when the variable is
unset, and it writes to a file rather than stdout, which the JSON-RPC server
owns.

It is worth the twenty lines. A `#private` field in the pruner subclass threw
`Cannot write private member` on every compaction attempt — a mounted service
reaches its own methods through a proxy — and the engine swallowed each failure,
so the session log showed no compaction at all and no error anywhere. Three
black-box bisection runs blamed the wrong component; the first traced run named
the line. (One of those bisects was invalid on top of that: a patch cannot
change a row's `name`, it only guards it, so the "revert to the stock pruner"
layer was silently skipped with a warning on stderr.)

## Install and iterate

```bash
python -m autorl.fpg                                 # after editing the vocabulary profile
python -m autorl.harness .runs/dsh-home              # first install
python -m autorl.harness .runs/dsh-home --reinstall  # after editing the bundle
```

`src/vocabulary.js` is generated, so a profile edit reaches the tool only after
`python -m autorl.fpg` — which `scripts/check.sh` verifies, since the commit
hook runs that and nothing else.

`dsh plugin` shells out to `pnpm`; on a machine with only corepack, the install
writes that shim for the subprocess itself, so this needs no setup. It
uses `file:`, which copies the bundle into the profile package tree — the only
place its `@deepseek-ai/dsh-tools` peer import resolves — and installs its own
`@duckdb/node-api` dependency there, native prebuild included. An edit here
reaches a session only after `--reinstall`, and reinstall removes the package
before adding it: `add` on an unchanged `file:` spec is a pnpm no-op no matter
how much the directory changed, which silently keeps the old copy running while
every check reports success.

The bundle is plain ESM JavaScript with no build step, because it is small and a
training host should not need a TypeScript toolchain. If it grows past what
reads clearly untyped, port it to TypeScript with a `dist/` build.

Verify a change the way the harness documents it: `dsh --profile sdk-minimal
--patch agent/profiles/rca.patch.yml --dump-config` shows the composed rows, and
one real episode against a snapshot shows whether the row actually runs — a row
whose `inject` is unsatisfied stays dormant without an error. Each mechanism has
a prompt that exercises it: `SHOW TABLES` and a `read_json_auto('result.json')`
attempt (the tables appear, the escape is refused), an unfiltered `SELECT` over
`abnormal_traces` followed by its next page (the cap, the status line, and
`offset` all report honestly), and one ordinary incident (the submission arrives
and ends the turn).
