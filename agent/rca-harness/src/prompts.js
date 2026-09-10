import { ENTITY_TYPES, NODE_PREDICATES } from './vocabulary.js'

/**
 * Everything the model reads, in one file.
 *
 * Not tidiness: an audience boundary. These strings go to the agent being
 * evaluated, while the rest of this repository — the profile's comments, the
 * READMEs, `fpg`'s own field docs — is written for whoever builds the task.
 * Prose that crosses from the second audience to the first hands over the
 * answer. It has happened twice: `fpg` documents `root_causes` as "injection
 * points plus preconditions" and that phrase was copied into the tool schema,
 * telling the model its incident was staged; and a `link` definition once
 * listed which predicates go with it, which is the annotation's convention, not
 * a definition. Both read as ordinary prose in review.
 *
 * So the rule is structural: model-visible prose lives here and nowhere else,
 * and `tests/test_no_answer_leak.py` scans this file plus the scenario persona.
 * Nothing here imports the harness, which is what lets the test read it.
 *
 * What may be said: what a tool does, what the schema means, what the snapshot
 * contains. What may not: that a fault was injected, which faults exist, how an
 * answer is scored, which values the annotation prefers, what the corpus or the
 * testbed is called.
 */

/** Model-facing tool names, so the prose and the registrations cannot disagree. */
export const SQL_TOOL = 'sql'
export const NOTE_TOOL = 'take_note'
export const SUBMIT_TOOL = 'submit_result'

/** The `sql` tool. Limits are interpolated so the text cannot outlive the config. */
export const sqlDescription = (maxRows, maxChars) =>
  'Query the incident snapshot with DuckDB SQL. Each telemetry file is a table named after '
  + 'the file without its extension (for example `abnormal_logs`, `normal_metrics`, '
  + '`abnormal_traces`). Start with `SHOW TABLES`, then `DESCRIBE <table>` for its columns. '
  + `A result is capped at ${maxRows} rows and about ${maxChars} characters; the status line `
  + 'reports how many rows matched, and `offset` pages through them. Prefer aggregating over '
  + `paging. After reviewing the result, call \`${NOTE_TOOL}\` to record your finding — this also `
  + 'compacts the sql result out of context, leaving a file reference. The finding then lives in '
  + 'the notebook, which you can read back at any time.'

export const SQL_STATEMENT_DESCRIPTION = 'One DuckDB SQL statement.'
export const SQL_OFFSET_DESCRIPTION =
  'Skip this many matched rows before returning; for paging a capped result.'

/** The `take_note` tool. */
export const NOTE_DESCRIPTION =
  'The investigation notebook: your external store, which survives context '
  + 'compaction. With `content`, records a finding. WITHOUT `content`, returns '
  + 'the whole notebook — call it that way whenever you need a statement, a '
  + 'number or a finding from earlier in the investigation. Write after each sql '
  + 'query: older results are compacted to file references between steps and a '
  + 'checkpoint keeps conclusions rather than evidence, so the notebook is where '
  + 'a finding survives. A write confirms without echoing the notebook, so '
  + 'reading it back is a deliberate call.'

export const NOTE_CONTENT_DESCRIPTION =
  'The finding to record: what you observed, the SQL that showed it, and what it implies. '
  + 'Omit to read the notebook instead of writing to it.'

/** What the model is told when it has queried without noting. */
export const noteReminder = (used, noteLimit) =>
  `Note reminder: ${used} queries have run since your last note.\n`
  + `Call \`${NOTE_TOOL}\` with what they established before the next query. Older `
  + 'results are compacted away and only the notebook survives, so an unwritten '
  + `finding is lost. After ${noteLimit} unnoted queries \`${SQL_TOOL}\` is refused `
  + 'until a note lands.'

/** What the model is told when the note policy closes querying. */
export const denialReason = used =>
  `${used} queries have run since your last note, so \`${SQL_TOOL}\` is closed until one `
  + `lands. Rephrasing this query will be denied too. Call \`${NOTE_TOOL}\` now with what `
  + 'those queries established — the statement, what its result showed, and what it '
  + 'implies — and querying reopens immediately. Older results are compacted to a file '
  + 'reference you cannot read back, so an unwritten finding is lost.'

/** The `submit_result` tool and its fields. */
export const submitDescription = vocabVersion =>
  'Submit the final fault propagation graph and its root causes. Call this exactly once, '
  + `when the investigation is complete; it ends the episode. Schema vocabulary: ${vocabVersion}.`

export const NODES_DESCRIPTION =
  'Propagation graph nodes: time-anchored, verifiable statements of the form "this entity '
  + 'exhibited this failure mode during this window", never free prose. A cause that fans out '
  + 'to several effects, or an effect that needs two causes at once, is why this is a graph. '
  + 'One node per thing that went wrong: propagation between two entities is an edge, so do '
  + 'not add a node to stand for the hop.'

export const NODE_ID_DESCRIPTION =
  'Node id, unique within this submission, referenced by edges and root_causes.'

/**
 * The subject vocabulary. Each kind's sentence is the profile's own, so what the
 * model is told and what the verifier enforces cannot say different things.
 */
export const SUBJECT_DESCRIPTION = [
  'The entity this node is about, as `<kind>:<name>`. Name a real entity from the snapshot. Kinds:',
  ...ENTITY_TYPES.map(entity => `- ${entity.prefix}: ${entity.brief}`),
].join('\n')

/** The predicate vocabulary, handed to the model as value plus meaning. */
export const PREDICATE_DESCRIPTION = [
  'The failure mode the subject exhibits — what is broken about it, never its cause or its',
  'consequence (those are separate nodes, joined by an edge). Pick the one value that fits:',
  ...NODE_PREDICATES.map(predicate => `- ${predicate.value}: ${predicate.brief}`),
].join('\n')

export const TIME_DESCRIPTION =
  'The window during which the subject exhibited the predicate, as the evidence bounds it. '
  + 'A cause may not start after its effect.'
export const TIME_START_DESCRIPTION =
  'ISO 8601 timestamp with a timezone offset, e.g. 2025-07-21T14:47:09+01:00. An instant has '
  + 'start == end.'
export const TIME_END_DESCRIPTION =
  'ISO 8601 timestamp with a timezone offset, not before start.'

export const EVIDENCE_DESCRIPTION =
  'What makes this node checkable rather than an opinion: the SQL you actually ran and '
  + 'what its result showed. Required unless hypothesis is true. Anyone can re-run it, so '
  + 'evidence that does not exist or does not support the predicate falsifies the node.'
export const EVIDENCE_LANGUAGE_DESCRIPTION =
  'How to re-execute the statement. Everything this harness can run is sql.'
export const EVIDENCE_STATEMENT_DESCRIPTION = 'The query verbatim and complete, as run.'
export const EVIDENCE_EXPLANATION_DESCRIPTION =
  'What the result showed, in terms the predicate is stated in.'

export const HYPOTHESIS_DESCRIPTION =
  'True for a step you believe is real but the snapshot cannot show. It is exempt from the '
  + 'evidence requirement: mark such a step rather than dropping it from the graph or asserting '
  + 'it as observed.'

export const EDGES_DESCRIPTION =
  'Causal edges between node ids, cause first. Existence and direction only: the propagation '
  + 'mechanism is not asked for.'
export const EDGE_SRC_DESCRIPTION = 'Cause-side node id.'
export const EDGE_DST_DESCRIPTION = 'Effect-side node id.'

export const ROOT_CAUSES_DESCRIPTION =
  'Node ids that are root causes, most confident first. State them; they are not read off the '
  + 'graph, so one missing edge cannot invent a root. More than one is allowed.'

/** What a call after the submission tells the model. */
export const alreadySubmitted = toolName =>
  `the root-cause analysis is already submitted, so \`${toolName}\` is not executed`

/**
 * What a rejected submission tells the model. Every one of these reaches it as
 * a tool error and is retried against, so they are prompts too.
 */
export const reject = {
  emptyNodeId: () => 'every node needs a non-empty id',
  duplicateNodeId: id => `duplicate node id ${JSON.stringify(id)}`,
  entityRef: (where, subject, prefixes) =>
    `${where}: subject ${JSON.stringify(subject)} is not an entity reference. Write `
    + `<kind>:<name> with kind one of ${prefixes.join(', ')} and name a bare identifier `
    + '(letters, digits, . _ - >). A description of what is wrong, or of which part of the '
    + 'entity is wrong, belongs in the predicate and the evidence.',
  timestamp: (where, bound, value) =>
    `${where}: time.${bound} ${JSON.stringify(value)} is not an ISO 8601 timestamp with a `
    + 'timezone offset',
  timeOrder: where => `${where}: time.start is after time.end`,
  evidenceRequired: where => `${where}: evidence is required unless hypothesis is true`,
  emptyList: field => `\`${field}\` must not be empty`,
  unknownEdgeEndpoint: (field, id) =>
    `edge.${field} ${JSON.stringify(id)} is not a declared node id`,
  selfLoop: id => `edge ${JSON.stringify(id)} -> itself is a self-loop`,
  unknownRootCause: id => `root cause ${JSON.stringify(id)} is not a declared node id`,
}

/** The `sql` result's status line, which tells the model how to read a capped result. */
export const nextPageHint = last =>
  `; for the next page repeat this statement with offset: ${last}, or aggregate/filter for a `
  + 'sharper answer'

/**
 * The RCA checkpoint instruction.
 *
 * `compaction-basic` ships a coding-assistant template — Files and Code, Errors
 * and Fixes, "preserve exact file paths, commands, function signatures". An RCA
 * episode has none of those, so half its sections compact to "(none)" and the
 * things that must survive have nowhere to go: which SQL already ran (so the
 * agent does not re-run it), what each result showed, the causal chain built so
 * far, and which hypotheses are still open.
 *
 * This template keeps exactly those, and keeps them in the vocabulary the
 * submission needs, so a compacted episode can still produce a graph.
 *
 * **No SQL text at all, and no evidence.** Twice now this template has carried
 * statements and twice the checkpoint outgrew the region it replaces. First it
 * asked for every query verbatim: across ten collected episodes that carried
 * 875 statements to cite 148, and the surplus grew with episode length. Scoping
 * it to the statements that ground a finding did not fix the shape, because that
 * set is monotone too — a finding is never un-established, so its statement is
 * never droppable. Measured over one episode's three passes, the sections that
 * accumulate by definition grew the summary from 10123 to 21886 characters
 * (Evidence Statements +5544, Ground Already Covered +3354, Established
 * Findings +1637) while Incident and Snapshot Schema did not move at all. The
 * region each pass could replace stayed at 9-12k tokens, so the net saving fell
 * from 5587 tokens to 2312, context settled at 20416 above the 19661 threshold
 * it had just compacted below, and the next step tripped compaction again —
 * MAX_TOKENS, then a 400, then the episode was over.
 *
 * The fix is not a smaller budget for evidence but a different home for it. The
 * notebook already persists across passes; what it lacked was a way to read it
 * back, so the checkpoint was evidence's only route out of a compaction pass.
 * With `take_note` readable (`notebook.js`), the checkpoint states where the
 * investigation stands and the notebook holds what it found — and the parts of
 * a checkpoint that are bounded by the incident rather than by the episode
 * (Incident, Snapshot Schema, the open questions) are the parts that stay.
 */
export const RCA_INSTRUCTION = [
  'You are now acting as a compaction engine for a root-cause analysis agent. Condense the investigation ABOVE into a structured checkpoint that lets another model resume it.',
  '',
  'The checkpoint carries the state of the investigation, NOT its evidence. The evidence is in the notebook, which survives compaction and which the resuming model reads by calling `take_note` with no argument. Do not copy findings, statements or numbers here that a notebook read would return.',
  '',
  'Output EXACTLY the Markdown structure below: keep every section, in order. Use terse bullets, not prose paragraphs. Write "(none)" for an empty section — never drop a section.',
  '',
  '## Incident',
  '- [the reported symptom: affected endpoints or services, the alerting signal, and the abnormal window]',
  '',
  '## Snapshot Schema',
  '- [tables seen so far and the columns that mattered, so they are not re-discovered]',
  '',
  '## Ground Already Covered',
  '- [what has been queried and what it showed or excluded — one line each, no SQL text]',
  '',
  '## Established Findings',
  '- [what the evidence settles, one line each — the claim, not the data behind it]',
  '',
  '## Causal Chain So Far',
  '- [the propagation edges established so far, cause first: A -> B]',
  '',
  '## Open Hypotheses',
  '- [asserted but not yet grounded, and the query that would settle each]',
  '',
  '## Ruled Out',
  '- [candidates the evidence excludes, and what excluded them — so they are not revisited]',
  '',
  '## Next Step',
  '- [the single next query or the decision to submit, or "(none)"]',
  '',
  'Rules:',
  '- No SQL text anywhere in the checkpoint. A query is one line under Ground Already Covered saying what it showed; the statement itself is in the notebook, and the resuming model reads it there when a submission has to cite it.',
  '- Preserve exact service names, metric names, table and column names, timestamps, and the abnormal window. These are what a resuming model cannot re-derive without re-querying.',
  '- Never promote a hypothesis to a finding. If the evidence did not settle it, it stays under Open Hypotheses.',
  '- Do NOT mention this summarization request or that the context was compacted.',
  '- Output only the checkpoint text: do not call any tool or take any other action.',
  '- If the conversation already contains a <compacted-summary> block, it is a PRIOR checkpoint. Do not copy it forward verbatim: preserve still-true facts, drop stale ones, and merge newer information into a single consolidated checkpoint under the same structure. The checkpoint must not grow from pass to pass — an investigation that has covered more ground states it more briefly, it does not accumulate.',
].join('\n')
