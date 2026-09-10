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
  + 'compaction. Three modes. With `content` and `id` it records a finding under '
  + 'that name. With `id` alone it hands those notes back. With neither it lists '
  + 'the notebook — one line per note, its name and its first line — so listing '
  + 'costs the same whether you have three notes or thirty. Write after each sql '
  + 'query: older results are compacted to file references between steps and a '
  + 'checkpoint keeps only where you stand, so the notebook is where a finding '
  + 'survives. Read by name whenever you need a statement, a number or a finding '
  + 'from earlier. A write confirms without echoing anything, so reading is '
  + 'always a deliberate call.'

export const NOTE_CONTENT_DESCRIPTION =
  'The finding to record. Open with a one-line summary, then what you observed, the SQL '
  + 'that showed it, and what it implies. Omit to read the notebook instead of writing to it.'

export const NOTE_ID_DESCRIPTION =
  'The note\'s name: lower case words joined by dashes, saying what the note is about — '
  + '`preserve-timeouts-on-travel-call`, not `note-7`. Written with `content`, it names the '
  + 'note, and writing to a name that already exists replaces it, which is how two partial '
  + 'findings become one. Written alone, it reads notes back: one name, or several as '
  + '`a-name, another-name`. Omit both this and `content` to list the notebook.'

/** What a note write confirms with, and the three ways a read comes back. */
export const noteRecorded = (id, count) => `Note \`${id}\` recorded (${count} in the notebook).`

export const noteReplaced = (id, count) =>
  `Note \`${id}\` replaced (${count} in the notebook).`

export const notebookEmpty = () => 'The notebook is empty.'

export const notebookIndex = body =>
  `--- Investigation Notebook ---\n${body}\n`
  + `(read one with \`${NOTE_TOOL}\` id: "a-name", or several with id: "a-name, another-name")`

export const unknownNoteId = (id, known) =>
  `no note ${JSON.stringify(id)} in the notebook; it holds ${known}`

/** What is left where an earlier notebook read used to be. */
export const foldedNotebookRead = () =>
  `(an earlier notebook read, folded away — call \`${NOTE_TOOL}\` again for the notes you need)`

/**
 * What the model is told when it has queried without noting. Twice per debt
 * cycle, not once per query: the notice is a message that stays in the surface
 * and only compaction can take it away, so repeating it spends the context it
 * is trying to protect.
 */
export const noteReminder = (used, noteLimit) =>
  `Note reminder: ${used} queries have run since your last note.\n`
  + `Call \`${NOTE_TOOL}\` with what they established before the next query. Older `
  + 'results are compacted away and only the notebook survives, so an unwritten '
  + `finding is lost. After ${noteLimit} unnoted queries \`${SQL_TOOL}\` is refused `
  + 'until a note lands.'

/** The last query before the gate closes. */
export const finalNoteReminder = (used, noteLimit) =>
  `Note reminder: ${used} queries have run since your last note, and \`${SQL_TOOL}\` `
  + `closes at ${noteLimit}. This is the last query you get before it does. Call `
  + `\`${NOTE_TOOL}\` with what these queries established and it reopens immediately.`

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
 * What a second attempt is told about the first. A retry that repeats the same
 * request is a coin flip; this is what makes it a different one.
 */
export const checkpointRetryHint = () =>
  '\nYour previous attempt at this checkpoint was rejected for being too long. It must be '
  + 'shorter than the conversation it replaces. Cut it hard: keep the draft answer, the open '
  + 'questions and the next step, collapse Covered Ground to the fewest lines that still say '
  + 'what is settled, and name notes instead of describing what is in them.'

/**
 * The notebook's index, carried with the checkpoint so that a note the
 * checkpoint did not cite is still reachable by name.
 */
export const notebookOutlinePreamble = (body, older) =>
  `Notes you have stored, by name — read any of them with \`${NOTE_TOOL}\` id:\n${body}`
  + (older > 0 ? `\n- (${older} earlier notes; \`${NOTE_TOOL}\` with no argument lists them all)` : '')
  + '\n'

/** The episode's own question, carried verbatim at the head of every checkpoint. */
export const incidentPreamble = incident =>
  `The incident you are investigating, as it was reported, verbatim:\n\n${incident}\n\n`

/**
 * The RCA checkpoint instruction.
 *
 * `compaction-basic` ships a coding-assistant template — Files and Code, Errors
 * and Fixes — which an RCA episode has nothing to put in.
 *
 * This one asks for where the investigation STANDS, not a record of what it
 * did. A record grows with the episode, so the checkpoint converges on the size
 * of the region it is meant to replace and a pass stops freeing anything. A
 * position is bounded by the incident: a draft of the answer, what is still
 * open, and one line per area covered. What a `take_note` read would return is
 * cited by name instead of restated — a name is a few characters where the
 * finding is a few hundred, so what stays monotone barely moves.
 *
 * Budgets are counted in items rather than characters: the model can count
 * bullets as it writes and cannot count characters it has not written yet.
 *
 * The instruction also has to argue with the persona. `compaction.js` keeps the
 * conversation's own system prompt in front of this request so the provider can
 * reuse its warm prefix, and that system prompt tells the agent a finding it
 * does not write down is lost. Read by the summarizer that is an instruction to
 * hoard, and it arrives first. Hence the second paragraph below: it says
 * outright that the rule does not apply here, and why.
 */
export const RCA_INSTRUCTION = [
  'You are now acting as a compaction engine for a root-cause analysis agent. Rewrite the investigation ABOVE into a checkpoint that lets another model resume it.',
  '',
  `For this request you are not the investigating agent, and its rule that an unwritten finding is lost does not apply to you. The notebook is a store that outlives this compaction, and the resuming model reads it with \`${NOTE_TOOL}\` — no argument for the index, \`id\` for the notes it wants. Every note is named for what it says, so a name is worth reading on its own. Everything a notebook read would return is already safe: do not restate it here, name the note instead.`,
  '',
  'The checkpoint is where the investigation STANDS, not a record of what it did. It replaces the previous checkpoint and must not be longer than it — an investigation that has covered more ground states its position more briefly, because more of it has settled.',
  '',
  'The reported incident and the list of note names are carried for you above whatever you write. Do not restate either: no description of the incident, and no inventory of the notebook. Name a note where a claim rests on it.',
  '',
  'Output EXACTLY the Markdown structure below: keep every section, in order. One line per bullet, nothing nested. Write "(none)" for an empty section — never drop a section.',
  '',
  '## Snapshot Schema',
  '- [one line per table that has mattered, with the columns used. Carry forward; do not re-derive]',
  '',
  '## Draft Answer',
  '- [the graph you would submit if asked right now — one line per node: <kind>:<name> | failure mode | window | the name of the note holding its evidence]',
  '- [then the edges established, cause first: A -> B]',
  '- [then which nodes are the root causes]',
  '',
  '## Open Questions',
  '- [what is not settled, and the query that would settle it]',
  '',
  '## Covered Ground',
  '- [one line per AREA already queried and what it settled or excluded, with the note names — not one line per query]',
  '',
  '## Next Step',
  '- [the single next query, or the decision to submit]',
  '',
  'Budgets, counted as you write:',
  '- At most 12 node lines, 8 edges, 6 open questions, 10 covered-ground lines.',
  '- When Covered Ground would run past 10 lines, merge instead of trimming: several queries over one area become one line naming the area, what it settled, and the notes.',
  '',
  'Rules:',
  '- A <compacted-summary> block above is the previous checkpoint, and it is what you are rewriting: carry Snapshot Schema over as it stands, update the rest, and let go of what no longer holds.',
  '- No SQL text, no query results, no log lines, no counts read off a result. Those are in the notebook; cite the note by name.',
  '- Entity, table, column and metric names and the abnormal window must be exact where you write them. Do not spell out anything else.',
  '- Drop what has stopped carrying weight: an answered question leaves Open Questions, a settled area collapses into one line, a node the evidence killed leaves the draft answer. Nothing here is permanent — the record is the notebook.',
  '- Never promote a hypothesis to a grounded node. A step you believe but cannot show stays under Open Questions, or goes into the draft answer marked hypothesis.',
  '- Do NOT mention this summarization request or that the context was compacted.',
  '- Output only the checkpoint text: do not call any tool or take any other action.',
].join('\n')
