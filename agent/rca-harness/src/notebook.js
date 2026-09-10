import { defineTool } from '@deepseek-ai/dsh-tools'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import { mkdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import {
  NOTE_CONTENT_DESCRIPTION,
  NOTE_DESCRIPTION,
  NOTE_ID_DESCRIPTION,
  NOTE_TOOL,
  SQL_TOOL,
  denialReason,
  finalNoteReminder,
  noteRecorded,
  noteReminder,
  noteReplaced,
  notebookEmpty,
  notebookIndex,
  unknownNoteId,
} from './prompts.js'

/**
 * The investigation notebook: the episode's external store, the policy that
 * makes it get written, and the debt the pruner reads.
 *
 * Context management is three mechanisms, in order of what they hold. The
 * notebook holds findings, addressed by name and read on demand. The pruner
 * shrinks noted sql results to a file reference and leaves the unnoted ones
 * whole (`pruner.js`). Compaction replaces the conversation with where the
 * investigation stands, citing note names for what it does not restate
 * (`prompts.js` RCA_INSTRUCTION). The three meet here: a note is what lets the
 * other two throw something away.
 *
 * The notes are also written to `resultRoot/<agent>/notes.md`, beside the
 * `qN.tsv` result files, so what an episode recorded is readable after it ends.
 * Nothing reads that file back in: within a run the map is the store.
 */

const PLUGIN_SOURCE = { kind: 'plugin', plugin: 'rca-note-policy' }

/** How much of a note's first line the index shows. */
const TITLE_CHARS = 120

/** How long a name may be, whether the model wrote it or it came from the summary line. */
const ID_CHARS = 48

/**
 * How many tool results have arrived since each session's last note.
 *
 * The gate and the pruner are separate rows in the tree and cannot share plugin
 * state, so the ledger is module state keyed by session id — which is what both
 * sides already hold, an `exec.agent.id` on one and a `session.id` on the other.
 */
const unnoted = new Map()
let inherited = 0

/** Which of a debt cycle's two notices has been sent, per session. */
const notified = new Map()
const FIRST_SENT = 1
const LAST_CALL_SENT = 2

/**
 * Every episode's notebook, keyed the way the ledger is keyed. Module state
 * rather than plugin state because the checkpoint is written by another row
 * (`compaction.js`), which needs the index of what the episode has stored.
 */
const notebooks = new Map()

/** A forked episode starts owing the parent's unnoted results. */
export function seedUnnoted(count) {
  inherited = count
}

/** How many of the newest results the pruner must leave whole. */
export function unnotedCount(sessionId) {
  return unnoted.get(sessionId) ?? inherited
}

/**
 * A note name: lower case words joined by dashes.
 *
 * The name is the point of the notebook being addressable. A checkpoint that
 * cites `preserve-timeouts-on-travel-call` says what it is pointing at; one
 * that cites `n7` says only that something was recorded, so a model deciding
 * whether the pointer is worth following has to follow it.
 */
function slug(text) {
  const cleaned = text.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '')
  if (cleaned.length <= ID_CHARS) return cleaned
  // A name taken from a summary line ends on a word: a name that cannot be read
  // is not a name.
  const cut = cleaned.slice(0, ID_CHARS)
  return (cut.includes('-') ? cut.slice(0, cut.lastIndexOf('-')) : cut).replace(/-+$/, '')
}

/**
 * Record `content` under the name the model gave, or under one taken from its
 * summary line, and return the name it landed on.
 *
 * Writing to a name that exists replaces that note. That is the one way the
 * notebook shrinks — two partial findings about one service become one, and the
 * name they were both cited under still resolves.
 */
function writeNote(notes, id, content) {
  const named = slug(id ?? '')
  if (named) {
    const replaced = notes.has(named)
    notes.set(named, content)
    return { id: named, replaced }
  }
  // No name given: the summary line is the name. A collision here is not a
  // request to consolidate, so it gets its own name rather than overwrite.
  const derived = slug(content.split('\n', 1)[0]) || 'note'
  let unique = derived
  for (let n = 2; notes.has(unique); n += 1) unique = `${derived}-${n}`
  notes.set(unique, content)
  return { id: unique, replaced: false }
}

/** One line per note: its name and the first line of what it says. */
function renderIndex(notes) {
  if (notes.size === 0) return notebookEmpty()
  return notebookIndex([...notes].map(([id, content]) => {
    const first = content.split('\n', 1)[0].trim()
    return `${id}  ${first.length > TITLE_CHARS ? `${first.slice(0, TITLE_CHARS)}…` : first}`
  }).join('\n'))
}

/** The notes named by a whitespace- or comma-separated list of names. */
function renderNotes(notes, spec) {
  return spec.split(/[\s,]+/).filter(Boolean).map(id => {
    const content = notes.get(id)
    if (content === undefined) throw new Error(unknownNoteId(id, [...notes.keys()].join(', ')))
    return `[${id}] ${content}`
  }).join('\n\n')
}

/**
 * The names in one episode's notebook, newest last, for the checkpoint to carry.
 *
 * A checkpoint cites the notes it used, which leaves a note it did not cite
 * invisible to the model resuming from it — written down and unreachable, which
 * is the failure the notebook exists to prevent. So the whole index rides along,
 * and it is bounded: past `limit` only the newest are listed, with a count of
 * what a full listing would add.
 */
export function notebookOutline(sessionId, limit) {
  const notes = notebooks.get(sessionId)
  if (notes === undefined || notes.size === 0) return undefined
  const names = [...notes.keys()]
  return { names: names.slice(-limit), older: Math.max(0, names.length - limit) }
}

/** The notebook a fresh episode starts with, or the one a fork inherits. */
function createNotes(seed) {
  const notes = new Map()
  for (const note of seed ?? []) writeNote(notes, note.id, note.content)
  return notes
}

/**
 * `take_note`: `content` writes a finding under a name, `id` alone reads those
 * notes back, neither returns the index.
 *
 * The read is what makes the write cheap, and the index is what keeps the read
 * cheap — a read that handed back the whole notebook would put the growth back
 * that the checkpoint no longer carries, and an episode that consults its notes
 * often would pay for all of them every time. For the same reason a write
 * confirms with a name and a count rather than echoing anything.
 */
export function registerNotebook(ctx, state) {
  function getNotebook(exec) {
    const key = state.key(exec)
    // A forked episode inherits the parent's notes under the parent's names.
    if (!notebooks.has(key)) notebooks.set(key, createNotes(state.notes))
    return notebooks.get(key)
  }

  const written = new Set()

  function persist(key, notes) {
    const dir = join(state.resultRoot, key)
    if (!written.has(key)) {
      mkdirSync(dir, { recursive: true })
      written.add(key)
    }
    // Rewritten whole rather than appended: a note can replace one already
    // there, and a notebook is a few tens of short entries.
    const body = [...notes].map(([id, content]) => `[${id}] ${content}`).join('\n\n')
    writeFileSync(join(dir, 'notes.md'), `${body}\n`, 'utf-8')
  }

  ctx.tools.register(defineTool({
    name: NOTE_TOOL,
    description: NOTE_DESCRIPTION,
    parameters: {
      // Both optional: `content` writes, `id` alone reads, neither lists.
      content: { type: 'string', description: NOTE_CONTENT_DESCRIPTION },
      id: { type: 'string', description: NOTE_ID_DESCRIPTION },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          note_id: { type: 'string' },
          replaced: { type: 'boolean' },
          notebook: { type: 'string' },
          count: { type: 'number' },
        },
      },
      render: (_args, value) => [{
        text: value.note_id
          ? (value.replaced ? noteReplaced(value.note_id, value.count) : noteRecorded(value.note_id, value.count))
          : value.notebook,
        type: 'text',
      }],
    },
    execute(args, exec) {
      const notes = getNotebook(exec)
      const content = args.content?.trim()
      const id = args.id?.trim()
      if (!content) {
        return Promise.resolve({
          note_id: '',
          replaced: false,
          notebook: id ? renderNotes(notes, id) : renderIndex(notes),
          count: notes.size,
        })
      }
      const written = writeNote(notes, id, content)
      persist(state.key(exec), notes)
      // The note landed, so the queries behind it are written down and the
      // pruner may shrink them. Cleared from inside the tool because only this
      // side knows the call was a write and not a read: clearing on a read would
      // make an empty `take_note` the way around `noteLimit`, reopening `sql`
      // without the finding the pruner needs in order to shrink anything.
      unnoted.set(state.key(exec), 0)
      notified.delete(state.key(exec))
      return Promise.resolve({
        note_id: written.id,
        replaced: written.replaced,
        notebook: '',
        count: notes.size,
      })
    },
  }))
}

/**
 * Make the notebook happen, in the order that costs least.
 *
 * A reminder rides the `sql` result: it costs no turn and the model can finish
 * the thought it is in. It is sent twice in a debt cycle — on reaching
 * `noteEvery`, and on the last query before the gate closes — because the
 * notice is a message the pruner cannot touch, so one per query would spend
 * more context than the results it is protecting. A denial past `noteLimit`
 * backs it up,
 * and is not there to nag — the pruner leaves every unnoted result whole, so an
 * episode that never notes has an unprunable tail and pressure it cannot
 * relieve. The limit is what bounds that tail. `submit_result` is never gated:
 * an episode ready to answer must be able to.
 */
export function registerNotePolicy(ctx, state, noteEvery, noteLimit) {
  ctx.on('tools/pre-execute', async (exec, next) => {
    if (exec.name !== SQL_TOOL) return next()
    const key = state.key(exec)
    const used = unnotedCount(key)
    // The reason says outright that rephrasing fails: answered as an ordinary
    // tool error, a denial buys a round of rewritten queries instead of a note.
    if (used >= noteLimit) return { kind: 'deny', reason: denialReason(used) }

    const decision = await next()
    if (decision.kind !== 'deny') unnoted.set(key, used + 1)
    return decision
  })

  // Observe and enrich, never veto: delegate first so a later listener can
  // still block or replace, then fold the notice onto whatever came back.
  ctx.on('tools/post-execute', async (exec, _result, next) => {
    const downstream = await next()
    if (exec.name !== SQL_TOOL) return downstream

    // Once per threshold crossed, not once per query, and by crossing rather
    // than by equality: a turn can call `sql` several times at once, so the
    // count jumps and an exact match would skip the notice altogether.
    const key = state.key(exec)
    const used = unnotedCount(key)
    const sent = notified.get(key) ?? 0
    const last = used >= noteLimit - 1
    if (last) {
      if (sent >= LAST_CALL_SENT) return downstream
      notified.set(key, LAST_CALL_SENT)
    } else {
      if (used < noteEvery || sent >= FIRST_SENT) return downstream
      notified.set(key, FIRST_SENT)
    }

    const notice = createUserMessage({
      content: [{
        type: 'text',
        text: last ? finalNoteReminder(used, noteLimit) : noteReminder(used, noteLimit),
      }],
      source: { ...PLUGIN_SOURCE, form: 'notice', summary: `${used} queries unnoted` },
    })
    return {
      ...downstream,
      additionalContexts: [notice, ...(downstream.additionalContexts ?? [])],
    }
  })
}
