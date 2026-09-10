import { ToolResultPruner } from '@deepseek-ai/dsh-compaction-tool-result-pruner'
import { trace } from './debug.js'
import { unnotedCount } from './notebook.js'
import { NOTE_TOOL, SQL_TOOL } from './prompts.js'

/**
 * The tool-result pruner, made note-aware.
 *
 * The shipped pruner trims every over-budget tool result whenever compaction
 * pressure qualifies. It cannot know which results the investigation has
 * already written down, so a finding can be replaced by a head, a marker and a
 * `(saved to qN.tsv)` tail before it was ever recorded.
 *
 * This subclass protects two things. The `sql` results that arrived since the
 * last note stay whole — bounded by `noteEvery` thanks to the note gate, so
 * pruning becomes strictly "after the finding was written down". And every
 * notebook read stays whole, because the checkpoint template keeps conclusions
 * and leaves the evidence in the notebook.
 *
 * The window counts `sql` results only, because the ledger does: `recordUnnoted`
 * fires for `sql` and nothing else (`notebook.js`). Taking the last N results
 * of *any* kind was the same set only while every `take_note` cleared the debt,
 * so that a note's own result could never sit inside the window. A contentless
 * `take_note` is a read that deliberately leaves the debt standing, and counting
 * its result would hand it a protection slot taken from the unnoted `sql` result
 * the window exists to keep whole.
 *
 * It relies on one property of the base class: `pruneSession` passes each
 * result's own `content` array to `pruneContent`, so identity is enough to
 * recognize a protected result without reimplementing the shadow-price
 * protocol, the replacement bookkeeping, or the surface rewrite.
 *
 * The protected set is module state rather than an instance field because a
 * mounted service reaches its own methods through a proxy, and a `#private`
 * field written through one throws. One pass is synchronous from `pruneSession`
 * through every `pruneContent` call, so a module-level set is exactly as scoped.
 */
let protectedContent = new WeakSet()

export class NoteAwarePruner extends ToolResultPruner {
  pruneSession(session) {
    protectedContent = new WeakSet()
    const unnoted = unnotedCount(session.id)
    const { sql, notes } = resultsByTool(session)
    // The unnoted window, over `sql` results only — see above.
    for (const event of unnoted > 0 ? sql.slice(-unnoted) : []) {
      protect(event)
    }
    // Every notebook read, whatever the debt. The checkpoint template keeps
    // conclusions and leaves the evidence here, so a read is the only route back
    // to a statement a submission has to cite. Cut to a 200-char head and a
    // 120-char tail it would still look like a notebook, with no indication of
    // which notes went missing — a stale full print is merely wasteful, but a
    // silently partial one gets cited.
    for (const event of notes) {
      protect(event)
    }
    const result = super.pruneSession(session)
    trace('pruneSession', { unnoted, notes: notes.length, pruned: result.pruned.length, charsRemoved: result.charsRemoved })
    return result
  }

  pruneContent(blocks) {
    if (protectedContent.has(blocks)) return null
    return super.pruneContent(blocks)
  }
}

function protect(event) {
  const block = event.data.message.content[0]
  if (block?.content !== undefined) protectedContent.add(block.content)
}

/**
 * The surface's tool results, split by which tool answered.
 *
 * A result names only the call it answers, so the `tool/call` events are what
 * say which tool that was.
 */
function resultsByTool(session) {
  const names = new Map()
  const results = []
  for (const seq of [...session.surface.nodes]) {
    const event = session.eventAt(seq)
    if (event?.type === 'tool/call') {
      names.set(event.data?.callId, event.data?.name)
    } else if (event?.type === 'tool/result') {
      results.push(event)
    }
  }
  const toolOf = event => names.get(event.data.message.content[0]?.toolCallId)
  return {
    sql: results.filter(event => toolOf(event) === SQL_TOOL),
    notes: results.filter(event => toolOf(event) === NOTE_TOOL),
  }
}

export default NoteAwarePruner
