import { ToolResultPruner } from '@deepseek-ai/dsh-compaction-tool-result-pruner'
import { trace } from './debug.js'
import { unnotedCount } from './note-ledger.js'

/**
 * The tool-result pruner, made note-aware.
 *
 * The shipped pruner trims every over-budget tool result whenever compaction
 * pressure qualifies. It cannot know which results the investigation has
 * already written down, so a result can be replaced by a head, a marker, and a
 * `(saved to qN.tsv)` tail the model has no tool to read back — the finding is
 * gone before it was ever recorded.
 *
 * This subclass protects the tail of the conversation: the tool results that
 * arrived since the last note stay whole, everything older is pruned normally.
 * With the note gate in front of `sql`, that window is bounded by `noteEvery`,
 * and pruning becomes strictly "after the finding was written down".
 *
 * It relies on one property of the base class: `pruneSession` passes each
 * result's own `content` array to `pruneContent`, so identity is enough to
 * recognize a protected result without reimplementing the shadow-price
 * protocol, the replacement bookkeeping, or the surface rewrite.
 *
 * The protected set is module state rather than an instance field because a
 * mounted service reaches its own methods through a proxy, and a `#private`
 * field written through one throws `Cannot write private member`. One pass is
 * synchronous from `pruneSession` through every `pruneContent` call, so a
 * module-level set is exactly as scoped as an instance field would have been.
 */
let protectedContent = new WeakSet()

export class NoteAwarePruner extends ToolResultPruner {
  pruneSession(session) {
    protectedContent = new WeakSet()
    const unnoted = unnotedCount(session.id)
    if (unnoted > 0) {
      const results = []
      for (const seq of [...session.surface.nodes]) {
        const event = session.eventAt(seq)
        if (event?.type === 'tool/result') results.push(event)
      }
      for (const event of results.slice(-unnoted)) {
        const block = event.data.message.content[0]
        if (block?.content !== undefined) protectedContent.add(block.content)
      }
    }
    const result = super.pruneSession(session)
    trace('pruneSession', { unnoted, pruned: result.pruned.length, charsRemoved: result.charsRemoved })
    return result
  }

  pruneContent(blocks) {
    if (protectedContent.has(blocks)) return null
    return super.pruneContent(blocks)
  }
}

export default NoteAwarePruner
