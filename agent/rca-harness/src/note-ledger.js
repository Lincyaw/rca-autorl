/**
 * How many tool results have arrived since each session's last note.
 *
 * The note gate and the note-aware pruner are separate rows in the tree, so
 * they cannot share the plugin's episode state; a Cordis service would work but
 * puts loader ordering between two halves of one mechanism. They are one
 * package in one process instead, keyed by session id, which is what both
 * sides already have — an `exec.agent.id` on one, a `session.id` on the other.
 */
const unnoted = new Map()

/** Count one result the current note has not covered yet. */
export function recordUnnoted(sessionId) {
  unnoted.set(sessionId, (unnoted.get(sessionId) ?? 0) + 1)
}

/** A note landed: everything before it is written down. */
export function clearUnnoted(sessionId) {
  unnoted.set(sessionId, 0)
}

/** How many of the newest results the pruner must leave whole. */
export function unnotedCount(sessionId) {
  return unnoted.get(sessionId) ?? 0
}
