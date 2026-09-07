import { appendFileSync } from 'node:fs'

/**
 * Append-only trace for the bundle's own decisions.
 *
 * stdout belongs to the JSON-RPC server, so a plugin that prints there corrupts
 * the protocol. This writes to the file named by `$RCA_HARNESS_LOG` and is inert
 * when the variable is unset, which keeps a training run silent while making a
 * debugging run answer "did this row even get called" without a bisect.
 */
const target = process.env.RCA_HARNESS_LOG ?? ''

export const debugEnabled = target.length > 0

export function trace(event, data = {}) {
  if (!debugEnabled) return
  try {
    appendFileSync(target, `${JSON.stringify({ at: new Date().toISOString(), event, ...data })}\n`)
  } catch {
    // Observability must never break the episode it observes.
  }
}
