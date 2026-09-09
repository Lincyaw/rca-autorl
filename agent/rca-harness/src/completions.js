import { appendFileSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'
import { trace } from './debug.js'

/** Stable Cordis plugin name. */
export const name = 'rca-completions'

/** Services this row waits for. */
export const inject = ['llm']

/**
 * Record the provider's response id for every LLM request of an episode.
 *
 * RL reward is per-completion: AReaL's proxy mints `chatcmpl-<uuid>`, returns
 * it as the response `id`, and keys its interaction cache by it, so
 * `set_reward(interaction_id, ...)` — and therefore any credit finer than "the
 * whole trajectory" — needs that id. Nothing in the session log carries it: the
 * adapter normalizes the provider stream into `StreamChunk`s and the id is not
 * one of their fields.
 *
 * It is not lost, though. `finish` carries `replayState`, whose `response` is
 * documented as "response-level adapter-private metadata (ids, native stop
 * reason)", and `llm-pi-ai` puts the provider's id there as `responseId`. That
 * crosses the `llm/stream` seam, so a waterfall listener can read it without an
 * adapter of our own and without touching the harness.
 *
 * The listener is a pass-through: it yields every chunk unchanged and only
 * observes the terminal one. Each line of the sidecar is one request in the
 * order the episode made them, which is what lets the trainer map a reward back
 * to the turn that earned it. `purpose` distinguishes the agent's own steps
 * from the compaction engine's summarizer call, which is also a completion the
 * proxy caches and which no policy should be credited for.
 */
export function apply(ctx, config = {}) {
  const root = String(config.root ?? '').trim()
  if (root.length === 0) throw new Error('rca-completions: root must be a non-empty path')
  mkdirSync(root, { recursive: true })
  // Per session, not per process: one runtime can serve several sessions, and
  // an ordinal shared across them would leave gaps in each one's file — which
  // is exactly the ordering the trainer reads to place a reward on a turn.
  const ordinals = new Map()
  // A route that delegates to another (the rca-sampling row) streams the same
  // completion through this seam twice, once per layer; the trainer counts
  // one request per agent step, so a response id is written once.
  const seen = new Map()

  ctx.on('llm/stream', (options, next) => record(options, next()), { global: true })

  async function* record(options, upstream) {
    const session = String(options.sessionId ?? 'unknown')
    const ordinal = ordinals.get(session) ?? 0
    ordinals.set(session, ordinal + 1)
    for await (const chunk of upstream) {
      if (chunk.type === 'finish') {
        const responseId = chunk.replayState?.response?.responseId
        const ids = seen.get(session) ?? new Set()
        seen.set(session, ids)
        if (responseId && ids.has(responseId)) {
          yield chunk
          continue
        }
        if (responseId) ids.add(responseId)
        const line = {
          ordinal,
          responseId: responseId ?? null,
          purpose: options.purpose ?? 'agent',
          provider: options.provider ?? null,
          model: options.model ?? null,
          reason: chunk.reason,
        }
        try {
          appendFileSync(join(root, `${session}.jsonl`), `${JSON.stringify(line)}\n`)
        } catch (error) {
          // A missing id costs credit resolution, never the episode.
          trace('completions:error', { message: String(error?.message ?? error) })
        }
        trace('completions', line)
      }
      yield chunk
    }
  }
}

export default { name, inject, apply }
