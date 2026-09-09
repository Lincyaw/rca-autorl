import { isAgentLoopRequest } from '@deepseek-ai/dsh-llm'
import { trace } from './debug.js'

/**
 * Two things the trainer decides about a request that `AgentOptions` cannot
 * carry: the sampling temperature, and the history a forked episode
 * continues from (spec §4).
 *
 * The temperature goes through the `agent/request` waterfall, which yields
 * the call config before the runtime prepares the call, so the value is in
 * the config from the start and every later equality check holds.
 *
 * The history cannot: that waterfall may not touch messages. The `llm/stream`
 * waterfall sees them and may short-circuit, so a forked episode answers the
 * loop's own request with a second call on the same route, the parent's
 * history spliced in after the child's incident prompt. Only the loop's
 * request is rewritten; the inner call, and every other call, passes through.
 */
export function registerSampling(ctx, temperature, prefix) {
  if (temperature !== undefined) {
    ctx.on('agent/request', async (_payload, next) => ({ ...await next(), temperature }))
  }
  if (prefix.length === 0) return

  /** The parent's history goes right after the child's incident prompt, or first once compaction has folded that away. */
  const continued = messages => {
    const at = messages.findIndex(message => message.source?.kind === 'user') + 1
    return [...messages.slice(0, at), ...prefix, ...messages.slice(at)]
  }

  ctx.on('llm/stream', async function* (options, next) {
    if (!isAgentLoopRequest(options)) return yield* next()
    trace('fork:forward', { prefix: prefix.length })
    yield* ctx.llm.stream({ ...options, messages: continued(options.messages) })
  }, { global: true, prepend: true })
  trace('fork:register', { prefix: prefix.length })
}

export function parseTemperature(value) {
  if (value === undefined || value === null || value === '') return undefined
  const parsed = Number(value)
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new Error(`rca-harness: bad temperature ${value}`)
  }
  return parsed
}
