import { LlmAdapter } from '@deepseek-ai/dsh-llm'
import { trace } from './debug.js'

/** The route a forked episode is switched to; it delegates to the launch's own. */
const FORK_ROUTE = 'rca-fork'

/**
 * Two things the trainer decides about a request that `AgentOptions` cannot
 * carry: the sampling temperature, and the history a forked episode
 * continues from (spec §4).
 *
 * The temperature goes through the `agent/request` waterfall, which yields
 * the call config before the runtime prepares the call, so the value is in
 * the config from the start and every later equality check holds. It is
 * route-independent and costs nothing on an ordinary episode.
 *
 * The history cannot: that waterfall may not touch messages, and the
 * `llm/stream` middleware sees them but dispatches the runtime's own copy. An
 * adapter does see them. When a prefix is given, the same waterfall switches
 * the episode to `FORK_ROUTE`, whose adapter prepares the call on the route
 * the launch named and streams it with the parent's history spliced in after
 * the child's own incident prompt. Ordinary episodes never pass through it.
 */
export function registerSampling(ctx, temperature, prefix) {
  const sampling = temperature === undefined ? {} : { temperature }
  const forked = prefix.length > 0
  let inner = ''

  ctx.on('agent/request', async (_payload, next) => {
    const config = await next()
    if (forked && config.provider !== FORK_ROUTE) inner = config.provider
    return { ...config, ...sampling, ...(forked ? { provider: FORK_ROUTE } : {}) }
  })
  if (!forked) return

  const llm = ctx.llm

  /** The history names this route; the inner adapter must see its own. */
  const relabel = messages => messages.map(message =>
    message.source?.provider === FORK_ROUTE
      ? { ...message, source: { ...message.source, provider: inner } }
      : message)

  /** The parent's history goes right after the child's incident prompt. */
  const continued = messages => {
    const at = messages.findIndex(message => message.role === 'user')
    if (at < 0) throw new Error('rca-harness: no incident prompt to fork after')
    return [...messages.slice(0, at + 1), ...prefix, ...messages.slice(at + 1)]
  }

  async function* forward(options) {
    const call = {
      provider: inner,
      model: options.model,
      ...(options.reasoningEffort === undefined ? {} : { reasoningEffort: options.reasoningEffort }),
      ...(options.maxTokens === undefined ? {} : { maxTokens: options.maxTokens }),
      ...(options.stop === undefined ? {} : { stop: options.stop }),
      ...sampling,
    }
    const prepared = await llm.prepareCall(call, options.signal)
    trace('fork:forward', { inner, prefix: prefix.length })
    yield* prepared.stream({ ...options, ...prepared.config, messages: relabel(continued(options.messages)) })
  }

  // Registration asks the adapter about its routes at once, before the inner
  // route necessarily exists, so nothing here touches `inner` until a call.
  class ForkAdapter extends LlmAdapter {
    providerInfo(id) {
      return { id, name: 'forked episode' }
    }

    async prepareCall(id, model, signal) {
      // The inner route's exact model, presented as this route's own.
      const info = await llm.resolveModelInfo(inner, model, signal)
      return { model: { ...info, provider: id }, stream: forward }
    }
  }

  llm.registerAdapter([FORK_ROUTE], new ForkAdapter())
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
