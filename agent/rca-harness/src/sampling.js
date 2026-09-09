import { LlmAdapter } from '@deepseek-ai/dsh-llm'
import { trace } from './debug.js'

/** Stable Cordis plugin name. */
export const name = 'rca-sampling'

/** Services this row waits for. */
export const inject = ['llm']

/**
 * A provider route that is another route plus a temperature.
 *
 * The harness builds each model call from `AgentOptions`, which carries the
 * provider, model, reasoning effort and output cap and nothing about sampling,
 * so a request leaves without a temperature and the serving side fills in its
 * own default. The adapters do forward `options.temperature` when it is set;
 * what is missing is a way to set it. The `llm/stream` middleware cannot,
 * because the runtime dispatches the options it prepared and refuses a
 * prepared call whose config changed.
 *
 * An adapter can. This row registers `config.provider` as a route whose
 * adapter prepares the same call on `config.inner`, with `config.temperature`
 * in the call config from the start, and streams through it. The runtime's
 * equality check between the prepared config and the dispatched options holds
 * because both carry the temperature. With no temperature configured the route
 * is a plain alias and the request is what it would have been.
 *
 * The trainer owns the number: it arrives as `RCA_TEMPERATURE` from the same
 * `gconfig` the rollout is trained under, and a greedy baseline is the same
 * route at zero.
 */
export function apply(ctx, config = {}) {
  const provider = String(config.provider ?? '').trim()
  const inner = String(config.inner ?? '').trim()
  if (provider.length === 0 || inner.length === 0) {
    throw new Error('rca-sampling: provider and inner are required')
  }
  const temperature = parseTemperature(config.temperature)
  const sampling = temperature === undefined ? {} : { temperature }
  const llm = ctx.llm

  /** The history names this route; the inner adapter must see its own. */
  const relabel = messages => messages.map(message =>
    message.source?.provider === provider
      ? { ...message, source: { ...message.source, provider: inner } }
      : message)

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
    trace('sampling', { provider, inner, ...sampling })
    yield* prepared.stream({ ...options, ...prepared.config, messages: relabel(options.messages) })
  }

  // Registration asks the adapter about its routes at once, before the inner
  // route necessarily exists, so nothing here touches `inner` until a call.
  class SamplingAdapter extends LlmAdapter {
    providerInfo(id) {
      return { id, name: temperature === undefined ? inner : `${inner} at temperature ${temperature}` }
    }

    async prepareCall(id, model, signal) {
      // The inner route's exact model, presented as this route's own.
      const info = await llm.resolveModelInfo(inner, model, signal)
      return { model: { ...info, provider: id }, stream: forward }
    }
  }

  ctx.llm.registerAdapter([provider], new SamplingAdapter())
  trace('sampling:register', { provider, inner, ...sampling })
}

function parseTemperature(value) {
  if (value === undefined || value === null || value === '') return undefined
  const parsed = Number(value)
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new Error(`rca-sampling: bad temperature ${value}`)
  }
  return parsed
}

export default { name, inject, apply }
