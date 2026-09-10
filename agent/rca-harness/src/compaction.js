import BasicCompactionEngine from '@deepseek-ai/dsh-compaction-basic'
import { checkpointRetryHint, incidentPreamble, notebookOutlinePreamble, RCA_INSTRUCTION } from './prompts.js'
import { BlockAssembler, createUserMessage } from '@deepseek-ai/dsh-llm'
import { notebookOutline } from './notebook.js'
import { trace } from './debug.js'

/**
 * `compaction-basic` with the RCA checkpoint template.
 *
 * `summarize` is the backend's documented sole hook ("Override this sole hook
 * for a template or remote summarizer"), so region selection, token accounting,
 * checkpoint landing, and the `<compacted-summary>` framing all stay with the
 * shipped engine — only the instruction, the one-shot call, and the pinned
 * incident are ours. The call keeps the conversation's own system prompt,
 * tools, and message prefix in front of the instruction, which is what lets the
 * provider reuse its warm prefix cache instead of re-reading the whole episode.
 *
 * The incident is pinned rather than summarized. Compaction is head-anchored,
 * so the episode's own question is inside the first region it replaces, and a
 * checkpoint that carried it would be carrying the model's paraphrase of the
 * task — which drifts, pass after pass, in the one piece of text that must not.
 * So the question is captured the first time it is about to be shadowed and
 * prepended verbatim to every checkpoint, and the template asks for the state
 * of the investigation only.
 */
export class RcaCompactionEngine extends BasicCompactionEngine {
  constructor(ctx, config) {
    super(ctx, config)
    // Registered after the engine's own pre-step listener, which catches a
    // failed pass and warns. That is where an episode used to continue with a
    // prompt too large to serve, reaching the endpoint as a bare 400 several
    // steps later. A compaction that has already been retried and still failed
    // ends the turn here instead, as `blocked`, which is a reason the trainer
    // can read.
    ctx.on('agent/pre-step', async ({ agent }, next) => {
      if (passes.get(agent.session.id) !== 'failed') return next()
      return { kind: 'reject' }
    })
  }

  /**
   * One retry, then the episode stops.
   *
   * Every failure mode lands here: a checkpoint the engine refuses for not
   * being smaller than the region, one the cap truncated, an endpoint error. A
   * failed pass leaves the surface as it was, so a second attempt is safe, and
   * the retry is told why the first was rejected rather than repeating it.
   */
  async compactIfNeeded(agent, trigger, signal) {
    const sessionId = agent.session.id
    for (let attempt = 1; attempt <= ATTEMPTS; attempt += 1) {
      try {
        const result = await super.compactIfNeeded(agent, trigger, signal)
        passes.delete(sessionId)
        return result
      } catch (error) {
        const message = String(error?.message ?? error)
        if (attempt === ATTEMPTS) {
          trace('compact:exhausted', { trigger, error: message })
          passes.set(sessionId, 'failed')
          throw error
        }
        trace('compact:failed', { trigger, error: message })
        // Only a checkpoint that was too big earns the corrective line: an
        // endpoint error is not the model's doing, and telling it to write less
        // would cut a checkpoint that was never the problem.
        const oversized = error?.code === 'MAX_TOKENS' || message.includes('not smaller')
        passes.set(sessionId, oversized ? 'retry-shorter' : 'retry')
      }
    }
  }

  async summarize(input, agent, signal) {
    trace('summarize:start', { messages: input.messages.length })
    const target = summarizationTarget(agent)
    const assembler = new BlockAssembler()
    const options = {
      provider: target.provider,
      model: target.model,
      messages: [
        ...input.messages,
        createUserMessage({
          content: [{
            type: 'text',
            text: passes.get(agent.session.id) === 'retry-shorter'
              ? RCA_INSTRUCTION + checkpointRetryHint()
              : RCA_INSTRUCTION,
          }],
          source: { kind: 'plugin', plugin: 'rca-compaction' },
        }),
      ],
      ...(input.system === undefined ? {} : { system: input.system }),
      ...(input.tools === undefined ? {} : { tools: [...input.tools] }),
      maxTokens: this.config.maxTokens,
      // A template fill is not a reasoning task, and reasoning is what fills
      // the output budget: the checkpoint itself is about a page. Only where
      // the route declares the effort — asking for one it does not declare is
      // a hard error, and the rollout route need not declare any.
      ...await lowReasoning(this.ctx, target, signal),
      sessionId: agent.session.id,
      purpose: 'compaction',
      ...(signal === undefined ? {} : { signal }),
    }

    for await (const chunk of this.ctx.llm.stream(options)) assembler.push(chunk)
    const finish = assembler.finish
    if (finish.kind === 'error' || finish.kind === 'aborted') {
      const error = new Error(finish.failure.message)
      error.code = finish.failure.code
      throw error
    }
    if (finish.kind === 'max-tokens') {
      const error = new Error('RCA checkpoint truncated at the token cap (incomplete checkpoint)')
      error.code = 'MAX_TOKENS'
      throw error
    }

    const rawOutput = assembler.blocks()
    if (rawOutput.some(block => block.type === 'image')) {
      throw new Error('compaction summary cannot contain image output')
    }
    const summary = rawOutput.filter(block => block.type === 'text')
    if (!summary.some(block => block.text.trim().length > 0)) {
      throw new Error('RCA checkpoint produced no text content')
    }
    return {
      summary: [...pinnedIncident(agent, input), ...pinnedOutline(agent), ...summary],
      rawOutput,
      llmStreamCall: true,
      provider: options.provider,
      model: options.model,
      maxTokens: this.config.maxTokens,
      ...(assembler.usage === undefined ? {} : { usage: assembler.usage }),
    }
  }
}

/** `{ reasoningEffort: 'low' }` where the routed model offers it, otherwise nothing. */
async function lowReasoning(ctx, target, signal) {
  try {
    const info = await ctx.llm.resolveModelInfo(target.provider, target.model, signal)
    const efforts = info.reasoning?.efforts ?? []
    return efforts.some(effort => effort.id === 'low') ? { reasoningEffort: 'low' } : {}
  } catch (error) {
    trace('summarize:effort-unresolved', { error: String(error?.message ?? error) })
    return {}
  }
}

/** How many note names a checkpoint lists before it starts counting the rest. */
const OUTLINE_LIMIT = 30

/**
 * The notebook's index, deterministic, beside the incident.
 *
 * The checkpoint cites the notes its own claims rest on; every other note would
 * be invisible to the model resuming from it, so the index rides along whole.
 * It is the one part of a checkpoint that grows with the episode, which is why
 * it is a list of names with a cap rather than anything the model writes.
 */
function pinnedOutline(agent) {
  const outline = notebookOutline(agent.session.id, OUTLINE_LIMIT)
  if (outline === undefined) return []
  const body = outline.names.map(name => `- ${name}`).join('\n')
  return [{ type: 'text', text: notebookOutlinePreamble(body, outline.older) }]
}

/** How many times one pressure point may be compacted before the episode stops. */
const ATTEMPTS = 2

/** Where each session's compaction stands: retrying, or given up. */
const passes = new Map()

/**
 * The episode's own question, verbatim, as the head of every checkpoint.
 *
 * Captured from the first region that contains it — a user-sourced message,
 * which is what tells it apart from a note reminder or a prior checkpoint, both
 * of which are plugin-sourced — and kept per session, since the next region
 * starts after it has already been replaced.
 */
const incidents = new Map()

function pinnedIncident(agent, input) {
  const sessionId = agent.session.id
  if (!incidents.has(sessionId)) {
    const asked = input.messages.find(
      message => message.role === 'user' && message.source?.kind === 'user',
    )
    const text = (asked?.content ?? [])
      .filter(block => block.type === 'text')
      .map(block => block.text)
      .join('\n')
      .trim()
    if (text.length === 0) return []
    incidents.set(sessionId, text)
  }
  return [{ type: 'text', text: incidentPreamble(incidents.get(sessionId)) }]
}

/** The checkpoint is written on the conversation's own route. */
function summarizationTarget(agent) {
  const routed = agent.session.requestHeader()?.config
  if (routed === undefined) throw new Error('no routed request to write the RCA checkpoint on')
  return routed
}

export default RcaCompactionEngine
