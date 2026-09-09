import BasicCompactionEngine from '@deepseek-ai/dsh-compaction-basic'
import { RCA_INSTRUCTION } from './prompts.js'
import { BlockAssembler, createUserMessage } from '@deepseek-ai/dsh-llm'
import { trace } from './debug.js'

/**
 * `compaction-basic` with the RCA checkpoint template.
 *
 * `summarize` is the backend's documented sole hook ("Override this sole hook
 * for a template or remote summarizer"), so region selection, token accounting,
 * checkpoint landing, and the `<compacted-summary>` framing all stay with the
 * shipped engine — only the instruction and the one-shot call are ours. The
 * call keeps the conversation's own system prompt, tools, and message prefix in
 * front of the instruction, which is what lets the provider reuse its warm
 * prefix cache instead of re-reading the whole episode.
 */
export class RcaCompactionEngine extends BasicCompactionEngine {
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
          content: [{ type: 'text', text: RCA_INSTRUCTION }],
          source: { kind: 'plugin', plugin: 'rca-compaction' },
        }),
      ],
      ...(input.system === undefined ? {} : { system: input.system }),
      ...(input.tools === undefined ? {} : { tools: [...input.tools] }),
      maxTokens: this.config.maxTokens,
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
      summary,
      rawOutput,
      llmStreamCall: true,
      provider: options.provider,
      model: options.model,
      maxTokens: this.config.maxTokens,
      ...(assembler.usage === undefined ? {} : { usage: assembler.usage }),
    }
  }
}

/** The checkpoint is written on the conversation's own route. */
function summarizationTarget(agent) {
  const routed = agent.session.requestHeader()?.config
  if (routed === undefined) throw new Error('no routed request to write the RCA checkpoint on')
  return routed
}

export default RcaCompactionEngine
