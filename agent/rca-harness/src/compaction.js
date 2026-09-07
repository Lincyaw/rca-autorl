import BasicCompactionEngine from '@deepseek-ai/dsh-compaction-basic'
import { BlockAssembler, createUserMessage } from '@deepseek-ai/dsh-llm'
import { trace } from './debug.js'

/**
 * The RCA checkpoint instruction.
 *
 * `compaction-basic` ships a coding-assistant template — Files and Code, Errors
 * and Fixes, "preserve exact file paths, commands, function signatures". An RCA
 * episode has none of those, so half its sections compact to "(none)" and the
 * things that must survive have nowhere to go: which SQL already ran (so the
 * agent does not re-run it), what each result showed, the causal chain built so
 * far, and which hypotheses are still open.
 *
 * This template keeps exactly those, and keeps them in the vocabulary the
 * submission needs, so a compacted episode can still produce a graph whose
 * nodes cite the statements that grounded them.
 *
 * Verbatim SQL is scoped to the statements that ground something. An earlier
 * revision asked for every query verbatim, on the reasoning that a re-run query
 * is a wasted step; across the first ten collected episodes that meant carrying
 * 875 statements to cite 148, and the surplus grew with episode length — 217
 * queries and 9 citations on the longest. A checkpoint that grows with the span
 * it replaces is one that eventually cannot replace it: those episodes spent
 * 45% of their compaction time on summaries discarded for overrunning the token
 * cap or for not being smaller than the region they shadowed. Knowing that
 * ground was covered is what prevents the re-run; the statement text is only
 * needed by the node that cites it.
 */
const RCA_INSTRUCTION = [
  'You are now acting as a compaction engine for a root-cause analysis agent. Condense the investigation ABOVE into a structured checkpoint that lets another model resume it with no loss of evidence.',
  '',
  'Output EXACTLY the Markdown structure below: keep every section, in order. Use terse bullets, not prose paragraphs. Write "(none)" for an empty section — never drop a section.',
  '',
  '## Incident',
  '- [the reported symptom: affected endpoints or services, the alerting signal, and the abnormal window]',
  '',
  '## Snapshot Schema',
  '- [tables seen so far and the columns that mattered, so they are not re-discovered]',
  '',
  '## Evidence Statements',
  '- [only the SQL that grounds a finding or an edge below: the statement verbatim and complete, and in one clause what its result showed]',
  '',
  '## Ground Already Covered',
  '- [what else was queried and what it showed or excluded — one line each, no SQL text]',
  '',
  '## Established Findings',
  '- [what the evidence settles, each tied to the statement above that settles it]',
  '',
  '## Causal Chain So Far',
  '- [the propagation edges established so far, cause first: A -> B because <evidence>]',
  '',
  '## Open Hypotheses',
  '- [asserted but not yet grounded, and the query that would settle each]',
  '',
  '## Ruled Out',
  '- [candidates the evidence excludes, and what excluded them — so they are not revisited]',
  '',
  '## Next Step',
  '- [the single next query or the decision to submit, or "(none)"]',
  '',
  'Rules:',
  '- Preserve verbatim only the statements under Evidence Statements: a submitted node must cite the statement that grounds it, so those have to survive exactly. Every other query goes under Ground Already Covered as one line without its SQL — knowing the ground was covered is what stops a re-run, and the statement itself is not needed for that.',
  '- Preserve exact service names, metric names, table and column names, timestamps, and numeric values. These are the evidence.',
  '- Never promote a hypothesis to a finding. If the evidence did not settle it, it stays under Open Hypotheses.',
  '- Do NOT mention this summarization request or that the context was compacted.',
  '- Output only the checkpoint text: do not call any tool or take any other action.',
  '- If the conversation already contains a <compacted-summary> block, it is a PRIOR checkpoint. Do not copy it forward verbatim: preserve still-true facts, drop stale ones, and merge newer information into a single consolidated checkpoint under the same structure.',
].join('\n')

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
  async compactIfNeeded(agent, trigger, signal) {
    try {
      // The two inputs of the pressure decision, so a "nothing to do" answer is
      // attributable: what the harness believes the window is, and what it
      // believes the conversation currently costs.
      const routed = agent.session.requestHeader()?.config
      const info = routed === undefined
        ? undefined
        : await this.ctx.llm.resolveModelInfo(routed.provider, routed.model, signal)
      const measured = this.ctx.tokenMeter.measure(agent.session)
      const result = await super.compactIfNeeded(agent, trigger, signal)
      trace('compactIfNeeded', {
        trigger,
        compacted: result !== null,
        routed: routed === undefined ? null : `${routed.provider}/${routed.model}`,
        contextWindow: info?.context?.contextWindow ?? null,
        thresholdRatio: this.config.thresholdRatio,
        measured: typeof measured === 'number' ? measured : JSON.stringify(measured).slice(0, 200),
      })
      return result
    } catch (error) {
      trace('compactIfNeeded:error', { trigger, message: String(error?.message ?? error) })
      throw error
    }
  }

  async summarize(input, agent, signal) {
    trace('summarize:start', { messages: input.messages.length })
    const target = summarizationTarget(this.config, agent)
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

/** The route that writes the checkpoint: configured pair, else the conversation's own. */
function summarizationTarget(config, agent) {
  if (config.summarizationProvider && config.summarizationModel) {
    return { provider: config.summarizationProvider, model: config.summarizationModel }
  }
  const routed = agent.session.requestHeader()?.config
  if (routed !== undefined) return routed
  const options = agent.options
  if (options.provider && options.model) {
    return { provider: options.provider, model: options.model }
  }
  throw new Error(
    'no provider/model available for the RCA checkpoint: route one request or set the '
    + 'summarization provider/model on the rca-compaction row',
  )
}

export default RcaCompactionEngine
