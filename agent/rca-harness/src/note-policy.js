import { createUserMessage } from '@deepseek-ai/dsh-llm'
import { clearUnnoted, recordUnnoted, unnotedCount } from './note-ledger.js'
import { NOTE_TOOL } from './notebook.js'
import { SQL_TOOL } from './sql-tool.js'

const PLUGIN_SOURCE = { kind: 'plugin', plugin: 'rca-note-policy' }

/**
 * Make the notebook happen.
 *
 * The notebook only pays off if it is written, and an episode with an advisory
 * instruction wrote one note in 84 tool calls: with nothing lost yet, a note is
 * a turn that does not advance the investigation, so the model defers it. By
 * the time something IS lost the pruner has already replaced those results with
 * a head, a marker, and a `(saved to qN.tsv)` tail the model cannot read back.
 *
 * Two pressures, in the order that costs least:
 *
 * - **A reminder rides the result.** Past `noteEvery` unnoted queries, the sql
 *   result carries a notice telling the model to write one. It costs no turn
 *   and the model can still finish the thought it is in the middle of. This is
 *   the mechanism that should do the work.
 * - **A denial backs it up.** Past `noteLimit` the next `sql` is refused. This
 *   is not there to nag: the note-aware pruner leaves every unnoted result
 *   whole, so an episode that never notes has an unprunable tail, and pressure
 *   it cannot relieve. The limit is what bounds that tail.
 *
 * A denial cost 21 wasted turns in one measured episode when it was the only
 * pressure and the model answered it by rephrasing the query, which is why the
 * reason says outright that rephrasing fails and the persona states the rule up
 * front. `submit_result` is never gated: an episode ready to answer must be
 * able to.
 */
export function registerNotePolicy(ctx, state, noteEvery, noteLimit) {
  ctx.on('tools/pre-execute', async (exec, next) => {
    const key = state.key(exec)

    if (exec.name === NOTE_TOOL) {
      const decision = await next()
      if (decision.kind !== 'deny') clearUnnoted(key)
      return decision
    }

    if (exec.name !== SQL_TOOL) return next()

    const used = unnotedCount(key)
    if (used >= noteLimit) {
      return {
        kind: 'deny',
        reason:
          `${used} queries have run since your last note, so \`${SQL_TOOL}\` is closed until one `
          + `lands. Rephrasing this query will be denied too. Call \`${NOTE_TOOL}\` now with what `
          + 'those queries established — the statement, what its result showed, and what it '
          + 'implies — and querying reopens immediately. Older results are compacted to a file '
          + 'reference you cannot read back, so an unwritten finding is lost.',
      }
    }

    const decision = await next()
    if (decision.kind !== 'deny') recordUnnoted(key)
    return decision
  })

  // Observe and enrich, never veto: delegate first so a later listener can
  // still block or replace, then fold the notice onto whatever came back.
  ctx.on('tools/post-execute', async (exec, _result, next) => {
    const downstream = await next()
    if (exec.name !== SQL_TOOL) return downstream

    const used = unnotedCount(state.key(exec))
    if (used < noteEvery) return downstream

    const notice = createUserMessage({
      content: [{
        type: 'text',
        text:
          `Note reminder: ${used} queries have run since your last note.\n`
          + `Call \`${NOTE_TOOL}\` with what they established before the next query. Older `
          + 'results are compacted away and only the notebook survives, so an unwritten '
          + `finding is lost. After ${noteLimit} unnoted queries \`${SQL_TOOL}\` is refused `
          + 'until a note lands.',
      }],
      source: { ...PLUGIN_SOURCE, form: 'notice', summary: `${used} queries unnoted` },
    })
    return {
      ...downstream,
      additionalContexts: [notice, ...(downstream.additionalContexts ?? [])],
    }
  })
}
