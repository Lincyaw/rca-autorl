import { readFileSync } from 'node:fs'
import z from '@deepseek-ai/schemastery'
import { isAgentLoopRequest } from '@deepseek-ai/dsh-llm'
import { registerNotebook, registerNotePolicy, seedUnnoted } from './notebook.js'
import { trace } from './debug.js'
import { registerSqlTool } from './sql-tool.js'
import { registerSubmitResult } from './submit-result.js'

/** Stable Cordis plugin name. */
export const name = 'rca-harness'

/** Services this row waits for; `sdk-minimal` provides `tools`. */
export const inject = ['tools', 'llm']

const positiveInteger = z.number().step(1).min(1).required()

/** Validated by the loader before `apply`; the bundle's own patch sets every limit. */
export const Config = z.object({
  snapshot: z.string().required(),
  resultRoot: z.string().required(),
  maxRows: positiveInteger,
  maxChars: positiveInteger,
  maxCellChars: positiveInteger,
  noteEvery: positiveInteger,
  noteLimit: positiveInteger,
  // The trainer's temperature as the environment carries it, and the fork
  // prefix file; both optional.
  temperature: z.string(),
  fork: z.string(),
})

/**
 * The RCA harness: the episode's whole action space. `sdk-minimal`'s shell and
 * editor rows are disabled by this bundle's patch, so the model sees exactly
 * three tools — `sql` to read the snapshot, `take_note` to record findings in a
 * persistent notebook, and `submit_result` to answer.
 */
export function apply(ctx, config) {
  const { snapshot, resultRoot, maxRows, maxChars, maxCellChars, noteEvery, noteLimit } = config
  const limits = { maxRows, maxChars, maxCellChars }
  if (noteLimit < noteEvery) {
    throw new Error(`rca-harness: noteLimit (${noteLimit}) must not be below noteEvery (${noteEvery})`)
  }
  // A forked episode starts with the parent's history, notebook and note debt
  // (spec §4); the temperature is the trainer's.
  const fork = readFork(config.fork)
  seedUnnoted(fork.unnoted)
  const temperature = parseTemperature(config.temperature)

  // Episode state, keyed by the calling agent so one runtime can serve several
  // sessions. Everything here lives and dies with this plugin's fiber.
  const state = {
    snapshot,
    resultRoot,
    submitted: new Set(),
    notes: fork.notes,
    key: episodeKey,
  }

  trace('apply', { snapshot, resultRoot, ...limits, noteEvery, noteLimit, temperature, fork: fork.messages.length })
  registerSampling(ctx, temperature, fork.messages)
  registerSqlTool(ctx, state, limits)
  registerNotebook(ctx, state)
  registerNotePolicy(ctx, state, noteEvery, noteLimit)
  registerSubmitResult(ctx, state)
}

/**
 * Two things the trainer decides about a request that `AgentOptions` cannot
 * carry: the sampling temperature, and the history a forked episode continues
 * from (spec §4).
 *
 * The temperature goes through the `agent/request` waterfall, which yields the
 * call config before the runtime prepares the call, so the value is in the
 * config from the start and every later equality check holds.
 *
 * The history cannot: that waterfall may not touch messages. The `llm/stream`
 * waterfall sees them and may short-circuit, so a forked episode answers the
 * loop's own request with a second call on the same route, the parent's history
 * spliced in after the child's incident prompt. Only the loop's request is
 * rewritten; the inner call, and every other call, passes through.
 */
function registerSampling(ctx, temperature, prefix) {
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

/**
 * Which episode a tool call belongs to.
 *
 * The agent's id is the session id, and it names both the in-memory state and
 * the directory of saved results. `exec.agent` is optional in the tool
 * contract, though, and a group of concurrent episodes shares one Harness home:
 * a constant fallback would have them writing over each other's `qN.tsv` and
 * `notes.md`. So a call that arrives without an agent is keyed to this process
 * instead of to a name every process would agree on.
 */
function episodeKey(exec) {
  const agent = exec.agent?.id
  if (agent !== undefined) return agent
  trace('key:no-agent', { tool: exec.name })
  return `detached-${process.pid}`
}

function parseTemperature(value) {
  if (value === undefined || value === null || value === '') return undefined
  const parsed = Number(value)
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new Error(`rca-harness: bad temperature ${value}`)
  }
  return parsed
}

/** The prefix file `autorl.fork.fork_prefix` writes, or an empty one. */
function readFork(path) {
  if (path === undefined || path === null || String(path).trim() === '') {
    return { messages: [], notes: [], unnoted: 0 }
  }
  const parsed = JSON.parse(readFileSync(String(path), 'utf8'))
  if (!Array.isArray(parsed.messages)) throw new Error('rca-harness: fork prefix has no messages')
  return {
    messages: parsed.messages,
    notes: Array.isArray(parsed.notes)
      ? parsed.notes.map(note => ({ id: String(note.id ?? ''), content: String(note.content ?? '') }))
      : [],
    unnoted: Number.isInteger(parsed.unnoted) && parsed.unnoted > 0 ? parsed.unnoted : 0,
  }
}
