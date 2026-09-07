import { defineTool } from '@deepseek-ai/dsh-tools'
import { DuckDBInstance } from '@duckdb/node-api'
import { mkdirSync, readdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'

/** Model-facing name of the one evidence tool. */
export const SQL_TOOL = 'sql'

/**
 * One DuckDB connection per episode, created on the first query.
 *
 * Every `*.parquet` in the snapshot is materialized as a table named after the
 * file stem. The JSON and text files beside them carry the injected fault, the
 * ground-truth causal graph, and the label, so they are never registered — and
 * `enable_external_access` is switched off once the tables exist, which turns
 * DuckDB's own readers (`read_parquet`, `read_json_auto`, `read_csv`) into
 * permission errors. Without that setting the query language is itself a path
 * out of the snapshot and straight onto the answer.
 */
async function openSnapshot(snapshot) {
  const tables = readdirSync(snapshot)
    .filter(entry => entry.endsWith('.parquet'))
    .map(entry => entry.slice(0, -'.parquet'.length))
    .sort()
  if (tables.length === 0) throw new Error(`no .parquet evidence in the snapshot ${snapshot}`)

  const instance = await DuckDBInstance.create(':memory:')
  const connection = await instance.connect()
  for (const table of tables) {
    await connection.run(`CREATE TABLE "${table}" AS SELECT * FROM read_parquet(?)`,
      [join(snapshot, `${table}.parquet`)])
  }
  await connection.run('SET enable_external_access=false')
  return { connection, tables }
}

/** JSON-safe projection of one DuckDB value; the canonical result must be lossless JSON. */
function jsonValue(value) {
  if (value === null || value === undefined) return null
  switch (typeof value) {
    case 'string':
    case 'boolean':
      return value
    case 'number':
      return Number.isFinite(value) ? value : null
    case 'bigint':
      return Number.isSafeInteger(Number(value)) ? Number(value) : value.toString()
    default:
      return typeof value.toString === 'function' ? value.toString() : String(value)
  }
}

/** How a cell reaches the model: NULL is a value, not a blank, and one log line is not a page. */
function cell(value, maxCellChars) {
  if (value === null) return 'NULL'
  const text = String(value)
  return text.length > maxCellChars ? `${text.slice(0, maxCellChars)}…(+${text.length - maxCellChars})` : text
}

/**
 * The model-facing view: one status line, then TSV. TSV rather than JSON because
 * the same table costs a fraction of the tokens, and the status line is what
 * makes a capped result actionable — it says how much was matched, which slice
 * came back, and the two ways forward (page on, or ask a narrower question).
 */
function renderTable(value) {
  if (value.matched_rows === 0) return '(0 rows)'
  const first = value.offset + 1
  const last = value.offset + value.rows.length
  const complete = value.offset === 0 && last === value.matched_rows
  const status = complete
    ? `${value.matched_rows} rows`
    : `rows ${first}-${last} of ${value.matched_rows}`
    + (last < value.matched_rows
      ? `; for the next page repeat this statement with offset: ${last}, or aggregate/filter for a sharper answer`
      : '')
  return [status, value.columns.join('\t'), ...value.rows.map(row => row.join('\t'))].join('\n')
}

/**
 * Save the full result to a TSV file so it survives context compaction.
 * Returns the file path relative to the results directory.
 */
function saveResultFile(state, value, statement) {
  const key = state.key({ agent: { id: 'root' } })
  if (!state.resultDirs) state.resultDirs = new Map()
  if (!state.resultCounters) state.resultCounters = new Map()

  let dir = state.resultDirs.get(key)
  if (!dir) {
    dir = join(state.snapshot, '.sql_results')
    mkdirSync(dir, { recursive: true })
    state.resultDirs.set(key, dir)
  }

  const n = (state.resultCounters.get(key) ?? 0) + 1
  state.resultCounters.set(key, n)
  const filename = `q${n}.tsv`
  const filepath = join(dir, filename)

  const header = value.columns.join('\t')
  const rows = value.rows.map(row => row.join('\t'))
  const meta = `-- Statement: ${statement}\n-- ${value.matched_rows} rows matched, offset ${value.offset}\n`
  writeFileSync(filepath, meta + header + '\n' + rows.join('\n') + '\n', 'utf-8')

  return { filename, filepath, queryNum: n }
}

/**
 * Register the `sql` tool. It is the episode's whole evidence surface: schema
 * discovery is `SHOW TABLES` and `DESCRIBE <table>`, so no second tool is
 * needed, and nothing outside the snapshot is reachable through it.
 *
 * Each result is also saved to a file so that when `take_note` compacts older
 * sql results out of context, the model can still reference what it queried.
 */
export function registerSqlTool(ctx, state, limits) {
  const { maxRows, maxChars, maxCellChars } = limits
  if (!state.pendingResults) state.pendingResults = []

  ctx.tools.register(defineTool({
    name: SQL_TOOL,
    description:
      'Query the incident snapshot with DuckDB SQL. Each telemetry file is a table named after '
      + 'the file without its extension (for example `abnormal_logs`, `normal_metrics`, '
      + '`abnormal_traces`). Start with `SHOW TABLES`, then `DESCRIBE <table>` for its columns. '
      + `A result is capped at ${maxRows} rows and about ${maxChars} characters; the status line `
      + 'reports how many rows matched, and `offset` pages through them. Prefer aggregating over '
      + 'paging. After reviewing the result, call `take_note` to record your finding — this also '
      + 'compacts the sql result out of context, keeping only your note and a file reference.',
    parameters: {
      statement: { type: 'string', required: true, description: 'One DuckDB SQL statement.' },
      offset: {
        type: 'integer',
        description: 'Skip this many matched rows before returning; for paging a capped result.',
      },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          columns: { type: 'array', required: true, items: { type: 'string' } },
          rows: { type: 'json', required: true },
          offset: { type: 'integer', required: true },
          matched_rows: { type: 'integer', required: true },
          truncated: { type: 'boolean', required: true },
          result_file: { type: 'string' },
        },
      },
      render: (_args, value) => [{
        type: 'text',
        text: renderTable(value) + (value.result_file ? `\n(saved to ${value.result_file})` : ''),
      }],
    },
    async execute(args, exec) {
      const statement = args.statement.trim()
      if (statement.length === 0) throw new Error('`statement` must be a non-empty SQL statement')
      const offset = args.offset ?? 0
      if (offset < 0) throw new Error('`offset` must not be negative')

      let session = state.snapshots.get(state.snapshot)
      if (session === undefined) {
        session = await openSnapshot(state.snapshot)
        state.snapshots.set(state.snapshot, session)
      }

      const reader = await session.connection.runAndReadAll(statement)
      const columns = reader.columnNames()
      const matched = reader.getRows()

      const rows = []
      let chars = columns.join('\t').length
      for (const row of matched.slice(offset)) {
        if (rows.length >= maxRows) break
        const projected = row.map(value => cell(jsonValue(value), maxCellChars))
        const width = projected.reduce((total, text) => total + text.length + 1, 0)
        if (rows.length > 0 && chars + width > maxChars) break
        chars += width
        rows.push(projected)
      }

      const result = {
        columns,
        rows,
        offset,
        matched_rows: matched.length,
        truncated: offset + rows.length < matched.length,
      }

      // Save to file and track for compaction by take_note.
      if (matched.length > 0) {
        const { filename } = saveResultFile(state, result, statement)
        result.result_file = filename
        state.pendingResults.push({
          statement,
          filename,
          matched_rows: matched.length,
          columns: columns.length,
        })
      }

      return result
    },
  }))
}
