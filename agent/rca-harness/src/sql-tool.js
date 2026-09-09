import { DuckDBInstance } from '@duckdb/node-api'
import { defineTool } from '@deepseek-ai/dsh-tools'
import { mkdirSync, readdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { nextPageHint, SQL_OFFSET_DESCRIPTION, SQL_STATEMENT_DESCRIPTION, SQL_TOOL, sqlDescription } from './prompts.js'


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
      ? nextPageHint(last)
      : '')
  return [status, value.columns.join('\t'), ...value.rows.map(row => row.join('\t'))].join('\n')
}

/**
 * Save the page to a TSV file so it survives context compaction. Returns the
 * file name, which is what the model-facing tail cites.
 *
 * The file goes under `resultRoot`, never under the snapshot. The snapshot is
 * the shared dataset directory a training run replays for every rollout of that
 * case: writing there mutates the dataset, breaks the read-only property the
 * whole tool roster exists to guarantee, and — with a group of 8 episodes on
 * one case — has every episode overwrite the others' `q1.tsv`. Keyed by the
 * calling agent so concurrent sessions in one runtime keep separate counters
 * and separate directories.
 */
function saveResultFile(counters, resultRoot, key, value, statement) {
  const dir = join(resultRoot, key)
  mkdirSync(dir, { recursive: true })
  const n = (counters.get(key) ?? 0) + 1
  counters.set(key, n)
  const filename = `q${n}.tsv`
  const header = value.columns.join('\t')
  const rows = value.rows.map(row => row.join('\t'))
  const meta = `-- Statement: ${statement}\n-- ${value.matched_rows} rows matched, offset ${value.offset}\n`
  writeFileSync(join(dir, filename), meta + header + '\n' + rows.join('\n') + '\n', 'utf-8')
  return filename
}

/** How many rows a statement matches, when the page did not reach its end. */
async function countRows(connection, statement) {
  const reader = await connection.runAndReadAll(`SELECT count(*) FROM (\n${statement.replace(/;\s*$/, '')}\n)`)
  return Number(reader.getRows()[0][0])
}

/**
 * Register the `sql` tool. It is the episode's whole evidence surface: schema
 * discovery is `SHOW TABLES` and `DESCRIBE <table>`, so no second tool is
 * needed, and nothing outside the snapshot is reachable through it.
 *
 * Each page is also saved to a file, so that once the pruner has folded the
 * result away the model can still cite what it queried.
 */
export function registerSqlTool(ctx, state, limits) {
  const { maxRows, maxChars, maxCellChars } = limits
  let session
  const counters = new Map()

  ctx.tools.register(defineTool({
    name: SQL_TOOL,
    description: sqlDescription(maxRows, maxChars),
    parameters: {
      statement: { type: 'string', required: true, description: SQL_STATEMENT_DESCRIPTION },
      offset: {
        type: 'integer',
        description: SQL_OFFSET_DESCRIPTION,
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

      session ??= await openSnapshot(state.snapshot)

      // Only the page is pulled into the process; a broad query over a large
      // table would otherwise materialize every row to show a fraction of them.
      const reader = await session.connection.runAndReadUntil(statement, offset + maxRows)
      const columns = reader.columnNames()
      const matchedRows = reader.done ? reader.currentRowCount : await countRows(session.connection, statement)

      const rows = []
      let chars = columns.join('\t').length
      for (const row of reader.getRows().slice(offset)) {
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
        matched_rows: matchedRows,
        truncated: offset + rows.length < matchedRows,
      }
      if (matchedRows > 0) {
        result.result_file = saveResultFile(counters, state.resultRoot, state.key(exec), result, statement)
      }
      return result
    },
  }))
}
