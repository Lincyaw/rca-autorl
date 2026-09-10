import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { COMPACTION_THRESHOLD, CONTEXT_WINDOW, type Episode } from './api'

/**
 * The context size per step, against the two lines that bound it.
 *
 * This is the view that shows whether compaction is paying for itself. A pass
 * should drop the curve well below the threshold; a drop that lands back above
 * it means the next step fires another pass, and the episode is in the loop that
 * ends at the window.
 */
export function ContextCurve({ episode }: { episode: Episode }) {
  const points = episode.steps
    .filter(step => step.input_tokens > 0)
    .map(step => ({ step: step.index, tokens: step.input_tokens }))

  if (points.length === 0) return <p className="muted">No request reached the model.</p>

  return (
    <div>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={points} margin={{ top: 8, right: 16, bottom: 4, left: 8 }}>
          <CartesianGrid stroke="#eee" />
          <XAxis dataKey="step" tick={{ fontSize: 11 }} />
          <YAxis domain={[0, CONTEXT_WINDOW]} tick={{ fontSize: 11 }} width={52} />
          <Tooltip formatter={(v: number) => `${v.toLocaleString()} tokens`} />
          <ReferenceLine
            y={CONTEXT_WINDOW}
            stroke="#c1121f"
            label={{ value: 'window 32768', position: 'insideTopRight', fontSize: 10 }}
          />
          <ReferenceLine
            y={COMPACTION_THRESHOLD}
            stroke="#e08a00"
            strokeDasharray="4 3"
            label={{ value: 'compacts at 19661', position: 'insideTopRight', fontSize: 10 }}
          />
          {episode.compactions.map((pass, i) => (
            <ReferenceLine
              key={i}
              x={pass.at_step}
              stroke={pass.landed ? '#2a7' : '#c1121f'}
              strokeDasharray="2 2"
            />
          ))}
          <Line
            type="monotone"
            dataKey="tokens"
            stroke="#25467a"
            strokeWidth={1.6}
            dot={{ r: 1.5 }}
          />
        </LineChart>
      </ResponsiveContainer>

      <PassTable episode={episode} />
    </div>
  )
}

/**
 * What each pass replaced and what the curve did about it.
 *
 * The net saving is the number to read: a pass that shadows 10k tokens and
 * leaves context where it was has written a checkpoint as large as the region.
 */
function PassTable({ episode }: { episode: Episode }) {
  if (episode.compactions.length === 0) return null
  const tokens = new Map(
    episode.steps.filter(s => s.input_tokens > 0).map(s => [s.index, s.input_tokens]),
  )
  const indices = [...tokens.keys()].sort((a, b) => a - b)

  const around = (step: number) => {
    const before = [...indices].reverse().find(i => i <= step)
    const after = indices.find(i => i > step)
    return {
      before: before === undefined ? 0 : tokens.get(before)!,
      after: after === undefined ? 0 : tokens.get(after)!,
    }
  }

  return (
    <table className="grid">
      <thead>
        <tr>
          <th>pass</th>
          <th>step</th>
          <th>shadowed</th>
          <th>summary</th>
          <th>context before</th>
          <th>after</th>
          <th>net</th>
          <th />
        </tr>
      </thead>
      <tbody>
        {episode.compactions.map((pass, i) => {
          const { before, after } = around(pass.at_step)
          const net = after && before ? after - before : 0
          return (
            <tr key={i}>
              <td>{i + 1}</td>
              <td>{pass.at_step}</td>
              <td>{pass.shadowed_tokens.toLocaleString()}</td>
              <td>{pass.summary.length.toLocaleString()} ch</td>
              <td>{before.toLocaleString()}</td>
              <td className={after > COMPACTION_THRESHOLD ? 'bad' : undefined}>
                {after.toLocaleString()}
              </td>
              <td className={net < 0 ? 'good' : 'bad'}>{net ? net.toLocaleString() : '—'}</td>
              <td>{pass.landed ? '' : <span className="bad">{pass.error}</span>}</td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}
