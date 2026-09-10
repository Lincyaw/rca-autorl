import type { CaseGroup, Episode } from './api'

/**
 * The K samples of one case side by side.
 *
 * The number this view exists for is the spread. RLOO centres a trajectory
 * against its siblings, so a case whose K samples score alike hands every one of
 * them a zero advantage and contributes nothing to the batch — the spread is
 * whether this case can teach anything at all.
 */
export function Siblings({
  group,
  onOpen,
}: {
  group: CaseGroup
  onOpen: (sessionId: string) => void
}) {
  const scores = group.episodes.map(e => e.score)
  const best = Math.max(...scores)
  const spread = best - Math.min(...scores)

  return (
    <div>
      <p>
        <strong>{group.episodes.length}</strong> sample(s) ·{' '}
        pass@1 <strong>{mean(scores).toFixed(3)}</strong> · pass@k{' '}
        <strong>{best.toFixed(3)}</strong> · spread{' '}
        <strong className={spread > 0 ? 'good' : 'bad'}>{spread.toFixed(3)}</strong>
        {spread === 0 && <span className="muted"> — zero advantage for every sibling</span>}
      </p>

      <table className="grid">
        <thead>
          <tr>
            <th>sample</th>
            <th>score</th>
            <th>finish</th>
            <th>steps</th>
            <th>sql</th>
            <th>notes</th>
            <th>peak ctx</th>
            <th>compactions</th>
          </tr>
        </thead>
        <tbody>
          {group.episodes.map(episode => (
            <tr key={episode.session_id}>
              <td>
                <button className="link" onClick={() => onOpen(episode.session_id)}>
                  {episode.session_id}
                </button>
              </td>
              <td className={episode.score === best && best > 0 ? 'good' : undefined}>
                {episode.submitted ? episode.score.toFixed(3) : '—'}
              </td>
              <td className={episode.finish === 'completed' ? 'good' : 'bad'}>
                {episode.finish}
                {episode.error && <div className="muted small">{episode.error}</div>}
              </td>
              <td>{episode.steps.length}</td>
              <td>{episode.tools.sql ?? 0}</td>
              <td>{episode.tools.take_note ?? 0}</td>
              <td>{episode.peak_input_tokens.toLocaleString()}</td>
              <td>
                {episode.compactions.length}
                {episode.compaction_failures > 0 && (
                  <span className="bad"> ({episode.compaction_failures} failed)</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <h4>Where they diverge</h4>
      <Divergence episodes={group.episodes} />
    </div>
  )
}

/**
 * The first step at which the samples stopped issuing the same call.
 *
 * Siblings share a prompt, so an identical prefix is expected; the step the
 * prefix ends at is where sampling started to matter.
 */
function Divergence({ episodes }: { episodes: Episode[] }) {
  // An episode killed on its first request has no trajectory to compare, and
  // including it puts the shared prefix at zero for everyone: two samples that
  // both opened with `SHOW TABLES` would read as having diverged immediately.
  const usable = episodes.filter(e => e.steps.length > 1)
  const stillborn = episodes.length - usable.length
  if (usable.length < 2) {
    return (
      <p className="muted">
        Needs two samples that got past their first step
        {stillborn > 0 && ` — ${stillborn} of ${episodes.length} died on the first request`}.
      </p>
    )
  }

  const shortest = Math.min(...usable.map(e => e.steps.length))
  let at = shortest
  for (let i = 0; i < shortest; i += 1) {
    const first = signature(usable[0].steps[i])
    if (usable.some(e => signature(e.steps[i]) !== first)) {
      at = i
      break
    }
  }

  const excluded =
    stillborn > 0 ? ` (${stillborn} sample(s) excluded: died on the first request)` : ''

  if (at === shortest) {
    return (
      <p className="muted">
        Identical for all {shortest} shared step(s) — these samples are the same trajectory
        {excluded}.
      </p>
    )
  }

  return (
    <div>
      <p>
        Shared prefix: <strong>{at}</strong> step(s) over {usable.length} sample(s){excluded}. At
        step <strong>{usable[0].steps[at]?.index}</strong> they part:
      </p>
      <ul className="plain">
        {usable.map(episode => (
          <li key={episode.session_id}>
            <span className="muted">{episode.session_id.slice(-8)}</span>{' '}
            <code>{signature(episode.steps[at]).slice(0, 160)}</code>
          </li>
        ))}
      </ul>
    </div>
  )
}

function signature(step: Episode['steps'][number] | undefined): string {
  if (!step) return ''
  return `${step.tool} ${step.arguments}`.replace(/\s+/g, ' ')
}

function mean(values: number[]): number {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0
}
