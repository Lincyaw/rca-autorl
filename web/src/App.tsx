import { useEffect, useState } from 'react'
import Markdown from 'react-markdown'
import { api, type CaseGroup, type Episode, type EpisodeRow, type Run } from './api'
import { ContextCurve } from './ContextCurve'
import { GraphDiff } from './GraphDiff'
import { Siblings } from './Siblings'
import { Timeline } from './Timeline'

type View = 'timeline' | 'context' | 'graph' | 'siblings'

export function App() {
  const [runs, setRuns] = useState<Run[]>([])
  const [run, setRun] = useState<string>('')
  const [rows, setRows] = useState<EpisodeRow[]>([])
  const [selected, setSelected] = useState<string>('')
  const [episode, setEpisode] = useState<Episode | null>(null)
  const [group, setGroup] = useState<CaseGroup | null>(null)
  const [view, setView] = useState<View>('timeline')
  const [report, setReport] = useState<string>('')
  const [error, setError] = useState<string>('')

  useEffect(() => {
    api.runs().then(found => {
      setRuns(found)
      if (found.length > 0) setRun(found[0].run)
    }, fail(setError))
  }, [])

  useEffect(() => {
    if (!run) return
    setRows([])
    setSelected('')
    setEpisode(null)
    setReport('')
    api.episodes(run).then(setRows, fail(setError))
  }, [run])

  // One episode and its sibling group travel together: the group is what makes
  // a single sample's score mean anything.
  useEffect(() => {
    if (!run || !selected) return
    api.episode(run, selected).then(loaded => {
      setEpisode(loaded)
      api.group(run, loaded.case).then(setGroup, fail(setError))
    }, fail(setError))
  }, [run, selected])

  return (
    <div className="app">
      <header>
        <h1>RCA trajectories</h1>
        <select value={run} onChange={e => setRun(e.target.value)}>
          {runs.map(r => (
            <option key={r.run} value={r.run}>
              {r.run} ({r.episodes})
            </option>
          ))}
        </select>
        <button
          className="link"
          onClick={() => api.report(run).then(setReport, fail(setError))}
          disabled={!run}
        >
          report
        </button>
        {error && <span className="bad">{error}</span>}
      </header>

      <div className="split">
        <EpisodeList rows={rows} selected={selected} onSelect={setSelected} />

        <main>
          {report && (
            <section className="report">
              <button className="link close" onClick={() => setReport('')}>
                close
              </button>
              <Markdown>{report}</Markdown>
            </section>
          )}

          {!episode && !report && <p className="muted">Pick an episode.</p>}

          {episode && (
            <>
              <EpisodeHead episode={episode} />
              <nav className="tabs">
                {(['timeline', 'context', 'graph', 'siblings'] as View[]).map(name => (
                  <button
                    key={name}
                    className={view === name ? 'tab on' : 'tab'}
                    onClick={() => setView(name)}
                  >
                    {name}
                  </button>
                ))}
              </nav>
              {view === 'timeline' && <Timeline episode={episode} />}
              {view === 'context' && <ContextCurve episode={episode} />}
              {view === 'graph' && <GraphDiff episode={episode} />}
              {view === 'siblings' &&
                (group ? (
                  <Siblings group={group} onOpen={setSelected} />
                ) : (
                  <p className="muted">Loading the group…</p>
                ))}
            </>
          )}
        </main>
      </div>
    </div>
  )
}

function EpisodeList({
  rows,
  selected,
  onSelect,
}: {
  rows: EpisodeRow[]
  selected: string
  onSelect: (id: string) => void
}) {
  return (
    <aside>
      {rows.length === 0 && <p className="muted">No episode.</p>}
      <ul className="plain">
        {rows.map(row => (
          <li key={row.session_id}>
            <button
              className={row.session_id === selected ? 'row on' : 'row'}
              onClick={() => onSelect(row.session_id)}
            >
              <span className="row-id">{row.session_id}</span>
              <span className={row.finish === 'completed' ? 'good' : 'bad'}>{row.finish}</span>
              <span className="muted">
                {row.steps} steps · {row.submitted ? row.score.toFixed(3) : 'no answer'}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </aside>
  )
}

function EpisodeHead({ episode }: { episode: Episode }) {
  return (
    <section className="head">
      <h2>{episode.session_id}</h2>
      <dl>
        <dt>case</dt>
        <dd>{episode.case}</dd>
        <dt>finish</dt>
        <dd className={episode.finish === 'completed' ? 'good' : 'bad'}>
          {episode.finish}
          {episode.error && ` — ${episode.error}`}
        </dd>
        <dt>score</dt>
        <dd>{episode.submitted ? episode.score.toFixed(4) : 'never submitted'}</dd>
        <dt>steps</dt>
        <dd>
          {episode.steps.length} ·{' '}
          {Object.entries(episode.tools)
            .map(([tool, n]) => `${tool} ${n}`)
            .join(' · ') || 'no call'}
        </dd>
        <dt>peak context</dt>
        <dd>{episode.peak_input_tokens.toLocaleString()}</dd>
        <dt>compactions</dt>
        <dd>
          {episode.compactions.length}
          {episode.compaction_failures > 0 && (
            <span className="bad"> — {episode.compaction_failures} failed</span>
          )}
        </dd>
      </dl>
    </section>
  )
}

/** Show what broke rather than a blank panel; a dead endpoint is the usual cause. */
function fail(set: (message: string) => void) {
  return (reason: unknown) => set(String(reason))
}
