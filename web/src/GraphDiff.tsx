import type { Axis, Episode } from './api'

/** The axes in the order the score weights them, per `difficulty.AXES`. */
const AXES: { key: string; label: string }[] = [
  { key: 'roots', label: 'root causes' },
  { key: 'subjects', label: 'nodes' },
  { key: 'edges', label: 'edges' },
]

/**
 * The submitted graph against the annotation, per axis.
 *
 * Not a drawn graph: the score is set arithmetic on subjects, and what a
 * reviewer needs is which subject was hit, missed or invented — a layout would
 * add a reading step without adding a fact. An episode that never submitted
 * still shows every truth element under "missed", which is the finding.
 */
export function GraphDiff({ episode }: { episode: Episode }) {
  if (!episode.axes) {
    return (
      <p className="muted">
        No ground truth on this machine for {episode.case} — the snapshot directory is gone.
      </p>
    )
  }

  return (
    <div>
      {!episode.submitted && (
        <p className="muted">
          This episode never submitted, so everything true is missed. It finished as{' '}
          <strong>{episode.finish}</strong>.
        </p>
      )}
      {AXES.map(({ key, label }) => {
        const axis = episode.axes?.[key]
        if (!axis) return null
        return <AxisPanel key={key} label={label} axis={axis} />
      })}
    </div>
  )
}

function AxisPanel({ label, axis }: { label: string; axis: Axis }) {
  const truth = axis.found.length + axis.missed.length
  const claimed = axis.found.length + axis.invented.length
  const recall = truth ? axis.found.length / truth : 1
  const precision = claimed ? axis.found.length / claimed : 0

  return (
    <section className="axis">
      <h4>
        {label}
        <span className="muted">
          {' '}
          recall {axis.found.length}/{truth} ({(recall * 100).toFixed(0)}%) · precision{' '}
          {axis.found.length}/{claimed} ({(precision * 100).toFixed(0)}%)
        </span>
      </h4>
      <div className="axis-cols">
        <Column title="found" tone="good" items={axis.found} />
        <Column title="missed" tone="bad" items={axis.missed} />
        <Column title="invented" tone="warn" items={axis.invented} />
      </div>
    </section>
  )
}

function Column({ title, tone, items }: { title: string; tone: string; items: string[] }) {
  return (
    <div>
      <div className={`col-head ${tone}`}>
        {title} ({items.length})
      </div>
      {items.length === 0 ? (
        <div className="muted">—</div>
      ) : (
        <ul className="plain">
          {items.map(item => (
            <li key={item}>
              <code>{item}</code>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
