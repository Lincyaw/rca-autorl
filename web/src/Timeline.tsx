import { useState } from 'react'
import type { Compaction, Episode, Step } from './api'

/**
 * The episode step by step, with the compaction passes in their place.
 *
 * The main view for auditing why an episode ended where it did. Steps are
 * collapsed to one line each — the tool, the statement, whether the result was
 * an error — because the question is usually "where did this go wrong", and only
 * then "what was it thinking".
 */
export function Timeline({ episode }: { episode: Episode }) {
  const [open, setOpen] = useState<number | null>(null)
  // Interleave: a pass is attributed to the step it interrupted, so it renders
  // after that step.
  const passesAfter = new Map<number, Compaction[]>()
  for (const pass of episode.compactions) {
    const list = passesAfter.get(pass.at_step) ?? []
    list.push(pass)
    passesAfter.set(pass.at_step, list)
  }

  return (
    <ol className="timeline">
      {episode.steps.map(step => (
        <li key={step.index}>
          <StepRow
            step={step}
            open={open === step.index}
            onToggle={() => setOpen(open === step.index ? null : step.index)}
          />
          {(passesAfter.get(step.index) ?? []).map((pass, i) => (
            <PassRow key={i} pass={pass} />
          ))}
        </li>
      ))}
    </ol>
  )
}

function StepRow({
  step,
  open,
  onToggle,
}: {
  step: Step
  open: boolean
  onToggle: () => void
}) {
  const argument = summarizeArguments(step.arguments)
  return (
    <div className={`step${step.result_error ? ' step-error' : ''}`}>
      <button className="step-head" onClick={onToggle}>
        <span className="step-index">{step.index}</span>
        <span className="step-tool">{step.tool || '—'}</span>
        <code className="step-arg">{argument}</code>
        <span className="step-tokens">{step.input_tokens.toLocaleString()}</span>
      </button>
      {open && (
        <div className="step-body">
          {step.reasoning && (
            <section>
              <h4>reasoning</h4>
              <pre>{step.reasoning}</pre>
            </section>
          )}
          {step.text && (
            <section>
              <h4>said</h4>
              <pre>{step.text}</pre>
            </section>
          )}
          {step.arguments && (
            <section>
              <h4>{step.tool}</h4>
              <pre>{pretty(step.arguments)}</pre>
            </section>
          )}
          {step.result_head && (
            <section>
              <h4>result{step.result_error ? ' (error)' : ''}</h4>
              <pre className={step.result_error ? 'bad' : undefined}>{step.result_head}</pre>
            </section>
          )}
        </div>
      )}
    </div>
  )
}

function PassRow({ pass }: { pass: Compaction }) {
  const [open, setOpen] = useState(false)
  return (
    <div className={`pass${pass.landed ? '' : ' pass-failed'}`}>
      <button className="step-head" onClick={() => setOpen(!open)}>
        <span className="step-index">⤵</span>
        <span className="step-tool">compaction</span>
        <span className="step-arg">
          {pass.landed
            ? `shadowed ${pass.shadowed_tokens.toLocaleString()} tokens, wrote ${pass.summary.length.toLocaleString()} characters`
            : pass.error}
        </span>
      </button>
      {open && pass.summary && (
        <div className="step-body">
          <pre>{pass.summary}</pre>
        </div>
      )}
    </div>
  )
}

/** The one line of a call worth showing collapsed. */
function summarizeArguments(raw: string): string {
  if (!raw) return ''
  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>
    // `sql` is the overwhelming majority, and its statement is the whole story.
    const first = parsed.statement ?? parsed.content ?? Object.values(parsed)[0]
    return String(first ?? '').replace(/\s+/g, ' ').slice(0, 150)
  } catch {
    return raw.slice(0, 150)
  }
}

function pretty(raw: string): string {
  try {
    return JSON.stringify(JSON.parse(raw), null, 2)
  } catch {
    return raw
  }
}
