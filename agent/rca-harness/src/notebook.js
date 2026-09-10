import { defineTool } from '@deepseek-ai/dsh-tools'
import { clearUnnoted } from './note-ledger.js'
import { NOTE_CONTENT_DESCRIPTION, NOTE_DESCRIPTION, NOTE_TOOL } from './prompts.js'


/**
 * Persistent investigation notebook: the episode's external store.
 *
 * `take_note` with `content` writes a finding; `take_note` with no argument
 * reads the notebook back. Both matter, and the read is what makes the write
 * cheap: a checkpoint can state a conclusion and leave the evidence here,
 * because a later step can ask for it.
 *
 * Without the read there was no way to look a finding up again, so a checkpoint
 * was the only thing that survived a compaction pass and the checkpoint
 * template had to carry every grounding SQL statement verbatim. Measured over
 * three passes of one episode that grew the summary from 10123 to 21886
 * characters while the region each pass could replace stayed at 9-12k tokens:
 * the passes stopped paying for themselves, context crossed the threshold it
 * had just compacted below, and the episode reached the serving window and died.
 *
 * So the write no longer echoes the notebook either. Echoing it put the whole
 * accumulated store back into context on every note — 13 notes meant printing
 * n1 for the thirteenth time — which is the same growth by another route.
 *
 * Context management is three mechanisms, in order of what they hold:
 * - The notebook holds findings, and is read on demand rather than echoed.
 * - The tool-result pruner shrinks noted sql results to a file reference,
 *   leaving whole only those not yet noted (`pruner.js`).
 * - Compaction replaces the conversation with a checkpoint of the metadata a
 *   resuming model needs, not of the evidence (`prompts.js` RCA_INSTRUCTION).
 *
 * The notebook is ephemeral to the episode (in-memory, keyed by agent id).
 */
export function registerNotebook(ctx, state) {
  const notebooks = new Map()

  function getNotebook(exec) {
    const key = state.key(exec)
    if (!notebooks.has(key)) {
      // A forked episode inherits the parent's notes, numbered as they were.
      notebooks.set(key, (state.notes ?? []).map((content, i) => ({ id: `n${i + 1}`, content })))
    }
    return notebooks.get(key)
  }

  function renderNotebook(notes) {
    if (!notes.length) return 'Notebook is empty.'
    return notes.map(n => `[${n.id}] ${n.content}`).join('\n')
  }

  ctx.tools.register(defineTool({
    name: NOTE_TOOL,
    description: NOTE_DESCRIPTION,
    parameters: {
      // Optional: omitting it is the read. `required: true` here was what left
      // the notebook write-only.
      content: {
        type: 'string',
        description: NOTE_CONTENT_DESCRIPTION,
      },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          note_id: { type: 'string' },
          notebook: { type: 'string' },
          count: { type: 'number' },
        },
      },
      // A write confirms and stops; only a read spends context on the notebook.
      render: (_args, value) => [{
        type: 'text',
        text: value.note_id
          ? `Note ${value.note_id} recorded (${value.count} in the notebook).`
          : `--- Investigation Notebook ---\n${value.notebook}`,
      }],
    },
    execute(args, exec) {
      const notes = getNotebook(exec)
      const content = args.content?.trim()
      if (!content) {
        return Promise.resolve({ note_id: '', notebook: renderNotebook(notes), count: notes.length })
      }
      const id = `n${notes.length + 1}`
      notes.push({ id, content })
      // The note landed, so the queries behind it are written down and the
      // pruner may shrink them. Cleared here rather than in the gate because
      // only this side knows a write happened rather than a read.
      clearUnnoted(state.key(exec))
      return Promise.resolve({ note_id: id, notebook: '', count: notes.length })
    },
  }))
}
