import { defineTool } from '@deepseek-ai/dsh-tools'
import { NOTE_CONTENT_DESCRIPTION, NOTE_DESCRIPTION, NOTE_TOOL } from './prompts.js'

export { NOTE_TOOL }

/**
 * Persistent investigation notebook.
 *
 * Each `take_note` call records a finding in a persistent in-memory notebook.
 * The tool result always renders the full notebook, so accumulated findings
 * stay visible even after the compaction-tool-result-pruner shrinks older sql
 * results between steps.
 *
 * Context management works through the combination of two mechanisms:
 * - This tool preserves key findings (what matters).
 * - The compaction-tool-result-pruner aggressively prunes old sql results to
 *   head + file-reference tail (what no longer needs to be inline).
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
      content: {
        type: 'string',
        required: true,
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
        },
      },
      render: (_args, value) => [{
        type: 'text',
        text: `Note ${value.note_id} recorded.\n\n--- Investigation Notebook ---\n${value.notebook}`,
      }],
    },
    execute(args, exec) {
      const notes = getNotebook(exec)
      const id = `n${notes.length + 1}`
      notes.push({ id, content: args.content.trim() })
      return Promise.resolve({
        note_id: id,
        notebook: renderNotebook(notes),
      })
    },
  }))

  state.getNotebook = (agentId) => {
    const key = agentId ?? 'root'
    return [...(notebooks.get(key) ?? [])]
  }
}
