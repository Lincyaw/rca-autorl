import { defineTool } from '@deepseek-ai/dsh-tools'

/** Model-facing name of the terminal submission tool. */
export const SUBMIT_TOOL = 'submit_result'

/**
 * The fault propagation graph the RCA agent must produce. The parameter DSL
 * carries types and required keys; `execute` hand-checks what the DSL cannot
 * express (non-empty arrays, non-empty ids).
 */
const PARAMETERS = {
  nodes: {
    type: 'array',
    required: true,
    description: 'Propagation graph nodes, each grounded in replayable evidence.',
    items: {
      type: 'object',
      additionalProperties: false,
      properties: {
        id: { type: 'string', required: true, description: 'Node id referenced by edges.' },
        subject: { type: 'string', required: true, description: 'Service or component the node is about.' },
        predicate: { type: 'string', required: true, description: 'What the subject does or suffers.' },
        time: {
          type: 'object',
          required: true,
          additionalProperties: false,
          properties: {
            start: { type: 'string', required: true, description: 'ISO 8601 start of the observation window.' },
            end: { type: 'string', required: true, description: 'ISO 8601 end of the observation window.' },
          },
        },
        evidence: {
          type: 'array',
          required: true,
          description: 'Replayable evidence from the snapshot. Empty only when hypothesis is true.',
          items: {
            type: 'object',
            additionalProperties: false,
            properties: {
              query: {
                type: 'object',
                required: true,
                additionalProperties: false,
                properties: {
                  language: { type: 'string', required: true, description: 'Query language, such as promql or sql.' },
                  statement: { type: 'string', required: true, description: 'The query that reproduces this evidence.' },
                },
              },
              explanation: { type: 'string', required: true, description: 'What the query result shows.' },
            },
          },
        },
        hypothesis: { type: 'boolean', description: 'True when the node is asserted without evidence.' },
      },
    },
  },
  edges: {
    type: 'array',
    required: true,
    description: 'Causal edges between node ids, cause first.',
    items: {
      type: 'object',
      additionalProperties: false,
      properties: {
        src: { type: 'string', required: true },
        dst: { type: 'string', required: true },
      },
    },
  },
  root_causes: {
    type: 'array',
    required: true,
    description: 'Node ids that are root causes of the incident.',
    items: { type: 'string' },
  },
}

/**
 * Register the terminal submission tool. The call concludes the agent turn, and
 * the guard keeps every later call in the same response from running, so one
 * episode produces exactly one submission. The arguments reach the trainer
 * through the session log's `tool/call` event; nothing else records them.
 */
export function registerSubmitResult(ctx, state) {
  ctx.tools.register(defineTool({
    name: SUBMIT_TOOL,
    description:
      'Submit the final fault propagation graph and its root causes. Call this exactly once, '
      + 'when the investigation is complete; it ends the episode.',
    parameters: PARAMETERS,
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          recorded: { type: 'boolean' },
          node_count: { type: 'integer' },
          root_cause_count: { type: 'integer' },
        },
      },
      render: (_args, value) => [{
        type: 'text',
        text: `Recorded ${value.node_count} node(s) and ${value.root_cause_count} root cause(s).`,
      }],
    },
    execute(args, exec) {
      if (args.nodes.length === 0) throw new Error('`nodes` must not be empty')
      if (args.root_causes.length === 0) throw new Error('`root_causes` must not be empty')
      const ids = new Set(args.nodes.map(node => node.id))
      for (const edge of args.edges) {
        for (const [field, id] of [['src', edge.src], ['dst', edge.dst]]) {
          if (!ids.has(id)) throw new Error(`edge.${field} ${JSON.stringify(id)} is not a declared node id`)
        }
      }
      for (const id of args.root_causes) {
        if (!ids.has(id)) throw new Error(`root cause ${JSON.stringify(id)} is not a declared node id`)
      }
      state.submitted.add(state.key(exec))
      exec.concludeTurn()
      return Promise.resolve({
        recorded: true,
        node_count: args.nodes.length,
        root_cause_count: args.root_causes.length,
      })
    },
  }))

  ctx.tools.guard(exec => state.submitted.has(state.key(exec))
    ? `the root-cause analysis is already submitted, so \`${exec.name}\` is not executed`
    : undefined)
}
