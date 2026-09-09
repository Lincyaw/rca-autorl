import { defineTool } from '@deepseek-ai/dsh-tools'
import { PARAMETERS, validate } from './contract.js'
import { alreadySubmitted, submitDescription, SUBMIT_TOOL } from './prompts.js'
import { VOCAB_VERSION } from './vocabulary.js'


/**
 * Register the terminal submission tool. The call concludes the agent turn, and
 * the guard keeps every later call in the same response from running, so one
 * episode produces exactly one submission. The arguments reach the trainer
 * through the session log's `tool/call` event; nothing else records them.
 */
export function registerSubmitResult(ctx, state) {
  ctx.tools.register(defineTool({
    name: SUBMIT_TOOL,
    description: submitDescription(VOCAB_VERSION),
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
      validate(args)
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
    ? alreadySubmitted(exec.name)
    : undefined)
}
