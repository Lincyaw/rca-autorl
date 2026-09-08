import {
  EDGE_DST_DESCRIPTION, EDGE_SRC_DESCRIPTION, EDGES_DESCRIPTION, EVIDENCE_DESCRIPTION,
  EVIDENCE_EXPLANATION_DESCRIPTION, EVIDENCE_LANGUAGE_DESCRIPTION, EVIDENCE_STATEMENT_DESCRIPTION,
  HYPOTHESIS_DESCRIPTION, NODE_ID_DESCRIPTION, NODES_DESCRIPTION, PREDICATE_DESCRIPTION,
  reject, ROOT_CAUSES_DESCRIPTION, SUBJECT_DESCRIPTION, TIME_DESCRIPTION, TIME_END_DESCRIPTION,
  TIME_START_DESCRIPTION,
} from './prompts.js'
import { ENTITY_REF_PATTERN, ENTITY_TYPES, NODE_PREDICATES, QUERY_LANGUAGES } from './vocabulary.js'

/**
 * The answer contract, kept apart from the tool that registers it.
 *
 * Structure lives here and prose lives in `prompts.js`, which is the audience
 * boundary: everything the model reads is in one file that can be scanned.
 * Nothing here imports the harness either, so the contract test can drive
 * `validate` with plain Node and check that it gives the same verdicts as
 * `autorl.fpg.parse_submission` on the same fixtures.
 */

const ENTITY_REF = new RegExp(ENTITY_REF_PATTERN)

/**
 * The fault propagation graph the RCA agent must produce: `ModelRCAOutput` of
 * the `fpg` schema, bound to the `microservices` vocabulary profile.
 *
 * The parameter DSL carries types, required keys, and the two closed
 * vocabularies it can express as `enum`; `validate` hand-checks the rest, since
 * the enforced JSON Schema subset has no `pattern`, no `minItems`, and no
 * cross-field rules. Together they are at least as strict as
 * `fpg.model_output.ModelRCAOutput`, so every call this tool accepts is one the
 * verifier can parse — `autorl.fpg.parse_submission` re-validates a stored
 * submission against the same profile, and `tests/test_submit_result_contract`
 * runs both sides over the same fixtures. Stricter in one place on purpose: a
 * timestamp must carry a timezone offset, which pydantic would let pass as a
 * naive datetime and which no reader could align against the snapshot.
 */
export const PARAMETERS = {
  nodes: {
    type: 'array',
    required: true,
    description: NODES_DESCRIPTION,
    items: {
      type: 'object',
      additionalProperties: false,
      properties: {
        id: { type: 'string', required: true, description: NODE_ID_DESCRIPTION },
        subject: { type: 'string', required: true, description: SUBJECT_DESCRIPTION },
        predicate: {
          type: 'string',
          required: true,
          enum: NODE_PREDICATES.map(predicate => predicate.value),
          description: PREDICATE_DESCRIPTION,
        },
        time: {
          type: 'object',
          required: true,
          additionalProperties: false,
          description: TIME_DESCRIPTION,
          properties: {
            start: { type: 'string', required: true, description: TIME_START_DESCRIPTION },
            end: { type: 'string', required: true, description: TIME_END_DESCRIPTION },
          },
        },
        evidence: {
          type: 'array',
          required: true,
          description: EVIDENCE_DESCRIPTION,
          items: {
            type: 'object',
            additionalProperties: false,
            properties: {
              query: {
                type: 'object',
                required: true,
                additionalProperties: false,
                properties: {
                  language: { type: 'string', required: true, enum: QUERY_LANGUAGES, description: EVIDENCE_LANGUAGE_DESCRIPTION },
                  statement: { type: 'string', required: true, description: EVIDENCE_STATEMENT_DESCRIPTION },
                },
              },
              explanation: { type: 'string', required: true, description: EVIDENCE_EXPLANATION_DESCRIPTION },
            },
          },
        },
        hypothesis: {
          type: 'boolean',
          description: HYPOTHESIS_DESCRIPTION,
        },
      },
    },
  },
  edges: {
    type: 'array',
    required: true,
    description: EDGES_DESCRIPTION,
    items: {
      type: 'object',
      additionalProperties: false,
      properties: {
        src: { type: 'string', required: true, description: EDGE_SRC_DESCRIPTION },
        dst: { type: 'string', required: true, description: EDGE_DST_DESCRIPTION },
      },
    },
  },
  root_causes: {
    type: 'array',
    required: true,
    description: ROOT_CAUSES_DESCRIPTION,
    items: { type: 'string' },
  },
}

/** ISO 8601 instant with an explicit offset — `+01` is not one, `+01:00` is. */
const ISO_8601 = /^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/

/**
 * The rules the schema DSL cannot state. Each violation is thrown as an
 * ordinary tool error, which ends nothing: the model sees what it broke and
 * retries inside the same turn. Exported so the contract test can run it over
 * the same fixtures it runs the pydantic models over.
 */
export function validate(args) {
  const ids = new Set()
  for (const node of args.nodes) {
    const where = `node ${JSON.stringify(node.id)}`
    if (node.id.length === 0) throw new Error(reject.emptyNodeId())
    if (ids.has(node.id)) throw new Error(reject.duplicateNodeId(node.id))
    ids.add(node.id)
    if (!ENTITY_REF.test(node.subject)) {
      throw new Error(reject.entityRef(where, node.subject, ENTITY_TYPES.map(e => e.prefix)))
    }
    for (const bound of ['start', 'end']) {
      if (!ISO_8601.test(node.time[bound])) {
        throw new Error(reject.timestamp(where, bound, node.time[bound]))
      }
    }
    if (Date.parse(node.time.start) > Date.parse(node.time.end)) {
      throw new Error(reject.timeOrder(where))
    }
    if (node.hypothesis !== true && node.evidence.length === 0) {
      throw new Error(reject.evidenceRequired(where))
    }
  }
  if (args.nodes.length === 0) throw new Error(reject.emptyList('nodes'))
  if (args.root_causes.length === 0) throw new Error(reject.emptyList('root_causes'))
  for (const edge of args.edges) {
    for (const [field, id] of [['src', edge.src], ['dst', edge.dst]]) {
      if (!ids.has(id)) throw new Error(reject.unknownEdgeEndpoint(field, id))
    }
    if (edge.src === edge.dst) throw new Error(reject.selfLoop(edge.src))
  }
  for (const id of args.root_causes) {
    if (!ids.has(id)) throw new Error(reject.unknownRootCause(id))
  }
}
