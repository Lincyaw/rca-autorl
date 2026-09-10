"""The notebook is the store the checkpoint cites instead of restating.

A compaction pass leaves a checkpoint that names notes; the resuming model reads
them back. So the properties that matter are that a name resolves, that listing
does not cost the notebook, that writing a name again consolidates rather than
duplicates, and that the gate in front of `sql` is settled by a write and not by
a read.

`notebook.js` imports the harness's own packages, which resolve only inside a
`dsh` runtime, so the module is copied into a workspace whose `node_modules`
carries stubs for the two of them.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from node import REPO_ROOT, run_node_module

SRC = REPO_ROOT / "agent" / "rca-harness" / "src"

STUBS = {
    "@deepseek-ai/dsh-tools": "export const defineTool = spec => spec\n",
    "@deepseek-ai/dsh-llm": "export const createUserMessage = message => ({ role: 'user', ...message })\n",
}

# One episode: the writes below, a listing, a read, then the gate. `noteEvery`
# is 2 and `noteLimit` 4, so the fifth query with no note behind it is refused,
# and the two reminders land on the second and third.
HARNESS = """
import { notebookOutline, registerNotebook, registerNotePolicy, unnotedCount } from %s

const tools = []
const listeners = {}
const ctx = { tools: { register: tool => tools.push(tool) }, on: (event, fn) => { listeners[event] = fn } }
const state = { resultRoot: %s, notes: [{ id: 'inherited', content: 'from the parent' }], key: () => 'ep' }
registerNotebook(ctx, state)
registerNotePolicy(ctx, state, 2, 4)
const [note] = tools
const out = { writes: [] }

const call = args => note.execute(args, {}).then(value => ({ value, text: note.output.render(args, value)[0].text }))
const query = () => listeners['tools/pre-execute']({ name: 'sql' }, () => Promise.resolve({ kind: 'allow' }))
const notice = async () => ((await listeners['tools/post-execute']({ name: 'sql' }, {}, () => Promise.resolve({}))).additionalContexts ?? []).length

for (const [id, content] of %s) out.writes.push((({ note_id, replaced }) => ({ note_id, replaced }))((await call({ id, content })).value))
out.confirmation = (await call({ id: 'inherited', content: 'rewritten' })).text
out.index = (await call({})).text
out.read = (await call({ id: 'preserve-severe-logs, inherited' })).text
try { await call({ id: 'no-such-note' }); out.unknown = null }
catch (error) { out.unknown = error.message }

out.gate = []
out.reminders = []
for (let i = 0; i < 5; i += 1) {
  out.gate.push((await query()).kind)
  out.reminders.push(await notice())
}

await call({ id: 'inherited' })
out.afterRead = (await query()).kind
await call({ id: 'a-fresh-finding', content: 'written down' })
out.afterWrite = (await query()).kind
out.debt = unnotedCount('ep')

// A turn that calls sql several times at once takes the count past both
// thresholds before any result is folded.
await call({ id: 'a-second-finding', content: 'written down' })
await query(); await query(); await query()
out.parallel = [await notice(), await notice(), await notice()]
out.outline = notebookOutline('ep', 30)
out.cappedOutline = notebookOutline('ep', 2)
out.noOutline = notebookOutline('no-such-episode', 30) ?? null
process.stdout.write(JSON.stringify(out))
"""

WRITES = [
    ["preserve-severe-logs", "1497 SEVERE lines\nthe statement\nso the service is failing"],
    ["", "travel service is healthy\nno errors in the window"],
    [
        "preserve-severe-logs",
        "1497 SEVERE lines, one template\nthe detail the index must not carry",
    ],
    ["", "travel service is healthy\na second look, same conclusion"],
    ["Preserve Timeouts on the travel call!", "the caller times out"],
    ["", "endpoint ts-ui-dashboard POST /api/v1/preserveservice/preserve is the symptom\nand more"],
]


class NotebookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._workspace = tempfile.TemporaryDirectory()
        root = Path(cls._workspace.name)
        for module in ("notebook.js", "prompts.js", "vocabulary.js"):
            shutil.copy(SRC / module, root / module)
        for package, source in STUBS.items():
            directory = root / "node_modules" / package
            directory.mkdir(parents=True)
            (directory / "package.json").write_text(
                json.dumps({"name": package, "type": "module", "main": "index.js"})
            )
            (directory / "index.js").write_text(source)
        cls.out = run_node_module(
            HARNESS
            % (
                json.dumps(str(root / "notebook.js")),
                json.dumps(str(root / "results")),
                json.dumps(WRITES),
            )
        )
        cls.notes_md = (root / "results" / "ep" / "notes.md").read_text()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._workspace.cleanup()

    def test_a_name_the_model_wrote_is_the_name_it_gets(self) -> None:
        self.assertEqual(
            self.out["writes"][0], {"note_id": "preserve-severe-logs", "replaced": False}
        )

    def test_a_write_without_a_name_is_named_by_its_summary_line(self) -> None:
        self.assertEqual(
            self.out["writes"][1], {"note_id": "travel-service-is-healthy", "replaced": False}
        )

    def test_writing_a_name_again_replaces_that_note(self) -> None:
        self.assertEqual(
            self.out["writes"][2], {"note_id": "preserve-severe-logs", "replaced": True}
        )
        self.assertIn("one template", self.out["read"])
        self.assertNotIn("so the service is failing", self.out["read"])

    def test_a_derived_name_never_overwrites(self) -> None:
        # Two findings that open the same way are two notes: the model did not
        # ask to consolidate them, so neither one is lost.
        self.assertEqual(
            self.out["writes"][3], {"note_id": "travel-service-is-healthy-2", "replaced": False}
        )

    def test_a_name_is_held_to_a_shape(self) -> None:
        self.assertEqual(self.out["writes"][4]["note_id"], "preserve-timeouts-on-the-travel-call")

    def test_a_long_summary_line_is_cut_on_a_word(self) -> None:
        self.assertEqual(self.out["writes"][5]["note_id"], "endpoint-ts-ui-dashboard-post-api-v1")

    def test_a_write_confirms_without_echoing_the_notebook(self) -> None:
        self.assertEqual(self.out["confirmation"], "Note `inherited` replaced (6 in the notebook).")

    def test_the_index_is_one_line_per_note(self) -> None:
        listed = [line for line in self.out["index"].splitlines() if "  " in line]
        self.assertEqual(len(listed), 6)
        self.assertTrue(
            any(line.startswith("preserve-severe-logs  1497 SEVERE") for line in listed)
        )
        self.assertNotIn("the detail the index must not carry", self.out["index"])
        self.assertIn("the detail the index must not carry", self.out["read"])

    def test_a_fork_inherits_its_parents_names(self) -> None:
        self.assertIn("[inherited] rewritten", self.out["read"])

    def test_an_unknown_name_says_what_the_notebook_holds(self) -> None:
        self.assertIn("no-such-note", self.out["unknown"])
        self.assertIn("preserve-severe-logs", self.out["unknown"])

    def test_the_notebook_is_on_disk_beside_the_result_files(self) -> None:
        self.assertIn("[preserve-severe-logs] 1497 SEVERE lines, one template", self.notes_md)

    def test_querying_closes_at_the_limit_and_a_write_reopens_it(self) -> None:
        self.assertEqual(self.out["gate"], ["allow", "allow", "allow", "allow", "deny"])
        # A read is not a note: it must not buy another query.
        self.assertEqual(self.out["afterRead"], "deny")
        self.assertEqual(self.out["afterWrite"], "allow")
        self.assertEqual(self.out["debt"], 1)

    def test_the_checkpoint_can_read_the_index_of_what_was_stored(self) -> None:
        # A note the checkpoint does not cite is still reachable by name, so the
        # whole index rides with it — capped, since it is the one part that
        # grows with the episode.
        self.assertEqual(self.out["outline"]["older"], 0)
        self.assertIn("preserve-severe-logs", self.out["outline"]["names"])
        self.assertEqual(len(self.out["cappedOutline"]["names"]), 2)
        self.assertEqual(self.out["cappedOutline"]["older"], 6)
        self.assertIsNone(self.out["noOutline"])

    def test_the_reminder_is_sent_twice_in_a_cycle_not_once_per_query(self) -> None:
        # noteEvery is 2 and noteLimit is 4: a notice on reaching the debt, one
        # more on the last query before the gate closes, and nothing else. The
        # notice is a message the pruner cannot reach, so repeating it would
        # spend the context it exists to protect.
        self.assertEqual(self.out["reminders"], [0, 1, 1, 0, 0])

    def test_a_parallel_turn_does_not_skip_the_notice(self) -> None:
        # Three queries run before any result is folded, so by the time the
        # notices are decided the count is past both thresholds at once. A rule
        # that matched a threshold exactly would send nothing; this sends the
        # one that matters, that the gate is about to close, and sends it once.
        self.assertEqual(self.out["parallel"], [1, 0, 0])


if __name__ == "__main__":
    unittest.main()
