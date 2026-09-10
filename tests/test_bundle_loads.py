"""Every module in the bundle has to import cleanly, or no episode starts.

A bundle row that names an export its neighbour does not have fails at plugin
load: `dsh` reports "plugin tree failed to load" and every episode of the run
times out waiting for the runtime. `node --check` does not catch it, because the
syntax is fine, and no other test imports the rows that only the harness loads.

So this test loads them — the whole `src/` tree, in a workspace whose
`node_modules` carries a stub for each `@deepseek-ai/*` package the bundle
imports, since those resolve only inside a `dsh` runtime.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from node import REPO_ROOT, run_node_module

SRC = REPO_ROOT / "agent" / "rca-harness" / "src"

# Each stub exports the names the bundle imports and nothing more: a missing one
# fails here rather than at plugin load, which is the point of the test.
STUBS = {
    # Every schema is built by chaining, so one self-returning proxy serves the
    # whole builder API.
    "@deepseek-ai/schemastery": (
        "const chain = new Proxy(function () {}, { get: () => chain, apply: () => chain })\n"
        "export default chain\n"
    ),
    "@deepseek-ai/dsh-tools": "export const defineTool = spec => spec\n",
    "@deepseek-ai/dsh-llm": (
        "export const createUserMessage = message => ({ role: 'user', ...message })\n"
        "export const isAgentLoopRequest = () => false\n"
        "export class BlockAssembler {}\n"
    ),
    "@deepseek-ai/dsh-compaction-basic": (
        "export default class BasicCompactionEngine {\n"
        "  constructor(ctx) { this.ctx = ctx }\n"
        "  compactIfNeeded() {}\n"
        "}\n"
    ),
    "@deepseek-ai/dsh-compaction-tool-result-pruner": (
        "export class ToolResultPruner {\n"
        "  pruneSession() { return { pruned: [], charsRemoved: 0 } }\n"
        "  pruneContent() { return null }\n"
        "  measureContent() { return 0 }\n"
        "}\n"
    ),
    "@duckdb/node-api": "export const DuckDBInstance = { create: () => Promise.resolve({}) }\n",
}

HARNESS = """
const loaded = []
for (const module of %s) {
  await import(module)
  loaded.push(module.split('/').pop())
}
process.stdout.write(JSON.stringify(loaded))
"""


class BundleLoadsTest(unittest.TestCase):
    def test_every_module_imports(self) -> None:
        modules = sorted(path.name for path in SRC.glob("*.js"))
        self.assertIn("pruner.js", modules)
        with tempfile.TemporaryDirectory() as workspace:
            root = Path(workspace)
            for module in modules:
                shutil.copy(SRC / module, root / module)
            for package, source in STUBS.items():
                directory = root / "node_modules" / package
                directory.mkdir(parents=True)
                (directory / "package.json").write_text(
                    json.dumps({"name": package, "type": "module", "main": "index.js"})
                )
                (directory / "index.js").write_text(source)
            loaded = run_node_module(
                HARNESS % json.dumps([str(root / module) for module in modules])
            )
        self.assertEqual(sorted(loaded), modules)


if __name__ == "__main__":
    unittest.main()
