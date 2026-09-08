"""Nothing the model reads may carry knowledge from the answer side.

The repository writes for two audiences. Most of it — the vocabulary profile's
comments, the READMEs, `fpg`'s own field documentation — is for whoever builds
the task and knows how the corpus was made. A small part is read by the agent
being evaluated. Prose that crosses from the first audience to the second hands
over the answer, and it reads as ordinary prose in review: `fpg` documents
`root_causes` as "injection points plus preconditions", which copied into the
tool schema tells the model its incident was staged; and a `link` definition
once named the predicates the annotation pairs it with.

So the boundary is a file. `agent/rca-harness/src/prompts.js` holds every
model-visible string on the JavaScript side, `agent/profiles/rca.patch.yml`
holds the persona, and this test reads both.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_JS = REPO_ROOT / "agent" / "rca-harness" / "src" / "prompts.js"
SCENARIO_DIR = REPO_ROOT / "agent" / "profiles"

# Terms that only someone who built the task would use. Each is a phrase the
# model has no way to learn from its snapshot, so seeing one means prose crossed
# the boundary. Matched case-insensitively on word-ish boundaries.
FORBIDDEN = {
    "staged incident": ["inject", "injected", "injection", "chaos", "fault injection"],
    "the answer": ["ground truth", "ground-truth", "annotation", "annotated", "annotator", "label"],
    "the grading": [
        "reward",
        "f1",
        "precision",
        "recall",
        "scored",
        "penalt",
        "credit",
        "evaluation",
        "evaluated",
        "benchmark",
    ],
    "the corpus": [
        "ops-lite",
        "rcabench",
        "train-ticket",
        "deathstarbench",
        "otel-demo",
        "hotel reservation",
        "testbed",
        "datapack",
        "corpus",
    ],
}

# Deliberately empty. An exception here is a claim that some answer-side phrase
# is safe to hand the model, which is the judgement that produced every one of
# the leaks above; make the prose plainer instead.
ALLOWED: set[str] = set()

HARNESS = """
import * as prompts from %s
const out = {}
for (const [name, value] of Object.entries(prompts)) {
  if (typeof value === 'string') out[name] = value
  // A description that interpolates config is still model-visible; call it with
  // stand-in arguments so its literal half is scanned too.
  else if (typeof value === 'function') out[name] = String(value(1, 2, ['a', 'b']))
  else if (Array.isArray(value)) out[name] = value.join('\\n')
}
process.stdout.write(JSON.stringify(out))
"""


def model_visible_strings() -> dict[str, str]:
    node = shutil.which("node")
    if node is None:
        raise unittest.SkipTest("node is not on PATH")
    completed = subprocess.run(
        [node, "--input-type=module", "-e", HARNESS % json.dumps(str(PROMPTS_JS))],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
        env={**os.environ},
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr)
    texts: dict[str, str] = json.loads(completed.stdout)
    for patch in sorted(SCENARIO_DIR.glob("*.patch.yml")):
        texts[patch.name] = patch.read_text(encoding="utf-8")
    return texts


def leaks(text: str) -> list[tuple[str, str]]:
    haystack = text.lower()
    for phrase in ALLOWED:
        haystack = haystack.replace(phrase, " ")
    found = []
    for kind, terms in FORBIDDEN.items():
        for term in terms:
            if re.search(rf"(?<![a-z]){re.escape(term)}", haystack):
                found.append((kind, term))
    return found


class NoAnswerLeakTest(unittest.TestCase):
    def test_no_model_visible_string_leaks(self) -> None:
        for name, text in model_visible_strings().items():
            with self.subTest(name):
                found = leaks(text)
                self.assertEqual(found, [], f"{name} carries answer-side wording: {found}")

    def test_the_scanner_would_catch_the_two_it_missed(self) -> None:
        """The regressions this test exists for, as literals."""
        self.assertTrue(leaks("possibly several: injection points plus preconditions"))
        self.assertTrue(leaks("A service in the benchmark system, instrumented or backing"))
        self.assertFalse(leaks("Query the incident snapshot with DuckDB SQL."))


if __name__ == "__main__":
    unittest.main()


BUNDLE_SRC = REPO_ROOT / "agent" / "rca-harness" / "src"

# JavaScript globals that look like constants.
JS_GLOBALS = {"JSON", "NaN", "Infinity", "URL", "URLSearchParams", "TextDecoder", "TextEncoder"}


def split_js(source: str) -> tuple[str, list[str]]:
    """Source with comments and string bodies removed, plus the literals removed.

    A regex cannot do this: apostrophes inside comments pair with quotes in code
    and match hundreds of characters of neither.
    """
    code: list[str] = []
    literals: list[str] = []
    index, size = 0, len(source)
    while index < size:
        pair = source[index : index + 2]
        if pair == "//":
            index = size if (index := source.find("\n", index)) < 0 else index
        elif pair == "/*":
            end = source.find("*/", index + 2)
            index = size if end < 0 else end + 2
        elif source[index] in "\"'`":
            quote, index = source[index], index + 1
            body: list[str] = []
            while index < size and source[index] != quote:
                # A template literal's `${...}` is code, not prose.
                if quote == "`" and source[index : index + 2] == "${":
                    depth, index = 1, index + 2
                    opened = index
                    while index < size and depth:
                        depth += (source[index] == "{") - (source[index] == "}")
                        index += 1
                    code.append(source[opened : index - 1])
                    continue
                body.append(source[index])
                index += 2 if source[index] == "\\" else 1
            index += 1
            literals.append("".join(body))
            code.append('""')
        else:
            code.append(source[index])
            index += 1
    return "".join(code), literals


class BundleConventionTest(unittest.TestCase):
    """Two rules the bundle broke, each cheap to state and to check.

    Neither is a substitute for running the thing. The harness runtime ships as
    a packed binary whose peer packages resolve only inside it, so no test here
    can mount the bundle and drive its handlers; the first real signal is an
    episode. These two catch the shapes that actually got through review.
    """

    def test_model_visible_prose_lives_in_one_file(self) -> None:
        """A sentence for the model, anywhere but `prompts.js`, is unscannable.

        A note reminder sat in `note-policy.js` and the leak scan above could
        not see it. Long literals elsewhere are the signature of that mistake.
        """
        for path in sorted(BUNDLE_SRC.glob("*.js")):
            # `compaction.js` throws to the compaction engine, which logs and
            # retries; none of it reaches the model. Its one model-visible
            # artifact, the checkpoint template, is in prompts.js already.
            if path.name in {"prompts.js", "vocabulary.js", "compaction.js"}:
                continue
            with self.subTest(path.name):
                _, literals = split_js(path.read_text("utf-8"))
                long_literals = [
                    literal for literal in literals if len(literal) >= 60 and " " in literal
                ]
                self.assertEqual(
                    [], long_literals, f"{path.name}: move model-visible prose to prompts.js"
                )

    def test_every_constant_a_module_uses_is_bound(self) -> None:
        """`NOTE_TOOL` outlived its import through a refactor and every check passed.

        It failed on the first tool call of a fifty-case collection run, which
        is an hour and a gateway bill later than here.
        """
        for path in sorted(BUNDLE_SRC.glob("*.js")):
            code, _ = split_js(path.read_text("utf-8"))
            # Whole statements, not lines: an import list wraps across several.
            bound = set(re.findall(r"(?:export\s+)?const\s+([A-Z][A-Z0-9_]*)", code))
            for names in re.findall(r"import\s*\{([^}]*)\}", code):
                bound.update(re.findall(r"[A-Z][A-Z0-9_]*", names))
            used = set(re.findall(r"(?<![.\w])([A-Z][A-Z0-9_]{2,})\b", code))
            with self.subTest(path.name):
                self.assertEqual(
                    set(), used - bound - JS_GLOBALS, f"{path.name}: constant used but not bound"
                )
