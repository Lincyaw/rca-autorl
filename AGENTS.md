# Repository Guidelines

## Project Structure & Module Organization
`src/autorl/` contains the first-party scaffold. Keep schema and interfaces in `contracts/`, runtime wiring in `runtime/`, task adapters in `tasks/`, tool and environment boundaries in `gateways/` and `tool_env/`, and runnable entrypoints in `experiments/agent_workflow/`. Training configs live in `configs/train/`, helper scripts in `scripts/`, and upstream AReaL code in `third_party/AReaL/`.

## Build, Test, and Development Commands
- `git submodule update --init --recursive` fetches the AReaL submodule.
- `uv sync` installs the project and editable local dependencies into `.venv/`.
- `./scripts/run_smoke.sh` runs the smoke training config with the repo venv on `PATH`.
- `PATH="$PWD/.venv/bin:$PATH" python3 -m autorl.experiments.agent_workflow.train --config configs/train/base.yaml` runs the baseline train path.
- `uv run ruff check src` lint-checks first-party Python code.
Rely on `third_party/AReaL/pyproject.toml` for shared runtime packages; avoid re-pinning AReaL-owned deps in the root project unless this repo adds a truly new requirement.

## Coding Style & Naming Conventions
Use Python 3.12+ conventions: 4-space indentation, explicit type hints, and short, interface-first modules. Follow existing naming: `snake_case` for modules/functions, `PascalCase` for classes, and lowercase YAML config names such as `smoke.yaml`. Add new behavior by extending task or runtime modules instead of branching the shared workflow when possible.

## Testing Guidelines
There is no first-party `tests/` directory yet. Validate changes with the smallest runnable path, usually `./scripts/run_smoke.sh` or a targeted `python3 -m autorl.experiments.agent_workflow.{train,eval,infer}` command. If tests are explicitly requested, place them under `tests/` using `test_<module>.py` naming and keep coverage focused on changed contracts, adapters, or loaders.

## Commit & Pull Request Guidelines
The repository has no commit history yet, so use intent-first, imperative commit subjects. For repo-managed commits, follow the workspace Lore format with trailers such as `Constraint:`, `Rejected:`, `Confidence:`, and `Tested:`. Pull requests should explain why the change is needed, list touched configs or modules, note any submodule or dataset assumptions, and include the exact validation command or smoke-log snippet.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
