# Repository Guidelines

## Project Structure & Module Organization
`src/autorl/agent.py` is the direct AReaL workflow around DeepSeek Harness, `harness.py` installs the RCA harness bundle into a `dsh` profile, and `train.py` / `train_sft.py` are the runnable RL and SFT entrypoints. SFT tokenization helpers live in `src/autorl/data/`. `agent/` holds the RCA harness: the `dsh` bundle `agent/rca-harness` (plain ESM JavaScript, no build step; its one npm dependency is the DuckDB binding) and the scenario patches in `agent/profiles/`; `agent/README.md` is its design. Training configs live in `configs/`, helper scripts in `scripts/`, and upstream AReaL code in `third_party/AReaL/`.

## Build, Test, and Development Commands
- `git submodule update --init --recursive` fetches the AReaL submodule.
- `uv sync` installs the project and editable local dependencies into `.venv/`.
- `./scripts/run_smoke.sh` runs the smoke training config with the repo venv on `PATH`.
- `PATH="$PWD/.venv/bin:$PATH" python3 -m autorl.train --config configs/train/dsh_rca_smoke.yaml` runs the RCA RL path.
- `python3 -m autorl.harness <dsh-home> [--reinstall]` installs the harness bundle into a profile (needs `pnpm` on `PATH`).
- `./scripts/check.sh` runs the required Ruff lint/format and mypy checks.
Rely on `third_party/AReaL/pyproject.toml` for shared runtime packages; avoid re-pinning AReaL-owned deps in the root project unless this repo adds a truly new requirement.

## Coding Style & Naming Conventions
Use Python 3.12+ conventions: 4-space indentation, explicit type hints, and short modules. Keep the integration shaped like AReaL's SWE example: DeepSeek Harness owns the agent loop, while this repository loads data, composes the harness, invokes the agent, computes reward, and launches training. Bundle code follows the harness's own extension contracts: register tools with `defineTool`, express policy as `tools/pre-execute` listeners or `ctx.tools.guard()`, and keep deployment-varying values in the row's config rather than in the source.

## Testing Guidelines
Run `./scripts/check.sh` before every commit. The installed pre-commit hook enforces it locally, and CI repeats it on pushes and pull requests. Run `python -m unittest discover -s tests` for unit tests and `./scripts/run_smoke.sh` for an end-to-end GPU smoke run. Keep tests focused on the workflow boundary, dataset loading, and SFT masking.

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
