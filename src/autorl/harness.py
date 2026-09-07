"""Installing the repo-owned RCA harness bundle into a DeepSeek Harness profile.

`agent/rca-harness` is a `dsh` bundle: `dsh plugin add` records it in the
profile's `package.json`, and its own `cordis.patch.yml` inserts the row that
registers the `sql`, `take_note`, and `submit_result` tools. Install is
a separate step, not part of a rollout: `dsh plugin` shells out to `pnpm`, and
concurrent rollouts must not race on one profile directory.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_DIR = REPO_ROOT / "agent" / "rca-harness"
BUNDLE_NAME = "@rca-autorl/dsh-rca-harness"
PROFILE = "sdk-minimal"


def scenario_patch(scenario: str) -> str:
    """Absolute path of the scenario's per-launch patch layer."""
    patch = REPO_ROOT / "agent" / "profiles" / f"{scenario}.patch.yml"
    if not patch.is_file():
        raise FileNotFoundError(f"no DeepSeek Harness patch for scenario {scenario!r}: {patch}")
    return str(patch)


def require_bundle(dsh_home: Path) -> None:
    """Fail with the exact fix when the profile has not installed the harness bundle."""
    if BUNDLE_NAME in _profile_dependencies(dsh_home):
        return
    raise RuntimeError(
        f"DeepSeek Harness profile {PROFILE!r} in {dsh_home} has no {BUNDLE_NAME}; "
        f"run `python -m autorl.harness {dsh_home}` once before training"
    )


def install_bundle(dsh_home: Path, *, reinstall: bool = False) -> bool:
    """Materialize the profile and add the bundle. Returns False when already installed.

    `file:` rather than `link:`: pnpm copies the bundle into the profile package
    tree, which is where its `@deepseek-ai/dsh-tools` peer import resolves
    through the installation fallback. A linked bundle would have to carry its
    own `node_modules`. The copy means an edit to `agent/rca-harness` reaches a
    session only after `reinstall`.
    """
    dsh_home.mkdir(parents=True, exist_ok=True)
    _run_dsh(dsh_home, ["--profile", PROFILE, "--dump-default-config"])
    installed = BUNDLE_NAME in _profile_dependencies(dsh_home)
    if installed:
        if not reinstall:
            return False
        # `add` on an unchanged `file:` spec is a no-op even when the directory
        # has changed, so a reinstall that only re-adds silently keeps the old
        # copy. Removing first is what makes pnpm copy the bundle again.
        _run_dsh(dsh_home, ["plugin", "--profile", PROFILE, "remove", BUNDLE_NAME])
    _run_dsh(dsh_home, ["plugin", "--profile", PROFILE, "add", f"file:{BUNDLE_DIR}"])
    return True


def _profile_dependencies(dsh_home: Path) -> dict[str, str]:
    manifest = dsh_home / "profiles" / PROFILE / "package.json"
    if not manifest.is_file():
        return {}
    data = json.loads(manifest.read_text(encoding="utf-8"))
    dependencies = data.get("dependencies") if isinstance(data, dict) else None
    return dict(dependencies) if isinstance(dependencies, dict) else {}


def _run_dsh(dsh_home: Path, args: list[str]) -> None:
    from deepseek_harness_runtime import resolve_bundled_launch_args

    argv = [*resolve_bundled_launch_args(), *args]
    env = {**os.environ, "DSH_HOME": str(dsh_home)}
    completed = subprocess.run(argv, env=env, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        tail = "\n".join((completed.stderr or completed.stdout).strip().splitlines()[-8:])
        raise RuntimeError(f"`dsh {' '.join(args)}` failed (exit {completed.returncode}):\n{tail}")


def main(argv: list[str]) -> None:
    reinstall = "--reinstall" in argv
    positional = [arg for arg in argv if arg != "--reinstall"]
    if len(positional) != 1:
        raise SystemExit("usage: python -m autorl.harness <dsh-home> [--reinstall]")
    home = Path(positional[0]).expanduser().resolve()
    added = install_bundle(home, reinstall=reinstall)
    print(f"{BUNDLE_NAME} {'installed into' if added else 'already in'} {home}/profiles/{PROFILE}")


if __name__ == "__main__":
    main(sys.argv[1:])
