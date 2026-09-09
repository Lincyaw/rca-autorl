"""Run a bundle module under Node, the way the contract and leak tests do."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_node_module(script: str, **env: str) -> Any:
    """The JSON the ESM `script` writes to stdout, or a skip when Node is absent."""
    node = shutil.which("node")
    if node is None:
        raise unittest.SkipTest("node is not on PATH")
    completed = subprocess.run(
        [node, "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
        env={**os.environ, **env},
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr)
    return json.loads(completed.stdout)
