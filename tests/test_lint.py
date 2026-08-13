"""Runs ruff as a real pytest gate --
`ruff.toml` pins the ruleset to the classic minimal E4/E7/E9/F select
rather than this ruff version's much broader real defaults (98 findings on a
cold run, only 2 of them F821 -- see ruff.toml's own header comment). F821
(undefined name) is the actual motivation: 62 of 64 files in this repo use
`from __future__ import annotations`, which makes an annotation-only
reference to a name a later edit removed invisible at runtime with no other
check catching it.

Deliberately fails, not skips, when ruff isn't installed -- a skipping lint
test isn't a gate. See requirements-dev.txt.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUFF = ROOT / ".venv" / "bin" / "ruff"


def test_ruff_check():
    if not RUFF.is_file():
        raise AssertionError(
            f"ruff not found at {RUFF} -- see requirements-dev.txt "
            "('.venv/bin/python3 -m pip install ruff')"
        )
    result = subprocess.run(
        [str(RUFF), "check", "--output-format=concise", "."],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
