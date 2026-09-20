"""Runs ruff as a real pytest gate --
`ruff.toml` selects its ruleset family by family rather than inheriting this
ruff version's defaults, which are not a superset of it: they would drop
E401/E402 and most of E7, weakening the gate. See ruff.toml's own header.
That selection is broad, not a syntax check, so a failure here can be any of
several hundred rules: line length, import order, bugbear, security, datetime
and modernization findings included.

F821 (undefined name) is the original motivation: 343 of the 348 Python files
ruff lints here use `from __future__ import annotations`, which makes an
annotation-only reference to a name a later edit removed invisible at runtime
with no other check catching it. **F821 finding zero is the healthy state for
a tripwire rule, not evidence it is unnecessary.** A cold `--isolated` run on
2026-09-19 found zero F821 and exactly 2 F841, both of them the deliberate
QApplication holds in `tools/verify_copy_paste.py` and
`tools/verify_iso_viewer_pick.py` that ruff.toml per-file-ignores. An earlier
version of this docstring mis-cited that F841 pair as F821.

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
        check=False,  # the assert below is the check, and wants ruff's output
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
