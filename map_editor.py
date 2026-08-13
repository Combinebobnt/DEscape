#!/usr/bin/env python3
"""Launches DEscape.

The easiest way to run this is LAUNCH_DESCAPE_LinuxMac.sh (Linux/macOS) or
LAUNCH_DESCAPE_Windows.bat (Windows) in the project root -- those also create the
venv and install dependencies automatically on first run. See README.md.

Can also be run directly from anywhere once the venv exists: `python3
map_editor.py` or `./map_editor.py`, optionally with a scenario path to open
immediately: `./map_editor.py examples/2_Joan_coop_2_v0_15.aoe2scenario`.

Re-execs itself under this project's venv interpreter if it isn't already running
under it, so you don't need to remember to `source .venv/bin/activate` first --
that's what running `python3 descape/viewer.py` directly was missing (that also
happened to break the `descape` package import: running a file inside a package
directly puts that directory on sys.path instead of its parent).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# venv layout differs by OS: Scripts/python.exe on Windows, bin/python3 elsewhere.
VENV_PYTHON = (
    ROOT / ".venv" / "Scripts" / "python.exe"
    if os.name == "nt"
    else ROOT / ".venv" / "bin" / "python3"
)


def _ensure_venv() -> None:
    if not VENV_PYTHON.exists():
        return
    # Comparing sys.executable would be unreliable here: .venv/bin/python3 is a
    # symlink to the system interpreter, so its resolved path is identical whether
    # or not the venv is actually active. sys.prefix is what actually differs --
    # it's set from the *invoked* path (via pyvenv.cfg next to it), not the
    # symlink's target.
    if Path(sys.prefix).resolve() == (ROOT / ".venv").resolve():
        return
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), __file__, *sys.argv[1:]])


if __name__ == "__main__":
    _ensure_venv()
    sys.path.insert(0, str(ROOT))
    from descape.viewer import main

    main()
