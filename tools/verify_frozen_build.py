#!/usr/bin/env python3
"""
Drives the PyInstaller onedir bundle's --self-check and --smoke modes
(packaging/entry_frozen.py) and reports their results. Build the bundle
first:

  .venv/bin/python3 -m PyInstaller packaging/descape.spec --noconfirm
  .venv/bin/python3 tools/verify_frozen_build.py

--self-check exists because missing bundled data is the dominant freeze bug
class, and it fails silently otherwise. --smoke goes further and actually
constructs a QApplication + ViewerWindow, catching a missing/unloadable Qt
platform plugin that --self-check can't see. Both run headless: --self-check
touches no Qt object at all, and --smoke forces QT_QPA_PLATFORM=offscreen.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _bundle_executable() -> Path:
    if sys.platform == "win32":
        return ROOT / "dist" / "DEscape" / "DEscape.exe"
    return ROOT / "dist" / "DEscape" / "DEscape"


def _run_mode(exe: Path, mode: str) -> int:
    print(f"=== {mode} ===")
    result = subprocess.run([str(exe), mode], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(f"verify_frozen_build: {mode} FAILED (exit {result.returncode})")
    else:
        print(f"verify_frozen_build: {mode} PASSED")
    return result.returncode


def main() -> int:
    exe = _bundle_executable()
    if not exe.is_file():
        print(f"No frozen build found at {exe} -- build it first:")
        print("  .venv/bin/python3 -m PyInstaller packaging/descape.spec --noconfirm")
        return 1

    # Run both even if --self-check fails, same report-every-failure stance
    # as _self_check() itself, so a red CI step shows both results at once.
    self_check_rc = _run_mode(exe, "--self-check")
    smoke_rc = _run_mode(exe, "--smoke")

    if self_check_rc != 0 or smoke_rc != 0:
        return 1

    total_size = sum(f.stat().st_size for f in exe.parent.rglob("*") if f.is_file())
    print(f"Bundle size: {total_size / (1024 * 1024):.1f} MiB ({exe.parent})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
