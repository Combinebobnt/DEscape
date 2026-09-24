#!/usr/bin/env python3
"""CI only: proves a from-source launch builds the native kernel. Runs
bootstrap.py's own setup steps (fresh .venv, dependencies, native build)
without launching the app, then asserts the venv's interpreter loads the
native backend. tests/test_bootstrap.py can't: it never spawns a real
subprocess.

Must run before anything else builds the kernel in place, or the build step
finds it current and this proves nothing.

    python tools/ci_bootstrap_native.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bootstrap

ASSERT_NATIVE = (
    "from descape import composite_backend as c; print(c.describe()); "
    "raise SystemExit(c.active_backend() != 'native')"
)


def main() -> int:
    prebuilt = [p for p in (ROOT / "descape").glob("_composite_native*") if p.suffix in (".so", ".pyd")]
    if prebuilt or bootstrap.VENV_DIR.exists():
        print(f"ci_bootstrap_native: not a fresh checkout ({prebuilt or bootstrap.VENV_DIR}), proves nothing")
        return 1
    reporter = bootstrap.ConsoleReporter()
    bootstrap.find_or_create_venv(reporter)
    bootstrap.install_dependencies(reporter)
    if not bootstrap.build_native_kernel(reporter):
        print("ci_bootstrap_native: bootstrap did not build the native kernel")
        return 1
    return subprocess.run([str(bootstrap.VENV_PYTHON), "-c", ASSERT_NATIVE], cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
