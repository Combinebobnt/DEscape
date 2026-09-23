"""Collection must not depend on this machine's data.

v0.7's first tag build failed CI because a module-level `parametrize` read
the machine-only Workshop corpus and errored the whole collection (fixed in
872e04b). A static lint rule against filesystem access at import would also
flag that fix's own `is_dir()` guard and conftest's corpus glob, so this
tests the actual failure instead: collect the suite with the corpus, the
Workshop folder and the DE install all hidden, and require a clean exit.

WORKSHOP_DIR is under `~`, so a fake HOME hides it. The DE install comes only
from AOE2DE_INSTALL_PATH or config.yaml under the platformdirs config dir, so
that var is removed and XDG_CONFIG_HOME faked too.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sanitized_env(tmp_path: Path) -> dict[str, str]:
    home = tmp_path / "home"
    config = tmp_path / "config"
    home.mkdir()
    config.mkdir()
    env = dict(os.environ)
    env.pop("AOE2DE_INSTALL_PATH", None)
    env["HOME"] = str(home)
    env["XDG_CONFIG_HOME"] = str(config)
    return env


def test_collection_succeeds_without_machine_data(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest", "--collect-only", "-q",
            # The = form: a bare path argument would become pytest's rootdir.
            "-p", "no:cacheprovider", f"--scenario-dir={empty}",
        ],
        cwd=ROOT,
        env=sanitized_env(tmp_path),
        check=False,  # the asserts below are the check, and want the output
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "ERROR collecting" not in output, output
