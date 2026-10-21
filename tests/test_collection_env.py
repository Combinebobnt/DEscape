"""Collection must not depend on this machine's data.

v0.7's first tag build failed CI because a module-level `parametrize` read
the machine-only Workshop corpus and errored the whole collection (fixed in
872e04b). A static lint rule against filesystem access at import would also
flag that fix's own `is_dir()` guard and conftest's corpus glob, so this
tests the actual failure instead: collect the suite with the corpus, the
Workshop folder and the DE install all hidden, and require a clean exit.

WORKSHOP_DIR is under `~`, so a fake HOME (USERPROFILE on Windows) hides it.
The DE install comes only from AOE2DE_INSTALL_PATH or config.yaml, so that
var is removed and the child points CONFIG_PATH into tmp_path before
anything imports it. XDG_CONFIG_HOME alone would not hide config.yaml on
Windows, where platformdirs ignores it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The same CONFIG_PATH override as test_theme_file.py's _STARTUP_PROBE, then pytest in-process.
_ISOLATED_PYTEST = """
import sys
from pathlib import Path
config = Path(sys.argv[1])
import descape.asset_source as asset_source
asset_source.CONFIG_PATH = config
asset_source.LEGACY_CONFIG_PATH = config.parent / "no-legacy-config.yaml"
import descape.settings as settings
settings.CONFIG_PATH = config
import pytest
code = pytest.main(sys.argv[2:])
print("CHILD_CONFIG_PATH", asset_source.CONFIG_PATH)
print("CHILD_SETTINGS_CONFIG_PATH", settings.CONFIG_PATH)
sys.exit(code)
"""


def sanitized_env(tmp_path: Path) -> dict[str, str]:
    home = tmp_path / "home"
    config = tmp_path / "config"
    home.mkdir()
    config.mkdir()
    env = dict(os.environ)
    env.pop("AOE2DE_INSTALL_PATH", None)
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    env["XDG_CONFIG_HOME"] = str(config)
    return env


def test_collection_succeeds_without_machine_data(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    env = sanitized_env(tmp_path)
    config = tmp_path / "config" / "config.yaml"
    result = subprocess.run(
        [
            sys.executable, "-c", _ISOLATED_PYTEST, str(config), "--collect-only", "-q",
            # The = form: a bare path argument would become pytest's rootdir.
            "-p", "no:cacheprovider", "-p", "no:xdist", f"--scenario-dir={empty}",
        ],
        cwd=ROOT,
        env=env,
        check=False,  # the asserts below are the check, and want the output
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "ERROR collecting" not in output, output
    # The isolation itself, checked on every OS rather than trusted.
    assert f"CHILD_CONFIG_PATH {config}\n" in result.stdout, output
    assert f"CHILD_SETTINGS_CONFIG_PATH {config}\n" in result.stdout, output
