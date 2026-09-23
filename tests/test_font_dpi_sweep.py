"""Re-runs the `font_sensitive` wrap/fit checks at a range of font DPIs.

testkit/qt_window.ensure_qapp() pins the test font and DPI so the default
tier's outcome does not depend on the machine. This sweep keeps that pin from
hiding real-world font variety: 120 is Windows 125% scaling, 144 is 150%.

QT_FONT_DPI is read once, at QApplication startup, and the process holds one
QApplication, so each DPI runs in its own subprocess. This module is not
itself `font_sensitive`, so the subprocess cannot recurse into it.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from testkit.qt_window import FONT_DPI_OVERRIDE_ENV, PYQT5_AVAILABLE

ROOT = Path(__file__).resolve().parent.parent

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_SWEPT = (72, 96, 100, 110, 120, 144, 168)
_TIMEOUT_S = 600


@pytest.fixture(scope="module")
def sweep_runs():
    """Starts every DPI's subprocess at once, so the sweep costs about one
    run's wall time instead of seven; each case then waits on its own."""
    procs = {}
    for dpi in _SWEPT:
        env = dict(os.environ)
        env[FONT_DPI_OVERRIDE_ENV] = str(dpi)
        procs[dpi] = subprocess.Popen(
            [sys.executable, "-m", "pytest", "-m", "font_sensitive", "-q", "-p", "no:cacheprovider", "tests"],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    yield procs
    for proc in procs.values():
        if proc.poll() is None:
            proc.kill()
            proc.communicate()


@pytest.mark.parametrize("dpi", _SWEPT)
def test_font_sensitive_checks_pass_at(dpi: int, sweep_runs) -> None:
    proc = sweep_runs[dpi]
    output, _ = proc.communicate(timeout=_TIMEOUT_S)
    # Exit 5 is "nothing collected": the marked set vanished, which is a failure too.
    failed = re.findall(r"^FAILED (\S+)", output, flags=re.MULTILINE)
    assert proc.returncode == 0, f"at {dpi} DPI: {failed or 'no FAILED lines'}\n{output[-4000:]}"
