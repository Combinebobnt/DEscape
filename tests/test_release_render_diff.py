"""GH #87: the map renders exactly like v0.6 outside the changes made on purpose since.

Drives tools/release_render_diff.py, which renders each file at the `v0.6` tag and
at this checkout in Flat, Stepped and Sloped with sprites off and on, and diffs
the full canvases. Its docstring lists what it neutralizes and why: sub-tile
unit placement, the Stepped contact-shadow terrace band, and the units whose
sprite art changed on purpose.

The controls below are what make a pass mean something. A single retextured
tile fails every comparison, and dropping any one neutralization fails exactly
the comparisons that neutralization exists for, so none of them is masking
more than it has to.

Corpus-marked: it needs the real examples/, an install via AOE2DE_INSTALL_PATH
(conftest hides config.yaml from the suite), and the `v0.6` tag in this clone.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from descape import asset_source

import conftest

pytestmark = pytest.mark.corpus

tool = conftest.load_verify_module("release_render_diff")

CONTROL_FILE = "C2_ElCid_coop_1_v0_16.aoe2scenario"
ALL_CONFIGS = {(style, sprites) for style in tool.STYLES for sprites in (False, True)}


def _require_install_and_tag():
    install = asset_source.get_install_path()
    if install is None:
        pytest.skip(
            "no AoE2:DE install visible: set AOE2DE_INSTALL_PATH to one. The configured "
            "config.yaml install is deliberately hidden by conftest._isolated_settings."
        )
    probe = subprocess.run(
        ["git", "-C", str(conftest.ROOT), "rev-parse", "--verify", "--quiet", f"{tool.BASE_REF}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    if probe.returncode:
        pytest.skip(f"no {tool.BASE_REF} tag in this clone (git fetch --tags)")
    return install


def test_the_render_matches_v06_outside_the_intended_changes(scenario_path, tmp_path):
    install = _require_install_and_tag()
    # run() deletes each pair's canvases once compared; the rest (base/, metas) is freed here, not kept with pytest's basetemps.
    with tempfile.TemporaryDirectory(dir=tmp_path) as work:
        results = tool.run([scenario_path], Path(work), install, pcts=(25, 50, 100, 200))
    assert {(r.job.style, r.job.sprites) for r in results} == ALL_CONFIGS
    failed = [r.line() for r in results if not r.ok]
    assert not failed, "\n".join(failed)


@pytest.mark.parametrize(
    ("inject", "must_fail"),
    [
        ("repaint-tile", ALL_CONFIGS),
        ("no-placement-patch", ALL_CONFIGS),
        ("no-exclusions", {(style, True) for style in tool.STYLES}),
        ("no-mask", {("stepped", False), ("stepped", True)}),
    ],
)
def test_every_control_fails_exactly_where_it_should(inject, must_fail, tmp_path):
    install = _require_install_and_tag()
    scenario = conftest.ROOT / "examples" / CONTROL_FILE
    if not scenario.is_file():
        pytest.skip(f"{CONTROL_FILE} is not in examples/")
    with tempfile.TemporaryDirectory(dir=tmp_path) as work:
        results = tool.run([scenario], Path(work), install, pcts=(50,), inject=inject)
        assert not list(Path(work).rglob("*.npy")), "run() left canvases behind"
    failed = {(r.job.style, r.job.sprites) for r in results if not r.ok}
    assert failed == must_fail, "\n".join(r.line() for r in results)
