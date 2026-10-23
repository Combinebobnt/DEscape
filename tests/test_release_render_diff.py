"""GH #87: the map renders exactly like the last release outside the changes made on purpose since.

Drives tools/release_render_diff.py, which renders each file at the `tool.BASE_REF`
tag and at this checkout in Flat, Stepped and Sloped with sprites off and on, and
diffs the full canvases. Its NEUTRALIZATIONS registry lists what it keeps out of
the comparison and why: one entry per change made on purpose since that release.

The controls below are what make a pass mean something, and they are generated
from that registry, so an entry cannot exist without its control. A single
retextured tile fails every comparison, a render whose Stepped skirts sample
another part of their tile's texture fails both Stepped comparisons, and dropping
any one entry fails exactly the comparisons that entry declares, so none of them
is hiding more than it has to. While a mask entry exists, the Stepped mask's size
is pinned too: it may not grow past its measured fraction of the canvas.

Corpus-marked: it needs the real examples/, an install via AOE2DE_INSTALL_PATH
(conftest hides config.yaml from the suite), and the base tag in this clone.
"""

from __future__ import annotations

import dataclasses
import subprocess
import tempfile
from pathlib import Path

import pytest

from descape import asset_source

import conftest

pytestmark = pytest.mark.corpus

tool = conftest.load_verify_module("release_render_diff")

MATCH_PCTS = (25, 50, 100, 200)
# Files outside the quick corpus, the controls' and the tool's own: no scenario_path cell renders them, so they get a match test.
EXTRA_FILES = sorted(({file for _inject, file, _configs in tool.controls()} - set(tool.QUICK_CORPUS)) | set(tool.EXTRA_MATCH_FILES))


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


def _example(name: str) -> Path:
    scenario = conftest.ROOT / "examples" / name
    if not scenario.is_file():
        pytest.skip(f"{name} is not in examples/")
    return scenario


def _assert_matches(scenario: Path, tmp_path: Path) -> None:
    install = _require_install_and_tag()
    # run() deletes each pair's canvases once compared; the rest (base/, metas) is freed here, not kept with pytest's basetemps.
    with tempfile.TemporaryDirectory(dir=tmp_path) as work:
        results = tool.run([scenario], Path(work), install, pcts=MATCH_PCTS)
    assert {(r.job.style, r.job.sprites) for r in results} == tool.ALL_CONFIGS
    failed = [r.line() for r in results if not r.ok]
    assert not failed, "\n".join(failed)
    if any(n.mask for n in tool.NEUTRALIZATIONS):
        unpinned = [r.line() for r in results if r.job.style == "stepped" and tool.mask_ceiling(scenario.name, r.job.pct) is None]
        assert not unpinned, "no mask ceiling pinned for:\n" + "\n".join(unpinned)


def test_the_render_matches_the_last_release_outside_the_intended_changes(scenario_path, tmp_path):
    _assert_matches(scenario_path, tmp_path)


@pytest.mark.parametrize("name", EXTRA_FILES)
def test_the_files_outside_the_quick_corpus_match_too(name, tmp_path):
    _assert_matches(_example(name), tmp_path)


@pytest.mark.parametrize(("inject", "control_file", "must_fail"), tool.controls(), ids=[c[0] for c in tool.controls()])
def test_every_control_fails_exactly_where_it_should(inject, control_file, must_fail, tmp_path):
    install = _require_install_and_tag()
    scenario = _example(control_file)
    with tempfile.TemporaryDirectory(dir=tmp_path) as work:
        results = tool.run([scenario], Path(work), install, pcts=(50,), inject=inject)
        assert not list(Path(work).rglob("*.npy")), "run() left canvases behind"
    failed = {(r.job.style, r.job.sprites) for r in results if not r.ok}
    assert failed == must_fail, "\n".join(r.line() for r in results)


# A stand-in mask entry: these three tests must hold whatever the registry holds, an empty one after a bump included.
PROBE = tool.Neutralization(
    "probe", "a stand-in entry", since=tool.BASE_REF, control_file=tool.ELCID, must_fail=tool.ALL_CONFIGS, mask=True
)


def test_a_mask_over_its_ceiling_fails_the_stepped_cells(tmp_path, monkeypatch):
    install = _require_install_and_tag()
    scenario = _example(tool.ELCID)
    # Pinned at 1% of the canvas, far under what the terrace mask covers on this file, as if the mask had grown.
    monkeypatch.setattr(tool, "NEUTRALIZATIONS", (PROBE,))
    monkeypatch.setattr(tool, "MASK_MEASURED", {tool.ELCID: {50: 0.01}})
    with tempfile.TemporaryDirectory(dir=tmp_path) as work:
        results = tool.run([scenario], Path(work), install, pcts=(50,), styles=("stepped",))
    assert len(results) == 2
    for result in results:
        assert any("ceiling" in problem for problem in result.problems), result.line()


def test_the_registry_refuses_an_entry_measured_against_another_release(monkeypatch):
    tool.check_registry()
    monkeypatch.setattr(tool, "NEUTRALIZATIONS", (dataclasses.replace(PROBE, since="v0.1"),))
    with pytest.raises(ValueError, match="measured against"):
        tool.check_registry()


def test_the_registry_refuses_a_control_that_must_fail_nowhere(monkeypatch):
    monkeypatch.setattr(tool, "NEUTRALIZATIONS", (dataclasses.replace(PROBE, must_fail=frozenset()),))
    with pytest.raises(ValueError, match="proves nothing"):
        tool.check_registry()
