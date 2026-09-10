"""Shared harness for the pytest migration of tools/verify_*.py.

Phase 1 scaffolding (see the migration plan): imports each verify_*.py
module and lets tests/test_legacy_adapter.py turn every check_* function
into an individually-named pytest test, with zero edits to the scripts
themselves. Module loading happens lazily, inside each test's own body
(via load_verify_module), not at collection time -- verify_iso_viewer_pick.py
and verify_copy_paste.py import PyQt5 at module level, and a collection-time
import failure there would take the whole session down instead of just the
gui-tier tests that actually need it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from testkit import qt_capture

ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = ROOT / "tools"

PYQT5_AVAILABLE = importlib.util.find_spec("PyQt5") is not None


def differing_ranges(before: bytes, after: bytes) -> list[tuple[int, int]]:
    """Maximal [start, end) spans where two equal-length buffers differ.

    Promoted here rather than copied a fifth time -- test_trigger_write_path.py,
    test_options_write_path.py, test_diplomacy_write_path.py, and
    test_player_options_write_path.py each already hold their own private
    copy; new locality tests should import this one instead."""
    assert len(before) == len(after)
    ranges: list[tuple[int, int]] = []
    start = None
    for i, (a, b) in enumerate(zip(before, after)):
        if a != b and start is None:
            start = i
        elif a == b and start is not None:
            ranges.append((start, i))
            start = None
    if start is not None:
        ranges.append((start, len(before)))
    return ranges

_QAPP = None


def ensure_qapp() -> None:
    """verify_iso_viewer_pick.py/verify_copy_paste.py's own main()s each
    construct exactly one QApplication, held in a module-level variable for
    the rest of the process, before any ViewerWindow() -- their own comment
    explains an unassigned QApplication(...) crashes the first ViewerWindow()
    since PyQt5 doesn't keep it alive via its own singleton registration
    alone. This suite never calls those main()s, so gui-tier tests must do
    the same setup themselves, once, or every ViewerWindow() construction
    aborts the whole interpreter (confirmed: a bare Fatal Python error, not
    a catchable exception -- there is no per-test recovery from skipping
    this)."""
    global _QAPP
    if not PYQT5_AVAILABLE or _QAPP is not None:
        return
    import os

    # Set directly rather than relying on verify_iso_viewer_pick.py/
    # verify_copy_paste.py's own module-level os.environ.setdefault side
    # effect -- this runs before load_verify_module() imports either script,
    # and QApplication() aborts immediately if it tries to reach a real
    # display that doesn't exist in this sandbox.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    _QAPP = QApplication.instance() or QApplication(["pytest"])


# Re-exported so the many existing `conftest.scene_rect_to_array(...)` call
# sites keep working; the implementation now lives in testkit/ so tools/
# scripts, which can't import conftest, can share it. See that module for
# why the 1:1 and zoom-parametrized captures stay separate functions.
scene_rect_to_array = qt_capture.scene_rect_to_array


def stepped_window(path, *, elev_step_pct: int | None = None, graphics_quality: int | None = None):
    """A real, shown ViewerWindow with `path` loaded in Stepped mode,
    offscreen -- the setup every gui-tier seam test needs, factored out
    because tests/test_seam_line.py's own module docstring names this
    exact gap ("nothing in this file mechanically guards" the geometry
    <-> render link) as the reason that file couldn't close it alone.

    elev_step_pct/graphics_quality, if given, are pinned via the settings
    MODULE GLOBALS directly, before ViewerWindow() is constructed (its
    __init__ reads settings.get_window_size(), and _render_current() reads
    both of these) -- not via settings.set_elev_step_pct()/
    set_graphics_quality(). Under _isolated_settings' autouse tmp_path
    redirect either approach is equally safe from touching the real
    config.yaml; module globals are used here only for consistency with
    tests/test_seam_line.py's own test_seam_changes_a_hill_at_every_pct_
    including_200, which already does this and documents why: the real
    setters persist to disk, which "a test has no business doing" even
    when the disk in question is a throwaway.

    Does NOT call terrain_style_combo.setCurrentText("Stepped"). Terrain
    Style already defaults to "stepped" (ViewerWindow._terrain_style), and
    on_terrain_style_changed() early-returns whenever the requested style
    matches the current one -- so every earlier gui test doing this
    (tests/test_lazy_viewport.py, tests/test_mip_viewer.py) was asserting
    the default, not exercising a real switch. A caller that genuinely
    needs to drive a style change should go through "Flat" first.

    Caller must call window.edit_history.mark_saved() before
    window.close() -- see tools/verify_iso_viewer_pick.py's own finally
    block: closeEvent() -> _confirm_discard_changes() pops a modal
    QMessageBox on a dirty document, which blocks forever offscreen with
    nothing to click it."""
    ensure_qapp()
    import descape.settings as settings_module
    from descape.viewer import ViewerWindow
    from PyQt5.QtWidgets import QApplication

    if elev_step_pct is not None:
        settings_module._elev_step_pct = elev_step_pct
    if graphics_quality is not None:
        settings_module._graphics_quality = graphics_quality

    window = ViewerWindow()
    window.load_scenario(path)
    if window.scenario is None:
        window.close()
        pytest.skip(f"{Path(path).name} failed to load")
    window.show()
    QApplication.processEvents()
    return window


def shown_window():
    """An empty, shown, 1200x800 ViewerWindow -- no scenario loaded. For
    tests that drive the window's own chrome (menus, undo stack, panels)
    and open a document themselves, or not at all."""
    from PyQt5.QtWidgets import QApplication

    from descape.viewer import ViewerWindow

    ensure_qapp()
    window = ViewerWindow()
    window.resize(1200, 800)
    window.show()
    QApplication.processEvents()
    return window


def blank_window(load: bool = True):
    """A ViewerWindow with the blank template loaded, never shown. Pass
    load=False for the same window with no document, which is how the
    has-a-map gating on toolbar and menu actions gets tested."""
    from descape.scenario_io import BLANK_TEMPLATE_PATH
    from descape.viewer import ViewerWindow

    ensure_qapp()
    window = ViewerWindow()
    if load:
        window.load_scenario(BLANK_TEMPLATE_PATH)
        assert window.scenario is not None, "blank template failed to load"
    return window


def terrain_edit_window():
    """blank_window() switched into Terrain mode -- the starting point for
    every paint/fill/param test."""
    window = blank_window()
    window.mode_combo.setCurrentText("Terrain")
    return window


def close_window(window) -> None:
    """Close without the dirty-document modal. Same requirement
    stepped_window() documents: closeEvent() -> _confirm_discard_changes()
    pops a QMessageBox that blocks forever offscreen."""
    window.edit_history.mark_saved()
    window.close()


def viewport_pos(map_view, tile_x: int, tile_y: int):
    """Viewport-space centre of tile (x, y), for synthesizing mouse events
    in Flat mode. Stepped/Sloped need the projection, not _tile_pixels --
    see tests/test_ruler_viewer.py's own style-aware version."""
    from PyQt5.QtCore import QPointF

    tile_px = map_view._tile_pixels
    scene_pt = QPointF((tile_x + 0.5) * tile_px, (tile_y + 0.5) * tile_px)
    return QPointF(map_view.mapFromScene(scene_pt))


def pytest_addoption(parser):
    parser.addoption(
        "--scenario-dir",
        action="store",
        default=None,
        help="Directory of .aoe2scenario files for the corpus tier (default: "
        "examples/ next to this repo). verify_copy_paste.py originally kept "
        "this overridable so it still runs from a worktree with no examples/ "
        "of its own; this option carries that forward for the whole suite.",
    )
    parser.addoption(
        "--corpus-full",
        action="store_true",
        default=False,
        help="Run the corpus/slow tier against every file in the scenario "
        "dir (~27 min) instead of the QUICK_CORPUS_NAMES subset (~15 min, "
        "the default -- see run_corpus_stress.sh). The default is meant for "
        "routine after-a-feature verification; pass this before a release "
        "or after touching rendering/write-path internals.",
    )


# A small cross-section of real map-size bands from examples/ (120/144/
# 168/200/220/240/480 -- see README.md's "File > New Map" section for the
# full confirmed-size list), including the 480x480 outlier that dominates
# full-corpus runtime the most (O(pixels) render comparisons). This is the
# DEFAULT corpus/slow-tier scope (no flag needed) -- pass --corpus-full for
# the complete corpus. Deliberately small and hand-picked, not "first N
# files sorted", so it stays a real cross-section rather than an arbitrary
# prefix if examples/ gains or loses files. Update by hand if that
# cross-section stops feeling representative -- there's no formula
# generating this list.
QUICK_CORPUS_NAMES = frozenset(
    {
        "2_Joan_coop_1_v0_13.aoe2scenario",  # 144x144
        "C2_ElCid_coop_1_v0_16.aoe2scenario",  # 120x120
        "F7_2_Dos Pilas (648).aoe2scenario",  # 480x480, the stress outlier
    }
)


def _scenario_dir(config: pytest.Config) -> Path:
    override = config.getoption("--scenario-dir")
    return Path(override) if override else ROOT / "examples"


def _corpus_files(config: pytest.Config) -> list[Path]:
    files = sorted(_scenario_dir(config).glob("*.aoe2scenario"))
    if not config.getoption("--corpus-full"):
        files = [f for f in files if f.name in QUICK_CORPUS_NAMES]
    return files


def _corpus_description(config: pytest.Config) -> str:
    """Human-readable source description for skip messages -- distinguishes
    "the scenario dir itself is empty" from "the default quick filter left
    nothing, because none of QUICK_CORPUS_NAMES exist here", which look
    identical from _corpus_files() alone."""
    base = str(_scenario_dir(config))
    if not config.getoption("--corpus-full"):
        return f"{base} (quick default: looking for {sorted(QUICK_CORPUS_NAMES)} -- pass --corpus-full for the complete corpus)"
    return base


@pytest.fixture
def corpus_files(request: pytest.FixtureRequest) -> list[Path]:
    """The 'files' family: checks that take the whole corpus at once."""
    files = _corpus_files(request.config)
    if not files:
        pytest.skip(f"no .aoe2scenario files in {_corpus_description(request.config)}")
    return files


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """The 'path'/'path_tmp' families: parametrizes any test that declares a
    `scenario_path` argument, one instance per real corpus file. A missing
    corpus produces one explicitly-skipped instance rather than silently
    collecting zero tests, so an empty examples/ still shows up as a named
    skip instead of just vanishing from the run."""
    if "scenario_path" not in metafunc.fixturenames:
        return
    files = _corpus_files(metafunc.config)
    if not files:
        reason = f"no .aoe2scenario files in {_corpus_description(metafunc.config)}"
        metafunc.parametrize(
            "scenario_path", [pytest.param(None, marks=pytest.mark.skip(reason=reason), id="no-corpus")]
        )
        return
    metafunc.parametrize("scenario_path", files, ids=[f.name for f in files])


_MODULE_CACHE: dict[str, object] = {}


def load_verify_module(name: str):
    """Imports tools/<name>.py by file path -- tools/ has no __init__.py, so
    this isn't a package import, matching how each script already resolves
    its own sibling imports via sys.path.insert(0, ROOT) at module level.

    Skips (doesn't error) if the file is already gone: Ordering step 8
    deletes verify scripts one at a time as each clears its own Commit B,
    before step 9 deletes this adapter -- during that window a script this
    adapter still has manifest/test entries for may no longer exist on
    disk. A clean, named skip (surfaced by pytest_terminal_summary below)
    is the point, not a FileNotFoundError spray across every one of that
    script's parametrized instances."""
    if name in _MODULE_CACHE:
        return _MODULE_CACHE[name]
    path = TOOLS_DIR / f"{name}.py"
    if not path.is_file():
        pytest.skip(f"{path.name} has already been migrated and deleted -- see migration_manifest.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Registered in sys.modules before exec: dataclasses' own machinery
    # (SyntheticTile/SyntheticUnit etc. in these scripts) looks up
    # sys.modules[cls.__module__] and crashes on a module that was never
    # registered there.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _MODULE_CACHE[name] = module
    return module


_SETTINGS_MEMOIZED_GLOBALS = (
    "_zoom_centered_on_cursor",
    "_graphics_quality",
    "_dark_mode",
    "_preload_zoom_levels",
    "_elev_step_pct",
    "_window_size",
    "_split_sizes",
    "_log_height",
    "_distance_ticks",
    "_distance_tick_interval",
    "_keybinds",
    "_overlay_colors",
    "_ruler_label_font_px",
    "_distance_tick_font_px",
    "_paint_trees",
    "_paint_eye_candy",
)


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch) -> None:
    """Every test gets its own throwaway config.yaml, never the developer's
    real one. settings.py does `from descape.asset_source import CONFIG_PATH`
    -- a bind-by-value import -- so this must patch descape.settings.CONFIG_PATH
    directly; patching asset_source's has no effect on it. Confirmed live,
    not just from reading: ViewerWindow.closeEvent() unconditionally calls
    settings.set_window_size() on every window close, which without this
    redirect writes straight through to the real config.yaml on every
    gui-tier test (caught after Phase 1's first two corpus runs already did
    this -- harmless only because the offscreen window's size happened to
    match what was last saved). Autouse, so every existing and future test
    gets this for free without declaring it."""
    import descape.asset_source as asset_source_module
    import descape.settings as settings_module

    fake_config_path = tmp_path / "config.yaml"
    # asset_source.py owns the original CONFIG_PATH and writes it too (its
    # own save_install_path_to_config) -- patched here as well so nothing in
    # this suite can ever reach the real file, even though nothing currently
    # exercises that particular call path.
    monkeypatch.setattr(asset_source_module, "CONFIG_PATH", fake_config_path)
    monkeypatch.setattr(settings_module, "CONFIG_PATH", fake_config_path)
    for name in _SETTINGS_MEMOIZED_GLOBALS:
        monkeypatch.setattr(settings_module, name, None)


def run_check(fn, *args) -> None:
    """Adapts a verify_*.py check's (ok, detail) return into a pytest
    outcome: ok is None -> skip, ok is False -> fail, ok is True -> pass. An
    exception from fn propagates as a normal pytest error -- the same
    severity the original scripts' own try/except-to-FAIL already gave it,
    so nothing needs to catch and reformat it here."""
    ok, detail = fn(*args)
    if ok is None:
        pytest.skip(detail)
    if not ok:
        pytest.fail(detail)


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    """'Silent skips must not read as green' (see the Tiers section this
    implements): one line naming how much of the run was deselected or
    skipped, so an empty examples/ or a default `-m` run doesn't look
    indistinguishable from full coverage."""
    deselected = len(terminalreporter.stats.get("deselected", []))
    skipped = len(terminalreporter.stats.get("skipped", []))
    if deselected or skipped:
        terminalreporter.write_line(
            f"tiers: {deselected} deselected (corpus/slow -- opt in with -m), "
            f"{skipped} skipped this run (see reasons above; not the same as passing)"
        )
