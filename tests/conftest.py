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

import contextlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest

from testkit import qt_capture, qt_window, settings_isolation

ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = ROOT / "tools"

# Re-exported so the 85 existing `conftest.ensure_qapp()` call sites across
# 30 files, and test_lazy_viewport.py's `conftest.PYQT5_AVAILABLE`
# pytestmark, keep working unchanged; same posture as the
# scene_rect_to_array re-export below. The implementation now lives in
# testkit/ so tools/ scripts, which can't import conftest, can share it.
PYQT5_AVAILABLE = qt_window.PYQT5_AVAILABLE
ensure_qapp = qt_window.ensure_qapp


def differing_ranges(before: bytes, after: bytes) -> list[tuple[int, int]]:
    """Maximal [start, end) spans where two equal-length buffers differ.

    Promoted here rather than copied a fifth time -- test_trigger_write_path.py,
    test_options_write_path.py, test_diplomacy_write_path.py, and
    test_player_options_write_path.py each already hold their own private
    copy; new locality tests should import this one instead."""
    assert len(before) == len(after)
    ranges: list[tuple[int, int]] = []
    start = None
    for i, (a, b) in enumerate(zip(before, after, strict=True)):
        if a != b and start is None:
            start = i
        elif a == b and start is not None:
            ranges.append((start, i))
            start = None
    if start is not None:
        ranges.append((start, len(before)))
    return ranges


def symlink_loop(tmp_path: Path) -> Path:
    """loop-a -> loop-b -> loop-a, returning loop-a."""
    a, b = tmp_path / "loop-a", tmp_path / "loop-b"
    a.symlink_to(b)
    b.symlink_to(a)
    return a


def resolve_like_py313(monkeypatch) -> None:
    """Python 3.13+'s non-strict Path.resolve() on a loop: no raise, and a ".."
    applied as text. Lets a loop-refusal test pin the stat check on CI's 3.11."""
    monkeypatch.setattr(Path, "resolve", lambda self, strict=False: Path(os.path.normpath(self.absolute())))


# -- GH #148: a fake install with portraits and stat icons, for the unit cards --

_UNIT_RED = (200, 10, 10, 255)
_BUILDING_GREEN = (10, 200, 10, 255)
_STAT_ICON_FILES = ("hp", "damage", "armor", "range-armor", "range")


def _fake_install(tmp_path):
    """Archer's unit portrait (017), a Castle portrait in buildings/ (007) and,
    as a trap, a unit portrait 007 too; plus the five stat icons at 66 px."""
    from PIL import Image

    ingame = tmp_path / "AoE2DE" / "widgetui" / "textures" / "ingame"
    for folder in ("units", "buildings", "staticons"):
        (ingame / folder).mkdir(parents=True)
    Image.new("RGBA", (32, 32), _UNIT_RED).save(ingame / "units" / "017_50730.DDS", format="DDS")
    Image.new("RGBA", (32, 32), _UNIT_RED).save(ingame / "units" / "007_50730.DDS", format="DDS")
    Image.new("RGBA", (32, 32), _BUILDING_GREEN).save(ingame / "buildings" / "007_castle.DDS", format="DDS")
    for name in _STAT_ICON_FILES:
        Image.new("RGBA", (66, 66), (30, 60, 90, 255)).save(ingame / "staticons" / f"{name}.png")
    return tmp_path / "AoE2DE"


class ManualExecutor:
    """A catalog_thumbs.ThumbLoader executor that holds each submitted decode until the test runs it."""

    def __init__(self):
        self.submitted: list[tuple] = []

    def submit(self, fn, *args):
        from concurrent.futures import Future

        future = Future()
        self.submitted.append((fn, args, future))
        return future

    def run(self, index: int) -> None:
        fn, args, future = self.submitted[index]
        future.set_result(fn(*args))

    def icons(self) -> list[int]:
        return [args[1] for _fn, args, _future in self.submitted]


def settle_thumbs(loader, timeout: float = 5.0) -> None:
    """Pumps events until a catalog_thumbs.ThumbLoader has answered everything
    asked of it: portraits decode on a worker thread (GH #68)."""
    import time

    from PyQt5.QtWidgets import QApplication

    deadline = time.monotonic() + timeout
    QApplication.processEvents()
    while not loader.idle():
        assert time.monotonic() < deadline, "the thumbnail loader never went idle"
        time.sleep(0.002)
        QApplication.processEvents()
    QApplication.processEvents()


@pytest.fixture
def fake_install(tmp_path):
    from descape import asset_source

    asset_source.set_install_path_override(_fake_install(tmp_path))
    yield
    asset_source.set_install_path_override(None)

# Re-exported so the many existing `conftest.scene_rect_to_array(...)` call
# sites keep working; the implementation now lives in testkit/ so tools/
# scripts, which can't import conftest, can share it. See that module for
# why the 1:1 and zoom-parametrized captures stay separate functions.
scene_rect_to_array = qt_capture.scene_rect_to_array


def stepped_window(
    path,
    *,
    elev_step_pct: int | None = None,
    graphics_quality: int | None = None,
    show: bool = True,
    size: int | None = None,
):
    """A real, shown ViewerWindow with `path` loaded in Stepped mode,
    offscreen -- the setup every gui-tier seam test needs, factored out
    because tests/test_seam_line.py's own module docstring names this
    exact gap ("nothing in this file mechanically guards" the geometry
    <-> render link) as the reason that file couldn't close it alone.

    The implementation is testkit.qt_window.stepped_window(); this is the
    pytest edge that turns its ScenarioLoadError into a skip. See that
    module for the settings-module-globals rule, the Stepped-combo no-op,
    and the mark_saved()-before-close() requirement."""
    try:
        return qt_window.stepped_window(
            path,
            elev_step_pct=elev_step_pct,
            graphics_quality=graphics_quality,
            show=show,
            size=size,
        )
    except qt_window.ScenarioLoadError as exc:
        pytest.skip(str(exc))


def shown_window(width: int = 1200, height: int = 800):
    """An empty, shown ViewerWindow (1200x800 unless told otherwise) -- no
    scenario loaded. For tests that drive the window's own chrome (menus,
    undo stack, panels) and open a document themselves, or not at all.
    show() + processEvents() matters for anything reading a splitter or
    other geometry: on an unshown window those assertions test Qt's layout
    fallback, not the code under test."""
    from PyQt5.QtWidgets import QApplication

    from descape.viewer import ViewerWindow

    ensure_qapp()
    window = ViewerWindow()
    window.resize(width, height)
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


def shown_terrain_window(width: int = 1200, height: int = 800):
    """terrain_edit_window(), but shown and processed -- what
    tests/test_toolbar_overflow.py needs and no existing fixture gives it.
    shown_window() has no map (View mode, nothing to paint), and
    terrain_edit_window() is built on blank_window(), documented as never
    shown -- geometry/sizeHint()-driven assertions on an unshown window test
    Qt's layout fallback, not real behaviour. resize() + processEvents()
    twice: once for the show to settle, once for the resize to take."""
    from PyQt5.QtWidgets import QApplication

    window = terrain_edit_window()
    window.show()
    QApplication.processEvents()
    window.resize(width, height)
    QApplication.processEvents()
    return window


def dialog_and_window():
    """An empty ViewerWindow (never shown) and a SettingsDialog built on it."""
    from descape.viewer import SettingsDialog, ViewerWindow

    ensure_qapp()
    window = ViewerWindow()
    return SettingsDialog(window), window


def close_window(window) -> None:
    """Close without the dirty-document modal. Same requirement
    stepped_window() documents: closeEvent() -> _confirm_discard_changes()
    pops a QMessageBox that blocks forever offscreen."""
    window.edit_history.mark_saved()
    window.close()


def menu_items(menu, path: str):
    """(path, action) for every non-separator action under `menu`, submenus included."""
    for action in menu.actions():
        if action.isSeparator():
            continue
        where = f"{path} > {action.text()}"
        yield where, action
        if action.menu() is not None:
            yield from menu_items(action.menu(), where)


@contextlib.contextmanager
def restoring_app_theme():
    """Yields the shared QApplication, then puts back all that viewer_dialogs.
    apply_theme() changes on it: style, app palette, the QPushButton class
    palette and the cached _LIGHT_PALETTE (cleared here, so the test captures
    its own). A theme left applied recolours every later test's widgets on
    that worker.

    Qt has no call that removes a class palette, so the one saved at setup
    (the app palette itself when none was set) is set back explicitly, after
    the app palette: an app palette equal to the live one clears nothing."""
    from PyQt5.QtGui import QPalette
    from PyQt5.QtWidgets import QApplication

    from descape import viewer_dialogs

    ensure_qapp()
    app = QApplication.instance()
    saved_palette = QPalette(app.palette())
    saved_class_palette = QPalette(QApplication.palette("QPushButton"))
    saved_style = app.style().objectName()
    saved_light = viewer_dialogs._LIGHT_PALETTE
    viewer_dialogs._LIGHT_PALETTE = None
    try:
        yield app
    finally:
        viewer_dialogs._LIGHT_PALETTE = saved_light
        app.setStyle(saved_style)
        app.setPalette(saved_palette)
        app.setPalette(saved_class_palette, "QPushButton")


def viewport_pos(map_view, tile_x: int, tile_y: int):
    """Viewport-space centre of tile (x, y), for synthesizing mouse events
    in Flat mode. Stepped/Sloped need the projection, not _tile_pixels --
    see tests/test_ruler_viewer.py's own style-aware version."""
    from PyQt5.QtCore import QPointF

    tile_px = map_view._tile_pixels
    scene_pt = QPointF((tile_x + 0.5) * tile_px, (tile_y + 0.5) * tile_px)
    return QPointF(map_view.mapFromScene(scene_pt))


def polygon_viewport_pos(map_view, tile_x: int, tile_y: int):
    """Viewport-space centre of tile (x, y)'s real footprint polygon, so it is
    correct in every terrain style. A different function from viewport_pos()
    above, which is tile-pixel arithmetic and Flat-only."""
    from PyQt5.QtCore import QPointF

    polygon = map_view._tile_polygon(tile_x, tile_y)
    assert polygon is not None, f"no footprint for ({tile_x}, {tile_y})"
    return QPointF(map_view.mapFromScene(polygon.boundingRect().center()))


def mouse_event(kind, pos, button, buttons):
    """A synthetic QMouseEvent with no keyboard modifiers, for driving
    MapView's own handlers directly. Promoted here from four byte-identical
    private copies (test_region_select.py, test_fill_tool.py,
    test_brush_stroke.py, test_ruler_viewer.py); the 2026-08-28 sweep that
    promoted viewport_pos() above scoped only two of those files and left
    this one behind. Callers that need modifiers construct their own.

    Not paired with viewport_pos() above, and neither is polygon_viewport_pos():
    those are position helpers, this only builds the event."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QMouseEvent

    return QMouseEvent(kind, pos, button, buttons, Qt.NoModifier)


_FONT_REFERENCE_TEXT = "Global Victory: Standard"


def pytest_configure(config) -> None:
    """Fires the font/DPI pin at session start, ahead of any fixture. Not left
    to pytest_report_header: `--no-header` skips that hook."""
    if PYQT5_AVAILABLE:
        ensure_qapp()


def pytest_report_header(config) -> str | None:
    """Records the resolved test font metrics in every run's header, so a CI
    log shows them."""
    if not PYQT5_AVAILABLE:
        return None
    from PyQt5.QtGui import QFontInfo, QFontMetrics
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance()
    info = QFontInfo(app.font())
    dpi = app.primaryScreen().logicalDotsPerInch()
    advance = QFontMetrics(app.font()).horizontalAdvance(_FONT_REFERENCE_TEXT)
    return (
        f"test font: {info.family()} {info.pointSize()}pt, logical DPI {dpi:g}, "
        f"{_FONT_REFERENCE_TEXT!r} = {advance} px"
    )


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
# generating this list. It also carries one old scenario version: the 1.37
# file is the only one corpus-wide that the Number of Players gate refuses,
# which test_verify_player_count_block_across_the_corpus needs.
QUICK_CORPUS_NAMES = frozenset(
    {
        "0_June_Event_Scenario.aoe2scenario",  # 200x200, scenario 1.37
        "2_Joan_coop_1_v0_13.aoe2scenario",  # 144x144
        "C2_ElCid_coop_1_v0_16.aoe2scenario",  # 120x120
        "F7_2_Dos Pilas (648).aoe2scenario",  # 480x480, the stress outlier
    }
)


# tools/gen_test_map.py's output: the quick/stress-tier (`test_map`) `scenario_path`
# param of every per-file test that no longer carries `corpus` (pytest_generate_tests).
TEST_MAP_PATH = ROOT / "tests" / "fixtures" / "test_map_240x240.aoe2scenario"
# The same map opened and saved once in the AoE2:DE 1.59 editor; never regenerated.
TEST_MAP_RESAVE_PATH = ROOT / "tests" / "fixtures" / "test_map_game_resave.aoe2scenario"


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
    skip instead of just vanishing from the run.

    A test that still carries `corpus` (function, class or module mark) gets
    only that. Any other test runs first on the generated test map (id
    `map240`) and its game resave (id `test_map_game_resave`), both marked
    `test_map`, then on each corpus file with `corpus` on that param alone,
    so the default tier is unchanged, `-m corpus` runs the files as before
    and run_corpus_quick/stress add the map items."""
    if "scenario_path" not in metafunc.fixturenames:
        return
    files = _corpus_files(metafunc.config)
    reason = f"no .aoe2scenario files in {_corpus_description(metafunc.config)}"
    if metafunc.definition.get_closest_marker("corpus"):
        if not files:
            metafunc.parametrize(
                "scenario_path", [pytest.param(None, marks=pytest.mark.skip(reason=reason), id="no-corpus")]
            )
            return
        metafunc.parametrize("scenario_path", files, ids=[f.name for f in files])
        return
    params = [
        pytest.param(TEST_MAP_PATH, marks=pytest.mark.test_map, id="map240"),
        pytest.param(TEST_MAP_RESAVE_PATH, marks=pytest.mark.test_map, id="test_map_game_resave"),
    ]
    if files:
        params += [pytest.param(f, marks=pytest.mark.corpus, id=f.name) for f in files]
    else:
        params.append(pytest.param(None, marks=[pytest.mark.corpus, pytest.mark.skip(reason=reason)], id="no-corpus"))
    metafunc.parametrize("scenario_path", params)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items) -> None:
    """Marks `test_map` on any item parametrized with a test map file (a
    tests/fixtures/ glob, say), before -m deselects, so it stays out of the
    default tier like the map items pytest_generate_tests makes."""
    names = {TEST_MAP_PATH.name, TEST_MAP_RESAVE_PATH.name}
    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec and any(isinstance(v, Path) and v.name in names for v in callspec.params.values()):
            item.add_marker(pytest.mark.test_map)


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


@pytest.fixture(autouse=True)
def _no_perf_trace_hooks_left():
    """A traced ViewerWindow installs perf_trace's gc hook and the stall
    watchdog; neither may outlive the test that turned them on."""
    yield
    from descape import perf_trace

    perf_trace.set_gc_hook(False)
    if "descape.viewer_canvas" in sys.modules:
        sys.modules["descape.viewer_canvas"].set_stall_watchdog(False)


@pytest.fixture(autouse=True)
def _aoe2_dataset():
    """GH #67: every test starts and ends with the AoE2 game data active."""
    from descape import game_dataset

    game_dataset.set_active(game_dataset.AOE2)
    yield
    game_dataset.set_active(game_dataset.AOE2)


@pytest.fixture(autouse=True)
def _no_gc_hold_left():
    """A warm started under the shared QApplication engages descape.gc_hold;
    its raised threshold2 would otherwise stop full collections for the rest
    of the worker's session."""
    yield
    if "descape.gc_hold" in sys.modules:
        sys.modules["descape.gc_hold"].reset()


@pytest.fixture(autouse=True)
def _no_mouse_button_left_held():
    """QApplication.mouseButtons() is process-global, and every warm tick backs
    off while it reads a held button (_IdleTimerDriver._held_backoff). On Qt
    5.15 QTest.mouseDClick() sets the button and sends no closing release, so
    one double-click test stalled every later warm on its xdist worker
    (TASK-005). Fails the test that left a button held, after releasing it.

    Reads QGuiApplication's, which no test monkeypatches (several patch
    QApplication.mouseButtons and may not have undone it yet here)."""
    yield
    widgets = sys.modules.get("PyQt5.QtWidgets")
    if widgets is None or widgets.QApplication.instance() is None:
        return
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QGuiApplication
    from PyQt5.QtTest import QTest

    held = int(QGuiApplication.mouseButtons())
    if not held:
        return
    sink = widgets.QWidget()
    for bit in range(held.bit_length()):
        if held & (1 << bit):
            QTest.mouseRelease(sink, Qt.MouseButton(1 << bit))
    pytest.fail(f"left mouse button(s) {held:#x} held: send the release QTest.mouseDClick() omits")


@pytest.fixture(autouse=True)
def _release_latched_modifiers():
    """A QTest click carrying a modifier leaves QApplication.keyboardModifiers()
    reporting it for the rest of the process, so a later test's ExtendedSelection
    setCurrentItem() acts as a Ctrl- or Shift-click (TASK-403 W1, TASK-406).
    Releases whatever the test left held."""
    yield
    widgets = sys.modules.get("PyQt5.QtWidgets")
    if widgets is None or widgets.QApplication.instance() is None:
        return
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    held = widgets.QApplication.keyboardModifiers()
    if held == Qt.NoModifier:
        return
    sink = widgets.QWidget()
    for modifier, key in (
        (Qt.ControlModifier, Qt.Key_Control),
        (Qt.ShiftModifier, Qt.Key_Shift),
        (Qt.AltModifier, Qt.Key_Alt),
        (Qt.MetaModifier, Qt.Key_Meta),
    ):
        if held & modifier:
            QTest.keyRelease(sink, key)
    sink.deleteLater()
    assert widgets.QApplication.keyboardModifiers() == Qt.NoModifier


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
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
    gets this for free without declaring it.

    The reset list is testkit.settings_isolation.MEMOIZED_GLOBALS, shared with
    the tools/ capture scripts; monkeypatch here rather than its
    isolate_settings(), so every value is restored after the test."""
    import descape.asset_source as asset_source_module
    import descape.settings as settings_module

    fake_config_path = tmp_path / "config.yaml"
    # asset_source.py owns the original CONFIG_PATH and writes it too (its
    # own save_install_path_to_config) -- patched here as well so nothing in
    # this suite can ever reach the real file, even though nothing currently
    # exercises that particular call path.
    monkeypatch.setattr(asset_source_module, "CONFIG_PATH", fake_config_path)
    monkeypatch.setattr(settings_module, "CONFIG_PATH", fake_config_path)
    for name in settings_isolation.MEMOIZED_GLOBALS:
        monkeypatch.setattr(settings_module, name, None)
    # Settings > Mods lists content_roots(), so a developer's AOE2DE_PROFILE_PATH must not reach it (GH #126).
    # setenv first so teardown restores "unset" even after testkit.isolate_settings() sets it (TASK-378).
    monkeypatch.setenv("AOE2DE_PROFILE_PATH", "")
    monkeypatch.delenv("AOE2DE_PROFILE_PATH", raising=False)
    # Mod art (TASK-378) is module state the viewer switches on; every test starts with it off.
    from descape import mod_graphics

    mod_graphics.reset()
    # So is x2 art (GH #241): a window at Enhanced turns it on.
    from descape import unit_sprites

    unit_sprites.set_hi_res(False)
    # A window turns it on, and with AOE2DE_INSTALL_PATH set the profile autodetects to this
    # machine's own mods; only a test that names a profile in AOE2DE_PROFILE_PATH gets any.
    real_roots = mod_graphics._candidate_roots
    monkeypatch.setattr(
        mod_graphics, "_candidate_roots", lambda: real_roots() if os.environ.get("AOE2DE_PROFILE_PATH") else []
    )
    # pytest-qt processes events after this fixture's teardown, so a live warm tick can cache the
    # real config's install between tests; clear it here, not only on teardown.
    asset_source_module.clear_install_caches()
    # The average colour is derived from get_terrain_texture_array(), which many
    # tests fake: a cached average must not outlive the fake that produced it.
    average = asset_source_module.get_terrain_average_color  # held: a test may monkeypatch the name
    average.cache_clear()
    yield
    average.cache_clear()


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


# Under xdist the controller never collects, so each worker reports its deselected count back through workeroutput.
_WORKER_DESELECTED: list[int] = []


def pytest_deselected(items) -> None:
    if items and hasattr(items[0].config, "workeroutput"):
        output = items[0].config.workeroutput
        output["deselected"] = output.get("deselected", 0) + len(items)


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error) -> None:
    _WORKER_DESELECTED.append(getattr(node, "workeroutput", {}).get("deselected", 0))


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    """'Silent skips must not read as green' (see the Tiers section this
    implements): one line naming how much of the run was deselected or
    skipped, so an empty examples/ or a default `-m` run doesn't look
    indistinguishable from full coverage."""
    # Every worker collects and deselects the same full set, so take one count, not the sum.
    deselected = len(terminalreporter.stats.get("deselected", [])) or max(_WORKER_DESELECTED, default=0)
    skipped = len(terminalreporter.stats.get("skipped", []))
    if deselected or skipped:
        terminalreporter.write_line(
            f"tiers: {deselected} deselected (corpus/slow/test_map -- opt in with -m), "
            f"{skipped} skipped this run (see reasons above; not the same as passing)"
        )
