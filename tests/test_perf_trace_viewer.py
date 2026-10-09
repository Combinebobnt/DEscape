"""Perf Trace's drag bracketing and idle `perf view` flush, wired through a
real MapView: a stroke's press and release reach begin_drag()/end_drag(),
and a pan with no drag prints its own line once the canvas goes quiet
(viewer_canvas's idle timer). The logic itself is pinned Qt-free in
tests/test_perf_trace.py."""

from __future__ import annotations

import gc
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from descape import debug_log, perf_trace

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def _spin_until(predicate, timeout_s: float) -> bool:
    from PyQt5.QtWidgets import QApplication

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def traced_window():
    from PyQt5.QtWidgets import QApplication

    window = conftest.terrain_edit_window()
    window._on_tool_selected("draw")
    window.paint_trees_check.setChecked(False)
    window.paint_eye_candy_check.setChecked(False)
    window.resize(600, 600)
    window.show()
    QApplication.processEvents()
    perf_trace.enable(True)
    debug_log.clear()
    yield window
    perf_trace.enable(False)
    perf_trace.reset()
    debug_log.clear()
    conftest.close_window(window)


def test_a_real_stroke_is_bracketed_and_drops_hover_phases(traced_window):
    from PyQt5.QtCore import QEvent, Qt

    map_view = traced_window.map_view
    pos = conftest.polygon_viewport_pos(map_view, 30, 30)
    map_view.mouseMoveEvent(conftest.mouse_event(QEvent.MouseMove, pos, Qt.NoButton, Qt.NoButton))
    assert perf_trace._current_step, "hover no longer records pick -- this test proves nothing"

    map_view.mousePressEvent(conftest.mouse_event(QEvent.MouseButtonPress, pos, Qt.LeftButton, Qt.LeftButton))
    assert perf_trace._drag_active
    assert "pick" not in perf_trace._phase_order or perf_trace._step_totals, "hover pick leaked into the drag"
    map_view.mouseReleaseEvent(conftest.mouse_event(QEvent.MouseButtonRelease, pos, Qt.LeftButton, Qt.NoButton))

    assert not perf_trace._drag_active
    assert "perf drag paint-terrain:" in debug_log.get_log_text()


def test_a_viewport_drag_line_names_the_platform_and_counts_coalesced_moves(traced_window):
    """GH #179: moves sent to the viewport go through MapView's coalescer, so
    the drag line carries the input counts, and the header the platform and
    steps per repaint call."""
    import re

    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtWidgets import QApplication

    map_view = traced_window.map_view
    viewport = map_view.viewport()

    def send(kind, tile, button, buttons):
        pos = conftest.polygon_viewport_pos(map_view, *tile)
        QApplication.sendEvent(viewport, conftest.mouse_event(kind, pos, button, buttons))

    send(QEvent.MouseButtonPress, (20, 20), Qt.LeftButton, Qt.LeftButton)
    for x in range(21, 29):
        send(QEvent.MouseMove, (x, 20), Qt.NoButton, Qt.LeftButton)
    assert _spin_until(lambda: map_view._move_stash is None and perf_trace._repaint_durations, timeout_s=5.0)
    send(QEvent.MouseButtonRelease, (28, 20), Qt.LeftButton, Qt.NoButton)

    text = debug_log.get_log_text()
    drag = text[text.index("perf drag paint-terrain:") :]
    header = drag.splitlines()[0]
    platform = re.escape(QApplication.platformName())
    assert re.search(rf", composite \w+, platform {platform}, \d+\.\d\d steps/repaint$", header), header
    match = re.search(r"\| input: moves (\d+), handled (\d+)", drag)
    assert match, drag
    assert int(match.group(1)) == 8 and 1 <= int(match.group(2)) < 8, drag


def test_a_trees_on_draw_stroke_end_splits_unit_plan_into_its_sub_phases(traced_window):
    """TASK-031.55: unit_plan's scan / plan / splices / begin / model / commit
    split prints on the stroke's end line beside unit_plan itself."""
    import re

    from PyQt5.QtCore import QEvent, Qt

    window = traced_window
    window.terrain_panel.set_terrain(10)  # FOREST_OAK, density 1000: every tile plants a tree
    window.paint_trees_check.setChecked(True)
    map_view = window.map_view
    start = conftest.polygon_viewport_pos(map_view, 30, 30)
    end = conftest.polygon_viewport_pos(map_view, 32, 30)
    map_view.mousePressEvent(conftest.mouse_event(QEvent.MouseButtonPress, start, Qt.LeftButton, Qt.LeftButton))
    map_view.mouseMoveEvent(conftest.mouse_event(QEvent.MouseMove, end, Qt.NoButton, Qt.LeftButton))
    map_view.mouseReleaseEvent(conftest.mouse_event(QEvent.MouseButtonRelease, end, Qt.LeftButton, Qt.NoButton))

    assert any(int(u.x) == 30 and int(u.y) == 30 for u in window.scenario.unit_manager.units[0]), "no tree planted"
    text = debug_log.get_log_text()
    drag = text[text.index("perf drag paint-terrain:") :]
    end_line = next(line for line in drag.splitlines() if "| end:" in line)
    names = re.findall(r"(unit_plan(?:\.\w+)?) [\d.]+", end_line)
    assert names[-1] == "unit_plan", end_line
    assert set(names[:-1]) == {f"unit_plan.{n}" for n in ("scan", "plan", "splices", "begin", "model", "commit")}, end_line


def test_a_real_set_elevation_stroke_traces_its_press_steps_and_release(traced_window):
    """set-elevation-untimed-stalls plan Step 2: the snapshot, the mutation and
    the commit are phases on the drag line, and the spans give it a wall."""
    import re

    from PyQt5.QtCore import QEvent, Qt

    window = traced_window
    window._on_tool_selected("set_level")
    window.elevation_level_spin.setValue(3)
    mm = window.scenario.map_manager
    assert mm.get_tile(30, 30).elevation != 3, "the stroke would write nothing -- vacuous"
    map_view = window.map_view
    start = conftest.polygon_viewport_pos(map_view, 30, 30)
    end = conftest.polygon_viewport_pos(map_view, 32, 30)
    map_view.mousePressEvent(conftest.mouse_event(QEvent.MouseButtonPress, start, Qt.LeftButton, Qt.LeftButton))
    map_view.mouseMoveEvent(conftest.mouse_event(QEvent.MouseMove, end, Qt.NoButton, Qt.LeftButton))
    map_view.mouseReleaseEvent(conftest.mouse_event(QEvent.MouseButtonRelease, end, Qt.LeftButton, Qt.NoButton))

    assert mm.get_tile(30, 30).elevation == 3
    text = debug_log.get_log_text()
    assert "perf drag set-elevation: 2 steps" in text, text
    drag = text[text.index("perf drag set-elevation:") :]
    # Set elevation takes a brush, so MapView names it (TASK-031.38).
    assert re.search(r", wall \d+ms, untimed \d+ms, brush 1 square, composite", drag), drag
    for name in ("stroke_snapshot", "stroke_mutate", "stroke_commit", "edit_actions", "gc_resume"):
        assert f"{name} " in drag, f"{name} missing from the drag line:\n{drag}"
    assert "span" not in drag and perf_trace._span_stack == []


def test_a_pan_with_no_drag_prints_its_own_view_line(traced_window):
    map_view = traced_window.map_view
    debug_log.clear()
    perf_trace._repaint_durations = []
    map_view.scale(1.25, 1.25)
    assert _spin_until(lambda: "perf view: repaint:" in debug_log.get_log_text(), timeout_s=5.0)
    assert "perf drag" not in debug_log.get_log_text()


# --- perf op / level events through the real viewer (perf-trace-coverage plan) --

UNITS_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"
_VILLAGER = 83


def _op_lines(prefix: str) -> list[str]:
    return [line for line in debug_log.get_log_text().splitlines() if f"perf op {prefix}" in line]


@pytest.fixture
def units_window():
    """units_120x120 in Flat, Units mode, never shown: every edit is synchronous."""
    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    window.load_scenario(UNITS_FIXTURE)
    assert window.scenario is not None, "fixture failed to load"
    window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText("Flat")
    window.mode_combo.setCurrentText("Units")
    perf_trace.enable(True)
    perf_trace.reset()
    debug_log.clear()
    yield window
    perf_trace.enable(False)
    perf_trace.reset()
    debug_log.clear()
    conftest.close_window(window)


def test_opening_the_debug_log_writes_the_pending_op_line_first(units_window, monkeypatch):
    from descape import viewer

    seen = []

    class _SnapshotDialog:
        def __init__(self, parent) -> None:
            seen.append(debug_log.get_log_text())

        def exec_(self) -> int:
            return 0

    monkeypatch.setattr(viewer, "DebugLogDialog", _SnapshotDialog)
    # No processEvents() between the op and the open: the idle flush must not run.
    with perf_trace.op("paste"):
        pass
    units_window._show_debug_log()
    assert len(seen) == 1 and "perf op paste:" in seen[0]


def _place(window, tile) -> None:
    from PyQt5.QtCore import QPointF, Qt

    tp = window.map_view._tile_pixels
    window.units_panel.select_object(_VILLAGER)
    window.units_panel.select_owner(1)
    window.place_unit_action.setChecked(True)
    window.on_unit_place(QPointF(tile[0] * tp + tp // 2, tile[1] * tp + tp // 2), Qt.NoModifier)


def test_the_first_place_unit_line_carries_the_model_build_and_the_second_does_not(units_window):
    window = units_window
    assert window.unit_edits is None, "the model is already built -- vacuous"
    _place(window, (70, 70))
    perf_trace.flush_idle()
    [first] = _op_lines("place-unit")
    assert "unit_model_build" in first and "untimed" in first
    debug_log.clear()
    _place(window, (72, 70))
    perf_trace.flush_idle()
    [second] = _op_lines("place-unit")
    assert "unit_model_build" not in second
    assert "unit_sources" in second or "unit_patch" in second, "the op no longer wraps the edit's tail"


def test_undo_prints_its_own_op_line(units_window):
    window = units_window
    _place(window, (70, 70))
    perf_trace.flush_idle()
    debug_log.clear()
    window.undo()
    perf_trace.flush_idle()
    assert _op_lines("undo:"), debug_log.get_log_text()


def test_a_held_nudge_coalesces_into_one_line(units_window):
    from PyQt5.QtCore import Qt

    window = units_window
    _place(window, (70, 70))
    window.pan_action.setChecked(True)
    perf_trace.flush_idle()
    debug_log.clear()
    for _ in range(5):
        assert window.on_unit_nudge(1, 0, Qt.NoModifier), "the nudge was refused -- vacuous"
    perf_trace.flush_idle()
    assert [line.split("] ", 1)[-1][: len("perf op nudge-unit x5:")] for line in _op_lines("nudge")] == [
        "perf op nudge-unit x5:"
    ]


def test_the_large_fill_dialog_is_its_own_phase_not_untimed(monkeypatch):
    import re

    from PyQt5.QtWidgets import QMessageBox

    import descape.viewer as viewer_module

    monkeypatch.setattr(viewer_module, "TERRAIN_UNIT_CONFIRM_THRESHOLD", 10)

    def slow_yes(*args, **kwargs):
        time.sleep(1.0)  # the user reading the dialog; 0.2 s lost to CI's ~280 ms untimed rest
        return QMessageBox.Yes

    monkeypatch.setattr(viewer_module.QMessageBox, "question", staticmethod(slow_yes))
    window = conftest.terrain_edit_window()
    try:
        window._on_tool_selected("fill")
        window.terrain_panel.set_terrain(10)
        window.paint_trees_check.setChecked(True)
        perf_trace.enable(True)
        perf_trace.reset()
        debug_log.clear()
        window.on_fill(0, 0, 0)
        perf_trace.flush_idle()
        [line] = _op_lines("fill")
        dialog_ms = float(re.search(r"confirm_dialog ([\d.]+)", line).group(1))
        untimed_ms = float(re.search(r"untimed ([\d.]+)", line).group(1))
        assert dialog_ms >= 950, line
        assert untimed_ms < dialog_ms, line
    finally:
        perf_trace.enable(False)
        perf_trace.reset()
        debug_log.clear()
        conftest.close_window(window)


@pytest.mark.parametrize("on", [True, False])
def test_the_stall_watchdog_follows_the_perf_trace_toggle(units_window, on):
    from descape import viewer_canvas

    action = units_window.perf_trace_action
    action.setChecked(not on)
    action.setChecked(on)
    timer = viewer_canvas._stall_timer
    assert (timer is not None and timer.isActive()) is on
    assert (perf_trace._on_gc in gc.callbacks) is on
    action.setChecked(False)
    assert viewer_canvas._stall_timer is None or not viewer_canvas._stall_timer.isActive()
    assert perf_trace._on_gc not in gc.callbacks


def test_the_first_paint_after_a_load_splits_its_repaint_into_composite_and_blit():
    perf_trace.enable(True)
    perf_trace.reset()
    debug_log.clear()
    window = conftest.stepped_window(UNITS_FIXTURE, size=600)
    try:
        [line] = [line for line in debug_log.get_log_text().splitlines() if "perf load: repaint" in line]
        assert " [composite " in line and "; blit " in line and "; other " in line, line
    finally:
        perf_trace.enable(False)
        perf_trace.reset()
        debug_log.clear()
        conftest.close_window(window)


def _switch_mode_line(window, mode: str) -> tuple[float, dict[str, float]]:
    """One mode switch's op line alone: (total ms, phase -> ms without untimed)."""
    perf_trace.flush_pending_op()
    debug_log.clear()
    window.mode_combo.setCurrentText(mode)
    perf_trace.flush_pending_op()
    [line] = _op_lines("mode-switch")
    assert " x2" not in line, line
    words = line.split(" | ")[1].split()
    phases = {name: float(ms) for name, ms in zip(words[::2], words[1::2], strict=True) if name != "untimed"}
    return float(line.split(": ")[1].split("ms")[0]), phases


def test_a_mode_switch_to_units_is_split_into_phases(units_window):
    _switch_mode_line(units_window, "Terrain")
    _total, phases = _switch_mode_line(units_window, "Units")
    for name in ("view_mode", "left_page", "camera_markers", "widen_left", "unit_index", "units_panel", "mode_status"):
        assert name in phases, phases
    assert "panel" not in phases, "the panel phase is for modes with a panel to populate"


def test_a_mode_switch_that_forces_pan_counts_the_nested_tool_switch_once(units_window, monkeypatch):
    """Leaving Units with a Units-only tool forces Pan inside the switch; that
    nested tool-switch op is a phase already, so no phase may wrap it too."""
    real_op = perf_trace.op

    @contextmanager
    def _slow(cm):
        with cm:
            time.sleep(0.03)
            yield

    monkeypatch.setattr(perf_trace, "op", lambda label: _slow(real_op(label)) if label == "tool-switch" else real_op(label))
    units_window.place_unit_action.setChecked(True)
    total, phases = _switch_mode_line(units_window, "Terrain")
    assert phases.get("tool-switch", 0.0) >= 30.0, phases
    assert sum(phases.values()) <= total + 1.0, f"a phase double counts: {phases} vs {total}"


def _wheel_in(view) -> None:
    from PyQt5.QtCore import QPoint, QPointF, Qt
    from PyQt5.QtGui import QWheelEvent

    center = QPointF(view.viewport().rect().center())
    view._end_wheel_gesture()
    view.wheelEvent(
        QWheelEvent(center, center, QPoint(0, 0), QPoint(0, 120), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    )


def test_a_zoom_into_an_unbuilt_level_names_the_build_on_the_view_line(monkeypatch):
    from PyQt5.QtWidgets import QApplication

    from descape import settings

    monkeypatch.setattr(settings, "_preload_zoom_levels", False)
    window = conftest.stepped_window(UNITS_FIXTURE)
    try:
        view = window.map_view
        QApplication.processEvents()
        start = view.viewport_chunk_target()[0]
        perf_trace.enable(True)
        perf_trace.reset()
        debug_log.clear()
        for _ in range(12):
            _wheel_in(view)
            QApplication.processEvents()
            if view.viewport_chunk_target()[0] != start:
                break
        mip = view.viewport_chunk_target()[0]
        assert mip != start, "twelve notches never changed level -- vacuous"
        perf_trace.flush_idle()
        text = debug_log.get_log_text()
        assert f"{mip} build " in text and "(repaint)" in text, text
        assert f"mip {start}->{mip}" in text, text
        assert "perf op zoom" in text
    finally:
        perf_trace.enable(False)
        perf_trace.reset()
        debug_log.clear()
        conftest.close_window(window)


def _texture_install(root: Path, terrain_ids) -> Path:
    """A fake install whose mapped .dds names hold small PNGs (PIL sniffs content)."""
    from PIL import Image

    from descape import asset_source

    tex_dir = root / asset_source.TERRAIN_TEXTURE_SUBPATH
    tex_dir.mkdir(parents=True, exist_ok=True)
    for tid in terrain_ids:
        Image.new("RGB", (16, 16), (90, 140, 60)).save(tex_dir / asset_source._terrain_texture_map()[tid], format="PNG")
    return root


def test_the_first_paint_after_a_load_names_its_terrain_texture_loads(tmp_path):
    import re

    from descape import asset_source
    from descape.scenario_io import load_map_and_units

    ids = {tile.terrain_id for tile in load_map_and_units(UNITS_FIXTURE).map_manager.terrain}
    asset_source.set_install_path_override(_texture_install(tmp_path, ids))
    perf_trace.enable(True)
    perf_trace.reset()
    debug_log.clear()
    window = conftest.stepped_window(UNITS_FIXTURE, size=600)
    try:
        # A prefetched file counts on the line its pool load finished in: the load op's or the first paint's.
        lines = [line for line in debug_log.get_log_text().splitlines() if "perf load: repaint" in line or "perf op load" in line]
        assert len(lines) == 2, lines
        assert sum(int(n) for n in re.findall(r"textures \+(\d+) files", " ".join(lines))) == len(ids), lines
    finally:
        perf_trace.enable(False)
        perf_trace.reset()
        debug_log.clear()
        conftest.close_window(window)
        asset_source.set_install_path_override(None)
