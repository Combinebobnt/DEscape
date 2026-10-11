"""TASK-031.61 A1: the UnitEditModel is built at idle after a load
(ViewerWindow._schedule_unit_model_prebuild), so the first unit edit doesn't
pay the scan inside its stroke end. Every ViewerWindow here ends with
edit_history.mark_saved() before close()."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

UNITS_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"
FOREST_OAK = 10


def _spy_builds(monkeypatch, fail: Exception | None = None) -> list:
    """Counts UnitEditModel constructions (the scenario each one was given)."""
    import descape.viewer as viewer_module

    real = viewer_module.UnitEditModel
    built = []

    def spy(scenario):
        built.append(scenario)
        if fail is not None:
            raise fail
        return real(scenario)

    monkeypatch.setattr(viewer_module, "UnitEditModel", spy)
    return built


def _pump(seconds: float = 0.3) -> None:
    from PyQt5.QtWidgets import QApplication

    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        QApplication.processEvents()
        time.sleep(0.005)


def _loaded(tmp_path, monkeypatch, fail: Exception | None = None):
    """A shown window with a tmp copy of the fixture loaded, before any idle turn."""
    path = tmp_path / "units.aoe2scenario"
    shutil.copyfile(UNITS_FIXTURE, path)
    window = conftest.shown_window()
    built = _spy_builds(monkeypatch, fail)
    window.load_scenario(path)
    return window, built, path


def test_a_shown_load_builds_the_model_at_idle_not_inside_the_load(tmp_path, monkeypatch) -> None:
    from descape import debug_log, perf_trace

    perf_trace.enable(True)
    debug_log.clear()
    window, built, _path = _loaded(tmp_path, monkeypatch)
    try:
        assert built == [] and window.unit_edits is None, "built inside the load"
        _pump()
        assert built == [window.scenario]
        assert window.unit_edits is not None and window.unit_edits.loaded is window.scenario
        perf_trace.flush_idle()
        assert "unit_model_prebuild" in debug_log.get_log_text()
        assert window._ensure_unit_edits() is window.unit_edits
        assert len(built) == 1
    finally:
        perf_trace.enable(False)
        conftest.close_window(window)


def test_the_idle_build_waits_while_a_mouse_button_is_held(tmp_path, monkeypatch) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    held = {"value": True}
    monkeypatch.setattr(
        QApplication, "mouseButtons", staticmethod(lambda: Qt.LeftButton if held["value"] else Qt.NoButton)
    )
    window, built, _path = _loaded(tmp_path, monkeypatch)
    try:
        _pump()
        assert built == [], "built mid-drag"
        held["value"] = False
        _pump()
        assert built == [window.scenario]
    finally:
        conftest.close_window(window)


def test_a_hidden_window_keeps_the_lazy_build(monkeypatch) -> None:
    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    built = _spy_builds(monkeypatch)
    window = ViewerWindow()
    window.load_scenario(UNITS_FIXTURE)
    try:
        _pump()
        assert built == [] and window.unit_edits is None
    finally:
        conftest.close_window(window)


def test_a_browse_only_save_after_the_idle_build_is_byte_identical(tmp_path, monkeypatch) -> None:
    import descape.viewer as viewer_module

    window, _built, path = _loaded(tmp_path, monkeypatch)
    try:
        _pump()
        assert window.unit_edits is not None
        dest = tmp_path / "saved.aoe2scenario"
        monkeypatch.setattr(viewer_module.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(dest), "")))
        window.save_as()
        assert dest.read_bytes() == path.read_bytes()
    finally:
        conftest.close_window(window)


def test_a_read_only_file_is_silent_at_idle_and_warns_on_the_first_draw(tmp_path, monkeypatch) -> None:
    from descape.unit_model import UnitEditsUnavailableError

    window, built, _path = _loaded(tmp_path, monkeypatch, fail=UnitEditsUnavailableError("read-only"))
    warned = []
    window._warn = lambda title, text: warned.append(title)
    try:
        _pump()
        assert len(built) == 1, "the idle build never ran"
        assert warned == [] and window.unit_edits is None
        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected("draw")
        window.terrain_panel.set_terrain(FOREST_OAK)
        window.paint_trees_check.setChecked(True)
        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(5, 5, 0)
        window.on_edit_stroke_end()
        assert warned == ["Units are read-only"]
        assert len(built) == 2
    finally:
        conftest.close_window(window)


def test_an_unexpected_build_error_at_idle_is_logged_and_left_to_the_first_edit(tmp_path, monkeypatch) -> None:
    """Never raised out of the idle timer into sys.excepthook (the crash
    dialog), browse-only sessions included; the sync path raises it in context."""
    import sys

    from descape import debug_log

    hooked = []
    monkeypatch.setattr(sys, "excepthook", lambda *exc: hooked.append(exc[1]))
    debug_log.clear()
    window, built, _path = _loaded(tmp_path, monkeypatch, fail=RuntimeError("boom"))
    try:
        _pump()
        assert len(built) == 1, "the idle build never ran"
        assert hooked == [] and window.unit_edits is None
        assert "Unit model prebuild failed: RuntimeError: boom" in debug_log.get_log_text()
        with pytest.raises(RuntimeError, match="boom"):
            window._ensure_unit_edits()
    finally:
        conftest.close_window(window)


def test_an_edit_before_idle_builds_once(tmp_path, monkeypatch) -> None:
    window, built, _path = _loaded(tmp_path, monkeypatch)
    try:
        model = window._ensure_unit_edits()
        _pump()
        assert len(built) == 1
        assert window.unit_edits is model
    finally:
        conftest.close_window(window)


def test_a_reload_or_close_before_idle_drops_the_build(tmp_path, monkeypatch) -> None:
    window, built, path = _loaded(tmp_path, monkeypatch)
    try:
        window.load_scenario(path)
        _pump()
        assert built == [window.scenario], "the first document's build ran"
        window.load_scenario(path)
        window.close_scenario()
        _pump()
        assert len(built) == 1
        assert window.unit_edits is None
    finally:
        conftest.close_window(window)


def test_a_resize_keeps_its_remapped_model_through_idle(tmp_path, monkeypatch) -> None:
    from PyQt5.QtWidgets import QMessageBox
    from test_resize_map import _fake_dialog, _questions

    import descape.viewer as viewer_module
    from descape.scenario_io import load_map_and_units

    window, built, _path = _loaded(tmp_path, monkeypatch)
    try:
        _pump()
        assert len(built) == 1
        _fake_dialog(monkeypatch, 109, "RIGHT")
        _questions(monkeypatch, QMessageBox.Yes)
        window.resize_map()
        model = window.unit_edits
        assert model is not None
        _pump()
        assert len(built) == 1, "an idle build ran over the resize's own model"
        assert window.unit_edits is model, "the idle build replaced the resize's model"
        dest = tmp_path / "resized.aoe2scenario"
        monkeypatch.setattr(viewer_module.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(dest), "")))
        window.save_as()
        reloaded = load_map_and_units(dest)
        positions = {u.reference_id: (u.x, u.y) for us in reloaded.unit_manager.units for u in us}
        assert positions == {201: (0.5, 4.5), 300: (9.5, 14.5), 301: (10.5, 14.5)}
    finally:
        conftest.close_window(window)


def test_no_build_fires_after_the_window_closes(tmp_path, monkeypatch) -> None:
    window, built, _path = _loaded(tmp_path, monkeypatch)
    conftest.close_window(window)
    _pump()
    assert built == []
