"""TASK-270: View > Saved Views on a real window. MapView's capture and recall
in every Elevation View (the tests/test_view_state_retention.py window shape),
then the window's keys, menu, refusals and the load, close, resize and Save As
hooks, then the registry rows. The settings store is tests/test_saved_views.py's."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from descape.saved_views import SLOTS, SavedView

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden_blank_120x120.aoe2scenario"
STYLES = ("flat", "flat_iso", "stepped", "sloped")
# A uniform 20x20 block raised LEVELS through the elevation edit path: equal-height neighbours hide no top face.
LOW, BLOCK, LEVELS = 50, 20, 3
MIDDLE = (60, 60)


def _process() -> None:
    from PyQt5.QtWidgets import QApplication

    QApplication.processEvents()
    QApplication.processEvents()


def _set_style(window, style: str) -> None:
    from descape import terrain_style

    # Isometric View first, then the combo, as tests/test_shape_tools_viewer.py does.
    window.iso_action.setChecked(style == "flat_iso")
    window.terrain_style_combo.setCurrentText(terrain_style.label_for("flat" if style.startswith("flat") else style))
    _process()


def _raised_window(style: str):
    from descape.elevation_tools import set_tiles_elevation
    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    window.load_scenario(FIXTURE)
    if window.scenario is None:
        window.close()
        pytest.skip(f"{FIXTURE.name} failed to load")
    block = range(LOW, LOW + BLOCK)
    set_tiles_elevation(window.scenario.map_manager, [(x, y, LEVELS) for y in block for x in block])
    _set_style(window, style)
    window.resize(1400, 1000)
    window.show()
    _process()
    window._render_current()
    _process()
    expected = {"flat": "flat", "flat_iso": "stepped", "stepped": "stepped", "sloped": "sloped"}[style]
    assert window.map_view._terrain_style == expected
    return window


def _scale(view) -> float:
    return abs(view.transform().determinant()) ** 0.5


def _tolerance(view) -> float:
    # Scrollbars move in whole view pixels, as test_view_state_retention.py allows; +1 for Sloped's int rise.
    return 3.0 / _scale(view) + 1.0


def _assert_at(view, scale: float, centre) -> None:
    after = view._viewport_center_scene()  # priv: the exact float centre a view keeps
    assert _scale(view) == pytest.approx(scale, rel=1e-6)
    assert after.x() == pytest.approx(centre.x(), abs=_tolerance(view))
    assert after.y() == pytest.approx(centre.y(), abs=_tolerance(view))


def _wander(view) -> None:
    view.scale(0.3, 0.3)
    view.centerOn(0, 0)
    _process()


@pytest.mark.parametrize("style", STYLES)
def test_a_view_round_trips_after_a_pan_and_a_zoom(style: str) -> None:
    window = _raised_window(style)
    try:
        view = window.map_view
        view.scale(4, 4)
        view.center_on_tile(*MIDDLE)
        _process()
        scale, centre = _scale(view), view._viewport_center_scene()
        saved = view.capture_saved_view()
        assert 60 < saved.x < 61 and 60 < saved.y < 61, saved
        assert saved.zoom == pytest.approx(scale / view._fit_baseline_scale(), rel=1e-9)
        _wander(view)
        assert view.apply_saved_view(saved)
        _process()
        _assert_at(view, scale, centre)
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize(("first", "second"), [("flat", "stepped"), ("sloped", "flat")])
def test_a_view_saved_in_one_elevation_view_recalls_its_map_point_in_another(first: str, second: str) -> None:
    window = _raised_window(first)
    try:
        view = window.map_view
        view.scale(4, 4)
        view.center_on_tile(*MIDDLE)
        _process()
        saved = view.capture_saved_view()
        assert (math.floor(saved.x), math.floor(saved.y)) == MIDDLE
        _set_style(window, second)
        assert view.apply_saved_view(saved)
        _process()
        centre = view._viewport_center_scene()
        target = view._map_point_scene(saved.x, saved.y)
        assert centre.x() == pytest.approx(target.x(), abs=_tolerance(view))
        assert centre.y() == pytest.approx(target.y(), abs=_tolerance(view))
        assert view._pick_tile(centre) == MIDDLE
        assert _scale(view) == pytest.approx(saved.zoom * view._fit_baseline_scale(), rel=1e-6)
    finally:
        conftest.close_window(window)


def test_a_centre_saved_past_a_stepped_diamond_corner_comes_back_there() -> None:
    from PyQt5.QtCore import QPointF

    window = _raised_window("stepped")
    try:
        view = window.map_view
        view.scale(4, 4)
        rect = view._map_rect
        view._center_on_exact(QPointF(rect.left() + 0.05 * rect.width(), rect.top() + 0.05 * rect.height()))
        _process()
        scale, centre = _scale(view), view._viewport_center_scene()
        assert view._pick_tile(centre) is None, "the bounding box's corner is off the diamond"
        saved = view.capture_saved_view()
        assert not (0 <= saved.x <= 120 and 0 <= saved.y <= 120), saved
        _wander(view)
        assert view.apply_saved_view(saved)
        _process()
        _assert_at(view, scale, centre)
    finally:
        conftest.close_window(window)


def test_a_stored_zoom_past_the_wheel_bounds_clamps_to_them() -> None:
    window = _raised_window("stepped")
    try:
        view = window.map_view
        assert view.apply_saved_view(SavedView(60.5, 60.5, 1000.0))
        assert _scale(view) == pytest.approx(view._max_linear_scale, rel=1e-6)
        assert view.apply_saved_view(SavedView(60.5, 60.5, 1e-6))
        assert _scale(view) == pytest.approx(view._min_linear_scale, rel=1e-6)
    finally:
        conftest.close_window(window)


def test_a_sloped_view_on_a_nw_se_folded_tile_round_trips() -> None:
    """GH #62: a dropped fold reads as NE-SW, so only a NW-SE tile can catch one."""
    from PyQt5.QtCore import QPointF

    from descape import iso_geometry

    window = _raised_window("sloped")
    try:
        view = window.map_view
        view.scale(40, 40)  # one level moves the point a few scene px, so the tolerance must sit well under that
        _process()
        cache = view._sloped_cache()
        cr, fold, proj = cache.corner_rise, cache.fold, view._iso_proj
        best = None
        for y in range(LOW - 2, LOW + BLOCK + 2):
            for x in range(LOW - 2, LOW + BLOCK + 2):
                if not fold[y, x] or cr[y, x] + cr[y + 1, x + 1] == cr[y, x + 1] + cr[y + 1, x]:
                    continue
                for fx, fy in ((0.3, 0.6), (0.6, 0.3), (0.35, 0.35), (0.65, 0.65)):
                    rise = iso_geometry.unit_rise_px(cr, x, y, fx, fy, True)
                    gap = abs(rise - iso_geometry.unit_rise_px(cr, x, y, fx, fy, False))
                    if best is None or gap > best[0]:
                        best = (gap, x, y, fx, fy, rise)
        assert best is not None, "the raised block has no non-planar NW-SE tile"
        gap, x, y, fx, fy, rise = best
        assert gap > 1.5 * _tolerance(view), "the fold must move the point by more than the tolerance"
        view._center_on_exact(QPointF(*iso_geometry.map_point_to_scene_f(x + fx, y + fy, rise, proj)))
        _process()
        scale, centre = _scale(view), view._viewport_center_scene()
        saved = view.capture_saved_view()
        assert (saved.x, saved.y) == pytest.approx((x + fx, y + fy), abs=0.05)
        _wander(view)
        assert view.apply_saved_view(saved)
        _process()
        _assert_at(view, scale, centre)
    finally:
        conftest.close_window(window)


# -- the window ---------------------------------------------------------------

_SAVE_KEY, _GO_KEY = "Ctrl+Alt+F7", "Ctrl+Alt+F8"


def _scenario_copy(tmp_path: Path, name: str) -> Path:
    from descape.scenario_io import BLANK_TEMPLATE_PATH

    path = tmp_path / name
    path.write_bytes(BLANK_TEMPLATE_PATH.read_bytes())
    return path


def _shown(path: Path | None = None, *, untitled: bool = False):
    from descape.scenario_io import BLANK_TEMPLATE_PATH

    window = conftest.shown_window(1200, 800)
    window.load_scenario(path or BLANK_TEMPLATE_PATH, untitled=untitled)
    assert window.scenario is not None
    _process()
    return window


def _press_key(window, key) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    window.activateWindow()
    QApplication.setActiveWindow(window)
    window.map_view.setFocus()
    QApplication.processEvents()
    QTest.keyClick(window.map_view, key, Qt.ControlModifier | Qt.AltModifier)
    _process()


def _bind(window) -> None:
    from descape import settings

    assert settings.set_keybind("savedview_save_1", _SAVE_KEY) == []
    assert settings.set_keybind("savedview_go_1", _GO_KEY) == []
    window.apply_keybind("savedview_save_1")
    window.apply_keybind("savedview_go_1")


def _transform(view) -> tuple:
    t = view.transform()
    return (t.m11(), t.m12(), t.m21(), t.m22(), view.horizontalScrollBar().value(), view.verticalScrollBar().value())


def test_bound_keys_save_a_view_and_go_back_to_it_with_no_document_edit(tmp_path) -> None:
    from PyQt5.QtCore import Qt

    from descape import settings

    path = _scenario_copy(tmp_path, "a.aoe2scenario")
    window = _shown(path)
    try:
        _bind(window)
        view = window.map_view
        view.scale(3, 3)
        view.center_on_tile(30, 40)
        _process()
        scale, centre = _scale(view), view._viewport_center_scene()
        records = len(window.edit_history.records)
        _press_key(window, Qt.Key_F7)
        assert window.status_log.toPlainText().splitlines()[-1] == "Saved view 1"
        assert len(window.edit_history.records) == records and not window.edit_history.is_dirty
        assert set(settings.get_saved_views(path)) == {1}
        _wander(view)
        _press_key(window, Qt.Key_F8)
        _assert_at(view, scale, centre)
    finally:
        conftest.close_window(window)


def test_an_empty_slot_logs_and_does_not_move() -> None:
    window = _shown()
    try:
        view = window.map_view
        before = _transform(view)
        window._keybind_actions["savedview_go_4"].trigger()
        line = window.status_log.toPlainText().splitlines()[-1]
        assert line == "View 4 is empty: save it with View > Saved Views > Save View"
        assert _transform(view) == before
    finally:
        conftest.close_window(window)


def test_go_is_refused_while_a_mouse_button_is_held_mid_stroke(monkeypatch) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication
    from test_shape_tools_viewer import _move, _press, _release

    window = _shown()
    try:
        window.mode_combo.setCurrentText("Terrain")
        window.iso_action.setChecked(False)
        window.terrain_style_combo.setCurrentText("Flat")
        window._on_tool_selected("draw")
        window.paint_trees_check.setChecked(False)
        window.paint_eye_candy_check.setChecked(False)
        window.terrain_panel.set_terrain(2)
        _process()
        view = window.map_view
        window._save_view(1)  # the whole-map view
        view.scale(3, 3)
        view.center_on_tile(20, 20)
        _process()
        monkeypatch.setattr(QApplication, "mouseButtons", staticmethod(lambda: Qt.LeftButton))
        _press(view, (20, 20))
        _move(view, (22, 20))
        mm = window.scenario.map_manager
        terrain = [t.terrain_id for t in mm.terrain]
        before = _transform(view)
        window._keybind_actions["savedview_go_1"].trigger()
        assert _transform(view) == before
        assert [t.terrain_id for t in mm.terrain] == terrain, "the stroke gained no tiles"
        assert window.status_log.toPlainText().splitlines()[-1] == "Go to View 1 needs every mouse button released"
        monkeypatch.setattr(QApplication, "mouseButtons", staticmethod(lambda: Qt.NoButton))
        _release(view, (22, 20))
    finally:
        conftest.close_window(window)


def _save_as(monkeypatch, window, path: Path) -> None:
    from PyQt5.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(path), "")))
    window.save_as()
    assert window.scenario.path == path and not window._untitled


def test_an_untitled_maps_views_are_kept_by_save_as(tmp_path, monkeypatch) -> None:
    from descape import settings

    target = tmp_path / "new.aoe2scenario"
    window = _shown(untitled=True)
    try:
        window.map_view.scale(2, 2)
        _process()
        window._save_view(1)
        assert window.status_log.toPlainText().splitlines()[-1] == "Saved view 1; it is kept once the map is saved"
        saved = window._saved_views[1]
        assert settings._load_config().get("saved_views") is None, "nothing persisted for an untitled map"
        _save_as(monkeypatch, window, target)
        window.close_scenario()
        assert window._saved_views == {}
        settings._saved_views = None  # priv: re-read config.yaml
        window.load_scenario(target)
        assert window._saved_views == {1: SavedView(round(saved.x, 4), round(saved.y, 4), float(f"{saved.zoom:.6g}"))}
    finally:
        conftest.close_window(window)


def test_save_as_with_no_views_leaves_the_targets_views_alone(tmp_path, monkeypatch) -> None:
    from descape import settings

    a = _scenario_copy(tmp_path, "a.aoe2scenario")
    stored = {2: SavedView(10.0, 20.0, 3.0), 5: SavedView(1.0, 2.0, 1.0)}
    settings.set_saved_views(a, stored)
    window = _shown(untitled=True)
    try:
        assert window._saved_views == {}
        _save_as(monkeypatch, window, a)
        assert settings.get_saved_views(a) == stored
        assert window._saved_views == stored, "the document is now a, with a's views"
        window._save_view(1)
        assert set(settings.get_saved_views(a)) == {1, 2, 5}
    finally:
        conftest.close_window(window)


def test_a_titled_document_with_no_views_does_not_adopt_the_targets(tmp_path, monkeypatch) -> None:
    """Only an untitled document takes the target's views: another map's places would point nowhere useful."""
    from descape import settings

    a, b = _scenario_copy(tmp_path, "a.aoe2scenario"), _scenario_copy(tmp_path, "b.aoe2scenario")
    stored = {2: SavedView(10.0, 20.0, 3.0)}
    settings.set_saved_views(b, stored)
    window = _shown(a)
    try:
        _save_as(monkeypatch, window, b)
        assert window._saved_views == {}
        assert settings.get_saved_views(b) == stored
    finally:
        conftest.close_window(window)


def test_a_saved_view_never_reaches_the_scenarios_bytes(tmp_path, monkeypatch) -> None:
    a = _scenario_copy(tmp_path, "a.aoe2scenario")
    without, with_view = tmp_path / "without.aoe2scenario", tmp_path / "with.aoe2scenario"
    window = _shown(a)
    try:
        _save_as(monkeypatch, window, without)
        window.map_view.scale(2, 2)
        window._save_view(3)
        assert not window.edit_history.is_dirty
        _save_as(monkeypatch, window, with_view)
        assert with_view.read_bytes() == without.read_bytes()
        window.save()
        assert with_view.read_bytes() == without.read_bytes()
    finally:
        conftest.close_window(window)


def test_a_resize_keeps_an_untitled_maps_views() -> None:
    from descape.scenario_resize import Anchor

    window = _shown(untitled=True)
    try:
        window._save_view(3)
        views = dict(window._saved_views)
        width = window.scenario.map_manager.map_width
        window._apply_resize(width + 4, width + 4, Anchor.CENTER)
        assert window.scenario.map_manager.map_width == width + 4, "the resize took"
        assert window._saved_views == views
    finally:
        conftest.close_window(window)


def test_a_failed_open_keeps_the_current_documents_views(tmp_path, monkeypatch) -> None:
    from PyQt5.QtWidgets import QMessageBox

    a = _scenario_copy(tmp_path, "a.aoe2scenario")
    bad = tmp_path / "bad.aoe2scenario"
    bad.write_bytes(b"not a scenario")
    window = _shown(a)
    try:
        window._save_view(2)
        views, scenario = dict(window._saved_views), window.scenario
        failures = []
        monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: failures.append(a)))
        window.load_scenario(bad)
        assert failures and window.scenario is scenario
        assert window._saved_views == views
    finally:
        conftest.close_window(window)


def test_the_menu_its_empty_labels_and_its_document_gate(tmp_path) -> None:
    from descape import settings
    from descape.viewer_common import KeepOpenMenu

    window = conftest.shown_window()
    try:
        items = dict(conftest.menu_items(window.menuBar(), "Menu"))
        top = "Menu > &View > Sa&ved Views"
        menu = items[top].menu()
        assert isinstance(menu, KeepOpenMenu), "a View submenu, like the rest"
        rows = [*window.saved_view_go_actions.values(), window.clear_saved_views_action]
        window.load_scenario(_scenario_copy(tmp_path, "a.aoe2scenario"))
        assert not any(menu._keeps_open(row) for row in rows), "a Go closes the menu and shows the map"  # priv
        window.close_scenario()
        for n in SLOTS:
            assert items[f"{top} > Go to View {n}"] is window.saved_view_go_actions[n]
            assert items[f"{top} > &Save View > Save View as {n}"] is window.saved_view_save_actions[n]
        assert items[f"{top} > Clear Saved Views"] is window.clear_saved_views_action
        actions = [*window.saved_view_go_actions.values(), *window.saved_view_save_actions.values()]
        actions.append(window.clear_saved_views_action)
        assert not any(action.isEnabled() for action in actions), "no document"
        window.load_scenario(_scenario_copy(tmp_path, "a.aoe2scenario"))
        assert all(action.isEnabled() for action in actions), "an empty slot's Go stays on too"
        window._save_view(1)
        items[top].menu().aboutToShow.emit()
        assert window.saved_view_go_actions[1].text() == "Go to View 1"
        assert window.saved_view_go_actions[3].text() == "Go to View 3 (empty)"
        window._clear_saved_views()
        assert settings.get_saved_views(window.scenario.path) == {}
        window._save_view(2)
        window.close_scenario()
        assert window._saved_views == {}
        assert not any(action.isEnabled() for action in actions)
    finally:
        conftest.close_window(window)


# -- registry -----------------------------------------------------------------


def test_the_nineteen_rows_ship_unbound_under_one_header_and_match_their_menu_texts() -> None:
    from descape import help_guide, settings

    ids = [f"savedview_go_{n}" for n in SLOTS] + [f"savedview_save_{n}" for n in SLOTS] + ["savedview_clear"]
    rows = [row for row in settings.REBINDABLE_ACTIONS if row[0].startswith("savedview_")]
    assert [aid for aid, _label, _default in rows] == ids
    assert all(default == "" for _aid, _label, default in rows)
    positions = [i for i, row in enumerate(settings.REBINDABLE_ACTIONS) if row[0].startswith("savedview_")]
    assert positions == list(range(positions[0], positions[0] + 19))
    dialog, window = conftest.dialog_and_window()
    try:
        for aid, label, _default in rows:
            assert window._keybind_actions[aid].text() == label
            assert window._keybind_actions[aid].shortcut().isEmpty()
        headers = [h.text() for _d, h, _r in dialog._keybind_sections]
        assert headers.count("<b>Saved Views</b>") == 1
    finally:
        dialog.close()
        conftest.close_window(window)
    page = help_guide.keys_markdown({"savedview_go_1": "Alt+1"}, {})
    assert "### Saved Views" in page and "| Go to View 1 | Alt+1 |" in page
