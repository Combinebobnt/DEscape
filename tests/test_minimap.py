"""GH #54's preview minimap: minimap.minimap_rgb()'s image, the MapView
transforms its viewport box rides on, and the Minimap dock's refresh, pan and
toggle in a real window.

Window tests use a shown window: the rebuild skips a dock that is not on
screen, which a never-shown window's never is. Refresh tests call the
debounced build directly or read the timer, with no sleeps.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from descape import asset_source, minimap
from descape.render_units import NON_BUILDING_SPAN, unit_mark_color, unit_tile_bounds
from descape.terrain_palette import (
    BUILDING_TILE_SPANS,
    PLAYER_COLORS,
    RESOURCE_COLORS,
    TREE_COLOR,
    TREE_UNIT_IDS,
    palette_color_for_terrain_id,
    tile_span,
)
from descape.unit_filter import UnitFilter
from testkit.fakes import FakeScenario, SyntheticTile, SyntheticUnit

import conftest

_W, _H = 12, 10
_FAKE_TEXTURE = Path("/nonexistent/fake_texture.dds")


def _scenario(units_by_player=None, terrain=lambda x, y: (x + y) % 3):
    tiles = [SyntheticTile(x, y, 0, terrain_id=terrain(x, y)) for y in range(_H) for x in range(_W)]
    units = units_by_player or [[] for _ in range(9)]
    return FakeScenario(_W, _H, tiles, units)


@pytest.fixture
def no_install(monkeypatch):
    monkeypatch.setattr(asset_source, "get_terrain_texture_path", lambda _tid: None)


def _const_with_span(span):
    return next(c for c, s in sorted(BUILDING_TILE_SPANS.items()) if s == span)


def _tree_const():
    return next(c for c in sorted(TREE_UNIT_IDS) if tile_span(c, NON_BUILDING_SPAN) == (1, 1))


def _resource_const():
    return next(
        c for c in sorted(RESOURCE_COLORS) if c not in BUILDING_TILE_SPANS and tile_span(c, NON_BUILDING_SPAN) == (1, 1)
    )


# -- minimap_rgb -------------------------------------------------------------


def test_terrain_takes_its_palette_colour_without_an_install(no_install) -> None:
    rgb, used_fallback = minimap.minimap_rgb(_scenario(), UnitFilter())
    assert rgb.shape == (_H, _W, 3) and rgb.dtype == np.uint8
    for y in range(_H):
        for x in range(_W):
            assert tuple(rgb[y, x]) == palette_color_for_terrain_id((x + y) % 3)
    assert not used_fallback, "no texture to wait for is no fallback"


def test_a_loaded_texture_gives_its_average_colour(monkeypatch) -> None:
    monkeypatch.setattr(asset_source, "get_terrain_texture_path", lambda tid: _FAKE_TEXTURE if tid == 1 else None)
    monkeypatch.setattr(asset_source, "_averages", {_FAKE_TEXTURE: (9, 9, 9)})
    monkeypatch.setattr(asset_source, "get_terrain_average_color", lambda tid: (1, 2, 3))
    rgb, used_fallback = minimap.minimap_rgb(_scenario(), UnitFilter())
    assert tuple(rgb[0, 1]) == (1, 2, 3)
    assert tuple(rgb[0, 0]) == palette_color_for_terrain_id(0)
    assert not used_fallback


def test_an_undecoded_texture_draws_its_palette_colour_and_says_so(monkeypatch) -> None:
    monkeypatch.setattr(asset_source, "get_terrain_texture_path", lambda tid: _FAKE_TEXTURE if tid == 1 else None)
    monkeypatch.setattr(asset_source, "_averages", {})

    def refuse(_tid):
        raise AssertionError("the minimap decoded a texture")

    monkeypatch.setattr(asset_source, "get_terrain_average_color", refuse)
    monkeypatch.setattr(asset_source, "get_terrain_texture_array", refuse)
    rgb, used_fallback = minimap.minimap_rgb(_scenario(), UnitFilter())
    assert tuple(rgb[0, 1]) == palette_color_for_terrain_id(1)
    assert used_fallback


def test_a_building_fills_its_footprint(no_install) -> None:
    const = _const_with_span((3, 3))
    units = [[] for _ in range(9)]
    units[2].append(SyntheticUnit(5.5, 4.5, const))
    rgb, _ = minimap.minimap_rgb(_scenario(units), UnitFilter())
    x0, x1, y0, y1 = unit_tile_bounds(units[2][0], _W, _H)
    assert (x1 - x0, y1 - y0) == (3, 3)
    assert (rgb[y0:y1, x0:x1] == PLAYER_COLORS[2]).all()
    assert tuple(rgb[y1, x0]) != PLAYER_COLORS[2]


def test_a_tree_is_dark_green_and_a_resource_has_its_own_colour(no_install) -> None:
    tree, resource = _tree_const(), _resource_const()
    units = [[] for _ in range(9)]
    units[0] += [SyntheticUnit(1.5, 1.5, tree), SyntheticUnit(3.2, 2.7, resource)]
    units[3].append(SyntheticUnit(6.5, 6.5, tree))
    rgb, _ = minimap.minimap_rgb(_scenario(units), UnitFilter())
    assert tuple(rgb[1, 1]) == TREE_COLOR
    assert tuple(rgb[2, 3]) == RESOURCE_COLORS[resource]
    assert tuple(rgb[6, 6]) == TREE_COLOR, "a player's tree is still a tree"


def test_units_the_filters_hide_are_left_out(no_install) -> None:
    tree = _tree_const()
    house = _const_with_span((2, 2))
    units = [[] for _ in range(9)]
    units[0].append(SyntheticUnit(1.5, 1.5, tree))
    units[1].append(SyntheticUnit(5.0, 5.0, house))
    # Player 2 is in the filter's players: only the garrison gate hides this one.
    units[2].append(SyntheticUnit(8.5, 2.5, 4, reference_id=7, garrisoned_in_id=3))
    terrain_only, _ = minimap.minimap_rgb(_scenario(), UnitFilter())
    shown, _ = minimap.minimap_rgb(_scenario(units), UnitFilter())
    hidden, _ = minimap.minimap_rgb(
        _scenario(units), UnitFilter(show_trees=False, players=frozenset({2, 3}), show_garrisoned=False)
    )
    assert not np.array_equal(shown, terrain_only)
    assert np.array_equal(hidden, terrain_only)


def test_a_player_colour_override_is_used(no_install) -> None:
    units = [[] for _ in range(9)]
    units[1].append(SyntheticUnit(2.5, 2.5, 4))
    scenario = _scenario(units)
    scenario.player_colors = (PLAYER_COLORS[0], (9, 99, 199), *PLAYER_COLORS[2:])
    rgb, _ = minimap.minimap_rgb(scenario, UnitFilter())
    assert tuple(rgb[2, 2]) == (9, 99, 199)


def _reference_rgb(scenario, unit_filter: UnitFilter) -> np.ndarray:
    """One unit at a time, in minimap_rgb()'s order: per player, single tiles, then footprints."""
    rgb, _ = minimap.minimap_rgb(_scenario(terrain=lambda x, y: (x * 7 + y) % 5), UnitFilter())
    rgb = rgb.copy()
    for player_id, units in enumerate(scenario.unit_manager.units):
        color = scenario.player_colors[player_id]
        kept = [u for u in units if unit_filter.matches(player_id, u)]
        for wanted_single in (True, False):
            for unit in kept:
                if (tile_span(unit.unit_const, NON_BUILDING_SPAN) == (1, 1)) != wanted_single:
                    continue
                bounds = unit_tile_bounds(unit, _W, _H)
                if bounds is not None:
                    x0, x1, y0, y1 = bounds
                    rgb[y0:y1, x0:x1] = unit_mark_color(unit, color)
    return rgb


@pytest.mark.parametrize("show_garrisoned", [True, False])
def test_the_vectorised_paths_match_a_per_unit_loop(no_install, show_garrisoned: bool) -> None:
    rng = random.Random(54)
    consts = [_tree_const(), _resource_const(), 4, 83, _const_with_span((2, 2)), _const_with_span((3, 3))]
    units = [[] for _ in range(9)]
    for i in range(400):
        player = rng.randrange(9)
        host = rng.choice([-1, -1, -1, 5, i])
        x, y = rng.uniform(-1.5, _W + 0.5), rng.uniform(-1.5, _H + 0.5)
        units[player].append(SyntheticUnit(x, y, rng.choice(consts), reference_id=i, garrisoned_in_id=host))
    scenario = _scenario(units, terrain=lambda x, y: (x * 7 + y) % 5)
    unit_filter = UnitFilter(show_garrisoned=show_garrisoned, players=frozenset({1, 2, 3, 5, 8}))
    rgb, _ = minimap.minimap_rgb(scenario, unit_filter)
    assert np.array_equal(rgb, _reference_rgb(scenario, unit_filter))


@pytest.mark.parametrize("show_inactive", [True, False])
def test_unit_states_fade_and_hide_per_unit_not_per_const(no_install, show_inactive: bool) -> None:
    """A const's units differ in Unit State, so one sample per const would colour or hide them all alike."""
    rng = random.Random(275)
    consts = [_tree_const(), _resource_const(), 4, 83, _const_with_span((2, 2))]
    units = [[] for _ in range(9)]
    for i in range(400):
        unit = SyntheticUnit(rng.uniform(0, _W), rng.uniform(0, _H), rng.choice(consts), reference_id=i)
        unit.status = rng.choice([2, 2, 7, 0, 42])
        units[rng.randrange(9)].append(unit)
    scenario = _scenario(units, terrain=lambda x, y: (x * 7 + y) % 5)
    unit_filter = UnitFilter(show_inactive=show_inactive)
    rgb, _ = minimap.minimap_rgb(scenario, unit_filter)
    assert np.array_equal(rgb, _reference_rgb(scenario, unit_filter))
    shown, _ = minimap.minimap_rgb(scenario, UnitFilter())
    assert np.array_equal(rgb, shown) == show_inactive


# -- the window ----------------------------------------------------------------


# The window tests below need a live ViewerWindow (tests/README.md, "Adding a test").
_NEEDS_QT = pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")


def _settle() -> None:
    from PyQt5.QtWidgets import QApplication

    QApplication.processEvents()
    QApplication.processEvents()


def _map_window(style: str = "Stepped", isometric: bool = True):
    """A shown 1500x900 window with the blank template open in `style`, minimap opened at its
    DEFAULT_HEIGHT (the 64 px floor would otherwise leave it at the log's height) and built."""
    from PyQt5.QtCore import Qt

    from descape.minimap_panel import DEFAULT_HEIGHT
    from descape.scenario_io import BLANK_TEMPLATE_PATH

    window = conftest.shown_window(1500, 900)
    window.minimap_dock.show()
    _settle()
    title = window.minimap_dock.height() - window.minimap_panel.height()
    window.resizeDocks([window.minimap_dock], [DEFAULT_HEIGHT + title], Qt.Vertical)
    window.iso_action.setChecked(isometric)
    window.terrain_style_combo.setCurrentText(style)
    window.load_scenario(BLANK_TEMPLATE_PATH)
    _settle()
    assert window.minimap_panel.height() == DEFAULT_HEIGHT
    window._rebuild_minimap()
    return window


def _corner_points(window):
    from PyQt5.QtCore import QPointF

    mm = window.scenario.map_manager
    w, h = mm.map_width, mm.map_height
    fit = window.minimap_panel.fit()
    assert fit is not None
    return {name: fit.map(QPointF(x, y)) for name, (x, y) in {"o": (0, 0), "w": (w, 0), "wh": (w, h), "h": (0, h)}.items()}


@pytest.mark.gui
@_NEEDS_QT
@pytest.mark.parametrize(("style", "isometric"), [("Stepped", True), ("Sloped", True), ("Flat", True)])
def test_the_isometric_views_draw_a_diamond_with_the_far_x_corner_on_top(style: str, isometric: bool) -> None:
    window = _map_window(style, isometric)
    try:
        corners = _corner_points(window)
        assert min(corners, key=lambda k: corners[k].y()) == "w"
        assert max(corners, key=lambda k: corners[k].y()) == "h"
        width = corners["wh"].x() - corners["o"].x()
        height = corners["h"].y() - corners["w"].y()
        assert abs(width / height - 2.0) < 0.02, "a 2:1 diamond, as the game's"
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@_NEEDS_QT
def test_flat_top_down_draws_a_square() -> None:
    window = _map_window("Flat", isometric=False)
    try:
        corners = _corner_points(window)
        assert abs(corners["o"].y() - corners["w"].y()) < 0.01
        assert abs(corners["o"].x() - corners["h"].x()) < 0.01
        side_x = corners["w"].x() - corners["o"].x()
        side_y = corners["h"].y() - corners["o"].y()
        assert side_x > 100 and abs(side_x - side_y) < 0.01
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@_NEEDS_QT
@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_tile_to_scene_is_the_ground_plane_and_none_after_close(style: str) -> None:
    from PyQt5.QtCore import QPointF, QRectF
    from PyQt5.QtGui import QTransform

    from descape import iso_geometry

    window = _map_window(style)
    try:
        to_scene = window.map_view.tile_to_scene()
        proj = window.map_view._iso_proj
        for mx, my in [(0, 0), (37.25, 80.5), (119.9, 3.0), (60, 60)]:
            point = to_scene.map(QPointF(mx, my))
            ax, ay = iso_geometry.map_point_to_screen(mx, my, 0, proj)
            assert abs(point.x() - ax) <= 1 and abs(point.y() - ay) <= 1
        window.close_scenario()
        assert window.map_view.tile_to_scene() is None
        panel = window.minimap_panel
        assert panel.viewport_polygon() is None and _draws_no_map(panel)
        # A view handed over again finds no image left to fit: the close dropped it.
        panel.set_view(QTransform(), QRectF(0, 0, 10, 10))
        assert panel.fit() is None and _draws_no_map(panel)
    finally:
        conftest.close_window(window)


def _draws_no_map(panel) -> bool:
    """The panel paints its fill colour alone: no map image, no border, no box."""
    from PyQt5.QtGui import QImage

    from descape.minimap_panel import FILL_COLOR

    painted = panel.grab().toImage().convertToFormat(QImage.Format_RGB32)
    blank = QImage(painted.size(), QImage.Format_RGB32)
    blank.fill(FILL_COLOR)
    return painted.width() > 0 and painted == blank


def _box_centre_tile(window):
    """The viewport box's centre, mapped back from minimap pixels to a map point."""
    poly = window.minimap_panel.viewport_polygon()
    centre = poly.boundingRect().center()
    inverse, ok = window.minimap_panel.fit().inverted()
    assert ok
    point = inverse.map(centre)
    return point.x(), point.y()


def _wheel_zoom_in(window, notches: int) -> None:
    """Real wheel notches over the map's centre, through MapView's own wheelEvent."""
    from PyQt5.QtCore import QPoint, QPointF, Qt
    from PyQt5.QtGui import QWheelEvent
    from PyQt5.QtWidgets import QApplication

    viewport = window.map_view.viewport()
    centre = QPointF(viewport.width() / 2, viewport.height() / 2)
    for _ in range(notches):
        event = QWheelEvent(
            centre, QPointF(viewport.mapToGlobal(centre.toPoint())), QPoint(0, 0), QPoint(0, 120),
            Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False,
        )
        QApplication.sendEvent(viewport, event)
    _settle()


def _zoom_in(window, factor: float = 4.0) -> None:
    from descape import zoom_preframe

    view = window.map_view
    zoom_preframe.zoom_about(view, factor, view.viewport().rect().center())
    view._note_viewport_changed()
    _settle()


@pytest.mark.gui
@_NEEDS_QT
@pytest.mark.parametrize(("style", "isometric"), [("Stepped", True), ("Flat", False)])
def test_the_box_follows_a_centre_a_scroll_and_a_zoom(style: str, isometric: bool) -> None:
    window = _map_window(style, isometric)
    try:
        _zoom_in(window)
        window.map_view.center_on_tile(30, 70)
        _settle()
        mx, my = _box_centre_tile(window)
        assert abs(mx - 30.5) <= 1 and abs(my - 70.5) <= 1
        before = window.minimap_panel.viewport_polygon().boundingRect()
        bar = window.map_view.horizontalScrollBar()
        bar.setValue(bar.value() + 200)
        _settle()
        scrolled = window.minimap_panel.viewport_polygon().boundingRect()
        assert scrolled.center().x() > before.center().x()
        _wheel_zoom_in(window, 3)
        zoomed = window.minimap_panel.viewport_polygon().boundingRect()
        assert zoomed.width() < scrolled.width() * 0.8, "a real wheel zoom shrinks the box"
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@_NEEDS_QT
def test_a_press_and_drag_on_the_minimap_recentres_the_view_and_returns_focus(monkeypatch) -> None:
    from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt
    from PyQt5.QtGui import QContextMenuEvent, QMouseEvent
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    window = _map_window("Stepped")
    try:
        _zoom_in(window)
        panel = window.minimap_panel
        fit = panel.fit()
        target = fit.map(QPointF(20.5, 90.5)).toPoint()
        QTest.mousePress(panel, Qt.LeftButton, Qt.NoModifier, target)
        _settle()
        mx, my = _box_centre_tile(window)
        assert abs(mx - 20.5) <= 1 and abs(my - 90.5) <= 1
        dragged = panel.fit().map(QPointF(70.5, 40.5)).toPoint()
        # Qt 5's QTest.mouseMove() carries no held button, so the drag's move is sent by hand.
        move = QMouseEvent(QEvent.MouseMove, QPointF(dragged), Qt.NoButton, Qt.LeftButton, Qt.NoModifier)
        QApplication.sendEvent(panel, move)
        _settle()
        mx, my = _box_centre_tile(window)
        assert abs(mx - 70.5) <= 1 and abs(my - 40.5) <= 1
        QTest.mouseRelease(panel, Qt.LeftButton, Qt.NoModifier, dragged)
        _settle()
        assert window.map_view.hasFocus() or window.focusWidget() is window.map_view
        assert panel.focusPolicy() == Qt.NoFocus
        before = _box_centre_tile(window)
        shown = _record_menus(monkeypatch)
        QTest.mouseClick(panel, Qt.RightButton, Qt.NoModifier, QPoint(5, 5))
        QApplication.sendEvent(panel, QContextMenuEvent(QContextMenuEvent.Mouse, QPoint(5, 5), panel.mapToGlobal(QPoint(5, 5))))
        _settle()
        assert _box_centre_tile(window) == before, "only the left button pans"
        assert shown == [[("Zoom In", True), ("Zoom Out", False), ("Fit Whole Map", True)]]
        for _ in range(8):
            panel.zoom_by(1)
        QApplication.sendEvent(panel, QContextMenuEvent(QContextMenuEvent.Mouse, QPoint(5, 5), panel.mapToGlobal(QPoint(5, 5))))
        assert shown[-1] == [("Zoom In", False), ("Zoom Out", True), ("Fit Whole Map", True)]
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@_NEEDS_QT
def test_dragging_the_divider_from_the_log_widens_the_minimap_past_its_hint_and_refits_it() -> None:
    """GH #54: the dock's Maximum width policy still lets the separator drag size it; the map stays centred."""
    from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt
    from PyQt5.QtGui import QMouseEvent
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    from descape.minimap_panel import DEFAULT_WIDTH

    window = _map_window("Stepped")
    try:
        log, dock, panel = window.log_dock, window.minimap_dock, window.minimap_panel
        assert panel.width() == DEFAULT_WIDTH and log.geometry().right() < dock.geometry().left()
        log_width, view_size = log.width(), window.map_view.size()
        start = QPoint((log.geometry().right() + dock.geometry().left()) // 2, dock.geometry().center().y())
        QTest.mousePress(window, Qt.LeftButton, Qt.NoModifier, start)
        # QMainWindow moves a separator on a 0 ms timer, so each step runs it before the next.
        for dx in (-10, -60, -120):
            move = QMouseEvent(QEvent.MouseMove, QPointF(start + QPoint(dx, 0)), Qt.NoButton, Qt.LeftButton, Qt.NoModifier)
            QApplication.sendEvent(window, move)
            _settle()
        QTest.mouseRelease(window, Qt.LeftButton, Qt.NoModifier, start + QPoint(-120, 0))
        _settle()
        assert panel.width() == DEFAULT_WIDTH + 120 and log.width() == log_width - 120
        assert window.map_view.size() == view_size, "the row only splits differently"
        drawn = _corner_points(window)
        left, right = min(p.x() for p in drawn.values()), max(p.x() for p in drawn.values())
        assert abs((left + right) / 2 - panel.width() / 2) <= 1, "the map re-fits centred in the wider panel"
        assert panel.viewport_polygon() is not None
    finally:
        conftest.close_window(window)


def _record_menus(monkeypatch) -> list:
    """Patches QMenu.exec_ to record each menu's action texts and close at once, so no modal popup runs."""
    from PyQt5.QtWidgets import QMenu

    shown: list = []
    monkeypatch.setattr(QMenu, "exec_", lambda menu, *args: shown.append([(a.text(), a.isEnabled()) for a in menu.actions()]))
    return shown


def _wheel_minimap(panel, pos, delta: int = 120, modifiers=None):
    from PyQt5.QtCore import QPoint, QPointF, Qt
    from PyQt5.QtGui import QWheelEvent
    from PyQt5.QtWidgets import QApplication

    pos = QPointF(pos)
    event = QWheelEvent(
        pos, QPointF(panel.mapToGlobal(pos.toPoint())), QPoint(0, 0), QPoint(0, delta),
        Qt.NoButton, Qt.NoModifier if modifiers is None else modifiers, Qt.NoScrollPhase, False,
    )
    QApplication.sendEvent(panel, event)
    return event


def _same_fit(a, b, tol: float = 0.5) -> bool:
    from PyQt5.QtCore import QPointF

    return all(
        abs(a.map(p).x() - b.map(p).x()) <= tol and abs(a.map(p).y() - b.map(p).y()) <= tol
        for p in (QPointF(0, 0), QPointF(120, 0), QPointF(120, 120), QPointF(0, 120))
    )


@pytest.mark.gui
@_NEEDS_QT
def test_the_wheel_zooms_the_minimap_never_the_map() -> None:
    from PyQt5.QtCore import QPointF, Qt

    from descape.minimap_panel import MAX_ZOOM, ZOOM_STEP

    window = _map_window("Stepped")
    try:
        panel = window.minimap_panel
        before = window.map_view.transform()
        cursor = QPointF(panel.width() * 0.3, panel.height() * 0.6)
        inverse, _ok = panel.fit().inverted()
        under = inverse.map(cursor)
        event = _wheel_minimap(panel, cursor)
        assert event.isAccepted()
        assert abs(panel.zoom - ZOOM_STEP) < 1e-9
        moved = panel.fit().map(under)
        assert abs(moved.x() - cursor.x()) <= 1 and abs(moved.y() - cursor.y()) <= 1
        assert window.map_view.transform() == before
        assert _wheel_minimap(panel, cursor, modifiers=Qt.ControlModifier).isAccepted()
        assert abs(panel.zoom - ZOOM_STEP) < 1e-9, "a modified wheel changes nothing"
        for _ in range(4):
            _wheel_minimap(panel, cursor, 30)
        assert abs(panel.zoom - ZOOM_STEP ** 2) < 1e-9, "four 30-unit deltas step once"
        for _ in range(12):
            _wheel_minimap(panel, cursor)
        assert panel.zoom == MAX_ZOOM
        for _ in range(12):
            _wheel_minimap(panel, cursor, -120)
        assert panel.zoom == 1.0
        assert window.map_view.transform() == before
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@_NEEDS_QT
def test_zoom_one_is_the_whole_map_fit() -> None:
    from PyQt5.QtCore import QPointF

    window = _map_window("Stepped")
    try:
        panel = window.minimap_panel
        whole = panel.fit()
        _wheel_minimap(panel, QPointF(40, 30))
        _wheel_minimap(panel, QPointF(40, 30))
        assert not _same_fit(panel.fit(), whole)
        panel.fit_whole_map()
        assert _same_fit(panel.fit(), whole)
    finally:
        conftest.close_window(window)


@_NEEDS_QT
def test_the_clamp_centres_a_span_that_fits_and_bands_none_that_overflows() -> None:
    from PyQt5.QtCore import QPointF, QRectF

    from descape.minimap_panel import MARGIN, clamped_offset

    box = QRectF(10, 20, 100, 50)
    # Scale 1 in a 200x100 panel: both axes fit, so it centres whatever the asked centre.
    offset = clamped_offset(box, 200, 100, 1.0, QPointF(0, 0))
    assert (offset.x(), offset.y()) == (40.0, 5.0)
    # Scale 4: both overflow. A centre past the box's low corner pins that corner at MARGIN.
    offset = clamped_offset(box, 200, 100, 4.0, QPointF(-500, -500))
    assert (offset.x(), offset.y()) == (MARGIN - 40.0, MARGIN - 80.0)
    offset = clamped_offset(box, 200, 100, 4.0, QPointF(900, 900))
    assert (offset.x(), offset.y()) == (200 - MARGIN - 440.0, 100 - MARGIN - 280.0)
    # Inside the range the centre lands on the panel's middle.
    offset = clamped_offset(box, 200, 100, 4.0, QPointF(60, 45))
    assert (offset.x(), offset.y()) == (100 - 240.0, 50 - 180.0)


@pytest.mark.gui
@_NEEDS_QT
def test_a_zoomed_minimap_shows_no_off_map_band_and_clamps_its_centre() -> None:
    from PyQt5.QtCore import QPointF

    from descape.minimap_panel import MARGIN, MAX_ZOOM

    window = _map_window("Flat", isometric=False)
    try:
        panel = window.minimap_panel
        corner = panel.fit().map(QPointF(0, 0))
        for _ in range(8):
            _wheel_minimap(panel, corner)
        assert panel.zoom == MAX_ZOOM
        drawn = panel.fit().map(panel._map_quad()).boundingRect()
        assert drawn.left() <= MARGIN + 0.5 and drawn.top() <= MARGIN + 0.5
        assert drawn.right() >= panel.width() - MARGIN - 0.5 and drawn.bottom() >= panel.height() - MARGIN - 0.5
        centre, again = panel._centre, panel.fit()
        assert panel._centre == centre and _same_fit(again, panel.fit(), 0.0), "fit() changes no state"
    finally:
        conftest.close_window(window)


def _zoomed_three_notches(panel) -> None:
    from PyQt5.QtCore import QPointF

    _wheel_minimap(panel, QPointF(panel.width() / 2, panel.height() / 2))
    _wheel_minimap(panel, QPointF(panel.width() / 2, panel.height() / 2))
    _wheel_minimap(panel, QPointF(panel.width() / 2, panel.height() / 2))


@pytest.mark.gui
@_NEEDS_QT
def test_a_zoomed_minimap_follows_the_view_at_its_edges() -> None:
    from descape.minimap_panel import FOLLOW_INSET

    window = _map_window("Flat", isometric=False)
    try:
        _zoom_in(window)
        window.map_view.center_on_tile(60, 60)
        _settle()
        panel = window.minimap_panel
        _zoomed_three_notches(panel)
        window.map_view.center_on_tile(62, 61)
        _settle()
        held = panel.fit()
        window.map_view.center_on_tile(61, 60)
        _settle()
        assert _same_fit(panel.fit(), held, 0.01), "a move inside the inner area scrolls nothing"
        window.map_view.center_on_tile(95, 60)
        _settle()
        assert not _same_fit(panel.fit(), held, 0.01)
        centre = panel.viewport_polygon().boundingRect().center()
        assert abs(centre.x() - panel.width() * (1 - FOLLOW_INSET)) <= 1
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@_NEEDS_QT
def test_an_edit_rebuild_does_not_move_a_zoomed_window() -> None:
    from PyQt5.QtCore import QPointF

    window = _map_window("Stepped")
    try:
        _zoom_in(window)
        panel = window.minimap_panel
        corner = panel.fit().map(QPointF(0, 120))
        for _ in range(4):
            _wheel_minimap(panel, corner)
        held = panel.fit()
        # A fallback build's retry returns before set_view(), which would test nothing.
        window._minimap_retrying = False
        window._rebuild_minimap()
        window._push_minimap_view()
        _settle()
        assert _same_fit(panel.fit(), held, 0.01)
        window.map_view.center_on_tile(110, 10)
        _settle()
        assert not _same_fit(panel.fit(), held, 0.01), "a real pan far from the window follows"
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@_NEEDS_QT
def test_a_drag_on_a_zoomed_minimap_holds_its_window_until_release() -> None:
    from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt
    from PyQt5.QtGui import QMouseEvent
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    window = _map_window("Flat", isometric=False)
    try:
        _zoom_in(window)
        window.map_view.center_on_tile(60, 60)
        _settle()
        panel = window.minimap_panel
        _zoomed_three_notches(panel)
        held = panel.fit()
        QTest.mousePress(panel, Qt.LeftButton, Qt.NoModifier, QPoint(panel.width() // 2, panel.height() // 2))
        edge = QPointF(panel.width() - 2, panel.height() / 2)
        move = QMouseEvent(QEvent.MouseMove, edge, Qt.NoButton, Qt.LeftButton, Qt.NoModifier)
        QApplication.sendEvent(panel, move)
        _settle()
        assert _same_fit(panel.fit(), held, 0.01), "no follow while the drag is held"
        QTest.mouseRelease(panel, Qt.LeftButton, Qt.NoModifier, edge.toPoint())
        _settle()
        assert not _same_fit(panel.fit(), held, 0.01), "the follow applies on release"
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@_NEEDS_QT
def test_a_new_document_resets_the_zoom() -> None:
    from descape.scenario_io import BLANK_TEMPLATE_PATH

    window = _map_window("Stepped")
    try:
        panel = window.minimap_panel
        _zoomed_three_notches(panel)
        assert panel.zoom > 1
        window.load_scenario(BLANK_TEMPLATE_PATH)
        _settle()
        assert panel.zoom == 1.0
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@_NEEDS_QT
def test_off_by_default_and_a_closed_minimap_builds_nothing_until_it_opens(monkeypatch) -> None:
    from descape.scenario_io import BLANK_TEMPLATE_PATH

    window = conftest.shown_window(1500, 900)
    try:
        action = window.minimap_dock.toggleViewAction()
        assert window.minimap_dock.isHidden() and not action.isChecked()
        calls = []
        real = minimap.minimap_rgb
        monkeypatch.setattr(minimap, "minimap_rgb", lambda *a: calls.append(1) or real(*a))
        window.load_scenario(BLANK_TEMPLATE_PATH)
        _settle()
        window._schedule_minimap()
        window._rebuild_minimap()
        assert calls == [], "a closed minimap builds nothing"
        window._minimap_timer.stop()
        action.trigger()
        _settle()
        assert window.minimap_dock.isVisible()
        assert window._minimap_timer.isActive(), "reopening schedules the build it skipped"
        window._rebuild_minimap()
        assert calls == [1]
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@_NEEDS_QT
def test_with_no_document_the_dock_stays_empty() -> None:
    window = conftest.shown_window(1500, 900)
    try:
        window.minimap_dock.show()
        _settle()
        window._schedule_minimap()
        assert not window._minimap_timer.isActive()
        window._rebuild_minimap()
        assert _draws_no_map(window.minimap_panel) and window.minimap_panel.fit() is None
    finally:
        conftest.close_window(window)


def _paint_one_tile(window, x: int, y: int, terrain_id: int) -> None:
    window.mode_combo.setCurrentText("Terrain")
    window._on_tool_selected("draw")
    window.terrain_panel.set_terrain(terrain_id)
    window.paint_trees_check.setChecked(False)
    window.paint_eye_candy_check.setChecked(False)
    window.brush_size_spin.setValue(1)
    window.on_edit_stroke_start()
    window.on_edit_stroke_tile(x, y, 0)
    window.on_edit_stroke_end()


@pytest.mark.gui
@_NEEDS_QT
def test_a_terrain_paint_and_its_undo_each_rebuild_and_a_ruler_does_not() -> None:
    from descape.edit_history import RulerDiffRecord

    window = _map_window("Stepped")
    try:
        window._minimap_timer.stop()
        window.edit_history.push_ruler_record(RulerDiffRecord("Add ruler", (), (), lambda _r, _i: None))
        assert not window._minimap_timer.isActive(), "a view-only record leaves the image alone"
        original = window.minimap_panel._image.copy()
        tile = window.scenario.map_manager.get_tile(10, 12)
        new_terrain = 0 if tile.terrain_id != 0 else 1
        window._minimap_timer.stop()
        _paint_one_tile(window, 10, 12, new_terrain)
        assert window._minimap_timer.isActive()
        window._rebuild_minimap()
        painted = window.minimap_panel._image.copy()
        assert painted.pixel(10, 12) != original.pixel(10, 12)
        window._minimap_timer.stop()
        window.undo()
        assert window._minimap_timer.isActive()
        window._rebuild_minimap()
        assert window.minimap_panel._image.pixel(10, 12) == original.pixel(10, 12)
        window._minimap_timer.stop()
        window.redo()
        assert window._minimap_timer.isActive()
        window._rebuild_minimap()
        assert window.minimap_panel._image.pixel(10, 12) == painted.pixel(10, 12)
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@_NEEDS_QT
def test_a_filters_change_rebuilds_with_the_new_filter(monkeypatch) -> None:
    window = _map_window("Stepped")
    try:
        seen = []
        real = minimap.minimap_rgb
        monkeypatch.setattr(minimap, "minimap_rgb", lambda scenario, unit_filter: seen.append(unit_filter) or real(scenario, unit_filter))
        window._minimap_timer.stop()
        window.show_trees_action.trigger()
        assert window._minimap_timer.isActive()
        window._rebuild_minimap()
        assert seen == [window._unit_filter] and not seen[0].show_trees
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@_NEEDS_QT
def test_a_re_render_with_no_history_change_schedules_a_rebuild() -> None:
    """A data-mod or game-data switch re-renders without touching the history,
    so _render_current() is the only thing that schedules that rebuild."""
    window = _map_window("Stepped")
    try:
        window._minimap_timer.stop()
        window._render_current()
        assert window._minimap_timer.isActive()
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@_NEEDS_QT
@pytest.mark.test_map
def test_a_second_load_drops_the_old_image_until_its_own_build() -> None:
    window = _map_window("Stepped")
    try:
        assert window.minimap_panel._image.width() == window.scenario.map_manager.map_width
        window.load_scenario(conftest.TEST_MAP_PATH)
        assert _draws_no_map(window.minimap_panel), "the old map's image never takes the new map's box"
        assert window._minimap_timer.isActive()
        window._rebuild_minimap()
        assert window.minimap_panel._image.width() == window.scenario.map_manager.map_width == 240
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@_NEEDS_QT
def test_a_fallback_build_retries_only_while_textures_load(monkeypatch) -> None:
    from descape.viewer_common import MINIMAP_DEBOUNCE_MS, MINIMAP_RETRY_MS

    window = _map_window("Stepped")
    try:
        window._schedule_minimap()
        window._minimap_timer.stop()
        count = [3]
        monkeypatch.setattr(asset_source, "loaded_texture_count", lambda: count[0])
        monkeypatch.setattr(minimap, "minimap_rgb", lambda *_a: (np.zeros((4, 4, 3), np.uint8), True))
        window._rebuild_minimap()
        assert window._minimap_timer.isActive() and window._minimap_timer.interval() == MINIMAP_RETRY_MS
        count[0] = 5
        window._minimap_timer.stop()
        window._rebuild_minimap()
        assert window._minimap_timer.isActive(), "a texture decoded since, so it builds and retries again"
        window._minimap_timer.stop()
        built = []
        monkeypatch.setattr(minimap, "minimap_rgb", lambda *_a: built.append(1) or (np.zeros((4, 4, 3), np.uint8), True))
        window._rebuild_minimap()
        assert built == [] and not window._minimap_timer.isActive(), "no progress ends the retries"
        count[0] = 6
        window._push_minimap_view()
        assert window._minimap_timer.isActive(), "a pan that decoded a texture schedules a build"
        assert window._minimap_timer.interval() == MINIMAP_DEBOUNCE_MS
        window._rebuild_minimap()
        assert built == [1]
        window._minimap_timer.stop()
        window._schedule_minimap()
        window._rebuild_minimap()
        assert built == [1, 1], "a real change builds whatever the texture count"
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@_NEEDS_QT
def test_a_zoomed_window_survives_a_panel_resize_a_main_view_zoom_and_keeps_its_zoom_over_an_orientation_round_trip() -> None:
    from PyQt5.QtCore import QPointF, Qt

    from descape.minimap_panel import DEFAULT_HEIGHT

    window = _map_window("Stepped")
    try:
        panel = window.minimap_panel
        corner = panel.fit().map(QPointF(0, 120))
        _wheel_minimap(panel, corner)
        held, zoom = panel.fit(), panel.zoom
        title = window.minimap_dock.height() - panel.height()
        # Tall enough that the zoomed span fits vertically, then back.
        window.resizeDocks([window.minimap_dock], [320 + title], Qt.Vertical)
        _settle()
        window.resizeDocks([window.minimap_dock], [DEFAULT_HEIGHT + title], Qt.Vertical)
        _settle()
        assert panel.height() == DEFAULT_HEIGHT and panel.zoom == zoom
        assert _same_fit(panel.fit(), held, 0.5), "the zoomed position comes back after a resize"
        view = window.map_view
        from descape import zoom_preframe

        # With the box in the window's middle, a main-view zoom about its centre moves nothing here.
        inverse, _ok = panel.fit().inverted()
        middle = inverse.map(QPointF(panel.width() / 2, panel.height() / 2))
        view.center_on_map_point(middle.x(), middle.y())
        _settle()
        held = panel.fit()
        zoom_preframe.zoom_about(view, 1.25, view.viewport().rect().center())
        view._note_viewport_changed()
        _settle()
        assert _same_fit(panel.fit(), held, 0.5), "a main-view zoom about its centre leaves the minimap alone"
        window.iso_action.setChecked(False)
        _settle()
        window.iso_action.setChecked(True)
        _settle()
        # The toggle refits the main view itself, so the window may follow its box; the zoom stays.
        assert panel.zoom == zoom
    finally:
        conftest.close_window(window)
