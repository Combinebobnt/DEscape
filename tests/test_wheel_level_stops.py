"""TASK-600: the wheel lands on every zoom level's own 1:1 scale (its stop),
from both directions, in each view, with the device-px snap and the
pre-rendered frames' twin prediction at every stop. The lattice itself is
tests/test_zoom_stops.py; the native stop's own command is test_native_tier.py."""

from __future__ import annotations

import pytest
from test_native_tier import _offset_viewport_off_grid, _wheel
from test_wheel_zoom import _notch_scale, _on_lattice, _scale

from descape.render_cache_common import MIP_SCALE_SLACK

import conftest

pytestmark = [pytest.mark.test_map, pytest.mark.gui]

WINDOW_SIZE = 700


def _window(view_kind: str):
    window = conftest.stepped_window(conftest.TEST_MAP_PATH, size=WINDOW_SIZE)
    if view_kind == "Sloped":
        window.terrain_style_combo.setCurrentText("Sloped")
    elif view_kind == "top-down Flat":
        window.terrain_style_combo.setCurrentText("Flat")
        window.iso_action.setChecked(False)
    return window


def _stop_level(view) -> int | None:
    """The level whose stop the view sits on, or None."""
    for level in view._canvas_item._cache.mip_levels():
        if abs(view.level_zoom_factor(level) - 1.0) <= MIP_SCALE_SLACK:
            return level
    return None


def _assert_on_device_grid(view, dpr: float) -> None:
    """The stop's level drawn 1:1, its origin on a whole device px of the window."""
    from descape import zoom_preframe

    level = _stop_level(view)
    assert level is not None and view._mip_for_zoom_factor(1.0) == level
    dt = view._canvas_item.deviceTransform(view.viewportTransform())
    offset = zoom_preframe.viewport_offset(view.viewport())
    for phys in ((offset.x() + dt.dx()) * dpr, (offset.y() + dt.dy()) * dpr):
        assert phys == pytest.approx(round(phys), abs=1e-6), (level, phys)


def _sweep(view, up: bool, on_stop=None) -> list[int]:
    """Single fresh-gesture notches until the bound refuses one; the levels whose stops were landed on."""
    visited = []
    for _ in range(200):
        before = _scale(view)
        _wheel(view, up)
        if _scale(view) == before:
            return visited
        assert _on_lattice(view) or view._max_linear_scale == pytest.approx(_scale(view), rel=1e-9) or (
            view._min_linear_scale == pytest.approx(_scale(view), rel=1e-9)
        ), "a notch landed between lattice points and bounds"
        level = _stop_level(view)
        if level is not None:
            visited.append(level)
            if on_stop is not None:
                on_stop(level)
    pytest.fail("never reached the bound")


@pytest.mark.parametrize("view_kind", ["Stepped", "Sloped", "top-down Flat"])
def test_the_wheel_lands_on_every_level_stop_from_both_directions(view_kind) -> None:
    window = _window(view_kind)
    try:
        view = window.map_view
        levels = view._canvas_item._cache.mip_levels()
        assert len(levels) >= 3, f"{view_kind}: a {len(levels)}-level ladder -- vacuous"
        finest = view._canvas_item._cache.mip_tile_px(levels[-1])
        assert finest == (128 if view_kind == "top-down Flat" else 192), "TASK-601's 192 stop is in the iso ladders"
        reachable = [
            lvl for lvl in levels
            if view._min_linear_scale <= _scale(view) * view.level_zoom_factor(lvl) <= view._max_linear_scale * (1 + 1e-12)
        ]
        assert reachable == levels, f"{view_kind}: stops outside the bounds {set(levels) - set(reachable)}"
        assert _stop_level(view) is None, "fit sits on a stop -- the off-lattice start unexercised"
        fit_level = view._mip_for_zoom_factor(1.0)
        inward = _sweep(view, True, lambda level: _assert_on_device_grid(view, 1.0))
        assert inward == [lvl for lvl in levels if lvl >= inward[0]] and inward[0] <= fit_level + 1, (view_kind, inward)
        # A ceiling on the finest stop (a big map in a small window) is where the outward sweep starts.
        start = _stop_level(view)
        outward = _sweep(view, False, lambda level: _assert_on_device_grid(view, 1.0))
        assert outward == [lvl for lvl in sorted(levels, reverse=True) if lvl != start], (view_kind, start, outward)
        assert start in (None, levels[-1])
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("dpr", [1.25, 1.5])
def test_every_level_stop_snaps_to_whole_device_px_and_the_twin_predicts_it(monkeypatch, dpr) -> None:
    """A notch onto any level's stop is snapped (TASK-599's snap, every stop
    now), and ZoomTwin.predict() with _twin_snap() equals the real notch's
    transform, so the pre-rendered frame keyed on it is hit."""
    from descape import zoom_preframe

    window = _window("Stepped")
    try:
        view = window.map_view
        view._zoom_on_cursor = False
        monkeypatch.setattr(view, "devicePixelRatioF", lambda: dpr)
        _offset_viewport_off_grid(view)
        view._capture_zoom_baseline()
        twin = zoom_preframe.ZoomTwin(view)
        anchor = view.viewport().rect().center()
        landed, snapped = [], []
        for up in (True, False):
            for _ in range(200):
                view._wheel_gesture_mip = view._mip_for_zoom_factor(1.0)
                factor = view._wheel_zoom_factor(1 if up else -1)
                view._wheel_gesture_mip = None
                if factor is None:
                    break
                snap = view._twin_snap(factor)
                predicted, _rect = twin.predict(factor, anchor, snap)
                unsnapped, _rect = twin.predict(factor, anchor)
                _wheel(view, up)
                assert predicted == view.viewportTransform(), (up, _scale(view))
                level = _stop_level(view)
                if level is not None:
                    _assert_on_device_grid(view, dpr)
                    landed.append(level)
                    snapped.append(unsnapped != predicted)
                else:
                    assert snap is None, "a notch off every stop snaps"
        levels = view._canvas_item._cache.mip_levels()
        assert set(landed) == set(levels), landed
        assert any(snapped), "no stop needed a snap at this DPR -- vacuous"
    finally:
        conftest.close_window(window)


def test_a_capture_at_the_reference_stop_is_the_level_itself() -> None:
    """At the 64 px stop (level 0, a unit device transform at DPR 1) the
    viewport shows the level's own chunk pixels byte for byte: no second
    resample. Pinned to the reference level, where the capture is exact."""
    import numpy as np
    from PyQt5.QtCore import QRect
    from PyQt5.QtGui import QImage
    from PyQt5.QtWidgets import QApplication

    from testkit import qt_capture

    window = _window("Stepped")
    try:
        view = window.map_view
        item, cache = view._canvas_item, view._canvas_item._cache
        view.centerOn(item.boundingRect().center())
        for _ in range(60):
            if _stop_level(view) == 0:
                break
            _wheel(view, True)
        assert _stop_level(view) == 0 and view.transform().m11() == pytest.approx(1.0, abs=1e-12)
        QApplication.processEvents()
        vp = view.viewport().rect()
        top_left = item.mapFromScene(view.mapToScene(vp.topLeft()))
        x0, y0 = round(top_left.x()), round(top_left.y())
        assert (x0, y0) == (top_left.x(), top_left.y()), "the stop is not on whole px"
        w, h = vp.width(), vp.height()
        cw, ch = cache.canvas_dims(0)
        assert x0 >= 0 and y0 >= 0 and x0 + w <= cw and y0 + h <= ch, "the viewport leaves the canvas"
        image = view.viewport().grab(QRect(0, 0, w, h)).toImage().convertToFormat(QImage.Format_RGB888)
        shown = qt_capture.qimage_rgb888_to_array(image)
        want = cache.render_rect(x0, y0, x0 + w, y0 + h, mip=0)[..., :3]
        assert shown.shape == want.shape
        assert np.array_equal(shown, want), f"{int(np.count_nonzero((shown != want).any(axis=2)))} px differ"
        # Control: the crop one px over differs, so the match is not two blank images.
        assert not np.array_equal(shown, cache.render_rect(x0 + 1, y0, x0 + 1 + w, y0 + h, mip=0)[..., :3])
    finally:
        conftest.close_window(window)


def test_a_quality_change_rebuilds_the_lattice() -> None:
    """Risk 1: a lattice cached across a ladder rebuild would land off-stop. Default -> Potato drops 96 and 128."""
    from descape import settings

    window = _window("Stepped")
    try:
        view = window.map_view
        _wheel(view, True)
        before = list(view._wheel_lattice()[0])
        settings.set_graphics_quality(2)
        window.refresh_map()
        levels = view._canvas_item._cache.mip_levels()
        assert max(view._canvas_item._cache.mip_tile_px(lvl) for lvl in levels) == 64
        after = list(view._wheel_lattice()[0])
        assert after != before
        view._wheel_lattice_cache = None
        assert view._wheel_lattice()[0] == after, "the cached lattice is not the fresh one"
        visited = _sweep(view, False) + _sweep(view, True)
        assert set(visited) == set(levels), visited
    finally:
        settings.set_graphics_quality(settings.GRAPHICS_QUALITY_DEFAULT)
        conftest.close_window(window)


def test_a_refused_flick_leaves_the_view_on_a_point() -> None:
    """Risk 2: the budget trims a multi-notch flick to a point, never between points."""
    from test_wheel_zoom import _wheel_units

    window = _window("Stepped")
    try:
        view = window.map_view
        _wheel(view, True)
        flicks = 0
        while _scale(view) < view._max_linear_scale * (1 - 1e-9):
            view._end_wheel_gesture()
            mip = view._mip_for_zoom_factor(1.0)
            want = _notch_scale(view, 1)
            _wheel_units(view, 1200)  # ten notches: always more than one level's worth
            flicks += 1
            assert abs(view._mip_for_zoom_factor(1.0) - mip) == 1, "the flick was not trimmed to one level"
            assert _on_lattice(view) or _scale(view) == pytest.approx(view._max_linear_scale, rel=1e-9)
            assert _scale(view) >= want * (1 - 1e-9)
            assert flicks < 20
        assert flicks >= 2
    finally:
        conftest.close_window(window)
