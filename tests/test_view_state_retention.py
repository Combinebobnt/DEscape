"""ViewerWindow._render_current()/MapView.set_source()'s reset_view parameter.

A full map redraw (ViewerWindow._apply_dirty()'s fallback when an edit's
incremental patch path declines, refresh_map()'s settings-triggered
re-render, or an Elevation View/terrain-style switch) used to always end in
MapView.set_isometric()'s fit-to-view path, silently yanking the user's
zoom/pan back to "whole map fits the viewport" even though the same document
was still on screen. reset_view=False lets those callers keep set_source()'s
side effects (fresh cache, scene rebuild) without losing the view.

The captured/restored view state is RELATIVE (zoom as a multiple of the
current fit baseline, pan as a fraction of map_rect), not absolute -- a
graphics-quality change or a terrain-style switch both rescale map_rect's
dimensions (tile_pixels_for_map() scales with quality; Stepped/Sloped's
projected canvas differs in size from Flat's plain tile grid), and an
absolute scene coordinate captured before that rescale lands on the wrong
part of the differently-scaled map afterward.

Driven the same way tests/test_zoom_bounds.py drives zoom: view.scale(...)
directly rather than synthetic QWheelEvents, reusing that file's _window()
fixture pattern.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden_blank_120x120.aoe2scenario"


def _window(style: str, isometric: bool):
    from PyQt5.QtWidgets import QApplication

    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    window.load_scenario(FIXTURE)
    if window.scenario is None:
        window.close()
        pytest.skip(f"{FIXTURE.name} failed to load")
    window.terrain_style_combo.setCurrentText(style)
    window.resize(1400, 1000)
    window.show()
    QApplication.processEvents()
    QApplication.processEvents()
    window.map_view.set_isometric(isometric)
    QApplication.processEvents()
    window.map_view.set_isometric(isometric)
    QApplication.processEvents()
    return window


def _current_scale(view) -> float:
    return abs(view.transform().determinant()) ** 0.5


def _current_center(view):
    return view.mapToScene(view.viewport().rect().center())


@pytest.mark.parametrize("style,isometric", [("Flat", False), ("Flat", True), ("Stepped", False)])
def test_reset_view_false_preserves_zoom_and_pan(style, isometric) -> None:
    from PyQt5.QtWidgets import QApplication

    window = _window(style, isometric)
    try:
        view = window.map_view
        for _ in range(3):
            view.scale(1.25, 1.25)
        view.centerOn(view._map_rect.center().x() * 0.25, view._map_rect.center().y() * 0.25)
        QApplication.processEvents()

        before_scale = _current_scale(view)
        before_center = _current_center(view)
        fit = view._fit_baseline_scale()
        assert fit is not None
        assert before_scale > fit * 1.5  # actually zoomed in, not sitting at fit

        window._render_current(reset_view=False)
        QApplication.processEvents()

        after_scale = _current_scale(view)
        after_center = _current_center(view)
        assert after_scale == pytest.approx(before_scale, rel=1e-6)
        # QGraphicsView's scrollbars are integer-valued in VIEW pixels, so
        # centerOn() is lossy by up to ~1 view pixel each time it's called --
        # once capturing before_center, once restoring it. At scene-units-
        # per-view-pixel (1 / after_scale) that's a few scene units at this
        # zoom, nowhere near "reset to fit" (hundreds of units on a 7680-wide
        # map) if this regressed.
        pixel_tolerance = 3.0 / after_scale
        assert after_center.x() == pytest.approx(before_center.x(), abs=pixel_tolerance)
        assert after_center.y() == pytest.approx(before_center.y(), abs=pixel_tolerance)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_reset_view_default_still_fits() -> None:
    """Every pre-existing caller passes no reset_view argument at all, so the
    default must keep resetting to fit -- this is the regression guard for
    that default, not a test of the new behavior."""
    from PyQt5.QtWidgets import QApplication

    window = _window("Flat", False)
    try:
        view = window.map_view
        for _ in range(3):
            view.scale(1.25, 1.25)
        QApplication.processEvents()
        fit = view._fit_baseline_scale()
        assert fit is not None
        assert _current_scale(view) > fit * 1.5

        window._render_current()
        QApplication.processEvents()

        assert _current_scale(view) == pytest.approx(fit, rel=1e-6)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_reset_view_false_survives_a_graphics_quality_change() -> None:
    """refresh_map() (reset_view=False) is exactly what Settings > Appearance's
    graphics-quality slider fires -- see ViewerWindow._apply_graphics_quality.
    That setting rescales tile_pixels_for_map()'s result, which rescales the
    WHOLE scene coordinate system (map_rect gets wider/narrower in scene
    units for the same map), so an absolute captured center point lands on a
    different part of the map after the change even though relative_scale is
    preserved -- the fraction-of-map_rect form this pins is what makes the
    restored center track the map itself instead of a stale coordinate."""
    import descape.settings as settings_module
    from PyQt5.QtWidgets import QApplication

    window = _window("Flat", False)
    try:
        view = window.map_view
        for _ in range(3):
            view.scale(1.25, 1.25)
        # Off-center, not the map's midpoint -- a midpoint survives a uniform
        # rescale by coincidence even with the bug this test is pinned against.
        view.centerOn(view._map_rect.width() * 0.2, view._map_rect.height() * 0.7)
        QApplication.processEvents()

        before_scale = _current_scale(view)
        before_fit = view._fit_baseline_scale()
        before_rect = view._map_rect
        before_center = _current_center(view)
        before_frac = (
            (before_center.x() - before_rect.left()) / before_rect.width(),
            (before_center.y() - before_rect.top()) / before_rect.height(),
        )

        # DOWN, not up: tile_pixels_for_map() clamps quality 4 ("Enhanced")'s
        # doubling to a no-op on a map this size (render.py's own "small map"
        # branch), so an upward nudge wouldn't actually rescale anything here.
        # GRAPHICS_QUALITY_MIN ("Potatest") halves twice, well clear of that.
        old_quality = settings_module.get_graphics_quality()
        settings_module._graphics_quality = settings_module.GRAPHICS_QUALITY_MIN
        assert settings_module.get_graphics_quality() != old_quality  # actually changed something
        window.refresh_map()
        QApplication.processEvents()

        after_scale = _current_scale(view)
        after_fit = view._fit_baseline_scale()
        after_rect = view._map_rect
        assert after_rect.width() != pytest.approx(before_rect.width())  # tile_pixels did rescale
        after_center = _current_center(view)
        after_frac = (
            (after_center.x() - after_rect.left()) / after_rect.width(),
            (after_center.y() - after_rect.top()) / after_rect.height(),
        )

        # ABSOLUTE scale legitimately changes here (map_rect shrank 4x, so
        # the same real-world view needs a 4x bigger scale) -- what must be
        # preserved is the zoom relative to each render's own fit baseline.
        assert after_scale / after_fit == pytest.approx(before_scale / before_fit, rel=1e-6)
        # Same view-pixel quantization slack as test_reset_view_false_
        # preserves_zoom_and_pan above, expressed per axis as a map_rect
        # fraction instead of a scene distance.
        scene_tolerance = 3.0 / after_scale
        assert after_frac[0] == pytest.approx(before_frac[0], abs=scene_tolerance / after_rect.width())
        assert after_frac[1] == pytest.approx(before_frac[1], abs=scene_tolerance / after_rect.height())
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_terrain_style_switch_approximates_the_same_view() -> None:
    """on_terrain_style_changed() also passes reset_view=False now: a style
    switch changes map_rect's dimensions (Flat's plain tile grid vs Stepped's
    isometric-projected canvas), same class of rescale as the graphics-
    quality test above, plus the projection itself changes shape -- so this
    only pins the zoom-relative-to-fit and map_rect-fraction surviving the
    switch, not a pixel-exact reproduction of the same on-screen content."""
    from PyQt5.QtWidgets import QApplication

    window = _window("Flat", False)
    try:
        view = window.map_view
        for _ in range(3):
            view.scale(1.25, 1.25)
        view.centerOn(view._map_rect.width() * 0.2, view._map_rect.height() * 0.7)
        QApplication.processEvents()

        before_scale = _current_scale(view)
        before_fit = view._fit_baseline_scale()
        before_rect = view._map_rect
        before_center = _current_center(view)
        before_frac = (
            (before_center.x() - before_rect.left()) / before_rect.width(),
            (before_center.y() - before_rect.top()) / before_rect.height(),
        )

        window.terrain_style_combo.setCurrentText("Stepped")
        QApplication.processEvents()
        assert window._terrain_style == "stepped"

        after_scale = _current_scale(view)
        after_fit = view._fit_baseline_scale()
        after_rect = view._map_rect
        after_center = _current_center(view)
        after_frac = (
            (after_center.x() - after_rect.left()) / after_rect.width(),
            (after_center.y() - after_rect.top()) / after_rect.height(),
        )

        assert after_scale / after_fit == pytest.approx(before_scale / before_fit, rel=1e-6)
        scene_tolerance = 3.0 / after_scale
        assert after_frac[0] == pytest.approx(before_frac[0], abs=scene_tolerance / after_rect.width())
        assert after_frac[1] == pytest.approx(before_frac[1], abs=scene_tolerance / after_rect.height())
    finally:
        window.edit_history.mark_saved()
        window.close()
