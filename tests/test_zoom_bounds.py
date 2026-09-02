"""MapView's zoom floor/ceiling, and that they survive a viewport resize.

Both bounds are anchored to a fit-to-view baseline (see
MapView._capture_zoom_baseline). That baseline used to be read straight off
the current transform, which is only a fit reading because both callers ran
immediately after a fit. Recomputing it on resize from the transform is a
REGRESSION, not a fix: a resize taken while the user is zoomed to 3x fit
re-anchors the floor to that zoom and strands them above fit, unable to zoom
back out to the whole map. _fit_baseline_scale() derives it from geometry
instead, so a resize can't do that.

test_zoom_out_stays_reachable_after_a_resize is the one that discriminates a
correct fix from that regression; every other test here passes under both.

Zoom is driven with view.scale(...) rather than synthetic QWheelEvents -- the
technique tests/test_lazy_viewport.py's docstring already establishes.
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

EQUIVALENCE_CASES = [
    ("Stepped", True, 1400, 1000),
    ("Stepped", True, 1000, 1400),
    ("Stepped", True, 1600, 900),
    ("Flat", True, 1400, 1000),
    ("Flat", True, 1600, 900),
    ("Flat", False, 1400, 1000),
    ("Flat", False, 1000, 1400),
    ("Flat", False, 1100, 1100),
]


def _window(style: str, isometric: bool, width: int, height: int):
    """A shown ViewerWindow fitted AT its final size, so the transform is a
    live fit reading rather than a stale one from the default window size.

    set_isometric() is called TWICE on purpose, and that is load-bearing for
    the equivalence test. fitInView (and set_isometric's own uniform-scale
    path) computes against the viewport as it is when it runs; applying the
    resulting transform can add or remove a scrollbar, which changes
    viewport().rect() afterwards. So the first fit's transform can be a few
    percent off the fit for the viewport that actually ended up on screen.
    The second fit sees the settled viewport and converges exactly -- measured
    across 15 style/size combinations, every one converged on the second fit
    and stayed converged on a third.
    """
    from PyQt5.QtWidgets import QApplication

    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    window.load_scenario(FIXTURE)
    if window.scenario is None:
        window.close()
        pytest.skip(f"{FIXTURE.name} failed to load")
    window.terrain_style_combo.setCurrentText(style)
    window.resize(width, height)
    window.show()
    QApplication.processEvents()
    QApplication.processEvents()
    window.map_view.set_isometric(isometric)
    QApplication.processEvents()
    window.map_view.set_isometric(isometric)
    QApplication.processEvents()
    return window


@pytest.mark.parametrize("style,isometric,width,height", EQUIVALENCE_CASES)
def test_the_derived_fit_scale_reproduces_the_real_transform(style, isometric, width, height) -> None:
    """_fit_baseline_scale() is hand-rolled arithmetic standing in for what
    fitInView (and set_isometric's own uniform-scale path) actually did. This
    is what proves it reproduces their numbers rather than merely resembling
    them -- including fitInView's undocumented 2 px per side, and the
    squash/rotate/uniform-scale composition on the isometric path.

    Covers all three branches of _fit_baseline_scale: stepped, flat+isometric,
    and flat+non-isometric. See _window() on why the fit runs twice.
    """
    window = _window(style, isometric, width, height)
    try:
        view = window.map_view
        derived = view._fit_baseline_scale()
        assert derived is not None
        actual = abs(view.transform().determinant()) ** 0.5
        assert derived == pytest.approx(actual, rel=1e-9)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_bare_view_with_no_source_still_gets_its_bounds_set() -> None:
    """_fit_baseline_scale() returns None with no map -- there is no fit to
    derive. _capture_zoom_baseline must fall back to the transform there
    rather than leaving the bounds unset."""
    from descape.map_view import MapView

    conftest.ensure_qapp()
    noop = lambda *a, **k: None  # noqa: E731
    view = MapView(noop, noop, noop, noop, noop, noop, noop, noop, noop, noop, noop, noop, noop, noop)
    assert view._map_rect is None
    assert view._fit_baseline_scale() is None
    view._capture_zoom_baseline()
    baseline = abs(view.transform().determinant()) ** 0.5
    assert view._min_linear_scale == pytest.approx(MapView.MIN_ZOOM_FRACTION_OF_FIT * baseline)
    assert view._max_linear_scale == pytest.approx(MapView.MAX_ZOOM_MULTIPLE_OF_FIT * baseline)


def test_zoom_out_stays_reachable_after_a_resize() -> None:
    """THE discriminating test: fit must stay reachable after a resize taken
    while zoomed in.

    Careful with what this asserts. Resizing shorter CHANGES the fit scale, so
    "returns to the fit captured before the resize" would fail a correct fix.
    Work it out from MIN_ZOOM_FRACTION_OF_FIT = 0.5 instead:

    - Correct (geometry): floor = 0.5 * new_fit, strictly BELOW the new fit, so
      fit stays reachable.
    - Regression (transform read on resize): the resize happens at ~1.95x fit,
      so floor = 0.5 * 1.95 * old_fit -- ABOVE the new fit, which is only
      ~0.66 * old_fit after the shorter resize. Zoom-out jams before reaching
      the whole map.

    The mip ladder is held out at 1.0 rather than chosen away. The plan for
    this work assumed flat/non-isometric would give _coarsest_mip_scale() ==
    1.0; measured, it is 8.0 there, and no elev_step_pct from 10 to 100
    changes that. A depth of 8 divides the floor by 8 and would let fit stay
    reachable under the regression too, i.e. it would stop this test
    discriminating anything. Depth arithmetic is already covered thoroughly by
    tests/test_mip_viewer.py and is orthogonal to what this pins.
    """
    from PyQt5.QtWidgets import QApplication

    window = _window("Flat", False, 1400, 1000)
    try:
        view = window.map_view
        view._coarsest_mip_scale = lambda: 1.0

        # Zoom in well past fit, then resize the viewport under the user.
        for _ in range(3):
            view.scale(1.25, 1.25)
        QApplication.processEvents()
        window.resize(1400, 700)
        QApplication.processEvents()
        QApplication.processEvents()

        new_fit = view._fit_baseline_scale()
        assert new_fit is not None

        # Zoom out the way wheelEvent does until its own guard refuses.
        for _ in range(60):
            current = abs(view.transform().determinant()) ** 0.5
            if current * 0.8 < view._min_linear_scale:
                break
            view.scale(0.8, 0.8)
        reachable = abs(view.transform().determinant()) ** 0.5

        # The whole map is still reachable. Under the regression this is
        # ~1.5x new_fit and the assertion fails.
        assert reachable <= new_fit * (1.0 + 1e-9)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_zoom_ceiling_tracks_a_resize_too() -> None:
    """The same staleness in the other direction: growing the viewport raises
    the fit scale, and the ceiling must follow rather than pinning the user to
    the old window's maximum.

    _finest_mip_scale() held at 1.0 for the same reason
    test_zoom_out_stays_reachable_after_a_resize holds _coarsest_mip_scale()
    at 1.0: mip depth arithmetic is covered thoroughly by
    tests/test_mip_viewer.py and is orthogonal to what this test pins (resize
    staleness, not mip depth). Flat/non-isometric measures 0.5 here (one
    level finer than the reference), which would otherwise scale the
    expected ceiling by 2x for a reason unrelated to the resize this test is
    about."""
    from PyQt5.QtWidgets import QApplication

    from descape.map_view import MapView

    window = _window("Flat", False, 1000, 700)
    try:
        view = window.map_view
        view._finest_mip_scale = lambda: 1.0
        window.resize(1600, 1200)
        QApplication.processEvents()
        QApplication.processEvents()
        fit = view._fit_baseline_scale()
        assert fit is not None
        assert view._max_linear_scale == pytest.approx(MapView.MAX_ZOOM_MULTIPLE_OF_FIT * fit)
    finally:
        window.edit_history.mark_saved()
        window.close()
