"""Each terrain style reads its own elevation step setting (GH #288).

Stepped and Flat + Isometric View read `elev_step_pct`, Sloped reads
`sloped_elev_step_pct`. The flat-map oracle in tests/test_sloped_render.py
passes one explicit pct to both renders, so it no longer sees this wiring:
these tests do, at the helper, at both cache constructors and through a
real viewer's style switches.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from descape import render_iso, render_paint, render_sloped, settings
from descape.render_cache import IsoChunkCache, SlopedChunkCache
from testkit.fakes import FakeScenario, SyntheticTile

import conftest

STEPPED_PCT, SLOPED_PCT = 50, 100
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden_blank_120x120.aoe2scenario"


@pytest.fixture
def two_steps(monkeypatch):
    monkeypatch.setattr(settings, "_elev_step_pct", STEPPED_PCT)
    monkeypatch.setattr(settings, "_sloped_elev_step_pct", SLOPED_PCT)


def _scenario(size: int = 20) -> FakeScenario:
    tiles = [SyntheticTile(x=x, y=y, elevation=(x + y) % 3, terrain_id=14) for y in range(size) for x in range(size)]
    return FakeScenario(size, size, tiles, [[] for _ in range(9)])


@pytest.mark.parametrize("stepped, sloped", [(50, 100), (150, 25)])
def test_the_helper_gives_sloped_its_own_stop_and_the_rest_steppeds(monkeypatch, stepped, sloped) -> None:
    monkeypatch.setattr(settings, "_elev_step_pct", stepped)
    monkeypatch.setattr(settings, "_sloped_elev_step_pct", sloped)
    assert {style: render_paint.elev_step_pct_for(style) for style in ("flat", "stepped", "sloped")} == {
        "flat": stepped,
        "stepped": stepped,
        "sloped": sloped,
    }


def test_each_cache_builds_its_ladder_from_its_own_styles_setting(two_steps) -> None:
    scn = _scenario()
    elevations, proj = render_iso.elevations_and_proj(scn)
    assert proj == render_iso.elevations_and_proj(scn, elev_step_pct=STEPPED_PCT)[1]
    assert IsoChunkCache(scn, elevations, proj, proj.tile_px)._levels[0].proj == proj
    s_elev, corner_rise, s_proj = render_sloped.sloped_elevations_and_proj(scn)
    assert s_proj == render_sloped.sloped_elevations_and_proj(scn, elev_step_pct=SLOPED_PCT)[2]
    assert s_proj.elev_step != proj.elev_step
    assert SlopedChunkCache(scn, s_elev, corner_rise, s_proj, s_proj.tile_px)._levels[0].proj == s_proj


def test_an_explicit_override_that_did_not_build_the_proj_fails_loudly(two_steps) -> None:
    scn = _scenario()
    elevations, proj = render_iso.elevations_and_proj(scn)
    with pytest.raises(AssertionError):
        IsoChunkCache(scn, elevations, proj, proj.tile_px, elev_step_pct=SLOPED_PCT)
    s_elev, corner_rise, s_proj = render_sloped.sloped_elevations_and_proj(scn)
    with pytest.raises(AssertionError):
        SlopedChunkCache(scn, s_elev, corner_rise, s_proj, s_proj.tile_px, elev_step_pct=STEPPED_PCT)
    # The matching override is the supported way to render at another stop.
    s_elev, corner_rise, s_proj = render_sloped.sloped_elevations_and_proj(scn, elev_step_pct=STEPPED_PCT)
    SlopedChunkCache(scn, s_elev, corner_rise, s_proj, s_proj.tile_px, elev_step_pct=STEPPED_PCT)


def _centre_point(view) -> tuple[float, float]:
    point = view._pick_map_point(view._viewport_center_scene())  # priv: the restore's own reading
    assert point is not None, "viewport centre is off the map"
    return point


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
@pytest.mark.parametrize(
    "zoom_steps, fx, fy, max_drift",
    [
        # The view restore keeps a fraction of the whole canvas rect, which grows unevenly between the
        # two styles once their steps differ. Measured 2026-11-27 on this 120x120 map at 50 vs 100:
        # 0.03 tiles at fit, 0 at the middle, 2.19-2.28 near the north and south corners at any zoom.
        (0, 0.5, 0.5, 0.1),
        (6, 0.3, 0.6, 1.0),
        (6, 0.5, 0.2, 2.5),
        (6, 0.5, 0.8, 2.5),
    ],
)
def test_a_style_switch_renders_at_that_styles_step_and_keeps_the_view(two_steps, zoom_steps, fx, fy, max_drift) -> None:
    from PyQt5.QtWidgets import QApplication

    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        window.load_scenario(FIXTURE)
        assert window.scenario is not None, f"{FIXTURE.name} failed to load"
        window.resize(1400, 1000)
        window.show()
        QApplication.processEvents()
        window.terrain_style_combo.setCurrentText("Stepped")
        QApplication.processEvents()
        expected = {
            "stepped": render_iso.elevations_and_proj(window.scenario, elev_step_pct=STEPPED_PCT)[1],
            "sloped": render_sloped.sloped_elevations_and_proj(window.scenario, elev_step_pct=SLOPED_PCT)[2],
        }
        assert expected["stepped"].elev_step != expected["sloped"].elev_step

        view = window.map_view
        for _ in range(zoom_steps):
            view.scale(1.25, 1.25)
        view.centerOn(view._map_rect.width() * fx, view._map_rect.height() * fy)
        QApplication.processEvents()
        drifts = []
        for style in ("sloped", "stepped"):
            before = _centre_point(view)
            window.terrain_style_combo.setCurrentText(style.capitalize())
            QApplication.processEvents()
            assert window._terrain_style == style
            assert window._iso_proj == expected[style], style
            assert window._cache._levels[0].proj == expected[style], style
            after = _centre_point(view)
            drifts.append(max(abs(after[0] - before[0]), abs(after[1] - before[1])))
        assert max(drifts) <= max_drift, f"view restore drifted {drifts} tiles across the two switches"

        # Flat + Isometric View draws through Stepped's projection, so it reads Stepped's stop.
        window.terrain_style_combo.setCurrentText("Flat")
        QApplication.processEvents()
        window.iso_action.setChecked(True)
        QApplication.processEvents()
        assert window._render_style == "stepped"
        assert window._iso_proj == expected["stepped"]
    finally:
        conftest.close_window(window)
