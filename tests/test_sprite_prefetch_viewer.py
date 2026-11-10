"""The viewer starts the sprite prefetch (TASK-031.47) on a load, right after
the parse and before the chunk cache exists, only for Stepped and Sloped with
sprites on, and drops it on the first paint and on any later render. The
prefetch itself is pinned in tests/test_sprite_prefetch.py."""

from __future__ import annotations

from pathlib import Path

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

UNITS_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"


@pytest.fixture
def recorded(monkeypatch):
    """(events, window): prefetch and discard calls and IsoChunkCache constructions, in order."""
    from descape import sprite_prefetch, viewer

    events: list = []
    real_prefetch = sprite_prefetch.prefetch_level_sprites

    def prefetch(scenario, unit_filter, max_fit_half_w=None):
        submitted = real_prefetch(scenario, unit_filter, max_fit_half_w)
        events.append(("prefetch", scenario, unit_filter, max_fit_half_w, submitted))
        return submitted

    monkeypatch.setattr(sprite_prefetch, "prefetch_level_sprites", prefetch)
    real_discard = sprite_prefetch.discard
    monkeypatch.setattr(sprite_prefetch, "discard", lambda: (events.append(("discard",)), real_discard())[1])

    class _RecordingCache(viewer.IsoChunkCache):
        def __init__(self, *args, **kwargs) -> None:
            events.append(("cache",))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(viewer, "IsoChunkCache", _RecordingCache)
    window = conftest.stepped_window(UNITS_FIXTURE, show=False)
    yield events, window
    conftest.close_window(window)


def _kinds(events) -> list[str]:
    return [event[0] for event in events]


def test_a_load_starts_the_prefetch_before_building_the_cache(recorded):
    events, window = recorded
    kinds = _kinds(events)
    assert "prefetch" in kinds and kinds.index("prefetch") < kinds.index("cache"), kinds
    _kind, scenario, unit_filter, max_fit_half_w, _submitted = next(e for e in events if e[0] == "prefetch")
    assert scenario is window.scenario and unit_filter == window._unit_filter
    # The viewport's width at load time (an unshown window's resizes after) over the map's diamond span.
    mm = window.scenario.map_manager
    assert 0 < max_fit_half_w * (mm.map_width + mm.map_height) < 10_000


def test_a_re_render_starts_none_and_discards(recorded):
    events, window = recorded
    events.clear()
    window._render_current(reset_view=False)
    kinds = _kinds(events)
    assert "prefetch" not in kinds and "discard" in kinds and kinds.index("discard") < kinds.index("cache"), kinds


def test_sprites_off_starts_none(recorded):
    events, window = recorded
    window._on_sprites_toggled(False)
    events.clear()
    window.load_scenario(UNITS_FIXTURE)
    assert "prefetch" not in _kinds(events), events


def test_top_down_flat_starts_none(recorded):
    events, window = recorded
    window.terrain_style_combo.setCurrentText("Flat")
    window.iso_action.setChecked(False)
    assert window._render_style == "flat", "fixture no longer reaches real Flat"
    events.clear()
    window.load_scenario(UNITS_FIXTURE)
    assert "prefetch" not in _kinds(events), events


def test_sloped_starts_one(recorded):
    events, window = recorded
    window.terrain_style_combo.setCurrentText("Sloped")
    assert window._render_style == "sloped"
    events.clear()
    window.load_scenario(UNITS_FIXTURE)
    assert "prefetch" in _kinds(events), events


def test_the_numpy_backend_still_calls_it(recorded):
    """The viewer leaves the backend gate to sprite_prefetch; that it then
    submits nothing is pinned in test_sprite_prefetch.py, against a real
    install (this fixture has none, so every load here submits nothing)."""
    from descape import composite_backend

    events, window = recorded
    events.clear()
    with composite_backend.use_backend("numpy"):
        window.load_scenario(UNITS_FIXTURE)
    assert len([e for e in events if e[0] == "prefetch"]) == 1, events


def test_the_first_paint_discards(recorded):
    events, window = recorded
    events.clear()
    window._on_canvas_paint_timed(0.001, 0)
    assert ("discard",) in events
    # With no report pending too: the early return comes after the discard.
    window._pending_paint_report = None
    events.clear()
    window._on_canvas_paint_timed(0.001, 0)
    assert ("discard",) in events
