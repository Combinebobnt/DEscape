"""TASK-270: View > Saved Views, Qt-free. settings.py's per-scenario store
(saved_views.py's cleaning) and the float projection pair a view's centre goes
through in Stepped and Sloped. The window and MapView halves are in
tests/test_saved_views_viewer.py."""

from __future__ import annotations

import math

import pytest
import yaml

from descape import iso_geometry, settings
from descape.saved_views import SavedView

import conftest


def _reread() -> None:
    settings._saved_views = None  # priv: the memo, so the next get reads config.yaml


def test_views_round_trip_per_resolved_path(tmp_path):
    a, b = tmp_path / "a.aoe2scenario", tmp_path / "b.aoe2scenario"
    views = {1: SavedView(12.5, 30.25, 2.0), 9: SavedView(-3.0, 241.0, 0.5)}
    settings.set_saved_views(a, views)
    settings.set_saved_views(b, {2: SavedView(1.0, 2.0, 3.0)})
    assert settings.get_saved_views(a) == views
    assert settings.get_saved_views(tmp_path / "x" / ".." / "a.aoe2scenario") == views
    _reread()
    assert settings.get_saved_views(a) == views
    assert settings.get_saved_views(b) == {2: SavedView(1.0, 2.0, 3.0)}
    assert settings.get_saved_views(tmp_path / "never.aoe2scenario") == {}
    got = settings.get_saved_views(a)
    got[3] = SavedView(0.0, 0.0, 1.0)
    assert 3 not in settings.get_saved_views(a), "a copy, not the store"


@conftest.requires_symlinks
def test_a_symlinked_path_reads_the_same_entry(tmp_path):
    a, link = tmp_path / "a.aoe2scenario", tmp_path / "link.aoe2scenario"
    a.write_bytes(b"a")
    link.symlink_to(a)
    views = {4: SavedView(7.0, 8.0, 2.0)}
    settings.set_saved_views(a, views)
    assert settings.get_saved_views(link) == views
    _reread()
    assert settings.get_saved_views(link) == views


def test_empty_views_remove_the_entry(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    settings.set_saved_views(a, {1: SavedView(1.0, 1.0, 1.0)})
    settings.set_saved_views(b, {1: SavedView(2.0, 2.0, 1.0)})
    settings.set_saved_views(a, {})
    _reread()
    assert settings.get_saved_views(a) == {}
    rows = settings._load_config()["saved_views"]
    assert [row["scenario"] for row in rows] == [settings._scenario_key(b)]


def test_malformed_rows_and_slots_are_dropped(tmp_path):
    good = {"x": 4.0, "y": 5.0, "zoom": 1.5}
    a, b = settings._scenario_key(tmp_path / "a"), settings._scenario_key(tmp_path / "b")
    rows = [
        "junk",
        {"scenario": 3, "views": {1: good}},
        {"views": {1: good}},
        {"scenario": a, "views": "junk"},
        {
            "scenario": b,
            "views": {
                "1": good,
                0: good,
                10: good,
                True: good,
                2: {"x": "4", "y": 5.0, "zoom": 1.0},
                3: {"x": 4.0, "y": 5.0, "zoom": 0},
                4: {"x": math.nan, "y": 5.0, "zoom": 1.0},
                5: {"x": True, "y": 5.0, "zoom": 1.0},
                6: {"x": 10**400, "y": 5.0, "zoom": 1.0},
                7: {"x": 4.0, "y": 5.0, "zoom": math.inf},
                8: {"x": 4, "y": -5, "zoom": 2},
                9: good,
            },
        },
        {"scenario": a, "views": {1: good}},
    ]
    settings.CONFIG_PATH.write_text(yaml.safe_dump({"saved_views": rows}))
    assert settings.get_saved_views(tmp_path / "b") == {8: SavedView(4.0, -5.0, 2.0), 9: SavedView(4.0, 5.0, 1.5)}
    assert settings.get_saved_views(tmp_path / "a") == {1: SavedView(4.0, 5.0, 1.5)}


@pytest.mark.parametrize("raw", ["junk", {"a": 1}, None, 7])
def test_a_non_list_value_reads_as_no_views(tmp_path, raw):
    settings.CONFIG_PATH.write_text(yaml.safe_dump({"saved_views": raw}))
    assert settings.get_saved_views(tmp_path / "a") == {}


def test_the_cap_keeps_the_most_recently_saved_most_recent_first(tmp_path):
    cap = settings.MAX_SAVED_VIEW_SCENARIOS
    paths = [tmp_path / f"m{i}" for i in range(cap + 3)]
    for i, path in enumerate(paths):
        settings.set_saved_views(path, {1: SavedView(float(i), 0.0, 1.0)})
    settings.set_saved_views(paths[3], {1: SavedView(99.0, 0.0, 1.0)})  # re-saved: moves to the front
    _reread()
    keys = [row["scenario"] for row in settings._load_config()["saved_views"]]
    expected = [paths[3]] + [p for p in reversed(paths) if p != paths[3]]
    assert keys == [settings._scenario_key(p) for p in expected[:cap]]
    assert settings.get_saved_views(paths[0]) == {}, "the least recently saved fell off"
    assert settings.get_saved_views(paths[3]) == {1: SavedView(99.0, 0.0, 1.0)}


def test_no_write_when_unchanged(tmp_path, monkeypatch):
    a = tmp_path / "a"
    writes = []
    real = settings._save_config
    monkeypatch.setattr(settings, "_save_config", lambda config: (writes.append(1), real(config)))
    views = {1: SavedView(1.0, 2.0, 3.0)}
    settings.set_saved_views(a, views)
    settings.set_saved_views(a, dict(views))
    settings.set_saved_views(a, {1: SavedView(1.00000001, 2.0, 3.0000000001)})  # equal once rounded
    settings.set_saved_views(tmp_path / "b", {})
    assert len(writes) == 1


def test_stored_values_are_rounded_for_a_readable_file(tmp_path):
    a = tmp_path / "a"
    settings.set_saved_views(a, {4: SavedView(1.234567891, -9.87654321, 3.14159265358979)})
    row = settings._load_config()["saved_views"][0]["views"][4]
    assert row == {"x": 1.2346, "y": -9.8765, "zoom": 3.14159}
    _reread()
    assert settings.get_saved_views(a) == {4: SavedView(1.2346, -9.8765, 3.14159)}


# T2: the float projection pair, over a grid of points on and off a 40x24 map.
_PROJ = iso_geometry.canvas_size_and_origin(40, 24, 64, 0, 3)
_GRID = [(mx / 4 - 5, my / 3 - 4) for mx in range(0, 200, 7) for my in range(0, 100, 5)]


def test_ground_map_point_inverts_the_float_projection_at_rise_zero():
    for mx, my in _GRID:
        sx, sy = iso_geometry.map_point_to_scene_f(mx, my, 0, _PROJ)
        back = iso_geometry.ground_map_point(sx, sy, _PROJ)
        assert back == pytest.approx((mx, my), abs=1e-9), (mx, my)
        # A known rise comes back through sy + rise.
        sx, sy = iso_geometry.map_point_to_scene_f(mx, my, 37, _PROJ)
        assert iso_geometry.ground_map_point(sx, sy + 37, _PROJ) == pytest.approx((mx, my), abs=1e-9)


def test_map_point_to_screen_is_the_float_projection_rounded_where_it_rounds():
    for mx, my in _GRID:
        for rise in (0, 16, 37):
            sx, sy = iso_geometry.map_point_to_scene_f(mx, my, rise, _PROJ)
            # round() before the trailing + half_h, as map_point_to_screen() does (its docstring).
            expected = (round(sx), round(sy - _PROJ.half_h) + _PROJ.half_h)
            assert iso_geometry.map_point_to_screen(mx, my, rise, _PROJ) == expected, (mx, my, rise)
