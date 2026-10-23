"""A garrison edit repaints its host's flag (GH #203, TASK-264 Step 4).

A link, an unload, Garrison Add or a single delete names only the occupant in
its UnitSplice list, while the host's garrison flag appears or clears. Where
the occupant sits on the host's own tile, as every UI link path puts it, the
cache's shared-tile component splice already redraws the host. An occupant a
file stores off its host's tile (the fixture's Villager 203) touches none of
the host's tiles, so the viewer adds a re-anchor splice per flipped host
(_flag_flip_splices()). Each check compares the visible level, as the
window's cache serves it, against a fresh cache built from the same scenario:
a flag left on, or one never drawn, shows as stale pixels.

Its own module, not tests/test_unit_edit_viewer.py: that one strips the
install (its autouse _no_install_env), so no sprite and no flag would draw
and every check there would pass vacuously. The synthetic install below
draws every const these edits touch, each host with a flag at a real offset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtCore import Qt

from descape import asset_source, iso_geometry, render, unit_sprites
from descape.render_cache import IsoChunkCache, SlopedChunkCache

import conftest
from test_unit_sprites import build_sld

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"
T = unit_sprites.NATIVE_TILE_W
HOUSE, TOWER, VILLAGER, ARCHER = 70, 79, 83, 4
REF_HOUSE, REF_VILLAGER_IN_HOUSE, REF_ARCHER_P2 = 200, 203, 300
TOWER_POINT = (16.5, 14.5)
# Every edit below lands inside these tiles: the house at (10, 10), its villager, the tower, the archer.
FOCUS = (4, 4, 28, 28)


@pytest.fixture
def art(tmp_path, monkeypatch):
    """A synthetic install for the four consts edited here, the two hosts
    carrying a flag at their real .dat offsets. Every canvas and offset stays
    inside the real MAX_SPRITE_REACH_*. No player colour: build_sld's colours
    are shades of red, which player 1's blue tint would turn all to black, and
    a flag black on a black house clears without changing a pixel."""
    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)

    def add(name, canvas, colour_base, angles=1):
        art = build_sld(angles, canvas=canvas, colour_base=colour_base, playercolor=False)
        (graphics / f"{name}.sld").write_bytes(art)
        return {"graphic_id": 1, "file_name": name, "angle_count": angles, "mirroring_mode": 0, "frame_count": 1}

    add("t_flag_x1", T // 2, 25)

    def flag(const, dx, dy):
        return {"unit_id": const, "file_name": "t_flag_x1", "angle_count": 1, "frame_count": 1, "dx": dx, "dy": dy}

    table = {
        HOUSE: {**add("t_house_x1", 2 * T, 3), "garrison_flags": [flag(HOUSE, -17, -61)]},
        TOWER: {**add("t_tower_x1", 2 * T, 7), "garrison_flags": [flag(TOWER, -1, -179)]},
        VILLAGER: add("t_villager_x1", T, 11, angles=16),
        ARCHER: add("t_archer_x1", T, 15, angles=16),
    }
    monkeypatch.setattr(unit_sprites, "graphic_map", lambda: table)
    asset_source.set_install_path_override(tmp_path)
    unit_sprites.clear_caches()
    yield table
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()


def _window(style: str, monkeypatch, show_garrisoned: bool):
    """The units fixture in `style`, Units mode, sprites on, with the focus
    tiles' rect of level 0 pinned as the visible level and made resident."""
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.load_scenario(FIXTURE_PATH)
    assert window.scenario is not None, "fixture failed to load"
    window.terrain_style_combo.setCurrentText(style)
    window.show_garrisoned_action.setChecked(show_garrisoned)
    window.mode_combo.setCurrentText("Units")
    cache = window._cache
    assert cache.sprites_enabled
    assert isinstance(cache, SlopedChunkCache if style == "Sloped" else IsoChunkCache)
    window._test_rect = _focus_rect(cache)
    cx0, cy0, cx1, cy1 = cache.chunk_index_range(0, *window._test_rect)
    monkeypatch.setattr(window.map_view, "viewport_chunk_target", lambda: (0, cx0, cy0, cx1, cy1))
    _level(window)
    return window


def _focus_rect(cache) -> tuple[int, int, int, int]:
    """Level 0's rect over FOCUS, padded by the sprite reach and clamped."""
    x0, y0, x1, y1 = FOCUS
    corners = ((x0, y0), (x1, y0), (x0, y1), (x1, y1))
    boxes = np.array([iso_geometry.tile_screen_bounds_swept(x, y, cache.proj) for x, y in corners])
    left, up, right, down = render._sprite_reach_px(cache.proj)
    w, h = cache.canvas_dims(0)
    return (
        max(0, int(boxes[:, 0].min()) - left), max(0, int(boxes[:, 1].min()) - up),
        min(w, int(boxes[:, 2].max()) + right), min(h, int(boxes[:, 3].max()) + down),
    )


def _level(window) -> np.ndarray:
    return window._cache.render_rect(*window._test_rect, mip=0).copy()


def _fresh(window) -> np.ndarray:
    cache = window._cache
    common = {"unit_filter": cache.unit_filter, "sprites": True, "layers": cache.layers}
    if isinstance(cache, SlopedChunkCache):
        fresh = SlopedChunkCache(
            window.scenario, cache.elevations.copy(), cache.corner_rise.copy(), cache.proj, cache.tile_px, **common
        )
    else:
        fresh = IsoChunkCache(window.scenario, cache.elevations.copy(), cache.proj, cache.tile_px, **common)
    fresh.set_grid(cache.grid)
    return fresh.render_rect(*window._test_rect, mip=0)


def _check(window, what: str) -> np.ndarray:
    now, fresh = _level(window), _fresh(window)
    if not np.array_equal(now, fresh):
        ys, xs = np.nonzero((now != fresh).any(axis=2))
        pytest.fail(f"{what}: {len(xs)} stale px, in x {xs.min()}..{xs.max()}, y {ys.min()}..{ys.max()}")
    return now


def _live(window, reference_id: int):
    return next(u for units in window.scenario.unit_manager.units for u in units if u.reference_id == reference_id)


def _entry(window, player_id: int, reference_id: int):
    entry = window.map_view._unit_index.entry_for_key((player_id, reference_id))
    assert entry is not None, (player_id, reference_id)
    return entry


def _place_tower(window):
    """Player 1's empty Watch Tower on a free tile, selected (a setup edit)."""
    model = window._ensure_unit_edits()
    with window._unit_edit(model, "Place tower", [1]):
        tower = model.add(1, TOWER, *TOWER_POINT)
    window._rebuild_unit_index()
    window._selection = [(1, tower.reference_id)]
    window._refresh_selection_view()
    return tower


class _Steps:
    """Runs one edit, then its undo and redo, checking each against a fresh
    render and that the host's flag really flipped (so no check is vacuous)."""

    def __init__(self, window):
        self.window = window
        self.flips: list[frozenset[int]] = []
        real = window._flag_flip_splices

        def recorded(changed):
            out = real(changed)
            self.flips.append(frozenset(s.unit.reference_id for s in out))
            return out

        window._flag_flip_splices = recorded

    def run(self, what: str, edit, host_ref: int, flagged_after: bool) -> None:
        window = self.window
        before = _check(window, f"before {what}")
        assert (host_ref in render.garrison_host_ids(window.scenario)) is not flagged_after
        edit()
        assert (host_ref in render.garrison_host_ids(window.scenario)) is flagged_after, what
        after = _check(window, what)
        assert not np.array_equal(before, after), f"{what} changed no pixel"
        window.undo()
        assert np.array_equal(_check(window, f"undo of {what}"), before)
        window.redo()
        assert np.array_equal(_check(window, f"redo of {what}"), after)


@pytest.mark.parametrize("show_garrisoned", [False, True], ids=["occupants_hidden", "occupants_shown"])
@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_link_unload_and_garrison_add_repaint_the_hosts_flag(art, monkeypatch, style, show_garrisoned) -> None:
    """Equality only, not a control: each of these puts the occupant on the
    tower's point or takes it off it, so the cache's shared-tile component
    splice redraws the tower with or without the flip splices."""
    window = _window(style, monkeypatch, show_garrisoned)
    try:
        tower = _place_tower(window)
        steps = _Steps(window)
        host = tower.reference_id
        steps.run("link", lambda: window._garrison_units(_entry(window, 1, host), [_entry(window, 2, REF_ARCHER_P2)]),
                  host, True)
        archer = _live(window, REF_ARCHER_P2)
        steps.run("unload", lambda: window._unload_units(_entry(window, 1, host), [(2, archer)]), host, False)
        steps.run("garrison add", lambda: window._garrison_add_const(_entry(window, 1, host), VILLAGER), host, True)
    finally:
        conftest.close_window(window)


def _off_tile_occupant(window):
    """The fixture's Villager 203, stored two tiles from its House 200: no
    tile of its own touches the house, so only the flip splices reach it."""
    villager = _live(window, REF_VILLAGER_IN_HOUSE)
    house = _live(window, REF_HOUSE)
    assert villager.garrisoned_in_id == REF_HOUSE
    assert (int(villager.x), int(villager.y)) not in set(render.unit_occupied_tiles(house, 120, 120))
    return villager


def _delete_off_tile_occupant(window) -> None:
    """Single Delete, a scoped edit naming the villager alone."""
    window._selection = [(1, REF_VILLAGER_IN_HOUSE)]
    window._refresh_selection_view()
    window.on_unit_delete(Qt.NoModifier)


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_deleting_an_off_tile_occupant_clears_its_hosts_flag(art, monkeypatch, style) -> None:
    window = _window(style, monkeypatch, show_garrisoned=True)
    try:
        _off_tile_occupant(window)
        steps = _Steps(window)
        steps.run("delete", lambda: _delete_off_tile_occupant(window), REF_HOUSE, False)
        # The delete, its undo and its redo each reached the house through the flip splices alone.
        assert steps.flips == [frozenset({REF_HOUSE})] * 3, steps.flips
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("step", ["delete", "undo"])
@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_without_the_flip_splices_an_off_tile_occupants_host_is_left_stale(art, monkeypatch, style, step) -> None:
    """The positive control for the off-tile case above (plan amendment 2):
    with the host splices taken away, the delete leaves the house's flag on
    screen, and the undo of a correct delete fails to bring it back."""
    window = _window(style, monkeypatch, show_garrisoned=True)
    try:
        _off_tile_occupant(window)
        if step == "undo":
            _delete_off_tile_occupant(window)
            _check(window, "delete with the flip splices")
        window._flag_flip_splices = lambda changed: []
        _check(window, f"before the {step}")
        if step == "delete":
            _delete_off_tile_occupant(window)
        else:
            window.undo()
        assert (REF_HOUSE in render.garrison_host_ids(window.scenario)) is (step == "undo")
        stale = int((_level(window) != _fresh(window)).any(axis=2).sum())
        # One whole synthetic flag, T // 2 native px square at the template's 2/3 scale.
        assert stale == (round(T // 2 * unit_sprites.sprite_scale(window._cache.proj.half_w))) ** 2, stale
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_a_wholesale_garrison_delete_then_its_scoped_undo(art, monkeypatch, style) -> None:
    """Garrison Delete repaints wholesale, so the cache must restamp its hosts
    there (_record_flag_hosts()): the undo that follows is scoped, and it
    finds the house's flag flip only against a current stamp."""
    window = _window(style, monkeypatch, show_garrisoned=True)
    try:
        _off_tile_occupant(window)
        steps = _Steps(window)
        before = _check(window, "before")
        window._selection = [(1, REF_HOUSE)]
        window._refresh_selection_view()
        window._on_garrison_delete([REF_VILLAGER_IN_HOUSE])
        assert REF_HOUSE not in render.garrison_host_ids(window.scenario)
        after = _check(window, "garrison delete")
        assert not np.array_equal(before, after)
        assert steps.flips == [], "Garrison Delete is the wholesale path"
        window.undo()
        assert np.array_equal(_check(window, "undo of garrison delete"), before)
        assert steps.flips == [frozenset({REF_HOUSE})], "the undo is the scoped path"
    finally:
        conftest.close_window(window)
