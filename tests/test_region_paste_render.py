"""Paste Region's unit-source refresh order (copy-paste perf plan, Step 3):
a paste carrying units refreshes the cache's unit sources BEFORE its tile
patch, so the patch's single level rebuild sees consistent units, elevations
and units_by_tile, and no later gen bump throws that build away. Since the
paste-undo-membership plan (Step 3) that refresh is a real add-splice batch:
it splices the levels, or updates units_by_tile in place when the batch is
refused or the elevation patch will bump the gen anyway
(_elevation_bumps_anyway()). Checked against a fresh render (pixels) and by
counting level builds across the paste and the next full paint. Stepped, the
style the stress log's stalls were in.

Same offscreen technique as tests/test_region_paste.py; every window here
is closed through edit_history.mark_saved() first.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_region_paste import _TERRAIN_A, _make_region
from test_sprite_edit_bbox import CONST as SPRITE_CONST
from test_sprite_edit_bbox import sprite_install  # noqa: F401 -- pytest fixture, imported for its name

from descape import render, render_cache

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

MARK_CONST = 83  # a villager: a coloured mark with sprites off
MILL_CONST = 68  # a 2x2 building
RUN_WALL_CONST = 72  # a 1x1 rotation-variant wall: adding one refuses the splice (const:add)
SRC = (10, 10, 14, 14)
DST = (40, 40)


def _canvas(window, mip: int = 0) -> np.ndarray:
    canvas_w, canvas_h = window._cache.canvas_dims(mip)
    return window._cache.render_rect(0, 0, canvas_w, canvas_h, mip=mip).copy()


def _visible_mip(window) -> int:
    """The level _apply_dirty's patch rebuilds after a gen bump (its
    rebuild_levels): the viewport's own, whatever the unshown window reports."""
    target = window.map_view.viewport_chunk_target()
    assert target is not None, "no visible level, so the patch rebuilds every level"
    return target[0]


def _fresh(window) -> np.ndarray:
    canvas_w, canvas_h = window._cache.canvas_dims(0)
    full = render.render_terrain_iso_with_proj(
        window.scenario, with_units=True, with_sprites=window._cache.sprites_enabled
    )[0]
    return full[:canvas_h, :canvas_w]


def _seeded_window(const: int, sprites: bool = False, block: str = "plain"):
    """Units in the source block and already standing in the paste's target
    block (a populated area), a raised source block copied with all three
    categories checked, and the whole mip-0 canvas resident. `block` adds to
    the source block: `wall` a rotation-variant wall (the paste refuses
    `const:add`), `garrison` a GAIA mill holding a player-2 villager on its own
    tile (two pasted units on one tile: a component splice)."""
    window = conftest.terrain_edit_window()
    window.show_sprites_action.setChecked(sprites)
    assert window._cache.sprites_enabled is sprites
    window._on_tool_selected("select")
    model = window._ensure_unit_edits()
    model.begin_unit_edit([0, 1, 2])
    for i in range(3):
        model.add(player=1, unit_const=const, x=SRC[0] + 0.5 + i, y=SRC[1] + 1.5, z=0.0, rotation=0.0)
        model.add(player=2, unit_const=const, x=DST[0] + 0.5 + i, y=DST[1] + 1.5, z=0.0, rotation=0.0)
    if block == "wall":
        model.add(player=1, unit_const=RUN_WALL_CONST, x=SRC[0] + 0.5, y=SRC[1] + 3.5, z=0.0, rotation=0.0)
    elif block == "garrison":
        mill = model.add(player=0, unit_const=MILL_CONST, x=SRC[0] + 3.0, y=SRC[1] + 3.0, z=0.0, rotation=0.0)
        model.add(
            player=2, unit_const=MARK_CONST, x=SRC[0] + 3.0, y=SRC[1] + 3.0, z=0.0, rotation=0.0,
            garrisoned_in_id=mill.reference_id,
        )
        window.show_garrisoned_action.setChecked(True)  # as _fresh()'s default filter draws it
    model.commit_unit_edit("seed", window.edit_history)
    window._after_unit_mutation()
    _make_region(window, *SRC, _TERRAIN_A, elevation=2)
    for check in (window.paste_terrain_check, window.paste_elevation_check, window.paste_units_check):
        check.setChecked(True)
    _canvas(window)
    return window


def _paste(window) -> None:
    window._hover_tile = DST
    window.paste_region()


@pytest.mark.parametrize("sprites", [False, True])
def test_a_units_and_elevation_paste_over_units_matches_a_fresh_render(sprites, request) -> None:
    if sprites:
        request.getfixturevalue("sprite_install")
    window = _seeded_window(SPRITE_CONST if sprites else MARK_CONST, sprites)
    try:
        before = _canvas(window)
        _paste(window)
        after = _canvas(window)
        assert not np.array_equal(after, before), "the paste changed nothing on screen"
        assert np.array_equal(after, _fresh(window))
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_multi_unit_paste_depoisons_the_unit_classes_once(monkeypatch) -> None:
    """Plan Step 4: the paste's add() loop runs inside UnitEditModel.batch_adds()."""
    from descape import library_compat

    window = _seeded_window(MARK_CONST)
    try:
        assert len(window._region_clipboard.units) >= 3, "the block carries too few units to prove a batch"
        calls = []
        real = library_compat.depoison
        monkeypatch.setattr(library_compat, "depoison", lambda: (calls.append(1), real())[1])
        _paste(window)
        assert len(calls) == 1, f"expected one depoison() for the whole paste, got {len(calls)}"
    finally:
        window.edit_history.mark_saved()
        window.close()


def _count_builds(window, monkeypatch) -> list[int]:
    """Every level install from here on, by mip: a real build or a warm's install."""
    cache = window._cache
    builds: list[int] = []
    real_commit = cache._commit_level

    def commit(level_mip, *args, **kwargs):
        builds.append(level_mip)
        return real_commit(level_mip, *args, **kwargs)

    monkeypatch.setattr(cache, "_commit_level", commit)
    return builds


@pytest.mark.parametrize("elevation_splice", ["spliced", "wholesale"])
def test_a_units_and_elevation_paste_builds_the_level_once(elevation_splice, monkeypatch) -> None:
    """At most one level build across the paste and the next full paint.
    `spliced`: the 3+3 units share tiles, so the units splice as a component
    and the elevation re-anchor splices too: no build and no gen bump.
    `wholesale` is the stress log's shape: the elevation re-anchor is past its
    cap, so the paste skips the level splice (_elevation_bumps_anyway()),
    updates units_by_tile in place and the patch's gen bump costs one build."""
    if elevation_splice == "wholesale":
        monkeypatch.setattr(render_cache, "_ELEV_SPLICE_MAX_UNITS", 0)
    window = _seeded_window(MARK_CONST)
    try:
        cache = window._cache
        mip = _visible_mip(window)
        _canvas(window, mip)  # the visible level resident, so the patch rebuilds it in place
        builds = _count_builds(window, monkeypatch)
        gen = cache._source_gen
        _paste(window)
        _canvas(window, mip)
        if elevation_splice == "wholesale":
            assert cache._source_gen != gen, "the paste bumped no gen, so this proves nothing"
            assert builds == [mip], f"expected one mip {mip} build across the paste and the paint, got {builds}"
        else:
            assert cache._source_gen == gen, "a paste under every cap bumped the gen"
            assert builds == [], f"expected no level build, got {builds}"
        assert np.array_equal(_canvas(window), _fresh(window))
    finally:
        window.edit_history.mark_saved()
        window.close()


def _spy_in_place(window, monkeypatch) -> list[bool]:
    """Every _update_units_by_tile_in_place() answer from here on."""
    cache = window._cache
    answers: list[bool] = []
    real = cache._update_units_by_tile_in_place

    def spy(changed):
        answers.append(real(changed))
        return answers[-1]

    monkeypatch.setattr(cache, "_update_units_by_tile_in_place", spy)
    return answers


def _block_window(block: str, monkeypatch):
    """_seeded_window() for one of BLOCKS, with `over_cap` pinning both unit
    splice caps under the block's unit count and `no_elevation` leaving the
    Elevation category unchecked."""
    window = _seeded_window(MARK_CONST, block=block)
    if block == "over_cap":
        monkeypatch.setattr(render_cache, "UNIT_SPLICE_MAX_UNITS", 1)
        monkeypatch.setattr(render_cache, "_COMPONENT_SPLICE_MAX_UNITS", 1)
    if block == "no_elevation":
        window.paste_elevation_check.setChecked(False)
    return window


# block -> whether the paste's own unit update takes the in-place path (else it splices the levels).
BLOCKS = {"wall": True, "over_cap": True, "garrison": False, "no_elevation": False}


def _paste_undo_redo(window, before_step=None, after_step=None) -> None:
    """Paste, undo, redo. The whole canvas is re-rendered right before each
    step (GOTCHAS: an evicted chunk redraws fresh and would hide a stale one),
    and each step's render must equal a fresh one, failing as `<step>: ...`."""
    for step, act in (("paste", lambda: _paste(window)), ("undo", window.undo), ("redo", window.redo)):
        _canvas(window)
        _canvas(window, _visible_mip(window))
        if before_step is not None:
            before_step(step)
        act()
        assert np.array_equal(_canvas(window), _fresh(window)), f"{step}: differs from a fresh render"
        if after_step is not None:
            after_step(step)


@pytest.mark.parametrize("block", sorted(BLOCKS))
def test_a_paste_its_undo_and_redo_take_the_blocks_units_path_and_match_a_fresh_render(block, monkeypatch) -> None:
    """A wall (const:add/remove) or an over-cap block updates units_by_tile in
    place and bumps the gen, on the paste and on its undo and redo; a
    garrisoned occupant (two pasted units on one tile) splices its component;
    an elevation-free block splices (its undo and redo take the tiles-first
    scoped path). Every step matches a fresh render and builds the visible
    level exactly once on the in-place path, and not at all on a splice."""
    window = _block_window(block, monkeypatch)
    try:
        cache = window._cache
        mip = _visible_mip(window)
        before = _canvas(window)
        answers = _spy_in_place(window, monkeypatch)
        builds = _count_builds(window, monkeypatch)
        state = {}

        def before_step(step):
            del answers[:], builds[:]
            state["gen"] = cache._source_gen

        def after_step(step):
            assert (True in answers) is BLOCKS[block], f"{step}: in-place answers {answers}"
            assert (cache._source_gen != state["gen"]) is BLOCKS[block], f"{step}: the gen moved against the path"
            # Measured: once after the in-place path's gen bump, never after a splice.
            expected = 1 if BLOCKS[block] else 0
            assert builds.count(mip) == expected, f"{step}: mip {mip} built {builds.count(mip)} times, not {expected}"
            if step == "paste":
                assert not np.array_equal(_canvas(window), before), "the paste changed nothing on screen"

        _paste_undo_redo(window, before_step, after_step)
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("broken", ["paste", "undo", "redo"])
def test_the_sequence_check_names_the_one_step_whose_update_went_stale(broken, monkeypatch) -> None:
    """Control for the test above (GOTCHAS: prove a step-k break fails naming
    step k): the wall block's in-place update is skipped on step `broken`
    only, claiming success, and the check must fail there and nowhere earlier."""
    window = _block_window("wall", monkeypatch)
    try:
        cache = window._cache
        real = cache._update_units_by_tile_in_place
        current = {}

        def update(changed):
            if current.get("step") == broken:
                cache._units_version += 1
                return True
            return real(changed)

        monkeypatch.setattr(cache, "_update_units_by_tile_in_place", update)
        with pytest.raises(AssertionError, match=f"{broken}: differs from a fresh render"):
            _paste_undo_redo(window, before_step=lambda step: current.update(step=step))
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("elevation_splice", ["spliced", "wholesale"])
def test_the_undo_and_redo_of_a_units_and_elevation_paste_build_the_level_at_most_once(
    elevation_splice, monkeypatch
) -> None:
    """Undo and redo of the paste run the unit sources before the tile patch.
    `wholesale` (elevation re-anchor past its cap): exactly one build across
    the op and the next full paint, where the tiles-first order paid two.
    `spliced`: none, and the gen does not move."""
    if elevation_splice == "wholesale":
        monkeypatch.setattr(render_cache, "_ELEV_SPLICE_MAX_UNITS", 0)
    window = _seeded_window(MARK_CONST)
    try:
        cache = window._cache
        mip = _visible_mip(window)
        _paste(window)
        builds = _count_builds(window, monkeypatch)
        for step, act in (("undo", window.undo), ("redo", window.redo)):
            _canvas(window, mip)
            del builds[:]
            gen = cache._source_gen
            act()
            _canvas(window, mip)
            if elevation_splice == "wholesale":
                assert cache._source_gen != gen, f"{step}: no gen bump, so this proves nothing"
                assert builds == [mip], f"{step}: expected one mip {mip} build, got {builds}"
            else:
                assert cache._source_gen == gen, f"{step}: a paste undo under every cap bumped the gen"
                assert builds == [], f"{step}: expected no level build, got {builds}"
            assert np.array_equal(_canvas(window), _fresh(window)), step
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_units_only_paste_in_units_mode_can_pick_a_pasted_unit() -> None:
    """No tile record: Draw release's shape, and the pick index is rebuilt."""
    from PyQt5.QtCore import QPointF

    window = _seeded_window(MARK_CONST)
    try:
        window.paste_terrain_check.setChecked(False)
        window.paste_elevation_check.setChecked(False)
        window.mode_combo.setCurrentText("Units")
        assert window.map_view._unit_index is not None
        count = sum(len(units) for units in window.scenario.unit_manager.units)
        window._hover_tile = (DST[0] + 20, DST[1])
        window.paste_region()
        pasted = window.scenario.unit_manager.units[1][-1]
        assert sum(len(units) for units in window.scenario.unit_manager.units) == count + 3
        center = window.map_view._tile_polygon(int(pasted.x), int(pasted.y)).boundingRect().center()
        picked = window.map_view.pick_unit_at(QPointF(center))
        assert picked is not None and picked.unit is pasted, "a pasted unit is not in the pick index"
        assert np.array_equal(_canvas(window), _fresh(window))
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.fixture
def traced(monkeypatch):
    """Perf Trace on, with fresh module state restored afterwards."""
    from descape import debug_log, perf_trace

    debug_log.clear()
    for name, value in perf_trace._fresh_state().items():
        monkeypatch.setattr(perf_trace, name, value)
    monkeypatch.setattr(perf_trace, "_idle_scheduler", None)
    monkeypatch.setattr(perf_trace, "_sprite_counter", lambda: (0, 0))
    monkeypatch.setattr(perf_trace, "_enabled", True)
    yield
    debug_log.clear()


def _paste_line() -> str:
    from descape import debug_log, perf_trace

    perf_trace.flush_pending_op()
    return next(line for line in debug_log.get_log_text().splitlines() if "perf op paste" in line)


@pytest.mark.parametrize("elevation_splice", ["spliced", "wholesale"])
def test_the_paste_perf_line_says_whether_it_skipped_the_level_splice(elevation_splice, traced, monkeypatch) -> None:
    """Over the elevation cap the predictor skips the level splice, which the
    op line reports as `splice_refused=skipped` plus `units_in_place`; under
    it the units splice and no refusal is reported."""
    if elevation_splice == "wholesale":
        monkeypatch.setattr(render_cache, "_ELEV_SPLICE_MAX_UNITS", 0)
    window = _seeded_window(MARK_CONST)
    try:
        from descape import debug_log, perf_trace

        perf_trace.enable(True)  # the window applied its own (off) setting on construction
        debug_log.clear()
        _paste(window)
        line = _paste_line()
        if elevation_splice == "wholesale":
            assert "splice_refused=skipped" in line and "units_in_place" in line, line
        else:
            assert "splice_refused" not in line and "units_in_place" not in line, line
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("elevation_splice", ["spliced", "wholesale"])
def test_the_elevation_predictor_is_its_own_phase_on_the_paste_undo_and_redo_lines(
    elevation_splice, traced, monkeypatch
) -> None:
    """The predictor used to run outside every phase, so its cost showed only
    as `untimed`. Undo and redo time both of its calls under the one name."""
    import re

    if elevation_splice == "wholesale":
        monkeypatch.setattr(render_cache, "_ELEV_SPLICE_MAX_UNITS", 0)
    window = _seeded_window(MARK_CONST)
    try:
        from descape import debug_log, perf_trace

        perf_trace.enable(True)  # the window applied its own (off) setting on construction
        debug_log.clear()
        _paste(window)
        window.undo()
        window.redo()
        perf_trace.flush_pending_op()
        lines = debug_log.get_log_text().splitlines()
        for label in ("paste", "undo", "redo"):
            [line] = [entry for entry in lines if f"perf op {label}:" in entry]
            assert re.search(r" elev_predict \d+\.\d", line), line
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_the_elevation_predictor_counts_the_seed_set_each_cache_starts_from(style, monkeypatch) -> None:
    """_elevation_bumps_anyway() counts the units whose own tile is in E on
    Stepped, and on Sloped (which reads four corners) also those on a tile
    whose corner the edit moves: a unit beside a raised tile counts on Sloped
    only. The map is edited and the cache is not, as when the viewer asks.
    Hidden units never count."""
    window = conftest.terrain_edit_window()
    try:
        window.terrain_style_combo.setCurrentText(style)
        assert window._render_style == style.lower()
        model = window._ensure_unit_edits()
        model.begin_unit_edit([1])
        model.add(player=1, unit_const=MARK_CONST, x=30.5, y=30.5, z=0.0, rotation=0.0)
        model.commit_unit_edit("seed", window.edit_history)
        window._after_unit_mutation()
        window.scenario.map_manager.get_tile(31, 30).elevation += 1
        cap_name = "_SLOPED_ELEV_SPLICE_MAX_UNITS" if style == "Sloped" else "_ELEV_SPLICE_MAX_UNITS"
        monkeypatch.setattr(render_cache, cap_name, 0)
        assert window._elevation_bumps_anyway({(30, 30)})
        assert window._elevation_bumps_anyway({(31, 30)}) is (style == "Sloped")
        assert not window._elevation_bumps_anyway({(60, 60)})
        assert not window._elevation_bumps_anyway(set())
        monkeypatch.setattr(render_cache, cap_name, 1)
        assert not window._elevation_bumps_anyway({(30, 30), (31, 30)}), "one seed is not past a cap of 1"
        monkeypatch.setattr(render_cache, cap_name, 0)
        window.player_actions[1].setChecked(False)
        assert not window._elevation_bumps_anyway({(30, 30)}), "a hidden unit seeds nothing"
    finally:
        window.edit_history.mark_saved()
        window.close()
