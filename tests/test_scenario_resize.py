"""descape/scenario_resize.py (TASK-107, GH #45): resizing an existing map
from any of nine anchors, square-only until non-square maps clear their
in-game gate (TASK-032).

Default tier runs on the shipped fixtures; the corpus-marked test at the
bottom resizes real examples/ files through the full write path.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from descape import library_compat, player_fields, render, trigger_geometry
from descape.scenario_io import (
    BLANK_TEMPLATE_PATH,
    TERRAIN_STRUCT_SIZE,
    load_map_and_units,
    load_map_and_units_from_bytes,
    parse_triggers,
)
from descape.scenario_new import BLANK_TERRAIN_STRUCT, blank_body
from descape.scenario_resize import (
    TRIGGERS_NONE,
    TRIGGERS_REMAPPED,
    Anchor,
    ResizeRefusedError,
    origin_offset,
    remap_terrain,
    resize_body,
    resize_plan,
    resize_scenario,
)
from descape.scenario_write import _compress_bytes, build_patched, write_scenario
from descape.terrain_palette import tile_span
from descape.unit_model import UnitEditModel

FIXTURES = Path(__file__).resolve().parent / "fixtures"
UNITS_FIXTURE = FIXTURES / "units_120x120.aoe2scenario"
TRIGGERS_FIXTURE = FIXTURES / "triggers_120x120.aoe2scenario"


def _save_and_reload(result, tmp_path, name="resized.aoe2scenario"):
    out = tmp_path / name
    write_scenario(result.loaded, out, backup=False, units=result.unit_edits, triggers=result.trigger_edits)
    return load_map_and_units(out)


def _all_units(loaded):
    return {u.reference_id: u for units in loaded.unit_manager.units for u in units}


# -- origin_offset -------------------------------------------------------------


@pytest.mark.parametrize(
    ("anchor", "expected"),
    [
        (Anchor.TOP_LEFT, (0, 0)),
        (Anchor.TOP, (2, 0)),
        (Anchor.TOP_RIGHT, (5, 0)),
        (Anchor.LEFT, (0, 3)),
        (Anchor.CENTER, (2, 3)),
        (Anchor.RIGHT, (5, 3)),
        (Anchor.BOTTOM_LEFT, (0, 7)),
        (Anchor.BOTTOM, (2, 7)),
        (Anchor.BOTTOM_RIGHT, (5, 7)),
    ],
)
def test_origin_offset_for_every_anchor_on_a_grow_with_odd_deltas(anchor, expected) -> None:
    assert origin_offset(100, 100, 105, 107, anchor) == expected


@pytest.mark.parametrize(
    ("anchor", "expected"),
    [
        (Anchor.TOP_LEFT, (0, 0)),
        (Anchor.CENTER, (-3, -4)),  # floors: -5 // 2, -7 // 2
        (Anchor.BOTTOM_RIGHT, (-5, -7)),
        (Anchor.TOP_RIGHT, (-5, 0)),
        (Anchor.BOTTOM_LEFT, (0, -7)),
    ],
)
def test_origin_offset_on_a_shrink_with_odd_deltas(anchor, expected) -> None:
    assert origin_offset(105, 107, 100, 100, anchor) == expected


# -- terrain -------------------------------------------------------------------


def test_growing_a_blank_map_from_top_left_is_blank_body_byte_for_byte() -> None:
    """Free cross-check against scenario_new's own byte oracle: a grown blank
    map is exactly a blank map generated at that size."""
    donor = load_map_and_units(BLANK_TEMPLATE_PATH)
    grown = resize_body(donor, 168, 168, Anchor.TOP_LEFT, build_patched(donor))
    assert grown == blank_body(donor, 168)


def _block(loaded) -> bytes:
    mm = loaded.map_manager
    off = loaded.terrain_block_offset
    return loaded.decompressed_body[off : off + TERRAIN_STRUCT_SIZE * mm.map_width * mm.map_height]


def _struct(block: bytes, w: int, x: int, y: int) -> bytes:
    i = (y * w + x) * TERRAIN_STRUCT_SIZE
    return block[i : i + TERRAIN_STRUCT_SIZE]


def _patterned_blank():
    """The blank template with every tile's terrain_id a function of (x, y),
    as an unsaved in-memory edit."""
    loaded = load_map_and_units(BLANK_TEMPLATE_PATH)
    mm = loaded.map_manager
    for i, tile in enumerate(mm.terrain):
        x, y = i % mm.map_width, i // mm.map_width
        tile.terrain_id = (x * 7 + y * 13) % 41
    return loaded


@pytest.mark.parametrize(
    ("size", "anchor"),
    [(100, Anchor.BOTTOM_RIGHT), (101, Anchor.CENTER), (137, Anchor.CENTER), (150, Anchor.TOP_RIGHT)],
)
def test_terrain_lands_at_its_new_index_and_only_the_cut_tiles_drop(size, anchor) -> None:
    loaded = _patterned_blank()
    old = build_patched(loaded)
    old_block = old.body[old.terrain_offset : old.terrain_offset + TERRAIN_STRUCT_SIZE * 120 * 120]
    result = resize_scenario(loaded, size, size, anchor)
    new_block = _block(result.loaded)
    dx, dy = result.plan.dx, result.plan.dy
    kept = 0
    for ny in range(size):
        for nx in range(size):
            ox, oy = nx - dx, ny - dy
            got = _struct(new_block, size, nx, ny)
            if 0 <= ox < 120 and 0 <= oy < 120:
                assert got == _struct(old_block, 120, ox, oy), (nx, ny)
                kept += 1
            else:
                assert got == BLANK_TERRAIN_STRUCT, (nx, ny)
    assert kept == 120 * 120 - result.plan.tiles_lost
    assert size * size - kept == result.plan.tiles_gained


def test_no_adjacent_pair_differs_by_more_than_one_after_growing_a_raised_border() -> None:
    """The game-crash guard: new tiles extend the nearest old edge tile's
    elevation instead of dropping to 0 beside a raised border."""
    loaded = load_map_and_units(BLANK_TEMPLATE_PATH)
    mm = loaded.map_manager
    for i, tile in enumerate(mm.terrain):
        x, y = i % 120, i // 120
        tile.elevation = max(0, 3 - min(x, y, 119 - x, 119 - y))
    result = resize_scenario(loaded, 151, 151, Anchor.CENTER)
    terrain = result.loaded.map_manager.terrain
    w = 151
    elevation = [[terrain[y * w + x].elevation for x in range(w)] for y in range(w)]
    for y in range(w):
        for x in range(w):
            for ex, ey in ((1, 0), (0, 1), (1, 1), (1, -1)):
                if 0 <= x + ex < w and 0 <= y + ey < w:
                    assert abs(elevation[y][x] - elevation[y + ey][x + ex]) <= 1, (x, y, ex, ey)
    assert elevation[0][0] == elevation[150][150] == elevation[0][75] == 3


def test_remap_terrain_extends_corners_from_the_corner_tile() -> None:
    """2x2 grown to 4x4 around the centre: each new corner copies its
    nearest old corner's elevation, never 0."""
    block = b"".join(_filler_with(e) for e in (1, 2, 3, 4))
    grown = remap_terrain(block, 2, 2, 4, 4, 1, 1)
    elevations = [grown[i * TERRAIN_STRUCT_SIZE + 1] for i in range(16)]
    assert elevations == [1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]


def _filler_with(elevation: int) -> bytes:
    return BLANK_TERRAIN_STRUCT[:1] + bytes((elevation,)) + BLANK_TERRAIN_STRUCT[2:]


# -- units ---------------------------------------------------------------------


def test_units_translate_keep_rotation_bytes_and_lose_the_off_map_ones(tmp_path) -> None:
    """units_120x120: 8 units, both rotation encodings, the one non-empty
    caption. A BOTTOM_RIGHT shrink by 20 cuts x, y < 20: players 0 and 1
    (100-102, 200-203) go, 300 and 301 survive at (0.5, 0.5) and (1.5, 0.5)."""
    loaded = load_map_and_units(UNITS_FIXTURE)
    before = _all_units(loaded)
    plan = resize_plan(loaded, 100, 100, Anchor.BOTTOM_RIGHT)
    assert (plan.dx, plan.dy, plan.units_deleted, plan.units_moved) == (-20, -20, 6, 2)
    result = resize_scenario(loaded, 100, 100, Anchor.BOTTOM_RIGHT)
    reloaded = _save_and_reload(result, tmp_path)
    after = _all_units(reloaded)
    assert set(after) == {300, 301}
    for ref, unit in after.items():
        old = before[ref]
        assert (unit.x, unit.y, unit.z) == (old.x - 20, old.y - 20, old.z)
        assert struct.pack("<f", unit.rotation) == struct.pack("<f", old.rotation)
        assert unit.unit_const == old.unit_const
    assert after[300].caption_string == "Fixture caption"


def test_a_host_leaving_the_map_takes_its_occupant_even_when_the_occupant_stays_on(tmp_path) -> None:
    """Host 200 sits at x = 10.5, its occupant 203 at 12.5. Anchored RIGHT,
    120 to 109 is (dx, dy) = (-11, -6): the host lands at x = -0.5 and the
    occupant at 1.5, so the occupant still goes and the count includes it."""
    loaded = load_map_and_units(UNITS_FIXTURE)
    plan = resize_plan(loaded, 109, 109, Anchor.RIGHT)
    assert (plan.dx, plan.dy) == (-11, -6)
    assert plan.units_deleted == 5  # 100, 101, 102, 200, and 203 inside it
    result = resize_scenario(loaded, 109, 109, Anchor.RIGHT)
    assert result.units_deleted == 5
    after = _all_units(_save_and_reload(result, tmp_path))
    assert set(after) == {201, 300, 301}
    assert (after[201].x, after[201].y) == (0.5, 4.5)


CASTLE = 82  # 4x4


@pytest.mark.parametrize(
    ("x", "anchor", "deleted"),
    [
        (117.0, Anchor.TOP_LEFT, True),
        (116.0, Anchor.TOP_LEFT, False),
        (3.0, Anchor.BOTTOM_RIGHT, True),
        (4.0, Anchor.BOTTOM_RIGHT, False),
    ],
)
def test_a_footprint_the_cut_leaves_partly_off_the_map_is_deleted(tmp_path, x, anchor, deleted) -> None:
    """A Castle at x = 117.0 covers tiles 115..118: its anchor survives a
    120 -> 118 top-left shrink but column 118 does not, a state
    replace_refusal() refuses. At x = 116.0 (114..117) it fits and stays.
    The low side mirrors it: 3.0 (tiles 1..4) under a bottom-right cut of 2."""
    loaded = load_map_and_units(UNITS_FIXTURE)
    units = UnitEditModel(loaded)
    castle = units.add(1, CASTLE, x, 60.0)
    plan = resize_plan(loaded, 118, 118, anchor)
    assert plan.units_deleted == int(deleted)
    result = resize_scenario(loaded, 118, 118, anchor, units=units)
    assert result.units_deleted == int(deleted)
    after = _all_units(_save_and_reload(result, tmp_path))
    assert (castle.reference_id in after) is not deleted
    for unit in after.values():
        span_x, span_y = tile_span(unit.unit_const, render.NON_BUILDING_SPAN)
        x0, y0 = render._span_start(unit.x, span_x), render._span_start(unit.y, span_y)
        assert 0 <= x0 <= 118 - span_x and 0 <= y0 <= 118 - span_y, unit.reference_id


@pytest.mark.parametrize(("x", "kept_edge"), [(1.0, Anchor.LEFT), (119.0, Anchor.RIGHT)])
def test_a_footprint_already_overhanging_the_old_edge_is_not_the_cuts(x, kept_edge) -> None:
    """Only what the cut takes counts: a Castle hanging over the old map's
    edge (tiles -1..2 or 117..120) survives a grow, and a shrink anchored on
    that same edge, since neither cuts any of its on-map tiles."""
    loaded = load_map_and_units(UNITS_FIXTURE)
    units = UnitEditModel(loaded)
    units.add(1, CASTLE, x, 60.0)
    assert resize_plan(loaded, 130, 130, Anchor.TOP_LEFT).units_deleted == 0
    assert resize_plan(loaded, 130, 130, Anchor.CENTER).units_deleted == 0
    shrink = resize_plan(loaded, 118, 118, kept_edge)
    assert shrink.dx == (0 if kept_edge is Anchor.LEFT else -2)
    assert shrink.units_deleted == 0


def test_a_grow_from_top_left_needs_no_unit_edits() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    result = resize_scenario(loaded, 144, 144, Anchor.TOP_LEFT)
    assert result.unit_edits is None and result.units_deleted == 0
    assert set(_all_units(result.loaded)) == set(_all_units(loaded))


# -- triggers ------------------------------------------------------------------


def _shapes(loaded):
    manager = parse_triggers(loaded)
    vocabulary = library_compat.load_vocabulary(loaded.scenario_version)
    return [
        s
        for i, t in enumerate(manager.triggers)
        for s in trigger_geometry.shapes_for_trigger(t, vocabulary, trigger_index=i)
    ]


@pytest.mark.parametrize(("size", "anchor"), [(100, Anchor.BOTTOM_RIGHT), (150, Anchor.CENTER)])
def test_trigger_coordinates_translate_and_clamp_through_their_field_groups(tmp_path, size, anchor) -> None:
    loaded = load_map_and_units(TRIGGERS_FIXTURE)
    before = _shapes(loaded)
    assert before, "the fixture must carry coordinate-bearing entries (GH #41 Phase 2)"
    plan = resize_plan(loaded, size, size, anchor)
    assert plan.triggers == TRIGGERS_REMAPPED
    result = resize_scenario(loaded, size, size, anchor)
    after = _shapes(_save_and_reload(result, tmp_path))
    assert [(s.trigger_index, s.entry_ref, s.fields) for s in after] == [
        (s.trigger_index, s.entry_ref, s.fields) for s in before
    ]
    for old, new in zip(before, after, strict=True):
        expected = tuple(
            min(max(c + (plan.dx if "_x" in f else plan.dy), 0), size - 1)
            for f, c in zip(old.fields, old.coords, strict=True)
        )
        assert new.coords == expected, (old, new)


def test_a_file_without_triggers_reports_none_to_remap() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    plan = resize_plan(loaded, 100, 100, Anchor.CENTER)
    assert plan.triggers == TRIGGERS_NONE
    assert "Trigger" not in plan.summary()


def test_unwritable_triggers_are_left_alone_and_the_plan_warns(monkeypatch) -> None:
    loaded = load_map_and_units(TRIGGERS_FIXTURE)
    parse_triggers(loaded)
    loaded.trigger_write_supported = False
    plan = resize_plan(loaded, 100, 100, Anchor.CENTER)
    assert plan.triggers == "stale"
    assert "left as they are" in plan.summary()
    result = resize_scenario(loaded, 100, 100, Anchor.CENTER)
    assert result.trigger_edits is None


# -- cameras -------------------------------------------------------------------


def _with_camera(x_value: int):
    """The blank template with player 1's Point of View x (Map.initial_player_views,
    GAIA-first, so index 1) stored as x_value, reloaded so it parses."""
    donor = load_map_and_units(BLANK_TEMPLATE_PATH)
    pov = [f for f in player_fields.camera_fields(donor) if f.target.codec == "s32" and f.axis == "x"]
    body = bytearray(donor.decompressed_body)
    struct.pack_into("<i", body, pov[1].target.offset, x_value)
    return load_map_and_units_from_bytes(donor.header_bytes + _compress_bytes(bytes(body)), "camera.aoe2scenario")


def test_cameras_past_the_new_edge_clamp_and_unset_ones_stay_unset() -> None:
    loaded = _with_camera(110)
    result = resize_scenario(loaded, 100, 100, Anchor.TOP_LEFT)
    fields = player_fields.camera_fields(result.loaded)
    pov_x = [f.value for f in fields if f.target.codec == "s32" and f.axis == "x"]
    assert pov_x[1] == 99
    assert all(v == -1 for i, v in enumerate(pov_x) if i != 1)
    editor = [f.value for f in fields if f.target.codec in ("f32", "s16")]
    assert set(editor) == {60}  # the template's (60, 60), inside 100x100: untouched


# -- carried edits and refusals ------------------------------------------------


def test_pending_unsaved_edits_are_carried_not_the_on_disk_state(tmp_path) -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    loaded.map_manager.terrain[50 * 120 + 50].terrain_id = 2
    units = UnitEditModel(loaded)
    moved = _all_units(loaded)[300]
    units.set_position(moved, 40.5, 41.5, moved.z)
    result = resize_scenario(loaded, 140, 140, Anchor.BOTTOM_RIGHT, units=units)
    reloaded = _save_and_reload(result, tmp_path)
    assert reloaded.map_manager.terrain[70 * 140 + 70].terrain_id == 2
    assert (_all_units(reloaded)[300].x, _all_units(reloaded)[300].y) == (60.5, 61.5)


@pytest.mark.parametrize(
    ("size", "match"),
    [((100, 120), "Only square"), ((120, 120), "already"), ((60, 60), "outside supported range")],
)
def test_refusals(size, match) -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    plan = resize_plan(loaded, *size, Anchor.CENTER)
    assert plan.refusal is not None and match in plan.refusal
    with pytest.raises(ResizeRefusedError, match=match):
        resize_scenario(loaded, *size, Anchor.CENTER)


def test_unwritable_units_allow_only_a_top_left_grow() -> None:
    import dataclasses

    loaded = dataclasses.replace(load_map_and_units(UNITS_FIXTURE), units_write_supported=False)
    assert resize_plan(loaded, 140, 140, Anchor.TOP_LEFT).refusal is None
    assert "only be grown from the top-left" in resize_plan(loaded, 140, 140, Anchor.CENTER).refusal


def test_resize_leaves_the_source_document_untouched() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    body = loaded.decompressed_body
    positions = {r: (u.x, u.y) for r, u in _all_units(loaded).items()}
    resize_scenario(loaded, 100, 100, Anchor.BOTTOM_RIGHT)
    assert loaded.decompressed_body == body
    assert loaded.map_manager.map_width == 120
    assert {r: (u.x, u.y) for r, u in _all_units(loaded).items()} == positions


def test_batch_api_resize_map_takes_an_anchor_name(tmp_path) -> None:
    from descape import batch_api

    loaded = batch_api.load(UNITS_FIXTURE)
    result = batch_api.resize_map(loaded, 100, 100, "bottom_right")
    assert result.plan.anchor is Anchor.BOTTOM_RIGHT
    out = tmp_path / "batch.aoe2scenario"
    batch_api.save(result.loaded, out, units=result.unit_edits, triggers=result.trigger_edits)
    assert load_map_and_units(out).map_manager.map_width == 100


def test_batch_api_resize_map_carries_pending_option_and_message_edits(tmp_path) -> None:
    from descape import batch_api
    from descape.messages_model import MessagesEditModel
    from descape.options_model import OptionsEditModel

    loaded = batch_api.load(TRIGGERS_FIXTURE)
    options = OptionsEditModel(loaded)
    lock_teams = 1 - options.original_value("lock_teams")
    options.set_value("lock_teams", lock_teams)
    messages = MessagesEditModel(loaded)
    messages.set_value("victory", "resized victory text")
    result = batch_api.resize_map(loaded, 130, 130, "CENTER", options=options, messages=messages)
    out = tmp_path / "batch.aoe2scenario"
    batch_api.save(result.loaded, out, units=result.unit_edits, triggers=result.trigger_edits)
    reloaded = load_map_and_units(out)
    assert reloaded.map_manager.map_width == 130
    assert OptionsEditModel(reloaded).original_value("lock_teams") == lock_teams
    assert MessagesEditModel(reloaded).original_value("victory") == "resized victory text"


# -- corpus --------------------------------------------------------------------


@pytest.mark.corpus
@pytest.mark.gui
def test_verify_resize_corpus(scenario_path) -> None:
    """tools/verify_resize.py's check_file(): the same resize driven through
    a real ViewerWindow, saved with File > Save and read back."""
    import conftest

    conftest.ensure_qapp()
    ok, detail = conftest.load_verify_module("verify_resize").check_file(scenario_path)
    if ok is None:
        pytest.skip(detail)
    assert ok, detail


@pytest.mark.corpus
def test_resize_corpus_round_trip(scenario_path, tmp_path) -> None:
    """A real file shrunk from BOTTOM_RIGHT by 24 tiles, through the real
    write path and back: dimensions, the deleted count, every surviving
    unit's position, and the elevation adjacency rule over the whole grid."""
    loaded = load_map_and_units(scenario_path)
    mm = loaded.map_manager
    if mm.map_width != mm.map_height:
        pytest.skip("non-square source")
    size = mm.map_width - 24
    plan = resize_plan(loaded, size, size, Anchor.BOTTOM_RIGHT)
    if plan.refusal is not None:
        pytest.skip(plan.refusal)
    before = {(p, i): (u.reference_id, u.x, u.y) for p, us in enumerate(loaded.unit_manager.units) for i, u in enumerate(us)}
    before_shapes = _shapes(loaded) if plan.triggers == TRIGGERS_REMAPPED else None
    result = resize_scenario(loaded, size, size, Anchor.BOTTOM_RIGHT)
    reloaded = _save_and_reload(result, tmp_path)
    if before_shapes is not None:
        after_shapes = _shapes(reloaded)
        assert len(after_shapes) == len(before_shapes)
        for old, new in zip(before_shapes, after_shapes, strict=True):
            assert new.coords == tuple(min(max(c - 24, 0), size - 1) for c in old.coords), (old, new)
    rm = reloaded.map_manager
    assert (rm.map_width, rm.map_height) == (size, size)
    after = sum(len(u) for u in reloaded.unit_manager.units)
    assert after == len(before) - plan.units_deleted
    by_ref = {}
    for ref, x, y in before.values():
        by_ref.setdefault(ref, []).append((x - 24, y - 24))
    for units in reloaded.unit_manager.units:
        for u in units:
            assert (u.x, u.y) in by_ref[u.reference_id]
    # A shrink is a pure crop: every kept struct is the old one at (x + 24, y + 24).
    old_block, new_block = _block(loaded), _block(reloaded)
    for y in range(size):
        row = (y * size) * TERRAIN_STRUCT_SIZE
        old_row = ((y + 24) * mm.map_width + 24) * TERRAIN_STRUCT_SIZE
        assert new_block[row : row + size * TERRAIN_STRUCT_SIZE] == old_block[old_row : old_row + size * TERRAIN_STRUCT_SIZE]


def _max_step(loaded) -> int:
    import numpy as np

    mm = loaded.map_manager
    e = np.frombuffer(_block(loaded), dtype=np.uint8)[1::TERRAIN_STRUCT_SIZE].astype(int)
    e = e.reshape(mm.map_height, mm.map_width)
    steps = [np.abs(np.diff(e, axis=0)).max(), np.abs(np.diff(e, axis=1)).max()]
    steps.append(np.abs(e[1:, 1:] - e[:-1, :-1]).max())
    steps.append(np.abs(e[1:, :-1] - e[:-1, 1:]).max())
    return int(max(steps))


@pytest.mark.corpus
def test_resize_corpus_grow_adds_no_steeper_step(scenario_path) -> None:
    """Edge extension can only repeat an existing neighbour pair, so a grown
    map's steepest 8-neighbour elevation step is never steeper than the
    source's."""
    loaded = load_map_and_units(scenario_path)
    mm = loaded.map_manager
    size = mm.map_width + 24
    plan = resize_plan(loaded, size, size, Anchor.CENTER)
    if plan.refusal is not None:
        pytest.skip(plan.refusal)
    result = resize_scenario(loaded, size, size, Anchor.CENTER)
    assert _max_step(result.loaded) <= _max_step(loaded)
