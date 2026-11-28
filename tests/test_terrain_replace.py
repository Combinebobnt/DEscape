"""descape/terrain_replace.py, the Replace Brush's Qt-free match and write,
on test_fill_tools.py's fake map. The viewer stroke is
tests/test_replace_brush_viewer.py."""

from __future__ import annotations

import itertools

import pytest
from test_fill_tools import FakeMapManager, FakeTile, _grid

from descape import brush, terrain_replace, tool_options
from descape.fill_tools import fill_indices
from descape.terrain_replace import FLOOD_CONTIGUOUS, FLOOD_GLOBAL, ReplaceStroke, press, replace_tiles
from descape.terrain_target import CLEAR, LAYER, TERRAIN

G, F, W, D = ord("G"), ord("F"), ord("W"), ord("D")


def _text(mm) -> list[str]:
    width = mm.map_width
    row = "".join(chr(t.terrain_id) for t in mm.terrain)
    return [row[i : i + width] for i in range(0, len(row), width)]


def _all(mm) -> list[tuple[int, int]]:
    return [(x, y) for y in range(mm.map_height) for x in range(mm.map_width)]


def _stroke(mm, x, y, target, flood=FLOOD_GLOBAL) -> ReplaceStroke:
    stroke = ReplaceStroke(flood)
    press(stroke, mm, x, y, target)
    return stroke


def test_the_terrain_target_writes_only_the_press_tiles_terrain() -> None:
    mm = _grid(["GGFW", "GFGW", "WGGG"])
    stroke = _stroke(mm, 0, 0, TERRAIN)
    changed = replace_tiles(mm, _all(mm), stroke.ref, TERRAIN, D)
    assert _text(mm) == ["DDFW", "DFDW", "WDDD"]
    assert sorted(changed) == [0, 1, 4, 6, 9, 10, 11]


def test_a_layered_tile_of_the_press_terrain_matches_on_terrain_and_its_layer_clears() -> None:
    mm = _grid(["GGF"])
    mm.terrain[1].layer = W
    mm.terrain[2].layer = G  # a Forest tile layered with Grass is still Forest
    stroke = _stroke(mm, 0, 0, TERRAIN)
    assert replace_tiles(mm, _all(mm), stroke.ref, TERRAIN, D) == [0, 1]
    assert [(t.terrain_id, t.layer) for t in mm.terrain] == [(D, -1), (D, -1), (F, G)]


def test_the_layer_target_writes_only_tiles_with_the_press_tiles_pair() -> None:
    mm = _grid(["GGGF"])
    mm.terrain[1].layer = W  # same base, different layer: untouched
    stroke = _stroke(mm, 0, 0, LAYER)
    assert stroke.ref == (G, -1)
    assert replace_tiles(mm, _all(mm), stroke.ref, LAYER, D) == [0, 2]
    assert [(t.terrain_id, t.layer) for t in mm.terrain] == [(G, D), (G, W), (G, D), (F, -1)]


def test_clear_layer_clears_only_that_pairs_layer() -> None:
    mm = _grid(["GGGG"])
    for i, layer in enumerate((W, D, W, -1)):
        mm.terrain[i].layer = layer
    stroke = _stroke(mm, 0, 0, CLEAR)
    assert replace_tiles(mm, _all(mm), stroke.ref, CLEAR, F) == [0, 2]
    assert [t.layer for t in mm.terrain] == [-1, D, -1, -1]
    assert all(t.terrain_id == G for t in mm.terrain)


@pytest.mark.parametrize("target", [TERRAIN, LAYER])
def test_contiguous_leaves_a_cut_off_patch_and_global_writes_it(target) -> None:
    rows = ["GGFGG", "GGFGG"]
    mm = _grid(rows)
    stroke = _stroke(mm, 0, 0, target, FLOOD_CONTIGUOUS)
    assert stroke.region == frozenset({0, 1, 5, 6})
    replace_tiles(mm, _all(mm), stroke.ref, target, D, stroke.region)
    after = [(t.terrain_id, t.layer) for t in mm.terrain]
    written = (D, -1) if target == TERRAIN else (G, D)
    assert [i for i, pair in enumerate(after) if pair == written] == [0, 1, 5, 6]

    mm = _grid(rows)
    stroke = _stroke(mm, 0, 0, target, FLOOD_GLOBAL)
    assert stroke.region is None
    replace_tiles(mm, _all(mm), stroke.ref, target, D, stroke.region)
    assert [i for i, t in enumerate(mm.terrain) if (t.terrain_id, t.layer) == written] == [0, 1, 3, 4, 5, 6, 8, 9]


def test_the_press_tile_governs_the_match_even_outside_a_square_outline_footprint() -> None:
    """Size 5, square outline: the centre is not painted, yet it is the reference."""
    mm = _grid(["GGGGG", "GFFFG", "GFWFG", "GFFFG", "GGGGG"])
    footprint = brush.brush_tiles(2, 2, 5, brush.BRUSH_SHAPE_SQUARE_OUTLINE, 5, 5, 1)
    assert (2, 2) not in footprint
    stroke = _stroke(mm, 2, 2, TERRAIN)
    assert stroke.ref == (W, -1)
    # The ring is all Grass, which is not Water: nothing changes.
    assert replace_tiles(mm, footprint, stroke.ref, TERRAIN, D) == []
    mm.terrain[0].terrain_id = W
    assert replace_tiles(mm, footprint, stroke.ref, TERRAIN, D) == [0]


def _paint_can_cases():
    """Every target, a paint value equal to the press tile's base among the
    values, and a region with and without layered tiles."""
    for target, layered in itertools.product((TERRAIN, LAYER, CLEAR), (False, True)):
        for paint in (G, F, W, D):
            yield target, layered, paint


@pytest.mark.parametrize(("target", "layered", "paint"), list(_paint_can_cases()))
def test_contiguous_over_the_whole_map_changes_exactly_paint_cans_set(target, layered, paint) -> None:
    def make():
        mm = _grid(["GGFGG", "GWFGG", "GGGFF", "FFGGW"])
        if layered:
            for i in (1, 7, 12, 13):
                mm.terrain[i].layer = W
            mm.terrain[0].layer = W  # the press tile itself layered, so the pair region is the layered set
        return mm

    for x, y in ((0, 0), (3, 0), (2, 2), (4, 3)):
        expected = set(fill_indices(make(), x, y, target, paint))
        mm = make()
        stroke = _stroke(mm, x, y, target, FLOOD_CONTIGUOUS)
        changed = replace_tiles(mm, _all(mm), stroke.ref, target, paint, stroke.region)
        assert set(changed) == expected, (x, y)
        assert len(changed) == len(set(changed))


def test_off_map_and_out_of_region_tiles_are_skipped_never_raised_on() -> None:
    mm = _grid(["GGG", "GGG"])
    tiles = [(-1, 0), (3, 0), (0, 2), (0, -1), (1, 0), (2, 1)]
    assert replace_tiles(mm, tiles, (G, -1), TERRAIN, D, region=frozenset({1})) == [1]
    assert replace_tiles(mm, tiles, (G, -1), TERRAIN, D) == [5]


def test_an_off_map_press_takes_no_reference_and_changes_nothing() -> None:
    """Flat picks x == map_width on the map rect's far edge: its flat index
    would wrap onto the next row's first tile, which here matches."""
    mm = _grid(["GGG", "GGG"])
    for x, y in ((3, 0), (-1, 0), (0, 2)):
        stroke = _stroke(mm, x, y, TERRAIN, FLOOD_CONTIGUOUS)
        assert stroke.pressed and stroke.ref is None and stroke.region is None
        assert replace_tiles(mm, _all(mm), stroke.ref, TERRAIN, D, stroke.region) == []
    assert all(t.terrain_id == G for t in mm.terrain)


def test_a_no_op_write_is_not_reported() -> None:
    mm = _grid(["GG"])
    mm.terrain[1].layer = W
    assert replace_tiles(mm, _all(mm), (G, -1), TERRAIN, G) == [1]


def test_an_unknown_target_raises() -> None:
    mm = FakeMapManager([FakeTile(G)], 1, 1)
    with pytest.raises(ValueError):
        replace_tiles(mm, [(0, 0)], (G, -1), "bogus", D)
    with pytest.raises(ValueError):
        replace_tiles(mm, [(0, 0)], None, "bogus", D)
    with pytest.raises(ValueError):
        terrain_replace.matches(mm.terrain[0], (G, -1), "bogus")


def test_the_tool_options_literals_are_the_modules_floods() -> None:
    spec = tool_options.option("replace_flood")
    assert spec.choices == terrain_replace.FLOODS == (FLOOD_GLOBAL, FLOOD_CONTIGUOUS)
    assert spec.default == FLOOD_GLOBAL
