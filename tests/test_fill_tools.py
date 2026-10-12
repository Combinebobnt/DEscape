"""Algorithm-level coverage for descape/fill_tools.py's Paint Can flood fill.

Uses a fake-tile/fake-map-manager protocol (plain objects exposing
map_width/map_height/terrain, and terrain_id/elevation/layer per tile)
rather than real AoE2ScenarioParser objects, matching tests/test_edit_
history.py's own precedent for exactly this reason: fill_tools only ever
reads mm.map_width/map_height/terrain and each tile's terrain_id/layer, so
staying duck-typed keeps these tests fast and independent of the parser.
FakeTile is duplicated here rather than imported from test_edit_history.py,
matching that module's own deliberately self-contained style.
"""

from __future__ import annotations

import pytest

from descape.edit_history import EditHistory
from descape.fill_tools import contiguous_region, fill_indices, flood_fill, flood_fill_terrain
from descape.terrain_target import CLEAR, LAYER, TERRAIN


class FakeTile:
    def __init__(self, terrain_id: int = 0, elevation: int = 0, layer: int = -1):
        self.terrain_id = terrain_id
        self.elevation = elevation
        self.layer = layer


class FakeMapManager:
    def __init__(self, terrain: list[FakeTile], map_width: int, map_height: int):
        self.terrain = terrain
        self.map_width = map_width
        self.map_height = map_height


def _grid(rows: list[str]) -> FakeMapManager:
    """Builds a FakeMapManager from a list of equal-length strings, one
    character per tile, row-major -- character value used directly as
    terrain_id (ord()) so distinct letters are trivially distinct terrains."""
    height = len(rows)
    width = len(rows[0])
    assert all(len(r) == width for r in rows)
    terrain = [FakeTile(terrain_id=ord(ch)) for row in rows for ch in row]
    return FakeMapManager(terrain, width, height)


def test_uniform_map_fills_every_tile() -> None:
    mm = FakeMapManager([FakeTile(terrain_id=0) for _ in range(120 * 120)], 120, 120)
    filled = flood_fill_terrain(mm, 60, 60, terrain_id=5)
    assert len(filled) == 120 * 120
    assert all(t.terrain_id == 5 and t.layer == -1 for t in mm.terrain)


def test_returned_indices_are_unique() -> None:
    mm = FakeMapManager([FakeTile(terrain_id=0) for _ in range(50 * 50)], 50, 50)
    filled = flood_fill_terrain(mm, 0, 0, terrain_id=1)
    assert len(set(filled)) == len(filled)


def test_same_terrain_fill_clears_only_the_layered_tiles() -> None:
    mm = _grid(["AAA", "AAB", "AAB"])
    mm.terrain[1].layer = 9
    mm.terrain[4].layer = 7  # two A tiles carry a layered terrain
    mm.terrain[5].layer = 7  # a B tile outside the clicked region keeps its own
    filled = flood_fill_terrain(mm, 0, 0, terrain_id=ord("A"))
    assert sorted(filled) == [1, 4]
    assert [t.layer for t in mm.terrain] == [-1, -1, -1, -1, -1, 7, -1, -1, -1]
    assert "".join(chr(t.terrain_id) for t in mm.terrain) == "AAAAABAAB"


def test_same_terrain_fill_with_no_layers_writes_nothing() -> None:
    mm = _grid(["AAA", "AAA", "AAA"])
    assert flood_fill_terrain(mm, 1, 1, terrain_id=ord("A")) == []
    assert all((t.terrain_id, t.layer) == (ord("A"), -1) for t in mm.terrain)


def test_no_op_fill_pushes_no_undo_record() -> None:
    mm = _grid(["AA", "AA"])
    hist = EditHistory()
    dirty = hist.apply("Fill terrain", mm.terrain, lambda: flood_fill_terrain(mm, 0, 0, ord("A")))
    assert dirty == []
    assert hist.records == []
    assert not hist.can_undo
    assert not hist.is_dirty


def test_region_bounded_by_map_edges() -> None:
    mm = _grid(["AAA", "AAA", "AAA"])
    filled = flood_fill_terrain(mm, 0, 0, terrain_id=ord("B"))
    assert len(filled) == 9
    assert all(t.terrain_id == ord("B") for t in mm.terrain)


def test_left_edge_does_not_wrap_to_previous_row() -> None:
    # Column 0 and the rightmost column are both A; everything between is B.
    # A flat-index wraparound bug would let a fill from (0, 1) leak across
    # the row boundary onto column width-1 of the row above/below.
    mm = _grid(["ABBBA", "ABBBA", "ABBBA"])
    filled = flood_fill_terrain(mm, 0, 1, terrain_id=ord("C"))
    width = mm.map_width
    for i in filled:
        x = i % width
        assert x == 0, f"fill leaked to x={x} (index {i})"


def test_diagonal_neighbours_do_not_leak() -> None:
    mm = _grid(["ABB", "BAB", "BBA"])  # A only on the main diagonal
    filled = flood_fill_terrain(mm, 0, 0, terrain_id=ord("C"))
    assert filled == [0]


def test_enclosed_region_does_not_escape_its_border() -> None:
    mm = _grid(
        [
            "AAAAA",
            "ABBBA",
            "ABABA",
            "ABBBA",
            "AAAAA",
        ]
    )
    filled = flood_fill_terrain(mm, 2, 2, terrain_id=ord("C"))
    assert filled == [2 * mm.map_width + 2]


def test_border_itself_is_not_repainted() -> None:
    mm = _grid(
        [
            "AAAAA",
            "ABBBA",
            "ABABA",
            "ABBBA",
            "AAAAA",
        ]
    )
    flood_fill_terrain(mm, 2, 2, terrain_id=ord("C"))
    width = mm.map_width
    for y, row in enumerate(["AAAAA", "ABBBA", "AB_BA", "ABBBA", "AAAAA"]):
        for x, ch in enumerate(row):
            if ch == "_":
                continue
            assert mm.terrain[y * width + x].terrain_id == ord(ch)


def test_non_square_map_fills_without_error() -> None:
    # Passes only because fill_tools never calls mm.get_tile()/get_tile_safe()
    # -- both raise/return-None for every coordinate on a non-square map on
    # real AoE2ScenarioParser MapManagers (map_size raises when width !=
    # height). This fake has no such method at all, so a regression that
    # reintroduces a get_tile() call would fail loudly with AttributeError.
    mm = _grid(["AAAAAA", "AAAAAA", "AAAAAA"])
    filled = flood_fill_terrain(mm, 0, 0, terrain_id=ord("B"))
    assert len(filled) == 18


def test_off_map_click_returns_empty() -> None:
    mm = _grid(["AA", "AA"])
    before = [t.terrain_id for t in mm.terrain]
    for x, y in [(-1, 0), (0, -1), (mm.map_width, 0), (0, mm.map_height)]:
        assert flood_fill_terrain(mm, x, y, terrain_id=ord("B")) == []
    assert [t.terrain_id for t in mm.terrain] == before


def test_fill_clears_stale_layer() -> None:
    mm = _grid(["AA", "AA"])
    for t in mm.terrain:
        t.layer = 7
    flood_fill_terrain(mm, 0, 0, terrain_id=ord("B"))
    assert all(t.layer == -1 for t in mm.terrain)


def test_contiguous_region_does_not_mutate() -> None:
    mm = _grid(["AAB", "AAB"])
    before = [(t.terrain_id, t.elevation, t.layer) for t in mm.terrain]
    region = contiguous_region(mm, 0, 0)
    assert len(region) == 4
    assert [(t.terrain_id, t.elevation, t.layer) for t in mm.terrain] == before


def test_undo_redo_round_trip() -> None:
    mm = _grid(["AAA", "AAA"])
    mm.terrain[0].layer = -1
    mm.terrain[1].layer = 7
    mm.terrain[2].layer = -1
    mm.terrain[3].layer = 7
    mm.terrain[4].layer = -1
    mm.terrain[5].layer = 7
    original = [(t.terrain_id, t.layer) for t in mm.terrain]

    hist = EditHistory()
    dirty = hist.apply("Fill terrain", mm.terrain, lambda: flood_fill_terrain(mm, 0, 0, ord("B")))
    assert len(dirty) == 6
    assert len(hist.records) == 1
    assert hist.can_undo
    assert not hist.can_redo

    hist.undo(mm.terrain)
    assert [(t.terrain_id, t.layer) for t in mm.terrain] == original
    assert not hist.can_undo
    assert hist.can_redo

    hist.redo(mm.terrain)
    assert all(t.terrain_id == ord("B") and t.layer == -1 for t in mm.terrain)
    assert hist.can_undo
    assert not hist.can_redo


def _layered(rows: list[str], layers: list[str]) -> FakeMapManager:
    """_grid(rows) with a layer per tile from `layers`: '.' is none, any other
    character its ord(), the same convention as the terrain rows."""
    mm = _grid(rows)
    flat = "".join(layers)
    assert len(flat) == len(mm.terrain)
    for tile, ch in zip(mm.terrain, flat, strict=True):
        tile.layer = -1 if ch == "." else ord(ch)
    return mm


def _state(mm: FakeMapManager) -> list[tuple[int, int, int]]:
    return [(t.terrain_id, t.elevation, t.layer) for t in mm.terrain]


def _layer_rows(mm: FakeMapManager) -> list[str]:
    flat = "".join("." if t.layer == -1 else chr(t.layer) for t in mm.terrain)
    return [flat[i : i + mm.map_width] for i in range(0, len(flat), mm.map_width)]


def test_contiguous_region_default_ignores_the_layer() -> None:
    mm = _layered(["AAB", "AAB"], ["x.x", ".y."])
    assert sorted(contiguous_region(mm, 0, 0)) == [0, 1, 3, 4]


def test_contiguous_region_match_layer_bounds_by_the_pair() -> None:
    # One base terrain, an x layer in two patches that touch only diagonally.
    mm = _layered(["AAAA", "AAAA", "AAAB"], ["xx..", "x.x.", "..xx"])
    assert sorted(contiguous_region(mm, 0, 0, match_layer=True)) == [0, 1, 4]
    assert sorted(contiguous_region(mm, 2, 1, match_layer=True)) == [6, 10]  # stops at the B base
    assert sorted(contiguous_region(mm, 3, 2, match_layer=True)) == [11]
    assert sorted(contiguous_region(mm, 2, 0, match_layer=True)) == [2, 3, 7]  # the unlayered pair


def test_layer_fill_covers_only_the_clicked_pair_region() -> None:
    mm = _layered(["AAAB", "AAAB", "AAAB"], ["x...", "x...", "...."])
    before_ids = [t.terrain_id for t in mm.terrain]
    filled = flood_fill(mm, 1, 0, LAYER, ord("y"))
    # The unlayered A tiles only: not the x patch, not the B column.
    assert sorted(filled) == [1, 2, 5, 6, 8, 9, 10]
    assert _layer_rows(mm) == ["xyy.", "xyy.", "yyy."]
    assert [t.terrain_id for t in mm.terrain] == before_ids


def test_layer_fill_on_the_same_layer_is_a_no_op() -> None:
    mm = _layered(["AAA", "AAA"], ["xx.", "xx."])
    before = _state(mm)
    assert fill_indices(mm, 0, 0, LAYER, ord("x")) == []
    assert flood_fill(mm, 0, 0, LAYER, ord("x")) == []
    assert _state(mm) == before


def test_layer_fill_with_the_base_terrain_clears_the_patch() -> None:
    mm = _layered(["AAA", "AAA"], ["xx.", "xxy"])
    filled = flood_fill(mm, 0, 0, LAYER, ord("A"))
    assert sorted(filled) == [0, 1, 3, 4]
    assert _layer_rows(mm) == ["...", "..y"]
    assert all(t.terrain_id == ord("A") for t in mm.terrain)


def test_layer_fill_with_the_base_terrain_on_an_unlayered_tile_is_a_no_op() -> None:
    mm = _layered(["AAA", "AAA"], ["xx.", "xx."])
    before = _state(mm)
    assert flood_fill(mm, 2, 0, LAYER, ord("A")) == []
    assert _state(mm) == before


def test_clear_fill_removes_one_patch_only() -> None:
    # Two x patches and a y patch on one base; the click clears its own patch.
    mm = _layered(["AAAAA", "AAAAA"], ["xx.xx", "xy.xx"])
    filled = flood_fill(mm, 0, 0, CLEAR, ord("Q"))
    assert sorted(filled) == [0, 1, 5]
    assert _layer_rows(mm) == ["...xx", ".y.xx"]
    assert all(t.terrain_id == ord("A") for t in mm.terrain)


def test_clear_fill_on_an_unlayered_tile_is_a_no_op() -> None:
    mm = _layered(["AAA", "AAA"], ["xx.", "xx."])
    before = _state(mm)
    assert fill_indices(mm, 2, 0, CLEAR, 0) == []
    assert flood_fill(mm, 2, 0, CLEAR, 0) == []
    assert _state(mm) == before


def test_layer_fill_pushes_one_undo_record_and_round_trips() -> None:
    mm = _layered(["AAB", "AAB"], ["x..", "..."])
    original = _state(mm)
    hist = EditHistory()
    dirty = hist.apply("Fill layer", mm.terrain, lambda: flood_fill(mm, 1, 0, LAYER, ord("y")))
    assert sorted(dirty) == [1, 3, 4]
    assert len(hist.records) == 1
    hist.undo(mm.terrain)
    assert _state(mm) == original
    hist.redo(mm.terrain)
    assert _layer_rows(mm) == ["xy.", "yy."]


def test_fill_indices_mutates_nothing() -> None:
    mm = _layered(["AAB", "AAB", "BBB"], ["x..", ".x.", "..."])
    before = _state(mm)
    for target in (TERRAIN, LAYER, CLEAR):
        for terrain_id in (ord("A"), ord("B"), ord("x"), ord("z")):
            for y in range(mm.map_height):
                for x in range(mm.map_width):
                    fill_indices(mm, x, y, target, terrain_id)
    assert _state(mm) == before


def test_fill_indices_is_exactly_what_flood_fill_changes() -> None:
    rows, layers = ["AAB", "AAB", "BBB"], ["x..", ".x.", "..."]
    for target in (TERRAIN, LAYER, CLEAR):
        for terrain_id in (ord("A"), ord("B"), ord("x"), ord("z")):
            for y in range(3):
                for x in range(3):
                    mm = _layered(rows, layers)
                    before = _state(mm)
                    predicted = fill_indices(mm, x, y, target, terrain_id)
                    written = flood_fill(mm, x, y, target, terrain_id)
                    changed = [i for i, (a, b) in enumerate(zip(before, _state(mm), strict=True)) if a != b]
                    assert written == predicted
                    assert sorted(predicted) == changed, (target, chr(terrain_id), x, y)


def test_fill_indices_terrain_target_matches_the_two_cases() -> None:
    mm = _layered(["AAB", "AAB"], ["x..", "..y"])
    assert sorted(fill_indices(mm, 0, 0, TERRAIN, ord("C"))) == [0, 1, 3, 4]  # differing id: the region
    assert fill_indices(mm, 0, 0, TERRAIN, ord("A")) == [0]  # same id: its layered tiles
    assert fill_indices(mm, 2, 0, TERRAIN, ord("B")) == [5]


def test_fill_indices_off_map_and_unknown_target() -> None:
    mm = _layered(["AA", "AA"], ["x.", ".."])
    for target in (TERRAIN, LAYER, CLEAR):
        for x, y in [(-1, 0), (0, -1), (mm.map_width, 0), (0, mm.map_height)]:
            assert fill_indices(mm, x, y, target, ord("B")) == []
    with pytest.raises(ValueError):
        fill_indices(mm, 0, 0, "overlay", ord("B"))
    with pytest.raises(ValueError):
        flood_fill(mm, 0, 0, "overlay", ord("B"))
    assert _layer_rows(mm) == ["x.", ".."]


def test_layer_and_clear_fill_on_a_non_square_map() -> None:
    # 6 wide, 2 tall: the same no-get_tile contract as the terrain fill above.
    mm = _layered(["AAAAAB", "AAAAAB"], ["xx....", "xx...."])
    assert sorted(flood_fill(mm, 5, 1, LAYER, ord("y"))) == [5, 11]
    assert sorted(flood_fill(mm, 0, 1, CLEAR, 0)) == [0, 1, 6, 7]
    assert _layer_rows(mm) == [".....y", ".....y"]
    # The right column must not wrap onto the next row's first tile.
    assert sorted(flood_fill(mm, 0, 0, LAYER, ord("z"))) == [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]


def test_blank_template_fill_covers_whole_map() -> None:
    from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units

    scenario = load_map_and_units(BLANK_TEMPLATE_PATH)
    mm = scenario.map_manager
    filled = flood_fill_terrain(mm, 0, 0, terrain_id=10)
    assert len(filled) == mm.map_width * mm.map_height
    assert all(t.terrain_id == 10 for t in mm.terrain)
