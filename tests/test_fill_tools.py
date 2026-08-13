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

from descape.edit_history import EditHistory
from descape.fill_tools import contiguous_region, flood_fill_terrain


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


def test_source_equals_target_writes_nothing() -> None:
    mm = _grid(["AAA", "AAA", "AAA"])
    mm.terrain[4].layer = 7  # center tile carries a real second terrain
    filled = flood_fill_terrain(mm, 1, 1, terrain_id=ord("A"))
    assert filled == []
    assert mm.terrain[4].layer == 7
    assert all(t.terrain_id == ord("A") for t in mm.terrain)


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


def test_blank_template_fill_covers_whole_map() -> None:
    from descape.scenario_io import load_map_and_units, BLANK_TEMPLATE_PATH

    scenario = load_map_and_units(BLANK_TEMPLATE_PATH)
    mm = scenario.map_manager
    filled = flood_fill_terrain(mm, 0, 0, terrain_id=10)
    assert len(filled) == mm.map_width * mm.map_height
    assert all(t.terrain_id == 10 for t in mm.terrain)
