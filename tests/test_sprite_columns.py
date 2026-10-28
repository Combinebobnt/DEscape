"""Multi-tile sprites paint per screen column (TASK-221).

A sprite piece used to paint once, at its footprint's last tile in depth order
(or its slot), so a unit standing just outside a building's front-left or
front-right face sorted before the building and was painted over. Each piece
is now cut into screen columns, each painting at the earlier of its column's
front footprint tile and the piece's own key (render._sprite_columns()).

Synthetic throughout: a fake unit_sprites.sprite_pieces_for draws solid
rectangles sized from the tile, so coverage is countable by colour. The
behaviour tests read render.SPRITE_COLUMN_SLICING as it stands, so setting it
False (the revert) fails them; the structural ones carry non-vacuity asserts.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest
from test_clipped_sprites import _full, _stitched, backend, synthetic_textures  # noqa: F401 -- fixtures
from test_invalidate_units_splice import (
    _apply,
    _call_counts,
    _make_cache,
    _move_splice,
    _oracle,
    _splice_eligible,
    _sprite_layer,
)
from test_invalidate_units_splice import _place as _template_place
from test_invalidate_units_splice import _scenario as _template_scenario
from test_sprite_chunks import MAP_H, MAP_W, Unit, _scenario

from descape import render, unit_sprites

from test_unit_sprites import CONST as UNIT

STYLES = ("stepped", "sloped")
BUILDING = UNIT + 2  # its span comes from the `building` fixture
DIAGONAL_GATE = 659  # a real BUILDING_TILE_OFFSETS const: six tiles along x == y
OWN_POINT = next(c for c, span in render.DOODAD_TILE_SPANS.items() if span[0] > 1)
UNIT_COLOUR = (255, 0, 255)
PIECE_COLOURS = ((0, 200, 40), (40, 0, 200), (200, 120, 0))


def _solid(w: int, h: int, colour) -> np.ndarray:
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = colour
    rgba[..., 3] = 255
    return rgba


class _Art:
    """The fake sprite_pieces_for. A 1x1 unit is two half-widths wide (its own
    two screen columns) and stands 3 half-heights tall; a building covers its
    footprint's columns exactly and stands 4 half-heights above its diamond.
    `slots` makes the building a composite of full-size pieces with those
    slots. Draws are built once per half_w: the splice and memo machinery
    compare them by identity."""

    def __init__(self):
        self.slots: list | None = None
        self.overhang = 0
        self._made: dict = {}

    def __call__(self, unit_const, rotation, team_index, half_w, *args, **kwargs):
        key = (unit_const, half_w)
        if key not in self._made:
            self._made[key] = self._make(unit_const, half_w)
        return self._made[key]

    def _make(self, unit_const, half_w):
        half_h = half_w // 2
        if unit_const == UNIT:
            draw = unit_sprites.SpriteDraw(_solid(2 * half_w, 4 * half_h, UNIT_COLOUR), half_w, 3 * half_h)
            return [unit_sprites.SpritePiece(draw, 0, 0)]
        if unit_const not in (BUILDING, DIAGONAL_GATE, OWN_POINT):
            return []
        span_x, span_y = render.tile_span(unit_const, render.NON_BUILDING_SPAN)
        n = span_x + span_y
        w, h = (n + 2 * self.overhang) * half_w, (n + 4) * half_h
        hotspot_x, hotspot_y = w // 2, h - n * half_h // 2
        slots = self.slots or [None]
        # A composite's pieces are nudged apart a few px, so each shows past the
        # others and their paint order changes pixels.
        return [
            unit_sprites.SpritePiece(
                unit_sprites.SpriteDraw(_solid(w, h, colour), hotspot_x - 3 * i, hotspot_y + 5 * i), 0, 0, slot=slot
            )
            for i, (slot, colour) in enumerate(zip(slots, PIECE_COLOURS[: len(slots)], strict=True))
        ]


@pytest.fixture
def art(monkeypatch):
    fake = _Art()
    monkeypatch.setattr(unit_sprites, "sprite_pieces_for", fake)
    monkeypatch.setitem(render.BUILDING_TILE_SPANS, BUILDING, (3, 3))
    return fake


def _units(placed):
    """(const, x, y) triples to player 1's list, in that order."""
    return _scenario([[], [Unit(x, y, const) for const, x, y in placed]])


def _render(style: str, scn, stitched: bool) -> np.ndarray:
    return _stitched(style, scn) if stitched else _full(style, scn, with_sprites=True)[0]


def _unit_px(img: np.ndarray, rect) -> int:
    x0, y0, x1, y1 = rect
    return int(np.all(img[y0:y1, x0:x1, :3] == UNIT_COLOUR, axis=-1).sum())


def _unit_rect(style: str, scn) -> tuple[int, int, int, int]:
    _img, elevations, corner_rise, proj = _full(style, scn, with_sprites=True)
    layer = render.sprite_draws_by_anchor(scn, proj, elevations, corner_rise=corner_rise)
    for entries in layer.by_anchor.values():
        for draw, ax, ay, _rise in entries:
            if tuple(draw.rgba[0, 0, :3]) == UNIT_COLOUR:
                h, w = draw.rgba.shape[:2]
                return ax - draw.hotspot_x, ay - draw.hotspot_y, ax - draw.hotspot_x + w, ay - draw.hotspot_y + h
    raise AssertionError("the unit resolved no sprite")


def _covered(style: str, building, tile, stitched: bool, unit_first: bool) -> tuple[int, int]:
    """(px of the unit's sprite the building covers, the unit's own px)."""
    unit = (UNIT, tile[0] + 0.5, tile[1] + 0.5)
    alone = _units([unit])
    both = _units([unit, building] if unit_first else [building, unit])
    rect = _unit_rect(style, alone)
    own = _unit_px(_render(style, alone, stitched), rect)
    return own - _unit_px(_render(style, both, stitched), rect), own


# Per case: the building, the tiles in front of a face (a unit there must be
# fully visible), and the tiles behind it (covered, the control). The front
# tiles are the ones the single-anchor render got wrong, plus two it got right.
CASES = {
    # Footprint 4-6 / 4-6, key (4, 6): (3, 5) is at the front-left face, (6, 7) the front-right.
    "barracks": ((BUILDING, 5.5, 5.5), [(3, 5), (6, 7), (5, 7), (3, 7)], [(7, 3)]),
    # Footprint 4-7 / 5, key (4, 5): (6, 6) is in front of its long face.
    "gate": ((BUILDING, 6.0, 5.5), [(6, 6), (7, 6)], [(6, 4)]),
    # Tiles (4, 4) (5, 5) (6, 6) (7, 7) (5, 6) (6, 5), key (5, 6): (4, 5) is in front of its left end.
    "diagonal gate": ((DIAGONAL_GATE, 6.0, 6.0), [(4, 5), (7, 8)], [(5, 4)]),
}


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("style", STYLES)
def test_a_unit_in_front_of_a_face_draws_over_the_building(art, backend, style, case, monkeypatch):  # noqa: F811
    building, front, behind = CASES[case]
    if case == "gate":
        monkeypatch.setitem(render.BUILDING_TILE_SPANS, BUILDING, (4, 1))
    failures = []
    for stitched in (False, True):
        for unit_first in (False, True):
            for tile in front + behind:
                covered, own = _covered(style, building, tile, stitched, unit_first)
                assert own > 0, f"the unit at {tile} drew nothing"
                if (tile in front) != (covered == 0):
                    where = "stitched" if stitched else "full"
                    failures.append(f"{tile} {where} unit_first={unit_first}: {covered} of {own} px covered")
    assert not failures, failures


def _building_px(img: np.ndarray, colour=PIECE_COLOURS[0]) -> int:
    return int(np.all(img[..., :3] == colour, axis=-1).sum())


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("raised", [(3, 5), (6, 7)], ids=["left-face", "right-face"])
def test_higher_ground_in_front_of_a_face_hides_its_lower_ink(art, style, raised, monkeypatch):
    """The intended change on uneven ground: a face's columns paint at their
    front tile, so a nearer tile standing higher than the building's base now
    covers that face's lower ink (the GH #107 rule, per column). The single
    anchor painted the whole building after it."""
    scn = _units([(BUILDING, 5.5, 5.5)])
    flat = _building_px(_render(style, scn, False))
    scn.map_manager.get_tile(*raised).elevation = 2
    now = _building_px(_render(style, scn, False))
    with monkeypatch.context() as m:
        m.setattr(render, "SPRITE_COLUMN_SLICING", False)
        single = _building_px(_render(style, scn, False))
    assert single <= flat
    assert now < single, f"raised {raised}: {now} building px against the single anchor's {single}"


def test_a_moved_slice_paints_after_a_unit_on_its_own_tile(art, backend):  # noqa: F811
    """A unit inside the footprint, on the tile a face column moved to: the
    moved slice paints after every unmoved entry there, as the whole building
    did from its later key, so the building still covers the unit."""
    building = (BUILDING, 5.5, 5.5)
    scn = _units([building])
    _img, elevations, _corner_rise, proj = _full("stepped", scn, with_sprites=True)
    layer = render.sprite_draws_by_anchor(scn, proj, elevations)
    assert any(type(d) is unit_sprites.MovedSlice for d, *_ in layer.by_anchor[(4, 4)]), "no slice moved to (4, 4)"
    for unit_first in (False, True):
        covered, own = _covered("stepped", building, (4, 4), True, unit_first)
        assert covered == own > 0


def test_a_composite_drawn_alone_is_byte_identical_to_the_single_anchor(art, backend, monkeypatch):  # noqa: F811
    """Moved slices from several pieces meet on one tile in their own keys'
    depth order, which is the order the pieces painted in before. Pasture-like
    slots, out of depth order in the piece list, and the pieces overlap a few
    px apart, so any other order shows."""
    art.slots = [[1, 2], [0, 2], [0, 1]]
    scn = _units([(BUILDING, 5.5, 5.5)])
    layer, _proj = _layer(scn)
    assert any(
        sum(type(d) is unit_sprites.MovedSlice for d, *_ in v) >= 2 for v in layer.by_anchor.values()
    ), "no tile holds moved slices of two pieces, so this proves nothing"
    for style in STYLES:
        for stitched in (False, True):
            sliced = _render(style, scn, stitched)
            with monkeypatch.context() as m:
                m.setattr(render, "SPRITE_COLUMN_SLICING", False)
                single = _render(style, scn, stitched)
            assert np.array_equal(sliced, single), (style, stitched)


def _layer(scn):
    _img, elevations, _corner_rise, proj = _full("stepped", scn, with_sprites=True)
    return render.sprite_draws_by_anchor(scn, proj, elevations), proj


def _piece_of(draw, pieces) -> int:
    [index] = [i for i, p in enumerate(pieces) if np.shares_memory(draw.rgba, p.draw.rgba)]
    return index


def test_moved_slices_at_one_tile_keep_their_pieces_depth_order(art):
    art.slots = [[1, 2], [0, 2], [0, 1]]
    scn = _units([(BUILDING, 5.5, 5.5)])
    layer, proj = _layer(scn)
    pieces = unit_sprites.sprite_pieces_for(BUILDING, 0.0, 1, proj.half_w)
    # (4, 4) is the front of the leftmost column: every piece's key is later, so all three move there.
    order = [_piece_of(d, pieces) for d, *_ in layer.by_anchor[(4, 4)]]
    assert all(type(d) is unit_sprites.MovedSlice for d, *_ in layer.by_anchor[(4, 4)])
    # Keys (4, 5), (5, 6) and (4, 6) in depth order: pieces 2, 0, 1.
    assert order == [2, 0, 1]


@pytest.mark.parametrize("overhang", [0, 2])
@pytest.mark.parametrize(
    "const, at", [(BUILDING, (5.5, 5.5)), (BUILDING, (0.5, 5.5)), (DIAGONAL_GATE, (6.0, 6.0))],
    ids=["barracks", "off-the-left-edge", "diagonal-gate"],
)
def test_slices_partition_each_piece_inside_the_footprint(art, const, at, overhang):
    """Every column of every piece paints exactly once, at a footprint tile;
    columns off the occupied strips (art overhang, or a footprint clamped at
    the map edge) stay at the piece's own key, unmoved."""
    art.overhang = overhang
    art.slots = [None, [0, 0]] if const == BUILDING else None
    scn = _units([(const, *at)])
    layer, proj = _layer(scn)
    tiles = render.unit_occupied_tiles(scn.unit_manager.units[1][0], MAP_W, MAP_H)
    lo = min(x + y for x, y in tiles)
    hi = max(x + y for x, y in tiles) + 1
    pieces = unit_sprites.sprite_pieces_for(const, 0.0, 1, proj.half_w)
    key = unit_sprites.sprite_anchor_tile(tiles)
    span_x, span_y = render.tile_span(const, render.NON_BUILDING_SPAN)
    # A slot counts from the unclamped footprint start, then clamps onto the map (_resolve_unit_sprite()).
    start = render._span_start(at[0], span_x), render._span_start(at[1], span_y)
    lo_x, hi_x = min(x for x, _ in tiles), max(x for x, _ in tiles)
    lo_y, hi_y = min(y for _, y in tiles), max(y for _, y in tiles)

    def own_key(piece):
        if piece.slot is None:
            return key
        return min(max(start[0] + piece.slot[0], lo_x), hi_x), min(max(start[1] + piece.slot[1], lo_y), hi_y)

    covered = [np.zeros(p.draw.rgba.shape[1], dtype=int) for p in pieces]
    moved = overhung = 0
    for k, entries in layer.by_anchor.items():
        assert k in tiles
        for draw, px, _py, _rise in entries:
            i = _piece_of(draw, pieces)
            parent = pieces[i].draw
            own = own_key(pieces[i])
            c0 = parent.hotspot_x - draw.hotspot_x
            c1 = c0 + draw.rgba.shape[1]
            assert np.array_equal(draw.rgba, parent.rgba[:, c0:c1])
            covered[i][c0:c1] += 1
            assert (k != own) == (type(draw) is unit_sprites.MovedSlice)
            moved += k != own
            strips = {(px - draw.hotspot_x + c - proj.origin_x) // proj.half_w for c in (0, c1 - c0 - 1)}
            if not all(lo <= s <= hi for s in strips):
                assert k == own, f"an overhanging column moved to {k}"
                overhung += 1
    assert all((c == 1).all() for c in covered), "a column painted twice or not at all"
    assert moved, "nothing moved, so this proves nothing"
    if overhang or at[0] < 1.5:
        assert overhung, "no column overhangs the occupied strips, so the K rule went unchecked"


def test_a_strip_with_no_tile_takes_its_left_neighbours_front():
    """_sprite_columns() on a sparse footprint, (0, 0) and (3, 3): strips 2-5
    have no tile on either line, so they take strip 1's front, (0, 0), and
    only the strips over (3, 3) stay at the key."""
    proj = _layer(_units([]))[1]
    hw = proj.half_w
    runs = render._sprite_columns([(0, 0), (3, 3)], (3, 3), proj.origin_x, 8 * hw, proj)
    assert runs == [((0, 0), 0, 6 * hw, True), ((3, 3), 6 * hw, 8 * hw, False)]


def test_one_tile_units_and_own_point_doodads_paint_whole(art):
    scn = _units([(UNIT, 2.5, 2.5), (OWN_POINT, 8.5, 8.5)])
    layer, proj = _layer(scn)
    entries = [e for v in layer.by_anchor.values() for e in v]
    assert len(entries) == 2
    whole = {id(p.draw) for c in (UNIT, OWN_POINT) for p in unit_sprites.sprite_pieces_for(c, 0.0, 1, proj.half_w)}
    assert {id(d) for d, *_ in entries} == whole
    assert render.tile_span(OWN_POINT, render.NON_BUILDING_SPAN)[0] > 1, "the doodad is one tile, so this proves nothing"


@pytest.mark.parametrize("style", STYLES)
def test_moving_a_building_beside_a_unit_splices_like_a_fresh_render(art, style, monkeypatch):
    """The splice re-derives every column key of the moved building: it ends
    with its left face beside the unit, so a key left behind or missing shows
    as pixels against a fresh render."""
    scenario = _template_scenario()
    _template_place(scenario, 1, UNIT, 38.5, 40.5)
    building = _template_place(scenario, 1, BUILDING, 41.5, 40.5)
    cache = _make_cache(style, scenario, sprites=True)
    canvas_w, canvas_h = cache.canvas_dims(0)
    counts = _call_counts(monkeypatch)

    index = next(i for i, u in enumerate(scenario.unit_manager.units[1]) if u is building)
    splice = _move_splice(scenario, 1, index, building, 40.5, 40.5)
    assert _splice_eligible(cache.units_by_tile, splice), "fixture is not testing the splice path"
    _apply(cache, splice)

    assert counts == {"building_bboxes": 0, "sprites": 0}

    def keyed(layer):
        return {k: [(id(d), px, py, rise) for d, px, py, rise in v] for k, v in layer.by_anchor.items()}

    layer = _sprite_layer(cache)
    assert any(type(d) is unit_sprites.MovedSlice for v in layer.by_anchor.values() for d, *_ in v)
    assert keyed(layer) == keyed(_sprite_layer(_make_cache(style, scenario, sprites=True)))
    stitched = cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)
    assert np.array_equal(stitched, _oracle(style, scenario, sprites=True)[:canvas_h, :canvas_w])


def test_a_slice_group_lives_exactly_as_long_as_its_parent():
    parent = unit_sprites.SpriteDraw(_solid(8, 4, UNIT_COLOUR), 4, 2)
    a = unit_sprites.slice_draw(parent, 0, 4, None)
    assert unit_sprites.slice_draw(parent, 0, 4, None) is a
    moved = unit_sprites.slice_draw(parent, 0, 4, (1, 2))
    assert type(moved) is unit_sprites.MovedSlice and moved.rgba is a.rgba
    assert id(parent) in unit_sprites._slice_groups
    key = id(parent)
    del parent, a, moved
    gc.collect()
    assert key not in unit_sprites._slice_groups
