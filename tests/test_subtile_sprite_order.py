"""Sprites sharing an anchor tile paint by sub-tile depth, not unit-list order
(TASK-016, GH #324).

render_sprites._accumulate_contribution() merges each unmoved entry by its
render_sprites._sprite_order(): the ground point's `y - x`, with a slotted
piece on the footprint's back edge only at -inf and on its front edge only at
+inf. Synthetic throughout, on test_sprite_columns' solid-rectangle art with a
`ground` per building piece. The behaviour tests read
render_sprites.SUBTILE_SPRITE_ORDER as it stands, so setting it False (the
revert) fails them; each also checks the switched-off render gets its case
wrong, so it is not vacuous.
"""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager

import numpy as np
import pytest
from test_clipped_sprites import backend, synthetic_textures  # noqa: F401 -- fixtures
from test_invalidate_units_splice import (
    _apply_and_check,
    _call_counts,
    _make_convert_cache,
    _move_splice,
    _place,
    _scenario,
    shared_art_install,  # noqa: F401 -- pytest fixture, imported for its name
)
from test_sprite_chunks import Unit
from test_sprite_chunks import _scenario as _synthetic_scenario
from test_sprite_columns import BUILDING, STYLES, UNIT, UNIT_COLOUR, _Art, _layer, _piece_of, _render, _solid

from descape import render_sprites, render_units, unit_sprites
from descape.terrain_palette import BUILDING_TILE_SPANS, tile_span
from testkit.render_family import render_global, set_render_global

from test_unit_sprites import CONST as SPRITE_CONST

OTHER = UNIT + 1
OTHER_COLOUR = (0, 255, 255)
assert tile_span(OTHER, render_units.NON_BUILDING_SPAN) == (1, 1)


class _GroundArt(_Art):
    """_Art plus a second 1x1 const in its own colour, and `grounds`: each
    building piece's (mx, my), as unit_graphic_map.json's pieces carry it."""

    def __init__(self):
        super().__init__()
        self.grounds: list | None = None

    def _make(self, unit_const, half_w):
        if unit_const == OTHER:
            half_h = half_w // 2
            draw = unit_sprites.SpriteDraw(_solid(2 * half_w, 4 * half_h, OTHER_COLOUR), half_w, 3 * half_h)
            return [unit_sprites.SpritePiece(draw, 0, 0)]
        pieces = super()._make(unit_const, half_w)
        if unit_const != BUILDING or self.grounds is None:
            return pieces
        return [dataclasses.replace(p, ground=g) for p, g in zip(pieces, self.grounds, strict=True)]


@pytest.fixture
def art(monkeypatch):
    fake = _GroundArt()
    monkeypatch.setattr(unit_sprites, "sprite_pieces_for", fake)
    monkeypatch.setitem(BUILDING_TILE_SPANS, BUILDING, (3, 3))
    return fake


def _players(*lists) -> object:
    """One list of (const, x, y) per player, from GAIA up."""
    return _synthetic_scenario([[Unit(x, y, const) for const, x, y in placed] for placed in lists])


def _winner(style: str, stitched: bool, a, b, both) -> str:
    """Which of two one-sprite scenarios `both` shows over their overlap: "a", "b" or "mixed"."""
    ground = _render(style, _players(), stitched)
    a_img, b_img, both_img = (_render(style, s, stitched) for s in (a, b, both))
    overlap = np.any(a_img != ground, axis=-1) & np.any(b_img != ground, axis=-1)
    assert overlap.any(), "the two sprites do not overlap, so this proves nothing"
    if np.array_equal(both_img[overlap], a_img[overlap]):
        return "a"
    return "b" if np.array_equal(both_img[overlap], b_img[overlap]) else "mixed"


@contextmanager
def _unordered():
    """SUBTILE_SPRITE_ORDER off, for each test's in-test revert check."""
    with pytest.MonkeyPatch.context() as m:
        set_render_global(m, "SUBTILE_SPRITE_ORDER", False)
        yield


# Both on tile (5, 5) at x + y = 11, one screen column apart in depth: NEAR (d = +0.5) is in front.
NEAR = (5.25, 5.75)
FAR = (5.75, 5.25)


@pytest.mark.parametrize("style", STYLES)
def test_the_nearer_of_two_units_on_one_tile_wins_their_overlap(art, backend, style):  # noqa: F811
    """Owners 0/1 and 1/0, so either one comes first in the walk."""
    near, far = (UNIT, *NEAR), (OTHER, *FAR)
    for near_owner in (0, 1):
        lists = [[near], [far]] if near_owner == 0 else [[far], [near]]
        a = _players([near]) if near_owner == 0 else _players([], [near])
        b = _players([], [far]) if near_owner == 0 else _players([far])
        both = _players(*lists)
        for stitched in (False, True):
            assert _winner(style, stitched, a, b, both) == "a", f"near owned by {near_owner}, stitched={stitched}"
            with _unordered():
                # Walk order: the later player's unit paints last.
                assert _winner(style, stitched, a, b, both) == ("a" if near_owner == 1 else "b")


@pytest.mark.parametrize("style", STYLES)
def test_two_units_at_one_point_keep_walk_order(art, style):
    point = (5.5, 5.5)
    for first, second in ((UNIT, OTHER), (OTHER, UNIT)):
        a, b = _players([(first, *point)]), _players([], [(second, *point)])
        both = _players([(first, *point)], [(second, *point)])
        for stitched in (False, True):
            assert _winner(style, stitched, a, b, both) == "b"
            ordered = _render(style, both, stitched)
            with _unordered():
                assert np.array_equal(ordered, _render(style, both, stitched))


# Building (3x3) at (5.5, 5.5), tiles 4..6, one slotted piece: (slot, ground, unit point, piece paints over the unit).
EDGE_CASES = {
    # Back corner (6, 4): on both back edges, -inf; the unit at d = -1.4 still paints over it.
    "back": ([2, 0], (1.5, -1.5), (6.2, 4.8), False),
    # Front corner (4, 6): +inf, over a unit at d = +1.4.
    "front": ([0, 2], (-1.5, 1.5), (4.8, 6.2), True),
    # Left corner (4, 4), one back and one front edge: its point depth 0, so d = +0.4 is in front of it...
    "side-behind": ([0, 0], (-1.5, -1.5), (4.3, 4.7), False),
    # ...and d = -0.4 behind it.
    "side-in-front": ([0, 0], (-1.5, -1.5), (4.7, 4.3), True),
}


@pytest.mark.parametrize("case", EDGE_CASES)
@pytest.mark.parametrize("style", STYLES)
def test_a_slotted_edge_piece_sorts_against_a_unit_on_its_slot_tile(art, backend, style, case):  # noqa: F811
    slot, ground, point, piece_over = EDGE_CASES[case]
    art.slots, art.grounds = [slot], [ground]
    building, unit = (BUILDING, 5.5, 5.5), (UNIT, *point)
    a, b = _players([], [building]), _players([], [unit])
    layer, _proj = _layer(b)
    assert list(layer.by_anchor) == [(int(point[0]), int(point[1]))]
    assert (int(point[0]), int(point[1])) == (4 + slot[0], 4 + slot[1]), "the unit is not on the piece's slot tile"
    want = "a" if piece_over else "b"
    for unit_first in (True, False):
        both = _players([], [unit, building] if unit_first else [building, unit])
        for stitched in (False, True):
            assert _winner(style, stitched, a, b, both) == want, f"unit_first={unit_first}, stitched={stitched}"
        if unit_first != piece_over:
            with _unordered():
                assert _winner(style, False, a, b, both) != want, f"list order got {case} right by luck"


def test_a_composites_own_pieces_keep_list_order_on_one_tile(art, backend):  # noqa: F811
    """Two pieces on the centre tile whose own orders decrease: the merge keeps
    a contribution's entries in list order, so drawn alone it is byte-identical
    to the walk-order append."""
    art.slots, art.grounds = [[1, 1], [1, 1]], [(-0.4, 0.4), (0.4, -0.4)]
    scn = _players([], [(BUILDING, 5.5, 5.5)])
    layer, proj = _layer(scn)
    pieces = unit_sprites.sprite_pieces_for(BUILDING, 0.0, 1, proj.half_w)
    unmoved = [e for e in layer.by_anchor[(5, 5)] if type(e[0]) is not unit_sprites.MovedSlice]
    assert [_piece_of(e[0], pieces) for e in unmoved] == [0, 1]
    assert unmoved[0][4] > unmoved[1][4], "the pieces' orders do not decrease, so this proves nothing"
    for style in STYLES:
        for stitched in (False, True):
            ordered = _render(style, scn, stitched)
            with _unordered():
                assert np.array_equal(ordered, _render(style, scn, stitched)), (style, stitched)


def test_the_order_value_reads_edges_only_on_slotted_pieces():
    draw = unit_sprites.SpriteDraw(_solid(2, 2, UNIT_COLOUR), 1, 1)

    def order(ground, slot=(0, 0)):
        return render_sprites._sprite_order(unit_sprites.SpritePiece(draw, 0, 0, slot=slot, ground=ground), 10.0, 20.0, (4, 4))

    assert order((2.0, -2.0)) == order((1.3, -2.0)) == order((2.0, -1.3)) == -np.inf
    assert order((-2.0, 2.0)) == order((-1.3, 2.0)) == np.inf
    assert order((-2.0, -2.0)) == 10.0 and order((2.0, 2.0)) == 10.0
    assert order((2.0, -2.0), slot=None) == 6.0
    assert order((0.5, 0.25)) == 9.75


# --- splices ----------------------------------------------------------------

SHARED_TILE = (40, 40)


@pytest.mark.parametrize("landing", [(40.1, 40.9), (40.9, 40.1)], ids=["in-front", "behind"])
@pytest.mark.parametrize("style", STYLES)
def test_a_unit_moved_onto_a_shared_tile_merges_by_depth_like_a_fresh_cache(style, landing, shared_art_install, monkeypatch):  # noqa: F811
    """A membership component: a third unit lands on a tile two others hold,
    listed against their depth order, in front of both or behind both."""
    assert render_global("SUBTILE_SPRITE_ORDER")
    scenario = _scenario()
    x, y = SHARED_TILE
    _place(scenario, 1, SPRITE_CONST, x + 0.25, y + 0.75)
    _place(scenario, 2, SPRITE_CONST, x + 0.75, y + 0.25)
    mover = _place(scenario, 3, SPRITE_CONST, 10.5, 10.5)
    cache = _make_convert_cache(style, scenario, True)
    counts = _call_counts(monkeypatch)

    index = next(i for i, u in enumerate(scenario.unit_manager.units[3]) if u is mover)
    splice = _move_splice(scenario, 3, index, mover, *landing)
    _apply_and_check(style, cache, scenario, True, counts, [splice])

    orders = [entry[4] for entry in cache._levels[0].sprites.by_anchor[SHARED_TILE]]
    want = sorted([0.5, -0.5, landing[1] - landing[0]])
    assert orders == want and orders != [0.5, -0.5, landing[1] - landing[0]], orders
