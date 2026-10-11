"""The Cliff tool's auto-connecting chain drag: its path, piece resolution and
layout, exercised headless (descape.cliff_chain has no Qt dependency; the
ViewerWindow wiring is tested separately in tests/test_cliff_tool_viewer.py).

Expected rows below (suffix, rotation, position, bounds, resolved) were
derived by an independent scratch oracle written from the cliff-chain plan's
text over descape/cliff_connectivity.json, never read back off the resolver.
The oracle brute-forces every suffix assignment under the reversible
pair-joint score `resolve_chain()` documents, then replays rotations, the
offset layout and the one contact pass. Transition counts that decide each
case are quoted beside it.
"""

from __future__ import annotations

import random
from itertools import pairwise

import numpy as np
import pytest

from descape import asset_source, cliff_catalog, cliff_chain, cliff_connectivity, iso_geometry, unit_sprites

# n_cliff_default_x1's own pieces, suffix 1..9 (01 is 3x3, 02 1x3, 03 3x1,
# 04/05 2x3, 06/07 3x2, 08/09 2x2).
DEFAULT = {suffix: 263 + suffix for suffix in range(1, 10)}
DEFAULT_01 = DEFAULT[1]
DEFAULT_02 = DEFAULT[2]
DEFAULT_08 = DEFAULT[8]
N, S, W, E = (0, -1), (0, 1), (-1, 0), (1, 0)


def _stroke(*, const=DEFAULT_01, frame=3, existing=(), width=120, height=120):
    return cliff_chain.ChainStroke(
        map_width=width,
        map_height=height,
        existing=list(existing),
        fallback_const=const,
        fallback_frame=frame,
    )


def _drag(stroke, tiles):
    for tile_x, tile_y in tiles:
        stroke.add_cursor_tile(tile_x, tile_y)
    return stroke.nodes()


def _path(press, steps, **kwargs):
    stroke = _stroke(**kwargs)
    stroke.set_path(press, steps)
    return stroke.nodes()


def _rows(nodes):
    return [(n.unit_const, n.rotation, n.x, n.y, n.bounds, n.resolved) for n in nodes]


def test_piece_suffix_reads_the_trailing_digit_of_every_cliff_label():
    """The pooling key cliff_connectivity's table is built on. Every cliff
    const must yield one, or the chain silently can't inherit a size from a
    neighbour of that piece."""
    for pieces in cliff_catalog.families().values():
        for piece in pieces:
            assert cliff_catalog.piece_suffix(piece.unit_const) in range(1, 10), piece


def test_piece_for_suffix_stays_in_the_selected_sub_group():
    """"Cliff Sand" holds two sub-groups sharing one graphic (CLF01..CLF08
    and the full Desert family), so several suffixes appear twice in it --
    the nearest-const tiebreak is what keeps a chain inside the one the user
    picked from."""
    assert cliff_catalog.piece_for_suffix(1339, 2) == 1340  # CLF01 -> CLF02
    assert cliff_catalog.piece_for_suffix(1849, 2) == 1850  # Desert 01 -> Desert 02
    # CLF has no 05 piece at all; falling through to Desert's is the only
    # answer that isn't None.
    assert cliff_catalog.piece_for_suffix(1339, 5) == 1853
    assert cliff_catalog.piece_for_suffix(DEFAULT_01, 8) == DEFAULT_08


def test_piece_for_suffix_agrees_with_piece_suffix_and_parses_no_config(monkeypatch):
    """It reads families()' own labels rather than calling piece_suffix() per
    piece, which parsed config.yaml each time (~25 ms per Cliff stroke start)."""
    from descape import asset_source

    families = cliff_catalog.families()
    calls = []
    real_get_language = asset_source.get_language

    def counting_get_language() -> str:
        calls.append(1)
        return real_get_language()

    suffix_of = {p.unit_const: cliff_catalog.piece_suffix(p.unit_const) for ps in families.values() for p in ps}
    expected = {}
    for pieces in families.values():
        for piece in pieces:
            for suffix in range(1, 10):
                same = [p.unit_const for p in pieces if suffix_of[p.unit_const] == suffix]
                expected[piece.unit_const, suffix] = (
                    min(same, key=lambda c: (abs(c - piece.unit_const), c)) if same else None
                )
    monkeypatch.setattr(asset_source, "get_language", counting_get_language)
    for (const, suffix), want in expected.items():
        assert cliff_catalog.piece_for_suffix(const, suffix) == want, (const, suffix)
    assert calls == []


def test_a_drag_against_existing_cliffs_parses_no_config(monkeypatch):
    """Every cursor event re-resolves the run with its contacts, and each
    existing-cliff contact needs that cliff's suffix: piece_suffix() parsed
    config.yaml per call (~10 ms each, ~300 ms an event beside 90 contacts)."""
    from descape import asset_source

    cliff_catalog.suffixes()
    assert cliff_catalog.suffixes() == {c: cliff_catalog.piece_suffix(c) for c in cliff_catalog.suffixes()}
    calls = []
    real_get_language = asset_source.get_language

    def counting_get_language() -> str:
        calls.append(1)
        return real_get_language()

    monkeypatch.setattr(asset_source, "get_language", counting_get_language)
    existing = [cliff_chain.ExistingCliff(unit_const=DEFAULT_02, bounds=(10, 11, 7, 10))]
    nodes = _drag(_stroke(existing=existing), [(10, y) for y in range(10, 22)])
    assert nodes[0].dirset == frozenset({N, S})
    assert calls == []


def test_a_lone_click_places_the_picked_piece_verbatim():
    """Stage 1's own gesture, unchanged by the widening to a drag stroke: a
    node with no neighbours has no dirset to resolve, so it takes the
    Piece/Frame picker's values rather than the table's empty-dirset row
    (43% suffix purity, by far the table's weakest)."""
    nodes = _drag(_stroke(const=DEFAULT_01, frame=3), [(10, 10)])
    assert len(nodes) == 1
    assert (nodes[0].unit_const, nodes[0].rotation) == (DEFAULT_01, 3)
    assert (nodes[0].x, nodes[0].y) == (11.5, 11.5)  # span_anchor(10, 10, 3, 3)
    assert not nodes[0].resolved


def test_a_straight_drag_steps_one_footprint_at_a_time():
    """A step is taken when the cursor leaves the head's realized footprint,
    and each node sits where the measured offset puts it: a straight south
    drag resolves to 3x3 pieces (below), three tiles apart, sharing edges."""
    nodes = _drag(_stroke(), [(10, y) for y in range(10, 25)])
    assert [n.bounds for n in nodes] == [
        (10, 13, 10, 13), (10, 13, 13, 16), (10, 13, 16, 19), (10, 13, 19, 22), (10, 13, 22, 25)
    ]
    for earlier, later in pairwise(nodes):
        assert cliff_connectivity.adjacency_dir(earlier.bounds, later.bounds) == (0, 1)


def test_a_fast_drag_that_skips_tiles_still_emits_every_node():
    """MapView only calls on_edit_stroke_tile once per distinct cursor tile,
    and a fast drag skips whole tiles between mouse-move events -- the walk
    is a loop for exactly this reason, so a gap-free run doesn't depend on
    how fast the user moved."""
    slow = _drag(_stroke(), [(10, y) for y in range(10, 25)])
    fast = _drag(_stroke(), [(10, 10), (10, 24)])
    assert [n.bounds for n in fast] == [n.bounds for n in slow]


def test_a_run_interior_resolves_through_the_measured_table():
    """A straight south drag, four nodes. Every pair keeps its size (S s1 ->
    NS s1 127 and s2 -> s2 90; NS -> NS s1 484, s2 540; NS -> N s1 155, s2 48),
    so the run is all s1 or all s2, and the edge joints favour s1 (score 0.02
    against -1.77): a plain straight is 3x3. Rotations follow the measured
    rotation pairs: 18 is the {S} s1 cap's commonest, NS s1 after 18 is 2 (64
    vs 58), after 2 is 1 (118 vs 111), and N s1 after 1 is 19 (78/80)."""
    nodes = _drag(_stroke(), [(10, y) for y in range(10, 22)])
    assert _rows(nodes) == [
        (DEFAULT_01, 18, 11.5, 11.5, (10, 13, 10, 13), True),
        (DEFAULT_01, 2, 11.5, 14.5, (10, 13, 13, 16), True),
        (DEFAULT_01, 1, 11.5, 17.5, (10, 13, 16, 19), True),
        (DEFAULT_01, 19, 11.5, 20.5, (10, 13, 19, 22), True),
    ]


def test_a_straight_east_drag_is_3x3_by_the_same_rule():
    """WE pairs keep their size too (E -> WE s1 131, s3 61; WE -> WE s1 385,
    s3 585; WE -> W s1 117, s3 102), and s1 wins the run (0.04 against -1.50)."""
    nodes = _path((10, 10), [E, E, E])
    assert _rows(nodes) == [
        (DEFAULT_01, 16, 11.5, 11.5, (10, 13, 10, 13), True),
        (DEFAULT_01, 5, 14.5, 11.5, (13, 16, 10, 13), True),
        (DEFAULT_01, 4, 17.5, 11.5, (16, 19, 10, 13), True),
        (DEFAULT_01, 17, 20.5, 11.5, (19, 22, 10, 13), True),
    ]


def test_face_inherits_along_a_run_but_a_corner_takes_its_own():
    """An L, south then east: the corner sizes both legs. NS s1 -> {N,E} is
    never seen (one s6), while NS s2 -> {N,E} s8 is 223/225, and {N,E} s8 ->
    WE is s3 182/182, so the vertical leg is 1x3 s2, the corner the 2x2 s8 and
    the horizontal leg 3x1 s3 (the plan's Context table). Layout reads the
    measured offsets: the s3 leg leaves the corner on its lower row."""
    nodes = _path((10, 10), [S, S, E, E])
    assert _rows(nodes) == [
        (DEFAULT[2], 22, 10.5, 11.5, (10, 11, 10, 13), True),
        (DEFAULT[2], 8, 10.5, 14.5, (10, 11, 13, 16), True),
        (DEFAULT[8], 13, 11.0, 17.0, (10, 12, 16, 18), True),
        (DEFAULT[3], 10, 13.5, 17.5, (12, 15, 17, 18), True),
        (DEFAULT[3], 21, 16.5, 17.5, (15, 18, 17, 18), True),
    ]


_L_TURNS = {
    # NS s2 -> {N,W} s4 215 / s5 216 against s1 -> s6 277, but {N,W} s4/s5 ->
    # WE is s1 (146, 154): thin vertical leg, 2x3 corner, 3x3 horizontal leg.
    "south_west": (
        [S, S, W, W],
        [
            (DEFAULT[2], 22, 40.5, 41.5, (40, 41, 40, 43), True),
            (DEFAULT[2], 8, 40.5, 44.5, (40, 41, 43, 46), True),
            (DEFAULT[4], 6, 40.0, 47.5, (39, 41, 46, 49), True),
            (DEFAULT[1], 4, 37.5, 47.5, (36, 39, 46, 49), True),
            (DEFAULT[1], 16, 34.5, 47.5, (33, 36, 46, 49), True),
        ],
    ),
    # NS s1 -> {S,E} s6 180 / s7 194, both -> WE s3 (252, 253): the mirror
    # image, 3x3 vertical leg, 3x2 corner, thin horizontal leg.
    "north_east": (
        [N, N, E, E],
        [
            (DEFAULT[1], 19, 41.5, 41.5, (40, 43, 40, 43), True),
            (DEFAULT[1], 1, 41.5, 38.5, (40, 43, 37, 40), True),
            (DEFAULT[7], 0, 41.5, 36.0, (40, 43, 35, 37), True),
            (DEFAULT[3], 10, 44.5, 35.5, (43, 46, 35, 36), True),
            (DEFAULT[3], 21, 47.5, 35.5, (46, 49, 35, 36), True),
        ],
    ),
    # NS s1 -> {S,W} has no records at all; NS s2 -> {S,W} s9 195 / s8 127,
    # both -> WE s3 (196, 125): thin legs either side of a 2x2.
    "north_west": (
        [N, N, W, W],
        [
            (DEFAULT[2], 23, 40.5, 41.5, (40, 41, 40, 43), True),
            (DEFAULT[2], 8, 40.5, 38.5, (40, 41, 37, 40), True),
            (DEFAULT[9], 9, 40.0, 36.0, (39, 41, 35, 37), True),
            (DEFAULT[3], 10, 37.5, 35.5, (36, 39, 35, 36), True),
            (DEFAULT[3], 20, 34.5, 35.5, (33, 36, 35, 36), True),
        ],
    ),
}


@pytest.mark.parametrize("name", sorted(_L_TURNS))
def test_each_l_turn_sizes_its_legs_from_its_corner(name):
    """The other three L turns (south then east is the test above): which
    leg is thin depends on the corner the pairs allow."""
    steps, rows = _L_TURNS[name]
    assert _rows(_path((40, 40), steps)) == rows


def test_a_zig_zag_changes_leg_size_between_turns():
    """South, east, north: an {N,E} turn (2x2 s8, thin legs both sides) then a
    {W,N} turn, where WE s3 -> {W,N} is s6 320/321 and {W,N} s6 -> NS is s1
    277/277. So the middle leg stays a thin 3x1 and the last leg is 3x3: one
    stroke, three leg sizes."""
    nodes = _path((40, 40), [S, S, E, E, N, N])
    assert _rows(nodes) == [
        (DEFAULT[2], 22, 40.5, 41.5, (40, 41, 40, 43), True),
        (DEFAULT[2], 8, 40.5, 44.5, (40, 41, 43, 46), True),
        (DEFAULT[8], 13, 41.0, 47.0, (40, 42, 46, 48), True),
        (DEFAULT[3], 10, 43.5, 47.5, (42, 45, 47, 48), True),
        (DEFAULT[6], 14, 46.5, 47.0, (45, 48, 46, 48), True),
        (DEFAULT[1], 2, 46.5, 44.5, (45, 48, 43, 46), True),
        (DEFAULT[1], 18, 46.5, 41.5, (45, 48, 40, 43), True),
    ]


def test_a_ten_step_staircase_drag_lays_2x2_steps_and_ends_at_the_cursor():
    """A fast diagonal drag from (10, 10) to (26, 26). The walk takes ten
    cardinal steps, larger overshoot first; the stair's corners are 2x2 s8
    (the {W,S} and {N,E} rows' joint favourite) and its caps thin. The
    oracle replays the walk rule over its own layouts to get the steps."""
    stroke = _stroke()
    nodes = _drag(stroke, [(10, 10), (26, 26)])
    assert stroke.steps == [E, S, S, E, S, E, S, E, S, E]
    assert _rows(nodes) == [
        (DEFAULT[3], 20, 11.5, 10.5, (10, 13, 10, 11), True),
        (DEFAULT[9], 9, 14.0, 11.0, (13, 15, 10, 12), True),
        (DEFAULT[2], 7, 14.5, 13.5, (14, 15, 12, 15), True),
        (DEFAULT[8], 13, 15.0, 16.0, (14, 16, 15, 17), True),
        (DEFAULT[8], 9, 17.0, 17.0, (16, 18, 16, 18), True),
        (DEFAULT[8], 13, 18.0, 19.0, (17, 19, 18, 20), True),
        (DEFAULT[8], 9, 20.0, 20.0, (19, 21, 19, 21), True),
        (DEFAULT[8], 13, 21.0, 22.0, (20, 22, 21, 23), True),
        (DEFAULT[8], 9, 23.0, 23.0, (22, 24, 22, 24), True),
        (DEFAULT[8], 13, 24.0, 25.0, (23, 25, 24, 26), True),
        (DEFAULT[3], 21, 26.5, 25.5, (25, 28, 25, 26), True),
    ]
    assert _distance((26, 26), nodes[-1].bounds) <= 2


def test_the_chain_disagrees_with_each_node_s_own_majority():
    """Per node, the commonest suffix for its shape draws the south-east L
    as s1 cap, s2, s8, s3, s1 cap ({S} s1 347 vs s2 259, {W} s1 327 vs s3
    321). The pairs never put an s1 cap on a thin leg (S s1 -> NS is s1
    127/127, WE s3 -> W is s3 102/109), so the chain makes both caps thin."""
    steps = [S, S, E, E]
    tables = cliff_connectivity.chain_tables()
    majority = [
        max(tables.candidates(d, range(1, 10)), key=lambda s, d=d: (tables.node_n[(d, s)], -s))
        for d in cliff_chain.path_dirsets(steps)
    ]
    assert majority == [1, 2, 8, 3, 1]
    by_const = {const: suffix for suffix, const in DEFAULT.items()}
    assert [by_const[n.unit_const] for n in _path((40, 40), steps)] == [2, 2, 8, 3, 3]


def test_an_unpinned_junction_falls_back_to_the_picked_frame():
    """A straight run squeezed between two existing cliffs: the middle node's
    contacts make its shape {N,S,W,E}, which the table has no row for, so it
    keeps the picked piece and frame. It splits the run, and both caps
    resolve alone as s1 (their priors) but stay unresolved: the pair to the
    middle node is not a measured one."""
    existing = [
        cliff_chain.ExistingCliff(unit_const=DEFAULT_01, bounds=(7, 10, 13, 16)),
        cliff_chain.ExistingCliff(unit_const=DEFAULT_01, bounds=(13, 16, 13, 16)),
    ]
    assert cliff_connectivity.resolve(frozenset({N, S, W, E})) is None
    nodes = _path((10, 10), [S, S], frame=7, existing=existing)
    assert _rows(nodes) == [
        (DEFAULT_01, 18, 11.5, 11.5, (10, 13, 10, 13), False),
        (DEFAULT_01, 7, 11.5, 14.5, (10, 13, 13, 16), False),
        (DEFAULT_01, 19, 11.5, 17.5, (10, 13, 16, 19), False),
    ]
    assert nodes[1].dirset == frozenset({N, S, W, E})


def test_a_drag_starting_against_an_existing_s2_cliff_joins_it():
    """The first layout puts a 3x3 cap under the existing 1x3, so node 0 takes
    that contact: its shape becomes {N,S}, and the contact term (an s2 {S}
    cap's southern NS neighbour is s2 90/90) moves the whole run to s2. The
    rerun's 1x3 node 0 still touches the existing piece, so all resolve."""
    existing = [cliff_chain.ExistingCliff(unit_const=DEFAULT_02, bounds=(10, 11, 7, 10))]
    nodes = _path((10, 10), [S, S, S], existing=existing)
    assert _rows(nodes) == [
        (DEFAULT_02, 8, 10.5, 11.5, (10, 11, 10, 13), True),
        (DEFAULT_02, 8, 10.5, 14.5, (10, 11, 13, 16), True),
        (DEFAULT_02, 8, 10.5, 17.5, (10, 11, 16, 19), True),
        (DEFAULT_02, 23, 10.5, 20.5, (10, 11, 19, 22), True),
    ]
    assert nodes[0].dirset == frozenset({N, S})


def test_a_node_overlapping_an_existing_cliff_is_skipped_but_the_run_continues():
    """A cliff already on the map blocks its own tiles without truncating the
    drag: the node on it is dropped and the path keeps walking past it."""
    existing = [cliff_chain.ExistingCliff(unit_const=DEFAULT_01, bounds=(10, 13, 13, 16))]
    nodes = _drag(_stroke(existing=existing), [(10, y) for y in range(10, 25)])
    assert (10, 13, 13, 16) not in [n.bounds for n in nodes]
    assert (10, 13, 16, 19) in [n.bounds for n in nodes]


def test_a_run_stops_at_the_map_edge_rather_than_hanging_off_it():
    """The whole footprint has to fit: unit_tile_bounds() clamps a
    half-off-map cliff into a box that no longer matches the lattice every
    adjacency lookup here assumes."""
    nodes = _drag(_stroke(width=20, height=20), [(10, y) for y in range(10, 30)])
    assert nodes
    for node in nodes:
        assert node.bounds[1] <= 20 and node.bounds[3] <= 20


def test_a_path_truncates_at_its_first_node_off_the_map():
    """The layout, not just the walk, cuts the run: a path laid past the edge
    keeps only the nodes before the first one that does not fit, even if a
    later one would land back on the map."""
    nodes = _path((10, 10), [S, S, S, S, W, W, N], width=20, height=20)
    assert [n.bounds for n in nodes] == [(10, 13, 10, 13), (10, 13, 13, 16), (10, 13, 16, 19)]


@pytest.mark.parametrize("const", [DEFAULT_01, DEFAULT_02, DEFAULT_08])
def test_the_anchor_round_trips_through_render_s_own_span_start(const):
    """span_anchor is _span_start's inverse (test_cliffs.py pins that
    directly); this asserts the chain actually places on it, so a node's
    tile bounds are the ones the renderer will report for it."""
    from descape.render import unit_tile_bounds

    node = _drag(_stroke(const=const), [(10, 10)])[0]
    placed = type("U", (), {"x": node.x, "y": node.y, "unit_const": const})()
    assert unit_tile_bounds(placed, 120, 120) == node.bounds


def test_a_one_tile_wobble_beside_a_thin_leg_does_not_turn_a_corner():
    """The step test widens the head to 3 tiles per axis: the 1x3 leg above
    sits on x=10 only, and the cursor wobbling onto x=11 stays a straight
    run (the same four nodes as the s2 contact drag above)."""
    existing = [cliff_chain.ExistingCliff(unit_const=DEFAULT_02, bounds=(10, 11, 7, 10))]
    stroke = _stroke(existing=existing)
    nodes = _drag(stroke, [(10 + y % 2, y) for y in range(10, 22)])
    assert stroke.steps == [S, S, S]
    assert [(n.unit_const, n.bounds) for n in nodes] == [
        (DEFAULT_02, (10, 11, 10, 13)), (DEFAULT_02, (10, 11, 13, 16)),
        (DEFAULT_02, (10, 11, 16, 19)), (DEFAULT_02, (10, 11, 19, 22)),
    ]


def test_dragging_back_over_the_run_does_not_step_onto_it():
    stroke = _stroke()
    _drag(stroke, [(10, y) for y in range(10, 19)] + [(10, y) for y in range(18, 9, -1)])
    assert stroke.steps == [S, S]
    assert len(stroke.nodes()) == 3


def _staircase(start, end):
    """A 4-connected cursor path, x first, the shape MapView's interpolation sends."""
    (x, y), tiles = start, [start]
    while (x, y) != end:
        if x != end[0] and (abs(end[0] - x) >= abs(end[1] - y)):
            x += 1 if end[0] > x else -1
        else:
            y += 1 if end[1] > y else -1
        tiles.append((x, y))
    return tiles


def _distance(tile, bounds):
    x0, x1, y0, y1 = bounds
    return max(x0 - tile[0], tile[0] - (x1 - 1), y0 - tile[1], tile[1] - (y1 - 1), 0)


def test_a_diagonal_drag_lays_a_staircase_that_ends_at_the_cursor():
    """Steps are cardinal only, so a 45-degree drag walks both axes in turn,
    and the run's last node stays within 2 tiles of the cursor."""
    stroke = _stroke()
    nodes = _drag(stroke, _staircase((10, 10), (40, 40)))
    assert set(stroke.steps) == {E, S}
    assert len(stroke.steps) >= 10
    assert _distance((40, 40), nodes[-1].bounds) <= 2


def test_a_fast_diagonal_drag_lays_the_same_staircase_as_a_slow_one():
    slow = _stroke()
    _drag(slow, _staircase((10, 10), (40, 40)))
    fast = _stroke()
    _drag(fast, [(10, 10), (40, 40)])
    assert fast.steps == slow.steps
    assert _rows(fast.nodes()) == _rows(slow.nodes())


def _realized(node, nodes, existing):
    dirs = {cliff_connectivity.adjacency_dir(node.bounds, other.bounds) for other in nodes if other is not node}
    dirs |= {cliff_connectivity.adjacency_dir(node.bounds, c.bounds) for c in existing}
    dirs.discard(None)
    return dirs


def _random_drag(rng, stroke):
    x = y = 60
    stroke.add_cursor_tile(x, y)
    for _ in range(rng.randint(3, 60)):
        dx, dy = rng.choice((N, S, W, E))
        x, y = min(max(x + dx, 0), 119), min(max(y + dy, 0), 119)
        stroke.add_cursor_tile(x, y)
    return stroke.nodes()


def test_random_drags_keep_every_layout_invariant():
    """200 seeded random-walk drags: consecutive nodes share the edge their
    step names, nothing overlaps, everything is on the map, every resolved
    node's realized neighbours are the shape it was resolved for, and the
    same drag twice gives the same run."""
    rng = random.Random(20261009)
    resolved = total = 0
    for _ in range(200):
        seed = rng.random()
        stroke = _stroke()
        nodes = _random_drag(random.Random(seed), stroke)
        steps = stroke.steps
        path = cliff_chain.path_dirsets(steps)
        by_index = {n.index: n for n in nodes}
        for i, node in by_index.items():
            if i + 1 in by_index:
                assert cliff_connectivity.adjacency_dir(node.bounds, by_index[i + 1].bounds) == steps[i]
            assert node.dirset >= path[i]
            if node.resolved:
                assert _realized(node, nodes, []) == node.dirset
        for a, b in pairwise(sorted(n.index for n in nodes)):
            assert a < b
        for i, a in enumerate(nodes):
            x0, x1, y0, y1 = a.bounds
            assert min(x0, y0) >= 0 and max(x1, y1) <= 120
            for b in nodes[:i]:
                assert not cliff_chain._overlap(a.bounds, b.bounds)
        again = _random_drag(random.Random(seed), _stroke())
        assert _rows(again) == _rows(nodes)
        resolved += sum(n.resolved for n in nodes)
        total += len(nodes)
    assert resolved > total // 4


def _shape(nodes):
    x0 = min(n.bounds[0] for n in nodes)
    y0 = min(n.bounds[2] for n in nodes)
    return sorted((b[0] - x0, b[1] - x0, b[2] - y0, b[3] - y0) for b in (n.bounds for n in nodes))


def _from_the_other_end(steps):
    return [(-dx, -dy) for dx, dy in reversed(steps)]


_TURNS = [(a, b) for a in (N, S, W, E) for b in (N, S, W, E) if a[0] != b[0] and a[1] != b[1]]


@pytest.mark.parametrize(("first", "second"), _TURNS)
def test_an_l_turn_lays_the_same_bounds_drawn_from_either_end(first, second):
    steps = [first, first, second, second]
    assert _shape(_path((40, 40), steps)) == _shape(_path((40, 40), _from_the_other_end(steps)))


def test_random_drags_lay_the_same_bounds_drawn_from_either_end():
    """The fuzz drags above, each laid again from its far end. Paths come from
    the live walk, which never steps onto an earlier node: a set_path() path
    may cross itself, and which of two overlapping nodes is dropped then
    depends on drawing order by design."""
    rng = random.Random(20261009)
    for _ in range(200):
        stroke = _stroke()
        nodes = _random_drag(random.Random(rng.random()), stroke)
        back = _path((200, 200), _from_the_other_end(stroke.steps), width=400, height=400)
        assert _shape(back) == _shape(nodes), stroke.steps


# -- ground truth: the drawn run's ink (needs a real AoE2:DE install) ----

# The south-then-west drag below, as the one-suffix-per-stroke resolver this
# walk replaced placed it with Cliff (Default) 01 picked (recorded at eebce52):
# eight 3x3 pieces, (const, rotation, x, y).
_ONE_SUFFIX_L = [
    (DEFAULT_01, 18, 41.5, 41.5), (DEFAULT_01, 1, 41.5, 44.5), (DEFAULT_01, 0, 41.5, 47.5),
    (DEFAULT_01, 17, 41.5, 50.5), (DEFAULT_01, 17, 38.5, 50.5), (DEFAULT_01, 4, 35.5, 50.5),
    (DEFAULT_01, 4, 32.5, 50.5), (DEFAULT_01, 16, 29.5, 50.5),
]
_L_DRAG = [(40, y) for y in range(40, 52)] + [(x, 51) for x in range(39, 27, -1)]
_INK_TILE_PX = 64


def _require_install():
    if asset_source.get_install_path() is None:
        pytest.skip("no AoE2:DE install visible -- set AOE2DE_INSTALL_PATH to one")


def _ink(placed) -> tuple[np.ndarray, int]:
    """The run's sprite pieces pasted where render puts them on flat ground,
    as an alpha > 0 mask, plus one ground tile's diamond area in pixels."""
    proj = iso_geometry.canvas_size_and_origin(120, 120, _INK_TILE_PX, 0, 0)
    pastes = []
    for const, rotation, x, y in placed:
        ax, ay = iso_geometry.map_point_to_screen(x, y, 0, proj)
        pieces = unit_sprites.sprite_pieces_for(const, float(rotation), 0, proj.half_w)
        assert pieces, (const, rotation)
        pastes.extend(
            (ax + p.dx - p.draw.hotspot_x, ay + p.dy - p.draw.hotspot_y, p.draw.rgba[..., 3] > 0) for p in pieces
        )
    x0 = min(px for px, _py, _a in pastes)
    y0 = min(py for _px, py, _a in pastes)
    x1 = max(px + a.shape[1] for px, _py, a in pastes)
    y1 = max(py + a.shape[0] for _px, py, a in pastes)
    mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    for px, py, a in pastes:
        mask[py - y0 : py - y0 + a.shape[0], px - x0 : px - x0 + a.shape[1]] |= a
    return mask, 2 * proj.half_w * proj.half_h


def _walls(mask: np.ndarray, tile_area: int) -> list[int]:
    """Sizes of the 4-connected ink components at least one ground tile's
    diamond in area: smaller ones are the art's own loose debris, which every
    piece scatters around its base (at most ~150 px at this zoom)."""
    seen = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    sizes = []
    for y0, x0 in zip(*np.nonzero(mask), strict=True):
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        stack, size = [(int(y0), int(x0))], 0
        while stack:
            y, x = stack.pop()
            size += 1
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < height and 0 <= xx < width and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        if size >= tile_area:
            sizes.append(size)
    return sizes


def _placed(nodes):
    return [(n.unit_const, n.rotation, n.x, n.y) for n in nodes]


@pytest.mark.parametrize(
    "steps",
    [[S, S, E, E], [S, S, W, W], [N, N, E, E], [N, N, W, W], [S, S, E, E, N, N], [S, S, E, E, S, S]],
    ids=["l_south_east", "l_south_west", "l_north_east", "l_north_west", "u_turn", "zig_zag"],
)
def test_a_resolved_run_draws_one_connected_wall(steps):
    _require_install()
    assert len(_walls(*_ink(_placed(_path((40, 40), steps))))) == 1


def test_the_l_drag_draws_one_wall_where_the_one_suffix_resolver_drew_two():
    """The connectivity check's negative control: the same drag, south then
    west, splits into two walls as the replaced resolver drew it."""
    _require_install()
    assert len(_walls(*_ink(_placed(_drag(_stroke(), _L_DRAG))))) == 1
    assert len(_walls(*_ink(_ONE_SUFFIX_L))) == 2
