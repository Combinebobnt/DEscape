"""The Cliff tool's auto-connecting chain drag: its path, piece resolution and
layout, exercised headless (descape.cliff_chain has no Qt dependency; the
ViewerWindow wiring is tested separately in tests/test_cliff_tool_viewer.py).

Expected pieces below are typed in from the measured shape table of the
facing model (GH #44 round 2), never read back off the resolver: a piece's
size is which way its face points, and the drag direction picks it. An edge
drawn east or south is tall (T), one drawn west or north is thin (t), and a
node's piece is the table's for its typed arms. Where the table gives a
straight two frames they are art twins, the lower one when the footprint's
low corner has `((x0 + y0) // 3) % 2 == 0`, else the higher.
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
SUFFIX_OF = {const: suffix for suffix, const in DEFAULT.items()}
N, S, W, E = (0, -1), (0, 1), (-1, 0), (1, 0)
T, t = True, False

# The measured shape table: typed arms -> (suffix, frames).
SHAPE_TABLE = [
    ({N: T, S: T}, 1, (1, 2)),
    ({N: t, S: t}, 2, (7, 8)),
    ({E: T, W: T}, 1, (4, 5)),
    ({E: t, W: t}, 3, (10, 11)),
    ({E: T, N: T}, 1, (3,)),
    ({E: t, N: t}, 8, (13,)),
    ({S: T, W: T}, 1, (15,)),
    ({S: t, W: t}, 9, (9,)),
    ({N: T, W: t}, 6, (14,)),
    ({N: t, W: T}, 5, (6,)),
    ({E: T, S: t}, 4, (12,)),
    ({E: t, S: T}, 6, (0,)),
    ({E: T}, 1, (16,)),
    ({W: T}, 1, (17,)),
    ({S: T}, 1, (18,)),
    ({N: T}, 1, (19,)),
    ({E: t}, 3, (20,)),
    ({W: t}, 3, (21,)),
    ({S: t}, 2, (22,)),
    ({N: t}, 2, (23,)),
]


def _stroke(*, const=DEFAULT_01, frame=3, existing=(), width=120, height=120, side=None):
    return cliff_chain.ChainStroke(
        map_width=width,
        map_height=height,
        existing=list(existing),
        fallback_const=const,
        fallback_frame=frame,
        side=side,
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


def _suffixes(nodes):
    return [SUFFIX_OF[n.unit_const] for n in nodes]


def _twin(frames, bounds):
    """The table's frame for a footprint at `bounds`: the position rule above."""
    return frames[0] if ((bounds[0] + bounds[2]) // 3) % 2 == 0 else frames[-1]


def _table_piece(arms):
    return next((suffix, frames) for typed, suffix, frames in SHAPE_TABLE if typed == arms)


def _typed_arms(steps, side=1):
    """Each path node's arms, typed by the edge: east and south tall on side 1."""
    out = [{} for _ in range(len(steps) + 1)]
    for i, step in enumerate(steps):
        tall = (step in (E, S)) == (side == 1)
        out[i][step] = tall
        out[i + 1][(-step[0], -step[1])] = tall
    return out


def _assert_on_the_table(nodes, steps, side=1):
    """Every node is the table's piece for its typed path arms, at the frame
    its position gives, and resolved."""
    assert [n.index for n in nodes] == list(range(len(steps) + 1))
    for node, arms in zip(nodes, _typed_arms(steps, side), strict=True):
        suffix, frames = _table_piece(arms)
        assert (SUFFIX_OF[node.unit_const], node.rotation) == (suffix, _twin(frames, node.bounds)), (node, arms)
        assert node.arms == frozenset(arms)
        assert node.resolved, node
    for (earlier, later), step in zip(pairwise(nodes), steps, strict=True):
        assert cliff_connectivity.adjacency_dir(earlier.bounds, later.bounds) == step


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
    stroke = _stroke(existing=existing)
    nodes = _drag(stroke, [(10, y) for y in range(10, 22)])
    assert nodes[0].arms == frozenset({N, S})
    assert stroke.rewrites()
    assert calls == []


# -- the press tile is a centre ---------------------------------------------


def test_a_lone_click_places_the_picked_piece_verbatim():
    """A node with no neighbours has no arm to resolve, so it takes the
    Piece/Frame picker's values, centred on the clicked tile."""
    nodes = _drag(_stroke(const=DEFAULT_01, frame=3), [(10, 10)])
    assert len(nodes) == 1
    assert (nodes[0].unit_const, nodes[0].rotation) == (DEFAULT_01, 3)
    assert nodes[0].bounds == (9, 12, 9, 12)
    assert (nodes[0].x, nodes[0].y) == (10.5, 10.5)  # span_anchor(9, 9, 3, 3)
    assert not nodes[0].resolved


@pytest.mark.parametrize(
    ("const", "bounds"),
    [(DEFAULT_01, (29, 32, 39, 42)), (DEFAULT_02, (30, 31, 39, 42)), (DEFAULT_08, (29, 31, 39, 41))],
    ids=["3x3", "1x3", "2x2"],
)
def test_the_first_piece_is_centred_on_the_press_tile(const, bounds):
    """The low corner is the press tile less half the span per axis, so an
    even span keeps the press tile as its high one."""
    assert [n.bounds for n in _drag(_stroke(const=const), [(30, 40)])] == [bounds]


def test_the_first_piece_stays_under_the_press_when_its_span_changes():
    """Drawn east the first node is a 3x3 cap, drawn west a 3x1, drawn north
    a 1x3: each is centred on the same press tile."""
    assert _path((30, 40), [E, E])[0].bounds == (29, 32, 39, 42)
    assert _path((30, 40), [W, W])[0].bounds == (29, 32, 40, 41)
    assert _path((30, 40), [N, N])[0].bounds == (30, 31, 39, 42)


@pytest.mark.parametrize(
    ("press", "const", "bounds"),
    [
        ((119, 40), DEFAULT_01, (117, 120, 39, 42)),
        ((0, 0), DEFAULT_01, (0, 3, 0, 3)),
        ((40, 119), DEFAULT_02, (40, 41, 117, 120)),
        ((0, 119), DEFAULT_08, (0, 2, 118, 120)),
    ],
)
def test_a_press_at_the_map_edge_clamps_the_first_piece_onto_the_map(press, const, bounds):
    assert [n.bounds for n in _drag(_stroke(const=const), [press])] == [bounds]


def test_a_press_off_the_map_starts_nothing():
    stroke = _stroke()
    assert _drag(stroke, [(120, 40), (-1, 3)]) == []
    assert _drag(stroke, [(50, 50)])[0].bounds == (49, 52, 49, 52)


# -- the drag direction picks the facing ------------------------------------


@pytest.mark.parametrize(
    ("step", "suffix", "caps", "twins"),
    [(E, 1, (16, 17), (4, 5)), (S, 1, (18, 19), (1, 2)), (W, 3, (21, 20), (10, 11)), (N, 2, (23, 22), (7, 8))],
    ids=["east", "south", "west", "north"],
)
def test_a_straight_is_front_facing_drawn_east_or_south_and_back_facing_the_other_way(step, suffix, caps, twins):
    """East and south are all 3x3 s1, the rock face; west is all 3x1 s3 and
    north all 1x3 s2, the low ridge. The ends are that size's own caps."""
    nodes = _path((40, 40), [step] * 4)
    assert _suffixes(nodes) == [suffix] * 5
    assert [nodes[0].rotation, nodes[-1].rotation] == list(caps)
    assert [n.rotation for n in nodes[1:-1]] == [_twin(twins, n.bounds) for n in nodes[1:-1]]
    assert all(n.resolved for n in nodes)
    _assert_on_the_table(nodes, [step] * 4)


def test_a_straight_drag_steps_one_footprint_at_a_time():
    """A step is taken when the cursor leaves the head's realized footprint,
    and each node sits where the measured offset puts it: a straight south
    drag is 3x3 pieces three tiles apart, sharing edges, from the centred one."""
    nodes = _drag(_stroke(), [(10, y) for y in range(10, 24)])
    assert [n.bounds for n in nodes] == [
        (9, 12, 9, 12), (9, 12, 12, 15), (9, 12, 15, 18), (9, 12, 18, 21), (9, 12, 21, 24)
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


def test_a_straight_east_run_row_by_row():
    """Four nodes drawn east from a press at (11, 11): tall caps 16 and 17
    around the s1 straight's twins, the higher (5) where the low corner's
    (x0 + y0) // 3 is odd."""
    assert _rows(_path((11, 11), [E, E, E])) == [
        (DEFAULT_01, 16, 11.5, 11.5, (10, 13, 10, 13), True),
        (DEFAULT_01, 5, 14.5, 11.5, (13, 16, 10, 13), True),
        (DEFAULT_01, 4, 17.5, 11.5, (16, 19, 10, 13), True),
        (DEFAULT_01, 17, 20.5, 11.5, (19, 22, 10, 13), True),
    ]


# The eight directed L turns: the suffix of each of the five nodes (cap, leg,
# corner, leg, cap), and the frames of the start cap, corner and end cap. Both
# legs tall gives the all-3x3 corners (3 and 15); a mixed turn takes the 3x2
# or 2x3 piece whose own legs are tall on one arm and thin on the other.
_L_TURNS = {
    "south_east": ((S, E), [1, 1, 1, 1, 1], (18, 3, 17)),
    "east_south": ((E, S), [1, 1, 1, 1, 1], (16, 15, 19)),
    "north_west": ((N, W), [2, 2, 9, 3, 3], (23, 9, 20)),
    "west_north": ((W, N), [3, 3, 8, 2, 2], (21, 13, 22)),
    "south_west": ((S, W), [1, 1, 6, 3, 3], (18, 14, 20)),
    "west_south": ((W, S), [3, 3, 6, 1, 1], (21, 0, 19)),
    "east_north": ((E, N), [1, 1, 5, 2, 2], (16, 6, 22)),
    "north_east": ((N, E), [2, 2, 4, 1, 1], (23, 12, 17)),
}


@pytest.mark.parametrize("name", sorted(_L_TURNS))
def test_each_directed_l_turn_takes_the_table_s_corner_and_legs(name):
    (first, second), suffixes, (start, corner, end) = _L_TURNS[name]
    steps = [first, first, second, second]
    nodes = _path((40, 40), steps)
    assert _suffixes(nodes) == suffixes
    assert [nodes[0].rotation, nodes[2].rotation, nodes[4].rotation] == [start, corner, end]
    assert all(n.resolved for n in nodes)
    _assert_on_the_table(nodes, steps)


def test_the_same_l_drawn_the_other_way_is_the_other_facing():
    """South then east turns in on an all-3x3 corner; the same line drawn
    back, west then north, is thin legs either side of a 2x2."""
    assert _suffixes(_path((40, 40), [S, S, E, E])) == [1, 1, 1, 1, 1]
    assert _suffixes(_path((46, 46), [W, W, N, N])) == [3, 3, 8, 2, 2]


def test_a_zig_zag_alternates_front_and_back_legs():
    """South, east, north: tall, tall, then thin. The first corner is the
    all-3x3 {N, E}, the second the 2x3 with its west leg tall and its north
    leg thin, and the last leg is the 1x3 ridge."""
    steps = [S, S, E, E, N, N]
    nodes = _path((40, 40), steps)
    assert _suffixes(nodes) == [1, 1, 1, 1, 5, 2, 2]
    assert [n.rotation for n in (nodes[2], nodes[4], nodes[6])] == [3, 6, 22]
    _assert_on_the_table(nodes, steps)


def test_a_staircase_drag_lays_table_pieces_and_ends_at_the_cursor():
    """A fast diagonal drag from (10, 10) to (26, 26): cardinal steps, larger
    overshoot first, every node the table's piece for its typed arms."""
    stroke = _stroke()
    nodes = _drag(stroke, [(10, 10), (26, 26)])
    assert set(stroke.steps) == {E, S} and len(stroke.steps) >= 6
    _assert_on_the_table(nodes, stroke.steps)
    assert _distance((26, 26), nodes[-1].bounds) <= 2


def test_the_run_is_not_each_shape_s_own_majority_piece():
    """Per shape, the corpus's commonest piece for the south-east L is s1,
    s2, s8, s3, s1 (`resolve()`): mixed facings no map draws in one line.
    The drag keeps the face on one side instead, here all s1."""
    steps = [S, S, E, E]
    majority = [cliff_connectivity.resolve(d)[0] for d in cliff_chain.path_dirsets(steps)]
    assert majority == [1, 2, 8, 3, 1]
    assert _suffixes(_path((40, 40), steps)) == [1, 1, 1, 1, 1]


def test_twins_are_stable_while_a_drag_grows():
    """A straight's twin comes from where its footprint sits, so growing the
    run never changes a piece already laid, and neighbours alternate."""
    seen: dict[tuple, int] = {}
    for length in range(2, 9):
        nodes = _path((40, 40), [E] * length)
        for node in nodes[1:-1]:
            assert seen.setdefault(node.bounds, node.rotation) == node.rotation
        assert [n.bounds for n in nodes[:-1]] == [n.bounds for n in _path((40, 40), [E] * (length - 1))]
    assert [seen[b] for b in sorted(seen)] == [5, 4, 5, 4, 5, 4, 5]


def test_set_path_drops_the_previous_path_s_plan():
    stroke = _stroke()
    stroke.set_path((40, 40), [E, E])
    first = _rows(stroke.nodes())
    stroke.set_path((70, 70), [E, E])
    moved = stroke.nodes()
    assert [n.bounds for n in moved] == [(69, 72, 69, 72), (72, 75, 69, 72), (75, 78, 69, 72)]
    stroke.set_path((40, 40), [E, E])
    assert _rows(stroke.nodes()) == first


# -- reversal: the other way round is the other side ------------------------


def _reversed(steps):
    return [(-dx, -dy) for dx, dy in reversed(steps)]


def _centre(bounds):
    x0, x1, y0, y1 = bounds
    return x0 + (x1 - x0) // 2, y0 + (y1 - y0) // 2


def _plain(nodes, steps):
    """No node dropped and no two non-adjacent nodes sharing an edge."""
    if [n.index for n in nodes] != list(range(len(steps) + 1)):
        return False
    return not any(
        cliff_connectivity.adjacency_dir(a.bounds, b.bounds) in (N, S, W, E)
        for i, a in enumerate(nodes)
        for b in nodes[i + 2 :]
    )


def _same_run_from_the_far_end(steps, side):
    """Whether `steps` with the face on `side` equals the reversed path with
    it on the other side, pressed at the centre of the forward run's last
    node, node for node. None when the forward run is not a plain one."""
    forward = _path((200, 200), steps, width=400, height=400, side=side)
    if not _plain(forward, steps):
        return None
    back = _path(_centre(forward[-1].bounds), _reversed(steps), width=400, height=400, side=-side)
    return [(n.unit_const, n.rotation, n.bounds) for n in reversed(back)] == [
        (n.unit_const, n.rotation, n.bounds) for n in forward
    ]


_TURNS = [(a, b) for a in (N, S, W, E) for b in (N, S, W, E) if a[0] != b[0] and a[1] != b[1]]


@pytest.mark.parametrize(("first", "second"), _TURNS)
@pytest.mark.parametrize("side", [1, -1])
def test_an_l_turn_is_the_same_run_drawn_from_its_far_end_on_the_other_side(first, second, side):
    assert _same_run_from_the_far_end([first, first, second, second], side) is True


def test_random_paths_are_the_same_run_drawn_from_the_far_end_on_the_other_side():
    """200 seeded random paths of 3 to 12 steps, each step going straight on
    as often as it turns. The ones with a node dropped or touching a
    non-adjacent node of its own run are set aside (which of two overlapping
    nodes goes depends on drawing order by design); the rest must match."""
    rng = random.Random(20261013)
    checked = 0
    for _ in range(200):
        steps = [rng.choice((N, S, W, E))]
        for _ in range(rng.randint(2, 11)):
            dx, dy = steps[-1]
            steps.append(rng.choice([(dx, dy), (dx, dy), (dy, dx), (-dy, -dx)]))
        same = _same_run_from_the_far_end(steps, rng.choice((1, -1)))
        if same is None:
            continue
        assert same, steps
        checked += 1
    assert checked >= 150


# -- existing cliffs: contacts, blockers and the slide ----------------------


def _cliff(suffix, bounds, rotation=None):
    return cliff_chain.ExistingCliff(unit_const=DEFAULT[suffix], bounds=bounds, rotation=rotation)


def test_a_click_whose_centred_piece_overlaps_a_cliff_slides_off_it_and_joins():
    """An s2 at (10,11,7,10), a click at (10,10). Centred, the picked 3x3
    covers the cliff's last tile; it slides one south to (9,12,10,13), where
    the cliff above makes it a thin north cap, a 1x3. That recentres onto the
    cliff, so the slide runs again at the new span: flush under it."""
    existing = [_cliff(2, (10, 11, 7, 10), rotation=22)]
    nodes = _path((10, 10), [], existing=existing)
    assert nodes, "the click must place a cliff"
    assert _rows(nodes) == [(DEFAULT_02, 23, 10.5, 11.5, (10, 11, 10, 13), True)]
    assert nodes[0].arms == frozenset({N})


def test_the_slide_takes_the_smallest_shift_that_still_covers_the_press_tile():
    """A 3x3 cliff at (10,13,6,9). A click at (11,9) centres on (10,13,8,11),
    one row into it: one south clears it. x goes before y and the negative
    shift first on a tie, so beside a cliff to the east the piece goes west."""
    assert _path((11, 9), [], existing=[_cliff(1, (10, 13, 6, 9))])[0].bounds == (10, 13, 9, 12)
    assert _path((11, 9), [], existing=[_cliff(1, (12, 15, 8, 11))])[0].bounds == (9, 12, 8, 11)
    # Clear already: no slide, and a diagonal touch alone changes nothing.
    assert _path((11, 10), [], existing=[_cliff(1, (13, 16, 6, 9))])[0].bounds == (10, 13, 9, 12)


def test_a_click_with_no_clear_shift_places_nothing():
    """A one-row gap between two 3x3 cliffs: every 3x3 over the press tile
    overlaps one of them, so the node is dropped as it always was."""
    existing = [_cliff(1, (10, 13, 7, 10)), _cliff(1, (10, 13, 11, 14))]
    assert _path((11, 10), [], existing=existing) == []


def test_a_drag_starting_against_an_existing_thin_cliff_continues_it_thin():
    """A free start under an s2: drawn south the default side is tall, which
    cannot continue a thin piece (no straight has one leg of each), so the run
    takes the other side and is all 1x3, flush under the cliff."""
    existing = [_cliff(2, (10, 11, 7, 10), rotation=22)]
    stroke = _stroke(existing=existing)
    stroke.set_path((10, 10), [S, S, S])
    assert _rows(stroke.nodes()) == [
        (DEFAULT_02, 7, 10.5, 11.5, (10, 11, 10, 13), True),
        (DEFAULT_02, 8, 10.5, 14.5, (10, 11, 13, 16), True),
        (DEFAULT_02, 7, 10.5, 17.5, (10, 11, 16, 19), True),
        (DEFAULT_02, 23, 10.5, 20.5, (10, 11, 19, 22), True),
    ]
    assert stroke.nodes()[0].arms == frozenset({N, S})
    assert stroke.side == -1


def test_a_free_start_beside_a_tall_cliff_keeps_the_default_side():
    """Under a 3x3 the default side's tall start already joins: no flip."""
    stroke = _stroke(existing=[_cliff(1, (9, 12, 6, 9), rotation=19)])
    stroke.set_path((10, 10), [S, S])
    assert stroke.side == 1
    assert _suffixes(stroke.nodes()) == [1, 1, 1]
    assert stroke.nodes()[0].arms == frozenset({N, S})


def test_a_contact_the_table_has_no_piece_for_keeps_the_path_shape():
    """A straight run squeezed between two existing cliffs: the middle node
    would have four arms, which no piece has, so it stays the straight its
    path arms make it, and stays resolved (both of those arms are joined)."""
    existing = [_cliff(1, (7, 10, 13, 16)), _cliff(1, (13, 16, 13, 16))]
    nodes = _path((11, 11), [S, S], frame=7, existing=existing)
    assert _rows(nodes) == [
        (DEFAULT_01, 18, 11.5, 11.5, (10, 13, 10, 13), True),
        (DEFAULT_01, 2, 11.5, 14.5, (10, 13, 13, 16), True),
        (DEFAULT_01, 19, 11.5, 17.5, (10, 13, 16, 19), True),
    ]
    assert nodes[1].arms == frozenset({N, S})


def test_a_shape_with_no_piece_keeps_the_picked_piece_and_frame(monkeypatch):
    """Marble has 23 frames, so the thin north cap (frame 23) does not exist
    there: that node keeps the picker's piece and frame and is flagged."""
    marble_01 = 2178
    monkeypatch.setattr(cliff_catalog, "frame_count", lambda _const: 23)
    nodes = _path((40, 40), [N, N], const=marble_01, frame=5)
    assert [(n.unit_const, n.rotation, n.resolved) for n in nodes] == [
        (marble_01, 5, False),
        (cliff_catalog.piece_for_suffix(marble_01, 2), _twin((7, 8), nodes[1].bounds), True),
        (cliff_catalog.piece_for_suffix(marble_01, 2), 22, True),
    ]


def test_a_node_overlapping_an_existing_cliff_is_skipped_but_the_run_continues():
    """A cliff already on the map blocks its own tiles without truncating the
    drag: the node on it is dropped and the path keeps walking past it."""
    existing = [_cliff(1, (10, 13, 13, 16))]
    nodes = _drag(_stroke(existing=existing), [(11, y) for y in range(11, 26)])
    assert (10, 13, 13, 16) not in [n.bounds for n in nodes]
    assert (10, 13, 10, 13) in [n.bounds for n in nodes]
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
    later one would land back on the map. The last kept node was shaped for a
    neighbour it does not get, so it is flagged."""
    nodes = _path((11, 11), [S, S, S, S, W, W, N], width=20, height=20)
    assert [n.bounds for n in nodes] == [(10, 13, 10, 13), (10, 13, 13, 16), (10, 13, 16, 19)]
    assert [n.resolved for n in nodes] == [True, True, False]


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
    run (the same four nodes as the thin continuation above)."""
    existing = [_cliff(2, (10, 11, 7, 10), rotation=22)]
    stroke = _stroke(existing=existing)
    nodes = _drag(stroke, [(10 + y % 2, y) for y in range(10, 22)])
    assert stroke.steps == [S, S, S]
    assert [(n.unit_const, n.bounds) for n in nodes] == [
        (DEFAULT_02, (10, 11, 10, 13)), (DEFAULT_02, (10, 11, 13, 16)),
        (DEFAULT_02, (10, 11, 16, 19)), (DEFAULT_02, (10, 11, 19, 22)),
    ]


def test_dragging_back_over_the_run_does_not_step_onto_it():
    stroke = _stroke()
    _drag(stroke, [(10, y) for y in range(10, 18)] + [(10, y) for y in range(17, 9, -1)])
    assert stroke.steps == [S, S]
    assert len(stroke.nodes()) == 3


# -- a press on an existing cliff anchors the run on it ----------------------

_TALL_CAP_FRAME = {E: 16, W: 17, S: 18, N: 19}
_THIN_CAP = {E: (3, 20), W: (3, 21), S: (2, 22), N: (2, 23)}
_NAMES = {N: "north", S: "south", W: "west", E: "east"}
_SPANS = {1: (3, 3), 2: (1, 3), 3: (3, 1)}


def _end_cap(arm, tall):
    """[the cap, its one neighbour]: an end cap at (40, 40) whose run goes
    off along `arm`, the same size throughout."""
    suffix, frame = (1, _TALL_CAP_FRAME[arm]) if tall else _THIN_CAP[arm]
    sx, sy = _SPANS[suffix]
    cap = (40, 40 + sx, 40, 40 + sy)
    x, y = 40 + arm[0] * sx, 40 + arm[1] * sy
    return [_cliff(suffix, cap, rotation=frame), _cliff(suffix, (x, x + sx, y, y + sy))]


def _anchored(existing, steps):
    stroke = _stroke(existing=existing)
    stroke.set_path((existing[0].bounds[0], existing[0].bounds[2]), steps)
    assert stroke.anchor == 0
    return stroke


_IN_LINE = [(arm, tall) for tall in (True, False) for arm in (N, S, W, E)]
_STRAIGHT_TWINS = {(True, 0): (1, 2), (True, 1): (4, 5), (False, 0): (7, 8), (False, 1): (10, 11)}


@pytest.mark.parametrize(
    ("arm", "tall"), _IN_LINE, ids=[f"{'tall' if tall else 'thin'}_cap_run_to_the_{_NAMES[a]}" for a, tall in _IN_LINE]
)
def test_a_run_continued_in_line_off_an_end_cap_keeps_its_size_and_reshapes_the_cap(arm, tall):
    """The cap passes its own run's side on, so the new pieces are its size,
    and the old cap becomes the straight twin its position gives."""
    existing = _end_cap(arm, tall)
    away = (-arm[0], -arm[1])
    stroke = _anchored(existing, [away, away])
    nodes = stroke.nodes()
    cap_suffix = SUFFIX_OF[existing[0].unit_const]
    assert _suffixes(nodes) == [cap_suffix, cap_suffix]
    assert all(n.resolved for n in nodes)
    assert cliff_connectivity.adjacency_dir(existing[0].bounds, nodes[0].bounds) == away
    twins = _STRAIGHT_TWINS[(tall, 1 if arm[0] else 0)]
    assert stroke.rewrites() == [cliff_chain.ChainRewrite(0, _twin(twins, existing[0].bounds))]
    assert stroke.unjoined() == 0
    # The far end is that size's own cap, facing back along the run.
    assert nodes[-1].rotation == (_TALL_CAP_FRAME[arm] if tall else _THIN_CAP[arm][1])


# A tall cap turned sideways: the all-3x3 corners {N, E} (frame 3) and {S, W}
# (frame 15) exist, {N, W} and {E, S} do not.
_SIDEWAYS_TALL = [
    (N, E, 3), (N, W, None), (S, W, 15), (S, E, None), (E, N, 3), (E, S, None), (W, S, 15), (W, N, None),
]


@pytest.mark.parametrize(
    ("arm", "step", "corner"),
    _SIDEWAYS_TALL,
    ids=[f"tall_cap_run_to_the_{_NAMES[a]}_dragged_{_NAMES[d]}" for a, d, _ in _SIDEWAYS_TALL],
)
def test_a_run_started_sideways_off_a_tall_cap_joins_where_an_all_tall_corner_exists(arm, step, corner):
    existing = _end_cap(arm, True)
    stroke = _anchored(existing, [step, step])
    nodes = stroke.nodes()
    assert len(nodes) == 2
    assert cliff_connectivity.adjacency_dir(existing[0].bounds, nodes[0].bounds) == step
    if corner is None:
        # That join needs a different-size piece: the old cap is left alone.
        assert stroke.rewrites() == []
        assert stroke.unjoined() == 1
        assert _suffixes(nodes) == [3 if step[0] else 2] * 2
    else:
        assert stroke.rewrites() == [cliff_chain.ChainRewrite(0, corner)]
        assert stroke.unjoined() == 0
        assert _suffixes(nodes) == [1, 1]


_SIDEWAYS_THIN = [(arm, step) for arm in (N, S, W, E) for step in (N, S, W, E) if arm[0] * step[0] + arm[1] * step[1] == 0]


@pytest.mark.parametrize(
    ("arm", "step"),
    _SIDEWAYS_THIN,
    ids=[f"thin_cap_run_to_the_{_NAMES[a]}_dragged_{_NAMES[d]}" for a, d in _SIDEWAYS_THIN],
)
def test_a_run_started_sideways_off_a_thin_cap_leaves_it_alone(arm, step):
    """No thin piece turns a corner at its own size. The first node has no
    measured offset off a thin piece's broadside, so it abuts flush."""
    existing = _end_cap(arm, False)
    stroke = _anchored(existing, [step, step])
    nodes = stroke.nodes()
    assert len(nodes) == 2
    assert cliff_connectivity.adjacency_dir(existing[0].bounds, nodes[0].bounds) == step
    assert stroke.rewrites() == []
    assert stroke.unjoined() == 1


def test_a_lone_piece_takes_the_side_that_matches_its_own_leg():
    """A lone 3x3 dragged west would be thin by default; its west leg is
    tall, so the run is tall and it becomes that run's east-facing cap. A lone
    1x3 dragged south continues thin. Dragged east its broadside is tall only
    nominally (no such cap), so the default stands and it is left alone."""
    tall = _anchored([_cliff(1, (40, 43, 40, 43), rotation=3)], [W, W])
    assert tall.side == -1
    assert _suffixes(tall.nodes()) == [1, 1]
    assert tall.rewrites() == [cliff_chain.ChainRewrite(0, 17)]

    thin = _anchored([_cliff(2, (40, 41, 40, 43), rotation=7)], [S, S])
    assert thin.side == -1
    assert _suffixes(thin.nodes()) == [2, 2]
    assert thin.rewrites() == [cliff_chain.ChainRewrite(0, 22)]

    broadside = _anchored([_cliff(2, (40, 41, 40, 43), rotation=7)], [E, E])
    assert broadside.side == 1
    assert _suffixes(broadside.nodes()) == [1, 1]
    assert (broadside.rewrites(), broadside.unjoined()) == ([], 1)


def test_a_piece_already_on_its_joined_frame_is_not_rewritten_or_counted():
    stroke = _anchored([_cliff(1, (40, 43, 40, 43), rotation=17)], [W, W])
    assert (stroke.rewrites(), stroke.unjoined()) == ([], 0)


def test_a_mid_run_anchor_leaves_the_side_to_the_drag_and_is_never_rewritten():
    """A press on the middle of a tall north-south run, dragged east: the
    default side. The T it becomes keeps its straight frame, so nothing is
    counted; a thin run's middle has no such T and is counted, not rewritten.
    Whatever frame the middle piece is on, it stays on it."""
    tall = [_cliff(1, (40, 43, 40, 43), rotation=19), _cliff(1, (40, 43, 37, 40)), _cliff(1, (40, 43, 43, 46))]
    stroke = _anchored(tall, [E, E])
    assert stroke.side == 1
    assert _suffixes(stroke.nodes()) == [1, 1]
    assert (stroke.rewrites(), stroke.unjoined()) == ([], 0)

    thin = [_cliff(2, (40, 41, 40, 43), rotation=7), _cliff(2, (40, 41, 37, 40)), _cliff(2, (40, 41, 43, 46))]
    stroke = _anchored(thin, [E, E])
    assert (stroke.rewrites(), stroke.unjoined()) == ([], 1)


def test_an_anchor_is_never_placed_and_a_press_on_one_with_no_step_places_nothing():
    existing = _end_cap(N, True)
    stroke = _anchored(existing, [])
    assert stroke.nodes() == []
    assert stroke.head_bounds() is None
    assert (stroke.rewrites(), stroke.unjoined(), stroke.side) == ([], 0, None)
    stroke.set_path((41, 41), [S])
    assert [n.bounds for n in stroke.nodes()] == [(40, 43, 43, 46)]
    assert [n.index for n in stroke.nodes()] == [1]
    assert stroke.head_bounds() == (40, 43, 43, 46)


def test_a_walk_steps_from_the_anchor_s_own_footprint():
    """The cursor has to leave the anchor's box before the first step, and
    the first node abuts the anchor, wherever in it the press landed."""
    stroke = _stroke(existing=_end_cap(N, True))
    nodes = _drag(stroke, [(42, 42), (42, 43), (42, 47)])
    assert stroke.anchor == 0
    assert stroke.steps == [S, S]
    assert [n.bounds for n in nodes] == [(40, 43, 43, 46), (40, 43, 46, 49)]


def test_a_refused_first_step_leaves_the_side_unset():
    """A lone 3x3 at the east edge: a first step east has no room, so no step
    is taken and no side is kept. The step west that follows is then tall
    (the piece's own west leg), not the thin the default side would draw."""
    stroke = _stroke(existing=[_cliff(1, (116, 119, 40, 43), rotation=3)])
    _drag(stroke, [(117, 41), (119, 41)])
    assert (stroke.steps, stroke.side, stroke.nodes()) == ([], None, [])
    nodes = _drag(stroke, [(113, 41)])
    assert stroke.steps == [W]
    assert stroke.side == -1
    assert _suffixes(nodes) == [1]
    assert stroke.rewrites() == [cliff_chain.ChainRewrite(0, 17)]


def test_a_free_run_ending_beside_an_end_cap_reshapes_it():
    """Not only the anchor: a lone click under a tall north-south run's
    south-facing end makes a new cap there and the old one a straight."""
    existing = [_cliff(1, (40, 43, 37, 40), rotation=19), _cliff(1, (40, 43, 34, 37), rotation=18)]
    stroke = _stroke(existing=existing)
    stroke.set_path((41, 41), [])
    assert _rows(stroke.nodes()) == [(DEFAULT_01, 19, 41.5, 41.5, (40, 43, 40, 43), True)]
    assert stroke.rewrites() == [cliff_chain.ChainRewrite(0, _twin((1, 2), (40, 43, 37, 40)))]


def test_a_run_passing_beside_a_straight_or_a_corner_rewrites_nothing():
    """Only an end cap or a lone piece is reshaped. A run that starts against
    the middle of a straight, or against a corner, leaves both as they are."""
    straight = [_cliff(1, (40, 43, 37, 40), rotation=1), _cliff(1, (40, 43, 34, 37)), _cliff(1, (40, 43, 40, 43))]
    stroke = _stroke(existing=straight)
    stroke.set_path((44, 38), [E, E])
    assert stroke.nodes()[0].bounds == (43, 46, 37, 40)
    assert (stroke.rewrites(), stroke.unjoined()) == ([], 0)

    corner = [_cliff(1, (40, 43, 40, 43), rotation=3), _cliff(1, (40, 43, 37, 40)), _cliff(1, (43, 46, 40, 43))]
    stroke = _stroke(existing=corner)
    stroke.set_path((38, 41), [W, W])
    assert cliff_connectivity.adjacency_dir(stroke.nodes()[0].bounds, corner[0].bounds) == E
    assert (stroke.rewrites(), stroke.unjoined()) == ([], 0)


def test_a_run_of_the_other_facing_arriving_end_on_leaves_the_cap_alone():
    """A join needs both legs the same type. A tall run drawn east up to a
    thin run's west end stays a tall cap there (no piece has one leg of
    each), so the thin cap is not made a straight: left alone and counted."""
    thin = [_cliff(3, (60, 63, 40, 41), rotation=20), _cliff(3, (63, 66, 40, 41))]
    stroke = _stroke(existing=thin)
    stroke.set_path((52, 40), [E, E])
    last = stroke.nodes()[-1]
    assert cliff_connectivity.adjacency_dir(last.bounds, thin[0].bounds) == E
    assert (SUFFIX_OF[last.unit_const], last.rotation, last.arms) == (1, 17, frozenset({W}))
    assert (stroke.rewrites(), stroke.unjoined()) == ([], 1)


def test_a_run_dragged_through_its_anchor_s_run_leaves_the_far_cap_alone():
    """A press on a tall run's south cap, dragged north through the run: the
    pieces that come out above it are thin, so the tall north cap they meet
    end-on keeps its frame."""
    tall = [_cliff(1, (40, 43, 43, 46), rotation=19), _cliff(1, (40, 43, 40, 43), rotation=1),
            _cliff(1, (40, 43, 37, 40), rotation=18)]
    stroke = _anchored(tall, [N, N, N, N])
    above = [n for n in stroke.nodes() if n.bounds[3] <= 37]
    assert above and {SUFFIX_OF[n.unit_const] for n in above} == {2}
    assert cliff_connectivity.adjacency_dir(tall[2].bounds, above[0].bounds) == N
    assert stroke.rewrites() == []
    assert stroke.unjoined() == 1


def test_existing_cliffs_carry_what_a_rewrite_needs():
    """rotation, owner and the live unit, off a scenario-shaped stand-in; a
    stored rotation that is no frame index reads as None."""
    from types import SimpleNamespace

    def unit(const, x, y, rotation):
        return SimpleNamespace(unit_const=const, x=x, y=y, rotation=rotation)

    gaia = [unit(DEFAULT_01, 11.5, 11.5, 19.0), unit(4, 5.5, 5.5, 0.0)]
    other = [unit(DEFAULT_02, 30.5, 31.5, float("nan"))]
    scenario = SimpleNamespace(unit_manager=SimpleNamespace(units=[gaia, other]))
    found = cliff_chain.existing_cliffs(scenario, 120, 120)
    assert [(c.unit_const, c.bounds, c.rotation, c.player) for c in found] == [
        (DEFAULT_01, (10, 13, 10, 13), 19, 0),
        (DEFAULT_02, (30, 31, 30, 33), None, 1),
    ]
    assert found[0].unit is gaia[0] and found[1].unit is other[0]
    # The pre-existing two-field construction still stands.
    assert cliff_chain.ExistingCliff(unit_const=DEFAULT_01, bounds=(0, 3, 0, 3)).unit is None


def test_a_prebuilt_index_gives_the_same_run():
    existing = _end_cap(N, True)
    index = cliff_chain.index_cliffs(existing)
    shared = cliff_chain.ChainStroke(
        map_width=120, map_height=120, existing=existing, fallback_const=DEFAULT_01, fallback_frame=3, existing_index=index
    )
    shared.set_path((41, 41), [S, S])
    own = _anchored(existing, [S, S])
    assert _rows(shared.nodes()) == _rows(own.nodes())
    assert shared.rewrites() == own.rewrites()


# -- walks -------------------------------------------------------------------


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
    node has a neighbour on each arm it was shaped for, and the same drag
    twice gives the same run."""
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
            assert node.arms >= path[i]
            if node.resolved:
                assert node.arms <= _realized(node, nodes, [])
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
    assert resolved > total * 3 // 4


# -- ground truth: the drawn run's ink (needs a real AoE2:DE install) ----

# The south-then-west drag below, as the one-suffix-per-stroke resolver two
# models ago placed it with Cliff (Default) 01 picked (recorded at eebce52):
# eight 3x3 pieces, (const, rotation, x, y).
_ONE_SUFFIX_L = [
    (DEFAULT_01, 18, 41.5, 41.5), (DEFAULT_01, 1, 41.5, 44.5), (DEFAULT_01, 0, 41.5, 47.5),
    (DEFAULT_01, 17, 41.5, 50.5), (DEFAULT_01, 17, 38.5, 50.5), (DEFAULT_01, 4, 35.5, 50.5),
    (DEFAULT_01, 4, 32.5, 50.5), (DEFAULT_01, 16, 29.5, 50.5),
]
_L_DRAG = [(40, y) for y in range(40, 52)] + [(x, 51) for x in range(39, 27, -1)]
_INK_TILE_PX = 64
# The art's own scale: a tile is 96 px wide, the size the heights below were measured at.
_NATIVE_HALF_W = 48
_FRONT_MIN_INK_PX = 150
_BACK_MAX_INK_PX = 110


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


_INK_RUNS = {
    "front_straight_east": [E, E, E, E],
    "front_straight_south": [S, S, S, S],
    "back_straight_west": [W, W, W, W],
    "back_straight_north": [N, N, N, N],
    "all_3x3_corner_south_east": [S, S, E, E],
    "all_3x3_corner_east_south": [E, E, S, S],
    "l_south_west": [S, S, W, W],
    "l_west_south": [W, W, S, S],
    "l_east_north": [E, E, N, N],
    "l_north_east": [N, N, E, E],
    "l_north_west": [N, N, W, W],
    "l_west_north": [W, W, N, N],
    "u_turn": [S, S, E, E, N, N],
    "zig_zag": [S, S, E, E, S, S],
}


@pytest.mark.parametrize("name", sorted(_INK_RUNS))
def test_a_resolved_run_draws_one_connected_wall(name):
    _require_install()
    assert len(_walls(*_ink(_placed(_path((40, 40), _INK_RUNS[name]))))) == 1


def test_the_l_drag_draws_one_wall_where_the_one_suffix_resolver_drew_two():
    """The connectivity check's negative control: the same drag, south then
    west, splits into two walls as the first resolver drew it."""
    _require_install()
    assert len(_walls(*_ink(_placed(_drag(_stroke(), _L_DRAG))))) == 1
    assert len(_walls(*_ink(_ONE_SUFFIX_L))) == 2


def _straight_ink_heights(nodes) -> list[int]:
    """Decoded ink height, at the art's own scale, of each two-arm node."""
    heights = []
    for node in nodes:
        if len(node.arms) != 2:
            continue
        (piece,) = unit_sprites.sprite_pieces_for(node.unit_const, float(node.rotation), 0, _NATIVE_HALF_W)
        rows = np.nonzero((piece.draw.rgba[..., 3] > 0).any(axis=1))[0]
        heights.append(int(rows[-1] - rows[0] + 1))
    return heights


def _is_front(heights) -> bool:
    return bool(heights) and all(h >= _FRONT_MIN_INK_PX for h in heights)


def _is_back(heights) -> bool:
    return bool(heights) and all(h <= _BACK_MAX_INK_PX for h in heights)


@pytest.mark.parametrize(("step", "front"), [(E, True), (S, True), (W, False), (N, False)], ids=["east", "south", "west", "north"])
def test_a_straight_s_ink_is_the_rock_face_or_the_ridge_by_drag_direction(step, front):
    """Connectivity cannot tell front from back, the ink's height can: a
    front straight's frames are a rock face (192 px measured), a back
    straight's the low ridge (85 to 90 px)."""
    _require_install()
    heights = _straight_ink_heights(_path((40, 40), [step] * 4))
    assert len(heights) == 3
    assert (_is_front(heights), _is_back(heights)) == (front, not front), heights


def test_the_ink_height_check_fails_a_run_dressed_for_the_wrong_side():
    """The height check's own control: the same eastward path dressed for the
    other side is the ridge, and must not pass as a front straight."""
    _require_install()
    wrong = _straight_ink_heights(_path((40, 40), [E] * 4, side=-1))
    assert len(wrong) == 3
    assert not _is_front(wrong) and _is_back(wrong), wrong
