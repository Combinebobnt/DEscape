"""Track B Stage 2 of the 2026-09-02 cliffs plan: the auto-connecting chain
drag's geometry and piece resolution, exercised headless (descape.cliff_chain
has no Qt dependency -- the ViewerWindow wiring is tested separately in
tests/test_cliff_tool_viewer.py).
"""

from __future__ import annotations

import pytest

from descape import cliff_catalog, cliff_chain, cliff_connectivity, terrain_palette

# n_cliff_default_x1's own pieces: 01 is 3x3, 02 is 1x3, 08 is 2x2.
DEFAULT_01 = 264
DEFAULT_02 = 265
DEFAULT_08 = 271


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
    """Nodes sit on the first node's own span lattice, not wherever the
    cursor happened to leave the previous footprint -- 3x3 pieces are three
    tiles apart, and consecutive bounds share an edge exactly."""
    nodes = _drag(_stroke(), [(10, y) for y in range(10, 25)])
    assert [n.bounds for n in nodes] == [
        (10, 13, 10, 13), (10, 13, 13, 16), (10, 13, 16, 19), (10, 13, 19, 22), (10, 13, 22, 25)
    ]
    for earlier, later in zip(nodes, nodes[1:]):
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
    """Interior nodes take the {N, S} entry's own rotations; the two ends
    take the single-neighbour entries', which are different frames -- the
    whole point of resolving per node rather than stamping one piece."""
    nodes = _drag(_stroke(), [(10, y) for y in range(10, 22)])
    interior = nodes[1:-1]
    assert all(n.resolved for n in nodes)
    for node in interior:
        assert node.rotation in cliff_connectivity.rotations_for(frozenset({(0, -1), (0, 1)}), 1)
    assert nodes[0].rotation in cliff_connectivity.rotations_for(frozenset({(0, 1)}), 1)
    assert nodes[-1].rotation in cliff_connectivity.rotations_for(frozenset({(0, -1)}), 1)


def test_face_inherits_along_a_run_but_a_corner_takes_its_own():
    """The plan's Decisions addendum, second rule, read as "keep the previous
    node's face when this node's candidates allow it". Taken literally it
    would carry a straight run's frame onto the corner, whose candidate set
    is disjoint from the run's -- writing a shape that doesn't join."""
    tiles = [(10, y) for y in range(10, 32)] + [(x, 31) for x in range(11, 22)]
    nodes = _drag(_stroke(), tiles)
    # Anchors run 10, 13, ... 31; the corner is at 31 and the node above it
    # sees it diagonally, so the strictly-{N, S} stretch is 13 through 22.
    straight = [n for n in nodes if n.bounds[0] == 10 and n.bounds[2] in (13, 16, 19, 22)]
    assert len(straight) == 4
    assert len({n.rotation for n in straight}) == 1
    corner = next(n for n in nodes if n.bounds == (10, 13, 31, 34))
    assert corner.rotation in cliff_connectivity.rotations_for(frozenset({(0, -1), (1, 0)}), 1)
    assert corner.rotation not in {n.rotation for n in straight}


def test_an_unpinned_junction_falls_back_to_the_picked_frame():
    """A pure diagonal run's dirset never reached the table's min_n of 30,
    so it is one of the ~5% of shapes with no pinned entry -- the plan's
    Decisions addendum, third rule: use the picker, never a guess."""
    nodes = _drag(_stroke(frame=7), [(10 + i, 10 + i) for i in range(12)])
    assert len(nodes) > 2
    assert cliff_connectivity.resolve(frozenset({(-1, -1), (1, 1)})) is None
    assert all(n.rotation == 7 for n in nodes)
    assert not any(n.resolved for n in nodes)


def test_the_stroke_inherits_an_adjacent_existing_cliff_s_piece_size():
    """Decisions addendum, first rule. The existing 1x3 "02" piece sits
    directly above the first node, so the whole run switches off the picked
    3x3 "01" -- and the lattice steps by the inherited span, not the picked
    one."""
    existing = [cliff_chain.ExistingCliff(unit_const=DEFAULT_02, bounds=(10, 11, 7, 10))]
    nodes = _drag(_stroke(const=DEFAULT_01, existing=existing), [(10, y) for y in range(10, 20)])
    assert all(n.unit_const == DEFAULT_02 for n in nodes)
    assert terrain_palette.tile_span(DEFAULT_02, (1, 1)) == (1, 3)
    assert [n.bounds for n in nodes] == [
        (10, 11, 10, 13), (10, 11, 13, 16), (10, 11, 16, 19), (10, 11, 19, 22)
    ]
    # The existing cliff is a real neighbour, so the first node is a run
    # interior rather than an end cap.
    assert nodes[0].rotation in cliff_connectivity.rotations_for(frozenset({(0, -1), (0, 1)}), 2)


def test_a_node_overlapping_an_existing_cliff_is_skipped_but_the_run_continues():
    """A cliff already on the map blocks its own tiles without truncating the
    drag -- the lattice head keeps walking past it, which is why it is
    tracked separately from the last emitted anchor."""
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


@pytest.mark.parametrize("const", [DEFAULT_01, DEFAULT_02, DEFAULT_08])
def test_the_anchor_round_trips_through_render_s_own_span_start(const):
    """span_anchor is _span_start's inverse (test_cliffs.py pins that
    directly); this asserts the chain actually places on it, so a node's
    tile bounds are the ones the renderer will report for it."""
    from descape.render import unit_tile_bounds

    node = _drag(_stroke(const=const), [(10, 10)])[0]
    placed = type("U", (), {"x": node.x, "y": node.y, "unit_const": const})()
    assert unit_tile_bounds(placed, 120, 120) == node.bounds


def test_a_run_uses_the_rotations_measured_for_its_own_piece_size():
    """The defect the L-shaped write-path proof caught: the table's
    top-level rotations belong to whichever piece size was that dirset's
    corpus majority, so a run pinned to a different size drew that other
    size's frames -- a thin ridge between tall end caps. Same drag, two
    piece sizes, two different frame sets."""
    run = frozenset({(0, -1), (0, 1)})
    big = _drag(_stroke(const=DEFAULT_01), [(10, y) for y in range(10, 22)])[1:-1]
    thin = _drag(_stroke(const=DEFAULT_02), [(10, y) for y in range(10, 22)])[1:-1]
    assert {n.rotation for n in big} <= set(cliff_connectivity.rotations_for(run, 1))
    assert {n.rotation for n in thin} <= set(cliff_connectivity.rotations_for(run, 2))
    assert {n.rotation for n in big}.isdisjoint({n.rotation for n in thin})
