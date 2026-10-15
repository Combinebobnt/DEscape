"""The measured cliff connectivity tables and the facing model's lookup over
them. See descape/cliff_connectivity.py's module docstring for the tables'
shape and tools/gen_cliff_connectivity.py's for the measurement methodology
and why the older per-dirset majority's purity is as low as it is.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from descape import cliff_connectivity

import conftest

ROOT = Path(__file__).resolve().parent.parent
PRIVATE_CORPUS = ROOT / "maintainer" / "scenarios_for_descape"


def test_adjacency_dir_cardinal_and_diagonal() -> None:
    # A: tile (5,5)..(6,6) (a 1x1 box, half-open bounds (5,6,5,6))
    a = (5, 6, 5, 6)
    north = (5, 6, 4, 5)
    south = (5, 6, 6, 7)
    east = (6, 7, 5, 6)
    west = (4, 5, 5, 6)
    ne = (6, 7, 4, 5)
    nw = (4, 5, 4, 5)
    se = (6, 7, 6, 7)
    sw = (4, 5, 6, 7)
    far = (9, 10, 9, 10)

    assert cliff_connectivity.adjacency_dir(a, north) == (0, -1)
    assert cliff_connectivity.adjacency_dir(a, south) == (0, 1)
    assert cliff_connectivity.adjacency_dir(a, east) == (1, 0)
    assert cliff_connectivity.adjacency_dir(a, west) == (-1, 0)
    assert cliff_connectivity.adjacency_dir(a, ne) == (1, -1)
    assert cliff_connectivity.adjacency_dir(a, nw) == (-1, -1)
    assert cliff_connectivity.adjacency_dir(a, se) == (1, 1)
    assert cliff_connectivity.adjacency_dir(a, sw) == (-1, 1)
    assert cliff_connectivity.adjacency_dir(a, far) is None


def test_adjacency_dir_is_symmetric_under_swap() -> None:
    a = (10, 13, 10, 13)  # a 3x3 box
    b = (13, 15, 10, 12)  # a 2x2 box flush to its east
    forward = cliff_connectivity.adjacency_dir(a, b)
    backward = cliff_connectivity.adjacency_dir(b, a)
    assert forward == (1, 0)
    assert backward == (-1, 0)


def test_resolve_pinned_vertical_run() -> None:
    # The single best-populated dirset in the measured corpus (n=2496): a
    # straight two-neighbour (N, S) run resolves to suffix 2 (the family's
    # "02" 1x3 piece), with rotations 7/8 as its two observed faces.
    dirset = frozenset({(0, -1), (0, 1)})
    result = cliff_connectivity.resolve(dirset)
    assert result is not None
    suffix, rotations = result
    assert suffix == 2
    assert set(rotations) == {7, 8}


def test_resolve_unknown_dirset_returns_none() -> None:
    # An 8-direction dirset with an already-covered opposite pair repeated
    # is impossible; a nonsense placeholder key was never pinned.
    assert cliff_connectivity.resolve(frozenset({(2, 2)})) is None


def test_rotations_for_is_conditioned_on_suffix() -> None:
    """The same neighbour shape uses different frames at different piece
    sizes, because a 1x3 footprint holds thin shapes and a 3x3 holds big
    ones. This is why Track B Stage 2's drag reads rotations_for() and never
    resolve(): borrowing the majority suffix's rotations for a run pinned to
    another size draws a wrong-scale frame -- measured as 90px of decoded
    ink where 192px was correct."""
    run = frozenset({(0, -1), (0, 1)})
    at_3x3 = cliff_connectivity.rotations_for(run, 1)
    at_1x3 = cliff_connectivity.rotations_for(run, 2)
    assert at_3x3 == (1, 2)
    assert at_1x3 == (8, 7)
    # resolve()'s own rotations are the majority suffix's -- suffix 2 here.
    assert cliff_connectivity.resolve(run)[1] == at_1x3


def test_rotations_for_unpinned_pair_returns_none() -> None:
    """Splitting by suffix makes every bucket sparser, so a pair below
    min_suffix_n is simply absent. A caller must fall back to its own manual
    value; falling back to resolve() would reintroduce the cross-suffix bug
    while still looking resolved."""
    assert cliff_connectivity.rotations_for(frozenset({(0, -1), (0, 1)}), 9) is None
    assert cliff_connectivity.rotations_for(frozenset({(2, 2)}), 1) is None


def test_metadata_matches_shipped_table() -> None:
    meta = cliff_connectivity.metadata()
    assert meta["min_n"] == 30
    assert meta["min_suffix_n"] == 10
    # Regression floor, not an exact pin -- a corpus re-measurement can move
    # these a little; a future change that tanks them is the real signal.
    assert meta["coverage"] >= 0.90
    assert 0.50 <= meta["purity_within_coverage"] <= 0.75


def _synthetic_l(gen):
    """A 1x3 vertical straight, a 2x2 {N,E} corner below it, a 3x1 east cap,
    plus a lone 3x3 far away (no neighbours, so in no projection)."""
    boxes = [((10, 11, 0, 3), 2, (1, 3)), ((10, 12, 3, 5), 8, (2, 2)), ((12, 15, 3, 4), 3, (3, 1)), ((40, 43, 40, 43), 1, (3, 3))]
    nodes = [gen.ScannedNode(suffix=s, rotation=s * 10, bounds=b, span=span) for b, s, span in boxes]
    adjacency = [
        [(j, d) for j, other in enumerate(nodes) if j != i and (d := cliff_connectivity.adjacency_dir(n.bounds, other.bounds))]
        for i, n in enumerate(nodes)
    ]
    return gen.ScannedFile(name="synthetic", nodes=nodes, adjacency=adjacency)


def test_offset_projections_on_a_synthetic_l() -> None:
    gen = conftest.load_verify_module("gen_cliff_connectivity")
    files = [_synthetic_l(gen)]
    offsets, by_span, stats = gen.build_offsets(files)
    assert {"a": [[[0, 1]], [1, 3]], "d": [0, 1], "b": [[[0, -1], [1, 0]], [2, 2]], "offsets": [[[0, 3], 1]]} in offsets
    assert {"a": [[[0, -1], [1, 0]], [2, 2]], "d": [1, 0], "b": [[[-1, 0]], [3, 1]], "offsets": [[[2, 0], 1]]} in offsets
    assert len(offsets) == 4 and len(by_span) == 4
    assert stats["offset_modal_share"] == 1.0

    # The dirset projection still sees the lone node (empty dirset) as a record.
    counts = gen.dirset_counts(files)
    assert counts[frozenset()] == {(1, 10): 1}


def test_shape_projection_on_a_synthetic_l() -> None:
    """One row per (cardinal arms, suffix); the lone node has no arm and no row."""
    gen = conftest.load_verify_module("gen_cliff_connectivity")
    files = [_synthetic_l(gen)]
    assert gen.build_shapes(files, min_suffix_n=1) == [
        {"arms": [[-1, 0]], "suffix": 3, "n": 1, "frames": [[30, 1]]},
        {"arms": [[0, -1], [1, 0]], "suffix": 8, "n": 1, "frames": [[80, 1]]},
        {"arms": [[0, 1]], "suffix": 2, "n": 1, "frames": [[20, 1]]},
    ]
    assert gen.build_shapes(files, min_suffix_n=2) == []


def test_shape_rows_ignore_diagonal_neighbours() -> None:
    """A 3x3 corner's two 3x3 flanks touch each other diagonally. The exact
    dirset keys of `entries` drop all three nodes for that; the shape rows
    key on cardinal arms alone, so the corner is still a {N, E} row."""
    gen = conftest.load_verify_module("gen_cliff_connectivity")
    boxes = [((10, 13, 10, 13), 3), ((10, 13, 7, 10), 1), ((13, 16, 10, 13), 4)]
    nodes = [gen.ScannedNode(suffix=1, rotation=r, bounds=b, span=(3, 3)) for b, r in boxes]
    adjacency = [
        [(j, d) for j, other in enumerate(nodes) if j != i and (d := cliff_connectivity.adjacency_dir(n.bounds, other.bounds))]
        for i, n in enumerate(nodes)
    ]
    scanned = gen.ScannedFile(name="corner", nodes=nodes, adjacency=adjacency)
    assert scanned.dirset(1) == {(0, 1), (1, 1)}
    assert scanned.arms(1) == {(0, 1)}
    rows = gen.build_shapes([scanned], min_suffix_n=1)
    assert {"arms": [[0, -1], [1, 0]], "suffix": 1, "n": 1, "frames": [[3, 1]]} in rows
    assert len(rows) == 3


def test_one_stroke_classes_are_the_twenty_typed_arm_sets() -> None:
    gen = conftest.load_verify_module("gen_cliff_connectivity")
    classes = gen.one_stroke_classes()
    assert len(classes) == 20 and len({tuple(sorted(c.items())) for c in classes}) == 20
    # A corner's two dressings: {N, E} and {S, W} match, {N, W} and {S, E} differ.
    assert {(0, -1): True, (1, 0): True} in classes and {(0, -1): False, (1, 0): False} in classes
    assert {(0, -1): True, (-1, 0): False} in classes and {(0, -1): False, (-1, 0): True} in classes
    assert {(0, -1): True, (-1, 0): True} not in classes
    assert {(0, 1): True, (1, 0): True} not in classes


def test_shipped_table_carries_the_offset_projections_and_no_retired_field() -> None:
    data = cliff_connectivity._data()
    assert set(data) == {
        "_comment", "metadata", "entries", "projection_metadata", "shape_metadata", "offsets", "offsets_by_span", "shapes",
    }
    assert set(data["projection_metadata"]) == {
        "offset_pairs", "offset_modal_share", "offset_by_span_pairs", "offset_by_span_modal_share",
    }
    # 3x3 beside 3x3 is always flush and aligned, the layout every all-3x3 corner needs.
    spans = {(tuple(o["a"]), tuple(o["d"]), tuple(o["b"])): o["offsets"] for o in data["offsets_by_span"]}
    assert spans[((3, 3), (1, 0), (3, 3))][0][0] == [3, 0]
    assert spans[((3, 3), (0, -1), (3, 3))][0][0] == [0, -3]


N, S, W, E = (0, -1), (0, 1), (-1, 0), (1, 0)
CAP_S, CAP_N, RUN_NS = frozenset({S}), frozenset({N}), frozenset({N, S})


def _js(dirset):
    return sorted(list(d) for d in dirset)


# -- the facing model: resolve_shape() and rewrite_frames() ------------------

T, t = True, False
ALL_SUFFIXES = range(1, 10)

# The measured shape table (GH #44 round 2), typed in from the design table
# and never read back off the resolver: arms with each leg tall (T) or thin
# (t) -> (suffix, frames). Two frames are a straight's art twins. Where two
# suffixes share a footprint (s9/s8, s5/s4, s6/s7) the commoner is listed.
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


def _arms_id(arms) -> str:
    names = {N: "N", S: "S", W: "W", E: "E"}
    return "".join(f"{names[d]}{'T' if tall else 't'}" for d, tall in arms.items())


def test_the_shape_table_has_one_cell_per_frame() -> None:
    """The 20 classes use each of the 24 frames exactly once."""
    assert sorted(f for _, _, frames in SHAPE_TABLE for f in frames) == list(range(24))


@pytest.mark.parametrize(("arms", "suffix", "frames"), SHAPE_TABLE, ids=[_arms_id(a) for a, _, _ in SHAPE_TABLE])
def test_resolve_shape_returns_the_measured_table(arms, suffix, frames) -> None:
    assert cliff_connectivity.resolve_shape(arms, allowed_suffixes=ALL_SUFFIXES, frame_limit=24) == (suffix, frames)


def test_leg_and_edge_types() -> None:
    """A leg is tall when the piece is 3 tiles across that arm; an edge is
    tall when travelled east or south with the face on side 1."""
    assert cliff_connectivity.DRAG_FACE_SIDE == 1
    assert [cliff_connectivity.edge_tall(d, 1) for d in (E, S, W, N)] == [T, T, t, t]
    assert [cliff_connectivity.edge_tall(d, -1) for d in (E, S, W, N)] == [t, t, T, T]
    # s2 is 1x3: thin along its run, tall only nominally on its broadside.
    assert [cliff_connectivity.leg_tall(2, d) for d in (N, S, W, E)] == [t, t, T, T]
    assert [cliff_connectivity.leg_tall(3, d) for d in (N, S, W, E)] == [T, T, t, t]
    assert all(cliff_connectivity.leg_tall(1, d) for d in (N, S, W, E))
    assert not any(cliff_connectivity.leg_tall(8, d) for d in (N, S, W, E))
    # 2x3 and 3x2 are the mixed corners' pieces.
    assert [cliff_connectivity.leg_tall(4, d) for d in (N, W)] == [t, T]
    assert [cliff_connectivity.leg_tall(6, d) for d in (N, W)] == [T, t]


def test_nominal_span_is_every_cliff_const_s_real_span_but_one() -> None:
    """Leg types read NOMINAL_SPAN by suffix. Short Marble Cliff 1 (2190) is
    the one piece whose real footprint differs, 3x2 for a suffix-1 3x3."""
    from descape import cliff_catalog, unit_sprites
    from descape.terrain_palette import tile_span

    suffixes = cliff_catalog.suffixes()
    odd = {
        const: tile_span(const, (1, 1))
        for const in unit_sprites.cliff_consts()
        if tile_span(const, (1, 1)) != cliff_connectivity.NOMINAL_SPAN[suffixes[const]]
    }
    assert odd == {2190: (3, 2)}
    assert cliff_connectivity.NOMINAL_SPAN == {
        1: (3, 3), 2: (1, 3), 3: (3, 1), 4: (2, 3), 5: (2, 3), 6: (3, 2), 7: (3, 2), 8: (2, 2), 9: (2, 2)
    }


def test_twin_frames_are_read_off_the_straight_rows() -> None:
    assert cliff_connectivity.shape_tables().twins == {1: 2, 2: 1, 4: 5, 5: 4, 7: 8, 8: 7, 10: 11, 11: 10}


def _shape_doc(*extra, straights=((1, 2), (7, 8), (4, 5), (10, 11))):
    """A hand-written `shapes` document: the four straight rows a ShapeTables
    needs for its twin map, plus `extra` rows (arms, suffix, [[frame, n], ...])."""
    ns_tall, ns_thin, we_tall, we_thin = straights
    rows = [
        ((N, S), 1, [[ns_tall[0], 300], [ns_tall[1], 200]]),
        ((N, S), 2, [[ns_thin[0], 300], [ns_thin[1], 200]]),
        ((W, E), 1, [[we_tall[0], 300], [we_tall[1], 200]]),
        ((W, E), 3, [[we_thin[0], 300], [we_thin[1], 200]]),
        *extra,
    ]
    return {
        "shapes": [
            {"arms": _js(arms), "suffix": suffix, "n": sum(n for _, n in frames), "frames": frames}
            for arms, suffix, frames in rows
        ]
    }


def test_a_table_whose_straights_are_not_the_four_twin_pairs_does_not_build() -> None:
    assert cliff_connectivity.ShapeTables(_shape_doc()).twins[10] == 11
    with pytest.raises(ValueError, match="twin frames"):
        cliff_connectivity.ShapeTables(_shape_doc(straights=((1, 2), (7, 8), (4, 5), (10, 12))))
    with pytest.raises(ValueError, match="no straight row"):
        cliff_connectivity.ShapeTables({"shapes": _shape_doc()["shapes"][:3]})


def test_resolve_shape_takes_the_commonest_row_over_the_floor_and_ties_low() -> None:
    corner = (N, E)
    tables = cliff_connectivity.ShapeTables(
        _shape_doc(
            (corner, 8, [[13, 150]]), (corner, 9, [[13, 150]]), ((S, W), 9, [[9, 99]]), ((S, W), 8, [[9, 100]]),
            ((N, W), 4, [[6, 120]]), ((N, W), 5, [[6, 180]]),
        )
    )

    def resolve(arms, allowed=ALL_SUFFIXES, limit=24):
        return cliff_connectivity.resolve_shape(arms, allowed_suffixes=allowed, frame_limit=limit, tables=tables)

    assert resolve({N: t, W: T}) == (5, (6,))  # the commoner of two twin suffixes
    assert resolve({N: t, E: t}) == (8, (13,))  # a tie goes to the lower suffix
    assert resolve({N: t, E: t}, allowed={9}) == (9, (13,))
    assert resolve({S: t, W: t}) == (8, (9,))  # s9's 99 is under SHAPE_FLOOR
    assert resolve({S: t, W: t}, allowed={9}) is None
    assert resolve({N: T, E: T}) is None  # no row with those leg types
    assert resolve({N: t, W: t}) is None  # no row with that arm set


def test_resolve_shape_filters_frame_24_the_family_limit_and_cap_frames() -> None:
    """Frame 24 is stored in the corpus but no family has 25 frames; Marble
    has 23, so its thin north cap (frame 23) has nothing; and a two-arm row
    topped by a cap frame is two runs abutting, not a corner."""
    tables = cliff_connectivity.ShapeTables(
        _shape_doc(
            ((N, E), 8, [[24, 150], [13, 140]]), ((S, W), 1, [[17, 110], [15, 20]]), ((N,), 2, [[23, 165], [7, 17]]),
            ((E,), 3, [[24, 150], [20, 140]]),
        )
    )

    def resolve(arms, limit=24):
        return cliff_connectivity.resolve_shape(arms, allowed_suffixes=ALL_SUFFIXES, frame_limit=limit, tables=tables)

    assert resolve({N: t, E: t}) is None
    assert resolve({E: t}) is None
    assert resolve({E: t}, limit=25) is None
    assert resolve({S: T, W: T}) is None
    assert resolve({N: t}) == (2, (23,))  # a cap frame is right on a one-arm row
    assert resolve({N: t}, limit=23) is None
    assert resolve({N: T, S: T}, limit=2) == (1, (1,))  # the twin past the limit is dropped
    # No install: frame_count() falls back to the .dat's 25, which must not admit 24.
    assert resolve({N: t, E: t}, limit=25) is None
    shipped = cliff_connectivity.resolve_shape
    assert shipped({N: t}, allowed_suffixes=ALL_SUFFIXES, frame_limit=23) is None
    assert shipped({S: t}, allowed_suffixes=ALL_SUFFIXES, frame_limit=23) == (2, (22,))


def test_resolve_shape_honours_the_family_s_own_pieces() -> None:
    """CLF01..08 has no 05, 07 or 09: the twin suffix that exists is used."""
    clf = {1, 2, 3, 4, 6, 8}
    assert cliff_connectivity.resolve_shape({N: t, W: T}, allowed_suffixes=clf, frame_limit=24) == (4, (6,))
    assert cliff_connectivity.resolve_shape({S: t, W: t}, allowed_suffixes=clf, frame_limit=24) == (8, (9,))
    assert cliff_connectivity.resolve_shape({E: t, S: T}, allowed_suffixes={7}, frame_limit=24) == (7, (0,))


def test_rewrite_frames_gives_a_cap_its_joined_frame_at_its_own_size() -> None:
    def rewrite(arms, suffix, limit=24):
        return cliff_connectivity.rewrite_frames(frozenset(arms), suffix, frame_limit=limit)

    # An end cap that gains the arm opposite: the straight's twins.
    assert rewrite({N, S}, 1) == (1, 2)
    assert rewrite({W, E}, 1) == (4, 5)
    assert rewrite({N, S}, 2) == (7, 8)
    assert rewrite({W, E}, 3) == (10, 11)
    # A tall cap that gains a side arm: the two all-3x3 corners exist.
    assert rewrite({E, N}, 1) == (3,)
    assert rewrite({S, W}, 1) == (15,)
    # The other two do not: that join needs a different-size piece.
    assert rewrite({N, W}, 1) is None
    assert rewrite({E, S}, 1) is None
    # Nor does a thin cap turn a corner at its own size.
    assert rewrite({S, E}, 2) is None
    assert rewrite({N, W}, 3) is None
    # A lone piece that gains its first arm becomes that cap.
    assert rewrite({W}, 1) == (17,)
    assert rewrite({S}, 2) == (22,)
    assert rewrite({E}, 2) is None
    # A twin suffix that is not its class's commonest still has its own row.
    assert rewrite({S, W}, 8) == (9,)
    # An all-tall straight with a side neighbour keeps its straight frames.
    assert rewrite({N, S, E}, 1) == (1, 2)
    assert rewrite({N, S}, 1, limit=2) == (1,)


def test_shipped_table_carries_the_shape_rows_and_their_fit() -> None:
    data = cliff_connectivity._data()
    rows = {(frozenset(tuple(d) for d in r["arms"]), r["suffix"]): r for r in data["shapes"]}
    assert all(1 <= len(arms) <= 3 for arms, _ in rows)
    assert all(r["n"] == sum(n for _, n in r["frames"]) >= data["metadata"]["min_suffix_n"] for r in rows.values())
    # The all-3x3 corner the exact dirset keys never reached.
    assert rows[(frozenset({N, E}), 1)]["frames"][0][0] == 3
    assert rows[(frozenset({S, W}), 1)]["frames"][0][0] == 15
    shape = data["shape_metadata"]
    assert shape["shape_floor"] == cliff_connectivity.SHAPE_FLOOR == 100
    assert (shape["classes"], shape["classes_at_floor"]) == (24, 24)
    assert shape["leg_agreement"] >= 0.97
    assert shape["corner_one_sided"] >= 0.95
    assert shape["on_table"] >= 0.90


def test_offset_reads_the_full_key_then_the_span_backoff() -> None:
    """The offsets live on the same tables class as the shape rows."""
    doc = _shape_doc()
    doc["offsets"] = [
        {"a": [_js(CAP_S), [1, 3]], "d": list(S), "b": [_js(CAP_N), [2, 2]], "offsets": [[[0, 3], 9], [[-1, 3], 2]]}
    ]
    doc["offsets_by_span"] = [
        {"a": [1, 3], "d": list(S), "b": [2, 2], "offsets": [[[-1, 3], 50]]},
        {"a": [3, 3], "d": list(E), "b": [3, 1], "offsets": [[[3, 1], 4]]},
    ]
    t = cliff_connectivity.ShapeTables(doc)
    assert cliff_connectivity.offset(CAP_S, (1, 3), S, CAP_N, (2, 2), tables=t) == (0, 3)
    assert cliff_connectivity.offset(RUN_NS, (1, 3), S, CAP_N, (2, 2), tables=t) == (-1, 3)
    assert cliff_connectivity.offset(RUN_NS, (3, 3), E, CAP_N, (3, 1), tables=t) == (3, 1)
    assert cliff_connectivity.offset(RUN_NS, (3, 3), W, CAP_N, (3, 1), tables=t) is None
    # A document with no offsets at all (the generator's own fit pass) still builds.
    assert cliff_connectivity.ShapeTables(_shape_doc()).offsets == {}


def test_the_retired_resolver_is_gone() -> None:
    """The statistical walk the facing lookup replaced: nothing may still reach for it."""
    for name in ("resolve_chain", "ChainTables", "chain_tables", "DOMINANCE_FLOOR"):
        assert not hasattr(cliff_connectivity, name), name


@pytest.mark.corpus
def test_regenerates_identically_from_the_private_corpus() -> None:
    """Drift guard in the test_config_example.py mould -- skipped, not
    failed, when the private maintainer corpus this table was measured from
    isn't present on disk (it's a separate gitignored repo, not guaranteed
    to exist for every checkout)."""
    if not PRIVATE_CORPUS.is_dir():
        pytest.skip(f"private corpus not present at {PRIVATE_CORPUS}")

    gen = conftest.load_verify_module("gen_cliff_connectivity")
    doc = json.loads(json.dumps(gen.build_document(gen.scan([ROOT / "examples", PRIVATE_CORPUS]), min_n=30)))

    # The facing model's own fit, with its pre-registered floors: a miss means
    # the model is wrong for this corpus, not that a floor needs tuning.
    shape = doc["shape_metadata"]
    assert shape["edge_ends_agreeing"] / shape["edge_ends"] >= 0.97, shape
    assert shape["corners_one_sided"] / shape["corners"] >= 0.95, shape
    assert shape["drag_nodes_on_table"] / shape["drag_nodes"] >= 0.90, shape
    assert (shape["classes"], shape["classes_at_floor"]) == (24, 24), shape

    shipped = cliff_connectivity._data()
    assert doc["metadata"] == shipped["metadata"], (
        "descape/cliff_connectivity.json is stale -- re-run "
        "tools/gen_cliff_connectivity.py against examples/ and the private corpus"
    )
    for field in (
        "entries",
        "projection_metadata",
        "shape_metadata",
        "offsets",
        "offsets_by_span",
        "shapes",
    ):
        assert doc[field] == shipped[field], field
    assert set(doc) == set(shipped)


def _held_out(name: str) -> bool:
    return int(hashlib.sha1(name.encode(), usedforsecurity=False).hexdigest(), 16) % 2 == 1


def _drag_chains(scanned, drag_shapes) -> list[list[tuple[int, tuple[int, int] | None]]]:
    """Maximal paths over drag-shape nodes as [(node, step to the next)], the
    last step None. A node whose neighbours are not all on the path is not a
    drag node."""
    n = len(scanned.nodes)
    dirsets = [scanned.dirset(i) for i in range(n)]
    ok = [
        scanned.nodes[i].suffix is not None and dirsets[i] in drag_shapes and len(scanned.adjacency[i]) == len(dirsets[i])
        for i in range(n)
    ]
    nbrs = {i: [(j, d) for j, d in scanned.adjacency[i] if ok[j]] for i in range(n) if ok[i]}
    seen: set[int] = set()
    chains = []
    for start in sorted(nbrs, key=lambda k: (len(nbrs[k]) > 1, k)):
        if start in seen:
            continue
        path, prev, cur = [], None, start
        while True:
            seen.add(cur)
            nxt = [(j, d) for j, d in nbrs[cur] if j != prev and j not in seen]
            if not nxt:
                path.append((cur, None))
                break
            j, d = min(nxt)
            path.append((cur, d))
            prev, cur = cur, j
        chains.append(path)
    return chains


@pytest.mark.corpus
def test_held_out_cliffs_sit_on_the_shape_table_and_lay_out_by_the_modal_offset() -> None:
    """The held-out gates, with their pre-registered floors. Corpus files
    split by sha1 parity of their names; tables come from the build half only
    and are scored on the other. (i) At least 85% of held-out one- and
    two-arm nodes sit on their class's resolved piece and frame: the facing
    model is not fitted to the files it is scored on. (ii) At least 90% of
    consecutive pairs along held-out drag chains (maximal paths of caps,
    straights and L corners) sit at offset()'s modal offset for their key.
    (iii) At least 99% of the edges of a held-out all-3x3 corner do, keyed on
    cardinal arms as the drag asks: the corners the old keys never reached."""
    if not PRIVATE_CORPUS.is_dir():
        pytest.skip(f"private corpus not present at {PRIVATE_CORPUS}")
    gen = conftest.load_verify_module("gen_cliff_connectivity")
    drag_shapes = {d for d in gen.DRAG_DIRSETS if len(d) <= 2}
    files = gen.scan([ROOT / "examples", PRIVATE_CORPUS])
    held = [f for f in files if _held_out(f.name)]
    build_doc = gen.build_document([f for f in files if not _held_out(f.name)])
    tables = cliff_connectivity.ShapeTables(build_doc)

    score = gen.build_shape_metadata(held, build_doc["shapes"])
    assert score["drag_nodes"] > 1000
    assert score["drag_nodes_on_table"] / score["drag_nodes"] >= 0.85, f"(i) held-out shape score {score}"

    layout_hit = layout_n = 0
    for scanned in held:
        for chain in _drag_chains(scanned, drag_shapes):
            idx = [i for i, _ in chain]
            steps = [d for _, d in chain[:-1]]
            dirsets = [scanned.dirset(i) for i in idx]
            for k, d in enumerate(steps):
                a, b = scanned.nodes[idx[k]], scanned.nodes[idx[k + 1]]
                real = (b.bounds[0] - a.bounds[0], b.bounds[2] - a.bounds[2])
                layout_n += 1
                layout_hit += cliff_connectivity.offset(dirsets[k], a.span, d, dirsets[k + 1], b.span, tables=tables) == real
    assert layout_n > 1000
    assert layout_hit / layout_n >= 0.90, f"(ii) per-step layout {layout_hit}/{layout_n}"

    corner_hit, corner_n = gen.all_tall_corner_offsets(held, tables)
    assert corner_n > 1000
    assert corner_hit / corner_n >= 0.99, f"(iii) all-3x3 corner edges {corner_hit}/{corner_n}"
