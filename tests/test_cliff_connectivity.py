"""Track B Stage 2 blocking measurement (2026-09-02 cliffs plan). See
descape/cliff_connectivity.py's module docstring for the table's shape and
tools/gen_cliff_connectivity.py's for the measurement methodology and why
its purity figures revise the plan's own informal ~80% estimate downward.
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


def test_transition_and_offset_projections_on_a_synthetic_l() -> None:
    gen = conftest.load_verify_module("gen_cliff_connectivity")
    files = [_synthetic_l(gen)]
    transitions, rotation_transitions = gen.build_transitions(files)
    cap_s, corner, cap_e = [[[0, 1]], 2], [[[0, -1], [1, 0]], 8], [[[-1, 0]], 3]
    assert transitions == [
        {"a": cap_e, "d": [-1, 0], "b": corner, "n": 1},
        {"a": corner, "d": [0, -1], "b": cap_s, "n": 1},
        {"a": corner, "d": [1, 0], "b": cap_e, "n": 1},
        {"a": cap_s, "d": [0, 1], "b": corner, "n": 1},
    ]
    assert {"a": [[[0, 1]], 2, 20], "d": [0, 1], "b": [[[0, -1], [1, 0]], 8, 80], "n": 1} in rotation_transitions
    assert len(rotation_transitions) == 4

    offsets, by_span, stats = gen.build_offsets(files)
    assert {"a": [[[0, 1]], [1, 3]], "d": [0, 1], "b": [[[0, -1], [1, 0]], [2, 2]], "offsets": [[[0, 3], 1]]} in offsets
    assert {"a": [[[0, -1], [1, 0]], [2, 2]], "d": [1, 0], "b": [[[-1, 0]], [3, 1]], "offsets": [[[2, 0], 1]]} in offsets
    assert len(offsets) == 4 and len(by_span) == 4
    assert stats["offset_modal_share"] == 1.0

    # The dirset projection still sees the lone node (empty dirset) as a record.
    counts = gen.dirset_counts(files)
    assert counts[frozenset()] == {(1, 10): 1}


def test_shipped_table_carries_the_transition_projections() -> None:
    data = cliff_connectivity._data()
    assert data["projection_metadata"]["transition_pairs"] == sum(t["n"] for t in data["transitions"])
    # Straights keep their size: a {N,S} s2 node's southern {N,S} neighbour is s2 far more often than s1.
    ns = [[0, -1], [0, 1]]
    counts = {t["b"][1]: t["n"] for t in data["transitions"] if t["a"] == [ns, 2] and t["d"] == [0, 1] and t["b"][0] == ns}
    assert counts[2] > 50 * counts.get(1, 1)


N, S, W, E = (0, -1), (0, 1), (-1, 0), (1, 0)
CAP_S, CAP_N, RUN_NS = frozenset({S}), frozenset({N}), frozenset({N, S})


def _js(dirset):
    return sorted(list(d) for d in dirset)


def _tables(node_n, transitions=(), rotation_transitions=(), offsets=(), offsets_by_span=(), top=None):
    """A ChainTables over a hand-written document. node_n: {(dirset, suffix): n};
    transitions: (dirset_a, sa, d, dirset_b, sb, n); rotation_transitions the
    same with a rotation after each suffix."""
    by_dirset: dict = {}
    for (dirset, suffix), n in node_n.items():
        rotations = (top or {}).get((dirset, suffix), [0, 1])
        by_dirset.setdefault(dirset, {})[str(suffix)] = {"rotations": rotations, "n": n}
    doc = {
        "metadata": {"min_suffix_n": 10},
        "entries": [{"dirset": _js(d), "rotations_by_suffix": v} for d, v in by_dirset.items()],
        "transitions": [{"a": [_js(a), sa], "d": list(d), "b": [_js(b), sb], "n": n} for a, sa, d, b, sb, n in transitions],
        "rotation_transitions": [
            {"a": [_js(a), sa, ra], "d": list(d), "b": [_js(b), sb, rb], "n": n} for a, sa, ra, d, b, sb, rb, n in rotation_transitions
        ],
        "offsets": [{"a": [_js(a), list(sa)], "d": list(d), "b": [_js(b), list(sb)], "offsets": o} for a, sa, d, b, sb, o in offsets],
        "offsets_by_span": [{"a": list(sa), "d": list(d), "b": list(sb), "offsets": o} for sa, d, sb, o in offsets_by_span],
    }
    return cliff_connectivity.ChainTables(doc)


def test_resolve_chain_follows_the_pair_not_each_node_s_majority() -> None:
    """Both caps alone would take s1 (100 vs 90). The corpus only ever shows
    an s2 south of either, so the joint score moves node 1 to s2 while node 0
    keeps its own prior's s1: a per-node majority would draw s1, s1."""
    t = _tables(
        {(CAP_S, 1): 100, (CAP_S, 2): 90, (CAP_N, 1): 100, (CAP_N, 2): 90},
        transitions=[(CAP_S, 1, S, CAP_N, 2, 50), (CAP_S, 2, S, CAP_N, 2, 50)],
    )
    out = cliff_connectivity.resolve_chain([CAP_S, CAP_N], [S], allowed_suffixes=range(1, 10), tables=t)
    assert [s for s, _, _ in out] == [1, 2]
    assert [r for _, _, r in out] == [True, True]


def test_resolve_chain_lets_a_fixed_contact_pick_the_size() -> None:
    node = {(RUN_NS, 1): 100, (RUN_NS, 2): 95}
    pair = [(RUN_NS, 2, S, RUN_NS, 2, 80), (RUN_NS, 2, S, RUN_NS, 1, 1)]
    t = _tables(node, transitions=pair)
    alone = cliff_connectivity.resolve_chain([RUN_NS], [], allowed_suffixes=range(1, 10), tables=t)
    north_s2 = [[(N, 2, RUN_NS)]]
    touching = cliff_connectivity.resolve_chain([RUN_NS], [], allowed_suffixes=range(1, 10), contacts=north_s2, tables=t)
    assert alone[0][0] == 1
    assert touching[0][0] == 2


def test_resolve_chain_ties_go_to_the_lower_suffix() -> None:
    t = _tables({(RUN_NS, 7): 40, (RUN_NS, 3): 40})
    assert cliff_connectivity.resolve_chain([RUN_NS], [], allowed_suffixes=range(1, 10), tables=t)[0][0] == 3


def test_resolve_chain_drops_dominated_and_disallowed_suffixes() -> None:
    """Q3's dominance floor: s2 at 9% of s1's count is no candidate, so with
    s1 disallowed nothing survives and the node falls back verbatim; that
    splits the run, and its neighbours resolve on their own."""
    t = _tables({(RUN_NS, 1): 100, (RUN_NS, 2): 9, (CAP_S, 2): 50, (CAP_N, 2): 50})
    out = cliff_connectivity.resolve_chain([CAP_S, RUN_NS, CAP_N], [S, S], allowed_suffixes={2}, tables=t)
    assert out[1] == (None, None, False)
    assert out[0][0] == 2 and out[2][0] == 2
    assert not out[0][2] and not out[2][2]
    assert cliff_connectivity.resolve_chain([RUN_NS], [], allowed_suffixes={1, 2}, tables=t)[0][0] == 1


def test_resolve_chain_flags_a_pair_below_the_floor_but_keeps_its_suffix() -> None:
    t = _tables({(CAP_S, 2): 50, (CAP_N, 2): 50}, transitions=[(CAP_S, 2, S, CAP_N, 2, 9)])
    out = cliff_connectivity.resolve_chain([CAP_S, CAP_N], [S], allowed_suffixes={2}, tables=t)
    assert [(s, ok) for s, _, ok in out] == [(2, False), (2, False)]
    t = _tables({(CAP_S, 2): 50, (CAP_N, 2): 50}, transitions=[(CAP_S, 2, S, CAP_N, 2, 10)])
    out = cliff_connectivity.resolve_chain([CAP_S, CAP_N], [S], allowed_suffixes={2}, tables=t)
    assert [(s, ok) for s, _, ok in out] == [(2, True), (2, True)]


def test_resolve_chain_rotations_follow_the_measured_pairs() -> None:
    """Node 1's own commonest rotation is 9, but node 0's (4) is only ever
    followed by 8 across this pair, so the run takes 4 then 8."""
    node = {(CAP_S, 2): 50, (CAP_N, 2): 50}
    t = _tables(
        node,
        transitions=[(CAP_S, 2, S, CAP_N, 2, 30), (CAP_N, 2, N, CAP_S, 2, 30)],
        rotation_transitions=[
            (CAP_S, 2, 4, S, CAP_N, 2, 8, 16),
            (CAP_S, 2, 5, S, CAP_N, 2, 9, 14),
            (CAP_N, 2, 8, N, CAP_S, 2, 4, 16),
            (CAP_N, 2, 9, N, CAP_S, 2, 5, 14),
            (CAP_N, 2, 9, E, CAP_S, 2, 0, 40),
        ],
    )
    out = cliff_connectivity.resolve_chain([CAP_S, CAP_N], [S], allowed_suffixes={2}, tables=t)
    assert [r for _, r, _ in out] == [4, 8]
    # An unattested pair falls back to inherit-the-face within the top two.
    t = _tables(node, top={(CAP_S, 2): [6, 3], (CAP_N, 2): [7, 3]}, rotation_transitions=[(CAP_S, 2, 3, E, CAP_S, 2, 3, 40)])
    out = cliff_connectivity.resolve_chain([CAP_S, CAP_N], [S], allowed_suffixes={2}, tables=t)
    assert [r for _, r, _ in out] == [3, 3]


def test_offset_reads_the_full_key_then_the_span_backoff() -> None:
    t = _tables(
        {},
        offsets=[(CAP_S, (1, 3), S, CAP_N, (2, 2), [[[0, 3], 9], [[-1, 3], 2]])],
        offsets_by_span=[((1, 3), S, (2, 2), [[[-1, 3], 50]]), ((3, 3), E, (3, 1), [[[3, 1], 4]])],
    )
    assert cliff_connectivity.offset(CAP_S, (1, 3), S, CAP_N, (2, 2), tables=t) == (0, 3)
    assert cliff_connectivity.offset(RUN_NS, (1, 3), S, CAP_N, (2, 2), tables=t) == (-1, 3)
    assert cliff_connectivity.offset(RUN_NS, (3, 3), E, CAP_N, (3, 1), tables=t) == (3, 1)
    assert cliff_connectivity.offset(RUN_NS, (3, 3), W, CAP_N, (3, 1), tables=t) is None


def test_the_memoized_conditional_is_keyed_on_its_support() -> None:
    """A drag re-resolves through one cached ChainTables; the same context
    normalized over a different candidate set must not reuse a stale value."""
    import math

    t = _tables({}, transitions=[(CAP_S, 1, S, CAP_N, 2, 30), (CAP_S, 1, S, CAP_N, 3, 30)])
    key = (CAP_S, 1, S, CAP_N)
    assert t.conditional(False, key, 2, (1, 2, 3)) == pytest.approx(math.log(30.5 / 61.5))
    assert t.conditional(False, key, 2, (1, 2)) == pytest.approx(math.log(30.5 / 31))


def test_resolve_chain_rejects_a_step_count_mismatch() -> None:
    with pytest.raises(ValueError, match="steps"):
        cliff_connectivity.resolve_chain([CAP_S, CAP_N], [], allowed_suffixes={2})


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

    shipped = cliff_connectivity._data()
    assert doc["metadata"] == shipped["metadata"], (
        "descape/cliff_connectivity.json is stale -- re-run "
        "tools/gen_cliff_connectivity.py against examples/ and the private corpus"
    )
    for field in (
        "entries",
        "projection_metadata",
        "transitions",
        "rotation_transitions",
        "offsets",
        "offsets_by_span",
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
def test_held_out_chains_propagate_style_and_lay_out_by_the_modal_offset() -> None:
    """Gate C of the cliff-chain plan, with its pre-registered floors. Corpus
    files split by sha1 parity of their names; tables come from the build half
    only, and held-out chains (maximal paths of caps, straights and L corners)
    are replayed. (i) Seeding node 0 with its stored suffix, at least 80% of
    nodes 1..n of each 3+ chain resolve to their stored suffix: style carries
    through transitions. (ii) At least 90% of consecutive held-out pairs, on
    their stored suffixes, sit at offset()'s modal offset for their key."""
    if not PRIVATE_CORPUS.is_dir():
        pytest.skip(f"private corpus not present at {PRIVATE_CORPUS}")
    gen = conftest.load_verify_module("gen_cliff_connectivity")
    drag_shapes = {d for d in gen.DRAG_DIRSETS if len(d) <= 2}
    files = gen.scan([ROOT / "examples", PRIVATE_CORPUS])
    tables = cliff_connectivity.ChainTables(gen.build_document([f for f in files if not _held_out(f.name)]))
    seeded_hit = seeded_n = layout_hit = layout_n = 0
    for scanned in (f for f in files if _held_out(f.name)):
        for chain in _drag_chains(scanned, drag_shapes):
            idx = [i for i, _ in chain]
            steps = [d for _, d in chain[:-1]]
            dirsets = [scanned.dirset(i) for i in idx]
            stored = [scanned.nodes[i].suffix for i in idx]
            for k, d in enumerate(steps):
                a, b = scanned.nodes[idx[k]], scanned.nodes[idx[k + 1]]
                real = (b.bounds[0] - a.bounds[0], b.bounds[2] - a.bounds[2])
                layout_n += 1
                layout_hit += cliff_connectivity.offset(dirsets[k], a.span, d, dirsets[k + 1], b.span, tables=tables) == real
            if len(chain) < 3:
                continue
            back = (-steps[0][0], -steps[0][1])
            contacts = [[(back, stored[0], dirsets[0])]] + [[] for _ in idx[2:]]
            seeded = cliff_connectivity.resolve_chain(
                dirsets[1:], steps[1:], allowed_suffixes=range(1, 10), contacts=contacts, tables=tables
            )
            seeded_n += len(seeded)
            seeded_hit += sum(s == want for (s, _, _), want in zip(seeded, stored[1:], strict=True))
    assert seeded_n > 1000 and layout_n > 1000
    assert seeded_hit / seeded_n >= 0.80, f"(i) seeded propagation {seeded_hit}/{seeded_n}"
    assert layout_hit / layout_n >= 0.90, f"(ii) per-step layout {layout_hit}/{layout_n}"
