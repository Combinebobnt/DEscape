"""Track B Stage 2 blocking measurement (2026-09-02 cliffs plan). See
descape/cliff_connectivity.py's module docstring for the table's shape and
tools/gen_cliff_connectivity.py's for the measurement methodology and why
its purity figures revise the plan's own informal ~80% estimate downward.
"""

from __future__ import annotations

from pathlib import Path

import conftest
import pytest

from descape import cliff_connectivity

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
    # The single best-populated dirset in the measured corpus (n=2466): a
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


@pytest.mark.corpus
def test_regenerates_identically_from_the_private_corpus() -> None:
    """Drift guard in the test_config_example.py mould -- skipped, not
    failed, when the private maintainer corpus this table was measured from
    isn't present on disk (it's a separate gitignored repo, not guaranteed
    to exist for every checkout)."""
    if not PRIVATE_CORPUS.is_dir():
        pytest.skip(f"private corpus not present at {PRIVATE_CORPUS}")

    gen = conftest.load_verify_module("gen_cliff_connectivity")
    dirset_to_result = gen.scan([ROOT / "examples", PRIVATE_CORPUS])
    entries, metadata = gen.build_table(dirset_to_result, min_n=30)

    shipped = cliff_connectivity._data()
    assert metadata == shipped["metadata"], (
        "descape/cliff_connectivity.json is stale -- re-run "
        "tools/gen_cliff_connectivity.py against examples/ and the private corpus"
    )
    assert entries == shipped["entries"]
