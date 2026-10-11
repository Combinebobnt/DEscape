#!/usr/bin/env python3
"""Regenerates descape/cliff_connectivity.json: the Track B Stage 2 blocking
measurement from the 2026-09-02 cliffs plan -- a `neighbour-direction-set ->
(suffix, rotation candidates)` table for cliffs, measured from real scenario
placements (this is corpus geometry, not .dat content, so it belongs here
rather than in gen_unit_render_data.py).

**Install-free, corpus-scan tool** in the scan_wall_rotation.py mould:
`--scenario-dir` defaults to `examples/` and may be repeated. The shipped
table pools `examples/` with a private, cliff-heavy corpus (288 scenario
files, 23,313 cliff placements) -- re-run it against any richer corpus to
reproduce or update.

**Adjacency is tile-block, not centre-distance.** Two cliffs are neighbours
when their `render.unit_tile_bounds()` footprints share an edge (cardinal)
or touch at exactly one corner (diagonal) -- the same approach
`scan_wall_rotation.py`'s `_neighbour_mask` uses for walls, generalised from
4 directions to 8 since cliff footprints vary in size (3x3 down to 1x3) and
a centre-to-centre angle fragments one true relationship across several
octants when the two neighbours have different spans. Verified this isn't
the source of the low purity below: an earlier centre-distance-and-angle
measurement gave the same accuracy figures within 1%.

**The dirset key is over the FULL neighbour set, any suffix** -- a "vertical
run interior" node is one whose only neighbours are north and south,
regardless of what piece those neighbours are. `suffix` in the table's
values is the trailing digit of `object_catalog.object_name()` (e.g. "Cliff
(Default) 04" -> 4), which is stable across all ten families -- confirmed by
decoding real frames: rotation index N is the same shape (matching ink
bbox size and hotspot, within a few px) across Default/Desert/Snow/
Limestone/Terrace at N in {0, 1, 3, 6, 9, 13}, so pooling rotation data
across families is safe. Only the suffix (which of the family's 9 unit_consts
was placed) varies the footprint span; rotation alone selects the frame.

**Measured purity is well below the plan's informal ~80% estimate, and
that's a real finding, not a bug in this tool.** Two experiments (see the
2026-09-05 Track B Stage 2 execution notes in the plan) isolated why:

1. For any given dirset, TWO different suffixes are both common (e.g. the
   plain vertical-run dirset splits ~55%/45% between the 3x3 "01" piece and
   the 1x3 "02" piece) -- which SIZE of piece to use is a real stylistic
   choice human map-makers make, not something neighbour geometry
   determines. This, not the rotation/face axis, is the dominant source of
   unpredictability.
2. Within one suffix, the residual is genuinely the "face" pair the plan
   named: restricted to strict run-interior nodes (dirset exactly {N,S} or
   {E,W}, single suffix throughout), runs are constant-rotation ~34% of the
   time and cleanly period-2 alternating ~36% of the time -- ~70% combined
   "the observed rotation is one of exactly two acceptable values," which
   IS close to the plan's estimate once suffix choice is factored out.

So the table below picks the single most-common suffix per dirset (a
canonicalising choice a placement tool is allowed to make, unlike a corpus
classifier) and keeps the top-2 rotations as the value -- "top-2" because a
face is a real either-or, not noise to average away. Coverage is reported
per entry (`suffix_fraction`, `rotation_fraction`) so a consumer can see how
confident any given lookup is instead of trusting a single blended number.

**Each entry also carries `rotations_by_suffix`, and a consumer that pins
one piece size must read that, not the top-level `rotations`.** Added
2026-09-05 for Track B Stage 2's drag, which pins one suffix per run: the
top-level rotations are the MAJORITY suffix's, and rotation turns out to be
strongly conditioned on suffix, because a 1x3 footprint holds thin shapes
and a 3x3 holds big ones. A `{N, S}` run at suffix 1 uses rotation 1/2
(192px of decoded ink); the same dirset at suffix 2 uses 8/7 (90px). Mixing
them drew a run as a thin ridge between tall end caps. Purely additive --
a different projection of the same `scan()` output, so every field above
and every metadata figure is unchanged. Splitting by suffix makes the
buckets sparser, so a pair below `--min-suffix-n` is simply absent and the
consumer must fall back to its own manual value rather than borrow the
majority suffix's rotations.

**Transition and offset projections** (added 2026-10-09 for the chain drag's
transition-conditioned walk). Which piece size a corner takes depends on the
size of the straight it meets, so per-node majorities compose into runs the
corpus never contains. `transitions` counts (dirset, suffix) -d-> (dirset,
suffix) over every cardinal edge between nodes whose dirset is a drag shape
(cap, straight, L) or a T, and `rotation_transitions` the same with
rotations; both are raw counts, the resolver applies its own floor.
`offsets` is where a neighbour's footprint sits relative to this node's
(low corner minus low corner), keyed on realized span rather than suffix,
with a span-only backoff `offsets_by_span` over every cardinal edge.
`projection_metadata` reports the offset tables' modal share. All of it is
the same `scan()` output as above, so the original fields are unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from descape import cliff_catalog, unit_sprites
from descape.cliff_connectivity import adjacency_dir as _adjacency_dir
from descape.render import NON_BUILDING_SPAN, unit_tile_bounds
from descape.scenario_io import load_map_and_units
from descape.terrain_palette import tile_span

Bounds = tuple[int, int, int, int]
Direction = tuple[int, int]


def _corpus_files(scenario_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in scenario_dir.iterdir()
        if p.suffix in (".aoe2scenario", ".scx2", ".scx", ".scn") or p.name == "play_Test"
    )


def _suffix_map() -> dict[int, int | None]:
    """unit_const -> the trailing digit of its object_catalog label (see
    module docstring for why this, not the raw unit_const, is the pooling
    key across families).

    The digit itself is read by `cliff_catalog.piece_suffix()` rather than
    by a regex kept here, so this table and the placement tool that consumes
    it can't drift apart on what a suffix is."""
    return {const: cliff_catalog.piece_suffix(const) for const in unit_sprites.cliff_consts()}


_CARDINALS: tuple[Direction, ...] = ((0, -1), (0, 1), (-1, 0), (1, 0))

# The dirsets the transition and offset projections keep: every cardinal-only
# neighbour set of 1-3 directions. Sizes 1 and 2 are the 10 drag shapes (4
# caps, 2 straights, 4 L corners); size 3 is the T rows a drag against an
# existing cliff produces.
DRAG_DIRSETS: frozenset[frozenset[Direction]] = frozenset(
    frozenset(c)
    for c in (
        [(a,) for a in _CARDINALS]
        + [(a, b) for i, a in enumerate(_CARDINALS) for b in _CARDINALS[i + 1 :]]
        + [tuple(d for d in _CARDINALS if d != skip) for skip in _CARDINALS]
    )
)


@dataclass(frozen=True)
class ScannedNode:
    # None for a cliff const with no catalog suffix: still a neighbour for
    # every other node's dirset, never a record of its own.
    suffix: int | None
    rotation: int
    bounds: Bounds
    # From tile_span(), never from bounds: unit_tile_bounds() clamps at the map edge.
    span: tuple[int, int]


@dataclass(frozen=True)
class ScannedFile:
    name: str
    nodes: list[ScannedNode]
    # adjacency[i] lists (j, direction from i to j) for every touching node j.
    adjacency: list[list[tuple[int, Direction]]]

    def dirset(self, i: int) -> frozenset[Direction]:
        return frozenset(d for _, d in self.adjacency[i])


def scan(scenario_dirs: list[Path]) -> list[ScannedFile]:
    """Every cliff placement in every scenario file under `scenario_dirs`,
    per file, with its tile-block adjacency. Every projection below reads
    this one scan, so a new field can't disagree with the shipped ones on
    which nodes or neighbours exist."""
    consts = unit_sprites.cliff_consts()
    suffix_of = _suffix_map()
    files: list[ScannedFile] = []

    for path in [p for d in scenario_dirs for p in _corpus_files(d)]:
        try:
            scenario = load_map_and_units(path)
        except Exception:
            continue
        mm = scenario.map_manager
        tile_w, tile_h = mm.map_width, mm.map_height

        nodes: list[ScannedNode] = []
        for units in scenario.unit_manager.units:
            for unit in units:
                if unit.unit_const not in consts:
                    continue
                bounds = unit_tile_bounds(unit, tile_w, tile_h)
                if bounds is not None:
                    nodes.append(
                        ScannedNode(
                            suffix=suffix_of.get(unit.unit_const),
                            rotation=int(float(unit.rotation)),
                            bounds=bounds,
                            span=tile_span(unit.unit_const, NON_BUILDING_SPAN),
                        )
                    )

        adjacency: list[list[tuple[int, Direction]]] = []
        for i, node in enumerate(nodes):
            row: list[tuple[int, Direction]] = []
            for j, other in enumerate(nodes):
                if i == j:
                    continue
                direction = _adjacency_dir(node.bounds, other.bounds)
                if direction is not None:
                    row.append((j, direction))
            adjacency.append(row)
        files.append(ScannedFile(name=path.name, nodes=nodes, adjacency=adjacency))

    return files


def dirset_counts(files: list[ScannedFile]) -> dict[frozenset[Direction], Counter[tuple[int, int]]]:
    """dirset -> Counter[(suffix, rotation)]: the projection the shipped
    `entries` and `metadata` are built from."""
    result: dict[frozenset[Direction], Counter[tuple[int, int]]] = defaultdict(Counter)
    for f in files:
        for i, node in enumerate(f.nodes):
            if node.suffix is None:
                continue
            result[f.dirset(i)][(node.suffix, node.rotation)] += 1
    return result


def _dirset_json(dirset: frozenset[Direction]) -> list[list[int]]:
    return sorted(list(d) for d in dirset)


def _drag_pairs(files: list[ScannedFile]):
    """(file, i, d, j) for every cardinal adjacency whose two ends both have
    a drag or T dirset, in both directions (each edge is seen from each end)."""
    for f in files:
        dirsets = [f.dirset(i) for i in range(len(f.nodes))]
        for i, row in enumerate(f.adjacency):
            if dirsets[i] not in DRAG_DIRSETS:
                continue
            for j, d in row:
                if d in _CARDINALS and dirsets[j] in DRAG_DIRSETS:
                    yield f, dirsets, i, d, j


def build_transitions(files: list[ScannedFile]) -> tuple[list[dict], list[dict]]:
    """`transitions` and `rotation_transitions`: raw counts of
    (dirset_i, suffix_i[, rotation_i]) -d-> (dirset_j, suffix_j[, rotation_j])
    over every cardinal edge between drag/T nodes. No count floor here: the
    resolver applies its own, and a floor at build time would turn rare but
    real pairs into zeros."""
    by_suffix: Counter = Counter()
    by_rotation: Counter = Counter()
    for f, dirsets, i, d, j in _drag_pairs(files):
        a, b = f.nodes[i], f.nodes[j]
        if a.suffix is None or b.suffix is None:
            continue
        ka, kb = tuple(sorted(dirsets[i])), tuple(sorted(dirsets[j]))
        by_suffix[(ka, a.suffix, d, kb, b.suffix)] += 1
        by_rotation[(ka, a.suffix, a.rotation, d, kb, b.suffix, b.rotation)] += 1

    transitions = [
        {"a": [[list(x) for x in ka], sa], "d": list(d), "b": [[list(x) for x in kb], sb], "n": n}
        for (ka, sa, d, kb, sb), n in sorted(by_suffix.items())
    ]
    rotation_transitions = [
        {"a": [[list(x) for x in ka], sa, ra], "d": list(d), "b": [[list(x) for x in kb], sb, rb], "n": n}
        for (ka, sa, ra, d, kb, sb, rb), n in sorted(by_rotation.items())
    ]
    return transitions, rotation_transitions


def _offset_records(counter: dict, full_key: bool) -> list[dict]:
    out = []
    for key, offsets in sorted(counter.items()):
        rows = sorted(offsets.items(), key=lambda kv: (-kv[1], kv[0]))
        if full_key:
            ka, span_a, d, kb, span_b = key
            a: list = [[list(x) for x in ka], list(span_a)]
            b: list = [[list(x) for x in kb], list(span_b)]
        else:
            span_a, d, span_b = key
            a, b = list(span_a), list(span_b)
        out.append({"a": a, "d": list(d), "b": b, "offsets": [[list(o), n] for o, n in rows]})
    return out


def _modal_share(counter: dict) -> float:
    total = sum(sum(c.values()) for c in counter.values())
    modal = sum(max(c.values()) for c in counter.values())
    return round(modal / total, 4) if total else 0.0


def build_offsets(files: list[ScannedFile]) -> tuple[list[dict], list[dict], dict]:
    """`offsets`: neighbour footprint low corner minus this node's, keyed
    (dirset_i, span_i, d, dirset_j, span_j) over drag/T edges. `offsets_by_span`
    is the span-only backoff (span_i, d, span_j) over EVERY cardinal edge, so
    a dirset the full key lacks still finds the geometry its spans imply.
    Keyed on realized span, never suffix: s4 and s5 are both 2x3 and the
    question is where the footprints touch, not which art they carry."""
    full: dict = defaultdict(Counter)
    for f, dirsets, i, d, j in _drag_pairs(files):
        a, b = f.nodes[i], f.nodes[j]
        offset = (b.bounds[0] - a.bounds[0], b.bounds[2] - a.bounds[2])
        full[(tuple(sorted(dirsets[i])), a.span, d, tuple(sorted(dirsets[j])), b.span)][offset] += 1
    span_only: dict = defaultdict(Counter)
    for f in files:
        for i, row in enumerate(f.adjacency):
            a = f.nodes[i]
            for j, d in row:
                if d not in _CARDINALS:
                    continue
                b = f.nodes[j]
                span_only[(a.span, d, b.span)][(b.bounds[0] - a.bounds[0], b.bounds[2] - a.bounds[2])] += 1
    stats = {
        "offset_pairs": sum(sum(c.values()) for c in full.values()),
        "offset_modal_share": _modal_share(full),
        "offset_by_span_pairs": sum(sum(c.values()) for c in span_only.values()),
        "offset_by_span_modal_share": _modal_share(span_only),
    }
    return _offset_records(full, True), _offset_records(span_only, False), stats


def build_document(files: list[ScannedFile], min_n: int = 30, min_suffix_n: int = 10) -> dict:
    """The whole JSON document, as main() writes it and the drift test
    compares it."""
    entries, metadata = build_table(dirset_counts(files), min_n, min_suffix_n)
    transitions, rotation_transitions = build_transitions(files)
    offsets, offsets_by_span, offset_stats = build_offsets(files)
    return {
        "_comment": "Cliff neighbour-direction-set -> (suffix, rotation candidates) -- see tools/gen_cliff_connectivity.py",
        "metadata": metadata,
        "entries": entries,
        "projection_metadata": {
            "transition_pairs": sum(t["n"] for t in transitions),
            **offset_stats,
        },
        "transitions": transitions,
        "rotation_transitions": rotation_transitions,
        "offsets": offsets,
        "offsets_by_span": offsets_by_span,
    }


def dump_document(doc: dict) -> str:
    """indent=2 like before for the original fields; the new record lists
    hold one record per line, which keeps the file a few hundred KB."""
    parts = []
    for key, value in doc.items():
        if isinstance(value, list) and key != "entries":
            rows = ",\n".join("    " + json.dumps(r, separators=(",", ":")) for r in value)
            text = "[\n" + rows + "\n  ]" if value else "[]"
        else:
            text = json.dumps(value, indent=2).replace("\n", "\n  ")
        parts.append(f"  {json.dumps(key)}: {text}")
    return "{\n" + ",\n".join(parts) + "\n}\n"


def build_table(
    dirset_to_result: dict[frozenset[Direction], Counter[tuple[int, int]]],
    min_n: int,
    min_suffix_n: int = 10,
) -> tuple[list[dict], dict]:
    total_all = sum(sum(c.values()) for c in dirset_to_result.values())
    entries = []
    covered_mass = 0
    covered_correct = 0

    for dirset, counter in sorted(dirset_to_result.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(counter.values())
        if n < min_n:
            continue
        by_suffix: dict[int, Counter[int]] = defaultdict(Counter)
        for (suffix, rotation), count in counter.items():
            by_suffix[suffix][rotation] += count
        top_suffix, suffix_n = max(
            ((s, sum(c.values())) for s, c in by_suffix.items()), key=lambda kv: kv[1]
        )
        top_rotations = [r for r, _ in by_suffix[top_suffix].most_common(2)]
        top_rotation_n = sum(c for r, c in by_suffix[top_suffix].items() if r in top_rotations)

        entries.append(
            {
                "dirset": sorted(list(d) for d in dirset),
                "suffix": top_suffix,
                "rotations": top_rotations,
                "n": n,
                "suffix_fraction": round(suffix_n / n, 4),
                "rotation_fraction": round(top_rotation_n / suffix_n, 4),
                # Rotation is strongly conditioned on suffix, and a consumer
                # that pins one piece SIZE for a whole run (Track B Stage 2's
                # drag, per the plan's Decisions addendum) needs the
                # rotations measured for THAT size, not for whichever size
                # happened to be this dirset's majority. Measured: a {N, S}
                # run at suffix 1 uses rotation 2 (192px of ink), the same
                # dirset at suffix 2 uses 8/7 (90px) -- a 1x3 footprint
                # holds thin shapes and a 3x3 holds big ones, so borrowing
                # across suffixes draws a wrong-scale piece. Additive: the
                # majority-suffix fields above are untouched, and this is a
                # different projection of the same scan, not new data.
                "rotations_by_suffix": {
                    str(suffix): {
                        "rotations": [r for r, _ in rotations.most_common(2)],
                        "n": sum(rotations.values()),
                    }
                    for suffix, rotations in sorted(by_suffix.items())
                    if sum(rotations.values()) >= min_suffix_n
                },
            }
        )
        covered_mass += n
        covered_correct += top_rotation_n

    metadata = {
        "total_records": total_all,
        "min_n": min_n,
        "min_suffix_n": min_suffix_n,
        "covered_records": covered_mass,
        "coverage": round(covered_mass / total_all, 4) if total_all else 0.0,
        "purity_within_coverage": round(covered_correct / covered_mass, 4) if covered_mass else 0.0,
        "purity_over_all_records": round(covered_correct / total_all, 4) if total_all else 0.0,
    }
    return entries, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--scenario-dir",
        type=Path,
        action="append",
        default=None,
        help="May be repeated to pool several corpora; defaults to examples/ alone",
    )
    parser.add_argument("--min-n", type=int, default=30, help="Minimum record count to pin a dirset entry")
    parser.add_argument(
        "--min-suffix-n",
        type=int,
        default=10,
        help="Minimum record count to pin one (dirset, suffix) pair's own rotations. Lower than "
        "--min-n because splitting by suffix makes every bucket sparser; a pair below it is simply "
        "absent, and a consumer must fall back rather than borrow another suffix's rotations",
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent.parent / "descape" / "cliff_connectivity.json"
    )
    args = parser.parse_args()

    scenario_dirs = args.scenario_dir or [Path(__file__).resolve().parent.parent / "examples"]
    for scenario_dir in scenario_dirs:
        if not scenario_dir.is_dir():
            print(f"No such scenario directory: {scenario_dir}", file=sys.stderr)
            sys.exit(1)

    doc = build_document(scan(scenario_dirs), args.min_n, args.min_suffix_n)
    args.output.write_text(dump_document(doc))

    metadata, entries, proj = doc["metadata"], doc["entries"], doc["projection_metadata"]
    print(f"Scanned {scenario_dirs}: {metadata['total_records']} cliff records")
    print(f"Pinned {len(entries)} dirsets at n >= {args.min_n}, covering {metadata['coverage']:.1%} of records")
    print(
        f"Purity within that coverage: {metadata['purity_within_coverage']:.1%} "
        f"({metadata['purity_over_all_records']:.1%} of all records)"
    )
    print(
        f"Transitions: {len(doc['transitions'])} keys over {proj['transition_pairs']} drag/T edge ends, "
        f"{len(doc['rotation_transitions'])} with rotations"
    )
    print(
        f"Offset modal share: {proj['offset_modal_share']:.1%} by (dirset, span) over {proj['offset_pairs']} pairs, "
        f"{proj['offset_by_span_modal_share']:.1%} by span only over {proj['offset_by_span_pairs']}"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
