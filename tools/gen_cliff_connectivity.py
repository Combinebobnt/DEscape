#!/usr/bin/env python3
"""Regenerates descape/cliff_connectivity.json: which piece a cliff takes
from its neighbours, measured from real scenario placements (this is corpus
geometry, not .dat content, so it belongs here rather than in
gen_unit_render_data.py).

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

**Piece size is the facing, and that is what the shape projection measures**
(GH #44 round 2). A piece's leg on an arm is tall (front, the rock face
toward the camera) when its footprint is 3 tiles across that arm, else thin
(back, the low ridge). Both ends of a join agree on that, and a run keeps its
face on one side through every corner, so a run has two dressings and each
node's piece follows from its arms and that side. `shapes` counts (cardinal
arm set of 1-3 directions, suffix) with every frame seen there; raw counts,
the resolver applies its own floor. Diagonal neighbours are ignored for the
key: the two 3x3 flanks of a 3x3 corner touch each other diagonally, so an
exact dirset key never reaches the all-3x3 corners. `shape_metadata` reports
how well the model fits the same scan (`build_shape_metadata()`): leg-type
agreement over edge ends, one-sided corners, and how many one- and two-arm
nodes sit on their class's resolved piece and frame.

**The per-dirset majority table (`entries`, `metadata`) is the older
measurement, and its low purity is the same fact seen from the other side.**
For any given dirset TWO suffixes are both common (the plain vertical-run
dirset splits ~55%/45% between the 3x3 "01" piece and the 1x3 "02" piece):
they are the front-facing and the back-facing dressing of one line, which a
neighbour set alone cannot tell apart. That is why purity is 61% within the
pinned dirsets, where it was first read as a stylistic size choice. Within
one suffix a straight's two frames are art twins, in no fixed order along a
run. The table picks the single most-common suffix per dirset and keeps its
top-2 rotations; coverage is reported per entry (`suffix_fraction`,
`rotation_fraction`). Nothing is placed from it any more.

**Each entry also carries `rotations_by_suffix`**: the top-level rotations
are the MAJORITY suffix's, and rotation is strongly conditioned on suffix,
because a 1x3 footprint holds thin shapes and a 3x3 holds big ones. A
`{N, S}` run at suffix 1 uses rotation 1/2 (192px of decoded ink); the same
dirset at suffix 2 uses 8/7 (90px). A pair below `--min-suffix-n` is simply
absent.

**Offset projections.** `offsets` is where a neighbour's footprint sits
relative to this node's (low corner minus low corner), over every cardinal
edge between nodes whose dirset is a drag shape (cap, straight, L) or a T,
keyed on realized span rather than suffix, with a span-only backoff
`offsets_by_span` over every cardinal edge. `projection_metadata` reports
the offset tables' modal share.

Every projection reads the one `scan()`, so no field can disagree with
another on which nodes or neighbours exist.
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
from descape.cliff_connectivity import NOMINAL_SPAN, SHAPE_FLOOR, ShapeTables, edge_tall, leg_tall
from descape.cliff_connectivity import adjacency_dir as _adjacency_dir
from descape.cliff_connectivity import offset as _offset
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

# The dirsets the full-key offset projection keeps: every cardinal-only
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

    def arms(self, i: int) -> frozenset[Direction]:
        """The cardinal half of dirset(): the shape projection's key."""
        return frozenset(d for _, d in self.adjacency[i] if d in _CARDINALS)


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


def build_shapes(files: list[ScannedFile], min_suffix_n: int = 10) -> list[dict]:
    """`shapes`: one row per (cardinal arm set of 1-3 directions, suffix) at
    n >= `min_suffix_n`, with every frame seen there, commonest first. Raw
    counts: the resolver applies its own floor."""
    counts: dict[tuple, Counter[int]] = defaultdict(Counter)
    for f in files:
        for i, node in enumerate(f.nodes):
            arms = f.arms(i)
            if node.suffix is not None and 1 <= len(arms) <= 3:
                counts[(tuple(sorted(arms)), node.suffix)][node.rotation] += 1
    return [
        {
            "arms": [list(d) for d in arms],
            "suffix": suffix,
            "n": sum(frames.values()),
            "frames": [[frame, n] for frame, n in sorted(frames.items(), key=lambda kv: (-kv[1], kv[0]))],
        }
        for (arms, suffix), frames in sorted(counts.items())
        if sum(frames.values()) >= min_suffix_n
    ]


def one_stroke_classes() -> list[dict[Direction, bool]]:
    """The 20 typed arm sets one drag can ask for: 8 end caps, 4 straights
    and each of the 4 corners dressed for either side."""
    classes: list[dict[Direction, bool]] = []
    for a in _CARDINALS:
        classes += [{a: True}, {a: False}]
    for i, a in enumerate(_CARDINALS):
        for b in _CARDINALS[i + 1 :]:
            for side in (1, -1):
                # In through `a` (travelling away from it), out through `b`.
                classes.append({a: edge_tall((-a[0], -a[1]), side), b: edge_tall(b, side)})
    return classes


def build_shape_metadata(files: list[ScannedFile], shapes: list[dict]) -> dict:
    """The facing model's own fit, over the same scan. `leg_agreement`: both
    ends of a cardinal join have the same leg type. `corner_one_sided`: a
    two-arm corner's own legs are one of the two dressings a one-sided run
    gives it. `on_table`: a 1-2 arm node sits on its class's resolved piece
    (twin suffixes share a footprint and count together) at one of its
    frames. `classes`: the (class, frame) cells of the resolved table, and how
    many hold SHAPE_FLOOR nodes or more."""
    tables = ShapeTables({"shapes": shapes})
    everything = range(1, 10)
    valid = one_stroke_classes()
    ends = agree = 0
    for f in files:
        for i, row in enumerate(f.adjacency):
            for j, d in row:
                a, b = f.nodes[i].suffix, f.nodes[j].suffix
                if d in _CARDINALS and a is not None and b is not None:
                    ends += 1
                    agree += leg_tall(a, d) == leg_tall(b, (-d[0], -d[1]))
    corners = one_sided = drag_nodes = on_table = 0
    for f in files:
        for i, node in enumerate(f.nodes):
            arms = f.arms(i)
            if node.suffix is None or not 1 <= len(arms) <= 2:
                continue
            drag_nodes += 1
            typed = {d: leg_tall(node.suffix, d) for d in arms}
            if len(arms) == 2 and sum(d[0] for d in arms) != 0:
                corners += 1
                one_sided += typed in valid
            found = tables.resolve(typed, allowed_suffixes=everything, frame_limit=24) if typed in valid else None
            if found is not None:
                on_table += NOMINAL_SPAN[node.suffix] == NOMINAL_SPAN[found[0]] and node.rotation in found[1]
    row_frames = {(frozenset(tuple(d) for d in r["arms"]), r["suffix"]): dict(r["frames"]) for r in shapes}
    cells = []
    for typed in valid:
        found = tables.resolve(typed, allowed_suffixes=everything, frame_limit=24)
        if found is not None:
            cells += [row_frames[(frozenset(typed), found[0])][frame] for frame in found[1]]

    def share(part: int, whole: int) -> float:
        return round(part / whole, 4) if whole else 0.0

    return {
        "shape_floor": SHAPE_FLOOR,
        "edge_ends": ends,
        "edge_ends_agreeing": agree,
        "leg_agreement": share(agree, ends),
        "corners": corners,
        "corners_one_sided": one_sided,
        "corner_one_sided": share(one_sided, corners),
        "drag_nodes": drag_nodes,
        "drag_nodes_on_table": on_table,
        "on_table": share(on_table, drag_nodes),
        "classes": len(cells),
        "classes_at_floor": sum(1 for n in cells if n >= SHAPE_FLOOR),
        "smallest_class_n": min(cells, default=0),
    }


def all_tall_corner_offsets(files: list[ScannedFile], tables) -> tuple[int, int]:
    """(hits, edge ends) over every cardinal edge of an all-3x3 corner (s1 at
    frame 3 on {N, E} or frame 15 on {S, W}), seen from both ends: whether
    `cliff_connectivity.offset()` over `tables` lays that neighbour where the
    file has it. Keyed on cardinal arms, as the chain drag asks."""
    corners = {(frozenset({(0, -1), (1, 0)}), 3), (frozenset({(0, 1), (-1, 0)}), 15)}
    hits = ends = 0
    for f in files:
        for i, node in enumerate(f.nodes):
            if node.suffix != 1 or (f.arms(i), node.rotation) not in corners:
                continue
            for j, d in f.adjacency[i]:
                if d not in _CARDINALS:
                    continue
                other = f.nodes[j]
                real = (other.bounds[0] - node.bounds[0], other.bounds[2] - node.bounds[2])
                back = (-d[0], -d[1])
                ends += 2
                hits += _offset(f.arms(i), node.span, d, f.arms(j), other.span, tables=tables) == real
                hits += _offset(f.arms(j), other.span, back, f.arms(i), node.span, tables=tables) == (-real[0], -real[1])
    return hits, ends


def build_document(files: list[ScannedFile], min_n: int = 30, min_suffix_n: int = 10) -> dict:
    """The whole JSON document, as main() writes it and the drift test
    compares it."""
    entries, metadata = build_table(dirset_counts(files), min_n, min_suffix_n)
    offsets, offsets_by_span, offset_stats = build_offsets(files)
    shapes = build_shapes(files, min_suffix_n)
    return {
        "_comment": (
            "Cliff placements measured from real scenarios. The Cliff tool reads `shapes` (typed cardinal arms "
            "-> piece suffix and frames) and `offsets`/`offsets_by_span` (where a neighbour's footprint sits); "
            "`entries` is the older per-dirset majority table, kept as a measurement. "
            "See tools/gen_cliff_connectivity.py."
        ),
        "metadata": metadata,
        "entries": entries,
        "projection_metadata": offset_stats,
        "shape_metadata": build_shape_metadata(files, shapes),
        "offsets": offsets,
        "offsets_by_span": offsets_by_span,
        "shapes": shapes,
    }


def dump_document(doc: dict) -> str:
    """indent=2 for `entries` and the metadata blocks; the record lists hold
    one record per line, which keeps the file small."""
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
        f"Offset modal share: {proj['offset_modal_share']:.1%} by (dirset, span) over {proj['offset_pairs']} pairs, "
        f"{proj['offset_by_span_modal_share']:.1%} by span only over {proj['offset_by_span_pairs']}"
    )
    shape = doc["shape_metadata"]
    print(
        f"Shapes: {len(doc['shapes'])} rows; leg agreement {shape['leg_agreement']:.1%} over {shape['edge_ends']} "
        f"edge ends, one-sided corners {shape['corner_one_sided']:.1%} of {shape['corners']}, on the table "
        f"{shape['on_table']:.1%} of {shape['drag_nodes']} 1-2 arm nodes, {shape['classes_at_floor']}/"
        f"{shape['classes']} classes at n >= {shape['shape_floor']} (smallest {shape['smallest_class_n']})"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
