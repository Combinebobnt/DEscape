#!/usr/bin/env python3
"""Backtests tools/structure_infer.py against versions whose true structure is known.

What it is for: the inference tool's first real run is the next game patch,
so its trust comes from hiding versions the library already knows and
checking what it infers for them. tests/test_structure_backtest.py drives it.

- `compare_files()` classifies an inferred structure against the true one,
  by value, field by field: each true field must be read from the same
  bytes (offset and size, every instance) under the inferred structure, by
  a column at the same path prefix that may stand for it: the same name, a
  new `inferred_v...` field (so `inferred_v1_59_0` matches `capture_flag`),
  or a base name that the base-to-truth diff renames to it, one to one
  (`rename_pairs()`). A base field read from another true field's bytes is
  a misread even where the values agree.
  EXACT: every value equal. SHAPE: the same bytes, but read with another
  type, merged into a raw run, or under another layout signature (a fixed
  repeat where the truth's is dependent). WRONG: any true field read from
  other bytes or by another field.
- `classify()` turns an InferResult into EXACT, SHAPE, AMBIGUOUS (the truth
  among the kept layout and its rivals), REFUSED or WRONG (an unambiguous
  result that misreads, or an ambiguous set without the truth).
- `history_edits()` diffs two library structures into edits, each in or out
  of the search's vocabulary, so the test can keep that list honest.
- `simulated_reference()` stands in for a re-save's old file: the truth's
  values at the base structure's field paths.
- `write_side()` loads a file through DEscape with the inferred structure as
  a repo structure (the library's own definition hidden, its trigger
  vocabulary copied beside it) and checks, against the true library
  structure: every unit, condition and effect field reads the same; a save
  with every unit dirty rewrites the body identically, and one with every
  trigger dirty too matches the same save made under the truth; distinctive
  edits after each 1.59 insert point (a capture flag, a condition's last two
  4-byte fields) plus one unit east and a 3x3 terrain corner read back as
  exactly those edits. Each half runs in its own interpreter.

Read-only on every input; write_side() writes only into the folder it is given.
"""

from __future__ import annotations

import argparse
import difflib
import json
import shutil
import subprocess
import sys
import time
from bisect import bisect_right
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from descape import structure_walk as sw
from tools import structure_infer as si

ROOT = Path(__file__).resolve().parent.parent

EXACT = "EXACT"
SHAPE = "SHAPE"
AMBIGUOUS = "AMBIGUOUS"
REFUSED = "REFUSED"
WRONG = "WRONG"

# The search's edit vocabulary in history_edits() kinds; everything else refuses by design.
IN_VOCABULARY = frozenset({"insert_fixed", "insert_4x", "insert_str", "delete", "size_change", "to_str", "rename", "retype"})


# --- comparing parses -------------------------------------------------------


@dataclass
class Comparison:
    """How one inferred structure reads the files, against the truth."""

    verdict: str  # EXACT, SHAPE or WRONG
    misread: list[tuple[str, ...]] = field(default_factory=list)  # true paths read from other bytes
    reshaped: list[tuple[str, ...]] = field(default_factory=list)  # same bytes, another type or a raw merge
    names: dict[tuple[str, ...], tuple[str, ...]] = field(default_factory=dict)  # inferred path -> true path
    files: int = 0
    structure: Mapping[str, Any] | None = None  # the layout judged, which `names` refers to


def _norm(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, (list, tuple)):
        return tuple(_norm(v) for v in value)
    if isinstance(value, float) and value != value:
        return "nan"  # NaN != NaN, yet the same bytes read the same
    return value


def _same_values(a: sw.Column, b: sw.Column) -> bool:
    return [_norm(v) for v in a.value_list()] == [_norm(v) for v in b.value_list()]


Renames = Mapping[tuple[str, ...], frozenset[tuple[str, str]]]


def rename_pairs(base: Mapping[str, Any], truth: Mapping[str, Any]) -> dict[tuple[str, ...], frozenset[tuple[str, str]]]:
    """Each truth field-path prefix mapped to its block's (base name, true name) renames: within
    each replaced run of names, paired from the front, then from the back, while fixed sizes match."""
    blocks_a, blocks_b = _blocks(base), _blocks(truth)
    by_block = si.block_prefixes(sw.compile_structure(truth))
    out: dict[tuple[str, ...], frozenset[tuple[str, str]]] = {}
    for path, block_b in blocks_b.items():
        if path not in blocks_a or path not in by_block:
            continue
        block_a = blocks_a[path]
        ra, rb = block_a["retrievers"], block_b["retrievers"]
        na, nb = list(ra), list(rb)
        sa = [_fixed_size(ra[x], block_a) for x in na]
        sb = [_fixed_size(rb[x], block_b) for x in nb]
        pairs: set[tuple[str, str]] = set()
        for op, a1, a2, b1, b2 in difflib.SequenceMatcher(a=na, b=nb, autojunk=False).get_opcodes():
            if op != "replace":
                continue
            i1, i2, j1, j2 = a1, a2, b1, b2
            while i1 < i2 and j1 < j2 and sa[i1] is not None and sa[i1] == sb[j1]:
                pairs.add((na[i1], nb[j1]))
                i1, j1 = i1 + 1, j1 + 1
            while i1 < i2 and j1 < j2 and sa[i2 - 1] is not None and sa[i2 - 1] == sb[j2 - 1]:
                pairs.add((na[i2 - 1], nb[j2 - 1]))
                i2, j2 = i2 - 1, j2 - 1
        if pairs:
            for prefix in by_block[path][1]:
                out[prefix] = frozenset(pairs)
    return out


def _may_frame(inferred: tuple[str, ...], truth: tuple[str, ...], renames: Renames) -> bool:
    """Whether the inferred column `inferred` may stand for the true field `truth`: at the same
    prefix, the same name, a new inferred field, or a base name the history renames to it."""
    if inferred[:-1] != truth[:-1]:
        return False
    name = inferred[-1]
    return name == truth[-1] or name.startswith(si.INFERRED_PREFIX) or (name, truth[-1]) in renames.get(truth[:-1], ())


def compare_walks(truth: sw.Walk, inferred: sw.Walk, out: Comparison, renames: Renames) -> None:
    """Adds `truth`'s fields that `inferred` (the same buffer) misreads or reshapes to `out`.
    A base field read from a true field's bytes is a misread: the bytes match, the meaning doesn't."""
    by_start: dict[tuple[int, int], list[tuple[tuple[str, ...], sw.Column]]] = {}
    starts: list[tuple[int, int, tuple[str, ...], sw.Column]] = []
    for path, col in inferred.columns.items():
        if len(col):
            by_start.setdefault((col.offsets[0], col.sizes[0]), []).append((path, col))
            starts.append((col.offsets[0], col.sizes[0], path, col))
    starts.sort(key=lambda t: (t[0], -t[1]))
    keys = [s[0] for s in starts]
    for path, col in truth.columns.items():
        if not any(col.sizes):
            continue  # no bytes, no evidence (1.41's __END_OF_FILE_MARK__ reads 0 at the end)
        framed = [(p, c) for p, c in by_start.get((col.offsets[0], col.sizes[0]), []) if c.offsets == col.offsets and c.sizes == col.sizes]
        named = [(p, c) for p, c in framed if _may_frame(p, path, renames)]
        exact = next((p for p, c in named if _same_values(col, c)), None)
        if exact is not None:
            out.names.setdefault(exact, path)
            continue
        if named:
            out.reshaped.append(path)
            continue
        if framed:
            out.misread.append(path)
            continue
        # A raw run covering the field in every instance reads its bytes, unnamed.
        i = bisect_right(keys, col.offsets[0]) - 1
        while i >= 0 and starts[i][0] + starts[i][1] < col.offsets[0] + col.sizes[0]:
            i -= 1
        cover = starts[i] if i >= 0 else None
        if cover is not None and isinstance(cover[3].value(0), bytes) and _covers(cover[3], col) and _may_frame(cover[2], path, renames):
            out.reshaped.append(path)
        else:
            out.misread.append(path)


def _covers(raw: sw.Column, col: sw.Column) -> bool:
    if len(raw) != len(col):
        return False
    return all(ro <= o and o + s <= ro + rs for ro, rs, o, s in zip(raw.offsets, raw.sizes, col.offsets, col.sizes, strict=True))


def _signatures(structure: Mapping[str, Any]) -> dict[tuple[str, ...], tuple]:
    compiled = sw.compile_structure(structure)
    return {block: si.layout_signature(si.block_dict(structure, block)["retrievers"]) for block in si.block_prefixes(compiled)}


def compare_files(inferred: Mapping[str, Any], truth: Mapping[str, Any], raws: Sequence[bytes], base: Mapping[str, Any]) -> Comparison:
    """`inferred` against `truth` on every file the truth walks; `base` is the structure the
    search started from, whose renames into the truth may frame a true field."""
    out = Comparison(EXACT, structure=inferred)
    t_compiled, i_compiled = sw.compile_structure(truth), sw.compile_structure(inferred)
    renames = rename_pairs(base, truth)
    for raw in raws:
        try:
            t = sw.walk_file(t_compiled, raw)
        except (sw.WalkError, sw.StructureError):
            continue  # the truth can't read it either: no evidence
        out.files += 1
        try:
            i = sw.walk_file(i_compiled, raw)
        except (sw.WalkError, sw.StructureError) as e:
            out.misread.append(("<walk>", str(e)))
            continue
        compare_walks(t.header, i.header, out, renames)
        compare_walks(t.body, i.body, out, renames)
    out.misread = sorted(set(out.misread))
    out.reshaped = sorted(set(out.reshaped))
    if out.misread:
        out.verdict = WRONG
    elif out.reshaped or _signatures(inferred) != _signatures(truth):
        out.verdict = SHAPE
    return out


# --- classifying a result ---------------------------------------------------


@dataclass
class Outcome:
    kind: str
    status: str
    detail: str = ""
    comparison: Comparison | None = None
    seconds: float = 0.0
    result: si.InferResult | None = None


def rebuild(base: Mapping[str, Any], edits: Sequence[si.Edit]) -> dict:
    structure = dict(base)
    for edit in edits:
        structure = si.apply_edit(structure, edit)
    return structure


def _swappable(edits: Sequence[si.Edit], i: int, rival: si.Edit) -> bool:
    """Whether `rival` can replace edits[i] without staling a later edit: the same block and
    shape, and every later edit in that block sits past both positions. A rival it skips can only
    turn AMBIGUOUS into a false WRONG, never hide a misread."""
    kept = edits[i]
    if (rival.block, rival.count, len(rival.fields)) != (kept.block, kept.count, len(kept.fields)):
        return False
    past = max(kept.index, rival.index) + max(kept.count, len(kept.fields))
    return all(later.index >= past for later in edits[i + 1 :] if later.block == kept.block)


def _resolve_ambiguity(base: dict, result: si.InferResult, truth: Mapping[str, Any], raws: Sequence[bytes]) -> Comparison:
    """The best layout among each ambiguous edit's same-shape rivals, one record at a time
    (edits in different blocks don't interact), judged on every file."""
    edits = [r.edit for r in result.edits]
    best = compare_files(rebuild(base, edits), truth, raws, base)
    for i, record in enumerate(result.edits):
        if not record.ambiguous or best.verdict != WRONG:
            continue
        for rival in record.rivals:
            if not _swappable(edits, i, rival):
                continue
            trial = [*edits[:i], rival, *edits[i + 1 :]]
            try:
                c = compare_files(rebuild(base, trial), truth, raws, base)
            except ValueError:
                continue
            if len(c.misread) < len(best.misread):
                edits, best = trial, c
    return best


def classify(result: si.InferResult, truth: Mapping[str, Any], raws: Sequence[bytes], base: Mapping[str, Any]) -> Outcome:
    """EXACT/SHAPE for a decided result that reads like the truth, AMBIGUOUS when the truth is
    among the candidates, REFUSED, else WRONG. `base` is the structure the search started from."""
    if result.status == si.STATUS_REFUSED:
        return Outcome(REFUSED, result.status, result.refusal or "")
    comparison = compare_files(result.structure, truth, raws, base)
    ambiguous = result.status == si.STATUS_AMBIGUOUS or any(r.ambiguous for r in result.edits)
    if not ambiguous:
        detail = ", ".join("/".join(p) for p in comparison.misread[:4])
        return Outcome(comparison.verdict, result.status, detail, comparison)
    if comparison.verdict == WRONG:
        comparison = _resolve_ambiguity(dict(base), result, truth, raws)
    if comparison.verdict == WRONG:
        missed = ", ".join("/".join(p) for p in comparison.misread[:4])
        return Outcome(WRONG, result.status, f"the truth is not among the candidates: {missed}", comparison)
    return Outcome(AMBIGUOUS, result.status, "; ".join(result.unresolved), comparison)


def base_paths(structure: Mapping[str, Any]) -> set[tuple[str, ...]]:
    compiled = sw.compile_structure(structure)
    return {
        (*prefix, f.name)
        for spec, prefixes in si.block_prefixes(compiled).values()
        for prefix in prefixes
        for f in spec.fields
        if f.kind != "struct"
    }


def simulated_reference(truth: Mapping[str, Any], raw: bytes, base: Mapping[str, Any]) -> dict[tuple[str, ...], list]:
    """What a real old file would give at the base's field paths: the truth's values there."""
    walked = sw.walk_file(truth, raw)
    keep = base_paths(base)
    return {p: c.value_list() for w in (walked.header, walked.body) for p, c in w.columns.items() if p in keep}


def run(
    base_version: str,
    truth_version: str,
    samples: Sequence[Path],
    *,
    compare: Sequence[Path] = (),
    mode: str = "blind",
    terrain_ids: Collection[int],
    reference: tuple[Path, Path] | None = None,
) -> Outcome:
    """Infer `truth_version` from `base_version` on `samples` and classify it on samples plus `compare`.
    mode "blind", "simulated" (reference_values from samples[0]) or "reference" (a real pair)."""
    base = si.load_structure(base_version)
    truth = si.load_structure(truth_version)
    kwargs: dict[str, Any] = {}
    if mode == "simulated":
        kwargs["reference_values"] = simulated_reference(truth, Path(samples[0]).read_bytes(), base)
    elif mode == "reference":
        kwargs["reference"] = reference
    t0 = time.monotonic()
    result = si.infer((base_version, base), list(samples), terrain_ids=terrain_ids, library_check=None, **kwargs)
    raws = [Path(p).read_bytes() for p in dict.fromkeys([*samples, *compare])]
    outcome = classify(result, truth, raws, base)
    outcome.seconds = time.monotonic() - t0
    outcome.result = result
    return outcome


# --- the history of real bumps ----------------------------------------------


def _blocks(structure: Mapping[str, Any]) -> dict[tuple[str, ...], Mapping[str, Any]]:
    out: dict[tuple[str, ...], Mapping[str, Any]] = {}

    def visit(path: tuple[str, ...], block: Mapping[str, Any]) -> None:
        out[path] = block
        for name, sub in (block.get("structs") or {}).items():
            visit((*path, name), sub)

    for name, block in structure.items():
        visit((name,), block)
    return out


def _sig(name: str, r: Mapping[str, Any]) -> tuple:
    src = sw._repeat_source(name, r, "?")
    return (r["type"], r.get("repeat", 1), src[0] if src else None, r.get("is_list"))


def _fixed_size(r: Mapping[str, Any], block: Mapping[str, Any]) -> int | None:
    t = r["type"]
    if sw._repeat_source("x", r, "?"):
        return None
    repeat = r.get("repeat", 1)
    if t.startswith("struct:"):
        sub = block["structs"][t[7:]]
        sizes = [_fixed_size(rr, sub) for rr in sub["retrievers"].values()]
        return None if None in sizes else sum(sizes) * repeat
    if t.startswith("str"):
        return None
    letters = "".join(c for c in t if c.isalpha())
    bits = int("".join(c for c in t if c.isdigit()))
    return repeat * (bits // 8 if letters in ("u", "s", "f") else bits)


def _inserts(names: Sequence[str], block: Mapping[str, Any], rets: Mapping[str, Any]) -> list[tuple[str, str]]:
    """An inserted run split into single edits, (kind, detail)."""
    edits: list[tuple[str, str]] = []
    fixed: list[str] = []

    def flush() -> None:
        if fixed:
            size = sum(_fixed_size(rets[n], block) for n in fixed)
            four = all(rets[n]["type"] in ("s32", "u32", "f32") and rets[n].get("repeat", 1) == 1 for n in fixed)
            edits.append(("insert_4x" if four and len(fixed) > 1 else "insert_fixed", f"{'+'.join(fixed)} ({size} B)"))
            fixed.clear()

    for n in names:
        r = rets[n]
        src = sw._repeat_source(n, r, "?")
        if src:
            flush()
            edits.append(("insert_dependent", f"{n} {r['type']} repeat={src[0]}"))
        elif r["type"] in ("str16", "str32") and r.get("repeat", 1) == 1:
            flush()
            edits.append(("insert_str", f"{n} {r['type']}"))
        elif _fixed_size(r, block) is not None:
            fixed.append(n)
        else:
            flush()
            edits.append(("insert_other", f"{n} {r['type']} x{r.get('repeat', 1)}"))
    flush()
    return edits


def _delete(where: str, names: Sequence[str], rets: Mapping[str, Any], block: Mapping[str, Any], note: str = "") -> tuple[str, str, str]:
    """A deleted run as one edit, with its byte size when every field is fixed-size."""
    sizes = [_fixed_size(rets[n], block) for n in names]
    size = f" ({sum(sizes)} B{note})" if None not in sizes else (f" ({note.strip()})" if note else "")
    return (where, "delete" if len(names) <= 2 else "delete_run", ",".join(names) + size)


def history_edits(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """The edits from structure `a` to `b`, each (block, kind, detail); kind in IN_VOCABULARY or not."""
    blocks_a, blocks_b = _blocks(a), _blocks(b)
    edits: list[tuple[str, str, str]] = []
    for path, block_b in blocks_b.items():
        where = "/".join(path)
        if path not in blocks_a:
            if len(path) == 1:
                edits.append((where, "new_section", ""))
            continue
        ra, rb = blocks_a[path]["retrievers"], block_b["retrievers"]
        na, nb = list(ra), list(rb)
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=na, b=nb, autojunk=False).get_opcodes():
            if op == "equal":
                for x in na[i1:i2]:
                    sa, sb = _sig(x, ra[x]), _sig(x, rb[x])
                    if sa == sb:
                        continue
                    fa, fb = _fixed_size(ra[x], blocks_a[path]), _fixed_size(rb[x], block_b)
                    if sa[2] != sb[2]:
                        edits.append((where, "repeat_rule_change", f"{x}: {sa} -> {sb}"))
                    elif fa is not None and fb is not None:
                        edits.append((where, "size_change" if fa != fb else "retype", f"{x}: {sa[:2]} -> {sb[:2]} ({fa}->{fb} B)"))
                    elif fa is not None and sb[0] in ("str16", "str32") and sb[1] == 1:
                        edits.append((where, "to_str", f"{x}: {sa[:2]} -> {sb[:2]}"))
                    else:
                        edits.append((where, "type_change", f"{x}: {sa} -> {sb}"))
            elif op == "insert":
                edits.extend((where, k, d) for k, d in _inserts(nb[j1:j2], block_b, rb))
            elif op == "delete":
                edits.append(_delete(where, na[i1:i2], ra, blocks_a[path]))
            else:
                fa = [_fixed_size(ra[x], blocks_a[path]) for x in na[i1:i2]]
                fb = [_fixed_size(rb[x], block_b) for x in nb[j1:j2]]
                plain = all(_sig(x, ra[x])[2] is None for x in na[i1:i2]) and all(_sig(x, rb[x])[2] is None for x in nb[j1:j2])
                if None not in fa and None not in fb and sum(fa) == sum(fb) and plain:
                    edits.append((where, "rename", ",".join(na[i1:i2]) + " -> " + ",".join(nb[j1:j2])))
                    continue
                edits.append(_delete(where, na[i1:i2], ra, blocks_a[path], " replaced"))
                edits.extend((where, k, d) for k, d in _inserts(nb[j1:j2], block_b, rb))
    edits.extend(("/".join(path), "section_removed", "") for path in blocks_a if path not in blocks_b and len(path) == 1)
    return edits


_SEGMENT_OF = {"FileHeader": si.SEG_HEADER, "Units": si.SEG_UNITS, "Triggers": si.SEG_TRIGGERS, "Files": si.SEG_TRIGGERS}


def _size_delta(kind: str, detail: str) -> int | None:
    if kind in ("insert_fixed", "insert_4x"):
        return int(detail.rsplit("(", 1)[1].split(maxsplit=1)[0])
    if kind == "size_change":
        a, b = detail.rsplit("(", 1)[1].split(maxsplit=1)[0].split("->")
        return int(b) - int(a)
    if kind in ("delete", "delete_run") and detail.endswith(")") and " B" in detail.rsplit("(", 1)[1]:
        return -int(detail.rsplit("(", 1)[1].split(maxsplit=1)[0])
    return None


def cancelling_segments(edits: Sequence[tuple[str, str, str]]) -> list[str]:
    """Segments where several layout edits' size changes sum to zero: an anchor that
    trusts the segment's length would see no change there (review #9)."""
    by_segment: dict[str, list[int | None]] = {}
    for block, kind, detail in edits:
        if kind not in ("rename", "retype"):
            by_segment.setdefault(_SEGMENT_OF.get(block.split("/")[0], si.SEG_PREGRID), []).append(_size_delta(kind, detail))
    return sorted(s for s, deltas in by_segment.items() if len(deltas) > 1 and None not in deltas and sum(deltas) == 0)


def library_versions() -> list[str]:
    from descape import library_compat

    found = [d.name[1:] for d in library_compat.VERSIONS_DIR.glob("v*") if (d / "structure.json").is_file()]
    return sorted(found, key=si._version_key)


def history() -> dict[str, list[tuple[str, str, str]]]:
    """Every consecutive pair of library versions, "a->b", mapped to its edits."""
    versions = library_versions()
    return {f"{a}->{b}": history_edits(si.load_structure(a), si.load_structure(b)) for a, b in pairwise(versions)}


# --- the write side ---------------------------------------------------------


def transplant_names(
    inferred: Mapping[str, Any], names: Mapping[tuple[str, ...], tuple[str, ...]], skipped: list[str] | None = None
) -> dict:
    """`inferred` with each inferred field renamed to the true field it reads exactly. A rename
    whose true name the block already holds is skipped and added to `skipped`."""
    out = json.loads(json.dumps(inferred))
    compiled = sw.compile_structure(out)
    by_prefix = {prefix: block for block, (_, prefixes) in si.block_prefixes(compiled).items() for prefix in prefixes}
    for path, true_path in names.items():
        name = path[-1]
        if not name.startswith(si.INFERRED_PREFIX) or path[:-1] not in by_prefix:
            continue
        block = si.block_dict(out, by_prefix[path[:-1]])
        if true_path[-1] in block["retrievers"]:
            if skipped is not None:
                skipped.append(f"{'/'.join(path)} -> {true_path[-1]}")
            continue
        block["retrievers"] = {true_path[-1] if n == name else n: d for n, d in block["retrievers"].items()}
    return out


def _hidden_library(tmp: Path, version: str) -> Path:
    """A copy of the library's version folders without `version`."""
    from descape import library_compat

    copy = tmp / "library_versions"
    top = library_compat.VERSIONS_DIR

    def hide(d: str, names: list[str]) -> list[str]:
        return [n for n in names if n == f"v{version}" and Path(d) == top]

    shutil.copytree(top, copy, ignore=hide)
    return copy


# Edits after each 1.59 insert point (capture_flag, allow_in_fog), so a misplaced field reads back wrong.
CONDITION_EDITS = {"object_type2": 5, "allow_in_fog": 7}


def _plain(value: Any) -> Any:
    """A retriever value as JSON can carry it and compare it."""
    if isinstance(value, (bytes, bytearray)):
        return "bytes:" + bytes(value).hex()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and value != value:
        return "nan"
    return value


def _retrievers(obj: Any) -> dict[str, Any]:
    return {n: _plain(r.data) for n, r in obj.retriever_map.items()}


def _unit_names(structure: Mapping[str, Any]) -> list[str]:
    units = structure["Units"]
    players = units["structs"][units["retrievers"]["players_units"]["type"].removeprefix("struct:")]
    return list(players["structs"][players["retrievers"]["units"]["type"].removeprefix("struct:")]["retrievers"])


def field_dump(loaded: Any) -> dict[str, list[list[Any]]]:
    """Every unit's, condition's and effect's fields by name, as DEscape read them at load
    (units off its fast path's Unit objects, triggers off the library's retrievers)."""
    from descape.scenario_io import parse_triggers

    names = _unit_names(loaded._scenario.structure)
    dump: dict[str, list[list[Any]]] = {
        "units": [[{n: _plain(getattr(u, n, "<missing>")) for n in names} for u in player] for player in loaded.unit_manager.units]
    }
    if parse_triggers(loaded) is None:
        dump["conditions"] = dump["effects"] = [["unparsed"]]
        return dump
    triggers = loaded._scenario.sections["Triggers"].retriever_map["trigger_data"].data or []
    for kind in ("condition", "effect"):
        dump[f"{kind}s"] = [[_retrievers(c) for c in (t.retriever_map[f"{kind}_data"].data or [])] for t in triggers]
    return dump


def dump_diff(a: Mapping[str, Any], b: Mapping[str, Any]) -> tuple[dict[tuple, tuple[Any, Any]], list[str]]:
    """Every (kind, i, j, name) whose value differs between two dumps of one file, and every
    structural mismatch (object counts, a name only one side has)."""
    diffs: dict[tuple, tuple[Any, Any]] = {}
    problems: set[str] = set()
    for kind in a.keys() | b.keys():
        la, lb = a.get(kind, []), b.get(kind, [])
        if [len(x) for x in la] != [len(x) for x in lb]:
            problems.add(f"{kind}: object counts differ")
            continue
        for i, (xa, xb) in enumerate(zip(la, lb, strict=True)):
            for j, (oa, ob) in enumerate(zip(xa, xb, strict=True)):
                if not isinstance(oa, dict) or not isinstance(ob, dict):
                    if oa != ob:
                        problems.add(f"{kind}: {oa} against {ob}")
                    continue
                problems.update(f"{kind}: field {n} on one side only" for n in oa.keys() ^ ob.keys())
                diffs.update({(kind, i, j, n): (oa[n], ob[n]) for n in oa.keys() & ob.keys() if oa[n] != ob[n]})
    return diffs, sorted(problems)


def _summarize(diffs: Mapping[tuple, Any], limit: int = 6) -> str:
    by_field: dict[str, int] = {}
    for kind, _, _, name in diffs:
        by_field[f"{kind}.{name}"] = by_field.get(f"{kind}.{name}", 0) + 1
    return ", ".join(f"{k} x{n}" for k, n in sorted(by_field.items())[:limit])


def dirty_save(loaded: Any, out: Path, *, with_triggers: bool) -> None:
    """Save `loaded` with every unit dirty, and with `with_triggers` every trigger too."""
    from descape.scenario_write import write_scenario
    from descape.trigger_model import TriggerEditModel
    from descape.unit_model import UnitEditModel

    units = UnitEditModel(loaded)
    for u in [u for player_units in loaded.unit_manager.units for u in player_units]:
        units.set_position(u, u.x, u.y, u.z)
    triggers = None
    if with_triggers:
        triggers = TriggerEditModel(loaded)
        for i in range(triggers.trigger_count):
            triggers.mark_dirty(i)
    write_scenario(loaded, out, triggers=triggers, backup=False, units=units)


def _save_worker(structure: Path, sample: Path, version: str, out: Path) -> dict:
    """Load `sample` with `structure` as the repo's, the library's own structure and vocabulary
    hidden but the vocabulary copied beside it; record every field read, then save dirty copies
    (every unit; every unit and trigger) and an edited one."""
    from descape import batch_api, library_compat
    from descape.edit_history import EditHistory
    from descape.scenario_write import write_scenario
    from descape.trigger_model import TriggerEditModel
    from descape.unit_model import UnitEditModel

    repo = out / "repo_versions" / f"v{version}"
    repo.mkdir(parents=True)
    shutil.copy(structure, repo / "structure.json")
    for name in ("conditions.json", "effects.json"):
        shutil.copy(library_compat.VERSIONS_DIR / f"v{version}" / name, repo / name)
    library_compat.VERSIONS_DIR = _hidden_library(out, version)
    library_compat.REPO_VERSIONS_DIR = repo.parent
    library_compat.load_vocabulary.cache_clear()

    loaded = batch_api.load(sample)
    record: dict[str, Any] = {"structure_source": loaded.structure_source, "version": loaded.scenario_version, "fields": field_dump(loaded)}
    dirty_save(loaded, out / "dirty_units.aoe2scenario", with_triggers=False)
    dirty_save(batch_api.load(sample), out / "dirty_all.aoe2scenario", with_triggers=True)

    loaded = batch_api.load(sample)
    units = UnitEditModel(loaded)
    every = [u for player_units in loaded.unit_manager.units for u in player_units]
    asked: dict[str, Any] = {}
    if every:
        moved = every[0]
        asked["moved"] = [moved.reference_id, moved.x]
        units.set_position(moved, moved.x + 1.0, moved.y, moved.z)
        if units.supports_capture_flag:
            flagged = every[len(every) // 2]
            value = 2 if flagged.capture_flag != 2 else 1
            asked["flagged"] = [flagged.reference_id, value]
            units.set_capture_flag(flagged, value)
    triggers = TriggerEditModel(loaded)
    found = next(((t, 0) for t, trig in enumerate(triggers.manager().triggers) if trig.conditions), None)
    if found is not None:
        condition = triggers.manager().triggers[found[0]].conditions[found[1]]
        current = {n: getattr(condition, n, None) for n in CONDITION_EDITS}
        edits = {n: v if current[n] != v else v + 1 for n, v in CONDITION_EDITS.items() if current[n] is not None}
        triggers.begin_trigger_edit([found[0]])
        for name, value in edits.items():
            setattr(condition, name, value)
        triggers.commit_trigger_edit("backtest", EditHistory())
        asked["condition"] = [*found, edits]
    mm = loaded.map_manager
    old = mm.get_tile(0, 0).terrain_id
    new = 2 if old != 2 else 1
    for x in range(3):
        for y in range(3):
            batch_api.set_terrain(mm.get_tile(x, y), new)
    asked["painted"] = new
    record["asked"] = asked
    write_scenario(loaded, out / "edited.aoe2scenario", triggers=triggers, backup=False, units=units)
    return record


def _unit_at(dump: Mapping[str, Any], ref: int) -> tuple[int, int]:
    return next((p, k) for p, us in enumerate(dump["units"]) for k, u in enumerate(us) if u["reference_id"] == ref)


def _read_worker(sample: Path, out: Path, record: Mapping[str, Any]) -> list[str]:
    """Re-read the sample and the outputs with the true library structure: the inferred load must
    have read every field the same, and the edited file must differ by exactly what was asked."""
    from descape import batch_api
    from testkit.zlib_runtime import scenario_parts

    problems = []
    original = batch_api.load(sample)
    truth = field_dump(original)
    diffs, shape = dump_diff(truth, record["fields"])
    problems += [f"the inferred load: {p}" for p in shape]
    if diffs:
        problems.append(f"the inferred load reads other values than the truth: {_summarize(diffs)}")
    if scenario_parts((out / "dirty_units.aoe2scenario").read_bytes())[1] != original.decompressed_body:
        problems.append("a save with every unit dirty does not rewrite the body identically")
    # Re-serializing triggers isn't byte-identical even under the true structure, so the oracle is that save.
    dirty_save(batch_api.load(sample), out / "dirty_all_true.aoe2scenario", with_triggers=True)
    bodies = [scenario_parts((out / f"dirty_all{s}.aoe2scenario").read_bytes())[1] for s in ("", "_true")]
    if bodies[0] != bodies[1]:
        problems.append("a save with every unit and trigger dirty differs from the same save under the true structure")
    edited = batch_api.load(out / "edited.aoe2scenario")
    if edited.structure_source != "library":
        problems.append(f"the true re-read came from {edited.structure_source}")
    asked = record["asked"]
    expect: dict[tuple, Any] = {}
    if "moved" in asked:
        p, k = _unit_at(truth, asked["moved"][0])
        expect[("units", p, k, "x")] = None  # checked below, as a float32
    if "flagged" in asked:
        p, k = _unit_at(truth, asked["flagged"][0])
        expect[("units", p, k, "capture_flag")] = asked["flagged"][1]
    if "condition" in asked:
        t, c, edits = asked["condition"]
        expect.update({("conditions", t, c, n): v for n, v in edits.items()})
    diffs, shape = dump_diff(truth, field_dump(edited))
    problems += [f"the edited file: {p}" for p in shape]
    unasked = {k: v for k, v in diffs.items() if k not in expect}
    if unasked:
        problems.append(f"the edited file changed more than was asked: {_summarize(unasked)}")
    for key, want in expect.items():
        got = diffs[key][1] if key in diffs else None
        if want is None:
            if got is None or abs(got - (diffs[key][0] + 1.0)) > 1e-4:
                problems.append(f"unit {asked['moved'][0]} x reads {got}, not one tile east")
        elif got != want:
            problems.append(f"{'/'.join(map(str, key))} reads {got}, not {want}")
    w, h = original.map_manager.map_width, original.map_manager.map_height
    tiles = [(x, y) for x in range(w) for y in range(h)]
    diff = {(x, y) for x, y in tiles if original.map_manager.get_tile(x, y).terrain_id != edited.map_manager.get_tile(x, y).terrain_id}
    corner = {(x, y) for x in range(3) for y in range(3) if original.map_manager.get_tile(x, y).terrain_id != asked["painted"]}
    if diff != corner or any(edited.map_manager.get_tile(x, y).terrain_id != asked["painted"] for x in range(3) for y in range(3)):
        problems.append(f"terrain changed at {sorted(diff)[:9]}, expected the 3x3 corner")
    return problems


def write_side(
    inferred: Mapping[str, Any], names: Mapping[tuple[str, ...], tuple[str, ...]], sample: Path, version: str, work: Path
) -> list[str]:
    """Every problem with DEscape loading, saving and editing `sample` through the inferred
    structure (true names transplanted), or [] when it all checks out."""
    skipped: list[str] = []
    structure = work / "structure.json"
    structure.write_text(json.dumps(transplant_names(inferred, names, skipped), indent=1), encoding="utf-8")
    problems = [f"no true name transplanted, the block already has it: {s}" for s in skipped]
    me = [sys.executable, str(Path(__file__).resolve())]
    save = subprocess.run(
        [*me, "_save", str(structure), str(sample), version, str(work)], capture_output=True, text=True, cwd=ROOT, check=False
    )
    if save.returncode != 0:
        return [*problems, f"loading or saving through the inferred structure failed: {save.stderr.strip().splitlines()[-1:]}"]
    record = json.loads((work / "record.json").read_text(encoding="utf-8"))
    if (record["structure_source"], record["version"]) != ("repo", version):
        problems.append(f"loaded from {record['structure_source']} as {record['version']}, not the repo structure")
    read = subprocess.run([*me, "_read", str(sample), str(work)], capture_output=True, text=True, cwd=ROOT, check=False)
    if read.returncode != 0:
        return [*problems, f"re-reading with the true structure failed: {read.stderr.strip().splitlines()[-1:]}"]
    return problems + json.loads(read.stdout.strip().splitlines()[-1])


# --- command line -----------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("history", help="every library version bump as edits, in or out of the vocabulary")
    save = sub.add_parser("_save", help=argparse.SUPPRESS)
    save.add_argument("structure", type=Path)
    save.add_argument("sample", type=Path)
    save.add_argument("version")
    save.add_argument("out", type=Path)
    read = sub.add_parser("_read", help=argparse.SUPPRESS)
    read.add_argument("sample", type=Path)
    read.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    if args.command == "_save":
        record = _save_worker(args.structure, args.sample, args.version, args.out)
        (args.out / "record.json").write_text(json.dumps(record), encoding="utf-8")
    elif args.command == "_read":
        record = json.loads((args.out / "record.json").read_text(encoding="utf-8"))
        print(json.dumps(_read_worker(args.sample, args.out, record)))
    else:
        for pair, edits in history().items():
            out = sorted({k for _, k, _ in edits if k not in IN_VOCABULARY})
            print(f"## {pair}: {len(edits)} edits, {'all in the vocabulary' if not out else 'OUT: ' + ', '.join(out)}")
            for block, kind, detail in edits:
                print(f"   {'in ' if kind in IN_VOCABULARY else 'OUT'} {block:<45s} {kind:<20s} {detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
