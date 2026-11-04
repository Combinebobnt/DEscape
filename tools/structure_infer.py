"""Infers a new scenario version's structure from game-saved files.

What it is for: when AoE2:DE patches the scenario format and no structure
definition exists for the new version yet, this finds the layout change by
search instead of by hand. tools/infer_structure.py is the command line; this
module is the search, importable for the backtests.

Inputs are game-saved files of the new version (samples), the newest known
structure (the base), and optionally a reference pair: a file of a known
version plus the patched game's re-save of it, whose known values pin down
where bytes moved. A simulated reference (`reference_values`, a field path to
values map) stands in for the old file in backtests.

How it works:

- descape/format_scan.py's landmarks cut each file into four segments, each
  anchored on its own: the FileHeader (ends where the deflate body starts),
  DataHeader through Map (map_width sits at the scanned terrain grid), Units
  (starts at the grid end, and a plausible trigger version follows it), and
  Triggers plus Files (ends at the end of the body).
- A segment that walks cleanly under the base, meets its anchor, passes the
  plausibility filters and shows no layout break against the reference is
  unchanged. Otherwise the earliest evidence (a walk failure, an implausible
  field, a missed anchor, a reference break) bounds the edit from above, the
  last distinctive value that still matches the reference bounds it from
  below, and every single edit in that span is tried. The best is applied and
  the loop repeats, one edit at a time, inside per-block and total budgets.
- Several fitting candidates are scored against the reference pairwise, over
  only the values the two parse differently. A winner needs MARGIN over every
  rival on at least PAIR_MIN such values; otherwise the result is AMBIGUOUS
  and the rivals are listed. When nothing fits, the candidate that gets
  furthest is taken only if, with a reference, the bytes after the old
  evidence match it again.
- Candidates that give the edited block the same byte layout (an s32 inserted
  anywhere in a run of s32s) share one walk: each member's parse is the
  representative's with the field names moved along the run.
- Edits outside the vocabulary (a conditional or dependent repeat, a
  presence-flagged struct, a new section) find no fitting candidate and
  refuse with the located evidence, the route back to authoring by hand.

Walking is descape/structure_walk.py's, never the library's. The library only
checks the final structure, one file per subprocess through
tools/map_structure_divergence.py's worker, because a library load mutates
class state (tests/README.md, "Process-global hazard").
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from descape import format_scan as fs
from descape import structure_walk as sw

STATUS_INFERRED = "INFERRED"
STATUS_AMBIGUOUS = "AMBIGUOUS"
STATUS_REFUSED = "REFUSED"

SEG_HEADER = "FileHeader"
SEG_PREGRID = "DataHeader..Map"
SEG_UNITS = "Units"
SEG_TRIGGERS = "Triggers+Files"

# Plausibility filters. Each held for every unit, effect and condition of a
# 144-file corpus (1.37 to 1.59, measured 2026-11-01), so a true layout never trips one.
UNIT_STATUS_RANGE = (0, 7)
UNIT_CONST_RANGE = (0, 4095)  # corpus 1..2708, the frozen re-save keeps a const 0; headroom above
UNIT_Z_RANGE = (0.0, float(fs.ELEVATION_MAX))
CAPTURE_FLAG_RANGE = (-1, 3)
CAPTION_STRING_ID_MIN = -1
# A trigger struct's count field (static_value_N, right after the type) equals its
# 4-byte scalar fields minus 2. Exact for every effect and condition, 1.37 to 1.59.
COUNT_FIELD_BIAS = 2
TRIGGER_BLOCKS = frozenset({"EffectStruct", "ConditionStruct"})
UNIT_PATH = ("Units", "players_units", "units")

FIXED_INSERT_TYPES = {1: "s8", 2: "s16", 4: "s32", 8: "8"}
# The largest one-block fixed insert in the 1.36 to 1.59 history: 1.40's initial_player_views.
RAW_INSERT_MAX = 128
# With no anchor delta, every raw size 1..RAW_INSERT_MAX is tried only this close to the evidence.
RAW_SWEEP_POSITIONS = 6
MAX_POSITIONS = 400

# A reference mismatch is a layout break when at least half (and 8) of the next
# 4096 compared values mismatch too. The game's own rewrites stay below that: on
# the frozen test-map re-save, a 235-tile terrain cluster and 548 effect quantities.
BREAK_WINDOW = 4096
BREAK_RATIO = 0.5
BREAK_MIN = 8
BREAK_SHORT = 16  # and half of the next 16, so an isolated rewrite just before a real shift isn't blamed
# A mismatch the stream realigns after is a value change, not a shift: within the next
# REALIGN_WITHIN values, REALIGN_RUN distinctive values match with no mismatch between them. On
# the frozen re-save, player_data_3[0] has 7 of its 17 values rewritten, then [1] matches again.
REALIGN_RUN = 4
REALIGN_WITHIN = 32
SHIFT_RUN = 4  # unless the mismatch opens this many distinctive mismatches in a row (player_data_3[0]: 2)
# A path mismatching in nearly every instance is a break too, at its first mismatch: a field read
# from its neighbour's bytes after a wrong insert (frozen re-save, allow_in_fog put before inverted:
# 476 of 482). Under the truth the frozen re-save's worst per-object path is 0.69 (the game sets
# capture_flag on 69% of units), next initial_animation_frame at 0.087; per-player paths (8 or 9
# instances, player_data_3's lists rewritten in all 8) stay under WHOLE_MIN.
WHOLE_MIN = 16
WHOLE_RATIO = 0.95
# An edit that gets further without fitting must make this many more distinctive values match.
PARTIAL_MIN = 2
# Scoring: a candidate wins when, against each rival, the rival's mismatch fraction over the
# values the two parse differently is at least MARGIN higher. Margin gate (frozen re-save, 1.58 to
# 1.59): the truth scores 0.0..0.052 (the game's rewrites), every wrong single edit 1.0.
MARGIN = 0.25
PAIR_MIN = 4  # fewer differing values than this decide nothing: one rewrite or chance match could flip them

INFERRED_PREFIX = "inferred_v"


class InferInputError(Exception):
    """An input can't be used at all: no sample, no base, an unreadable file."""


class _BudgetExceeded(Exception):
    pass


@dataclass(frozen=True)
class Budget:
    edits_per_block: int = 6
    edits_total: int = 24
    walk_reads: int = 1_000_000_000  # field reads over every candidate walk of the run


DEFAULT_BUDGET = Budget()


# --- edits ------------------------------------------------------------------


@dataclass
class Edit:
    """Replace `count` retrievers of `block` (section, then struct type names)
    starting at `index` with `fields` ((name, definition) pairs). An insert has
    count 0, a delete no fields, a retype keeps the replaced field's name."""

    block: tuple[str, ...]
    index: int
    count: int
    fields: list[tuple[str, dict]]
    kind: str
    at: str = ""  # the field the edit sits before, for reports

    def describe(self) -> str:
        where = "/".join(self.block)
        return f"{self.kind} at {where}[{self.index}] (before {self.at or 'the end'})"

    def new_names(self) -> list[str]:
        return [n for n, _ in self.fields if n.startswith(INFERRED_PREFIX)]


def block_dict(structure: Mapping[str, Any], block: Sequence[str]) -> dict:
    node = structure[block[0]]
    for name in block[1:]:
        node = node["structs"][name]
    return node


def apply_edit(structure: Mapping[str, Any], edit: Edit) -> dict:
    """A new structure dict with `edit` applied. Only the dicts along the
    block's path are copied; everything else is shared with `structure`."""
    out = dict(structure)
    node = out[edit.block[0]] = dict(out[edit.block[0]])
    for name in edit.block[1:]:
        structs = node["structs"] = dict(node["structs"])
        node = structs[name] = dict(structs[name])
    items = list(node["retrievers"].items())
    if edit.index > len(items) or edit.index + edit.count > len(items):
        raise ValueError(f"{edit.describe()}: block has {len(items)} retrievers")
    items[edit.index : edit.index + edit.count] = [(n, dict(d)) for n, d in edit.fields]
    names = [n for n, _ in items]
    if len(set(names)) != len(names):
        raise ValueError(f"{edit.describe()}: duplicate retriever name")
    node["retrievers"] = dict(items)
    return out


def _read_deps(definition: Mapping[str, Any], slot: Mapping[str, int]) -> str:
    deps = definition.get("dependencies") or {}
    if not deps:
        return ""

    def target(t: str) -> str:
        section, _, name = t.partition(":")
        return f"self:#{slot[name]}" if section == "self" and name in slot else t

    mapped = {}
    for key, value in deps.items():
        items = value if isinstance(value, list) else [value]
        out = []
        for item in items:
            dep = dict(item)
            raw = dep.get("target")
            if isinstance(raw, str):
                dep["target"] = target(raw)
            elif isinstance(raw, list):
                dep["target"] = [target(t) for t in raw]
            out.append(dep)
        mapped[key] = out
    return json.dumps(mapped, sort_keys=True)


def layout_signature(retrievers: Mapping[str, Mapping[str, Any]]) -> tuple:
    """A block's byte layout with the names taken out: two blocks with the
    same signature read the same bytes field for field, so one walk serves
    both."""
    slot = {n: i for i, n in enumerate(retrievers)}
    return tuple(
        (d.get("type"), d.get("repeat", 1), d.get("is_list"), _read_deps(d, slot), n == sw.EOF_MARK) for n, d in retrievers.items()
    )


# --- reference values -------------------------------------------------------


def _as_np(values: Sequence[Any]) -> np.ndarray | list:
    if isinstance(values, np.ndarray):
        return values
    if values and all(type(v) is int for v in values):
        try:
            return np.asarray(values, dtype=np.int64)
        except OverflowError:
            return list(values)
    if values and all(type(v) is float for v in values):
        return np.asarray(values, dtype=np.float64)
    return list(values)


def _column_np(col: sw.Column) -> np.ndarray | list:
    values = col.values
    if values is None:
        return []
    if not isinstance(values, list):
        return np.frombuffer(values, dtype=np.int64 if values.typecode == "q" else np.float64)
    return _as_np(values)


class Reference:
    """Known values to compare a walk against, keyed by field path (section,
    then retriever names). Units pair by reference_id, everything else index
    by index."""

    def __init__(self, values: Mapping[tuple[str, ...], Sequence[Any]]) -> None:
        self.values = {tuple(p): _as_np(list(v)) for p, v in values.items() if p[-1] != "#"}
        refs = self.values.get((*UNIT_PATH, "reference_id"))
        self._unit_index: dict[int, int] = {}
        self._unit_player: np.ndarray | None = None
        if isinstance(refs, np.ndarray):
            seen = Counter(refs.tolist())
            self._unit_index = {r: i for i, r in enumerate(refs.tolist()) if seen[r] == 1}
            counts = self.values.get(UNIT_PATH)
            if isinstance(counts, np.ndarray) and int(counts.sum()) == len(refs):
                self._unit_player = np.repeat(np.arange(len(counts)), counts)
        self._player_sets: dict[tuple[str, ...], list[set]] = {}

    def player_set(self, path: tuple[str, ...], player: int) -> set:
        sets = self._player_sets.get(path)
        if sets is None:
            values = self.values[path]
            players = self._unit_player
            n = 0 if players is None else int(players.max()) + 1 if len(players) else 0
            sets = [set() for _ in range(n)]
            if players is not None:
                for v, p in zip(_tolist(values), players.tolist(), strict=True):
                    sets[p].add(_key(v))
            self._player_sets[path] = sets
        return sets[player] if player < len(sets) else set()


def _tolist(values: np.ndarray | list) -> list:
    return values.tolist() if isinstance(values, np.ndarray) else list(values)


def _key(v: Any) -> Any:
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    if isinstance(v, list):
        return tuple(_key(x) for x in v)
    return v


def _same(a: Any, b: Any) -> bool:
    return _key(a) == _key(b)


def _equal(a: np.ndarray | list, b: np.ndarray | list) -> np.ndarray:
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        eq = a == b
        if a.dtype.kind == "f" or b.dtype.kind == "f":
            eq |= np.isnan(a.astype(np.float64)) & np.isnan(b.astype(np.float64))
        return eq
    return np.fromiter((_same(x, y) for x, y in zip(_tolist(a), _tolist(b), strict=True)), dtype=bool, count=len(a))


def _distinctive(values: np.ndarray | list) -> np.ndarray:
    """Values unlikely to match by accident after a shift: not 0, 1 or -1,
    not an empty string, not one repeated byte."""
    if isinstance(values, np.ndarray):
        with np.errstate(invalid="ignore"):
            return np.isfinite(values) & (np.abs(values) > 1)
    out = []
    for v in values:
        if isinstance(v, str):
            out.append(len(v) > 1)
        elif isinstance(v, bytes):
            out.append(len(set(v)) > 1)
        elif isinstance(v, list):
            out.append(len(v) > 1)
        else:
            out.append(False)
    return np.asarray(out, dtype=bool)


@dataclass
class Events:
    """Compared values in read order: offset, size, matched, distinctive."""

    offsets: np.ndarray
    sizes: np.ndarray
    matched: np.ndarray
    distinctive: np.ndarray
    paths: list[tuple[str, ...]]
    path_index: np.ndarray


def compare(walk: sw.Walk, ref: Reference, sections: Collection[str], only: Collection[tuple[str, ...]] | None = None) -> Events:
    offs, sizes, matched, dist, idx = [], [], [], [], []
    paths: list[tuple[str, ...]] = []
    unit_refs = walk.columns.get((*UNIT_PATH, "reference_id"))
    pairing = None
    for path, col in walk.columns.items():
        if path[0] not in sections or path[-1] == "#" or path not in ref.values or not len(col):
            continue
        if only is not None and path not in only:
            continue
        new = _column_np(col)
        old = ref.values[path]
        o = np.frombuffer(col.offsets, dtype=np.int64)
        s = np.frombuffer(col.sizes, dtype=np.int64)
        if len(path) == len(UNIT_PATH) + 1 and path[: len(UNIT_PATH)] == UNIT_PATH and unit_refs is not None:
            if pairing is None:
                pairing = _pair_units(walk, ref)
            m = _paired_match(path, new, old, pairing, ref)
        else:
            n = min(len(new), len(old))
            m = np.zeros(len(new), dtype=bool)
            if n:
                m[:n] = _equal(new[:n], old[:n])
        paths.append(path)
        offs.append(o)
        sizes.append(s)
        matched.append(m)
        dist.append(_distinctive(new))
        idx.append(np.full(len(o), len(paths) - 1, dtype=np.int64))
    if not offs:
        empty = np.zeros(0, dtype=np.int64)
        return Events(empty, empty, empty.astype(bool), empty.astype(bool), paths, empty)
    offsets = np.concatenate(offs)
    order = np.argsort(offsets, kind="stable")
    return Events(
        offsets[order],
        np.concatenate(sizes)[order],
        np.concatenate(matched)[order],
        np.concatenate(dist)[order],
        paths,
        np.concatenate(idx)[order],
    )


def _pair_units(walk: sw.Walk, ref: Reference) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    refs = _tolist(_column_np(walk.columns[(*UNIT_PATH, "reference_id")]))
    old_index = np.asarray([ref._unit_index.get(r, -1) if type(r) is int else -1 for r in refs], dtype=np.int64)
    counts = walk.columns.get(UNIT_PATH)
    players = np.zeros(len(refs), dtype=np.int64)
    if counts is not None:
        c = np.asarray(_tolist(_column_np(counts)), dtype=np.int64)
        if int(c.sum()) >= len(refs):
            players = np.repeat(np.arange(len(c)), c)[: len(refs)]
    return old_index, players, np.asarray(refs, dtype=object)


def _paired_match(path, new, old, pairing, ref: Reference) -> np.ndarray:
    old_index, players, _ = pairing
    n = min(len(new), len(old_index))
    m = np.zeros(len(new), dtype=bool)
    paired = np.flatnonzero(old_index[:n] >= 0)
    if paired.size and len(old):
        valid = paired[old_index[paired] < len(old)]
        newv = new[valid] if isinstance(new, np.ndarray) else [new[i] for i in valid.tolist()]
        oldv = old[old_index[valid]] if isinstance(old, np.ndarray) else [old[j] for j in old_index[valid].tolist()]
        m[valid] = _equal(newv, oldv)
    newl = new if isinstance(new, list) else None
    for i in np.flatnonzero(old_index[:n] < 0).tolist():
        v = newl[i] if newl is not None else new[i].item()
        m[i] = _key(v) in ref.player_set(path, int(players[i]))
    return m


def first_break(events: Events, before: int | None = None) -> int | None:
    """Index of the first mismatch that the next BREAK_WINDOW values mostly
    share, or None."""
    miss = ~events.matched
    if before is not None:
        miss = miss & (events.offsets < before)
    if not miss.any():
        return None
    cs = np.concatenate(([0], np.cumsum(~events.matched)))
    n = len(miss)
    k = np.arange(n)
    hi = np.minimum(k + BREAK_WINDOW, n)
    window = cs[hi] - cs[k]
    need = np.maximum(np.ceil(BREAK_RATIO * (hi - k)), BREAK_MIN)
    hs = np.minimum(k + BREAK_SHORT, n)
    near = cs[hs] - cs[k]
    hits = np.flatnonzero(miss & (window >= need) & (near >= np.ceil(BREAK_RATIO * (hs - k))))
    if not hits.size:
        return None
    realigned = _realigns(events)
    solid = _solid_misses(events)
    for h in hits.tolist():
        if not realigned[h] or solid[h]:
            return h
    return None


def whole_path_break(events: Events) -> int | None:
    """Index of the first mismatch of any path with at least WHOLE_MIN values
    that mismatches in WHOLE_RATIO of them, or None. Not its first value: a
    field before the edit reads right in instance 0 and shifts from instance 1."""
    best = None
    n_paths = len(events.paths)
    if not n_paths or not len(events.matched):
        return None
    totals = np.bincount(events.path_index, minlength=n_paths)
    misses = np.bincount(events.path_index, weights=~events.matched, minlength=n_paths)
    for k in np.flatnonzero((totals >= WHOLE_MIN) & (misses >= WHOLE_RATIO * totals)).tolist():
        first = int(np.flatnonzero((events.path_index == k) & ~events.matched)[0])
        best = first if best is None else min(best, first)
    return best


def _solid_misses(events: Events) -> np.ndarray:
    """Per event: SHIFT_RUN mismatches (any value) from it on before the next
    distinctive match. A shift reads solid garbage until a string happens to
    resync it; a re-save's rewrites interleave with matches."""
    n = len(events.matched)
    good = np.append(np.flatnonzero(events.matched & events.distinctive), n)  # sentinel past the end
    miss_cs = np.concatenate(([0], np.cumsum(~events.matched)))
    k = np.arange(n)
    stop = good[np.searchsorted(good, k, side="left")]
    return miss_cs[stop] - miss_cs[k] >= SHIFT_RUN


def _realigns(events: Events) -> np.ndarray:
    """Per event: within the next REALIGN_WITHIN values a clean stretch
    starts, one no mismatch of any value interrupts, holding REALIGN_RUN
    distinctive matches."""
    n = len(events.matched)
    out = np.zeros(n, dtype=bool)
    if not n:
        return out
    miss = np.flatnonzero(~events.matched)
    # Clean stretches lie between mismatches: [miss[i] + 1, miss[i + 1]).
    starts = np.concatenate(([0], miss + 1))
    ends = np.append(miss, n)
    dcs = np.concatenate(([0], np.cumsum(events.matched & events.distinctive)))
    good = starts[(ends > starts) & (dcs[ends] - dcs[starts] >= REALIGN_RUN)]
    if not good.size:
        return out
    nxt = np.searchsorted(good, np.arange(n), side="right")  # the first clean stretch starting after each event
    has = nxt < len(good)
    out[has] = good[nxt[has]] - np.arange(n)[has] <= REALIGN_WITHIN
    return out


# --- probing one input ------------------------------------------------------


@dataclass
class FilterContext:
    map_size: tuple[int, int] | None = None
    trigger_range: tuple[float, float] | None = None


Anchor = Callable[[sw.Walk | None, sw.WalkError | None], "tuple[int, str, int | None] | None"]


@dataclass
class SegmentInput:
    """One file's bytes for one segment. `anchor` returns (evidence offset,
    reason, delta) when the walk misses the segment's landmark, else None;
    delta is the byte count the landmark says is missing (negative: extra)."""

    label: str
    data: bytes
    start: int
    anchor: Anchor
    ctx: FilterContext = field(default_factory=FilterContext)
    reference: Reference | None = None


@dataclass
class Probe:
    evidence: int | None
    reason: str
    delta: int | None
    walk: sw.Walk | None  # the full walk, or the partial one before a failure
    failed: bool
    stream: int | None = None  # the earliest evidence that misframes what follows (not a whole-path break)


def _np_values(walk: sw.Walk, path: tuple[str, ...]) -> tuple[sw.Column, np.ndarray] | None:
    col = walk.columns.get(path)
    if col is None or not len(col):
        return None
    values = _column_np(col)
    if not isinstance(values, np.ndarray):
        return None
    return col, values


def count_field_formula(fields: Sequence[sw.FieldSpec]) -> int:
    four = sum(1 for f in fields if f.kind in ("int", "float") and f.size == 4 and f.repeat == 1 and f.rule is None and not f.eof)
    return four - COUNT_FIELD_BIAS


def count_field(spec: sw.StructSpec) -> str | None:
    """A trigger struct's count field, found by name so an insert before it can't hide it."""
    if spec.name not in TRIGGER_BLOCKS:
        return None
    return next((f.name for f in spec.fields if f.name.startswith("static_value")), None)


def filter_violation(walk: sw.Walk, compiled: sw.CompiledStructure, ctx: FilterContext) -> tuple[int, str] | None:
    """The earliest field occurrence that fails a plausibility filter."""
    best: tuple[int, str] | None = None

    def note(col: sw.Column, bad: np.ndarray, why: str) -> None:
        nonlocal best
        hits = np.flatnonzero(bad)
        if hits.size:
            off = col.offsets[int(hits[0])]
            if best is None or off < best[0]:
                best = (off, why)

    for block, (spec, prefixes) in block_prefixes(compiled).items():
        for prefix in prefixes:
            if spec.name == "UnitStruct":
                _unit_filters(walk, prefix, ctx, note)
            count = count_field(spec)
            if count is not None:
                got = _np_values(walk, (*prefix, count))
                if got is not None:
                    want = count_field_formula(spec.fields)
                    note(got[0], got[1] != want, f"{'/'.join(block)}.{count} != {want}")
            if prefix == ("Triggers",) and ctx.trigger_range is not None:
                got = _np_values(walk, ("Triggers", "trigger_version"))
                if got is not None:
                    bad = np.asarray([not fs.plausible_trigger_version(float(v), ctx.trigger_range) for v in got[1].tolist()])
                    note(got[0], bad, f"trigger_version outside {ctx.trigger_range}")
    return best


def _unit_filters(walk: sw.Walk, prefix: tuple[str, ...], ctx: FilterContext, note) -> None:
    def get(name: str):
        return _np_values(walk, (*prefix, name))

    w, h = ctx.map_size if ctx.map_size else (math.inf, math.inf)
    with np.errstate(invalid="ignore"):
        for name, limit in (("x", w), ("y", h)):
            got = get(name)
            if got is not None:
                v = got[1].astype(np.float64)
                note(got[0], ~(np.isfinite(v) & (v >= 0) & (v <= limit)), f"unit {name} off the map")
        checks = (
            ("z", UNIT_Z_RANGE),
            ("status", UNIT_STATUS_RANGE),
            ("unit_const", UNIT_CONST_RANGE),
            ("capture_flag", CAPTURE_FLAG_RANGE),
            ("caption_string_id", (CAPTION_STRING_ID_MIN, math.inf)),
        )
        for name, (lo, hi) in checks:
            got = get(name)
            if got is not None:
                v = got[1].astype(np.float64)
                note(got[0], ~(np.isfinite(v) & (v >= lo) & (v <= hi)), f"unit {name} outside {lo}..{hi}")
    got = get("reference_id")
    if got is not None:
        v = got[1]
        bad = v < 0
        _, first = np.unique(v, return_index=True)
        dup = np.ones(len(v), dtype=bool)
        dup[first] = False
        note(got[0], bad | dup, "unit reference_id negative or repeated")


def block_prefixes(compiled: sw.CompiledStructure) -> dict[tuple[str, ...], tuple[sw.StructSpec, list[tuple[str, ...]]]]:
    """Each block's definition path mapped to its spec and the field-path
    prefixes it is read under."""
    cached = getattr(compiled, "_block_prefixes", None)
    if cached is not None:
        return cached
    out: dict[tuple[str, ...], tuple[sw.StructSpec, list[tuple[str, ...]]]] = {}

    def visit(spec: sw.StructSpec, block: tuple[str, ...], prefix: tuple[str, ...]) -> None:
        out.setdefault(block, (spec, []))[1].append(prefix)
        for f in spec.fields:
            if f.kind == "struct":
                visit(f.struct, (*block, f.struct.name), (*prefix, f.name))

    for name, spec in compiled.sections.items():
        visit(spec, (name,), (name,))
    compiled._block_prefixes = out
    return out


# --- candidate generation ---------------------------------------------------


@dataclass(frozen=True)
class Position:
    block: tuple[str, ...]
    index: int
    offset: int
    multiplicity: int  # instances read: exact on a complete walk, 0 when unknown


def positions(compiled: sw.CompiledStructure, walk: sw.Walk, lower: int, upper: int, limit: int | None = MAX_POSITIONS) -> list[Position]:
    """Every insertion point whose first occurrence was read inside
    [lower, upper], nearest `upper` first."""
    found = []
    for block, (spec, prefixes) in block_prefixes(compiled).items():
        for i in range(len(spec.fields) + 1):
            off, mult = None, 0
            for prefix in prefixes:
                o = None
                if i < len(spec.fields):
                    col = walk.columns.get((*prefix, spec.fields[i].name))
                    if col is not None and len(col):
                        o, n = col.offsets[0], len(col)
                elif len(prefix) == 1:
                    span = walk.sections.get(prefix[0])
                    if span is not None:
                        o, n = span[1], 1
                else:
                    inst = walk.columns.get((*prefix, "#"))
                    if inst is not None and len(inst):
                        o, n = inst.offsets[0] + inst.sizes[0], len(inst)
                if o is None and i > 0:
                    # Never read: the walk failed on this field, which starts where the previous one
                    # ends. A section has one instance; a struct's count is unknown (0), not the 1 read.
                    prev = walk.columns.get((*prefix, spec.fields[i - 1].name))
                    if prev is not None and len(prev):
                        o, n = prev.offsets[0] + prev.sizes[0], 1 if len(prefix) == 1 else 0
                if o is not None:
                    off = o if off is None else min(off, o)
                    mult += n
            if off is not None and lower <= off <= upper:
                found.append(Position(block, i, off, mult))
    found.sort(key=lambda p: (-p.offset, -len(p.block), -p.index))
    return found if limit is None else found[:limit]


class Names:
    """Hands out inferred field names, inferred_v<major>_<minor>_<n>."""

    def __init__(self, version: str) -> None:
        self.stem = INFERRED_PREFIX + version.replace(".", "_") + "_"
        self.next = 0

    def peek(self, k: int) -> list[str]:
        return [f"{self.stem}{self.next + j}" for j in range(k)]

    def commit(self, edit: Edit) -> None:
        self.next += len(edit.new_names())


def _fixed_size(f: sw.FieldSpec) -> int | None:
    if f.kind in ("int", "float", "data", "chars") and f.repeat == 1 and f.rule is None and not f.eof and f.is_list is not True:
        return f.size
    return None


def _int_type(f: sw.FieldSpec, n: int) -> str:
    if f.kind == "int" and n in (1, 2, 4):
        return f"{'s' if f.signed else 'u'}{n * 8}"
    return str(n)


def _rule_targets(spec: sw.StructSpec) -> set[str]:
    return {t for f in spec.fields if f.rule is not None for t in f.rule.targets}


def candidates(
    structure: Mapping[str, Any],
    compiled: sw.CompiledStructure,
    places: Sequence[Position],
    delta: int | None,
    names: Names,
    k_inserts: Mapping[tuple[str, ...], int] | None = None,
    walk: sw.Walk | None = None,
) -> list[Edit]:
    """Every single edit of the vocabulary at `places`, plus k-field 4-byte
    inserts into trigger structs whose count field says they grew."""
    specs = {b: s for b, (s, _) in block_prefixes(compiled).items()}
    new1, new2 = names.peek(2)
    out: list[Edit] = []
    for rank, p in enumerate(places):
        spec = specs[p.block]
        fields = spec.fields
        at = fields[p.index].name if p.index < len(fields) else ""

        def add(count: int, new: list[tuple[str, dict]], kind: str, p=p, at=at) -> None:
            out.append(Edit(p.block, p.index, count, new, kind, at))

        per = None if delta is None or p.multiplicity == 0 or delta % p.multiplicity else delta // p.multiplicity
        for t in FIXED_INSERT_TYPES.values():
            add(0, [(new1, {"type": t})], f"insert {t}")
        raw_sizes: list[int] = []
        if per is not None and 0 < per <= RAW_INSERT_MAX and per not in FIXED_INSERT_TYPES:
            raw_sizes = [per]
        elif (delta is None or p.multiplicity == 0) and rank < RAW_SWEEP_POSITIONS:
            raw_sizes = [n for n in range(1, RAW_INSERT_MAX + 1) if n not in FIXED_INSERT_TYPES]
        for n in raw_sizes:
            add(0, [(new1, {"type": str(n)})], f"insert raw {n}")
        add(0, [(new1, {"type": "str16"})], "insert str16")
        add(0, [(new1, {"type": "str32"})], "insert str32")
        add(0, [(new1, {"type": "u16"}), (new2, {"type": "str16"})], "insert u16 + str16")
        if p.index >= len(fields):
            continue
        f = fields[p.index]
        size = _fixed_size(f)
        if size is None or f.name in _rule_targets(spec):
            continue
        definition = dict(block_dict(structure, p.block)["retrievers"][f.name])
        for count in (1, 2):
            if p.index + count > len(fields):
                break
            run = fields[p.index : p.index + count]
            sizes = [_fixed_size(g) for g in run]
            if any(s is None for s in sizes) or any(g.name in _rule_targets(spec) for g in run):
                break
            removed = sum(sizes)
            if per is None or per == -removed:
                add(count, [], f"delete {count}")
            if count == 2 and per is not None and removed + per > 0:
                add(2, [(new1, {"type": str(removed + per)})], f"merge 2 into raw {removed + per}")
        new_sizes = [n for n in FIXED_INSERT_TYPES if n != size]
        if per is not None and size + per > 0 and size + per not in new_sizes and size + per != size:
            new_sizes.append(size + per)
        for n in new_sizes:
            t = _int_type(f, n)
            add(1, [(f.name, {**definition, "type": t})], f"retype {f.var} -> {t}")
        for t in ("str16", "str32"):
            add(1, [(f.name, {**definition, "type": t})], f"retype {f.var} -> {t}")
        if f.kind == "int" and size == 4:
            add(1, [(new1, {"type": "u16"}), (f.name, {**definition, "type": "str16"})], f"retype {f.var} -> u16 + str16")
    for block, k in (k_inserts or {}).items():
        spec = specs.get(block)
        if spec is None:
            continue
        new = names.peek(k)
        first_str = next((i for i, f in enumerate(spec.fields) if i >= 2 and _fixed_size(f) != 4), len(spec.fields))
        added = [(n, {"type": "s32"}) for n in new]
        for i in range(2, first_str + 1):
            at = spec.fields[i].name if i < len(spec.fields) else ""
            out.append(Edit(block, i, 0, added, f"insert {k} x s32", at))
    return out


def count_field_growth(compiled: sw.CompiledStructure, walk: sw.Walk) -> dict[tuple[str, ...], int]:
    """Trigger structs whose count field, in the first instance (read before
    any misframing of that block), is above the formula: block -> how many
    4-byte fields it gained."""
    out = {}
    for block, (spec, prefixes) in block_prefixes(compiled).items():
        count = count_field(spec)
        if count is None:
            continue
        seen: set[int] = set()
        for prefix in prefixes:
            got = _np_values(walk, (*prefix, count))
            if got is not None:
                seen.add(int(got[1][0]))
        if len(seen) == 1:
            k = seen.pop() - count_field_formula(spec.fields)
            if 0 < k <= 64:
                out[block] = k
    return out


# --- the per-segment search -------------------------------------------------


def _subset(structure: Mapping[str, Any], sections: Sequence[str]) -> dict:
    return {s: structure[s] for s in sections}


class _Group:
    """Candidates sharing one byte layout, so one walk per input."""

    def __init__(self, rep_structure: dict, block: tuple[str, ...], rep_names: list[str]) -> None:
        self.structure = rep_structure
        self.block = block
        self.rep_names = rep_names
        self.members: list[tuple[Edit, dict, list[str]]] = []
        self.compiled: sw.CompiledStructure | None = None
        self.prefixes: list[tuple[str, ...]] = []


def _translate(walk: sw.Walk, prefixes: Sequence[tuple[str, ...]], mapping: Mapping[str, str]) -> sw.Walk:
    if not mapping:
        return walk
    columns = {}
    for path, col in walk.columns.items():
        new = path
        for prefix in prefixes:
            n = len(prefix)
            if len(path) > n and path[:n] == prefix and path[n] in mapping:
                new = (*prefix, mapping[path[n]], *path[n + 1 :])
                break
        columns[new] = col
    return sw.Walk(sections=walk.sections, columns=columns, end=walk.end, reads=walk.reads)


@dataclass
class Choice:
    edit: Edit
    rivals: list[Edit]
    ambiguous: bool
    fits: bool
    note: str


@dataclass
class SegmentOutcome:
    structure: dict
    edits: list[tuple[Edit, Choice, str]]  # (edit, choice, evidence)
    refused: str | None = None


class SegmentSearch:
    """The localize-and-repair loop over one segment's inputs."""

    def __init__(
        self,
        name: str,
        sections: Sequence[str],
        inputs: Sequence[SegmentInput],
        names: Names,
        budget: Budget,
        spent: list[int],
        diff_pair: tuple[SegmentInput, SegmentInput] | None = None,
        diff_expect_new: bool = False,
    ) -> None:
        self.name = name
        self.sections = list(sections)
        self.inputs = list(inputs)
        self.names = names
        self.budget = budget
        self.spent = spent  # shared [reads] across segments
        self.diff_pair = diff_pair
        self.diff_expect_new = diff_expect_new
        self.ref_index = next((i for i, inp in enumerate(self.inputs) if inp.reference is not None), None)

    def _walk(self, compiled: sw.CompiledStructure, inp: SegmentInput) -> tuple[sw.Walk | None, sw.WalkError | None]:
        try:
            walk = sw.walk(compiled, inp.data, inp.start, self.sections, budget=sw.DEFAULT_READ_BUDGET)
            reads, result = walk.reads, (walk, None)
        except sw.WalkError as e:
            reads, result = (e.partial.reads if e.partial is not None else 0), (None, e)
        self.spent[0] += reads
        if self.spent[0] > self.budget.walk_reads:
            raise _BudgetExceeded(f"walker budget of {self.budget.walk_reads} field reads spent")
        return result

    def _evidence(
        self, compiled: sw.CompiledStructure, inp: SegmentInput, walk: sw.Walk | None, error: sw.WalkError | None
    ) -> Probe:
        found: list[tuple[int, bool, str]] = []  # (offset, soft, reason): hard evidence sorts first on a tie
        seen = walk if walk is not None else error.partial
        if error is not None:
            found.append((error.offset, False, f"walk fails at {error.path}: {error.reason}"))
        if seen is not None:
            v = filter_violation(seen, compiled, inp.ctx)
            if v is not None:
                found.append((v[0], False, v[1]))
        anchor = inp.anchor(walk, error)
        delta = None
        if anchor is not None:
            found.append((anchor[0], False, anchor[1]))
            delta = anchor[2]
        if inp.reference is not None and seen is not None:
            events = compare(seen, inp.reference, self.sections)
            k = first_break(events)
            if k is not None:
                path = events.paths[int(events.path_index[k])]
                found.append((int(events.offsets[k]), True, f"reference values break at {'.'.join(path)}"))
        stream = min((f[0] for f in found), default=None)
        if inp.reference is not None and seen is not None:
            k = whole_path_break(events)
            if k is not None:
                path = events.paths[int(events.path_index[k])]
                found.append((int(events.offsets[k]), True, f"reference values mismatch in nearly every {'.'.join(path)}"))
        if not found:
            return Probe(None, "", delta, seen, error is not None)
        off, _, why = min(found)
        return Probe(off, why, delta, seen, error is not None, stream)

    def _gain(self, probe: Probe, current: Events, inp: SegmentInput, lower: int) -> int:
        """How many more distinctive reference values the candidate matches
        than the current structure, from the edit's position to the
        candidate's own stream-level evidence (a whole-path break leaves the
        values after it framed)."""
        if probe.walk is None:
            return 0
        upper = probe.stream if probe.stream is not None else math.inf

        def ok(events: Events) -> int:
            return int((events.matched & events.distinctive & (events.offsets >= lower) & (events.offsets < upper)).sum())

        return ok(compare(probe.walk, inp.reference, self.sections)) - ok(current)

    def probe(self, structure: Mapping[str, Any], inp: SegmentInput) -> Probe:
        compiled = sw.compile_structure(_subset(structure, self.sections))
        walk, error = self._walk(compiled, inp)
        return self._evidence(compiled, inp, walk, error)

    def _lower_bound(self, probe: Probe, inp: SegmentInput) -> int:
        if inp.reference is None or probe.walk is None:
            return inp.start
        events = compare(probe.walk, inp.reference, self.sections)
        ok = np.flatnonzero(events.matched & events.distinctive & (events.offsets + events.sizes <= probe.evidence))
        if not ok.size:
            return inp.start
        k = int(ok[-1])
        return int(events.offsets[k] + events.sizes[k])

    def run(self, structure: dict) -> SegmentOutcome:
        current = structure
        done: list[tuple[Edit, Choice, str]] = []
        per_block: Counter = Counter()
        try:
            while True:
                probes = [self.probe(current, inp) for inp in self.inputs]
                bad = [i for i, p in enumerate(probes) if p.evidence is not None]
                if not bad:
                    return SegmentOutcome(current, done)
                primary = bad[0]
                p = probes[primary]
                inp = self.inputs[primary]
                located = f"{inp.label} @ {p.evidence}: {p.reason}"
                if len(done) >= self.budget.edits_total:
                    return SegmentOutcome(current, done, f"edit budget of {self.budget.edits_total} spent; still: {located}")
                choice = self._round(current, probes, primary)
                if choice is None:
                    return SegmentOutcome(current, done, f"no single edit in the vocabulary fits; first evidence: {located}")
                per_block[choice.edit.block] += 1
                if per_block[choice.edit.block] > self.budget.edits_per_block:
                    where = "/".join(choice.edit.block)
                    why = f"edit budget of {self.budget.edits_per_block} for {where} spent; still: {located}"
                    return SegmentOutcome(current, done, why)
                current = apply_edit(current, choice.edit)
                self.names.commit(choice.edit)
                done.append((choice.edit, choice, located))
        except _BudgetExceeded as e:
            return SegmentOutcome(current, done, str(e))

    def _groups(self, current: dict, edits: Sequence[Edit]) -> list[_Group]:
        groups: dict[tuple, _Group] = {}
        for edit in edits:
            try:
                new = apply_edit(current, edit)
            except ValueError:
                continue
            retrievers = block_dict(new, edit.block)["retrievers"]
            key = (edit.block, layout_signature(retrievers))
            group = groups.get(key)
            names = list(retrievers)
            if group is None:
                group = groups[key] = _Group(new, edit.block, names)
            group.members.append((edit, new, names))
        out = []
        for group in groups.values():
            try:
                group.compiled = sw.compile_structure(_subset(group.structure, self.sections))
            except sw.StructureError:
                continue
            spec_prefixes = block_prefixes(group.compiled).get(group.block)
            group.prefixes = spec_prefixes[1] if spec_prefixes else []
            out.append(group)
        return out

    def _member_probes(self, group: _Group, inp: SegmentInput) -> list[Probe]:
        walk, error = self._walk(group.compiled, inp)
        out = []
        for _, _, names in group.members:
            mapping = {r: m for r, m in zip(group.rep_names, names, strict=True) if r != m}
            w = None if walk is None else _translate(walk, group.prefixes, mapping)
            e = error
            if error is not None and error.partial is not None and mapping:
                e = sw.WalkError(error.path, error.offset, error.reason)
                e.partial = _translate(error.partial, group.prefixes, mapping)
            out.append(self._evidence(group.compiled, inp, w, e))
        return out

    def _round(self, current: dict, probes: Sequence[Probe], primary: int) -> Choice | None:
        p = probes[primary]
        inp = self.inputs[primary]
        compiled = sw.compile_structure(_subset(current, self.sections))
        lower = self._lower_bound(p, inp)
        seen = p.walk if p.walk is not None else sw.Walk()
        places = positions(compiled, seen, lower, p.evidence)
        grow = {}
        for probe in probes:
            if probe.walk is not None:
                for block, k in count_field_growth(compiled, probe.walk).items():
                    grow.setdefault(block, k)
        edits = candidates(current, compiled, places, p.delta, self.names, grow)
        offsets = {(q.block, q.index): q.offset for q in positions(compiled, seen, 0, len(inp.data), limit=None)}
        groups = self._groups(current, edits)
        kept: list[tuple[_Group, int, Probe]] = []
        for group in groups:
            for m, probe in enumerate(self._member_probes(group, inp)):
                # With a reference, an earlier evidence of another kind can still be progress (_gain decides).
                if probe.evidence is None or probe.evidence > p.evidence or inp.reference is not None:
                    kept.append((group, m, probe))
        if not kept:
            return None
        results = []  # (edit, fits, primary evidence, ref probe, structure, primary probe)
        by_group: dict[int, list[tuple[int, Probe]]] = {}
        for group, m, probe in kept:
            by_group.setdefault(id(group), []).append((m, probe))
        group_of = {id(g): g for g, _, _ in kept}
        for gid, members in by_group.items():
            group = group_of[gid]
            per_input = {primary: dict(members)}
            for j, other in enumerate(self.inputs):
                if j == primary:
                    continue
                probes_j = self._member_probes(group, other)
                per_input[j] = {m: probes_j[m] for m, _ in members}
            for m, _ in members:
                ok = True
                for j in range(len(self.inputs)):
                    if j == primary:
                        continue
                    before, after = probes[j].evidence, per_input[j][m].evidence
                    if after is not None and (before is None or after < before):
                        ok = False
                        break
                if not ok:
                    continue
                fits = all(per_input[j][m].evidence is None for j in per_input)
                ref_probe = per_input[self.ref_index][m] if self.ref_index is not None else None
                edit = group.members[m][0]
                results.append((edit, fits, per_input[primary][m].evidence, ref_probe, group.members[m][1], per_input[primary][m]))
        if not results:
            return None
        fitting = [r for r in results if r[1]]
        if fitting:
            pool, fits = fitting, True
        else:
            if inp.reference is not None:
                # Getting further proves nothing on its own (a junk insert can push a break or a
                # walk failure along); with a reference, the edit must make more values match.
                if p.walk is None:
                    return None
                current = compare(p.walk, inp.reference, self.sections)
                gains = [self._gain(r[5], current, inp, offsets.get((r[0].block, r[0].index), lower)) for r in results]
                best = max(gains)
                if best < PARTIAL_MIN:
                    return None
                pool, fits = [r for r, g in zip(results, gains, strict=True) if g == best], False
            else:
                results = [r for r in results if r[2] is not None and r[2] > p.evidence]
                if not results:
                    return None
                best = max(r[2] for r in results)
                pool, fits = [r for r in results if r[2] == best], False
        if fits and self.diff_pair is not None and len(pool) > 1:
            pool = self._diff_filter(pool) or pool
        return self._choose(pool, fits, offsets)

    def _diff_filter(self, pool):
        a, b = self.diff_pair
        out = []
        for r in pool:
            structure = r[4]
            wa = self.probe(structure, a).walk
            wb = self.probe(structure, b).walk
            if wa is not None and wb is not None and diff_rule_holds(wa, wb, self.diff_expect_new):
                out.append(r)
        return out

    def _choose(self, pool, fits: bool, offsets) -> Choice:
        def rank(r) -> tuple:
            e: Edit = r[0]
            order = ("insert", "retype", "delete", "merge")
            kind = next((i for i, k in enumerate(order) if e.kind.startswith(k)), len(order))
            return (-offsets.get((e.block, e.index), e.index), kind, len(e.fields), e.kind)

        pool = sorted(pool, key=rank)
        if len(pool) == 1:
            return Choice(pool[0][0], [], False, fits, "the only candidate that fits" if fits else "the only candidate that gets furthest")
        if self.ref_index is not None and all(r[3] is not None and r[3].walk is not None for r in pool):
            ref = self.inputs[self.ref_index].reference
            walks = [r[3].walk for r in pool]
            scores = score_candidates(walks, ref, self.sections)
            if scores is not None:
                i = min(range(len(pool)), key=lambda j: (scores[j], j))
                tied, closest = [i], None
                for j in range(len(pool)):
                    if j == i:
                        continue
                    d = pair_distance(walks[i], walks[j], ref, self.sections)
                    if d is None or d[2] < PAIR_MIN or d[1] - d[0] < MARGIN:
                        tied.append(j)
                    elif closest is None or d[1] - d[0] < closest[1] - closest[0]:
                        closest = d
                if len(tied) == 1:
                    rivals = [r[0] for j, r in enumerate(pool) if j != i]
                    a, b, n = closest
                    note = f"reference mismatch {a:.4f} against the closest rival's {b:.4f}, over the {n} values they parse differently"
                    return Choice(pool[i][0], rivals, False, fits, note)
                tied.sort()  # the pool's rank order: the kept edit is the best-ranked of the tied
                note = f"{len(tied)} candidates within MARGIN of each other on the values they parse differently"
                return Choice(pool[tied[0]][0], [pool[j][0] for j in tied[1:]], True, fits, note)
            note = f"{len(pool)} candidates parse every compared field identically"
            return Choice(pool[0][0], [r[0] for r in pool[1:]], True, fits, note)
        return Choice(pool[0][0], [r[0] for r in pool[1:]], True, fits, f"{len(pool)} candidates fit and nothing tells them apart")


def score_candidates(walks: Sequence[sw.Walk], ref: Reference, sections: Collection[str]) -> list[float] | None:
    """Each walk's mismatch fraction against `ref`, over only the compared
    paths whose parse differs between the walks. None when none differs."""
    paths = set()
    for w in walks:
        paths.update(p for p in w.columns if p in ref.values and p[0] in sections)
    differing = set()
    for p in paths:
        cols = [w.columns.get(p) for w in walks]
        if any(c is None for c in cols):
            differing.add(p)
            continue
        if all(c is cols[0] for c in cols):
            continue
        first = _column_np(cols[0])
        for c in cols[1:]:
            other = _column_np(c)
            if len(other) != len(first) or not bool(np.all(_equal(other, first))):
                differing.add(p)
                break
    if not differing:
        return None
    scores = []
    for w in walks:
        events = compare(w, ref, sections, only=differing)
        expected = sum(len(ref.values[p]) for p in differing)
        compared = max(len(events.matched), expected, 1)
        scores.append((compared - int(events.matched.sum())) / compared)
    return scores


def pair_distance(a: sw.Walk, b: sw.Walk, ref: Reference, sections: Collection[str]) -> tuple[float, float, int] | None:
    """Each walk's reference mismatch fraction over only the compared values
    the two parse differently, and how many those are. None when none
    differs. A path whose length differs counts whole."""
    paths = {p for w in (a, b) for p in w.columns if p in ref.values and p[0] in sections and p[-1] != "#"}
    masks: dict[tuple[str, ...], tuple[np.ndarray | None, np.ndarray | None]] = {}
    for p in paths:
        ca, cb = a.columns.get(p), b.columns.get(p)
        ca, cb = (c if c is not None and len(c) else None for c in (ca, cb))  # a partial walk leaves empty columns
        if ca is cb:
            continue
        if ca is None or cb is None or len(ca) != len(cb):
            masks[p] = (None if ca is None else np.ones(len(ca), dtype=bool), None if cb is None else np.ones(len(cb), dtype=bool))
            continue
        diff = ~_equal(_column_np(ca), _column_np(cb))
        if diff.any():
            masks[p] = (diff, diff)
    if not masks:
        return None
    out = []
    for side, w in enumerate((a, b)):
        events = compare(w, ref, sections, only=set(masks))
        miss = total = 0
        for k, p in enumerate(events.paths):
            mask = masks[p][side]
            if mask is None:
                continue
            m = events.matched[events.path_index == k]  # read order is element order within a path
            miss += int((~m[mask]).sum())
            total += int(mask.sum())
        for pair in masks.values():
            if pair[side] is None:  # this walk lacks the path: the other side's values all count as missed
                n = int(pair[1 - side].sum())
                miss += n
                total += n
        out.append((miss, total))
    (miss_a, total_a), (miss_b, total_b) = out
    return miss_a / max(total_a, 1), miss_b / max(total_b, 1), max(total_a, total_b)


def diff_rule_holds(a: sw.Walk, b: sw.Walk, expect_new: bool) -> bool:
    """Two files differing by one in-game edit: every object (struct
    instance, or a section's own fields) changes in at most one field, and
    with `expect_new` only inferred fields change."""
    changed: dict[tuple, set[str]] = {}
    for path, col in a.columns.items():
        if path[-1] == "#":
            continue
        other = b.columns.get(path)
        if other is None:
            return False
        va, vb = col.value_list(), other.value_list()
        if len(va) != len(vb):
            changed.setdefault((path[:-1], "count"), set()).add(path[-1])
            continue
        for i, (x, y) in enumerate(zip(va, vb, strict=True)):
            if not _same(x, y):
                obj = (path[:-1], i) if len(path) > 2 else (path[:1], 0)
                changed.setdefault(obj, set()).add(path[-1])
    if not changed:
        return False
    if any(len(v) > 1 for v in changed.values()):
        return False
    return not expect_new or all(n.startswith(INFERRED_PREFIX) for v in changed.values() for n in v)


# --- scenario files ---------------------------------------------------------


@dataclass
class ScenarioFile:
    path: Path
    raw: bytes
    version: str
    body: fs.Body
    sha256: str
    grid: fs.GridMatch | None = None
    grid_miss: str | None = None


def load_file(path: Path | str) -> ScenarioFile:
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise InferInputError(f"{path}: {e}") from e
    try:
        version = raw[:4].decode("ascii")
        if len(version) != 4 or not version[0].isdigit():
            raise ValueError(version)
    except (UnicodeDecodeError, ValueError) as e:
        raise InferInputError(f"{path}: not a scenario file (no version at the start)") from e
    body = fs.find_body(raw)
    if isinstance(body, str):
        raise InferInputError(f"{path}: {body}")
    return ScenarioFile(path, raw, version, body, hashlib.sha256(raw).hexdigest())


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(x) for x in version.split("."))


def known_versions() -> list[str]:
    from descape import library_compat

    found = {
        d.name[1:]
        for base in (library_compat.VERSIONS_DIR, library_compat.REPO_VERSIONS_DIR)
        for d in base.glob("v*")
        if (d / "structure.json").is_file()
    }
    return sorted(found, key=_version_key)


def load_structure(version: str) -> dict:
    """A known version's structure, library first, then the repo's, with
    library_compat.fix_structure() applied."""
    from descape import library_compat

    for base in (library_compat.VERSIONS_DIR, library_compat.REPO_VERSIONS_DIR):
        path = base / f"v{version}" / "structure.json"
        if path.is_file():
            structure = json.loads(path.read_text(encoding="utf-8"))
            library_compat.fix_structure(structure, version)
            return structure
    raise InferInputError(f"no structure.json for version {version}")


def resolve_base(base: str | Path | tuple[str, Mapping[str, Any]] | None) -> tuple[str, dict]:
    """(version, structure) from a version string, a structure.json path
    (its version from the parent directory name v<version>), a ready pair,
    or None for the newest known version."""
    from descape import library_compat

    if base is None:
        versions = known_versions()
        if not versions:
            raise InferInputError("no known structure to start from")
        base = versions[-1]
    if isinstance(base, tuple):
        version, structure = base
        structure = json.loads(json.dumps(structure))
        library_compat.fix_structure(structure, version)
        return version, structure
    if isinstance(base, Path) or (isinstance(base, str) and base.endswith(".json")):
        path = Path(base)
        version = path.parent.name.removeprefix("v")
        try:
            structure = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise InferInputError(f"{path}: {e}") from e
        library_compat.fix_structure(structure, version)
        return version, structure
    return base, load_structure(base)


def _section_names(structure: Mapping[str, Any]) -> list[str]:
    return list(structure)


def _terrain_stride(structure: Mapping[str, Any]) -> int:
    compiled = sw.compile_structure({"Map": structure["Map"]})
    terrain = next(f for f in compiled.sections["Map"].fields if f.name == "terrain_data")
    if terrain.struct is None or terrain.struct.fixed is None:
        raise InferInputError("the base Map's terrain_data is not a fixed-size struct")
    return terrain.struct.fixed.size


def _end_anchor(expected: int) -> Anchor:
    def anchor(walk: sw.Walk | None, error: sw.WalkError | None):
        if walk is None:
            return None
        if walk.end != expected:
            return min(walk.end, expected), f"segment ends at {walk.end}, the landmark is at {expected}", expected - walk.end
        return None

    return anchor


def _grid_anchor(grid: fs.GridMatch) -> Anchor:
    def anchor(walk: sw.Walk | None, error: sw.WalkError | None):
        seen = walk if walk is not None else error.partial if error is not None else None
        col = seen.columns.get(("Map", "map_width")) if seen is not None else None
        if col is None or not len(col):
            return None
        o = col.offsets[0]
        if o != grid.offset:
            return min(o, grid.offset), f"map_width read at {o}, the terrain grid is at {grid.offset}", grid.offset - o
        if walk is not None and walk.end != grid.end:
            return o, f"Map ends at {walk.end}, the terrain grid ends at {grid.end}", None
        return None

    return anchor


def _trigger_version_anchor(data: bytes, trigger_range: tuple[float, float]) -> Anchor:
    def anchor(walk: sw.Walk | None, error: sw.WalkError | None):
        if walk is None:
            return None
        if walk.end + 8 > len(data):
            return walk.end, f"Units ends at {walk.end}, too close to the end for a trigger version", None
        (value,) = struct.unpack_from("<d", data, walk.end)
        if not fs.plausible_trigger_version(value, trigger_range):
            return walk.end, f"no plausible trigger version after Units ({value!r})", None
        return None

    return anchor


# --- the result -------------------------------------------------------------


@dataclass
class EditRecord:
    segment: str
    edit: Edit
    evidence: str
    note: str
    rivals: list[Edit]
    ambiguous: bool


@dataclass
class InferResult:
    status: str
    version: str | None
    base_version: str | None
    structure: dict | None
    edits: list[EditRecord] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    refusal: str | None = None
    inputs: list[tuple[str, str, str]] = field(default_factory=list)  # (role, path, sha256)
    notes: list[str] = field(default_factory=list)
    report_lines: list[str] = field(default_factory=list)


LibraryCheck = Callable[[Path, Path], Mapping[str, Any]]


def divergence_check(path: Path, structure_path: Path) -> Mapping[str, Any]:
    """The library's own parse of `path` under `structure_path`, in a fresh
    interpreter (tools/map_structure_divergence.py's worker)."""
    from tools import map_structure_divergence as msd

    return msd.clean_check(path, structure_path)


def infer(
    base: str | Path | tuple[str, Mapping[str, Any]] | None,
    samples: Sequence[Path | str],
    *,
    reference: tuple[Path | str, Path | str] | None = None,
    reference_values: Mapping[tuple[str, ...], Sequence[Any]] | None = None,
    diff_pair: tuple[Path | str, Path | str] | None = None,
    diff_expect_new: bool = False,
    terrain_ids: Collection[int],
    trigger_versions: tuple[float, float] | None = None,
    budget: Budget = DEFAULT_BUDGET,
    accept_ambiguous: bool = False,
    library_check: LibraryCheck | None = divergence_check,
    vocabulary: Mapping[str, Collection[int]] | None = None,
) -> InferResult:
    """Infer the structure of the samples' version from `base`. Raises
    InferInputError when an input can't be used at all."""
    if not samples:
        raise InferInputError("no sample files")
    base_version, structure = resolve_base(base)
    sample_files = [load_file(p) for p in samples]
    versions = {f.version for f in sample_files}
    if len(versions) != 1:
        raise InferInputError(f"samples carry more than one version: {sorted(versions)}")
    version = versions.pop()
    result = InferResult(STATUS_REFUSED, version, base_version, None)
    files: list[ScenarioFile] = []
    roles: list[str] = []
    ref_values: dict[tuple[str, ...], list] | None = None
    ref_trigger: float | None = None
    if reference is not None:
        old = load_file(reference[0])
        new = load_file(reference[1])
        if new.version != version:
            raise InferInputError(f"the reference re-save is version {new.version}, the samples {version}")
        walked = sw.walk_file(load_structure(old.version), old.raw)
        ref_values = {p: c.value_list() for w in (walked.header, walked.body) for p, c in w.columns.items()}
        ref_trigger = (ref_values.get(("Triggers", "trigger_version")) or [None])[0]
        result.inputs.append(("reference (old)", str(old.path), old.sha256))
        files.append(new)
        roles.append("reference (re-save)")
    elif reference_values is not None:
        ref_values = {tuple(p): list(v) for p, v in reference_values.items()}
    for f in sample_files:
        if all(f.path.resolve() != g.path.resolve() for g in files):
            files.append(f)
            roles.append("sample")
    diff_files: list[ScenarioFile] = []
    if diff_pair is not None:
        diff_files = [load_file(p) for p in diff_pair]
        for f in diff_files:
            if f.version != version:
                raise InferInputError(f"{f.path}: the differential pair must be version {version}, not {f.version}")
    for role, f in zip(roles, files, strict=True):
        result.inputs.append((role, str(f.path), f.sha256))
    for f in diff_files:
        result.inputs.append(("differential pair", str(f.path), f.sha256))

    trigger_range = trigger_versions or fs.KNOWN_TRIGGER_VERSIONS.get(base_version)
    if trigger_range is None and isinstance(ref_trigger, float):
        trigger_range = (ref_trigger, ref_trigger)
    if trigger_range is None:
        known = fs.KNOWN_TRIGGER_VERSIONS.values()
        trigger_range = (min(lo for lo, _ in known), max(hi for _, hi in known))
        result.notes.append(f"No trigger versions known for base {base_version}: any of {trigger_range} plus headroom is plausible.")

    stride = _terrain_stride(structure)
    for f in [*files, *diff_files]:
        grid = fs.scan_terrain_grid(f.body.data, stride, terrain_ids)
        if isinstance(grid, fs.GridMiss):
            result.refusal = f"{f.path.name}: {grid.reason}"
            if grid.unknown_ids:
                result.refusal += " (regenerate descape/terrain_classes.json for the patched game first)"
            result.report_lines = render_report(result)
            return result
        f.grid = grid

    sections = _section_names(structure)
    if sections[0] != sw.HEADER_SECTION or "Map" not in sections or "Units" not in sections or "Triggers" not in sections:
        raise InferInputError(f"base {base_version} does not have the FileHeader..Map, Units, Triggers section order")
    pregrid = sections[1 : sections.index("Map") + 1]
    after_units = sections[sections.index("Units") + 1 :]
    names = Names(version)
    spent = [0]
    ref = Reference(ref_values) if ref_values is not None else None

    def inputs_for(segment: str, file_list: Sequence[ScenarioFile], starts: Mapping[Path, int] | None = None) -> list[SegmentInput]:
        out = []
        for f in file_list:
            ctx = FilterContext((f.grid.width, f.grid.height), trigger_range)
            r = ref if f is files[0] else None
            if segment == SEG_HEADER:
                out.append(SegmentInput(f.path.name, f.raw, 0, _end_anchor(f.body.start), ctx, r))
            elif segment == SEG_PREGRID:
                out.append(SegmentInput(f.path.name, f.body.data, 0, _grid_anchor(f.grid), ctx, r))
            elif segment == SEG_UNITS:
                out.append(SegmentInput(f.path.name, f.body.data, f.grid.end, _trigger_version_anchor(f.body.data, trigger_range), ctx, r))
            else:
                out.append(SegmentInput(f.path.name, f.body.data, starts[f.path], _end_anchor(len(f.body.data)), ctx, r))
        return out

    segments = ((SEG_HEADER, [sw.HEADER_SECTION]), (SEG_PREGRID, pregrid), (SEG_UNITS, ["Units"]), (SEG_TRIGGERS, after_units))
    starts: dict[Path, int] = {}
    for name, secs in segments:
        if name == SEG_TRIGGERS:
            compiled = sw.compile_structure({"Units": structure["Units"]})
            for f in [*files, *diff_files]:
                try:
                    starts[f.path] = sw.walk(compiled, f.body.data, f.grid.end, ["Units"], record=False).end
                except sw.WalkError as e:
                    raise AssertionError(f"Units no longer walks on {f.path.name} after its own segment fit") from e
        diff = None
        if diff_files:
            a, b = inputs_for(name, diff_files, starts)
            diff = (a, b)
        segment_inputs = inputs_for(name, files, starts) + (list(diff) if diff else [])
        search = SegmentSearch(name, secs, segment_inputs, names, budget, spent, diff, diff_expect_new)
        outcome = search.run(structure)
        structure = outcome.structure
        for edit, choice, evidence in outcome.edits:
            result.edits.append(EditRecord(name, edit, evidence, choice.note, choice.rivals, choice.ambiguous))
            if choice.ambiguous:
                span = sorted({e.index for e in [edit, *choice.rivals] if e.block == edit.block})
                result.unresolved.append(
                    f"{name}: {'/'.join(edit.block)} positions {span[0]}..{span[-1]}: {choice.note}; kept {edit.describe()}"
                )
        if outcome.refused is not None:
            result.refusal = f"{name}: {outcome.refused}"
            result.structure = structure
            result.report_lines = render_report(result)
            return result

    structure = emit(structure, result, [*files, *diff_files], version, base_version)
    result.structure = structure
    result.notes.extend(_coverage_notes(structure, files, sections))
    result.notes.extend(_vocabulary_notes(structure, files, sections, base_version, vocabulary))
    result.notes.extend(_unlinked_notes(result))
    result.status = STATUS_AMBIGUOUS if result.unresolved and not accept_ambiguous else STATUS_INFERRED
    if library_check is not None:
        failure = _library_gate(structure, [*files, *diff_files], library_check)
        if failure is not None:
            result.status = STATUS_REFUSED
            result.refusal = f"the library does not parse the inferred structure: {failure}"
    result.report_lines = render_report(result)
    return result


def _library_gate(structure: dict, files: Sequence[ScenarioFile], check: LibraryCheck) -> str | None:
    with tempfile.TemporaryDirectory(prefix="structure_infer_") as tmp:
        path = Path(tmp) / "structure.json"
        path.write_text(json.dumps(structure, indent=2), encoding="utf-8")
        for f in files:
            record = check(f.path, path)
            if record.get("status") != "CLEAN" or record.get("remainder") != 0:
                where = record.get("section"), record.get("retriever"), record.get("detail")
                return f"{f.path.name}: {record.get('status')} remainder={record.get('remainder')} {where}"
    return None


# --- emit -------------------------------------------------------------------


def _most_common(values: Sequence[Any]) -> Any:
    keyed = Counter(_key(v) for v in values)
    if not keyed:
        return None
    top = keyed.most_common(1)[0][0]
    return next(v for v in values if _key(v) == top)


def _default_for(type_name: str, value: Any) -> Any:
    if value is None:
        if type_name.startswith(("str", "c")):
            return ""
        if type_name.isdigit():
            return "00" * int(type_name)
        return -1 if type_name.startswith("s") else 0
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, list):
        return [_default_for(type_name, v) for v in value]
    return value


def _field_values(files: Sequence[ScenarioFile], structure: dict, block: tuple[str, ...], name: str) -> list[Any]:
    compiled = sw.compile_structure(structure)
    prefixes = block_prefixes(compiled).get(block, (None, []))[1]
    out: list[Any] = []
    for f in files:
        try:
            walked = sw.walk_file(compiled, f.raw)
        except (sw.WalkError, sw.StructureError):
            continue
        for w in (walked.header, walked.body):
            for prefix in prefixes:
                col = w.columns.get((*prefix, name))
                if col is not None:
                    out.extend(col.value_list())
        if out:
            break  # the first file that parses: the reference re-save when there is one
    return out


def emit(structure: dict, result: InferResult, files: Sequence[ScenarioFile], version: str, base_version: str) -> dict:
    """Defaults, comments and version fields for the inferred structure."""
    out = json.loads(json.dumps(structure))
    for record in result.edits:
        edit = record.edit
        retrievers = block_dict(out, edit.block)["retrievers"]
        for name, _ in edit.fields:
            definition = retrievers.get(name)
            if definition is None:
                continue  # a later edit removed it
            values = _field_values(files, out, edit.block, name)
            definition["default"] = _default_for(definition["type"], _most_common(values))
            how = "ambiguous, see INFERRED.md" if record.ambiguous else record.note
            definition["comment"] = f"Inferred for {version} from {base_version} ({edit.kind}; {how})"
        spec_block = sw.compile_structure({edit.block[0]: out[edit.block[0]]})
        spec = block_prefixes(spec_block)[edit.block][0]
        count = count_field(spec)
        if count is not None:
            values = _field_values(files, out, edit.block, count)
            if values:
                retrievers[count]["default"] = _most_common(values)
    for section, name in (("FileHeader", "version"), ("DataHeader", "version"), ("Triggers", "trigger_version")):
        retrievers = out.get(section, {}).get("retrievers", {})
        if name in retrievers:
            values = _field_values(files, out, (section,), name)
            if values:
                value = _most_common(values)
                if section == "DataHeader" and isinstance(value, float):
                    value = float(f"{value:.6g}")
                retrievers[name]["default"] = value
    return out


def _coverage_notes(structure: dict, files: Sequence[ScenarioFile], sections: Sequence[str]) -> list[str]:
    compiled = sw.compile_structure(structure)
    seen: set[tuple[str, ...]] = set()
    for f in files:
        try:
            walked = sw.walk_file(compiled, f.raw, record=True)
        except (sw.WalkError, sw.StructureError):
            continue
        for block, (spec, prefixes) in block_prefixes(compiled).items():
            for prefix in prefixes:
                for w in (walked.header, walked.body):
                    col = w.columns.get((*prefix, spec.fields[0].name))
                    if col is not None and len(col):
                        seen.add(block)
    unseen = [b for b in block_prefixes(compiled) if b not in seen]
    if not unseen:
        return []
    listed = ", ".join("/".join(b) for b in unseen)
    return [f"Blocks with no instance in any input, so unverified: {listed}. A sample that uses them checks them."]


def _vocabulary_notes(structure, files, sections, base_version, vocabulary) -> list[str]:
    if vocabulary is None:
        try:
            from descape import library_compat

            raw = library_compat.vocabulary_json(base_version)
            vocabulary = {kind: {int(k) for k in raw[kind]} for kind in ("conditions", "effects")}
        except (FileNotFoundError, ValueError, KeyError) as e:
            return [f"Vocabulary not checked: {e}"]
    compiled = sw.compile_structure(structure)
    used: dict[str, set[int]] = {"conditions": set(), "effects": set()}
    paths = {
        "effects": ("Triggers", "trigger_data", "effect_data", "effect_type"),
        "conditions": ("Triggers", "trigger_data", "condition_data", "condition_type"),
    }
    for f in files:
        try:
            walked = sw.walk_file(compiled, f.raw)
        except (sw.WalkError, sw.StructureError):
            continue
        for kind, path in paths.items():
            col = walked.body.columns.get(path)
            if col is not None:
                used[kind].update(int(v) for v in col.value_list())
    lines = []
    for kind in ("conditions", "effects"):
        missing = sorted(used[kind] - set(vocabulary[kind]))
        if missing:
            lines.append(f"The samples use {kind[:-1]} type ids {missing}, which the {base_version} {kind}.json lacks.")
    if not lines:
        lines.append(f"Every condition and effect type id the samples use is in the {base_version} vocabulary.")
    return lines


_CARRIERS = {"UnitStruct": "Unit", "ConditionStruct": "Condition"}


def _unlinked_notes(result: InferResult) -> list[str]:
    lines = []
    for record in result.edits:
        block = record.edit.block[-1]
        for name in record.edit.new_names():
            carrier = _CARRIERS.get(block)
            if carrier is not None:
                lines.append(f'UNLINKED_FIELDS: {carrier}: {{"{name}": <default>}} (descape/unlinked_fields.py)')
            elif block == "EffectStruct":
                lines.append(f"EffectStruct.{name} has no UNLINKED_FIELDS carrier (Unit and Condition only): the release needs one.")
    return lines


def render_report(result: InferResult) -> list[str]:
    """INFERRED.md, as lines."""
    lines = [f"# Inferred structure: {result.version} from {result.base_version}", "", f"Status: {result.status}", ""]
    if result.refusal:
        lines += [f"Refused: {result.refusal}", ""]
    lines += ["## Inputs", ""]
    lines += [f"- {role}: `{Path(path).name}` sha256 `{sha}`" for role, path, sha in result.inputs]
    lines += ["", "## Edits", ""]
    if not result.edits:
        lines.append("- none")
    for r in result.edits:
        lines.append(f"- {r.segment}: {r.edit.describe()}, new fields {r.edit.new_names() or '-'}")
        lines.append(f"  - evidence: {r.evidence}")
        lines.append(f"  - decided by: {r.note}")
        if r.rivals:
            shown = "; ".join(e.describe() for e in r.rivals[:8])
            more = f" and {len(r.rivals) - 8} more" if len(r.rivals) > 8 else ""
            lines.append(f"  - rivals: {shown}{more}")
    if result.unresolved:
        lines += ["", "## Unresolved", ""]
        lines += [f"- {u}" for u in result.unresolved]
        lines.append("- A differential pair (one in-game edit of the new feature, saved as a second file) resolves a span.")
    if result.notes:
        lines += ["", "## Notes", ""]
        lines += [f"- {n}" for n in result.notes]
    return lines
