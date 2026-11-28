"""Finds a scenario file's fixed landmarks without knowing its structure:
where the compressed body starts, where the terrain grid sits, and whether
a value is a believable trigger version.

tools/infer_structure.py splits a file of an unknown scenario version into
segments at these landmarks, so a layout change can be located before any
structure is trusted. Pure functions, no AoE2ScenarioParser import, so the
in-app compat view (TASK-057) can use them in the frozen build.

The constants are measured, not guessed: 144 corpus files (examples/ plus
the maintainer corpus, 1.37 to 1.59) walked under their own structures on
2026-11-01 (TASK-056 step 0). tests/test_format_scan.py pins the ones that
mirror another module's constant.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Callable, Collection
from dataclasses import dataclass

import numpy as np

from descape.structure_walk import Column, Walk

# The body's raw-deflate stream starts right after the FileHeader. The
# header's own header_length field is 0 in every corpus file, so the start is
# found by trial. Largest header measured: 2880 bytes.
BODY_SEARCH_START = 8  # past the c4 version and the u32 header_length
BODY_SEARCH_LIMIT = 8192
# A real body holds at least the fixed parts of every section; the smallest
# measured is 123598 bytes (a 120x120 map). Rejects a short accidental stream.
MIN_BODY_BYTES = 1 << 16
BODY_CAP = 1 << 30

# Map edge lengths: the corpus spans 120..480 (all square). 480 is
# scenario_new.MAX_MAP_TILES. 120 is the smallest real map, not
# scenario_new.MIN_MAP_TILES (61), which is the New Map generator's limit.
MAP_TILES_MIN = 120
MAP_TILES_MAX = 480
# Highest elevation measured on any tile (and iso_geometry.MAX_ELEVATION).
ELEVATION_MAX = 15
# Units opens with number_of_unit_sections, 9 in every file and version.
UNITS_FIRST_VALUE = 9

# Trigger versions per scenario version, measured on the corpus. The
# structure files' own trigger_version defaults are stale (4.7 in 1.58 and
# 1.59, whose files carry 4.9 and 5.0), so they are never used.
KNOWN_TRIGGER_VERSIONS: dict[str, tuple[float, float]] = {
    "1.37": (2.2, 2.2),
    "1.40": (2.4, 2.4),
    "1.41": (2.4, 2.4),
    "1.42": (2.4, 2.4),
    "1.43": (2.4, 2.4),
    "1.45": (2.5, 2.6),
    "1.47": (3.2, 3.5),
    "1.48": (3.5, 3.5),
    "1.49": (3.5, 3.5),
    "1.51": (3.5, 3.5),
    "1.53": (3.6, 3.6),
    "1.54": (4.1, 4.1),
    "1.55": (4.4, 4.5),
    "1.56": (4.5, 4.5),
    "1.57": (4.7, 4.7),
    "1.58": (4.9, 4.9),
    "1.59": (5.0, 5.0),
}
TRIGGER_VERSION_HEADROOM = 2.0

# The in-app compat view's bounds (TASK-057), lower than the tool's above
# because they must admit DEscape's own New Map output. Measured 2026-11-25
# over 170 files (New Map 61..119 tiles in both games, the library's
# v1.54..v1.59 defaults, the corpus, tests/fixtures): one body and one grid
# each, at the offsets the normal load finds.
COMPAT_MAP_TILES_MIN = 61  # scenario_new.MIN_MAP_TILES; admitted no second grid anywhere
# Half the smallest body measured (47307, a 61x61 Return of Rome New Map);
# the largest accidental stream found was 22166 bytes.
COMPAT_MIN_BODY_BYTES = 47307 // 2

# Unit plausibility filters: every unit of every corpus file passes them.
UNIT_STATUS_RANGE = (0, 7)
UNIT_CONST_RANGE = (0, 4095)  # corpus 1..2708, the frozen re-save keeps a const 0; headroom above
UNIT_Z_RANGE = (0.0, float(ELEVATION_MAX))
CAPTURE_FLAG_RANGE = (-1, 3)
CAPTION_STRING_ID_MIN = -1
UNITS_PREFIX = ("Units", "players_units", "units")


@dataclass(frozen=True)
class Body:
    """Where the compressed body starts and what it inflates to.
    `trailing` counts bytes after the deflate stream's end, which the game
    ignores (one 1.42 corpus file carries 21078)."""

    start: int
    data: bytes
    trailing: int


def _inflate(raw: bytes, start: int, cap: int) -> tuple[bytes, int] | None:
    inflater = zlib.decompressobj(-zlib.MAX_WBITS)
    try:
        data = inflater.decompress(memoryview(raw)[start:], cap + 1)
    except zlib.error:
        return None
    if not inflater.eof or len(data) > cap:
        return None
    return data, len(inflater.unused_data)


def inflate_at(raw: bytes, start: int, cap: int = BODY_CAP) -> Body | None:
    """The raw-deflate stream starting at `start`, or None when none starts
    there, it doesn't end inside `raw`, or it inflates past `cap`."""
    result = _inflate(raw, start, cap)
    return None if result is None else Body(start, result[0], result[1])


def find_body(raw: bytes, *, limit: int = BODY_SEARCH_LIMIT, min_bytes: int = MIN_BODY_BYTES, cap: int = BODY_CAP) -> Body | str:
    """The one offset in [BODY_SEARCH_START, limit) where a raw-deflate
    stream starts, ends, and inflates to at least `min_bytes` (at most
    `cap`). A start whose stream ends where a later one's does, with that
    later stream's output as its tail, is the later stream with junk in
    front and doesn't count. Returns the Body, or a reason string when no
    start or more than one start qualifies."""
    found: list[Body] = []
    for start in range(BODY_SEARCH_START, min(limit, len(raw))):
        result = _inflate(raw, start, cap)
        if result is None or len(result[0]) < min_bytes:
            continue
        found.append(Body(start, result[0], result[1]))
    # A start a few bytes early can decode junk, then fall into step with the
    # real stream at a block boundary: same end, output = junk + the real data.
    found = [a for a in found if not any(b.start > a.start and b.trailing == a.trailing and a.data.endswith(b.data) for b in found)]
    # A stream that runs to the end of the file wins over one with trailing
    # bytes: a short accidental stream can start inside the body (one corpus
    # file has one at 4266, 59341 bytes before the end).
    to_end = [b for b in found if b.trailing == 0]
    pool = to_end or found
    if not pool:
        return f"no raw-deflate stream of at least {min_bytes} bytes starts in bytes {BODY_SEARCH_START}..{limit}"
    if len(pool) > 1:
        return f"more than one deflate start qualifies: {[b.start for b in pool]}"
    return pool[0]


@dataclass(frozen=True)
class GridMatch:
    """The terrain grid: `offset` is where u32 map_width sits, tiles start
    8 bytes later, and `end` (where Units starts) is just past the last
    tile."""

    offset: int
    width: int
    height: int
    stride: int

    @property
    def tiles_start(self) -> int:
        return self.offset + 8

    @property
    def end(self) -> int:
        return self.tiles_start + self.width * self.height * self.stride


@dataclass(frozen=True)
class GridMiss:
    reason: str
    unknown_ids: tuple[int, ...] = ()


def _candidates(body: bytes, stride: int, size_range: tuple[int, int]) -> list[tuple[int, int, int]]:
    """(offset, w, h) where two u32s in range are followed, past w*h tiles,
    by the u32 Units anchor."""
    lo, hi = size_range
    found = []
    for align in range(4):
        count = (len(body) - align) // 4
        if count < 3:
            continue
        words = np.frombuffer(body, dtype="<u4", count=count, offset=align)
        ok = (words >= lo) & (words <= hi)
        for i in np.flatnonzero(ok[:-1] & ok[1:]):
            offset = align + 4 * int(i)
            w, h = int(words[i]), int(words[i + 1])
            anchor = offset + 8 + w * h * stride
            if anchor + 4 <= len(body) and int.from_bytes(body[anchor : anchor + 4], "little") == UNITS_FIRST_VALUE:
                found.append((offset, w, h))
    return sorted(found)


def scan_terrain_grid(
    body: bytes,
    stride: int,
    terrain_ids: Collection[int],
    size_range: tuple[int, int] = (MAP_TILES_MIN, MAP_TILES_MAX),
    elevation_bound: int = ELEVATION_MAX,
    *,
    id_offset: int = 0,
    elevation_offset: int = 1,
) -> GridMatch | GridMiss:
    """The one place in `body` that reads as map_width, map_height, then
    width*height tiles of `stride` bytes whose terrain ids are all in
    `terrain_ids` and whose elevations are all <= `elevation_bound`, then
    the u32 9 that opens Units. A candidate failing only on terrain ids
    names them, since a game update can add terrains (regenerate the table
    first)."""
    known = np.fromiter(sorted(set(terrain_ids)), dtype=np.int64)
    matches: list[GridMatch] = []
    unknown: set[int] = set()
    rejected = 0
    for offset, w, h in _candidates(body, stride, size_range):
        tiles = np.frombuffer(body, dtype=np.uint8, count=w * h * stride, offset=offset + 8).reshape(-1, stride)
        if int(tiles[:, elevation_offset].max()) > elevation_bound:
            rejected += 1
            continue
        ids = np.unique(tiles[:, id_offset])
        missing = ids[~np.isin(ids, known)]
        if missing.size:
            unknown.update(int(x) for x in missing)
            continue
        matches.append(GridMatch(offset, w, h, stride))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return GridMiss(f"{len(matches)} grid-shaped matches at offsets {[m.offset for m in matches]}")
    if unknown:
        return GridMiss(f"terrain ids not in the terrain table: {sorted(unknown)}", tuple(sorted(unknown)))
    return GridMiss(f"no terrain grid found ({rejected} candidate(s) had elevations over {elevation_bound})")


def plausible_trigger_version(value: float, known_range: tuple[float, float]) -> bool:
    """True when `value` is finite, has one decimal place, and lies between
    the lowest trigger version known for the base scenario version and the
    highest plus TRIGGER_VERSION_HEADROOM."""
    if not isinstance(value, float) or not math.isfinite(value):
        return False
    tenths = value * 10
    if abs(tenths - round(tenths)) > 1e-6:
        return False
    lo, hi = known_range
    return lo - 1e-9 <= value <= hi + TRIGGER_VERSION_HEADROOM + 1e-9


def column_array(walk: Walk, path: tuple[str, ...]) -> tuple[Column, np.ndarray] | None:
    """A walked column as a numpy array, None when absent, empty or not all ints or all floats."""
    col = walk.columns.get(path)
    if col is None or not len(col) or col.values is None:
        return None
    values = col.values
    if not isinstance(values, list):
        return col, np.frombuffer(values, dtype=np.int64 if values.typecode == "q" else np.float64)
    if all(type(v) is int for v in values):
        try:
            return col, np.asarray(values, dtype=np.int64)
        except OverflowError:
            return None
    if all(type(v) is float for v in values):
        return col, np.asarray(values, dtype=np.float64)
    return None


def unit_filter_violations(
    walk: Walk,
    prefix: tuple[str, ...],
    map_size: tuple[int, int] | None,
    note: Callable[[Column, np.ndarray, str], None],
) -> None:
    """Calls `note(column, bad_mask, reason)` for each unit filter over the
    UnitStruct fields read under `prefix`: x/y inside the map, z, status,
    unit_const, capture_flag and caption_string_id in range, reference_id
    non-negative and unique."""

    def get(name: str):
        return column_array(walk, (*prefix, name))

    w, h = map_size if map_size else (math.inf, math.inf)
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


def unit_filters(walk: Walk, width: int, height: int, prefix: tuple[str, ...] = UNITS_PREFIX) -> str | None:
    """The first unit filter a walk's units fail, as "<reason> at <offset>",
    or None when every unit passes."""
    best: tuple[int, str] | None = None

    def note(col: Column, bad: np.ndarray, why: str) -> None:
        nonlocal best
        hits = np.flatnonzero(bad)
        if hits.size:
            off = col.offsets[int(hits[0])]
            if best is None or off < best[0]:
                best = (off, why)

    unit_filter_violations(walk, prefix, (width, height), note)
    return None if best is None else f"{best[1]} at byte {best[0]}"
