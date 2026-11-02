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
from collections.abc import Collection
from dataclasses import dataclass

import numpy as np

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


def find_body(raw: bytes, *, limit: int = BODY_SEARCH_LIMIT, min_bytes: int = MIN_BODY_BYTES, cap: int = BODY_CAP) -> Body | str:
    """The one offset in [BODY_SEARCH_START, limit) where a raw-deflate
    stream starts, ends, and inflates to at least `min_bytes` (at most
    `cap`). Returns the Body, or a reason string when no start or more than
    one start qualifies."""
    found: list[Body] = []
    for start in range(BODY_SEARCH_START, min(limit, len(raw))):
        result = _inflate(raw, start, cap)
        if result is None or len(result[0]) < min_bytes:
            continue
        found.append(Body(start, result[0], result[1]))
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
