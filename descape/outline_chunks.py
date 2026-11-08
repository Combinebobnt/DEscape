"""Which map chunks of a unit-outline overlay (View > Footprint Outlines, Fish
Outlines) an edit made stale, so MapView rebuilds only those (TASK-031.42).
Qt-free; MapView builds the paths and ChunkedPathItem holds one per chunk.

The map is cut into CHUNK_TILES-square chunks, and a unit belongs to the chunk
holding its own tile (own_x, own_y). Each entry's record is its key plus its
signature: (player_id, reference_id, unit_const, x, y, own_x, own_y), every
per-unit input of unit_pick.unit_polygons() plus the player, which picks the
owner group. plan() diffs this pass's record set against the last one's and
marks a chunk dirty when:

- a record appeared or went away: an add, a remove, or a changed signature,
  which is both at once, so a move dirties its old and its new chunk;
- a record occurs more than once (a corrupt file's exact duplicate): its
  chunk, on every pass while it does and the pass after, since a set cannot
  see a copy count go from 2 to 1;
- a height field the geometry reads changed under a unit: every in-scope
  entry in index.by_tile of a changed tile. by_tile holds the unit's whole
  unit_tile_bounds() rect, a superset of unit_occupied_tiles() that also holds
  the own tile unit_rise_px_for() reads, so no reader is missed. A corner-grid
  field (Sloped's corner_rise) maps a changed corner to its up to four tiles.

Everything else unit_polygons() reads (style, tile size, map dims,
projection, the drape, a data mod's spans) is the caller's `context`, plus
the height fields' shapes and CHUNK_TILES. Any change there, or a reset(),
makes the plan full. A new input added to unit_polygons() must join the
record or the context, or an outline goes stale with every test of the old
inputs green.

A dirty chunk's members are gathered from index.by_tile over the chunk's own
tiles, so `entries` must be entries of `index`, and sorted back into index
order: Qt's cosmetic stroke draws a few pixels differently with subpath order
at low zoom, so an incremental chunk must match a full plan's exactly.
"""

from __future__ import annotations

import operator
from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field

import numpy as np

# Benched 8/16/32 on old-allies (tools/bench_footprint_refresh.py --chunk): 8 was best for a Sloped All edit.
CHUNK_TILES = 8

_RECORD = operator.attrgetter(
    "player_id", "unit.reference_id", "unit.unit_const", "unit.x", "unit.y", "own_x", "own_y"
)
_ORDER = operator.attrgetter("order")


def chunk_of(x: int, y: int) -> tuple[int, int]:
    """The chunk holding tile (x, y). Reads CHUNK_TILES at call time, so a test can shrink it."""
    return (x // CHUNK_TILES, y // CHUNK_TILES)


@dataclass(frozen=True)
class HeightField:
    """A height array the geometry reads: tile-indexed [y, x], or a corner grid one larger per axis."""

    array: np.ndarray
    corners: bool = False


@dataclass
class OutlinePlan:
    """`dirty`: each dirty chunk that has members, with all of them. `emptied`:
    the dirty chunks with none left. `full`: every chunk is dirty, and a chunk
    absent from `dirty` holds nothing (clear first)."""

    full: bool
    dirty: dict[tuple[int, int], list] = field(default_factory=dict)
    emptied: set[tuple[int, int]] = field(default_factory=set)


class OutlineChunks:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Forgets everything: the next plan() is full."""
        self._records: set | None = None
        self._dups: set = set()
        self._context = None
        self._heights: list[np.ndarray] = []

    def plan(self, entries: Sequence, index, context: Hashable, heights: Sequence[HeightField] = ()) -> OutlinePlan:
        tiles = CHUNK_TILES
        # _RECORD's fields, spelled out: a comprehension measured ~2x faster than map(_RECORD).
        records = [(e.player_id, (u := e.unit).reference_id, u.unit_const, u.x, u.y, e.own_x, e.own_y) for e in entries]
        current = set(records)
        dups = set()
        if len(current) != len(records):
            dups = {record for record, count in Counter(records).items() if count > 1}
        full_context = (context, tiles, tuple((h.array.shape, h.corners) for h in heights))
        if self._records is None or full_context != self._context:
            self._records, self._dups, self._context = current, dups, full_context
            self._heights = [np.array(h.array, copy=True) for h in heights]
            members: dict = {}
            for entry, record in zip(entries, records, strict=True):
                chunk = (record[5] // tiles, record[6] // tiles)
                bucket = members.get(chunk)
                if bucket is None:
                    members[chunk] = [entry]
                else:
                    bucket.append(entry)
            return OutlinePlan(full=True, dirty=members)

        changed = current ^ self._records
        changed |= dups | self._dups
        dirty = {(record[5] // tiles, record[6] // tiles) for record in changed}
        for field_, snapshot in zip(heights, self._heights, strict=True):
            changed_tiles = field_.array != snapshot
            if not changed_tiles.any():
                continue
            ys, xs = np.nonzero(changed_tiles)
            if field_.corners:
                h, w = field_.array.shape[0] - 1, field_.array.shape[1] - 1
                touched = {
                    (tx, ty)
                    for cx, cy in zip(xs.tolist(), ys.tolist(), strict=True)
                    for ty in (cy - 1, cy)
                    for tx in (cx - 1, cx)
                    if 0 <= tx < w and 0 <= ty < h
                }
            else:
                touched = set(zip(xs.tolist(), ys.tolist(), strict=True))
            for tile in touched:
                for order in index.by_tile.get(tile, ()):
                    entry = index.entries[order]
                    if _RECORD(entry) in current:
                        dirty.add((entry.own_x // tiles, entry.own_y // tiles))
            np.copyto(snapshot, field_.array)
        self._records, self._dups = current, dups

        plan = OutlinePlan(full=False)
        by_tile, indexed = index.by_tile, index.entries
        for cx, cy in dirty:
            bucket = []
            for ty in range(cy * tiles, cy * tiles + tiles):
                for tx in range(cx * tiles, cx * tiles + tiles):
                    for order in by_tile.get((tx, ty), ()):
                        entry = indexed[order]
                        if entry.own_x == tx and entry.own_y == ty and _RECORD(entry) in current:
                            bucket.append(entry)
            if bucket:
                # Index order, as a full plan lists them: subpath order changes a few stroked pixels.
                bucket.sort(key=_ORDER)
                plan.dirty[(cx, cy)] = bucket
            else:
                plan.emptied.add((cx, cy))
        return plan
