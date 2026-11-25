"""Own tile `(int(u.x), int(u.y))` -> the units standing there, kept current by
UnitEditModel's deltas rather than rebuilt on every unit edit.

`render_units.unit_own_tile_index()` serves an `OwnTileIndex` while it is
current: its `gen` equals `scenario.unit_gen` and `is_live()` holds. Any other
gen move (a mutator that applies no delta, a test or tool that bumps unit_gen
by hand after a direct list edit, a whole-list restore(), a dropped model)
leaves it stale, and the next read builds a fresh one. So an unwired writer
costs a rebuild, never a wrong answer.

The buckets hold units only. Each unit's `(player_id, i)` is resolved on read:
from the build's own walk until the first delta, then through the last
mutating model's `_locate()` (held weakly, so the memo never keeps a model or
its scenario alive). That is why reassign, swap and the tail shift of a
removal need no delta: the model already keeps `_pos` current.

A new x/y or membership writer in UnitEditModel passes its delta to
`_bump_unit_gen()`, or passes no index and leaves this to rebuild. It never
stamps the index without the delta: a stamped index with a wrong bucket is the
one way to get a stale answer, a missed co-occupant in the shared-tile splice.

Pure, no Qt. Main thread only: buckets mutate in place, and no worker job reads
the index.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Iterable


def own_tile(unit) -> tuple[int, int]:
    """The unit's own tile, the key every reader of the index looks up."""
    return int(unit.x), int(unit.y)


class OwnTileIndex:
    """Own tile -> units, with each unit's (player_id, i) resolved on read."""

    __slots__ = ("__weakref__", "_buckets", "_locate_ref", "_positions", "_scenario_ref", "gen")

    def __init__(self, buckets: dict, positions: dict, gen: int | None, scenario_ref=None):
        self._buckets: dict[tuple[int, int], list] = buckets
        # id(unit) -> (player_id, i) from the build's walk; dropped at the first stamp.
        self._positions: dict[int, tuple[int, int]] | None = positions
        # WeakMethod of the last mutating model's _locate, once a delta landed.
        self._locate_ref: weakref.WeakMethod | None = None
        self._scenario_ref = scenario_ref
        self.gen = gen

    @classmethod
    def build(cls, scenario, gen: int | None) -> OwnTileIndex:
        """A fresh walk of every unit list, player-major then list index, filter-free."""
        buckets: dict[tuple[int, int], list] = {}
        positions: dict[int, tuple[int, int]] = {}
        for player_id, units in enumerate(scenario.unit_manager.units):
            for i, u in enumerate(units):
                buckets.setdefault((int(u.x), int(u.y)), []).append(u)
                positions[id(u)] = (player_id, i)
        # A duck-typed test scenario (no unit_gen) is never memoized and needs no back-reference.
        return cls(buckets, positions, gen, weakref.ref(scenario) if gen is not None else None)

    def is_live(self) -> bool:
        """False once the model whose locator this index reads has been collected."""
        return self._positions is not None or (self._locate_ref is not None and self._locate_ref() is not None)

    def stamp(self, gen: int, locate: Callable) -> None:
        """Marks the index current at `gen` and reads positions through `locate`
        (a model's bound `_locate`) from now on. Only a mutator that applied its
        delta, or that moved no unit's own tile or membership, may call this."""
        held = self._locate_ref() if self._locate_ref is not None else None
        if held != locate:
            self._locate_ref = weakref.WeakMethod(locate)
        self._positions = None
        self.gen = gen

    def _where(self, locate, lists, unit, tile) -> tuple[int, int]:
        if locate is None:
            where = self._positions.get(id(unit))
        else:
            try:
                where = locate(unit)
            except ValueError:
                where = None
            # A model whose own lists fell out of step (a second model's membership edit) answers stale.
            if where is not None and not (where[1] < len(lists[where[0]]) and lists[where[0]][where[1]] is unit):
                where = None
        if where is None:
            raise RuntimeError(
                f"own-tile index drift: tile {tile} holds reference_id {getattr(unit, 'reference_id', '?')}, "
                f"which the unit lists do not hold where its locator says"
            )
        return where

    def get(self, tile, default=()):
        """The units on `tile` as [(player_id, i, unit)], sorted by (player_id, i):
        a fresh walk's order. `default` if no unit stands there."""
        bucket = self._buckets.get(tile)
        if bucket is None:
            return default
        locate = lists = None
        if self._positions is None:
            locate = self._locate_ref() if self._locate_ref is not None else None
            scenario = self._scenario_ref() if self._scenario_ref is not None else None
            if locate is None or scenario is None:
                raise RuntimeError(f"own-tile index read at tile {tile} after its model or scenario was collected")
            lists = scenario.unit_manager.units
        out = [(*self._where(locate, lists, u, tile), u) for u in bucket]
        if len(out) > 1:
            out.sort(key=lambda e: (e[0], e[1]))
        return out

    def tiles(self) -> Iterable[tuple[int, int]]:
        """Every own tile some unit stands on."""
        return self._buckets.keys()

    def as_dict(self) -> dict[tuple[int, int], list[tuple[int, int, object]]]:
        """{tile: [(player_id, i, unit)]}, the shape the old memoized dict had: the tests' oracle form."""
        return {tile: self.get(tile) for tile in self._buckets}

    # -- deltas, applied by UnitEditModel after its mutation --------------------

    def _drop(self, unit, tile) -> None:
        bucket = self._buckets.get(tile)
        if bucket is not None:
            for k, u in enumerate(bucket):
                if u is unit:
                    del bucket[k]
                    if not bucket:
                        del self._buckets[tile]
                    return
        raise RuntimeError(
            f"own-tile index drift: reference_id {getattr(unit, 'reference_id', '?')} is not in its bucket {tile}"
        )

    def moved(self, unit, old_tile: tuple[int, int]) -> None:
        """`unit`'s x/y were written; `old_tile` is its own tile before the write."""
        new_tile = own_tile(unit)
        if new_tile != old_tile:
            self._drop(unit, old_tile)
            self._buckets.setdefault(new_tile, []).append(unit)

    def added(self, units: Iterable) -> None:
        """`units` were appended to the unit lists."""
        for u in units:
            self._buckets.setdefault(own_tile(u), []).append(u)

    def removed(self, units: Iterable) -> None:
        """`units` left the unit lists. A removal writes no x/y, so each is found at its current own tile."""
        for u in {id(u): u for u in units}.values():
            self._drop(u, own_tile(u))


def check(scenario) -> OwnTileIndex | None:
    """Raises RuntimeError naming the first differing tile when the memoized
    index, if current, differs from a fresh walk. Returns that index, or None
    when there is none current to check. The tests' oracle; never called on a
    hot path (a fresh walk is exactly the cost the index exists to avoid)."""
    from descape import render_units

    index = render_units.live_own_tile_index(scenario)
    if index is None:
        return None

    def ids(mapping):
        return {tile: [(p, i, id(u)) for p, i, u in entries] for tile, entries in mapping.items()}

    live, fresh = ids(index.as_dict()), ids(OwnTileIndex.build(scenario, scenario.unit_gen).as_dict())
    for tile in sorted(live.keys() | fresh.keys()):
        if live.get(tile) != fresh.get(tile):
            raise RuntimeError(f"own-tile index differs from a fresh walk at {tile}: {live.get(tile)} vs {fresh.get(tile)}")
    return index
