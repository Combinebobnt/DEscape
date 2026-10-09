"""Qt-free search over placed objects for Edit > Find and Replace (GH #144).

Walks unit_references.all_units(), never the pick index: filtered,
garrisoned and off-map units are all searchable, and keys resolve the same
way (resolve_keys()). Rows hold plain values and a (player, reference_id)
key, never a Unit, so a stale row can only fail to resolve.
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from descape import object_catalog, unit_kind
from descape.find_text import FindPatternError, compile_pattern, refuse_forbidden_path
from descape.render import unit_tile_bounds
from descape.terrain_palette import TREE_UNIT_IDS
from descape.unit_references import all_units

__all__ = [
    "CATEGORIES",
    "KINDS",
    "FindCriteria",
    "FindPatternError",
    "FindRow",
    "compile_pattern",
    "find",
    "resolve_keys",
    "rows_to_csv",
]

CATEGORIES = ("Units", "Buildings", "Heroes", "Others")
KINDS = ("trees", "eye_candy", "walls", "invisible")


def _kind_consts(kind: str) -> frozenset[int]:
    if kind == "trees":
        return TREE_UNIT_IDS
    if kind == "eye_candy":
        return unit_kind.eye_candy_consts()
    if kind == "walls":
        return unit_kind.wall_consts()
    if kind == "invisible":
        return unit_kind.invisible_consts()
    raise ValueError(f"unknown kind {kind!r}")


@lru_cache(maxsize=1)
def _category_by_const() -> dict[int, str]:
    return {entry.id: entry.category for entry in object_catalog.objects()}


def category_of(unit_const: int) -> str | None:
    """The object_catalog category ("Units"/"Buildings"/"Heroes"/"Others"), or None for a const no dataset has."""
    return _category_by_const().get(unit_const)


@dataclass(frozen=True)
class FindCriteria:
    text: str = ""
    regex: bool = False
    case_sensitive: bool = False
    match_captions: bool = False
    players: frozenset[int] | None = None
    categories: frozenset[str] | None = None
    kinds: frozenset[str] | None = None
    consts: frozenset[int] | None = None
    # Half-open tiles (x0, y0, x1, y1); a unit matches when its clamped footprint intersects.
    area: tuple[int, int, int, int] | None = None
    include_garrisoned: bool = True
    include_off_map: bool = True


@dataclass(frozen=True)
class FindRow:
    key: tuple[int, int]
    reference_id: int
    unit_const: int
    name: str
    player: int
    x: float
    y: float
    z: float
    caption: str
    garrisoned_in: int
    occupants: int
    off_map: bool
    duplicate_ref: bool
    # Trigger conditions/effects referencing this reference_id; None until triggers are parsed.
    trigger_refs: int | None = None


def is_garrisoned(unit: Any) -> bool:
    """UnitFilter.matches' rule: -1 and a self-reference both read as not garrisoned."""
    host = getattr(unit, "garrisoned_in_id", -1)
    return host != -1 and host != unit.reference_id


def _caption(unit: Any) -> str:
    try:
        value = getattr(unit, "caption_string", "")
    except Exception:  # noqa: BLE001 -- the library's version-gated properties raise
        return ""
    return value if isinstance(value, str) else ""


def _id_query(text: str) -> int | None:
    """`#1103` or `1103` as an exact unit_const, else None."""
    stripped = text.strip()
    if stripped.startswith("#"):
        stripped = stripped[1:]
    return int(stripped) if stripped.isdigit() else None


def _intersects(bounds: tuple[int, int, int, int], area: tuple[int, int, int, int]) -> bool:
    bx0, bx1, by0, by1 = bounds
    ax0, ay0, ax1, ay1 = area
    return bx0 < ax1 and bx1 > ax0 and by0 < ay1 and by1 > ay0


def find(
    loaded: Any,
    criteria: FindCriteria,
    map_w: int,
    map_h: int,
    trigger_ref_counts: Mapping[int, int] | None = None,
) -> list[FindRow]:
    """Every placed object matching all of `criteria`'s facets, in file order.

    `trigger_ref_counts` (find_triggers.trigger_ref_counts()) fills
    FindRow.trigger_refs; None leaves the column blank and never parses triggers."""
    pattern = compile_pattern(criteria.text, criteria.regex, criteria.case_sensitive)
    id_query = _id_query(criteria.text) if criteria.text else None
    kind_consts: frozenset[int] | None = None
    if criteria.kinds is not None:
        kind_consts = frozenset().union(*(_kind_consts(k) for k in criteria.kinds)) if criteria.kinds else frozenset()
    units = list(all_units(loaded))
    ref_counts = Counter(int(u.reference_id) for _p, u in units)
    occupants = Counter(int(u.garrisoned_in_id) for _p, u in units if is_garrisoned(u))
    rows: list[FindRow] = []
    names: dict[int, str] = {}
    for player, unit in units:
        const = int(unit.unit_const)
        if criteria.players is not None and player not in criteria.players:
            continue
        if criteria.consts is not None and const not in criteria.consts:
            continue
        if criteria.categories is not None and category_of(const) not in criteria.categories:
            continue
        if kind_consts is not None and const not in kind_consts:
            continue
        garrisoned = is_garrisoned(unit)
        if garrisoned and not criteria.include_garrisoned:
            continue
        bounds = unit_tile_bounds(unit, map_w, map_h)
        if bounds is None and not criteria.include_off_map:
            continue
        if criteria.area is not None and (bounds is None or not _intersects(bounds, criteria.area)):
            continue
        name = names.get(const)
        if name is None:
            name = names[const] = object_catalog.display_name(const)
        caption = _caption(unit)
        if pattern is not None:
            hit = (
                pattern.search(name) is not None
                or (criteria.match_captions and pattern.search(caption) is not None)
                or id_query == const
            )
            if not hit:
                continue
        ref = int(unit.reference_id)
        rows.append(
            FindRow(
                key=(player, ref),
                reference_id=ref,
                unit_const=const,
                name=name,
                player=player,
                x=float(unit.x),
                y=float(unit.y),
                z=float(getattr(unit, "z", 0.0)),
                caption=caption,
                garrisoned_in=int(unit.garrisoned_in_id) if garrisoned else -1,
                occupants=occupants.get(ref, 0),
                off_map=bounds is None,
                duplicate_ref=ref_counts[ref] > 1,
                trigger_refs=None if trigger_ref_counts is None else trigger_ref_counts.get(ref, 0),
            )
        )
    return rows


def resolve_keys(loaded: Any, keys: Iterable[tuple[int, int]]) -> list[tuple[int, Any]]:
    """(player_id, unit) for each key that names exactly one live unit, in `keys` order.

    Over all_units(), never the pick index. A key matching two units (a
    duplicated reference_id inside one player's list) is ambiguous and dropped."""
    wanted = list(keys)
    want_set = set(wanted)
    found: dict[tuple[int, int], list] = {}
    for player, unit in all_units(loaded):
        key = (player, int(unit.reference_id))
        if key in want_set:
            found.setdefault(key, []).append(unit)
    return [(key[0], found[key][0]) for key in wanted if len(found.get(key, ())) == 1]


CSV_COLUMNS = (
    "name",
    "reference_id",
    "unit_const",
    "player",
    "x",
    "y",
    "z",
    "caption",
    "garrisoned_in",
    "occupants",
    "off_map",
    "duplicate_ref",
    "trigger_refs",
)


def rows_to_csv(rows: Iterable[FindRow], path: Path | str) -> int:
    """Writes `rows` as CSV with a header; returns the row count. Refuses
    (scenario_write.WriteBlockedError) a compatdata path before opening anything."""
    path = refuse_forbidden_path(path)
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for row in rows:
            writer.writerow(
                [
                    row.name,
                    row.reference_id,
                    row.unit_const,
                    row.player,
                    f"{row.x:g}",
                    f"{row.y:g}",
                    f"{row.z:g}",
                    row.caption,
                    row.garrisoned_in,
                    row.occupants,
                    int(row.off_map),
                    int(row.duplicate_ref),
                    "" if row.trigger_refs is None else row.trigger_refs,
                ]
            )
    return len(rows)
