"""Family/piece catalog for the Terrain-mode Cliff tool -- Track B Stage 1 of
the 2026-09-02 cliffs plan.

Cliffs are 96 GAIA `class_ == 34` consts: ten full nine-piece families
sharing one graphic per family, plus a six-piece partial family
(CLF01..CLF08, missing 05/07/09) that shares the Desert graphic. Family
membership here is `unit_sprites.cliff_consts()` -- the same set
tests/test_unit_footprints.py pins equal to
`terrain_palette.OBJECT_TILE_SPANS`' key set -- never object_catalog's name
strings, so a missing or renamed catalog entry can't silently drop a piece
from the picker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from descape import object_catalog, unit_sprites

# Falls back to the .dat's angle_count when no install is configured (no
# .sld to read a real frame count from) -- unit_sprites.variant_index()'s
# own fallback rule, A3.
_DEFAULT_ANGLE_COUNT = 24

_SUFFIX_RE = re.compile(r"(\d+)$")


@dataclass(frozen=True)
class CliffPiece:
    unit_const: int
    label: str  # object_catalog's own .dat "code", falling back to the bare const


@lru_cache(maxsize=1)
def families() -> dict[str, tuple[CliffPiece, ...]]:
    """Family label (the shared graphic's file_name) -> its pieces, sorted by
    unit_const. Not process-invalidated by an install change: file_name
    grouping and the .dat "code" labels are both install-independent, unlike
    the frame counts frame_count() below reads live.
    """
    graphics = unit_sprites.graphic_map()
    grouped: dict[str, list[CliffPiece]] = {}
    for const in sorted(unit_sprites.cliff_consts()):
        entry = graphics.get(const)
        family = str(entry["file_name"]) if entry else f"const_{const}"
        label = object_catalog.object_name(const) or str(const)
        grouped.setdefault(family, []).append(CliffPiece(unit_const=const, label=label))
    return {family: tuple(pieces) for family, pieces in grouped.items()}


def family_label(file_name: str) -> str:
    """A human family name derived from the shared graphic's own file_name
    (e.g. "n_cliff_marble_x1" -> "Cliff Marble") rather than from any one
    piece's object_catalog label -- the CLF01..CLF08 partial family shares
    its graphic (and so its families() grouping) with the real Desert
    family, and picking either piece's own label as the family name would
    hide the other."""
    stem = file_name.removeprefix("n_").removesuffix("_x1")
    return stem.replace("_", " ").title()


def piece_suffix(unit_const: int) -> int | None:
    """The trailing digit of the piece's own object_catalog label ("Cliff
    (Default) 04" -> 4, "Marble Cliff 7" -> 7), or None if it has no label
    to read one off.

    This is the cross-family pooling key `cliff_connectivity`'s measured
    table is built on -- the same suffix means the same footprint span in
    every family (Short Marble Cliff 1 excepted, which is why the span
    itself still comes from `terrain_palette.tile_span()` and never from
    this number). `tools/gen_cliff_connectivity.py` calls this rather than
    keeping its own copy of the regex, so the table and its consumers
    can't disagree about what a suffix is.
    """
    match = _SUFFIX_RE.search(object_catalog.object_name(unit_const) or "")
    return int(match.group(1)) if match else None


def piece_for_suffix(unit_const: int, suffix: int) -> int | None:
    """The const carrying `suffix` within `unit_const`'s own family, or None
    if that family has no such piece.

    Nearest-const tiebreak because one families() entry can hold two
    sub-groups with colliding suffixes: CLF01..CLF08 shares the Desert
    graphic, so "Cliff Sand" has fifteen pieces and two of most suffixes.
    Staying near the const the caller already has keeps a chain inside the
    sub-group the user picked from, falling through to the other only for
    the three suffixes CLF has no piece for (05/07/09).
    """
    for pieces in families().values():
        if not any(p.unit_const == unit_const for p in pieces):
            continue
        matches = [p.unit_const for p in pieces if piece_suffix(p.unit_const) == suffix]
        if not matches:
            return None
        return min(matches, key=lambda const: (abs(const - unit_const), const))
    return None


def frame_count(unit_const: int) -> int:
    """How many rotation values are legal for `unit_const`: the real .sld
    frame count when an install is configured (Marble/Short Marble: 23,
    every other cliff family: 24 -- tests/test_cliffs.py's own ground
    truth), falling back to the .dat's angle_count otherwise."""
    entry = unit_sprites.graphic_map().get(unit_const)
    if entry is None:
        return _DEFAULT_ANGLE_COUNT
    real = unit_sprites.sld_frame_count(str(entry["file_name"]))
    return real if real is not None else max(1, int(entry["angle_count"]))
