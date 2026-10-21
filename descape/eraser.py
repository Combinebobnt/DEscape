"""The Eraser brush's targets (GH #184): which placed units a drag removes.

Qt-free, so the default tier covers it headless. Every target is built from an
existing unit_kind set (or terrain_palette.TREE_UNIT_IDS, which the Filters
menu's Show Trees reads too), so the Eraser and the Filters menu cannot
disagree about what a category is.

A unit is erased if it matches ANY ticked target, with one veto: a cliff const
never matches unless Cliffs is ticked, whatever else is, since erasing one
piece breaks a cliff chain. Terrain Overlay is not a unit target at all; the
viewer clears per-tile `layer` for it.

The hit test reuses unit_pick's index (UnitIndex.by_tile keys every footprint
tile and already leaves out filter-hidden and off-map units), so a hidden unit
is never erased, the same rule Convert and click-pick follow.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, fields, replace

from descape import unit_kind
from descape.terrain_palette import TREE_UNIT_IDS
from descape.unit_filter import GAIA_PLAYER_ID
from descape.unit_pick import UnitEntry, UnitIndex


@dataclass(frozen=True)
class EraseTargets:
    """Which categories a drag erases. The defaults are the in-game Erase help
    text's own pair (string 30536, "erase units and buildings"), so a first
    stroke never takes trees, resources or terrain by surprise."""

    units: bool = True
    buildings: bool = True
    trees: bool = False
    invisible: bool = False
    gaia: bool = False
    overlay: bool = False
    cliffs: bool = False

    @property
    def any_units(self) -> bool:
        """Whether any target can match a unit (every one but Overlay)."""
        return self.units or self.buildings or self.trees or self.invisible or self.gaia or self.cliffs


# (field, menu label) in menu order. Cliffs is last and set apart in the menu:
# it is the one opt-in target "Everything" leaves alone.
TARGET_LABELS: tuple[tuple[str, str], ...] = (
    ("units", "Units"),
    ("buildings", "Buildings"),
    ("trees", "Trees"),
    ("overlay", "Terrain Overlay"),
    ("invisible", "Invisible Objects"),
    ("gaia", "Gaia Objects"),
    ("cliffs", "Cliffs"),
)

# GH #184's "Everything": its six targets, not Cliffs.
EVERYTHING_FIELDS: frozenset[str] = frozenset(f.name for f in fields(EraseTargets)) - {"cliffs"}


def everything(targets: EraseTargets) -> EraseTargets:
    """`targets` with every Everything target ticked; Cliffs is left as it was."""
    return replace(targets, **dict.fromkeys(EVERYTHING_FIELDS, True))


def unit_matches(targets: EraseTargets, player_id: int, unit_const: int) -> bool:
    """Whether a unit of `unit_const` owned by `player_id` is erased under
    `targets`: the union of the ticked targets, after the cliff veto."""
    if unit_const in unit_kind.cliff_consts():
        return targets.cliffs
    if targets.units and unit_const in unit_kind.mobile_unit_consts():
        return True
    if targets.buildings and (unit_const in unit_kind.building_consts() or unit_const in unit_kind.wall_consts()):
        return True
    is_tree = unit_const in TREE_UNIT_IDS
    if targets.trees and is_tree:
        return True
    if targets.invisible and unit_const in unit_kind.invisible_consts():
        return True
    # Trees owns GAIA's trees, matching the game's separate checkboxes.
    return targets.gaia and player_id == GAIA_PLAYER_ID and not is_tree


@dataclass
class EraseStroke:
    """One Eraser drag's state, held by the viewer from press to release.

    `index` and `model` are None when no unit target is ticked or the file's
    units are read-only; `overlay` is whether this drag clears `layer`."""

    targets: EraseTargets
    overlay: bool = False
    index: UnitIndex | None = None
    model: object | None = None  # the UnitEditModel whose edit is open for the drag
    touched_tiles: set[tuple[int, int]] = field(default_factory=set)
    done: set[int] = field(default_factory=set)  # unit ids already erased or refused
    erased: int = 0
    cascaded: int = 0  # of `erased`, garrison occupants taken with their host
    refused: int = 0
    reference_ids: list[int] = field(default_factory=list)
    layer_cleared: int = 0
    pending_layer_tiles: list[int] = field(default_factory=list)  # cleared tile indices, not yet repainted
    pending_splices: list = field(default_factory=list)  # render_cache.UnitSplice, not yet repainted
    # Resident mips not is_level_resident() at one of those erases: their layers can't size its repaint.
    pending_stale_levels: set[int] = field(default_factory=set)
    # The Cliff tool's right-button erase (GH #247): the cursor tiles alone, no brush.
    cursor_only: bool = False
    # Re-caps the cliff ends the erase leaves at release (cliff_chain.erase_rewrites()).
    cliff_ends: bool = False
    cliffs: tuple | None = None  # (the press's ExistingCliff list, its tile index)
    scenario_ref: object | None = None  # weakref to the scenario the press saw
    # unit_gen after this stroke's own last removal; None once something else changed the units.
    expected_gen: int | None = None
    erased_ids: set[int] = field(default_factory=set)  # id() of each unit actually removed
    reshaped: int = 0
    left_alone: int = 0
    ends_note: str | None = None


def collect_hits(
    targets: EraseTargets, index: UnitIndex, tiles: Iterable[tuple[int, int]], done: set[int]
) -> list[UnitEntry]:
    """Every entry in `index` whose footprint covers one of `tiles` and which
    `targets` erases, in tile then bucket order, skipping and then adding to `done`
    (unit ids). `done` is the stroke's own dedupe: a multi-tile unit sits in
    several tiles' buckets, and a drag crosses the same tile many times."""
    hits = []
    for tile in tiles:
        for order in index.by_tile.get(tile, ()):
            entry = index.entries[order]
            unit_id = id(entry.unit)
            if unit_id in done:
                continue
            if unit_matches(targets, entry.player_id, int(entry.unit.unit_const)):
                done.add(unit_id)
                hits.append(entry)
    return hits
