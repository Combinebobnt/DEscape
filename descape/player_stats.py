"""Per-player statistics for View mode's player readout (GH #5).

Qt-free leaf module, the same shape as map_analysis.py and unit_stats_table.py,
so every count is testable without a QApplication. Never imports batch_api
(which pulls in scenario_write) or Qt.

**The four unit buckets partition each player's list.** BUILDING_TILE_SPANS,
TREE_UNIT_IDS and eye_candy_consts() are pairwise disjoint, and wall_consts()
is a strict subset of BUILDING_TILE_SPANS (measured over the examples/
corpus, and pinned by tests/test_player_stats.py). Each bucket is counted by
its own membership test, never derived as a remainder, so a const-table
regeneration that broke disjointness shows up as a partition mismatch.

"Buildings" is BUILDING_TILE_SPANS membership, the rule
unit_pick.footprint_entries() and render._unit_color already use, not
BuildingInfo (470 consts vs 235).

**Triggers.** A trigger counts once for each player any of its conditions or
effects names in a PlayerId field (source or target player). Only the fields
the entry's own type uses are read: entry types that don't use source_player
still store a value there, and reading it unfiltered inflates the counts.
Nothing here ever triggers a parse: an unattempted parse reads as None.
"""

from __future__ import annotations

from dataclasses import dataclass

from descape import library_compat, unit_kind
from descape.scenario_io import LoadedScenario, parse_triggers
from descape.terrain_palette import BUILDING_TILE_SPANS, TREE_UNIT_IDS

NUM_PLAYER_SLOTS = 9  # 0 = GAIA, 1-8
_PLAYER_PRESENTATIONS = frozenset({"PlayerId"})


@dataclass(frozen=True)
class PlayerCounts:
    placements: int
    units: int  # in none of the three buckets below
    buildings: int
    walls: int  # subset of buildings
    trees: int
    eye_candy: int


@dataclass(frozen=True)
class TriggerSummary:
    per_player: dict[int, int]  # player_id -> distinct triggers naming it
    without_player: int  # triggers naming no player at all
    total: int


def counts_for(loaded: LoadedScenario, player_id: int) -> PlayerCounts:
    """Walks the raw per-player list (index 0 = GAIA). Never
    unit_pick.build_index(), which drops filtered and off-map units."""
    walls_set = unit_kind.wall_consts()
    eye_candy_set = unit_kind.eye_candy_consts()
    units = loaded.unit_manager.units[player_id]
    buildings = walls = trees = eye_candy = other = 0
    for unit in units:
        const = unit.unit_const
        if const in BUILDING_TILE_SPANS:
            buildings += 1
            if const in walls_set:
                walls += 1
        elif const in TREE_UNIT_IDS:
            trees += 1
        elif const in eye_candy_set:
            eye_candy += 1
        else:
            other += 1
    return PlayerCounts(len(units), other, buildings, walls, trees, eye_candy)


def _read(obj, attribute: str):
    # map_analysis._read()'s body, copied to keep this a leaf module.
    try:
        return getattr(obj, attribute, None)
    except Exception:  # noqa: BLE001 -- the library's version-gated properties raise
        return None


def trigger_summary(loaded: LoadedScenario) -> TriggerSummary | None:
    """None unless the Triggers section has already been parsed successfully
    and a vocabulary exists for this version. Goes back through the memoized
    parse_triggers() every call rather than holding the manager, whose field
    gating is process-global (trigger_model.py)."""
    if loaded.trigger_read_supported is not True:
        return None
    manager = parse_triggers(loaded)
    if manager is None or not library_compat.vocabulary_is_available(loaded.scenario_version):
        return None
    vocabulary = library_compat.load_vocabulary(loaded.scenario_version)
    kinds = (
        ("conditions", "condition_type", vocabulary.conditions, vocabulary.condition_presentation),
        ("effects", "effect_type", vocabulary.effects, vocabulary.effect_presentation),
    )
    per_player = dict.fromkeys(range(NUM_PLAYER_SLOTS), 0)
    without_player = 0
    for trigger in manager.triggers:
        players: set[int] = set()
        for list_name, type_attribute, entries, presentation in kinds:
            for ce in getattr(trigger, list_name):
                definition = entries.get(_read(ce, type_attribute))
                if definition is None:
                    continue
                for attribute in definition.attributes:
                    if presentation.get(attribute) not in _PLAYER_PRESENTATIONS:
                        continue
                    value = _read(ce, attribute)
                    values = value if isinstance(value, (list, tuple)) else [value]
                    players.update(v for v in values if isinstance(v, int) and 0 <= v < NUM_PLAYER_SLOTS)
        for player_id in players:
            per_player[player_id] += 1
        if not players:
            without_player += 1
    return TriggerSummary(per_player, without_player, len(manager.triggers))


def trigger_counts(loaded: LoadedScenario) -> dict[int, int] | None:
    summary = trigger_summary(loaded)
    return None if summary is None else summary.per_player


def triggers_without_player(loaded: LoadedScenario) -> int | None:
    summary = trigger_summary(loaded)
    return None if summary is None else summary.without_player
