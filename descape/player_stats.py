"""Per-player statistics for View mode's player readout (GH #5, GH #145).

Qt-free leaf module, the same shape as map_analysis.py and unit_stats_table.py,
so every count is testable without a QApplication. Never imports batch_api
(which pulls in scenario_write) or Qt.

**Classification is owner-independent.** A const lands in the same bucket
whoever owns it; only which rows the View panel *shows* depends on the
owner. Every placement lands in exactly one bucket, in this order:

1. Trebuchets (packed and unpacked, `TREBUCHET_CONSTS`) are Military units.
   This is the one exception to "BUILDING_TILE_SPANS membership is a
   building" (user decision 2026-09-26).
2. BUILDING_TILE_SPANS members are buildings, split into a partition:
   Walls & gates (`unit_kind.wall_consts()`), Economy and Military families,
   and the Other remainder. The .dat class cannot split buildings (houses,
   stables, castles, docks, markets and TCs are all class 3), so the
   families are curated: a family is a BuildingInfo enum name resolved to
   every const sharing that const's .dat string id, which is how the age
   variants (Barracks Age1 12, Age2 498, Age3 132, Age4 20) all land in it.
   Economy also takes class 49 (farms, fish traps, pastures) and the Pasture
   Post; Military also takes every class-52 tower. Blacksmith and University
   are Other because they train nothing; Dock is Economy although it also
   trains warships.
3. TREE_UNIT_IDS, then eye_candy_consts().
4. GAIA-side buckets: resource piles by class (gold mines, stone mines,
   berry bushes, fish, other piles), animals (prey, predators, livestock,
   controlled animals, and class-11 birds of type 70 only: class 11 also
   holds flares and dead bodies, which are not animals), cliffs and relics.
5. Everything else is a unit: Economy (villagers, trade carts and cogs,
   fishing ships), Military (combat classes plus trebuchets) and the Other
   remainder (flags, revealers, torches, invisible objects, kings and carts,
   transport ships). Both Other remainders are pinned by a corpus test, so a
   newly placed class or family fails there instead of landing in Other
   silently.

Water, Unique (`UnitInfo.unique_units(include_chronicles=True)`) and Heroes
(`HeroInfo`) are overlapping flags over the unit buckets, never part of the
partition sum.

**Resources on map** sum object_catalog.json's `resources` (the .dat's own
storage amounts, mapped food/wood/gold/stone in tools/gen_object_catalog.py,
Fish Storage counted as food) over every owner's placements. A placed unit
stores no amount, so these are the game's defaults; trigger effects that
change amounts are not reflected.

**Triggers.** A trigger counts once for each player any of its conditions or
effects names in a PlayerId field (source or target player). Only the fields
the entry's own type uses are read: entry types that don't use source_player
still store a value there, and reading it unfiltered inflates the counts.
Nothing here ever triggers a parse: an unattempted parse reads as None.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache

from AoE2ScenarioParser.datasets.buildings import BuildingInfo
from AoE2ScenarioParser.datasets.heroes import HeroInfo
from AoE2ScenarioParser.datasets.units import UnitInfo

from descape import library_compat, object_catalog, unit_kind
from descape.scenario_io import LoadedScenario, parse_triggers
from descape.terrain_palette import BUILDING_TILE_SPANS, TREE_UNIT_IDS

NUM_PLAYER_SLOTS = 9  # 0 = GAIA, 1-8
_PLAYER_PRESENTATIONS = frozenset({"PlayerId"})

# Same four consts unit_rotation treats as facing-angle trebuchets.
TREBUCHET_CONSTS = frozenset({42, 331, 1690, 1691})

# .dat classes (AoE2ScenarioParser's ObjectClass names in the comments).
_GOLD_MINE_CLASSES = frozenset({32})
_STONE_MINE_CLASSES = frozenset({8})
_BERRY_BUSH_CLASSES = frozenset({7})
_FISH_CLASSES = frozenset({5, 31, 33, 63})  # ocean, deep-sea, shore fish, oysters
_OTHER_PILE_CLASSES = frozenset({40, 41, 48})  # salvage, resource pile, ore mine
_ANIMAL_CLASSES = frozenset({9, 10, 58, 61})  # prey, predator, livestock, controlled
_BIRD_CLASS = 11  # MISCELLANEOUS: an animal only at type 70 (HAWKX); flares are type 10
_CREATABLE_TYPE = 70
_CLIFF_CLASSES = frozenset({34})
_RELIC_CLASSES = frozenset({42})
_FARM_CLASS = 49
_TOWER_CLASS = 52

ECONOMY_UNIT_CLASSES = frozenset({2, 4, 19, 21})  # trade boat, civilian, trade cart, fishing boat
# Archer, infantry, cavalry, siege, monk, warship, conquistador, war elephant,
# elephant archer, phalanx, petard, cavalry archer, monk with relic, hand
# cannoneer, two-handed swordsman, pikeman, scout, spearman, packed and
# unpacked siege, boarding boat, ballista, raider, cavalry raider.
MILITARY_UNIT_CLASSES = frozenset(
    {0, 6, 12, 13, 18, 22, 23, 24, 26, 28, 35, 36, 43, 44, 45, 46, 47, 50, 51, 53, 54, 55, 56, 57}
)
WATER_UNIT_CLASSES = frozenset({2, 20, 21, 22, 53})

_ECONOMY_FAMILIES = (
    "TOWN_CENTER", "HOUSE", "MILL", "LUMBER_CAMP", "MINING_CAMP", "MARKET", "DOCK",
    "FOLWARK", "MULE_CART", "FEITORIA", "SETTLEMENT", "TRADE_WORKSHOP", "HARBOR", "CARAVANSERAI",
)
# The Pasture Post (class 14, no enum member, no string id), which a pasture's owner places.
_EXTRA_ECONOMY_CONSTS = frozenset({1888})
_MILITARY_FAMILIES = (
    "BARRACKS", "ARCHERY_RANGE", "STABLE", "SIEGE_WORKSHOP", "CASTLE", "KREPOST", "DONJON",
    "FORTRESS", "CAMP_BARRACKS", "CAMP_ARCHERY_RANGE", "CAMP_STABLE", "WOODEN_FORT",
    "FORTIFIED_CHURCH", "OUTPOST", "FORTIFIED_OUTPOST",
)


@dataclass(frozen=True)
class PlayerCounts:
    placements: int
    units: int  # economy_units + military_units + other_units
    economy_units: int
    military_units: int  # includes trebuchets
    other_units: int
    water_units: int  # overlapping flags over units, not in any sum
    unique_units: int
    hero_units: int
    buildings: int  # economy_buildings + military_buildings + walls + other_buildings
    economy_buildings: int
    military_buildings: int
    walls: int  # walls and gates
    other_buildings: int
    trees: int
    eye_candy: int
    piles: int  # gold_mines + stone_mines + berry_bushes + fish + other_piles
    gold_mines: int
    stone_mines: int
    berry_bushes: int
    fish: int
    other_piles: int
    animals: int
    cliffs: int
    relics: int

    def partition_total(self) -> int:
        """The sum of every non-overlapping bucket, == placements."""
        return (
            self.units + self.buildings + self.trees + self.eye_candy
            + self.piles + self.animals + self.cliffs + self.relics
        )


@dataclass(frozen=True)
class ResourceTotals:
    food: int
    wood: int
    gold: int
    stone: int


@dataclass(frozen=True)
class TriggerSummary:
    per_player: dict[int, int]  # player_id -> distinct triggers naming it
    without_player: int  # triggers naming no player at all
    total: int


@dataclass(frozen=True)
class _Sets:
    gold_mines: frozenset[int]
    stone_mines: frozenset[int]
    berry_bushes: frozenset[int]
    fish: frozenset[int]
    other_piles: frozenset[int]
    animals: frozenset[int]
    cliffs: frozenset[int]
    relics: frozenset[int]
    economy_units: frozenset[int]
    military_units: frozenset[int]
    water_units: frozenset[int]
    unique_units: frozenset[int]
    hero_units: frozenset[int]
    economy_buildings: frozenset[int]
    military_buildings: frozenset[int]


def _family(objects: dict[int, dict], enum_name: str) -> frozenset[int]:
    """BuildingInfo[enum_name]'s const plus every const sharing its .dat
    string id (the age variants). A string id of 0 shares nothing."""
    const = BuildingInfo[enum_name].ID
    string_id = objects.get(const, {}).get("string_id", 0)
    if not string_id:
        return frozenset({const})
    return frozenset({const} | {c for c, e in objects.items() if e.get("string_id") == string_id})


def _classes(objects: dict[int, dict], classes: frozenset[int]) -> frozenset[int]:
    return frozenset(c for c, e in objects.items() if e.get("class") in classes)


@lru_cache(maxsize=1)
def _sets() -> _Sets:
    objects = object_catalog.dat_objects()
    birds = frozenset(
        c for c, e in objects.items() if e.get("class") == _BIRD_CLASS and e.get("type") == _CREATABLE_TYPE
    )
    economy_buildings = (
        _classes(objects, frozenset({_FARM_CLASS}))
        | _EXTRA_ECONOMY_CONSTS
        | frozenset().union(*(_family(objects, name) for name in _ECONOMY_FAMILIES))
    )
    military_buildings = _classes(objects, frozenset({_TOWER_CLASS})) | frozenset().union(
        *(_family(objects, name) for name in _MILITARY_FAMILIES)
    )
    return _Sets(
        gold_mines=_classes(objects, _GOLD_MINE_CLASSES),
        stone_mines=_classes(objects, _STONE_MINE_CLASSES),
        berry_bushes=_classes(objects, _BERRY_BUSH_CLASSES),
        fish=_classes(objects, _FISH_CLASSES),
        other_piles=_classes(objects, _OTHER_PILE_CLASSES),
        animals=_classes(objects, _ANIMAL_CLASSES) | birds,
        cliffs=_classes(objects, _CLIFF_CLASSES),
        relics=_classes(objects, _RELIC_CLASSES),
        economy_units=_classes(objects, ECONOMY_UNIT_CLASSES),
        military_units=_classes(objects, MILITARY_UNIT_CLASSES) | TREBUCHET_CONSTS,
        water_units=_classes(objects, WATER_UNIT_CLASSES),
        unique_units=frozenset(m.ID for m in UnitInfo.unique_units(include_chronicles=True)),
        hero_units=frozenset(m.ID for m in HeroInfo),
        economy_buildings=economy_buildings,
        military_buildings=military_buildings,
    )


def bucket_sets() -> dict[str, frozenset[int]]:
    """Every category const set by bucket name, for tests and tools. The
    building sets are family sets, applied only to BUILDING_TILE_SPANS
    members."""
    sets = _sets()
    return {f.name: getattr(sets, f.name) for f in fields(sets)}


def counts_for(loaded: LoadedScenario, player_id: int) -> PlayerCounts:
    """Walks the raw per-player list (index 0 = GAIA). Never
    unit_pick.build_index(), which drops filtered and off-map units."""
    s = _sets()
    walls_set = unit_kind.wall_consts()
    eye_candy_set = unit_kind.eye_candy_consts()
    units = loaded.unit_manager.units[player_id]
    n = {f.name: 0 for f in fields(PlayerCounts)}
    for unit in units:
        const = unit.unit_const
        if const not in TREBUCHET_CONSTS and const in BUILDING_TILE_SPANS:
            if const in walls_set:
                n["walls"] += 1
            elif const in s.economy_buildings:
                n["economy_buildings"] += 1
            elif const in s.military_buildings:
                n["military_buildings"] += 1
            else:
                n["other_buildings"] += 1
        elif const in TREE_UNIT_IDS:
            n["trees"] += 1
        elif const in eye_candy_set:
            n["eye_candy"] += 1
        elif const in s.gold_mines:
            n["gold_mines"] += 1
        elif const in s.stone_mines:
            n["stone_mines"] += 1
        elif const in s.berry_bushes:
            n["berry_bushes"] += 1
        elif const in s.fish:
            n["fish"] += 1
        elif const in s.other_piles:
            n["other_piles"] += 1
        elif const in s.animals:
            n["animals"] += 1
        elif const in s.cliffs:
            n["cliffs"] += 1
        elif const in s.relics:
            n["relics"] += 1
        else:
            if const in s.economy_units:
                n["economy_units"] += 1
            elif const in s.military_units:
                n["military_units"] += 1
            else:
                n["other_units"] += 1
            n["water_units"] += const in s.water_units
            n["unique_units"] += const in s.unique_units
            n["hero_units"] += const in s.hero_units
    n["placements"] = len(units)
    n["units"] = n["economy_units"] + n["military_units"] + n["other_units"]
    n["buildings"] = n["economy_buildings"] + n["military_buildings"] + n["walls"] + n["other_buildings"]
    n["piles"] = n["gold_mines"] + n["stone_mines"] + n["berry_bushes"] + n["fish"] + n["other_piles"]
    return PlayerCounts(**n)


@lru_cache(maxsize=1)
def _resource_amounts() -> dict[int, tuple[int, int, int, int]]:
    return {
        const: tuple(entry["resources"].get(key, 0) for key in ("food", "wood", "gold", "stone"))
        for const, entry in object_catalog.dat_objects().items()
        if entry.get("resources")
    }


def resources_on_map(loaded: LoadedScenario) -> ResourceTotals:
    """The game's default food/wood/gold/stone over every owner's placements
    (trees, animals and player livestock included)."""
    amounts = _resource_amounts()
    food = wood = gold = stone = 0
    for units in loaded.unit_manager.units:
        for unit in units:
            amount = amounts.get(unit.unit_const)
            if amount is not None:
                food += amount[0]
                wood += amount[1]
                gold += amount[2]
                stone += amount[3]
    return ResourceTotals(food, wood, gold, stone)


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
