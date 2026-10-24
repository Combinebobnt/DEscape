"""Map Analysis: a read-only pre-ship check pass over an open scenario.

Qt-free, so every check is testable without a QApplication; the dialog lives
in descape/analysis_dialog.py. The (label, check_fn) registry mirrors the
tools/verify_*.py harness shape, but lives here so a future CLI can share it.

**Three-valued, not pass/fail.** Each check returns a CheckResult that is
clean (no findings), has findings, or is unavailable (could not run, e.g.
the 1.54/trigger-3.9 set whose Triggers section does not parse). An
unavailable check is never reported as clean. Severity is a separate axis
on each Finding.

**Every tile walk flat-indexes mm.terrain.** MapManager.get_tile() raises on
every coordinate of a non-square map and get_tile_safe() returns None for
all of them, so a pass built on either would report a clean bill of health
on a map it never read.

**Severities were measured against all 20 examples/ files** (shipping
campaign scenarios, so anything firing there is mislabelled): elevation,
off-map, display-order and trigger-reference checks fire on 0/20; one
dangling unit reference fires (a disabled trigger); active-player-with-no-
units fires only on blank_map. Victory text is empty and unset on 20/20 and
instructions on 17/20, so the objectives check is instructions-only and
INFO. Stranded-unit reachability is approximate: is_water() is a name
match, only terrain is considered (not trees, buildings or cliffs, which
are GAIA objects), and elevation is ignored.

The off-grid checks were measured over the 21 loadable examples/ files: a
building off the half-tile grid fires on 0/21 (WARNING), a building on the
wrong parity for its span on 11/21 (INFO, the game writes those), and units
off their tile centre on 6/21 (player) and 7/21 (GAIA), both INFO.
Decorations, trees, fish and cliffs are left out on purpose.

A GAIA-only object (unit_kind.gaia_only_consts(), the set Place Unit gives
to GAIA and no owner change gives to a player: resources, trees and the
Gunpowder Keg, plus what each game's editor lists only for GAIA, GH #32:
wild animals and birds, Dummy Target, and the Others outside its player
list) in a player's list fires on 0/21 examples/ files (measured before
GH #32 widened the set; 3 other AoE2 files have none of the wider set
player-owned) and on the one known bad file, the GH #206 tester's scenario
(a player 1 Forage Bush that was missing in the official editor), so it is
a WARNING. Its message states only that measured fact: whether the game
drops such an object on its own is unconfirmed.

Objects in invalid positions (GH #173) reuse placement_rules' two rules,
which are its approximations of the game's, not the game's own. Over the 21
loadable examples/ files: a mobile unit on terrain its .dat row refuses fires
on 0/21 (WARNING); a building on refused terrain on 6/21 with 53 objects (a
House on Beach, a Stone Wall in shallow water; INFO); a mobile unit inside
another object's box on 7/21 with 20 (INFO). Invisible objects are not
blockers there: counting Map Revealers would add 4. Garrisoned units are
skipped in all three (they stand wherever their host does: 12 false hits,
all inside Transport Ships), and so are off-map ones. A unit on a bridge is
not on water: the units row reads each building's foundation terrain
(placement_rules.laid_terrain()), and a bridge lays 28, which the land rows
pass. GAIA scenery, resources and fish are left out as decoration noise.

GH #230's "Train and research buttons" check reports a Change Train
Location, Add Train Location or Change Research Location effect whose button
won't show or that takes another one away, one finding per effect with the
worst reason first: the unit, tech or building is not available to the
player's civ (civ_availability, which the game's behaviour matched on
2026-10-16) or is in its Disables list (a building placed on the map counts
as available), a lower-id unit the civ has hides it,
a unit covers a tech in the slot or a tech lands under a unit (measured: a
unit put on Barracks button 6 left Long Swordsman unresearchable, and
Chemistry put on Barracks button 1 stayed under the Militia), a
villager id or the builder's More/Cancel slot, and a value with no button,
all WARNING. Value 0 (no button) is INFO, as it may be on purpose. A random
or unset civ skips the civ reasons. A generic tech the player's starting age
already researched (measured 2026-10-16) gets no button either, and is not
covered by a unit. Moving or disabling the other unit
anywhere in the file keeps the hidden and covered reasons quiet. A tech
the file enables or researches makes the units it gives available. Measured
(2026-10-16) on three private test scenarios: 5 warnings, all true (four
buildings trained at Villager 293, an elite unit the file never enables);
before the `gives` and placed-building rules it fired 21, 16 of them false.
examples/ was not available for this measurement.

Nothing here constructs an edit model, and nothing may be named verify_* or
*_supported (both already mean byte-layout trust elsewhere).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from AoE2ScenarioParser.datasets.terrains import TerrainId
from AoE2ScenarioParser.objects.managers.trigger_manager import get_trigger_referencing_ce

from descape import civ_availability, command_buttons, library_compat, object_catalog, placement_rules, train_locations
from descape.command_buttons import TECH, UNIT
from descape.fill_tools import connected_regions
from descape.iso_geometry import MAX_ELEVATION, MIN_ELEVATION
from descape.messages_fields import STRING_ID_UNSET
from descape.player_fields import defined_player_ids
from descape.render import unit_tile_bounds
from descape.scenario_io import LoadedScenario, parse_triggers
from descape.terrain_palette import BUILDING_TILE_SPANS
from descape.unit_kind import gaia_only_consts, invisible_consts, mobile_unit_consts
from descape.unit_references import all_units, build_reference_index, references_in

Severity = Literal["error", "warning", "info"]

# A player-owned unit on a land region smaller than this is reported.
STRANDED_REGION_TILES = 25
# Per-check cap, so a badly broken map can't build a 100k-row dialog.
MAX_FINDINGS_PER_CHECK = 200

_TRIGGERS_UNAVAILABLE = "triggers could not be parsed for this file"


@dataclass(frozen=True)
class Finding:
    message: str
    severity: Severity
    tile: tuple[int, int] | None = None
    unit_key: tuple[int, int] | None = None  # (player_id, reference_id)


@dataclass(frozen=True)
class CheckResult:
    label: str
    findings: tuple[Finding, ...] = ()
    unavailable: str = ""

    @property
    def is_clean(self) -> bool:
        return not self.unavailable and not self.findings


@dataclass(frozen=True)
class AnalysisReport:
    results: tuple[CheckResult, ...]

    @property
    def finding_count(self) -> int:
        return sum(len(r.findings) for r in self.results)

    @property
    def headline(self) -> str:
        clean = sum(1 for r in self.results if r.is_clean)
        unavailable = sum(1 for r in self.results if r.unavailable)
        findings = self.finding_count
        detail = f"{findings} finding{'' if findings == 1 else 's'}"
        if unavailable:
            detail += f", {unavailable} unavailable"
        return f"Clean on {clean} of {len(self.results)} checks: {detail}"


def _capped(label: str, findings: list[Finding]) -> CheckResult:
    if len(findings) > MAX_FINDINGS_PER_CHECK:
        extra = len(findings) - MAX_FINDINGS_PER_CHECK
        severity = findings[0].severity
        findings = [*findings[:MAX_FINDINGS_PER_CHECK], Finding(f"...and {extra} more not listed", severity)]
    return CheckResult(label, tuple(findings))


def _all_units(loaded: LoadedScenario):
    """(player_id, unit) over the raw nine lists, index 0 = GAIA. Never
    unit_pick.build_index(), which drops off-map units by design."""
    return all_units(loaded)


def _read(obj, attribute: str):
    try:
        return getattr(obj, attribute, None)
    except Exception:  # noqa: BLE001 -- the library's version-gated properties raise
        return None


# -- checks -----------------------------------------------------------------


def check_off_map_units(loaded: LoadedScenario) -> CheckResult:
    label = "Off-map units"
    mm = loaded.map_manager
    findings = [
        Finding(
            f"Player {player_id} unit {unit.unit_const} (id {unit.reference_id}) at "
            f"({unit.x:g}, {unit.y:g}) is outside the map",
            "warning",
            unit_key=(player_id, unit.reference_id),
        )
        for player_id, unit in _all_units(loaded)
        if unit_tile_bounds(unit, mm.map_width, mm.map_height) is None
    ]
    return _capped(label, findings)


def _on_half_grid(v: float) -> bool:
    return (v * 2).is_integer()


def _on_span_grid(v: float, span: int) -> bool:
    # render.span_anchor()'s rule: tile + span/2, so odd spans end in .5 and even ones in .0.
    return v - math.floor(v) == (span % 2) / 2


def _owner(player_id: int) -> str:
    return "GAIA" if player_id == 0 else f"Player {player_id}"


def _parity(span: int) -> str:
    return ".5" if span % 2 else ".0"


def _wrong_parity_hint(wrong: list[tuple[str, int]]) -> str:
    """'(x and y should end in .0)', or per axis when they expect different
    endings (a 4x1 gate: '(x should end in .0, y in .5)')."""
    endings = {_parity(span) for _axis, span in wrong}
    if len(endings) == 1:
        return f"({' and '.join(axis for axis, _span in wrong)} should end in {endings.pop()})"
    (axis0, span0), (axis1, span1) = wrong
    return f"({axis0} should end in {_parity(span0)}, {axis1} in {_parity(span1)})"


def check_off_grid_buildings(loaded: LoadedScenario) -> CheckResult:
    """WARNING off the half-tile grid (no shipping file has one; only a free
    placement write makes it), INFO for the wrong parity, which the game
    itself writes in 11 of 21 shipping files."""
    label = "Off-grid buildings"
    mm = loaded.map_manager
    mobile = mobile_unit_consts()
    findings = []
    for player_id, unit in _all_units(loaded):
        span = BUILDING_TILE_SPANS.get(unit.unit_const)
        if span is None or unit.unit_const in mobile:
            continue
        if unit_tile_bounds(unit, mm.map_width, mm.map_height) is None:
            continue
        x, y = unit.x, unit.y
        where = f"{_owner(player_id)} building {unit.unit_const} (id {unit.reference_id}) at ({x:g}, {y:g})"
        if not (_on_half_grid(x) and _on_half_grid(y)):
            message = f"{where} is off the half-tile grid; no game-made building sits there"
            severity: Severity = "warning"
        else:
            wrong = [(axis, s) for axis, v, s in (("x", x, span[0]), ("y", y, span[1])) if not _on_span_grid(v, s)]
            if not wrong:
                continue
            message = (
                f"{where} is half a tile off the grid for a {span[0]}x{span[1]} footprint "
                f"{_wrong_parity_hint(wrong)}"
            )
            severity = "info"
        findings.append(Finding(message, severity, tile=(int(x), int(y)), unit_key=(player_id, unit.reference_id)))
    findings.sort(key=lambda f: f.severity != "warning")  # stable: the cap never hides a warning behind info
    return _capped(label, findings)


def _off_centre_units(loaded: LoadedScenario, gaia: bool) -> list[Finding]:
    """Mobile units not at a tile centre (.5), where snapped Place Unit puts
    them. Garrisoned units are skipped (their stored spot is not where they
    stand), and off-map ones belong to the off-map check."""
    mm = loaded.map_manager
    mobile = mobile_unit_consts()
    findings = []
    for player_id, unit in _all_units(loaded):
        if (player_id == 0) != gaia or unit.unit_const not in mobile:
            continue
        if _read(unit, "garrisoned_in_id") not in (None, -1):
            continue
        if _on_span_grid(unit.x, 1) and _on_span_grid(unit.y, 1):
            continue
        if unit_tile_bounds(unit, mm.map_width, mm.map_height) is None:
            continue
        findings.append(
            Finding(
                f"{_owner(player_id)} unit {unit.unit_const} (id {unit.reference_id}) at "
                f"({unit.x:g}, {unit.y:g}) is not at its tile's centre",
                "info",
                tile=(int(unit.x), int(unit.y)),
                unit_key=(player_id, unit.reference_id),
            )
        )
    return findings


def check_off_grid_player_units(loaded: LoadedScenario) -> CheckResult:
    return _capped("Off-grid player units (not at tile centre)", _off_centre_units(loaded, gaia=False))


def check_off_grid_gaia_units(loaded: LoadedScenario) -> CheckResult:
    return _capped("Off-grid GAIA units (not at tile centre)", _off_centre_units(loaded, gaia=True))


def check_gaia_only_ownership(loaded: LoadedScenario) -> CheckResult:
    """A GAIA-only object in a player's list, wherever it is: off-map and
    garrisoned ones are reported too, since the owner is wrong either way."""
    label = "GAIA-only objects owned by a player"
    gaia_only = gaia_only_consts()
    findings = [
        Finding(
            f"Player {player_id} object {unit.unit_const} (id {unit.reference_id}) at "
            f"({unit.x:g}, {unit.y:g}) is GAIA-only; game-made files only give those to GAIA",
            "warning",
            tile=(int(unit.x), int(unit.y)),
            unit_key=(player_id, unit.reference_id),
        )
        for player_id, unit in _all_units(loaded)
        if player_id != 0 and unit.unit_const in gaia_only
    ]
    return _capped(label, findings)


def _standing_units(loaded: LoadedScenario, consts: frozenset[int] | set[int]):
    """(player_id, unit) for each on-map, non-garrisoned unit of `consts`:
    a garrisoned unit stands wherever its host does, an off-map one is the
    off-map check's."""
    mm = loaded.map_manager
    for player_id, unit in _all_units(loaded):
        if unit.unit_const not in consts:
            continue
        if _read(unit, "garrisoned_in_id") not in (None, -1):
            continue
        if unit_tile_bounds(unit, mm.map_width, mm.map_height) is None:
            continue
        yield player_id, unit


def _placed(kind: str, player_id: int, unit, message: str, severity: Severity) -> Finding:
    return Finding(
        f"{_owner(player_id)} {kind} {unit.unit_const} (id {unit.reference_id}) at ({unit.x:g}, {unit.y:g}){message}",
        severity,
        tile=(int(unit.x), int(unit.y)),
        unit_key=(player_id, unit.reference_id),
    )


def check_units_on_wrong_terrain(loaded: LoadedScenario) -> CheckResult:
    """A mobile unit on a tile its .dat row refuses. A building's foundation
    counts (placement_rules.laid_terrain()), so a unit on a bridge is not on
    water."""
    laid = placement_rules.laid_terrain(loaded)
    findings = []
    for player_id, unit in _standing_units(loaded, mobile_unit_consts()):
        reason = placement_rules.terrain_refusal(loaded, unit.unit_const, unit.x, unit.y, laid)
        if reason is not None:
            findings.append(_placed("unit", player_id, unit, f": {reason}", "warning"))
    return _capped("Units on terrain they can't stand on", findings)


def check_buildings_on_wrong_terrain(loaded: LoadedScenario) -> CheckResult:
    """A building on a tile its .dat row refuses, shore rule included."""
    buildings = BUILDING_TILE_SPANS.keys() - mobile_unit_consts()
    findings = []
    for player_id, unit in _standing_units(loaded, buildings):
        reason = placement_rules.terrain_refusal(loaded, unit.unit_const, unit.x, unit.y)
        if reason is not None:
            findings.append(_placed("building", player_id, unit, f": {reason}", "info"))
    return _capped("Buildings on terrain they can't stand on (often deliberate)", findings)


def check_overlapping_units(loaded: LoadedScenario) -> CheckResult:
    """A mobile unit inside another object's box. Invisible objects (Map
    Revealers and the like) are not blockers here, though Place Unit keeps
    treating them as solid."""
    invisible = invisible_consts()
    hidden = [unit for _player_id, unit in _all_units(loaded) if unit.unit_const in invisible]
    grid = placement_rules.BlockerGrid(loaded, exclude=hidden)
    findings = []
    for player_id, unit in _standing_units(loaded, mobile_unit_consts()):
        found = grid.blocker(unit.unit_const, unit.x, unit.y, exclude=[unit])
        if found is None:
            continue
        owner, blocker = found
        where = f"{object_catalog.display_name(blocker.unit_const)} ({_owner(owner)}) at ({blocker.x:g}, {blocker.y:g})"
        findings.append(_placed("unit", player_id, unit, f" overlaps {where}", "info"))
    return _capped("Units overlapping another object", findings)


def _elevation_grid(loaded: LoadedScenario) -> np.ndarray:
    mm = loaded.map_manager
    width, height = mm.map_width, mm.map_height
    flat = np.fromiter((tile.elevation for tile in mm.terrain), dtype=np.int64, count=width * height)
    return flat.reshape(height, width)


def check_elevation(loaded: LoadedScenario) -> CheckResult:
    """Any 8-adjacent pair more than 1 apart, the library's own repair rule.
    ERROR because such a seam crashes the game on load (scenario_write.py)."""
    label = "Illegal elevation steps"
    e = _elevation_grid(loaded)
    findings: list[Finding] = []
    for y, x in np.argwhere((e < MIN_ELEVATION) | (e > MAX_ELEVATION)):
        findings.append(
            Finding(
                f"Tile ({x}, {y}) elevation {e[y, x]} is outside {MIN_ELEVATION}..{MAX_ELEVATION}",
                "error",
                tile=(int(x), int(y)),
            )
        )
    # (slice of the "a" tile, slice of its neighbour, neighbour offset)
    pairs = (
        (e[:, :-1], e[:, 1:], (1, 0)),
        (e[:-1, :], e[1:, :], (0, 1)),
        (e[:-1, :-1], e[1:, 1:], (1, 1)),
        (e[:-1, 1:], e[1:, :-1], (-1, 1)),
    )
    for a, b, (dx, dy) in pairs:
        x_base = 1 if dx < 0 else 0
        for y, x in np.argwhere(np.abs(a - b) > 1):
            ax, ay = int(x) + x_base, int(y)
            bx, by = ax + dx, ay + dy
            findings.append(
                Finding(
                    f"Tiles ({ax}, {ay}) and ({bx}, {by}) differ by {abs(int(e[ay, ax]) - int(e[by, bx]))} "
                    "elevation levels; the game may crash loading this map",
                    "error",
                    tile=(ax, ay),
                )
            )
    return _capped(label, findings)


def check_trigger_display_order(loaded: LoadedScenario) -> CheckResult:
    label = "Trigger display order"
    manager = parse_triggers(loaded)
    if manager is None:
        return CheckResult(label, unavailable=_TRIGGERS_UNAVAILABLE)
    order = list(manager.trigger_display_order)
    if sorted(order) == list(range(len(manager.triggers))):
        return CheckResult(label)
    return CheckResult(
        label,
        (
            Finding(
                f"Display order is not a permutation of 0..{len(manager.triggers) - 1} "
                f"({len(order)} entries for {len(manager.triggers)} triggers)",
                "error",
            ),
        ),
    )


def _trigger_label(trigger) -> str:
    disabled = "" if _read(trigger, "enabled") else ", disabled"
    return f'Trigger {trigger.trigger_id} "{trigger.name}"{disabled}'


def check_trigger_references(loaded: LoadedScenario) -> CheckResult:
    label = "Dangling trigger references"
    manager = parse_triggers(loaded)
    if manager is None:
        return CheckResult(label, unavailable=_TRIGGERS_UNAVAILABLE)
    ids = {t.trigger_id for t in manager.triggers}
    findings = []
    for trigger in manager.triggers:
        for ce in get_trigger_referencing_ce(trigger):
            target = ce.trigger_id
            if target == -1 or target in ids:  # -1 is unset, not dangling
                continue
            findings.append(
                Finding(f"{_trigger_label(trigger)} refers to trigger {target}, which does not exist", "warning")
            )
    return _capped(label, findings)


def check_unit_references(loaded: LoadedScenario) -> CheckResult:
    """Placed-unit reference_ids in condition/effect fields, not type
    constants. Only the fields each condition/effect type actually uses."""
    label = "Dangling unit references"
    manager = parse_triggers(loaded)
    if manager is None:
        return CheckResult(label, unavailable=_TRIGGERS_UNAVAILABLE)
    if not library_compat.vocabulary_is_available(loaded.scenario_version):
        return CheckResult(label, unavailable="no trigger vocabulary for this scenario version")
    vocabulary = library_compat.load_vocabulary(loaded.scenario_version)
    index = build_reference_index(loaded)
    kinds = (
        ("condition", "conditions", "condition_type", vocabulary.conditions, vocabulary.condition_presentation),
        ("effect", "effects", "effect_type", vocabulary.effects, vocabulary.effect_presentation),
    )
    findings = []
    for trigger in manager.triggers:
        for kind, list_name, type_attribute, entries, presentation in kinds:
            for ce in getattr(trigger, list_name):
                definition = entries.get(_read(ce, type_attribute))
                if definition is None:
                    continue
                for attribute, refs in references_in(ce, definition, presentation).items():
                    for ref in refs:
                        if ref in index.by_id:
                            continue
                        findings.append(
                            Finding(
                                f"{_trigger_label(trigger)}: {definition.name} {kind} "
                                f"{attribute} refers to unit id {ref}, which is not placed on the map",
                                "warning",
                            )
                        )
    return _capped(label, findings)


def check_garrison_links(loaded: LoadedScenario) -> CheckResult:
    """A unit garrisoned inside a reference_id no unit carries (GH #42).

    Worth reporting because such a unit is invisible under the default
    filter -- it is hidden as garrisoned, but there is no host to find it
    inside. Legal on disk and accepted by the in-game editor, so a warning,
    never an error.
    """
    label = "Dangling garrison links"
    ids = {unit.reference_id for _player_id, unit in _all_units(loaded)}
    findings = [
        Finding(
            f"Player {player_id} unit {unit.unit_const} (id {unit.reference_id}) is garrisoned in "
            f"unit id {_read(unit, 'garrisoned_in_id')}, which is not placed on the map",
            "warning",
            tile=(int(unit.x), int(unit.y)),
            unit_key=(player_id, unit.reference_id),
        )
        for player_id, unit in _all_units(loaded)
        if (host_id := _read(unit, "garrisoned_in_id")) not in (None, -1, unit.reference_id) and host_id not in ids
    ]
    return _capped(label, findings)


def check_players_without_units(loaded: LoadedScenario) -> CheckResult:
    """WARNING, never ERROR: a Create Object effect at time 0 is a
    legitimate way to spawn a player's starting forces."""
    label = "Active players with no placed units"
    units = loaded.unit_manager.units
    findings = [
        Finding(
            f"Player {player_id} is active but has no placed units (triggers may create them at runtime)",
            "warning",
        )
        for player_id in defined_player_ids(loaded)
        if player_id < len(units) and not units[player_id]
    ]
    return CheckResult(label, tuple(findings))


def check_map_shape(loaded: LoadedScenario) -> CheckResult:
    label = "Map shape"
    if loaded.map_is_square:
        return CheckResult(label)
    mm = loaded.map_manager
    return CheckResult(
        label,
        (
            Finding(
                f"Map is {mm.map_width}×{mm.map_height}, not square; elevation tools and "
                "unit reachability are unavailable for it",
                "info",
            ),
        ),
    )


def _messages(loaded: LoadedScenario):
    try:
        return loaded._scenario.sections["Messages"].retriever_map
    except (AttributeError, KeyError):
        return None


def check_instructions(loaded: LoadedScenario) -> CheckResult:
    """A set string-table id wins over typed text in game, so empty text
    with an id set can't be judged from the file."""
    label = "Scenario instructions"
    messages = _messages(loaded)
    if messages is None:
        return CheckResult(label, unavailable="the Messages section could not be read")
    text = messages["ascii_instructions"].data
    string_id = messages["instructions"].data
    if text:
        return CheckResult(label)
    if string_id != STRING_ID_UNSET:
        return CheckResult(label, unavailable=f"string-table id {string_id} is set; its in-game text can't be read")
    return CheckResult(
        label,
        (Finding("No scenario instructions text; the objectives screen will be empty unless triggers fill it", "info"),),
    )


def _land_fn(loaded: LoadedScenario) -> Callable[[int], bool]:
    """Terrain-only, deliberately: treating trees and buildings as blockers
    too raised the corpus fire rate from 4/20 to 11-14/20 files."""
    terrain = loaded.map_manager.terrain
    water_ids: dict[int, bool] = {}

    def is_land(i: int) -> bool:
        terrain_id = terrain[i].terrain_id
        if terrain_id not in water_ids:
            try:
                name = TerrainId(terrain_id).name
            except ValueError:
                name = ""
            # batch_api.is_water()'s rule, with an unknown id counted as land.
            water_ids[terrain_id] = "WATER" in name or "SHALLOW" in name
        return not water_ids[terrain_id]

    return is_land


def check_stranded_units(loaded: LoadedScenario) -> CheckResult:
    """Approximate: see the module docstring. GAIA is excluded, which is what
    takes the corpus fire rate from 9/20 to usable."""
    label = f"Player units on small land areas (<{STRANDED_REGION_TILES} tiles, approximate)"
    if not loaded.map_is_square:
        return CheckResult(label, unavailable="not run on a non-square map")
    mm = loaded.map_manager
    width, height = mm.map_width, mm.map_height
    region_size: dict[int, int] = {}
    for region in connected_regions(mm, _land_fn(loaded)):
        if len(region) < STRANDED_REGION_TILES:
            for i in region:
                region_size[i] = len(region)
    findings = []
    for player_id, unit in _all_units(loaded):
        if player_id == 0:
            continue
        x, y = int(unit.x), int(unit.y)
        if not (0 <= x < width and 0 <= y < height):
            continue
        size = region_size.get(y * width + x)
        if size is None:
            continue
        findings.append(
            Finding(
                f"Player {player_id} unit {unit.unit_const} (id {unit.reference_id}) at ({unit.x:g}, {unit.y:g}) "
                f"is on a {size}-tile land area; may be a deliberate outpost",
                "info",
                tile=(x, y),
                unit_key=(player_id, unit.reference_id),
            )
        )
    return _capped(label, findings)


def _location_label(edit: train_locations.LocationEdit) -> str:
    return f"{edit.label} (effect {edit.index})"


def _location_problem(
    edit: train_locations.LocationEdit, loaded: LoadedScenario, overrides, moved_away, placed: frozenset[int]
) -> tuple[str, Severity] | None:
    """(reason, severity) for the worst thing wrong with `edit`'s button, or None."""
    building, value, mover, kind = edit.building, edit.value, edit.mover, edit.kind
    if value == 0:
        return "button slot 0 gives no button", "info"
    if command_buttons.slot(value) is None:
        return f"button slot {value} is no button (1-15, or 21-35 for page 2)", "warning"
    if building in command_buttons.VILLAGER_IDS:
        return f"trained at Villager {building} moves nothing; use the builder (118)", "warning"
    reserved = command_buttons.reserved_label(building, value)
    if reserved:
        return f"slot {value} is the build menu's {reserved} button, so it has no visible button", "warning"
    civ = civ_availability.player_civ(loaded, edit.player)
    civ_text = civ_availability.civ_name(loaded, edit.player)
    start_age = civ_availability.player_start_age(loaded, edit.player)

    def name() -> str:
        # Only for a finding: tech_name() re-reads the settings on each call.
        return object_catalog.tech_name(mover) if kind == TECH else object_catalog.display_name(mover)

    reason = civ_availability.missing(civ, overrides, mover, kind, start_age)
    if reason:
        text = train_locations.missing_text(reason, kind, name(), civ_text, edit.player, start_age)
        how = train_locations.enable_hint(reason, kind)
        return f"{text}, so it gets no button" + (f"; enable it with {how}" if how else ""), "warning"
    if (
        not command_buttons.is_builder(building)
        and building not in placed
        and not civ_availability.available(civ, overrides, building, UNIT)
    ):
        return f"{civ_text} can't build {object_catalog.display_name(building)}", "warning"

    def keep(id_: int, item_kind: str) -> bool:
        if (id_, item_kind) in moved_away or overrides.turned_off((id_, item_kind)):
            return False
        return not civ_availability.missing(civ, overrides, id_, item_kind, start_age)

    where = f"{command_buttons.building_name(building)}, {command_buttons.slot_label(value)}"
    if kind == UNIT:
        hider = command_buttons.blocker(mover, building, value, keep)
        if hider is not None:
            return f"{name()} is hidden by {object_catalog.display_name(hider)} (the lower id shows) at {where}", "warning"
        if (building, value) not in command_buttons.home_slots(mover, UNIT):
            covered = command_buttons.covered_techs(building, value, keep)
            if covered:
                names = ", ".join(object_catalog.tech_name(tech) for tech in covered)
                return f"{name()} covers {names} at {where}, so it can't be researched", "warning"
    else:
        hider = command_buttons.covering_unit(mover, building, value, keep)
        if hider is not None:
            return f"{name()} is hidden by {object_catalog.display_name(hider)}, a unit in the slot at {where}", "warning"
    return None


def check_location_buttons(loaded: LoadedScenario) -> CheckResult:
    label = "Train and research buttons"
    manager = parse_triggers(loaded)
    if manager is None:
        return CheckResult(label, unavailable=_TRIGGERS_UNAVAILABLE)
    edits = train_locations.location_edits(manager)
    # A building placed on the map exists whatever the civ (Kreposts given to Magyars).
    placed = frozenset(unit.unit_const for _player, unit in _all_units(loaded))
    overrides: dict[int, civ_availability.Overrides] = {}
    findings = []
    for edit in edits:
        if edit.mover < 0 or edit.building < 0:
            continue
        if edit.player not in overrides:
            overrides[edit.player] = civ_availability.file_overrides(manager, loaded, edit.player)
        moved_away = {
            other.item
            for other in train_locations.for_player(edits, edit.player)
            if other is not edit and other.effect in train_locations.MOVES_AWAY
        }
        problem = _location_problem(edit, loaded, overrides[edit.player], moved_away, placed)
        if problem is not None:
            reason, severity = problem
            findings.append(Finding(f"{_location_label(edit)}: {reason}", severity))
    return _capped(label, findings)


CHECKS: tuple[tuple[str, Callable[[LoadedScenario], CheckResult]], ...] = (
    ("elevation", check_elevation),
    ("off_map_units", check_off_map_units),
    ("off_grid_buildings", check_off_grid_buildings),
    ("off_grid_player_units", check_off_grid_player_units),
    ("off_grid_gaia_units", check_off_grid_gaia_units),
    ("gaia_only_ownership", check_gaia_only_ownership),
    ("units_on_wrong_terrain", check_units_on_wrong_terrain),
    ("buildings_on_wrong_terrain", check_buildings_on_wrong_terrain),
    ("overlapping_units", check_overlapping_units),
    ("trigger_display_order", check_trigger_display_order),
    ("trigger_references", check_trigger_references),
    ("location_buttons", check_location_buttons),
    ("unit_references", check_unit_references),
    ("garrison_links", check_garrison_links),
    ("players_without_units", check_players_without_units),
    ("stranded_units", check_stranded_units),
    ("instructions", check_instructions),
    ("map_shape", check_map_shape),
)


def analyze(loaded: LoadedScenario) -> AnalysisReport:
    return AnalysisReport(tuple(check(loaded) for _, check in CHECKS))


# -- map markers ----------------------------------------------------------------

_SEVERITY_RANK: dict[str, int] = {"info": 0, "warning": 1, "error": 2}

MarkerAnchor = tuple[tuple[int, int], Severity, int]


def marker_tile(finding: Finding, map_w: int, map_h: int) -> tuple[int, int] | None:
    """The tile a finding's map marker sits on: its own tile clamped to the
    map, the way ViewerWindow._navigate_to_finding clamps. Unit-only findings
    (off-map units) have no tile and so no marker."""
    if finding.tile is None or map_w <= 0 or map_h <= 0:
        return None
    x, y = finding.tile
    return max(0, min(map_w - 1, int(x))), max(0, min(map_h - 1, int(y)))


def marker_anchors(report: AnalysisReport, map_w: int, map_h: int) -> list[MarkerAnchor]:
    """One (tile, worst severity, finding count) per located tile, in the
    order each tile first appears in the report."""
    anchors: dict[tuple[int, int], tuple[Severity, int]] = {}
    for result in report.results:
        for finding in result.findings:
            tile = marker_tile(finding, map_w, map_h)
            if tile is None:
                continue
            if tile not in anchors:
                anchors[tile] = (finding.severity, 1)
                continue
            worst, count = anchors[tile]
            if _SEVERITY_RANK.get(finding.severity, 0) > _SEVERITY_RANK.get(worst, 0):
                worst = finding.severity
            anchors[tile] = (worst, count + 1)
    return [(tile, severity, count) for tile, (severity, count) in anchors.items()]
