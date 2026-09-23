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

Nothing here constructs an edit model, and nothing may be named verify_* or
*_supported (both already mean byte-layout trust elsewhere).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from AoE2ScenarioParser.datasets.terrains import TerrainId
from AoE2ScenarioParser.objects.managers.trigger_manager import get_trigger_referencing_ce

from descape import library_compat
from descape.fill_tools import connected_regions
from descape.iso_geometry import MAX_ELEVATION, MIN_ELEVATION
from descape.messages_fields import STRING_ID_UNSET
from descape.player_fields import defined_player_ids
from descape.render import unit_tile_bounds
from descape.scenario_io import LoadedScenario, parse_triggers
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


CHECKS: tuple[tuple[str, Callable[[LoadedScenario], CheckResult]], ...] = (
    ("elevation", check_elevation),
    ("off_map_units", check_off_map_units),
    ("trigger_display_order", check_trigger_display_order),
    ("trigger_references", check_trigger_references),
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
