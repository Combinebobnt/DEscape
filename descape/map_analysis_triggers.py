"""Map Analysis's trigger checks (GH #282): coordinates, Create Object
spots, completeness, players and triggers that never start.

Qt-free, like descape/map_analysis.py, whose CHECKS registry lists these.
Each returns unavailable, never clean, when the Triggers section does not
parse, and every check but the never-started one also needs the scenario
version's trigger vocabulary, which says which fields each type uses.

Measured 2026-11-26 over the 21 loadable examples/ files, 17 of them with
parseable triggers. A location or an area wholly off the map fires on 0
(WARNING); an area reaching past the edge on 1 file, 3 areas reaching
x = 259 on a 240 map (INFO). A Create Object or Place Foundation unit on
terrain it can't stand on fires on 0 (WARNING); what the game does with
one is unconfirmed, so the message states only the terrain rule.
Incomplete triggers fire on 6 files, 21 triggers, mostly "has no effects"
(INFO). An entry naming an undefined player fires once (INFO). A disabled
trigger no Activate Trigger effect names fires on 4 files, 26 triggers
(INFO): campaigns keep debug and fallback triggers that way.
"""

from __future__ import annotations

from AoE2ScenarioParser.datasets.effects import EffectId
from AoE2ScenarioParser.objects.managers.trigger_manager import get_trigger_referencing_ce

from descape import library_compat, object_catalog, placement_rules, trigger_geometry, trigger_organize, trigger_status
from descape.map_analysis_types import _TRIGGERS_UNAVAILABLE, CheckResult, Finding, Severity, _capped, _read, _trigger_label
from descape.player_fields import defined_player_ids
from descape.scenario_io import LoadedScenario, parse_triggers
from descape.trigger_model import entry_display_order
from descape.unit_kind import mobile_unit_consts

_NO_VOCABULARY = "no trigger vocabulary for this scenario version"
_PLAYER_FIELDS = ("source_player", "target_player")
_KINDS = (
    (trigger_geometry.ENTRY_CONDITION, "conditions", "condition_type"),
    (trigger_geometry.ENTRY_EFFECT, "effects", "effect_type"),
)


def _manager_and_vocabulary(loaded: LoadedScenario, label: str):
    """(manager, vocabulary, None), or (None, None, the unavailable CheckResult)."""
    manager = parse_triggers(loaded)
    if manager is None:
        return None, None, CheckResult(label, unavailable=_TRIGGERS_UNAVAILABLE)
    if not library_compat.vocabulary_is_available(loaded.vocabulary_version):
        return None, None, CheckResult(label, unavailable=_NO_VOCABULARY)
    return manager, library_compat.load_vocabulary(loaded.vocabulary_version), None


def _entry_label(trigger, kind: str, index: int, vocabulary) -> str:
    """trigger_status.entry_label() numbered by display slot, as the entry tree shows it."""
    order = entry_display_order(trigger, kind)
    slot = order.index(index) if index in order else index
    entries = _read(trigger, f"{kind}s") or ()
    return trigger_status.entry_label(kind, slot, entries[index], vocabulary)


def _entries(trigger):
    """(kind, index, entry, type id) over conditions then effects, in stored order."""
    for kind, list_name, type_attribute in _KINDS:
        for index, entry in enumerate(_read(trigger, list_name) or ()):
            yield kind, index, entry, _read(entry, type_attribute)


def _clamped(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    return max(0, min(width - 1, x)), max(0, min(height - 1, y))


def check_trigger_coordinates(loaded: LoadedScenario) -> CheckResult:
    """A location, or an area or wall run with no tile on the map, is a
    WARNING; an area or run that only reaches past the edge is INFO. Unit
    shapes are derived from placed units, so they are not read."""
    label = "Trigger areas and locations off the map"
    manager, vocabulary, unavailable = _manager_and_vocabulary(loaded, label)
    if unavailable is not None:
        return unavailable
    mm = loaded.map_manager
    width, height = mm.map_width, mm.map_height
    findings = []
    for trigger_index, trigger in enumerate(manager.triggers):
        for shape in trigger_geometry.shapes_for_trigger(trigger, vocabulary, trigger_index=trigger_index):
            if shape.shape == trigger_geometry.SHAPE_LOCATION:
                x, y = shape.coords
                if x < width and y < height:
                    continue
                what, severity = f"location ({x}, {y}) is outside", "warning"
                corner = (x, y)
            else:
                x1, y1, x2, y2 = shape.coords
                low_x, low_y, high_x, high_y = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
                if high_x < width and high_y < height:
                    continue
                name = "area" if shape.shape == trigger_geometry.SHAPE_AREA else "wall run"
                where = f"{name} ({x1}, {y1})-({x2}, {y2})"
                if low_x >= width or low_y >= height:
                    what, severity = f"{where} is wholly outside", "warning"
                else:
                    what, severity = f"{where} reaches past the edge of", "info"
                corner = (low_x, low_y)
            entry = _entry_label(trigger, shape.entry_kind, shape.entry_index, vocabulary)
            findings.append(
                Finding(
                    f"{_trigger_label(trigger)}: {entry} {what} the {width}x{height} map",
                    severity,
                    tile=_clamped(*corner, width, height),
                    trigger_index=trigger_index,
                    trigger_entry=shape.entry_ref,
                )
            )
    findings.sort(key=lambda f: f.severity != "warning")  # stable: the cap never hides a warning behind info
    return _capped(label, findings)


def check_created_object_terrain(loaded: LoadedScenario) -> CheckResult:
    """A Create Object or Place Foundation effect whose unit can't stand on
    its location's terrain, by placement_rules.terrain_refusal() at the tile
    centre. A mobile unit reads the foundations under it, as the placed-unit
    check does. An off-map location is the coordinates check's finding."""
    label = "Created objects on terrain they can't stand on"
    manager, vocabulary, unavailable = _manager_and_vocabulary(loaded, label)
    if unavailable is not None:
        return unavailable
    mm = loaded.map_manager
    mobile = mobile_unit_consts()
    laid = None
    findings = []
    for trigger_index, trigger in enumerate(manager.triggers):
        for created in trigger_geometry.created_units_for_trigger(trigger, vocabulary, trigger_index=trigger_index):
            x, y = created.tile
            if x >= mm.map_width or y >= mm.map_height:
                continue
            if created.unit_const in mobile and laid is None:
                laid = placement_rules.laid_terrain(loaded)
            reason = placement_rules.terrain_refusal(
                loaded, created.unit_const, x + 0.5, y + 0.5, laid if created.unit_const in mobile else None
            )
            if reason is None:
                continue
            entry = _entry_label(trigger, trigger_geometry.ENTRY_EFFECT, created.entry_index, vocabulary)
            name = object_catalog.display_name(created.unit_const)
            findings.append(
                Finding(
                    f"{_trigger_label(trigger)}: {entry} puts {name} at ({x}, {y}): {reason}",
                    "warning",
                    tile=(x, y),
                    trigger_index=trigger_index,
                    trigger_entry=created.entry_ref,
                )
            )
    return _capped(label, findings)


def _named_trigger_ids(manager) -> frozenset[int]:
    return frozenset(ce.trigger_id for trigger in manager.triggers for ce in get_trigger_referencing_ce(trigger))


def check_incomplete_triggers(loaded: LoadedScenario) -> CheckResult:
    """The trigger list's own completeness rule (trigger_status, GH #166),
    one finding per trigger with its reasons joined, plus an entry type the
    vocabulary does not list. Dangling unit and trigger ids are left to the
    two dangling-reference checks: the context here accepts every id."""
    label = "Incomplete triggers"
    manager, vocabulary, unavailable = _manager_and_vocabulary(loaded, label)
    if unavailable is not None:
        return unavailable
    ids = frozenset(t.trigger_id for t in manager.triggers) | _named_trigger_ids(manager)
    context = trigger_status.StatusContext(vocabulary, ids)
    definitions = {trigger_geometry.ENTRY_CONDITION: vocabulary.conditions, trigger_geometry.ENTRY_EFFECT: vocabulary.effects}
    findings = []
    for trigger_index, trigger in enumerate(manager.triggers):
        status = trigger_status.evaluate_trigger(trigger, context).trigger
        reasons = list(status.reasons) if status.status is trigger_status.Status.PROBLEM else []
        for kind, index, _entry, type_id in _entries(trigger):
            if type_id not in (0, None) and type_id not in definitions[kind]:
                reasons.append(f"{_entry_label(trigger, kind, index, vocabulary)}: this version has no such {kind} type")
        if reasons:
            findings.append(Finding(f"{_trigger_label(trigger)}: {'; '.join(reasons)}", "info", trigger_index=trigger_index))
    return _capped(label, findings)


def check_trigger_players(loaded: LoadedScenario, defined_players: list[int] | None = None) -> CheckResult:
    """A condition or effect whose source or target player is one the
    scenario does not define. Read only where the entry's type uses the
    field: a field the type ignores can hold a leftover value. GAIA (0) and
    unset are fine. `defined_players` is the count as edited, as in
    map_analysis.analyze()."""
    label = "Trigger entries naming an undefined player"
    manager, vocabulary, unavailable = _manager_and_vocabulary(loaded, label)
    if unavailable is not None:
        return unavailable
    defined = set(defined_player_ids(loaded) if defined_players is None else defined_players)
    findings = []
    for trigger_index, trigger in enumerate(manager.triggers):
        for kind, index, entry, _type_id in _entries(trigger):
            definition = trigger_status.definition_for(kind, entry, vocabulary)
            if definition is None:
                continue
            for field in _PLAYER_FIELDS:
                if field not in definition.attributes:
                    continue
                player = _read(entry, field)
                if not isinstance(player, int) or player <= 0 or player in defined:
                    continue
                findings.append(
                    Finding(
                        f"{_trigger_label(trigger)}: {_entry_label(trigger, kind, index, vocabulary)} "
                        f"{field.replace('_', ' ')} is player {player}, which this scenario does not define",
                        "info",
                        trigger_index=trigger_index,
                        trigger_entry=(kind, index),
                    )
                )
    return _capped(label, findings)


def check_never_started_triggers(loaded: LoadedScenario) -> CheckResult:
    """A disabled trigger that no other trigger's Activate Trigger effect
    names, so nothing in the triggers starts it. Section dividers are skipped."""
    label = "Triggers that never start"
    manager = parse_triggers(loaded)
    if manager is None:
        return CheckResult(label, unavailable=_TRIGGERS_UNAVAILABLE)
    activated = {
        target
        for trigger in manager.triggers
        for effect in _read(trigger, "effects") or ()
        if _read(effect, "effect_type") == EffectId.ACTIVATE_TRIGGER
        and (target := _read(effect, "trigger_id")) != trigger.trigger_id
    }
    severity: Severity = "info"
    findings = [
        Finding(
            f'Trigger {trigger.trigger_id} "{trigger.name}" is disabled and no Activate Trigger effect names it',
            severity,
            trigger_index=trigger_index,
        )
        for trigger_index, trigger in enumerate(manager.triggers)
        if not _read(trigger, "enabled")
        and trigger.trigger_id not in activated
        and not trigger_organize.is_divider(str(_read(trigger, "name") or ""))
    ]
    return _capped(label, findings)
