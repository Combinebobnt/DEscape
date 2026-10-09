"""Qt-free search and replace-planning over triggers, for Edit > Find and
Replace's Triggers tab (GH #144 Part B).

Field classes come from each field's presentation (trigger_fields.field_specs()
over the file's own vocabulary), never from attribute names: `unit_object`
has presentation "" before v1.44, so a name rule misreads older files.

Nothing here mutates. find_triggers() returns hits; the plan_* functions turn
checked hits into a list of writes plus refusals, which the viewer applies in
one bracket declaring only the triggers that change. The manager is re-fetched
through scenario_io.parse_triggers() on every call, never held
(trigger_model.py's invariant 1).
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from descape import (
    library_compat,
    messages_fields,
    object_catalog,
    scenario_io,
    trigger_clipboard,
    trigger_fields,
    trigger_organize,
    unit_kind,
    unit_references,
)
from descape.find_text import FindPatternError, compile_pattern, refuse_forbidden_path

TEXT_SCOPES = ("names", "descriptions", "messages", "identifiers", "xs")
DEFAULT_SCOPES = frozenset({"names", "descriptions", "messages"})
TYPE_PRESENTATIONS = frozenset({"UnitInfo", "BuildingInfo"})

# str32 in the structure, but field_specs() classifies them INT (presentation
# "", default None), so a STR-only scan misses them.
EXTRA_TEXT_FIELDS = frozenset({("create_decision", "message_option1"), ("create_decision", "message_option2")})
# STR fields that hold a name, not prose. Same (entry name, attribute) shape as
# trigger_fields.PROSE_FIELDS; `sound_name` is an identifier on every entry.
IDENTIFIER_FIELDS = frozenset(
    {
        ("change_variable", "message"),
        ("load_key_value", "message"),
        ("store_key_value", "message"),
        ("delete_key", "message"),
    }
)
IDENTIFIER_ATTRIBUTES = frozenset({"sound_name"})

# How a text field is matched and written back.
MODE_PROSE = "prose"  # messages_fields display form, file newline token kept
MODE_XS = "xs"  # trigger_fields.xs_to_display / xs_from_display
MODE_LINE = "line"  # verbatim, one line

_TRIGGER_TEXT = (("name", "names", MODE_LINE), ("short_description", "descriptions", MODE_PROSE),
                 ("description", "descriptions", MODE_PROSE))


@dataclass(frozen=True)
class TriggerFindCriteria:
    text: str = ""
    regex: bool = False
    case_sensitive: bool = False
    text_scopes: frozenset[str] = DEFAULT_SCOPES
    type_consts: frozenset[int] | None = None
    object_refs: frozenset[int] | None = None
    enabled: bool | None = None


@dataclass(frozen=True)
class TriggerHit:
    trigger_index: int
    trigger_name: str
    enabled: bool
    kind: str  # "trigger", "condition" or "effect"
    entry_index: int  # -1 for the trigger's own fields
    entry_name: str
    attribute: str  # "" for a facet-free whole-trigger row
    field_class: str  # "text", "xs", "type", "object_ref" or "trigger"
    value_display: str
    found_value: Any
    spans: tuple[tuple[int, int], ...] = ()
    text_mode: str = ""

    @property
    def location(self) -> tuple[int, str, int, str]:
        return (self.trigger_index, self.kind, self.entry_index, self.attribute)


@dataclass(frozen=True)
class Write:
    trigger_index: int
    kind: str
    entry_index: int
    attribute: str
    new_value: Any


@dataclass
class ReplacePlan:
    writes: list[Write] = field(default_factory=list)
    refusals: list[tuple[TriggerHit, str]] = field(default_factory=list)

    @property
    def touched(self) -> list[int]:
        return sorted({w.trigger_index for w in self.writes})


def _read(obj: Any, attribute: str) -> Any:
    try:
        return getattr(obj, attribute, None)
    except Exception:  # noqa: BLE001 -- the library's version-gated properties raise
        return None


def _vocabulary(loaded):
    version = loaded.scenario_version
    if version and library_compat.vocabulary_is_available(version):
        return library_compat.load_vocabulary(version)
    return None


def text_scope(entry_name: str, spec: trigger_fields.FieldSpec) -> tuple[str, str] | None:
    """(scope, mode) for a text field of a condition/effect, or None if it is not text."""
    pair = (entry_name, spec.name)
    if pair in EXTRA_TEXT_FIELDS:
        return "messages", MODE_LINE
    if spec.kind != trigger_fields.STR:
        return None
    if spec.multiline == trigger_fields.XS:
        return "xs", MODE_XS
    if pair in IDENTIFIER_FIELDS or spec.name in IDENTIFIER_ATTRIBUTES:
        return "identifiers", MODE_LINE
    return "messages", MODE_PROSE if spec.multiline == trigger_fields.PROSE else MODE_LINE


def to_display(value: str, mode: str) -> str:
    if mode == MODE_PROSE:
        return messages_fields.normalize_for_display(value)[0]
    if mode == MODE_XS:
        return trigger_fields.xs_to_display(value)
    return value


def from_display(display: str, stored: str, mode: str) -> str:
    if mode == MODE_PROSE:
        return messages_fields.substitute_newlines(display, messages_fields.normalize_for_display(stored)[1])
    if mode == MODE_XS:
        return trigger_fields.xs_from_display(display)
    return display


def _ref_display(ids: Sequence[int]) -> str:
    return ", ".join(str(i) for i in ids)


def _entries(trigger, kind: str):
    return trigger.conditions if kind == "condition" else trigger.effects


def _walk(vocabulary, trigger):
    return trigger_clipboard.entry_fields(vocabulary, trigger.conditions, trigger.effects)


def find_triggers(loaded, criteria: TriggerFindCriteria) -> list[TriggerHit]:
    """One hit per matching field, grouped by trigger in list order. Facets AND
    per trigger: text (non-empty text), types and object refs each need a hit
    somewhere in the trigger, its own fields or its entries. With no facet set,
    each trigger passing `enabled` is one field_class "trigger" hit."""
    manager = scenario_io.parse_triggers(loaded)
    if manager is None:
        return []
    vocabulary = _vocabulary(loaded)
    pattern = compile_pattern(criteria.text, criteria.regex, criteria.case_sensitive)
    scopes = criteria.text_scopes
    results: list[TriggerHit] = []
    for t_index, trigger in enumerate(manager.triggers):
        enabled = bool(trigger.enabled)
        if criteria.enabled is not None and enabled != criteria.enabled:
            continue
        name = trigger.name or ""

        def hit(kind, e_index, entry_name, attribute, field_class, display, value, spans=(), mode="", _n=name,
                _i=t_index, _e=enabled):
            return TriggerHit(_i, _n, _e, kind, e_index, entry_name, attribute, field_class, display, value,
                              tuple(spans), mode)

        text_hits: list[TriggerHit] = []
        type_hits: list[TriggerHit] = []
        ref_hits: list[TriggerHit] = []
        if pattern is not None:
            for attribute, scope, mode in _TRIGGER_TEXT:
                value = _read(trigger, attribute)
                if scope in scopes and isinstance(value, str):
                    display = to_display(value, mode)
                    spans = [m.span() for m in pattern.finditer(display) if m.end() > m.start()]
                    if spans:
                        text_hits.append(hit("trigger", -1, "", attribute, "text", display, value, spans, mode))
        if vocabulary is not None:
            seen_entries = set()
            for f in _walk(vocabulary, trigger):
                entry_name = f.definition.name
                if pattern is not None:
                    classified = text_scope(entry_name, f.spec)
                    if classified is not None and classified[0] in scopes:
                        value = _read(f.entry, f.spec.attribute)
                        if isinstance(value, str):
                            display = to_display(value, classified[1])
                            spans = [m.span() for m in pattern.finditer(display) if m.end() > m.start()]
                            if spans:
                                field_class = "xs" if classified[1] == MODE_XS else "text"
                                text_hits.append(
                                    hit(f.kind, f.index, entry_name, f.spec.attribute, field_class, display, value,
                                        spans, classified[1])
                                )
                if criteria.type_consts is not None and f.spec.presentation in TYPE_PRESENTATIONS:
                    value = _read(f.entry, f.spec.attribute)
                    if isinstance(value, int) and value in criteria.type_consts:
                        type_hits.append(
                            hit(f.kind, f.index, entry_name, f.spec.attribute, "type",
                                object_catalog.combined_object_name(value) or str(value), value)
                        )
                if criteria.object_refs is not None and (f.kind, f.index) not in seen_entries:
                    seen_entries.add((f.kind, f.index))
                    refs = unit_references.references_in(f.entry, f.definition, f.presentation_map)
                    for attribute, ids in refs.items():
                        if criteria.object_refs.intersection(ids):
                            ref_hits.append(hit(f.kind, f.index, entry_name, attribute, "object_ref",
                                                _ref_display(ids), tuple(ids)))
        facets = [
            (pattern is not None, text_hits),
            (criteria.type_consts is not None, type_hits),
            (criteria.object_refs is not None, ref_hits),
        ]
        active = [hits for on, hits in facets if on]
        if not active:
            results.append(hit("trigger", -1, "", "", "trigger", name, name))
            continue
        if all(active):
            results.extend(h for hits in active for h in hits)
    return results


def trigger_ref_counts(loaded) -> dict[int, int] | None:
    """reference_id -> how many condition/effect fields reference it. None
    while triggers are unparsed: this never forces the (seconds-long) parse."""
    if loaded._trigger_manager is None:
        return None
    manager = scenario_io.parse_triggers(loaded)
    vocabulary = _vocabulary(loaded) if manager is not None else None
    if vocabulary is None:
        return None
    counts: Counter = Counter()
    for trigger in manager.triggers:
        seen = set()
        for f in _walk(vocabulary, trigger):
            if (f.kind, f.index) in seen:
                continue
            seen.add((f.kind, f.index))
            for ids in unit_references.references_in(f.entry, f.definition, f.presentation_map).values():
                for ref in set(ids):
                    counts[ref] += 1
    return dict(counts)


def entry_at(manager, trigger_index: int, kind: str, entry_index: int):
    """The live condition/effect (or the trigger itself for kind "trigger"), or None."""
    if not 0 <= trigger_index < len(manager.triggers):
        return None
    trigger = manager.triggers[trigger_index]
    if kind == "trigger":
        return trigger
    entries = _entries(trigger, kind)
    return entries[entry_index] if 0 <= entry_index < len(entries) else None


def _live_value(target, h: TriggerHit):
    value = _read(target, h.attribute)
    if h.field_class == "object_ref":
        values = value if isinstance(value, (list, tuple)) else (value,)
        return tuple(v for v in values if isinstance(v, int) and not isinstance(v, bool) and v != trigger_fields.UNSET)
    return value


def revalidate(loaded, hits: Iterable[TriggerHit]) -> tuple[list[TriggerHit], list[TriggerHit]]:
    """(valid, stale): a hit is valid while its trigger index still holds the
    same trigger name (own fields) or an entry of the same vocabulary type at
    (kind, entry_index), and the field still holds `found_value`. A stale hit
    is never written; trigger indices renumber on delete and undo."""
    manager = scenario_io.parse_triggers(loaded)
    vocabulary = _vocabulary(loaded) if manager is not None else None
    valid: list[TriggerHit] = []
    stale: list[TriggerHit] = []
    for h in hits:
        ok = False
        if manager is not None:
            target = entry_at(manager, h.trigger_index, h.kind, h.entry_index)
            if target is not None and h.kind == "trigger":
                ok = (target.name or "") == h.trigger_name and (
                    h.field_class == "trigger" or _live_value(target, h) == h.found_value
                )
            elif target is not None and vocabulary is not None:
                definitions = vocabulary.conditions if h.kind == "condition" else vocabulary.effects
                type_attribute = "condition_type" if h.kind == "condition" else "effect_type"
                definition = definitions.get(_read(target, type_attribute))
                ok = definition is not None and definition.name == h.entry_name and _live_value(target, h) == h.found_value
        (valid if ok else stale).append(h)
    return valid, stale


def _tag_or_divider_changes(old: str, new: str) -> bool:
    return (
        trigger_organize.parse_tag(old) != trigger_organize.parse_tag(new)
        or trigger_organize.is_divider(old) != trigger_organize.is_divider(new)
    )


def plan_text_replace(loaded, hits: Iterable[TriggerHit], pattern: re.Pattern, replacement: str,
                      regex: bool) -> ReplacePlan:
    """Writes for the checked text/xs hits. Regex mode treats `replacement` as a
    template (\\1 works); substring mode passes it through a function, so a
    backslash stays literal. A write that changes nothing is dropped. Raises
    FindPatternError for an invalid template."""
    manager = scenario_io.parse_triggers(loaded)
    plan = ReplacePlan()
    if manager is None:
        return plan
    repl = replacement if regex else (lambda _m: replacement)
    for h in hits:
        if h.field_class not in ("text", "xs"):
            continue
        target = entry_at(manager, h.trigger_index, h.kind, h.entry_index)
        stored = _read(target, h.attribute) if target is not None else None
        if not isinstance(stored, str):
            plan.refusals.append((h, "field no longer holds text"))
            continue
        display = to_display(stored, h.text_mode)
        try:
            new_display = pattern.sub(repl, display)
        except (re.error, IndexError) as e:
            raise FindPatternError(f"Invalid replacement: {e}") from e
        if new_display == display:
            continue
        if "\x00" in new_display:
            plan.refusals.append((h, "result contains a NUL character"))
            continue
        if h.text_mode == MODE_LINE and ("\n" in new_display or "\r" in new_display):
            plan.refusals.append((h, "newline in a single-line field"))
            continue
        if h.kind == "trigger" and h.attribute == "name" and _tag_or_divider_changes(stored, new_display):
            plan.refusals.append((h, "would move the trigger to another tag or section"))
            continue
        new_value = from_display(new_display, stored, h.text_mode)
        if new_value != stored:
            plan.writes.append(Write(h.trigger_index, h.kind, h.entry_index, h.attribute, new_value))
    return plan


def plan_type_replace(loaded, hits: Iterable[TriggerHit], old_consts: frozenset[int], new_const: int) -> ReplacePlan:
    """Writes `new_const` into the checked type-field hits whose live value is in
    `old_consts`. UnitInfo and BuildingInfo both hold any catalog object, so the
    category is not policed; an unknown const refuses every hit."""
    manager = scenario_io.parse_triggers(loaded)
    plan = ReplacePlan()
    if manager is None:
        return plan
    known = object_catalog.is_known_object(new_const)
    for h in hits:
        if h.field_class != "type":
            continue
        if not known:
            plan.refusals.append((h, f"unknown object type {new_const}"))
            continue
        target = entry_at(manager, h.trigger_index, h.kind, h.entry_index)
        value = _read(target, h.attribute) if target is not None else None
        if not isinstance(value, int) or value not in old_consts:
            plan.refusals.append((h, "field no longer holds that type"))
            continue
        if value != new_const:
            plan.writes.append(Write(h.trigger_index, h.kind, h.entry_index, h.attribute, new_const))
    return plan


# -- follow an object Replace into trigger type filters -----------------------

TIER_PAIRED = "a"
TIER_FILTER_ONLY = "b"
TIER_MIXED = "c"
TIER_REPORT = "d"
TIER_CLASS = "e"


@dataclass(frozen=True)
class FollowEntry:
    tier: str
    trigger_index: int
    trigger_name: str
    kind: str
    entry_index: int
    entry_name: str
    attribute: str
    old_const: int
    new_const: int | None
    reason: str = ""


@dataclass
class FollowPlan:
    entries: list[FollowEntry] = field(default_factory=list)

    def tier(self, tier: str) -> list[FollowEntry]:
        return [e for e in self.entries if e.tier == tier]

    def writes(self, include_filter_only: bool = False) -> list[Write]:
        tiers = {TIER_PAIRED, TIER_FILTER_ONLY} if include_filter_only else {TIER_PAIRED}
        return [
            Write(e.trigger_index, e.kind, e.entry_index, e.attribute, e.new_const)
            for e in self.entries
            if e.tier in tiers and e.new_const is not None
        ]

    def filter_only_summary(self) -> list[str]:
        """Tier (b), per effect type: "Create Object: 3 effects create Archer"."""
        groups = Counter((e.entry_name, e.old_const) for e in self.tier(TIER_FILTER_ONLY))
        lines = []
        for (entry_name, old_const), count in sorted(groups.items()):
            noun = "effect" if count == 1 else "effects"
            label = entry_name.replace("_", " ").title()
            lines.append(f"{label}: {count} {noun} name {object_catalog.display_name(old_const)}")
        return lines


def _object_type_of(const: int) -> int:
    """ObjectType for a const, approximately: BUILDING 2 for a .dat building,
    CIVILIAN 3 for a creatable villager-class unit, MILITARY 4 for any other
    creatable, OTHER 1 otherwise. Report-only use (tier e)."""
    entry = unit_kind._objects().get(const, {})
    dat_type = entry.get("type")
    if dat_type == 80:
        return 2
    if dat_type == 70:
        return 3 if entry.get("class") == 4 else 4
    return 1


def _class_filter_breaks(entry, old_const: int, new_const: int) -> str:
    """Why the effect's object_group/object_type filter matches the old const but not the new one, or ""."""
    objects = unit_kind._objects()
    group = _read(entry, "object_group")
    old_class, new_class = objects.get(old_const, {}).get("class"), objects.get(new_const, {}).get("class")
    if isinstance(group, int) and group >= 0 and old_class == group != new_class:
        return "its object class filter no longer matches"
    kind = _read(entry, "object_type")
    if isinstance(kind, int) and kind >= 1 and _object_type_of(old_const) == kind != _object_type_of(new_const):
        return "its object type filter no longer matches"
    return ""


def plan_follow_replace(loaded, replaced: Mapping[int, tuple[int, int]]) -> FollowPlan:
    """Sorts every entry naming a replaced unit's old const in a type field into
    tiers (a)-(e), by field role and never by field presence: only an effect's
    selected_object_ids together with the same effect's object_list_unit_id is a
    pair. `replaced` maps reference_id -> (old_const, new_const), built from the
    object Replace batch after its pre-flight, so unit consts are still old."""
    manager = scenario_io.parse_triggers(loaded)
    vocabulary = _vocabulary(loaded) if manager is not None else None
    plan = FollowPlan()
    if vocabulary is None or not replaced:
        return plan
    old_consts = {old for old, _new in replaced.values()}
    new_for_old: dict[int, set[int]] = {}
    for old, new in replaced.values():
        new_for_old.setdefault(old, set()).add(new)
    const_of = {int(u.reference_id): int(u.unit_const) for _p, u in unit_references.all_units(loaded)}
    for t_index, trigger in enumerate(manager.triggers):
        name = trigger.name or ""
        refs_by_entry: dict[tuple[str, int], dict[str, tuple[int, ...]]] = {}
        for f in _walk(vocabulary, trigger):
            if f.spec.presentation not in TYPE_PRESENTATIONS:
                continue
            value = _read(f.entry, f.spec.attribute)
            if not isinstance(value, int) or value not in old_consts:
                continue
            key = (f.kind, f.index)
            if key not in refs_by_entry:
                refs_by_entry[key] = unit_references.references_in(f.entry, f.definition, f.presentation_map)
            refs = refs_by_entry[key]

            def add(tier, new_const=None, reason="", _f=f, _v=value, _t=t_index, _n=name):
                plan.entries.append(
                    FollowEntry(tier, _t, _n, _f.kind, _f.index, _f.definition.name, _f.spec.attribute, _v,
                                new_const, reason)
                )

            if f.kind != "effect" or f.spec.attribute != "object_list_unit_id":
                add(TIER_REPORT, reason="this field does not filter the replaced unit")
                continue
            selected = refs.get("selected_object_ids", ())
            hit_refs = [r for r in selected if r in replaced and replaced[r][0] == value]
            if not selected:
                news = new_for_old.get(value, set())
                if len(news) != 1:
                    add(TIER_MIXED, reason="its type was replaced by more than one new type")
                    continue
                new_const = next(iter(news))
                broken = _class_filter_breaks(f.entry, value, new_const)
                add(TIER_CLASS if broken else TIER_FILTER_ONLY, new_const, broken)
                continue
            if not hit_refs:
                add(TIER_REPORT, reason="it does not select a replaced unit")
                continue
            unreplaced_same = [r for r in selected if r not in replaced and const_of.get(r) == value]
            pairs = {replaced[r] for r in hit_refs}
            if unreplaced_same or len(pairs) > 1:
                add(TIER_MIXED, reason="it selects replaced and unreplaced units of this type")
                continue
            new_const = next(iter(pairs))[1]
            broken = _class_filter_breaks(f.entry, value, new_const)
            add(TIER_CLASS if broken else TIER_PAIRED, new_const, broken)
    return plan


# -- CSV ------------------------------------------------------------------------

CSV_COLUMNS = ("trigger_index", "trigger_name", "enabled", "kind", "entry_index", "entry_name", "attribute",
               "field_class", "value")


def hits_to_csv(hits: Iterable[TriggerHit], path: Path | str) -> int:
    """find_objects.rows_to_csv()'s trigger counterpart, with the same compatdata refusal."""
    path = refuse_forbidden_path(path)
    hits = list(hits)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for h in hits:
            writer.writerow([h.trigger_index, h.trigger_name, int(h.enabled), h.kind, h.entry_index, h.entry_name,
                             h.attribute, h.field_class, h.value_display])
    return len(hits)
