"""The trigger clipboard (GH #27): copy a block of triggers, paste it back.

Qt-free, like region_clipboard.py: a frozen payload plus functions over a
TriggerManager, importable with no QApplication.

Copy + Paste moves a *linked* block: a (de)activate-trigger effect or a
trigger-active condition pointing inside the block points at the pasted copy
afterwards. A reference pointing outside the block keeps pointing at its
original target, found by object identity rather than by id, so it survives an
intervening delete or reorder. The panel's Duplicate button is the other verb:
it makes independent duplicates whose references still point at the originals.

Where identity does not survive, a reference lands on -1, the same outcome as
pasting into another document. Two cases:

- the target was deleted since the copy;
- a *content* edit on the target was undone since the copy. That undo swaps in
  a deepcopy of the trigger (trigger_model.py's restore(), Trap 5), so the
  recorded object is gone. A structural undo/redo hands back the same objects
  and is unaffected.

A copied block pastes with COPY_SUFFIX through trigger_organize.copied_name()
(GH #134): inside a divider's title (`--- X ---` becomes `--- X (copy) ---`),
so a pasted section header still heads a section of its own, and a bare
`------` verbatim.

Cut + Paste (GH #137) is a move. A cut block pastes without COPY_SUFFIX, and
the links other triggers held into it, which remove_triggers() reset to -1 in
place, are re-pointed at the pasted copies on a same-document paste, each only
while it still reads -1. That one guard covers an undone cut, a second paste of
the same block and a link set again by hand.

Conditions and effects have a clipboard of their own (GH #137's EntryBlock), a
second slot so copying an effect never drops a copied trigger block. A paste
goes into whatever trigger is current, links resolved by identity and the
cross-document policy below applied the same way.

Display order is the caller's job: import_triggers() flattens it, so the
caller captures it first and repairs it through the setter afterwards (see
trigger_model.display_order_with_block_inserted()).

Pasting into another document (GH #3): copy in A, open B, paste. The caller
gates this on an equal scenario_version. A block remembers the document it came
from, and a paste whose doc_id differs applies a policy per field class, keyed
on the version vocabulary's presentation rather than on attribute names:

- trigger links (TriggerId): relinked inside the block, -1 outside it, as above;
- placed units (Unit, Unit[]): cleared to the type's default, as the in-game
  editor does, since reference_ids name A's units, not B's;
- variables (VariableId): the id is kept, and a slot B has not named takes A's
  name in the same undo record. A slot B already named differently keeps B's;
- players, areas, strings, XS names, timers, AI signals: verbatim.

Documents are told apart by the window's _doc_id, which every load
regenerates. So Save, reopen the same file and paste clears unit references
even though the units exist. Accepted.

The restamp step exists because the manager restamps each pasted condition and
effect but not the trigger's own inner lists, which keep the source's _uuid. A
later "new effect" on a pasted trigger would then stamp A's uuid, and commit
into A's sections or raise once A is gone.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from AoE2ScenarioParser.objects.managers.trigger_manager import get_trigger_referencing_ce

from . import library_compat, trigger_fields, trigger_organize

COPY_SUFFIX = trigger_organize.COPY_SUFFIX

# GH #3's reference classes, as the version vocabulary presents them.
UNIT_PRESENTATIONS = frozenset({"Unit", "Unit[]"})
VARIABLE_PRESENTATIONS = frozenset({"VariableId"})


@dataclass(frozen=True)
class TriggerBlock:
    """A copied block of triggers.

    `triggers` are deepcopies, in source display order, so later edits to the
    sources never reach the clipboard. `external_targets` holds
    ((block_pos, ce_ordinal, live source Trigger), ...) for each reference
    whose target sat outside the block, ce_ordinal counting
    get_trigger_referencing_ce()'s own order.
    """

    triggers: tuple
    external_targets: tuple
    label: str
    # The window's _doc_id at copy time, and the source's scenario version.
    source_doc_id: str = ""
    scenario_version: str = ""
    # ((variable_id, name), ...) for every named variable the block references.
    variable_names: tuple = ()
    # GH #137: a cut block pastes without COPY_SUFFIX, and `inbound` holds
    # ((live referencing Trigger, live ce, block_pos), ...) for every link
    # from outside the block into it, which remove_triggers() reset to -1.
    from_cut: bool = False
    inbound: tuple = ()


@dataclass(frozen=True)
class PasteResult:
    """What paste_into() did. `triggers` are the stored pasted triggers, in
    block order; the counts are for the status line."""

    triggers: list
    cross_document: bool = False
    cleared_trigger_links: int = 0
    cleared_unit_refs: int = 0
    named_variables: int = 0
    variable_name_conflicts: int = 0
    # A cut block's inbound links: re-pointed at the paste, or left alone.
    relinked_inbound: int = 0
    skipped_inbound: int = 0


def _vocabulary(scenario_version: str):
    if scenario_version and library_compat.vocabulary_is_available(scenario_version):
        return library_compat.load_vocabulary(scenario_version)
    return None


def _read(entry: Any, attribute: str):
    try:
        return getattr(entry, attribute, None)
    except Exception:  # noqa: BLE001 -- the library's version-gated properties raise
        return None


def reference_fields(vocabulary: Any, conditions: Sequence, effects: Sequence, presentations: frozenset):
    """(entry, attribute, default) for every displayed field of these
    conditions and effects whose presentation is in `presentations`: a whole
    trigger's lists, or an EntryBlock's (GH #137). A type the vocabulary does
    not know is skipped.

    The default is the library's own for a new entry: type 0's defaults
    overlaid with the type's, since a type's JSON lists only its departures
    (Task Object has no selected_object_ids default of its own)."""
    for field in entry_fields(vocabulary, conditions, effects):
        if field.spec.presentation in presentations:
            yield field.entry, field.spec.attribute, field.default


@dataclass(frozen=True)
class EntryField:
    """One displayed field of one condition or effect, as entry_fields() walks them."""

    kind: str  # "condition" or "effect"
    index: int  # the entry's position in its trigger's list
    entry: Any
    definition: Any  # its VocabularyEntry
    presentation_map: Any  # the kind's attribute -> presentation map
    spec: trigger_fields.FieldSpec
    default: Any


def entry_fields(vocabulary: Any, conditions: Sequence, effects: Sequence):
    """Every displayed field of these conditions and effects, with where it sits.
    The one walker: reference_fields() and find_triggers both filter it. A type
    the vocabulary does not know is skipped. The default is type 0's overlaid
    with the type's, as reference_fields() documents."""
    groups = (
        ("condition", conditions, vocabulary.conditions, vocabulary.condition_presentation, "condition_type"),
        ("effect", effects, vocabulary.effects, vocabulary.effect_presentation, "effect_type"),
    )
    for kind, entries, definitions, presentation_map, type_attribute in groups:
        base = definitions.get(0)
        base_defaults = base.default_attributes if base is not None else {}
        for index, entry in enumerate(entries):
            definition = definitions.get(getattr(entry, type_attribute, None))
            if definition is None:
                continue
            defaults = {**base_defaults, **definition.default_attributes}
            for spec in trigger_fields.field_specs(definition, presentation_map, type_attribute):
                yield EntryField(kind, index, entry, definition, presentation_map, spec, defaults.get(spec.name))


def _variable_names(manager: Any, groups: Sequence, scenario_version: str) -> tuple:
    """`groups` is ((conditions, effects), ...)."""
    vocabulary = _vocabulary(scenario_version)
    if vocabulary is None:
        return ()
    referenced = set()
    for conditions, effects in groups:
        for entry, attribute, _default in reference_fields(vocabulary, conditions, effects, VARIABLE_PRESENTATIONS):
            value = _read(entry, attribute)
            if isinstance(value, int):
                referenced.add(int(value))
    return tuple(
        sorted((v.variable_id, v.name) for v in manager.variables if v.variable_id in referenced and v.name)
    )


def _cross_document_fixups(manager: Any, groups: Sequence, scenario_version: str, variable_names: tuple):
    """GH #3's policy over pasted (conditions, effects) groups: placed-unit
    references cleared, variable names carried. (cleared, named, conflicts)."""
    cleared_units = 0
    vocabulary = _vocabulary(scenario_version)
    for conditions, effects in groups if vocabulary is not None else ():
        for entry, attribute, default in reference_fields(vocabulary, conditions, effects, UNIT_PRESENTATIONS):
            if default is not None and _read(entry, attribute) != default:
                setattr(entry, attribute, copy.deepcopy(default))
                cleared_units += 1
    named = conflicts = 0
    existing = {v.variable_id: v for v in manager.variables}
    for variable_id, name in variable_names:
        slot = existing.get(variable_id)
        if slot is None:
            manager.add_variable(name, variable_id=variable_id)
            named += 1
        elif not slot.name:
            slot.name = name
            named += 1
        elif slot.name != name:
            conflicts += 1
    return cleared_units, named, conflicts


def copy_block(
    manager: Any, indices: Sequence[int], *, doc_id: str = "", scenario_version: str = "", cut: bool = False
) -> TriggerBlock:
    """A TriggerBlock of the triggers at `indices` (list indices), taken in
    display order whatever order `indices` arrives in. `doc_id` and
    `scenario_version` describe the source document for a later paste.

    `cut=True` is GH #137's Cut, taken just before the triggers are removed:
    the block also records every link into it from a trigger outside it, so a
    same-document paste can re-point what the removal reset to -1."""
    wanted = set(indices)
    live = list(manager.triggers)
    ordered = [index for index in manager.trigger_display_order if index in wanted]
    external = []
    for pos, index in enumerate(ordered):
        for ordinal, ce in enumerate(get_trigger_referencing_ce(live[index])):
            target = ce.trigger_id
            if target in wanted or not 0 <= target < len(live):
                continue
            external.append((pos, ordinal, live[target]))
    block_pos = {index: pos for pos, index in enumerate(ordered)}
    inbound = [
        (trigger, ce, block_pos[ce.trigger_id])
        for index, trigger in enumerate(live if cut else ())
        if index not in wanted
        for ce in get_trigger_referencing_ce(trigger)
        if ce.trigger_id in block_pos
    ]
    triggers = tuple(copy.deepcopy(live[index]) for index in ordered)
    label = f"{len(triggers)} trigger{'s' if len(triggers) != 1 else ''}"
    return TriggerBlock(
        triggers=triggers,
        external_targets=tuple(external),
        label=label,
        source_doc_id=doc_id,
        scenario_version=scenario_version,
        variable_names=_variable_names(manager, [(t.conditions, t.effects) for t in triggers], scenario_version),
        from_cut=cut,
        inbound=tuple(inbound),
    )


def _relink_inbound(manager: Any, block: TriggerBlock, before_len: int) -> tuple[int, int]:
    """Re-point a cut block's inbound links at the pasted copies, each only
    while it still reads -1: an undo of the cut, an earlier paste of the same
    block or a hand edit has set it since, and wins. (relinked, skipped)."""
    present = {id(trigger) for trigger in manager.triggers}
    relinked = skipped = 0
    for referrer, ce, pos in block.inbound:
        if (
            id(referrer) in present
            and any(c is ce for c in get_trigger_referencing_ce(referrer))
            and ce.trigger_id == -1
        ):
            ce.trigger_id = before_len + pos
            relinked += 1
        else:
            skipped += 1
    return relinked, skipped


def paste_into(manager: Any, block: TriggerBlock, *, doc_id: str = "") -> PasteResult:
    """Append a copy of `block` to `manager`. The new triggers, in block order,
    occupy list indices len-before .. len-after - 1: the append renumbers
    nothing.

    One import_triggers(index=-1) call. Any other index routes through
    move_triggers() and renumbers every id. A `doc_id` other than the block's
    source applies the cross-document policy in the module docstring.
    """
    before_len = len(manager.triggers)
    with warnings.catch_warnings():
        # Fires once per outside reference, exactly the case the relink below
        # repairs.
        warnings.simplefilter("ignore")
        manager.import_triggers(list(block.triggers), index=-1)
    # Re-resolved, never the return value: into an empty destination the
    # returned list is not what manager.triggers holds.
    pasted = list(manager.triggers[before_len:])
    for trigger in pasted:
        # The inner lists are deepcopied with the source's _uuid; the manager
        # restamps the entries but not the lists themselves.
        trigger._effects._uuid = manager._uuid
        trigger._conditions._uuid = manager._uuid

    position = {id(trigger): index for index, trigger in enumerate(manager.triggers)}
    cleared_links = 0
    for pos, ordinal, source in block.external_targets:
        index = position.get(id(source))
        if index is None or index >= before_len:
            cleared_links += 1
            continue
        get_trigger_referencing_ce(pasted[pos])[ordinal].trigger_id = index

    # A cut block reads as a move: no suffix. A copied divider takes it inside its title (GH #134).
    for trigger in pasted if not block.from_cut else ():
        trigger.name = trigger_organize.copied_name(trigger.name or "")

    if block.source_doc_id == doc_id:
        relinked, skipped = _relink_inbound(manager, block, before_len)
        return PasteResult(
            pasted, cleared_trigger_links=cleared_links, relinked_inbound=relinked, skipped_inbound=skipped
        )
    cleared_units, named, conflicts = _cross_document_fixups(
        manager, [(t.conditions, t.effects) for t in pasted], block.scenario_version, block.variable_names
    )
    return PasteResult(pasted, True, cleared_links, cleared_units, named, conflicts)


# -- GH #137: the entry clipboard -----------------------------------------------


@dataclass(frozen=True)
class EntryBlock:
    """Copied conditions and effects, a second slot beside the trigger one.

    `conditions` and `effects` are deepcopies in list order. `trigger_targets`
    holds ((kind, pos, live target Trigger), ...) for every trigger-referencing
    entry whose target was a real trigger, `pos` counting within its kind."""

    conditions: tuple
    effects: tuple
    trigger_targets: tuple
    label: str
    source_doc_id: str = ""
    scenario_version: str = ""
    variable_names: tuple = ()


@dataclass(frozen=True)
class EntryPasteResult:
    """What paste_entries() did. `refs` are the pasted (kind, index) rows."""

    refs: list
    cross_document: bool = False
    cleared_trigger_links: int = 0
    cleared_unit_refs: int = 0
    named_variables: int = 0
    variable_name_conflicts: int = 0


def entries_label(conditions: int, effects: int) -> str:
    parts = [_count(n, noun) for n, noun in ((conditions, "condition"), (effects, "effect")) if n]
    return " and ".join(parts) or "nothing"


def copy_entries(
    manager: Any, trigger_index: int, refs: Sequence, *, doc_id: str = "", scenario_version: str = ""
) -> EntryBlock:
    """An EntryBlock of trigger `trigger_index`'s entries at `refs`, ((kind,
    index), ...), each kind taken in list order."""
    live = list(manager.triggers)
    trigger = live[trigger_index]
    referencing = {id(ce) for ce in get_trigger_referencing_ce(trigger)}
    picked = {}
    targets = []
    for kind, entries in (("condition", trigger.conditions), ("effect", trigger.effects)):
        wanted = sorted({index for ref_kind, index in refs if ref_kind == kind and 0 <= index < len(entries)})
        picked[kind] = [entries[index] for index in wanted]
        for pos, entry in enumerate(picked[kind]):
            target = entry.trigger_id if id(entry) in referencing else -1
            if 0 <= target < len(live):
                targets.append((kind, pos, live[target]))
    conditions = tuple(copy.deepcopy(entry) for entry in picked["condition"])
    effects = tuple(copy.deepcopy(entry) for entry in picked["effect"])
    return EntryBlock(
        conditions=conditions,
        effects=effects,
        trigger_targets=tuple(targets),
        label=entries_label(len(conditions), len(effects)),
        source_doc_id=doc_id,
        scenario_version=scenario_version,
        variable_names=_variable_names(manager, [(conditions, effects)], scenario_version),
    )


def paste_entries(
    manager: Any, trigger_index: int, block: EntryBlock, anchor: tuple | None = None, *, doc_id: str = ""
) -> EntryPasteResult:
    """Insert a copy of `block` into trigger `trigger_index`: conditions after
    `anchor` when it is a ("condition", index) ref, else at the end of the
    conditions; effects likewise. UuidList.insert() stamps each entry with
    this manager. Trigger links resolve by identity, landing on -1 for a
    target deleted since the copy or in another document."""
    trigger = manager.triggers[trigger_index]
    position = {id(t): index for index, t in enumerate(manager.triggers)}
    cross = block.source_doc_id != doc_id
    pasted: dict = {"condition": [], "effect": []}
    refs = []
    for kind, source, entries in (
        ("condition", block.conditions, trigger.conditions),
        ("effect", block.effects, trigger.effects),
    ):
        at = anchor[1] + 1 if anchor is not None and anchor[0] == kind else len(entries)
        for offset, entry in enumerate(source):
            entries.insert(at + offset, copy.deepcopy(entry))
            pasted[kind].append(entries[at + offset])
            refs.append((kind, at + offset))
    cleared_links = 0
    for kind, pos, target in block.trigger_targets:
        index = None if cross else position.get(id(target))
        pasted[kind][pos].trigger_id = -1 if index is None else index
        cleared_links += index is None
    if not cross:
        return EntryPasteResult(refs, cleared_trigger_links=cleared_links)
    cleared_units, named, conflicts = _cross_document_fixups(
        manager, [(pasted["condition"], pasted["effect"])], block.scenario_version, block.variable_names
    )
    return EntryPasteResult(refs, True, cleared_links, cleared_units, named, conflicts)


def entry_paste_report(block: EntryBlock, result: EntryPasteResult) -> str:
    """paste_report()'s twin for an entry paste from another scenario."""
    head = f"Pasted {block.label} from another scenario"
    parts = [
        f"cleared {_count(n, noun)}"
        for n, noun in (
            (result.cleared_trigger_links, "trigger link"),
            (result.cleared_unit_refs, "unit reference"),
        )
        if n
    ]
    if result.named_variables:
        parts.append(f"named {_count(result.named_variables, 'variable')}")
    if result.variable_name_conflicts:
        parts.append(f"kept this scenario's name for {_count(result.variable_name_conflicts, 'variable')}")
    return f"{head}: {', '.join(parts)}." if parts else f"{head}; nothing needed clearing."


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def paste_report(result: PasteResult) -> str:
    """One status line for a cross-document paste: what the user has to
    re-set by hand."""
    head = f"Pasted {_count(len(result.triggers), 'trigger')} from another scenario"
    parts = []
    if result.cleared_trigger_links or result.cleared_unit_refs:
        cleared = [
            _count(n, noun)
            for n, noun in (
                (result.cleared_trigger_links, "trigger link"),
                (result.cleared_unit_refs, "unit reference"),
            )
            if n
        ]
        parts.append("cleared " + " and ".join(cleared))
    if result.named_variables:
        parts.append(f"named {_count(result.named_variables, 'variable')}")
    if result.variable_name_conflicts:
        n = result.variable_name_conflicts
        parts.append(f"kept this scenario's name for {_count(n, 'variable')}")
    if not parts:
        return f"{head}; nothing needed clearing."
    return f"{head}: {', '.join(parts)}."


def _cleared_since_copy(n: int) -> list[str]:
    # An undo swaps in a deep copy of the target (restore()'s Trap 5), which identity no longer finds.
    return [f"cleared {_count(n, 'trigger link')} (target changed or removed since the copy)"] if n else []


def relink_report(result: PasteResult) -> str:
    """The same-document twin of paste_report(): a cut block's inbound links
    and any outbound link left at -1, or "" when there were none."""
    head = f"Pasted {_count(len(result.triggers), 'trigger')}"
    parts = []
    if result.relinked_inbound:
        parts.append(f"reconnected {_count(result.relinked_inbound, 'trigger link')} into them")
    if result.skipped_inbound:
        n = result.skipped_inbound
        parts.append(f"left {_count(n, 'trigger link')} alone (changed or removed since the cut)")
    parts += _cleared_since_copy(result.cleared_trigger_links)
    return f"{head}: {', '.join(parts)}." if parts else ""


def entry_relink_report(block: EntryBlock, result: EntryPasteResult) -> str:
    """relink_report()'s twin for a same-document entry paste."""
    parts = _cleared_since_copy(result.cleared_trigger_links)
    return f"Pasted {block.label}: {', '.join(parts)}." if parts else ""
