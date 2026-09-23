"""The trigger clipboard (GH #27): copy a block of triggers, paste it back.

Qt-free, like region_clipboard.py: a frozen payload plus functions over a
TriggerManager, importable with no QApplication.

Copy + Paste moves a *linked* block: a (de)activate-trigger effect or a
trigger-active condition pointing inside the block points at the pasted copy
afterwards. A reference pointing outside the block keeps pointing at its
original target, found by object identity rather than by id, so it survives an
intervening delete or reorder. The panel's Copy button is the other verb: it
makes independent duplicates whose references still point at the originals.

Where identity does not survive, a reference lands on -1, the same outcome as
pasting into another document. Two cases:

- the target was deleted since the copy;
- a *content* edit on the target was undone since the copy. That undo swaps in
  a deepcopy of the trigger (trigger_model.py's restore(), Trap 5), so the
  recorded object is gone. A structural undo/redo hands back the same objects
  and is unaffected.

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

from . import library_compat, trigger_fields

COPY_SUFFIX = " (copy)"

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


def _vocabulary(scenario_version: str):
    if scenario_version and library_compat.vocabulary_is_available(scenario_version):
        return library_compat.load_vocabulary(scenario_version)
    return None


def _read(entry: Any, attribute: str):
    try:
        return getattr(entry, attribute, None)
    except Exception:  # noqa: BLE001 -- the library's version-gated properties raise
        return None


def reference_fields(vocabulary: Any, trigger: Any, presentations: frozenset):
    """(entry, attribute, default) for every displayed condition/effect field
    of `trigger` whose presentation is in `presentations`. A type the
    vocabulary does not know is skipped.

    The default is the library's own for a new entry: type 0's defaults
    overlaid with the type's, since a type's JSON lists only its departures
    (Task Object has no selected_object_ids default of its own)."""
    groups = (
        (trigger.conditions, vocabulary.conditions, vocabulary.condition_presentation, "condition_type"),
        (trigger.effects, vocabulary.effects, vocabulary.effect_presentation, "effect_type"),
    )
    for entries, definitions, presentation_map, type_attribute in groups:
        base = definitions.get(0)
        base_defaults = base.default_attributes if base is not None else {}
        for entry in entries:
            definition = definitions.get(getattr(entry, type_attribute, None))
            if definition is None:
                continue
            defaults = {**base_defaults, **definition.default_attributes}
            for spec in trigger_fields.field_specs(definition, presentation_map, type_attribute):
                if spec.presentation in presentations:
                    yield entry, spec.attribute, defaults.get(spec.name)


def _variable_names(manager: Any, triggers: Sequence, scenario_version: str) -> tuple:
    vocabulary = _vocabulary(scenario_version)
    if vocabulary is None:
        return ()
    referenced = set()
    for trigger in triggers:
        for entry, attribute, _default in reference_fields(vocabulary, trigger, VARIABLE_PRESENTATIONS):
            value = _read(entry, attribute)
            if isinstance(value, int):
                referenced.add(int(value))
    return tuple(
        sorted((v.variable_id, v.name) for v in manager.variables if v.variable_id in referenced and v.name)
    )


def copy_block(
    manager: Any, indices: Sequence[int], *, doc_id: str = "", scenario_version: str = ""
) -> TriggerBlock:
    """A TriggerBlock of the triggers at `indices` (list indices), taken in
    display order whatever order `indices` arrives in. `doc_id` and
    `scenario_version` describe the source document for a later paste."""
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
    triggers = tuple(copy.deepcopy(live[index]) for index in ordered)
    label = f"{len(triggers)} trigger{'s' if len(triggers) != 1 else ''}"
    return TriggerBlock(
        triggers=triggers,
        external_targets=tuple(external),
        label=label,
        source_doc_id=doc_id,
        scenario_version=scenario_version,
        variable_names=_variable_names(manager, triggers, scenario_version),
    )


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

    for trigger in pasted:
        trigger.name = (trigger.name or "") + COPY_SUFFIX

    if block.source_doc_id == doc_id:
        return PasteResult(pasted, cleared_trigger_links=cleared_links)
    cleared_units = 0
    vocabulary = _vocabulary(block.scenario_version)
    for trigger in pasted if vocabulary is not None else ():
        for entry, attribute, default in reference_fields(vocabulary, trigger, UNIT_PRESENTATIONS):
            if default is not None and _read(entry, attribute) != default:
                setattr(entry, attribute, copy.deepcopy(default))
                cleared_units += 1
    named = conflicts = 0
    existing = {v.variable_id: v for v in manager.variables}
    for variable_id, name in block.variable_names:
        slot = existing.get(variable_id)
        if slot is None:
            manager.add_variable(name, variable_id=variable_id)
            named += 1
        elif not slot.name:
            slot.name = name
            named += 1
        elif slot.name != name:
            conflicts += 1
    return PasteResult(pasted, True, cleared_links, cleared_units, named, conflicts)


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
