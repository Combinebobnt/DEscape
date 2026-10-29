"""Qt-free core of Swap Players (GH #185): players A and B exchange slots, so
the scenario plays as before with the two renumbered.

plan_swap() reads what the swap would write and refuses it whole when any of
it can't be written on this file. apply_swap() makes those writes, inside the
caller's open unit and trigger brackets, through the setters each editor
already uses. Nothing here constructs a model, so the dialog re-plans on every
change, and a tool or batch script runs the same code as the viewer.

What swaps:

- Always: the two unit lists (UnitEditModel.swap_players()), every Players
  mode field (player_fields.specs_for(), less AI type, which Personality
  writes), Personality itself and the disabled-object lists.
- With SwapOptions.diplomacy: the stance grid, as the permutation s swapping
  A and B (new[s(i)][s(j)] = old[i][j] over P1..P8, the diagonal included),
  and allied victory. Indices 8..15 are never touched.
- With SwapOptions.triggers: every PlayerId attribute a condition or effect's
  type lists in its VocabularyEntry.attributes, A to B and B to A. Every other
  value is left alone, and so is an attribute the type doesn't list and every
  entry of a type the vocabulary doesn't know (counted). Text is never touched.

Values are the file's own, overlaid with the options model's pending edits.
Only P1..P8 that the scenario defines can be swapped, never GAIA. What no
editor writes today (player_data_3's own fields, trade goods, ...) stays as
stored.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from descape import (
    ai_scripts,
    diplomacy_fields,
    disables_fields,
    find_triggers,
    library_compat,
    option_fields,
    options_model,
    player_fields,
)
from descape.edit_history import OptionsDiffRecord

PLAYER_ID_PRESENTATION = "PlayerId"
# AI type follows Personality (GH #126), which has its own entry below.
_NOT_SWAPPED_AS_FIELDS = frozenset({"player_type", "personality"})
_NUM_PLAYERS = player_fields.NUM_PLAYERS


@dataclass(frozen=True)
class SwapOptions:
    """The dialog's two checkboxes. Both on by default (user decision, 2026-10-28)."""

    triggers: bool = True
    diplomacy: bool = True


@dataclass(frozen=True)
class SwapGates:
    """What this file accepts writes to: file_gates(), which the viewer closes whole under its data lock."""

    player_fields: frozenset[str] = frozenset()  # bare ids, file_gates()
    diplomacy: bool = False
    disables: bool = False
    units: bool = False
    personality_custom: bool = False  # the Files library verifies, so a custom AI may move


@dataclass(frozen=True)
class SwapRefusal:
    reasons: tuple[str, ...]

    @property
    def reason(self) -> str:
        return "Can't swap: " + "; ".join(self.reasons) + "."


@dataclass
class SwapPlan:
    a: int
    b: int
    options: SwapOptions
    # field_id -> new value, only ids whose value changes.
    option_writes: dict[str, object] = field(default_factory=dict)
    # The two personalities differ, so A and B's current choices cross over.
    personality: bool = False
    trigger_writes: list[find_triggers.Write] = field(default_factory=list)
    unit_counts: tuple[int, int] = (0, 0)
    unknown_entries: int = 0
    settings_differ: int = 0
    diplomacy_cells: int = 0

    @property
    def touched_triggers(self) -> list[int]:
        return sorted({w.trigger_index for w in self.trigger_writes})

    @property
    def label(self) -> str:
        return f"Swap P{self.a} and P{self.b}"

    @property
    def writes_to(self) -> tuple[bool, bool, bool]:
        """Whether apply_swap() needs the options, units and triggers model."""
        return bool(self.option_writes or self.personality), any(self.unit_counts), bool(self.trigger_writes)

    @property
    def is_empty(self) -> bool:
        return not any(self.writes_to)

    def summary(self) -> str:
        """The dialog's preview line and the status line after a swap."""
        if self.is_empty:
            return f"Nothing differs between P{self.a} and P{self.b}."
        n_a, n_b = self.unit_counts
        parts = [f"Moves {n_a} and {n_b} unit{'' if n_a + n_b == 1 else 's'}"]
        if self.options.triggers:
            n, m = len(self.trigger_writes), len(self.touched_triggers)
            parts.append(f"{n} trigger field{'' if n == 1 else 's'} in {m} trigger{'' if m == 1 else 's'}")
        s = self.settings_differ
        parts.append(f"{s} setting{' differs' if s == 1 else 's differ'}")
        if self.options.diplomacy:
            d = self.diplomacy_cells
            parts.append(f"{d} diplomacy value{' changes' if d == 1 else 's change'}")
        text = "; ".join(parts) + "."
        if self.unknown_entries:
            k = self.unknown_entries
            text += f" {k} trigger entr{'y' if k == 1 else 'ies'} of an unknown type left as stored."
        return text


def file_gates(loaded, *, personality_custom: bool | None = None) -> SwapGates:
    """The panels' write gates read straight off the file. The viewer builds
    its own from this plus its data lock, passing `personality_custom` from
    its per-document cache, since computing it forces the trigger parse."""
    carrier = options_model.options_write_supported(loaded, option_fields.specs_for(loaded))
    fields: frozenset[str] = frozenset()
    if carrier and options_model.players_write_supported(loaded):
        fields = frozenset(player_fields.parse_player_field_id(key)[0] for key in player_fields.write_targets(loaded))
        if options_model.personality_write_supported(loaded):
            fields |= {"personality"}
    if "personality" not in fields:
        personality_custom = False
    elif personality_custom is None:
        personality_custom = options_model.personality_custom_supported(loaded)
    return SwapGates(
        player_fields=fields,
        diplomacy=carrier and options_model.diplomacy_write_supported(loaded),
        disables=carrier and options_model.disables_write_supported(loaded),
        units=bool(loaded.units_write_supported) and loaded.number_of_unit_sections == 9,
        personality_custom=personality_custom,
    )


def swap_partners(loaded, a: int, pending: Mapping[str, object] | None = None) -> list[int]:
    """The players A can swap with: the defined players (the count as
    edited), never GAIA and never A itself."""
    return [p for p in player_fields.defined_player_ids(loaded, pending) if p != a]


def _personality_view(loaded, pid: int, pending, personalities) -> tuple[str, int] | None:
    """(stored name, ai_type), the pair set_personality() treats as one
    choice, or None on a file with no personality rows."""
    if personalities is not None and pid in personalities:
        choice = personalities[pid]
        return choice.stored_name, choice.ai_type
    if player_fields.personality_field_id(pid) in pending:
        raise ValueError(f"P{pid} has a pending personality: pass its choice in `personalities`")
    specs = {s.field_id: s for s in player_fields.specs_for(loaded)}
    if "personality" not in specs or "player_type" not in specs:
        return None
    index = player_fields.PlayerArrayLayout.P1_TO_P8_THEN_GAIA.index_for(pid)
    name = loaded._scenario.sections["PlayerDataTwo"].retriever_map["ai_names"].data[index]
    return name, player_fields.current_value(loaded, specs["player_type"], pid)


def _diplomacy_value(loaded, cell_id: str, pending) -> int:
    if cell_id in pending:
        return pending[cell_id]
    parsed = diplomacy_fields.parse_cell_id(cell_id)
    if parsed[0] == "stance":
        return diplomacy_fields.stance_value(loaded, parsed[1], parsed[2])
    return diplomacy_fields.allied_victory_value(loaded, parsed[1])


_NAMED_READ_ONLY = 3  # names listed before "and N more", so a whole read-only tab stays one line


def _read_only_reason(labels: Sequence[str]) -> str:
    shown = ", ".join(labels[:_NAMED_READ_ONLY])
    if len(labels) > _NAMED_READ_ONLY:
        shown += f" and {len(labels) - _NAMED_READ_ONLY} more"
    verb = "differs" if len(labels) == 1 else "differ"
    return f"{shown} {verb} between the two players and can't be written on this file"


def _sigma(a: int, b: int, p: int) -> int:
    return b if p == a else a if p == b else p


def _read(entry, attribute: str):
    try:
        return getattr(entry, attribute, None)
    except Exception:  # noqa: BLE001 -- the library's version-gated properties raise
        return None


def _vocabulary(loaded):
    version = loaded.scenario_version
    if version and library_compat.vocabulary_is_available(version):
        return library_compat.load_vocabulary(version)
    return None


def trigger_writes(loaded, manager, a: int, b: int) -> tuple[list[find_triggers.Write], int]:
    """(writes, entries of an unknown type) for the trigger half: every
    PlayerId attribute an entry's type lists, A to B and B to A."""
    vocabulary = _vocabulary(loaded)
    writes: list[find_triggers.Write] = []
    unknown = 0
    for t_index, trigger in enumerate(manager.triggers):
        groups = (
            ("condition", trigger.conditions, "condition_type"),
            ("effect", trigger.effects, "effect_type"),
        )
        for kind, entries, type_attribute in groups:
            if vocabulary is None:
                unknown += len(entries)
                continue
            if kind == "condition":
                definitions, presentation = vocabulary.conditions, vocabulary.condition_presentation
            else:
                definitions, presentation = vocabulary.effects, vocabulary.effect_presentation
            for e_index, entry in enumerate(entries):
                definition = definitions.get(getattr(entry, type_attribute, None))
                if definition is None:
                    unknown += 1
                    continue
                for attribute in definition.attributes:
                    if presentation.get(attribute) != PLAYER_ID_PRESENTATION:
                        continue
                    value = _read(entry, attribute)
                    if isinstance(value, int) and not isinstance(value, bool) and value in (a, b):
                        writes.append(find_triggers.Write(t_index, kind, e_index, attribute, _sigma(a, b, value)))
    return writes, unknown


def plan_swap(
    loaded,
    a: int,
    b: int,
    opts: SwapOptions,
    pending: Mapping[str, object],
    editable: SwapGates,
    trigger_manager=None,
    personalities: Mapping[int, ai_scripts.AiChoice] | None = None,
) -> SwapPlan | SwapRefusal:
    """What swapping A and B would write, or why the file refuses it.

    `pending` is OptionsEditModel.pending_values() (empty without a model) and
    `personalities` that model's current_personality() for A and B, needed
    when either has a pending one. `trigger_manager` is the memoized
    parse_triggers() result, or None when the triggers box is unchecked; a
    checked box with no manager refuses. Constructs no model and opens nothing.
    """
    if not (1 <= a <= _NUM_PLAYERS and 1 <= b <= _NUM_PLAYERS) or a == b:
        raise ValueError(f"a swap takes two different players in 1..{_NUM_PLAYERS}, got {a} and {b}")
    defined = player_fields.defined_player_ids(loaded, pending)
    if a not in defined or b not in defined:
        return SwapRefusal((f"only defined players can be swapped, and this scenario has {len(defined)}",))
    plan = SwapPlan(a, b, opts)
    refusals: list[str] = []

    units = loaded.unit_manager.units
    plan.unit_counts = (len(units[a]), len(units[b]))
    if any(plan.unit_counts) and not editable.units:
        refusals.append("this file's units can't be written")

    read_only: list[str] = []
    for spec in player_fields.specs_for(loaded):
        if spec.field_id in _NOT_SWAPPED_AS_FIELDS:
            continue
        id_a = player_fields.player_field_id(spec.field_id, a)
        id_b = player_fields.player_field_id(spec.field_id, b)
        value_a = pending.get(id_a, player_fields.current_value(loaded, spec, a))
        value_b = pending.get(id_b, player_fields.current_value(loaded, spec, b))
        if value_a == value_b:
            continue
        plan.settings_differ += 1
        if spec.field_id not in editable.player_fields:
            read_only.append(spec.label)
            continue
        plan.option_writes[id_a] = value_b
        plan.option_writes[id_b] = value_a

    view_a = _personality_view(loaded, a, pending, personalities)
    view_b = _personality_view(loaded, b, pending, personalities)
    if view_a != view_b:
        plan.settings_differ += 1
        custom = ai_scripts.AI_TYPE_CUSTOM in (view_a[1], view_b[1])
        if "personality" not in editable.player_fields:
            read_only.append("Personality")
        elif custom and not editable.personality_custom:
            refusals.append("a custom AI can't be moved on this file: its AI script library could not be verified")
        else:
            plan.personality = True
    for category in disables_fields.CATEGORIES:
        id_a = disables_fields.disables_field_id(category, a)
        id_b = disables_fields.disables_field_id(category, b)
        ids_a = tuple(pending.get(id_a, disables_fields.current_ids(loaded, category, a)))
        ids_b = tuple(pending.get(id_b, disables_fields.current_ids(loaded, category, b)))
        if ids_a == ids_b:
            continue
        plan.settings_differ += 1
        if not editable.disables:
            read_only.append(f"disabled {category}")
            continue
        plan.option_writes[id_a] = ids_b
        plan.option_writes[id_b] = ids_a
    if read_only:
        refusals.append(_read_only_reason(read_only))

    if opts.diplomacy:
        diplomacy: dict[str, int] = {}
        for i in range(1, _NUM_PLAYERS + 1):
            for j in range(1, _NUM_PLAYERS + 1):
                cell = diplomacy_fields.stance_cell_id(i, j)
                source = diplomacy_fields.stance_cell_id(_sigma(a, b, i), _sigma(a, b, j))
                new = _diplomacy_value(loaded, source, pending)
                if new != _diplomacy_value(loaded, cell, pending):
                    diplomacy[cell] = new
            cell = diplomacy_fields.allied_victory_cell_id(i)
            new = _diplomacy_value(loaded, diplomacy_fields.allied_victory_cell_id(_sigma(a, b, i)), pending)
            if new != _diplomacy_value(loaded, cell, pending):
                diplomacy[cell] = new
        plan.diplomacy_cells = len(diplomacy)
        if diplomacy and not editable.diplomacy:
            refusals.append("diplomacy can't be written on this file (untick Swap diplomacy to swap the rest)")
        else:
            plan.option_writes.update(diplomacy)

    if opts.triggers:
        if trigger_manager is None:
            refusals.append("this file's triggers can't be read (untick Swap trigger players to swap the rest)")
        else:
            writes, plan.unknown_entries = trigger_writes(loaded, trigger_manager, a, b)
            if writes and not loaded.trigger_write_supported:
                refusals.append("this file's triggers can't be written (untick Swap trigger players to swap the rest)")
            else:
                plan.trigger_writes = writes

    if refusals:
        return SwapRefusal(tuple(refusals))
    return plan


def apply_swap(plan: SwapPlan, option_edits, unit_edits, trigger_manager, records: list) -> None:
    """The plan's writes, inside the caller's open unit and trigger brackets.
    `option_edits`/`unit_edits` may be None where the plan writes nothing of theirs.

    Each OptionsDiffRecord goes into the caller-owned `records` as soon as its
    write lands, so a raise partway still leaves every applied option edit
    with its record (a dirty model with no history record closes the
    document with no save prompt). The personality pair goes through
    set_personality(), recorded as choice keys."""
    label = plan.label

    def record(field_id: str, before) -> None:
        after = option_edits.current_value(field_id)
        if after != before:
            records.append(OptionsDiffRecord(label, field_id, before, after))

    for field_id, value in plan.option_writes.items():
        assert option_edits.can_set(field_id), f"the preflight passed {field_id!r}, which this file can't write"
        before = option_edits.current_value(field_id)
        option_edits.set_value(field_id, value)
        record(field_id, before)
    if plan.personality:
        choice_a = option_edits.current_personality(plan.a)
        choice_b = option_edits.current_personality(plan.b)
        for pid, choice in ((plan.a, choice_b), (plan.b, choice_a)):
            field_id = player_fields.personality_field_id(pid)
            assert option_edits.can_set(field_id), f"the preflight passed {field_id!r}, which this file can't write"
            before = option_edits.current_value(field_id)
            option_edits.set_personality(pid, choice)
            record(field_id, before)
    if any(plan.unit_counts):
        unit_edits.swap_players(plan.a, plan.b)
    if plan.trigger_writes:
        find_triggers.write_fields(trigger_manager, plan.trigger_writes)


def players_swapped(keys: Sequence[tuple[int, int]], a: int, b: int) -> list[tuple[int, int]]:
    """Units-mode selection keys (player, reference_id) after a swap."""
    return [(_sigma(a, b, player), ref) for player, ref in keys]
