#!/usr/bin/env python3
"""Asks AoE2:DE what its own Swap Players moves (GH #185), and compares that
with DEscape's swap.

`make [--source <file>] [--out <dir>]` writes two files into
build/swap_players_probe/ (gitignored):

- before.aoe2scenario: the real 1.59 editor blank with three players, built
  through the app's own models and write_scenario(). P1 and P2 differ on every
  Players field and each disables one object of its own; diplomacy is
  asymmetric with P3 as the witness; each player owns a few distinct units;
  one trigger names P1 and P2 as source and target players. `--source` builds
  the same from another 1.56+ scenario at least 93 tiles across, one whose P1
  and P2 already differ in data no editor writes, which the blank can't
  measure (a second pair, written with `--out` beside the first).
- after_descape.aoe2scenario: before.aoe2scenario reloaded fresh, so DEscape
  swaps the bytes the game will, then plan_swap() and apply_swap() of P1 and
  P2 with both checkboxes on.

`compare <before> <after_game> [<after_descape>]` reads each file with
DEscape's own readers and prints, per row, whether the game swapped it, left
it or did something else, the same for DEscape, and whether they agree. Rows
cover what DEscape swaps plus the per-player data no editor writes, so the
in-game pass shows whether the game moves those too. Such a row whose P1 and
P2 values are equal in before reads `unmeasured`, never `same`: that pair
can't show whether the game swaps it. Exits 1 on any disagreement.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import ai_scripts, batch_api, diplomacy_fields, disables_fields, library_compat, player_fields, player_swap
from descape.edit_history import EditHistory
from descape.options_model import OptionsEditModel
from descape.scenario_io import is_workshop_subscribed, parse_triggers
from descape.scenario_write import WriteBlockedError, write_scenario
from descape.trigger_model import TriggerEditModel
from descape.unit_model import UnitEditModel

SOURCE = ROOT / "tests" / "fixtures" / "real_blank_120x120_v159.aoe2scenario"
OUT = ROOT / "build" / "swap_players_probe"
A, B, WITNESS = 1, 2, 3

# field_id -> (P1, P2). Every Players field differs, so a mirrored or one-way write reads back wrong.
FIELDS: dict[str, tuple] = {
    "tribe_name": ("Probe Alpha", "Probe Beta"),
    "player_name_string_id": (-2, 44700),
    "civilization": ("BRITON-CIV", "FRANKISH-CIV"),
    "architecture": ("BYZANTINE-CIV", "JAPANESE-CIV"),
    "lock_civilization": (1, 0),
    "lock_personality": (0, 1),
    "starting_age": (3, 4),
    "food": (111, 222),
    "wood": (333, 444),
    "stone": (555, 666),
    "gold": (777, 888),
    "pop_limit": (75, 125),
    "color": (3, 5),
    "human": (1, 0),
    "base_priority": (3, 7),
    "initial_view_x": (10, 100),
    "initial_view_y": (10, 100),
}
PERSONALITIES = (ai_scripts.STANDARD, ai_scripts.NONE)
DISABLES = {("buildings", A): (82,), ("techs", B): (22,)}  # Castle for P1, Loom for P2
ALLY, NEUTRAL, ENEMY = 0, 1, 3
# P3 to P2 neutral too, so the witness's own row shows the column swap.
STANCES = {(A, B): ENEMY, (B, A): NEUTRAL, (A, WITNESS): ALLY, (WITNESS, A): ENEMY, (WITNESS, B): NEUTRAL}
ALLIED_VICTORY = {A: 1, B: 0, WITNESS: 0}
UNITS = {
    A: ((4, 20.5, 20.5), (74, 22.5, 20.5), (83, 24.5, 20.5)),  # Archer, Militia, Villager
    B: ((38, 90.5, 90.5), (125, 92.5, 90.5)),  # Knight, Monk
    WITNESS: ((448, 60.5, 60.5),),  # Scout Cavalry
}
TRIGGER_NAME = "Swap probe"
# (kind, type id, attributes): Own Objects, Diplomacy State, Change Ownership, Change Diplomacy, Send Chat.
TRIGGER_ENTRIES = (
    ("condition", 3, {"quantity": 1, "object_list": 4, "source_player": A}),
    ("condition", 24, {"quantity": ENEMY, "source_player": B, "target_player": A}),
    ("effect", 18, {"object_list_unit_id": 4, "source_player": A, "target_player": B,
                    "area_x1": 10, "area_y1": 10, "area_x2": 30, "area_y2": 30}),
    ("effect", 1, {"diplomacy": ENEMY, "source_player": B, "target_player": WITNESS}),
    ("effect", 3, {"source_player": B, "message": "Swap probe: P2 speaks"}),
)

EXPECTED = """In the AoE2:DE editor (Verification H1/H2):
  H1: open before, Players tab, Swap Players on P1 and P2, then Save As
      swap_probe_after_game, and run this tool's compare on the three files.
  H2: open after_descape. The Players tab shows P1 and P2's settings exchanged
      (P1 Probe Beta, Franks, Castle Age, food 222, Computer, None; P2 Probe
      Alpha, Britons, Feudal Age, food 111, Either, Standard), the Archer,
      Militia and Villager now P2's, the Knight and Monk P1's, the Swap probe
      trigger's player fields read swapped, and Save succeeds. P1 now carries
      P2's Name string ID 44700, so its name may show as that game string
      rather than Probe Beta; that is the string ID swapping too."""


def _refuse_subscribed(out: Path) -> None:
    if is_workshop_subscribed(out):
        raise WriteBlockedError(f"Refusing to write into a subscribed-mods (mods/subscribed/) folder: {out}")


def write_before(source: Path, path: Path) -> Path:
    """before.aoe2scenario, built through the real models and write_scenario()."""
    _refuse_subscribed(path)
    loaded = batch_api.load(source)
    history = EditHistory()
    options = OptionsEditModel(loaded)
    options.set_value(player_fields.PLAYER_COUNT_FIELD_ID, 3)
    for field_id, values in FIELDS.items():
        for pid, value in zip((A, B), values, strict=True):
            options.set_value(player_fields.player_field_id(field_id, pid), value)
    for pid, choice in zip((A, B), PERSONALITIES, strict=True):
        options.set_personality(pid, choice)
    for (category, pid), ids in DISABLES.items():
        options.set_value(disables_fields.disables_field_id(category, pid), ids)
    for (row, col), stance in STANCES.items():
        options.set_value(diplomacy_fields.stance_cell_id(row, col), stance)
    for pid, flag in ALLIED_VICTORY.items():
        options.set_value(diplomacy_fields.allied_victory_cell_id(pid), flag)

    units = UnitEditModel(loaded)
    units.begin_unit_edit(list(UNITS))
    for pid, placements in UNITS.items():
        for const, x, y in placements:
            units.add(pid, const, x, y)
    units.commit_unit_edit("Probe units", history)

    if parse_triggers(loaded) is None:
        raise RuntimeError(f"{source}: triggers do not parse")
    triggers = TriggerEditModel(loaded)
    index = triggers.trigger_count
    triggers.begin_trigger_edit()
    triggers.structural_edit(lambda manager: manager.add_trigger(TRIGGER_NAME))
    triggers.commit_trigger_edit("Add probe trigger", history)
    vocabulary = library_compat.load_vocabulary(loaded.scenario_version)
    triggers.begin_trigger_edit([index])
    trigger = triggers.manager().triggers[index]
    for kind, type_id, values in TRIGGER_ENTRIES:
        definitions = vocabulary.conditions if kind == "condition" else vocabulary.effects
        if kind == "condition":
            trigger._add_condition(type_id)
            entry = trigger.conditions[-1]
        else:
            trigger._add_effect(type_id)
            entry = trigger.effects[-1]
        for name, value in values.items():
            assert name in definitions[type_id].attributes, (type_id, name)
            setattr(entry, name, value)
    triggers.commit_trigger_edit("Fill probe trigger", history)

    path.parent.mkdir(parents=True, exist_ok=True)
    write_scenario(loaded, path, triggers=triggers, options=options, units=units, backup=False)
    return path


def write_after_descape(before: Path, path: Path, opts: player_swap.SwapOptions | None = None) -> Path:
    """`before` reloaded fresh, then DEscape's swap of P1 and P2, written."""
    _refuse_subscribed(path)
    opts = opts or player_swap.SwapOptions()
    loaded = batch_api.load(before)
    manager = parse_triggers(loaded) if opts.triggers else None
    plan = player_swap.plan_swap(loaded, A, B, opts, {}, player_swap.file_gates(loaded), manager)
    if isinstance(plan, player_swap.SwapRefusal):
        raise RuntimeError(plan.reason)
    history = EditHistory()
    options = OptionsEditModel(loaded)
    units = UnitEditModel(loaded)
    triggers = TriggerEditModel(loaded) if plan.trigger_writes else None
    units.begin_unit_edit([A, B])
    if triggers is not None:
        triggers.begin_trigger_edit(plan.touched_triggers)
    records: list = []
    player_swap.apply_swap(plan, options, units, triggers.manager() if triggers else None, records)
    units.commit_unit_edit(plan.label, history)
    if triggers is not None:
        triggers.commit_trigger_edit(plan.label, history)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_scenario(loaded, path, triggers=triggers, options=options, units=units, backup=False)
    return path


# -- compare ---------------------------------------------------------------------

UNWRITTEN = "Unwritten:"  # prefixes each row of per-player data no editor writes


def _sigma(p: int) -> int:
    return B if p == A else A if p == B else p


def _short(value) -> str:
    text = repr(value)
    if len(text) <= 40:
        return text
    return f"<{hashlib.sha256(text.encode()).hexdigest()[:10]}>"


def _pair(read):
    """A per-player row: (P1, P2), and the same swapped."""
    def row(loaded):
        value = (read(loaded, A), read(loaded, B))
        return value, (value[1], value[0])
    return row


def _struct_field(section: str, retriever: str, field: str, index_of):
    def read(loaded, pid):
        return loaded._scenario.sections[section].retriever_map[retriever].data[index_of(pid)].retriever_map[field].data
    return read


def _no_gaia(pid: int) -> int:
    return pid - 1


def _gaia_first_permuted(values):
    """A GAIA-first per-player list with P1's and P2's slots exchanged."""
    values = list(values)
    values[A], values[B] = values[B], values[A]
    return values


def _units(loaded, pid):
    return tuple(sorted(Counter((u.unit_const, u.x, u.y) for u in loaded.unit_manager.units[pid]).items()))


def _personality(loaded, pid):
    layout = player_fields.personality_layout(loaded)
    if layout is None:
        return None
    choice = player_fields.stored_personalities(loaded, layout)[pid]
    return choice.stored_name, choice.ai_type


def _trigger_locations(loaded):
    """(trigger, kind, entry, attribute) for every PlayerId attribute a known entry type lists."""
    manager = parse_triggers(loaded)
    vocabulary = library_compat.load_vocabulary(loaded.scenario_version)
    out = []
    for t, trigger in enumerate(manager.triggers):
        for kind, entries, definitions, presentation, type_attribute in (
            ("condition", trigger.conditions, vocabulary.conditions, vocabulary.condition_presentation, "condition_type"),
            ("effect", trigger.effects, vocabulary.effects, vocabulary.effect_presentation, "effect_type"),
        ):
            for e, entry in enumerate(entries):
                definition = definitions.get(getattr(entry, type_attribute))
                if definition is None:
                    continue
                out.extend(
                    (t, kind, e, attribute)
                    for attribute in definition.attributes
                    if presentation.get(attribute) == player_swap.PLAYER_ID_PRESENTATION
                )
    return out


def _trigger_value(loaded, location):
    t, kind, e, attribute = location
    manager = parse_triggers(loaded)
    if manager is None or t >= len(manager.triggers):
        return None
    entries = manager.triggers[t].conditions if kind == "condition" else manager.triggers[t].effects
    return getattr(entries[e], attribute, None) if e < len(entries) else None


def _rows(before) -> list[tuple[str, object]]:
    """(name, row) where row(loaded) -> (value, value if swapped, or None when unknown)."""
    rows: list[tuple[str, object]] = [
        (f"Players: {spec.label}", _pair(lambda loaded, pid, s=spec: player_fields.current_value(loaded, s, pid)))
        for spec in player_fields.specs_for(before)
        if spec.field_id != "personality"
    ]
    rows.append(("Players: Personality (name, AI type)", _pair(_personality)))
    rows.extend(
        (f"Disabled {category}", _pair(lambda loaded, pid, c=category: disables_fields.current_ids(loaded, c, pid)))
        for category in disables_fields.CATEGORIES
    )
    for row_p in (A, B, WITNESS):
        for col_p in (A, B, WITNESS):
            if row_p == col_p == WITNESS:
                continue

            def stance(loaded, r=row_p, c=col_p):
                return (
                    diplomacy_fields.stance_value(loaded, r, c),
                    diplomacy_fields.stance_value(loaded, _sigma(r), _sigma(c)),
                )
            rows.append((f"Diplomacy: P{row_p} to P{col_p}", stance))
    rows.append(("Diplomacy: allied victory", _pair(diplomacy_fields.allied_victory_value)))
    rows.append(("Units: (const, x, y) per player", _pair(_units)))
    rows.append((f"Units: P{WITNESS} (witness)", lambda loaded: (_units(loaded, WITNESS), _units(loaded, WITNESS))))
    for location in _trigger_locations(before):
        t, kind, e, attribute = location

        def trig(loaded, loc=location):
            value = _trigger_value(loaded, loc)
            return value, (_sigma(value) if isinstance(value, int) else value)
        rows.append((f"Trigger {t} {kind} {e} {attribute}", trig))

    # The second copies the Players fields write along with the first.
    pd3 = lambda field: _struct_field("Units", "player_data_3", field, _no_gaia)  # noqa: E731
    rows.append(("Mirror: player_data_3.color", _pair(pd3("color"))))
    rows.extend(
        (f"Mirror: player_data_4.{field}", _pair(_struct_field("Units", "player_data_4", field, _no_gaia)))
        for field in ("food_duplicate", "wood_duplicate", "stone_duplicate", "gold_duplicate", "population_limit")
    )
    # Per-player data no editor writes: the plan's Decision 5 list, reported so H1 shows what the game does.
    rows.extend(
        (f"Unwritten: player_data_3.{field}", _pair(pd3(field)))
        for field in ("constant_name", "editor_camera_x", "editor_camera_y", "initial_camera_x", "initial_camera_y",
                      "aok_allied_victory", "victory_version")
    )
    for field in ("diplomacy_for_interaction", "diplomacy_for_ai_system"):
        read = pd3(field)

        def dup(loaded, read=read):
            value = (list(read(loaded, A)), list(read(loaded, B)))
            return value, (_gaia_first_permuted(value[1]), _gaia_first_permuted(value[0]))
        rows.append((f"Unwritten: player_data_3.{field}", dup))
    p1_to_p8 = player_fields.PlayerArrayLayout.P1_TO_P8_THEN_GAIA.index_for
    rows.append(("Unwritten: player_data_1.cty_mode", _pair(_struct_field("DataHeader", "player_data_1", "cty_mode", p1_to_p8))))
    rows.extend(
        (f"Unwritten: resources.{field}", _pair(_struct_field("PlayerDataTwo", "resources", field, p1_to_p8)))
        for field in ("trade_goods", "ore_x_unused")
    )
    rows.extend(
        (f"Unwritten: player_data_4.{field}", _pair(_struct_field("Units", "player_data_4", field, _no_gaia)))
        for field in ("trade_goods_duplicate", "ore_x_duplicate")
    )

    def victories(loaded):
        data = bytes(loaded._scenario.sections["Diplomacy"].retriever_map["individual_victories"].data)
        size = len(data) // 16  # 16 per-player blocks, as the 16x16 stance grid
        value = (data[(A - 1) * size : A * size], data[(B - 1) * size : B * size])
        return value, (value[1], value[0])
    rows.append(("Unwritten: Diplomacy.individual_victories (P1, P2 blocks)", victories))
    rows.append((
        "Unwritten: PlayerDataTwo.strings (whole)",
        lambda loaded: (tuple(loaded._scenario.sections["PlayerDataTwo"].retriever_map["strings"].data), None),
    ))
    return rows


def classify(before_row, after_row, unwritten: bool = False) -> str:
    """swapped, left, other; `same` where swapping changes nothing, but
    `unmeasured` for an unwritten row there, since that row can't tell
    swapped from left; for a row of unknown layout, left or changed."""
    old, swapped = before_row
    new = after_row[0]
    if swapped is None:
        return "left" if new == old else "changed"
    if old == swapped:
        if new != old:
            return "other"
        return "unmeasured" if unwritten else "same"
    if new == swapped:
        return "swapped"
    if new == old:
        return "left"
    return "other"


def compare(before_path: Path, game_path: Path, descape_path: Path | None = None) -> int:
    before = batch_api.load(before_path)
    game = batch_api.load(game_path)
    ours = batch_api.load(descape_path) if descape_path is not None else None
    disagreements = unmeasured = 0
    print(f"{'row':58s} {'game':10s} {'DEscape':10s} agree")
    for name, row in _rows(before):
        before_row = row(before)
        unwritten = name.startswith(UNWRITTEN)
        game_class = classify(before_row, row(game), unwritten)
        unmeasured += game_class == "unmeasured"
        line = f"{name:58s} {game_class:10s}"
        if ours is not None:
            our_class = classify(before_row, row(ours), unwritten)
            agree = game_class == our_class
            disagreements += not agree
            line += f" {our_class:10s} {'yes' if agree else 'NO'}"
            if not agree:
                line += f"  before {_short(before_row[0])}, game {_short(row(game)[0])}, DEscape {_short(row(ours)[0])}"
        print(line)
    if ours is not None:
        print(f"\n{disagreements} disagreement{'' if disagreements == 1 else 's'}.")
    if unmeasured:
        print(f"{unmeasured} unmeasured: P1 and P2 store one value there in before, so this pair can't show a swap.")
    return 1 if disagreements else 0


def make(source: Path, out: Path) -> int:
    _refuse_subscribed(out)
    before = write_before(source, out / "before.aoe2scenario")
    print(f"wrote {before}")
    after = write_after_descape(before, out / "after_descape.aoe2scenario")
    print(f"wrote {after}\n")
    # Self-check: DEscape swaps every row it owns and leaves the unwritten ones
    # ("unmeasured" where the source stores one value for both, so no swap shows there).
    failures = 0
    loaded_before, loaded_after = batch_api.load(before), batch_api.load(after)
    for name, row in _rows(loaded_before):
        unwritten = name.startswith(UNWRITTEN)
        got = classify(row(loaded_before), row(loaded_after), unwritten)
        want = ("left", "unmeasured") if unwritten else ("swapped", "same")
        if got not in want:
            failures += 1
            print(f"  [FAIL] {name}: {got}, expected {' or '.join(want)}")
    print(f"self-check: {failures} failed\n")
    print(EXPECTED)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    make_parser = sub.add_parser("make", help="write before and after_descape")
    make_parser.add_argument("--source", type=Path, default=SOURCE)
    make_parser.add_argument("--out", type=Path, default=OUT)
    compare_parser = sub.add_parser("compare", help="compare the game's swap with DEscape's")
    compare_parser.add_argument("before", type=Path)
    compare_parser.add_argument("after_game", type=Path)
    compare_parser.add_argument("after_descape", type=Path, nargs="?")
    args = parser.parse_args()
    if args.command == "make":
        return make(args.source, args.out)
    return compare(args.before, args.after_game, args.after_descape)


if __name__ == "__main__":
    raise SystemExit(main())
