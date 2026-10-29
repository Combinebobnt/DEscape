"""Swap Players' Qt-free core (GH #185): descape/player_swap.py, plus
UnitEditModel.swap_players() and OptionsEditModel.can_set().

The default tier runs on tools/gen_swap_players_probe.py's before file, built
from the real 1.59 editor blank with P1 and P2 differing on every Players field,
asymmetric diplomacy, units and a trigger naming both. The test map's
player-held GAIA-only objects run in the test_map tier, and every examples/
file in the corpus tier.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from pathlib import Path

import pytest

from descape import ai_scripts, batch_api, diplomacy_fields, disables_fields, options_model, player_fields, player_swap, unit_kind
from descape.edit_history import CompositeDiffRecord, EditHistory
from descape.options_model import OptionsEditModel
from descape.player_swap import SwapGates, SwapOptions, SwapPlan, SwapRefusal, apply_swap, file_gates, plan_swap
from descape.scenario_io import parse_triggers
from descape.scenario_write import write_scenario
from descape.trigger_model import TriggerEditModel
from descape.unit_model import UnitEditModel

import conftest

A, B, WITNESS = 1, 2, 3
_BOTH = SwapOptions()


@pytest.fixture(scope="module")
def probe():
    return conftest.load_verify_module("gen_swap_players_probe")


@pytest.fixture(scope="module")
def before_path(probe, tmp_path_factory) -> Path:
    return probe.write_before(probe.SOURCE, tmp_path_factory.mktemp("swap") / "before.aoe2scenario")


class _Doc:
    """One open document with every model the viewer would build, swapped as
    ViewerWindow.swap_players() does: one composite record per swap."""

    def __init__(self, path: Path):
        self.loaded = batch_api.load(path)
        self.options = OptionsEditModel(self.loaded)
        self.units = UnitEditModel(self.loaded)
        writable = parse_triggers(self.loaded) is not None and self.loaded.trigger_write_supported
        self.triggers = TriggerEditModel(self.loaded) if writable else None
        self.history = EditHistory()

    def plan(self, a: int, b: int, opts: SwapOptions = _BOTH, gates: SwapGates | None = None):
        personalities = (
            {p: self.options.current_personality(p) for p in (a, b)} if self.options.personality_supported else None
        )
        manager = parse_triggers(self.loaded) if opts.triggers else None
        gates = gates if gates is not None else file_gates(self.loaded)
        return plan_swap(self.loaded, a, b, opts, self.options.pending_values(), gates, manager, personalities)

    def swap(self, a: int, b: int, opts: SwapOptions = _BOTH) -> SwapPlan:
        plan = self.plan(a, b, opts)
        assert isinstance(plan, SwapPlan), plan
        self.units.begin_unit_edit([a, b])
        if self.triggers is not None:
            self.triggers.begin_trigger_edit(plan.touched_triggers)
        records: list = []
        apply_swap(plan, self.options, self.units, self.triggers.manager() if self.triggers else None, records)
        children = [self.units.commit_unit_edit(plan.label, self.history, push=False)]
        if self.triggers is not None:
            children.append(self.triggers.commit_trigger_edit(plan.label, self.history, push=False))
        self.history.push_composite_record(CompositeDiffRecord(plan.label, [*children, *records]))
        return plan

    def undo(self) -> None:
        self.history.undo(self.loaded.map_manager.terrain, self.triggers, self.options, self.units)

    def redo(self) -> None:
        self.history.redo(self.loaded.map_manager.terrain, self.triggers, self.options, self.units)

    def save(self, path: Path):
        write_scenario(self.loaded, path, triggers=self.triggers, options=self.options, units=self.units, backup=False)
        reloaded = batch_api.load(path)
        parse_triggers(reloaded)
        return reloaded


def _field(loaded, field_id: str, pid: int):
    spec = next(s for s in player_fields.specs_for(loaded) if s.field_id == field_id)
    return player_fields.current_value(loaded, spec, pid)


def _player_values(loaded, pid: int) -> dict:
    return {s.field_id: player_fields.current_value(loaded, s, pid) for s in player_fields.specs_for(loaded)}


def _units_of(loaded, pid: int) -> Counter:
    return Counter((u.unit_const, u.x, u.y) for u in loaded.unit_manager.units[pid])


def _grid(loaded) -> list[list[int]]:
    rows = loaded._scenario.sections["Diplomacy"].retriever_map["per_player_diplomacy"].data
    return [list(row.retriever_map["stance_with_each_player"].data) for row in rows]


def _player_attributes(loaded) -> list[tuple]:
    """(trigger, kind, entry, attribute, value) for source_player/target_player on every entry."""
    out = []
    for t, trigger in enumerate(parse_triggers(loaded).triggers):
        for kind, entries in (("condition", trigger.conditions), ("effect", trigger.effects)):
            for e, entry in enumerate(entries):
                out.extend((t, kind, e, name, player_swap._read(entry, name)) for name in ("source_player", "target_player"))
    return out


def _sigma(p):
    return B if p == A else A if p == B else p


def _listed(loaded, t: int, kind: str, e: int, name: str) -> bool:
    """Whether the entry's own type lists `name` (the vocabulary, not a hand list)."""
    vocabulary = player_swap._vocabulary(loaded)
    if vocabulary is None:
        return False
    trigger = parse_triggers(loaded).triggers[t]
    entry = (trigger.conditions if kind == "condition" else trigger.effects)[e]
    definitions = vocabulary.conditions if kind == "condition" else vocabulary.effects
    definition = definitions.get(getattr(entry, f"{kind}_type"))
    return definition is not None and name in definition.attributes


def _expected_after_swap(loaded, rows) -> list[tuple]:
    """`rows` as a swap leaves them: listed attributes through the permutation, the rest as stored."""
    return [(t, k, e, n, _sigma(v) if _listed(loaded, t, k, e, n) else v) for t, k, e, n, v in rows]


# -- every category swaps and survives a save --------------------------------


def test_every_player_field_and_its_mirrors_swap_through_a_save_and_reload(before_path, tmp_path) -> None:
    doc = _Doc(before_path)
    before_a, before_b = _player_values(doc.loaded, A), _player_values(doc.loaded, B)
    assert all(before_a[k] != before_b[k] for k in before_a), "the probe makes every Players field differ"
    witness = _player_values(doc.loaded, WITNESS)

    doc.swap(A, B)
    after = doc.save(tmp_path / "after.aoe2scenario")

    assert _player_values(after, A) == before_b
    assert _player_values(after, B) == before_a
    assert _player_values(after, WITNESS) == witness
    # The second copies the first one's write carries: player_data_4 and player_data_3.
    pd4 = after._scenario.sections["Units"].retriever_map["player_data_4"].data
    pd3 = after._scenario.sections["Units"].retriever_map["player_data_3"].data
    caps = after._scenario.sections["Map"].retriever_map["per_player_population_cap"].data
    for pid, source in ((A, before_b), (B, before_a)):
        entry = pd4[pid - 1].retriever_map
        for resource in ("food", "wood", "stone", "gold"):
            assert entry[f"{resource}_duplicate"].data == source[resource]
        assert entry["population_limit"].data == source["pop_limit"] == caps[pid - 1]
        assert pd3[pid - 1].retriever_map["color"].data == source["color"]
    # The camera's GAIA-first array: P1 is index 1, P2 index 2, GAIA's slot 0 untouched.
    views = after._scenario.sections["Map"].retriever_map["initial_player_views"].data
    assert views[A].retriever_map["location_x"].data == before_b["initial_view_x"]
    assert views[B].retriever_map["location_y"].data == before_a["initial_view_y"]
    assert player_fields.verify_player_block(after)


def test_disables_and_units_swap_and_a_witness_keeps_its_own(before_path, tmp_path) -> None:
    doc = _Doc(before_path)
    ids = {(c, p): disables_fields.current_ids(doc.loaded, c, p) for c in disables_fields.CATEGORIES for p in (A, B, WITNESS)}
    units = {p: _units_of(doc.loaded, p) for p in (A, B, WITNESS)}
    assert ids[("buildings", A)] != ids[("buildings", B)] and ids[("techs", A)] != ids[("techs", B)]

    plan = doc.swap(A, B)
    after = doc.save(tmp_path / "after.aoe2scenario")

    assert plan.unit_counts == (sum(units[A].values()), sum(units[B].values()))
    for category in disables_fields.CATEGORIES:
        assert disables_fields.current_ids(after, category, A) == ids[(category, B)]
        assert disables_fields.current_ids(after, category, B) == ids[(category, A)]
        assert disables_fields.current_ids(after, category, WITNESS) == ids[(category, WITNESS)]
    assert (_units_of(after, A), _units_of(after, B), _units_of(after, WITNESS)) == (units[B], units[A], units[WITNESS])


def test_standard_and_none_personalities_swap(before_path, tmp_path) -> None:
    doc = _Doc(before_path)
    assert (doc.options.current_personality(A).stored_name, doc.options.current_personality(B).stored_name) == (
        ai_scripts.STANDARD.stored_name,
        ai_scripts.NONE.stored_name,
    )
    doc.swap(A, B)
    after = doc.save(tmp_path / "after.aoe2scenario")
    stored = player_fields.stored_personalities(after, player_fields.personality_layout(after))
    assert (stored[A].stored_name, stored[A].ai_type) == (ai_scripts.NONE.stored_name, ai_scripts.NONE.ai_type)
    assert (stored[B].stored_name, stored[B].ai_type) == (ai_scripts.STANDARD.stored_name, ai_scripts.STANDARD.ai_type)


def _custom_ai(name: str, text: bytes) -> ai_scripts.AiChoice:
    stem = name.rsplit(".", 1)[0].encode()
    library = ((stem + b".per\x00", text),)
    return ai_scripts.AiChoice(f"test:{name}", name, ai_scripts.AI_TYPE_CUSTOM, "test", text=text, library=library, resolved=True)


def _library(loaded) -> list[tuple[bytes, bytes]]:
    layout = player_fields.ai_library_layout(loaded)
    return [(e.name, player_fields.ai_library_entry_text(loaded.decompressed_body, e)) for e in layout.entries]


def test_a_stored_custom_ai_moves_with_its_name_type_script_and_library(before_path, tmp_path) -> None:
    """The library closure (serialize_ai_library_resize()) keeps the moved AI's entries."""
    doc = _Doc(before_path)
    doc.options.set_personality(A, _custom_ai("SwapProbe.ai", b"(defrule (true) => (do-nothing))"))
    custom_path = tmp_path / "custom.aoe2scenario"
    stored_before = doc.save(custom_path)
    layout = player_fields.personality_layout(stored_before)
    p1 = player_fields.stored_personalities(stored_before, layout)[A]
    assert p1.is_custom and p1.text
    library = _library(stored_before)
    assert library, "the custom AI's .per went into the Files library"

    doc = _Doc(custom_path)
    doc.swap(A, B)
    after = doc.save(tmp_path / "after.aoe2scenario")
    moved = player_fields.stored_personalities(after, player_fields.personality_layout(after))
    assert (moved[B].stored_name, moved[B].ai_type, moved[B].text) == (p1.stored_name, p1.ai_type, p1.text)
    assert (moved[A].stored_name, moved[A].ai_type) == (ai_scripts.NONE.stored_name, ai_scripts.NONE.ai_type)
    assert _library(after) == library


# -- diplomacy --------------------------------------------------------------------


def test_the_diplomacy_permutation_moves_rows_columns_diagonal_and_allied_victory(before_path, tmp_path) -> None:
    doc = _Doc(before_path)
    # A pending diagonal edit: the swap moves what is shown, and the diagonal is per-file data.
    doc.options.set_value(diplomacy_fields.stance_cell_id(A, A), 3)
    doc.options.set_value(diplomacy_fields.stance_cell_id(B, B), 0)
    old = {
        (i, j): doc.options.current_value(diplomacy_fields.stance_cell_id(i, j)) for i in range(1, 9) for j in range(1, 9)
    }
    allied = {p: doc.options.current_value(diplomacy_fields.allied_victory_cell_id(p)) for p in range(1, 9)}
    assert old[(A, B)] != old[(B, A)] and old[(A, WITNESS)] != old[(B, WITNESS)], "the probe's grid is asymmetric"
    assert old[(WITNESS, A)] != old[(WITNESS, B)], "the witness's own row distinguishes A from B"
    filler = [row[8:] for row in _grid(doc.loaded)[:8]] + _grid(doc.loaded)[8:]

    doc.swap(A, B)
    after = doc.save(tmp_path / "after.aoe2scenario")

    for i, j in old:
        assert diplomacy_fields.stance_value(after, i, j) == old[(_sigma(i), _sigma(j))], (i, j)
    for p in range(1, 9):
        assert diplomacy_fields.allied_victory_value(after, p) == allied[_sigma(p)]
    assert [row[8:] for row in _grid(after)[:8]] + _grid(after)[8:] == filler, "indices 8..15 stay verbatim"


def test_the_diplomacy_box_unticked_leaves_the_grid_alone(before_path, tmp_path) -> None:
    doc = _Doc(before_path)
    grid = _grid(doc.loaded)
    plan = doc.swap(A, B, SwapOptions(diplomacy=False))
    assert not any(k.startswith(("stance:", "allied_victory:")) for k in plan.option_writes)
    assert _grid(doc.save(tmp_path / "after.aoe2scenario")) == grid


# -- triggers ---------------------------------------------------------------------


def _add_trigger(doc: _Doc, entries) -> int:
    model = doc.triggers
    index = model.trigger_count
    model.begin_trigger_edit()
    model.structural_edit(lambda manager: manager.add_trigger("Swap edge cases"))
    model.commit_trigger_edit("Add", doc.history)
    model.begin_trigger_edit([index])
    trigger = model.manager().triggers[index]
    for kind, type_id, values in entries:
        if kind == "condition":
            trigger._add_condition(type_id)
            entry = trigger.conditions[-1]
        else:
            trigger._add_effect(type_id)
            entry = trigger.effects[-1]
        for name, value in values.items():
            setattr(entry, name, value)
    model.commit_trigger_edit("Fill", doc.history)
    return index


def _edge_case_file(before_path: Path, tmp_path: Path) -> Path:
    """The probe file plus a trigger of values the swap must leave alone."""
    doc = _Doc(before_path)
    _add_trigger(
        doc,
        (
            ("effect", 3, {"source_player": B, "target_player": A, "message": "stale"}),  # Send Chat lists no target
            ("effect", 1, {"source_player": 0, "target_player": -1, "diplomacy": 0}),  # GAIA and unset
            ("effect", 13, {"source_player": WITNESS}),  # Declare Victory, another player
            ("condition", 24, {"source_player": A, "target_player": A, "quantity": 3}),  # both fields A
        ),
    )
    path = tmp_path / "edge.aoe2scenario"
    doc.save(path)
    return path


def test_only_listed_player_fields_naming_a_or_b_are_rewritten_and_survive_a_reload(before_path, tmp_path) -> None:
    """Read back after a reload: a rewritten trigger not declared content_touched would
    have its stored bytes spliced back by the save."""
    doc = _Doc(_edge_case_file(before_path, tmp_path))
    before = _player_attributes(doc.loaded)
    edge = len(parse_triggers(doc.loaded).triggers) - 1
    assert (edge, "effect", 0, "target_player", A) in before, "a stale value Send Chat does not list"
    assert (edge, "effect", 1, "source_player", 0) in before and (edge, "effect", 1, "target_player", -1) in before
    expected = _expected_after_swap(doc.loaded, before)
    assert expected != before

    plan = doc.swap(A, B)
    after = _player_attributes(doc.save(tmp_path / "after.aoe2scenario"))

    assert after == expected
    assert (edge, "effect", 0, "target_player", A) in after, "the stale unlisted value stays as stored"
    assert (edge, "condition", 0, "target_player", B) in after, "a listed field naming A becomes B"
    assert (edge, "effect", 2, "source_player", WITNESS) in after, "another player stays"
    assert {w.attribute for w in plan.trigger_writes} == {"source_player", "target_player"}
    assert plan.touched_triggers == sorted({w.trigger_index for w in plan.trigger_writes})


def test_an_entry_of_an_unknown_type_is_counted_and_untouched(before_path, monkeypatch) -> None:
    doc = _Doc(before_path)
    vocabulary = player_swap._vocabulary(doc.loaded)
    effects = {k: v for k, v in vocabulary.effects.items() if k != 18}  # Change Ownership unknown
    monkeypatch.setattr(player_swap, "_vocabulary", lambda _loaded: dataclasses.replace(vocabulary, effects=effects))
    writes, unknown = player_swap.trigger_writes(doc.loaded, parse_triggers(doc.loaded), A, B)
    assert unknown == 1
    assert not [w for w in writes if w.kind == "effect" and w.entry_index == 0], "the unknown entry is not written"
    assert writes, "the known entries still are"


def test_the_trigger_box_unticked_leaves_every_trigger_alone(before_path) -> None:
    doc = _Doc(before_path)
    before = _player_attributes(doc.loaded)
    plan = doc.swap(A, B, SwapOptions(triggers=False))
    assert plan.trigger_writes == [] and _player_attributes(doc.loaded) == before
    assert not doc.triggers.has_edits


# -- the unit model -----------------------------------------------------------------


def test_swap_players_swaps_lists_and_blobs_in_order_and_undo_restores_the_layout(before_path) -> None:
    loaded = batch_api.load(before_path)
    model = UnitEditModel(loaded)
    host = model.add(A, 82, 40.5, 40.5)  # a Castle
    rider = model.add(A, 4, 40.5, 40.5, garrisoned_in_id=host.reference_id)
    model.warm_garrison_map()
    units_a, units_b = list(loaded.unit_manager.units[A]), list(loaded.unit_manager.units[B])
    blobs_a, blobs_b = list(model._blobs[A]), list(model._blobs[B])
    serialized = model.serialize()
    history = EditHistory()

    model.begin_unit_edit([A, B])
    model.swap_players(A, B)
    model.commit_unit_edit("swap", history)

    assert list(loaded.unit_manager.units[A]) == units_b and list(loaded.unit_manager.units[B]) == units_a
    assert model._blobs[A] == blobs_b and model._blobs[B] == blobs_a
    assert all(u._player == B for u in units_a) and all(u._player == A for u in units_b)
    assert rider.garrisoned_in_id == host.reference_id
    assert model.referencing(host) == [rider]
    model._check_alignment()  # _pos and the garrison map agree with a fresh walk

    history.undo(loaded.map_manager.terrain, units=model)
    assert model.serialize() == serialized
    assert list(loaded.unit_manager.units[A]) == units_a


@pytest.mark.parametrize(("a", "b"), [(0, 1), (1, 0), (1, 1), (1, 9), (-1, 2)])
def test_swap_players_refuses_gaia_one_player_or_out_of_range(before_path, a, b) -> None:
    model = UnitEditModel(batch_api.load(before_path))
    with pytest.raises(ValueError):
        model.swap_players(a, b)


def test_swap_players_refuses_inside_a_fields_only_edit(before_path) -> None:
    model = UnitEditModel(batch_api.load(before_path))
    model.begin_unit_edit([A, B], fields_only=True)
    with pytest.raises(RuntimeError):
        model.swap_players(A, B)


@pytest.mark.test_map
def test_a_player_held_gaia_only_object_moves_with_its_player() -> None:
    """Decision 3: the swap gives no object to a logical player that lacked it, so it is not refused."""
    doc = _Doc(conftest.TEST_MAP_PATH)
    held = [
        (p, u) for p in range(1, 9) for u in doc.loaded.unit_manager.units[p] if unit_kind.owner_refused(u.unit_const, p)
    ]
    assert held, "the test map holds player-owned GAIA-only objects"
    p, unit = held[0]
    q = 1 if p != 1 else 2
    doc.swap(p, q)
    assert any(u is unit for u in doc.loaded.unit_manager.units[q])


# -- apply_swap's records ---------------------------------------------------------


def test_a_raise_partway_keeps_every_applied_option_edit_recorded(before_path, monkeypatch) -> None:
    doc = _Doc(before_path)
    plan = doc.plan(A, B)
    real = doc.options.set_value
    calls = []

    def flaky(field_id, value):
        calls.append(field_id)
        if len(calls) == 5:
            raise ValueError("injected")
        real(field_id, value)

    monkeypatch.setattr(doc.options, "set_value", flaky)
    records: list = []
    with pytest.raises(ValueError, match="injected"):
        apply_swap(plan, doc.options, doc.units, None, records)
    assert [r.field_id for r in records] == calls[:4]
    assert set(doc.options.pending_values()) == set(calls[:4])


def test_file_gates_takes_the_callers_custom_ai_gate_without_computing_it(before_path, monkeypatch) -> None:
    """The viewer passes its per-document cache: computing the gate forces the trigger parse."""
    loaded = batch_api.load(before_path)
    gates = file_gates(loaded)
    assert gates.personality_custom, "the probe file's Files library verifies"
    monkeypatch.setattr(options_model, "personality_custom_supported", lambda _loaded: pytest.fail("computed anyway"))
    assert file_gates(loaded, personality_custom=True) == gates
    assert file_gates(loaded, personality_custom=False) == dataclasses.replace(gates, personality_custom=False)


def test_can_set_names_the_ids_set_value_accepts(before_path) -> None:
    model = OptionsEditModel(batch_api.load(before_path))
    for field_id in ("player:food:1", "stance:1:2", "allied_victory:3", "disabled:techs:2", "personality:4"):
        assert model.can_set(field_id), field_id
    for field_id in ("player:food:9", "personality:0", "player_count", "no_such_field"):
        assert not model.can_set(field_id), field_id


# -- refusals ---------------------------------------------------------------------


def _gates(before_path, **changes) -> SwapGates:
    return dataclasses.replace(file_gates(batch_api.load(before_path)), **changes)


def _refusal(plan) -> str:
    assert isinstance(plan, SwapRefusal), plan
    return plan.reason


def test_a_read_only_field_that_differs_refuses_and_names_it(before_path) -> None:
    loaded = batch_api.load(before_path)
    gates = _gates(before_path)
    gates = dataclasses.replace(gates, player_fields=gates.player_fields - {"food"})
    reason = _refusal(plan_swap(loaded, A, B, SwapOptions(), {}, gates, parse_triggers(loaded)))
    assert "Starting food" in reason and "Starting wood" not in reason


def test_an_equal_read_only_field_does_not_refuse(before_path) -> None:
    loaded = batch_api.load(before_path)
    assert _field(loaded, "lock_civilization", B) == _field(loaded, "lock_civilization", WITNESS)
    gates = _gates(before_path)
    gates = dataclasses.replace(gates, player_fields=gates.player_fields - {"lock_civilization"})
    plan = plan_swap(loaded, B, WITNESS, SwapOptions(), {}, gates, parse_triggers(loaded))
    assert isinstance(plan, SwapPlan) and "player:lock_civilization:2" not in plan.option_writes


@pytest.mark.parametrize(
    ("changes", "opts", "named"),
    [
        ({"units": False}, SwapOptions(), "units"),
        ({"disables": False}, SwapOptions(), "disabled buildings"),
        ({"diplomacy": False}, SwapOptions(), "diplomacy"),
    ],
)
def test_each_unwritable_category_that_would_change_refuses(before_path, changes, opts, named) -> None:
    loaded = batch_api.load(before_path)
    reason = _refusal(plan_swap(loaded, A, B, opts, {}, _gates(before_path, **changes), parse_triggers(loaded)))
    assert named in reason


def test_unticking_diplomacy_lifts_its_refusal(before_path) -> None:
    loaded = batch_api.load(before_path)
    plan = plan_swap(loaded, A, B, SwapOptions(diplomacy=False), {}, _gates(before_path, diplomacy=False), parse_triggers(loaded))
    assert isinstance(plan, SwapPlan)


def test_personality_refuses_when_unwritable_or_a_custom_ai_cannot_move(before_path) -> None:
    doc = _Doc(before_path)
    gates = file_gates(doc.loaded)
    assert "Personality" in _refusal(doc.plan(A, B, gates=dataclasses.replace(gates, player_fields=gates.player_fields - {"personality"})))
    assert isinstance(doc.plan(A, B, gates=dataclasses.replace(gates, personality_custom=False)), SwapPlan), "no custom AI"
    doc.options.set_personality(A, _custom_ai("SwapProbe.ai", b"(defrule (true) => (do-nothing))"))
    assert "custom AI" in _refusal(doc.plan(A, B, gates=dataclasses.replace(gates, personality_custom=False)))
    assert isinstance(doc.plan(A, B, gates=gates), SwapPlan)


def test_triggers_unreadable_or_unwritable_refuse_while_the_box_is_ticked(before_path) -> None:
    loaded = batch_api.load(before_path)
    gates = _gates(before_path)
    assert "can't be read" in _refusal(plan_swap(loaded, A, B, SwapOptions(), {}, gates, None))
    assert isinstance(plan_swap(loaded, A, B, SwapOptions(triggers=False), {}, gates, None), SwapPlan)
    manager = parse_triggers(loaded)
    loaded.trigger_write_supported = False
    assert "can't be written" in _refusal(plan_swap(loaded, A, B, SwapOptions(), {}, gates, manager))


def test_only_defined_players_can_be_swapped_and_never_gaia(before_path) -> None:
    loaded = batch_api.load(before_path)
    assert player_swap.swap_partners(loaded, A) == [B, WITNESS]
    assert player_swap.swap_partners(loaded, A, {player_fields.PLAYER_COUNT_FIELD_ID: 2}) == [B]
    assert "defined" in _refusal(plan_swap(loaded, A, 5, SwapOptions(), {}, _gates(before_path), None))
    with pytest.raises(ValueError):
        plan_swap(loaded, 0, A, SwapOptions(), {}, _gates(before_path), None)
    with pytest.raises(ValueError):
        plan_swap(loaded, A, A, SwapOptions(), {}, _gates(before_path), None)


def test_players_that_differ_in_nothing_shown_make_an_empty_plan(probe) -> None:
    """The editor blank's P1 and P2 differ only in colour; a pending edit evens that out."""
    doc = _Doc(probe.SOURCE)
    assert set(doc.plan(A, B).option_writes) == {"player:color:1", "player:color:2"}
    doc.options.set_value("player:color:2", doc.options.current_value("player:color:1"))
    plan = doc.plan(A, B)
    assert isinstance(plan, SwapPlan) and plan.is_empty


# -- swap twice, undo ---------------------------------------------------------------


def test_swapping_back_restores_every_value_and_undo_restores_the_bytes(before_path, tmp_path) -> None:
    zero = _Doc(before_path)
    zero_path = tmp_path / "zero.aoe2scenario"
    write_scenario(zero.loaded, zero_path, backup=False)
    zero_units = UnitEditModel(batch_api.load(before_path)).serialize()

    doc = _Doc(before_path)
    original = {p: _player_values(doc.loaded, p) for p in (A, B)}
    attributes = _player_attributes(doc.loaded)
    doc.swap(A, B)
    doc.swap(A, B)
    assert not doc.options.has_edits, "two swaps put every option value back to the file's own"
    assert doc.units.serialize() == zero_units
    back = doc.save(tmp_path / "back.aoe2scenario")
    assert {p: _player_values(back, p) for p in (A, B)} == original
    assert _player_attributes(back) == attributes

    doc.undo()
    doc.undo()
    undone = tmp_path / "undone.aoe2scenario"
    write_scenario(doc.loaded, undone, triggers=doc.triggers, options=doc.options, units=doc.units, backup=False)
    assert undone.read_bytes() == zero_path.read_bytes()
    doc.redo()
    assert doc.options.current_value("player:food:1") == original[B]["food"]


# -- the probe's compare ------------------------------------------------------------


def test_compare_lists_an_unwritten_row_equal_in_before_as_unmeasured(probe, before_path, capsys) -> None:
    """So a game result there can't be read as clearing it (TASK-529)."""
    loaded = batch_api.load(before_path)
    unwritten = {name: row(loaded) for name, row in probe._rows(loaded) if name.startswith(probe.UNWRITTEN)}
    equal = {name for name, (old, swapped) in unwritten.items() if swapped is not None and old == swapped}
    differ = {name for name, (old, swapped) in unwritten.items() if swapped is not None and old != swapped}
    assert equal and differ, "the blank-based probe has both kinds"

    assert probe.compare(before_path, before_path, before_path) == 0
    classes = {}
    for line in capsys.readouterr().out.splitlines():
        name = next((n for n in unwritten if line.startswith(n + " ")), None)
        if name is not None:
            classes[name] = line[len(name):].split()[:2]
    assert {name: classes[name] for name in equal} == {name: ["unmeasured", "unmeasured"] for name in equal}
    assert {name: classes[name] for name in differ} == {name: ["left", "left"] for name in differ}
    # A changed value is still reported where the two were equal: it isn't left alone either.
    assert probe.classify((1, 1), (2, None), unwritten=True) == "other"
    assert probe.classify((1, 1), (1, None)) == "same", "a row DEscape writes keeps same"


# -- corpus -------------------------------------------------------------------------


@pytest.mark.corpus
def test_swap_p1_and_p2_round_trips_on_the_corpus(scenario_path, tmp_path) -> None:
    loaded = batch_api.load(scenario_path)
    if len(player_fields.defined_player_ids(loaded)) < 2:
        pytest.skip("fewer than two players")
    gates = file_gates(loaded)
    if not (gates.player_fields and gates.diplomacy and gates.disables and gates.units):
        pytest.skip("a write gate fails on this file")
    parsed = parse_triggers(loaded) is not None
    # A file whose triggers can't be written still swaps, with that box unticked.
    opts = SwapOptions(triggers=parsed and bool(loaded.trigger_write_supported))
    zero_path = tmp_path / "zero.aoe2scenario"
    write_scenario(loaded, zero_path, backup=False)

    doc = _Doc(scenario_path)
    plan = doc.plan(A, B, opts)
    if isinstance(plan, SwapRefusal):
        pytest.skip(plan.reason)
    values = {p: _player_values(doc.loaded, p) for p in (A, B)}
    units = {p: _units_of(doc.loaded, p) for p in (A, B)}
    disabled = {(c, p): disables_fields.current_ids(doc.loaded, c, p) for c in disables_fields.CATEGORIES for p in (A, B)}
    stances = {(i, j): diplomacy_fields.stance_value(doc.loaded, i, j) for i in range(1, 9) for j in range(1, 9)}
    attributes = _player_attributes(doc.loaded) if parsed else []
    expected_attributes = _expected_after_swap(doc.loaded, attributes) if opts.triggers else attributes
    personalities = {}
    if doc.options.personality_supported:
        personalities = {p: (c.stored_name, c.ai_type) for p in (A, B) for c in [doc.options.current_personality(p)]}

    doc.swap(A, B, opts)
    after = doc.save(tmp_path / "after.aoe2scenario")
    if personalities:
        stored = player_fields.stored_personalities(after, player_fields.personality_layout(after))
        assert {p: (stored[p].stored_name, stored[p].ai_type) for p in (A, B)} == {A: personalities[B], B: personalities[A]}
    assert (_player_values(after, A), _player_values(after, B)) == (values[B], values[A])
    assert (_units_of(after, A), _units_of(after, B)) == (units[B], units[A])
    for c in disables_fields.CATEGORIES:
        assert (disables_fields.current_ids(after, c, A), disables_fields.current_ids(after, c, B)) == (
            disabled[(c, B)], disabled[(c, A)],
        )
    for (i, j), value in stances.items():
        assert diplomacy_fields.stance_value(after, _sigma(i), _sigma(j)) == value
    if parsed:
        assert _player_attributes(after) == expected_attributes
    assert len(plan.trigger_writes) == sum(a != b for a, b in zip(attributes, expected_attributes, strict=True))

    doc.swap(A, B, opts)
    assert not doc.options.has_edits
    back = doc.save(tmp_path / "back.aoe2scenario")
    assert {p: _player_values(back, p) for p in (A, B)} == values
    if parsed:
        assert _player_attributes(back) == attributes

    doc.undo()
    doc.undo()
    undone = tmp_path / "undone.aoe2scenario"
    write_scenario(doc.loaded, undone, triggers=doc.triggers, options=doc.options, units=doc.units, backup=False)
    assert undone.read_bytes() == zero_path.read_bytes()
