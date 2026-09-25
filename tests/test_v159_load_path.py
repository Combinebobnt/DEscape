"""Default-tier coverage for scenario version 1.59, read through the vendored
descape/versions/DE/v1.59/ definitions on the 0.8.3 pin.

Everything runs on tests/fixtures/v159_units_triggers.aoe2scenario
(tools/gen_v159_fixture.py). The two 1.59-only fields, `capture_flag` and
`allow_in_fog`, have no RetrieverObjectLink on 0.8.3, so the library leaves
them in their list slot when it re-slots objects on save. The slot-shift
tests below are the regression guard for descape/unlinked_fields.py: each
edit moves an object into a slot whose old value differs from its own, then
re-reads the saved file and checks the value by reference id or condition
identity, never by position.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from AoE2ScenarioParser.objects.data_objects.condition import Condition
from AoE2ScenarioParser.objects.data_objects.unit import Unit

from descape import library_compat, scenario_io, unlinked_fields
from descape.scenario_io import load_map_and_units, parse_triggers
from descape.scenario_write import write_scenario
from descape.trigger_model import TriggerEditModel
from descape.unit_model import UnitEditModel

import conftest

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "v159_units_triggers.aoe2scenario"


class _History:
    def push_unit_record(self, record) -> None:
        pass

    def push_trigger_record(self, record) -> None:
        pass


def _gen():
    return conftest.load_verify_module("gen_v159_fixture")


def _unit(loaded, reference_id: int) -> Unit:
    return next(u for units in loaded.unit_manager.units for u in units if u.reference_id == reference_id)


def _save_and_reload(loaded, tmp_path, name: str, **models):
    out = tmp_path / f"{name}.aoe2scenario"
    write_scenario(loaded, out, backup=False, **models)
    return load_map_and_units(out)


# -- the fixture itself -------------------------------------------------------


def test_fixture_matches_its_generator() -> None:
    assert _gen().build_fixture_bytes() == FIXTURE.read_bytes()


def test_fixture_loads_from_the_repo_structure_with_triggers() -> None:
    gen = _gen()
    loaded = load_map_and_units(FIXTURE)
    assert (loaded.scenario_version, loaded.structure_source) == ("1.59", "repo")
    assert loaded.terrain_write_supported and loaded.units_write_supported and loaded.messages_write_supported
    assert loaded.trigger_read_supported is None
    manager = parse_triggers(loaded)
    assert manager is not None and len(manager.triggers) == gen.TRIGGER_COUNT
    assert loaded.trigger_read_supported and loaded.trigger_write_supported
    UnitEditModel(loaded)
    TriggerEditModel(loaded)


def test_the_carrier_pulls_both_fields_at_load() -> None:
    gen = _gen()
    loaded = load_map_and_units(FIXTURE)
    assert {u.reference_id: u.capture_flag for units in loaded.unit_manager.units for u in units} == gen.CAPTURE_FLAGS
    manager = parse_triggers(loaded)
    got = {(t, c): cond.allow_in_fog for t, trig in enumerate(manager.triggers) for c, cond in enumerate(trig.conditions)}
    assert got == gen.ALLOW_IN_FOG


def test_a_no_op_save_is_byte_identical(tmp_path) -> None:
    loaded = load_map_and_units(FIXTURE)
    out = tmp_path / "same.aoe2scenario"
    write_scenario(loaded, out, backup=False)
    assert out.read_bytes() == FIXTURE.read_bytes()


def test_a_dirty_save_rewrites_every_unit_and_trigger_identically(tmp_path) -> None:
    """Review #14: a no-op save reuses the original compressed bytes, so it
    proves nothing about the write path. Every unit and trigger is dirtied
    here, so every one goes through the library commit and the carrier.

    Units must come back byte-identical. A dirty trigger carries the library's
    known string drift (descape/trigger_model.py's docstring), so the Triggers
    section is compared retriever by retriever: both files parsed, then both
    re-serialized the same way."""
    fixture_bytes = conftest.load_verify_module("_fixture_bytes")
    pristine = load_map_and_units(FIXTURE)
    parse_triggers(pristine)
    expected_section = fixture_bytes._game_style_bytes(pristine._scenario.sections["Triggers"])
    loaded = load_map_and_units(FIXTURE)
    units = UnitEditModel(loaded)
    for player_units in loaded.unit_manager.units:
        for u in player_units:
            units.set_position(u, u.x, u.y, u.z)
    triggers = TriggerEditModel(loaded)
    for index in range(triggers.trigger_count):
        triggers.mark_dirty(index)
    original = loaded.decompressed_body
    start, end = loaded.units_section_end, loaded.triggers_section_end

    reloaded = _save_and_reload(loaded, tmp_path, "dirty", units=units, triggers=triggers)
    assert reloaded.header_bytes == loaded.header_bytes
    assert reloaded.decompressed_body[:start] == original[:start]
    parse_triggers(reloaded)
    assert fixture_bytes._game_style_bytes(reloaded._scenario.sections["Triggers"]) == expected_section
    assert reloaded.decompressed_body[reloaded.triggers_section_end :] == original[end:]


# -- slot-shift regressions ---------------------------------------------------


def test_delete_then_move_keeps_each_flag_on_its_own_unit(tmp_path) -> None:
    """The measured review #2 case: delete a unit, move its list neighbour,
    and the neighbour must not inherit the deleted unit's flag."""
    gen = _gen()
    loaded = load_map_and_units(FIXTURE)
    model = UnitEditModel(loaded)
    model.remove(_unit(loaded, 100))  # GAIA, flag 1
    moved = _unit(loaded, 101)  # its neighbour, flag -1
    model.set_position(moved, moved.x + 1, moved.y, moved.z)

    flags = gen.read_capture_flags(_save_and_reload(loaded, tmp_path, "delete_move", units=model))
    expected = {ref: flag for ref, flag in gen.CAPTURE_FLAGS.items() if ref != 100}
    assert flags == expected


def test_reassign_then_move_keeps_each_flag_on_its_own_unit(tmp_path) -> None:
    gen = _gen()
    loaded = load_map_and_units(FIXTURE)
    model = UnitEditModel(loaded)
    model.reassign(_unit(loaded, 200), 2)  # P1 flag 2 -> P2
    moved = _unit(loaded, 201)  # slides into 200's old slot, flag -1
    model.set_position(moved, moved.x + 1, moved.y, moved.z)

    reloaded = _save_and_reload(loaded, tmp_path, "reassign_move", units=model)
    assert gen.read_capture_flags(reloaded) == gen.CAPTURE_FLAGS
    assert _unit(reloaded, 200) in reloaded.unit_manager.units[2]


def test_delete_then_add_writes_the_default_not_the_slots_old_value(tmp_path) -> None:
    gen = _gen()
    loaded = load_map_and_units(FIXTURE)
    model = UnitEditModel(loaded)
    model.remove(_unit(loaded, 301))  # P2 middle, flag 3
    added = model.add(player=2, unit_const=4, x=45.5, y=30.5)  # lands in 302's old slot (flag 2)
    flagged = model.add(player=2, unit_const=4, x=46.5, y=30.5, capture_flag=0)

    flags = gen.read_capture_flags(_save_and_reload(loaded, tmp_path, "delete_add", units=model))
    expected = {ref: flag for ref, flag in gen.CAPTURE_FLAGS.items() if ref != 301}
    expected[added.reference_id] = -1
    expected[flagged.reference_id] = 0
    assert flags == expected


def test_removing_a_condition_before_27_keeps_allow_in_fog_on_it(tmp_path) -> None:
    gen = _gen()
    loaded = load_map_and_units(FIXTURE)
    model = TriggerEditModel(loaded)
    model.begin_trigger_edit(content_touched=[gen.FOG_TRIGGER])
    trigger = model.manager().triggers[gen.FOG_TRIGGER]
    trigger.remove_condition(condition_index=0)
    trigger._add_condition(27)  # a fresh condition 27, into the freed tail slot
    model.commit_trigger_edit("Remove condition", _History())

    reloaded = _save_and_reload(loaded, tmp_path, "remove_condition", triggers=model)
    manager = parse_triggers(reloaded)
    kept = [(c.condition_type, c.allow_in_fog) for c in manager.triggers[gen.FOG_TRIGGER].conditions]
    assert kept == [(27, 1), (10, -1), (27, -1)]
    assert gen.read_allow_in_fog(reloaded)[(1, 0)] == gen.ALLOW_IN_FOG[(1, 0)]


def test_unit_undo_restores_the_carried_flag_with_the_unit(tmp_path) -> None:
    gen = _gen()
    loaded = load_map_and_units(FIXTURE)
    model = UnitEditModel(loaded)
    model.begin_unit_edit([0])
    model.remove(_unit(loaded, 100))
    record = model.commit_unit_edit("Delete", _History(), push=False)
    model.restore(record.before)
    moved = _unit(loaded, 101)
    model.set_position(moved, moved.x + 1, moved.y, moved.z)

    flags = gen.read_capture_flags(_save_and_reload(loaded, tmp_path, "undo", units=model))
    assert flags == gen.CAPTURE_FLAGS


def test_trigger_snapshot_restore_keeps_allow_in_fog() -> None:
    gen = _gen()
    loaded = load_map_and_units(FIXTURE)
    model = TriggerEditModel(loaded)
    before = model.snapshot([gen.FOG_TRIGGER])
    model.manager().triggers[gen.FOG_TRIGGER].conditions[gen.FOG_CONDITION].allow_in_fog = 7
    model.restore(before)
    assert model.manager().triggers[gen.FOG_TRIGGER].conditions[gen.FOG_CONDITION].allow_in_fog == 1


def test_depoison_leaves_carried_values_alone() -> None:
    loaded = load_map_and_units(FIXTURE)
    library_compat.depoison()
    assert _unit(loaded, 102).capture_flag == 3


# -- copies ---------------------------------------------------------------------


def test_trigger_copy_paste_keeps_allow_in_fog(tmp_path) -> None:
    from descape import trigger_clipboard

    gen = _gen()
    loaded = load_map_and_units(FIXTURE)
    model = TriggerEditModel(loaded)
    block = trigger_clipboard.copy_block(model.manager(), [gen.FOG_TRIGGER], doc_id="a", scenario_version="1.59")
    model.begin_trigger_edit()
    model.structural_edit(lambda m: trigger_clipboard.paste_into(m, block, doc_id="a"))
    model.commit_trigger_edit("Paste", _History())

    manager = parse_triggers(_save_and_reload(loaded, tmp_path, "trigger_paste", triggers=model))
    pasted = manager.triggers[gen.TRIGGER_COUNT]
    assert [(c.condition_type, c.allow_in_fog) for c in pasted.conditions] == [(10, -1), (27, 1), (10, -1)]


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_paste_region_keeps_capture_flag(tmp_path) -> None:
    gen = _gen()
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    try:
        window.load_scenario(FIXTURE)
        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected("select")
        window.on_region_selected((20, 10, 24, 11))  # P1's four units
        window.copy_region()
        window.paste_units_check.setChecked(True)
        window.on_hover((60, 60))
        window.paste_region()

        out = tmp_path / "paste.aoe2scenario"
        write_scenario(window.scenario, out, backup=False, **window._edit_model_kwargs())
        reloaded = load_map_and_units(out)
        flags = gen.read_capture_flags(reloaded)
        pasted = {
            int(u.x) - 40: flags[u.reference_id]
            for u in reloaded.unit_manager.units[1]
            if int(u.y) == 60
        }
        assert pasted == {20: 2, 21: -1, 22: 1, 23: 0}
        assert {ref: flags[ref] for ref in gen.CAPTURE_FLAGS} == gen.CAPTURE_FLAGS
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_mirroring_keeps_capture_flag(tmp_path) -> None:
    from descape.mirror_tools import plan_mirror, plan_mirror_units

    gen = _gen()
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    try:
        window.load_scenario(FIXTURE)
        loaded = window.scenario
        mm = loaded.map_manager
        model = window._ensure_unit_edits()
        tiles = plan_mirror(mm, 1, 0, False, False)
        unit_plan = plan_mirror_units(
            mm, 1, 0, loaded.unit_manager.units, tiles.source_indices, referencing=model.referencing
        )
        assert unit_plan.images and not unit_plan.blocked
        sources = {id(image): image.source.reference_id for image in unit_plan.images}
        window.on_mirror(tiles, unit_plan)

        out = tmp_path / "mirror.aoe2scenario"
        write_scenario(loaded, out, backup=False, **window._edit_model_kwargs())
        flags = gen.read_capture_flags(load_map_and_units(out))
        by_position = {
            (round(float(u.x), 3), round(float(u.y), 3)): u.reference_id for units in loaded.unit_manager.units for u in units
        }
        for image in unit_plan.images:
            ref = by_position[(round(image.x, 3), round(image.y, 3))]
            assert flags[ref] == gen.CAPTURE_FLAGS[sources[id(image)]], image
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- the vocabulary filter and the carrier table ------------------------------


def test_the_repo_vocabulary_drops_exactly_allow_in_fog() -> None:
    assert library_compat.repo_only_attributes("1.59") == {
        "conditions": frozenset({"allow_in_fog"}),
        "effects": frozenset(),
    }
    vocabulary = library_compat.load_vocabulary("1.59")
    for entries, presentation in (
        (vocabulary.conditions, vocabulary.condition_presentation),
        (vocabulary.effects, vocabulary.effect_presentation),
    ):
        assert "allow_in_fog" not in presentation
        for entry in entries.values():
            assert "allow_in_fog" not in entry.attributes and "allow_in_fog" not in entry.default_attributes
    # Private-name links (effect.py) are vocabulary names too, so they stay.
    assert "quantity" in vocabulary.effects[0].default_attributes
    assert library_compat.repo_only_attributes("1.58") == {"conditions": frozenset(), "effects": frozenset()}


def _struct(structure: dict, name: str) -> dict | None:
    for value in structure.values() if isinstance(structure, dict) else ():
        if isinstance(value, dict):
            if name in value:
                return value[name]
            found = _struct(value, name)
            if found is not None:
                return found
    return None


def _unlinked(versions_dir: Path, cls: type, struct_name: str) -> dict[str, set[str]]:
    out = {}
    for path in sorted(versions_dir.glob("v*/structure.json")):
        found = _struct(json.loads(path.read_text(encoding="utf-8")), struct_name)
        out[path.parent.name] = set(found["retrievers"]) - unlinked_fields.link_names(cls)
    return out


@pytest.mark.parametrize(("cls", "struct_name"), [(Unit, "UnitStruct"), (Condition, "ConditionStruct")])
def test_the_carrier_table_is_exactly_what_repo_structures_add(cls, struct_name) -> None:
    """UNLINKED_FIELDS equals, per class, the struct fields a repo structure
    has that the installed class doesn't link, beyond the constant/padding
    fields every library structure already leaves unlinked. After a pin bump
    that links them, this difference is empty and the carrier is dormant."""
    baseline = set().union(*_unlinked(library_compat.VERSIONS_DIR, cls, struct_name).values())
    added = set().union(*_unlinked(library_compat.REPO_VERSIONS_DIR, cls, struct_name).values()) - baseline
    assert added == set(unlinked_fields.UNLINKED_FIELDS[cls])


def test_the_carrier_is_dormant_before_1_59() -> None:
    loaded = load_map_and_units("tests/fixtures/units_120x120.aoe2scenario")
    entry = loaded._scenario.sections["Units"].retriever_map["players_units"].data[0].retriever_map["units"].data[0]
    assert unlinked_fields.active_fields(Unit, entry.retriever_map) == []
    assert not any(hasattr(u, "capture_flag") for units in loaded.unit_manager.units for u in units)


def test_scenario_io_names_the_repo_versions() -> None:
    assert scenario_io._repo_versions() == ["1.21", "1.59"]


# -- corpus tier: the one real game-saved 1.59 file ---------------------------

REAL_V159 = Path(__file__).resolve().parent.parent / "examples" / "8tp3w9j.aoe2scenario"


def _real_v159():
    if not REAL_V159.is_file():
        pytest.skip(f"{REAL_V159.name} is not in examples/")
    return load_map_and_units(REAL_V159)


def _raw_flags(loaded) -> dict[int, int]:
    return _gen().read_capture_flags(loaded)


@pytest.mark.corpus
def test_real_file_dirty_save_rewrites_every_unit_identically(tmp_path) -> None:
    loaded = _real_v159()
    assert (loaded.scenario_version, loaded.structure_source) == ("1.59", "repo")
    manager = parse_triggers(loaded)
    assert manager is not None and len(manager.triggers) == 38 and loaded.trigger_write_supported
    units = UnitEditModel(loaded)
    for player_units in loaded.unit_manager.units:
        for u in player_units:
            units.set_position(u, u.x, u.y, u.z)
    reloaded = _save_and_reload(loaded, tmp_path, "dirty", units=units)
    assert reloaded.decompressed_body == loaded.decompressed_body


@pytest.mark.corpus
def test_real_file_delete_then_move_keeps_flags_by_reference(tmp_path) -> None:
    """The review's measured case: delete GAIA ref 12004 (Once), move its
    list neighbour 14802 (Default); 14802 must still save as Default."""
    loaded = _real_v159()
    before = _raw_flags(loaded)
    assert (before[12004], before[14802]) == (1, -1)
    gaia = loaded.unit_manager.units[0]
    index = next(i for i, u in enumerate(gaia) if u.reference_id == 12004)
    assert gaia[index + 1].reference_id == 14802, "no longer list neighbours"
    model = UnitEditModel(loaded)
    model.remove(gaia[index])
    moved = _unit(loaded, 14802)
    model.set_position(moved, moved.x + 1, moved.y, moved.z)

    after = _raw_flags(_save_and_reload(loaded, tmp_path, "delete_move", units=model))
    del before[12004]
    assert after == before
