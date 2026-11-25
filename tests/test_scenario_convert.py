"""GH #307: converting a 1.58 scenario to 1.59 (descape/scenario_convert.py).

The game's own upgrade is the only oracle for what it accepts, so these pin
the part a session can: the step table against the library's own structure
files, a byte-identical blank against the shipped template (itself checked
against a game-saved 1.59 blank), every field of every unit and condition
carried, and a verify gate that is shown to fail.
"""

from __future__ import annotations

import dataclasses
import json
import struct

import pytest
from AoE2ScenarioParser.exceptions.asp_exceptions import EndOfFileError

from descape import library_compat, scenario_convert, scenario_io, settings
from descape.messages_model import MessagesEditModel
from descape.options_model import OptionsEditModel
from descape.scenario_convert import TARGET_VERSION, ConversionRefusedError, convert_scenario
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units, parse_triggers
from descape.scenario_write import WriteBlockedError, _compress_bytes, write_scenario
from descape.trigger_model import TriggerEditModel
from descape.unit_model import UnitEditModel

import conftest

FIXTURES = conftest.ROOT / "tests" / "fixtures"
V158_BLANK = FIXTURES / "blank_120x120_v158.aoe2scenario"
UNITS = FIXTURES / "units_120x120.aoe2scenario"
TRIGGERS = FIXTURES / "triggers_120x120.aoe2scenario"
FIND_TRIGGERS = FIXTURES / "find_triggers_120x120.aoe2scenario"
V159 = FIXTURES / "real_blank_120x120_v159.aoe2scenario"
ATILLA = conftest.ROOT / "examples" / "atilla_1_scn_resaved.aoe2scenario"
STEP = scenario_convert.STEPS["1.58"]
UNIT_FIELDS = (
    "x", "y", "z", "reference_id", "unit_const", "status", "rotation",
    "initial_animation_frame", "garrisoned_in_id", "caption_string_id", "caption_string",
)  # fmt: skip


def _gen():
    return conftest.load_verify_module("gen_v159_fixture")


def _structure(version: str) -> dict:
    return json.loads((library_compat.VERSIONS_DIR / f"v{version}" / "structure.json").read_text(encoding="utf-8"))


def _units(loaded) -> list[list[tuple]]:
    return [[tuple(getattr(u, f) for f in UNIT_FIELDS) for u in player] for player in loaded.unit_manager.units]


def _trigger_strings(loaded) -> list[str]:
    manager = parse_triggers(loaded)  # re-called before reading: the class gating is process-global
    triggers = [manager.triggers[i] for i in manager.trigger_display_order]
    return [f"{t.name}\n{t.get_content_as_string()}" for t in triggers]


def _condition_struct(version: str) -> dict:
    return _structure(version)["Triggers"]["structs"]["TriggerStruct"]["structs"]["ConditionStruct"]["retrievers"]


# -- the step table against the library ------------------------------------------


def test_the_step_is_the_whole_158_to_159_delta() -> None:
    """Fails when the library's 1.58 or 1.59 structure moves: the step table and
    its byte offsets were measured on exactly this delta."""
    old, new = _structure("1.58"), _structure("1.59")
    assert set(old) == set(new)
    for section in old:
        if section not in ("Units", "Triggers"):  # the header carriers differ in value, not in layout
            assert old[section] == new[section], section
    old_unit = scenario_convert.unit_struct_retrievers("1.58")
    new_unit = scenario_convert.unit_struct_retrievers("1.59")
    names = list(new_unit)
    assert names[names.index(STEP.unit_field) - 1] == STEP.unit_after
    assert {n: a for n, a in new_unit.items() if n != STEP.unit_field} == old_unit
    assert (new_unit[STEP.unit_field]["type"], new_unit[STEP.unit_field]["default"]) == ("s8", -1)
    assert struct.pack("<b", new_unit[STEP.unit_field]["default"]) == STEP.unit_fill
    assert scenario_convert.unit_insert_offset(STEP) == 29

    old_cond, new_cond = _condition_struct("1.58"), _condition_struct("1.59")
    cond_names = list(new_cond)
    assert cond_names[-2:] == [STEP.condition_field, "xs_function"]
    assert {n: a for n, a in new_cond.items() if n not in (STEP.condition_field, "static_value_33")} == {
        n: a for n, a in old_cond.items() if n != "static_value_33"
    }
    assert (old_cond["static_value_33"]["default"], new_cond["static_value_33"]["default"]) == STEP.condition_counts
    assert struct.pack("<i", new_cond[STEP.condition_field]["default"]) == STEP.condition_fill
    # Every counted field is an s32 between the type and xs_function, which convert_body() relies on.
    assert len(cond_names) - 3 == STEP.condition_counts[1]
    assert {new_cond[n]["type"] for n in cond_names[:-1]} == {"s32"}

    def vocabulary(version: str, kind: str) -> dict:
        return json.loads((library_compat.VERSIONS_DIR / f"v{version}" / f"{kind}.json").read_text(encoding="utf-8"))

    assert vocabulary("1.58", "effects") == vocabulary("1.59", "effects")
    old_c, new_c = vocabulary("1.58", "conditions"), vocabulary("1.59", "conditions")
    assert old_c.keys() == new_c.keys()
    for key in old_c:
        for part in ("attributes", "default_attributes"):
            extra = [a for a in new_c[key].get(part, ()) if a not in old_c[key].get(part, ())]
            assert extra in ([], [STEP.condition_field]), (key, part, extra)


def test_the_target_is_the_newest_version_descape_reads() -> None:
    """The tripwire: a game update that adds a version fails here until
    scenario_convert.STEPS gains a step to it, measured on its structure diff."""
    newest = library_compat.vocabulary_versions()[-1]
    assert newest == TARGET_VERSION, f"scenario_convert.STEPS needs a {TARGET_VERSION} to {newest} step"
    assert [step.target for step in scenario_convert.STEPS.values()] == [TARGET_VERSION]


# -- conversions ----------------------------------------------------------------


def test_the_158_blank_converts_to_the_shipped_159_template_byte_for_byte() -> None:
    result = convert_scenario(load_map_and_units(V158_BLANK))
    assert result.data == BLANK_TEMPLATE_PATH.read_bytes()
    assert (result.units, result.conditions) == (0, 0)


@pytest.mark.parametrize("path", [UNITS, TRIGGERS, FIND_TRIGGERS], ids=lambda p: p.stem)
def test_a_fixture_converts_with_every_unit_and_trigger_carried(path) -> None:
    gen = _gen()
    source = load_map_and_units(path)
    units_before = _units(source)
    triggers_before = _trigger_strings(source)
    result = convert_scenario(source)
    loaded = result.loaded
    assert (loaded.scenario_version, loaded.trigger_version) == (TARGET_VERSION, 5.0)
    assert loaded.terrain_write_supported and loaded.units_write_supported and loaded.messages_write_supported
    assert len(loaded.decompressed_body) - len(source.decompressed_body) == result.units + 4 * result.conditions
    assert result.units == sum(map(len, units_before))
    assert _units(loaded) == units_before
    assert _trigger_strings(loaded) == triggers_before
    assert loaded.trigger_write_supported
    assert set(gen.read_allow_in_fog(loaded).values()) <= {-1}
    assert len(gen.read_allow_in_fog(loaded)) == result.conditions

    slow = scenario_io.load_map_and_units_from_bytes(result.data, path, fast_units=False)
    assert _units(slow) == units_before
    assert set(gen.read_capture_flags(slow).values()) <= {-1}
    assert len(gen.read_capture_flags(slow)) == result.units


def test_a_converted_document_saves_and_reloads_as_a_full_159_one(tmp_path) -> None:
    """Every unit and trigger dirtied, a Capture Flag set and a condition 27
    added on the converted document: the library commit takes them all."""
    gen = _gen()
    loaded = convert_scenario(load_map_and_units(TRIGGERS)).loaded
    units = UnitEditModel(loaded)
    flagged = loaded.unit_manager.units[1][0]
    units.set_capture_flag(flagged, 2)
    for player in loaded.unit_manager.units:
        for u in player:
            units.set_position(u, u.x, u.y, u.z)
    triggers = TriggerEditModel(loaded)
    triggers.structural_edit(lambda m: m.add_trigger("Converted"))
    added = triggers.manager().triggers[-1]
    added._add_condition(27)
    assert added.conditions[0].allow_in_fog == -1, "the converted document is the active one"
    for index in range(triggers.trigger_count):
        triggers.mark_dirty(index)
    out = tmp_path / "converted.aoe2scenario"
    write_scenario(loaded, out, backup=False, units=units, triggers=triggers)

    back = load_map_and_units(out, fast_units=False)
    assert back.scenario_version == TARGET_VERSION
    assert gen.read_capture_flags(back)[flagged.reference_id] == 2
    manager = parse_triggers(back)
    assert back.trigger_write_supported and manager.triggers[-1].conditions[0].condition_type == 27
    fog = gen.read_allow_in_fog(back)
    assert set(fog.values()) == {-1} and len(fog) == 8


def _edited_158(path=TRIGGERS):
    """A 1.58 fixture with pending unit, trigger, option and message edits."""
    loaded = load_map_and_units(path)
    units = UnitEditModel(loaded)
    first = loaded.unit_manager.units[1][0]
    units.set_position(first, first.x + 2, first.y + 1, first.z)
    units.remove(loaded.unit_manager.units[1][1])
    units.add(1, 83, 30.5, 40.5)
    triggers = TriggerEditModel(loaded)
    manager = triggers.manager()
    manager.triggers[0].name = "Renamed before converting"
    manager.triggers[0]._add_effect(3)
    manager.triggers[0].effects[-1].message = "added before converting"
    triggers.mark_dirty(0)
    options = OptionsEditModel(loaded)
    options.set_value("lock_teams", 1 - options.original_value("lock_teams"))
    messages = MessagesEditModel(loaded)
    messages.set_value("hints", "converted hint")
    return loaded, {"units": units, "triggers": triggers, "options": options, "messages": messages}


def test_pending_edits_are_carried_and_the_source_is_untouched(tmp_path, monkeypatch) -> None:
    """The result equals a 1.58 save of the same models, reloaded. Saves made
    after a failed and after a successful conversion equal one made before
    (a zero-edit save would pass trivially: it reuses the original bytes)."""
    loaded, models = _edited_158()
    before = tmp_path / "before.aoe2scenario"
    write_scenario(loaded, before, backup=False, **models)

    def refuse(*args, **kwargs):
        raise WriteBlockedError("forced")

    with monkeypatch.context() as m:
        m.setattr(scenario_convert, "verify_conversion", refuse)
        with pytest.raises(WriteBlockedError, match="forced"):
            convert_scenario(loaded, **models)
    after_failure = tmp_path / "after_failure.aoe2scenario"
    write_scenario(loaded, after_failure, backup=False, **models)
    assert after_failure.read_bytes() == before.read_bytes()

    result = convert_scenario(loaded, **models)
    scenario_io.reactivate(loaded)
    after_success = tmp_path / "after_success.aoe2scenario"
    write_scenario(loaded, after_success, backup=False, **models)
    assert after_success.read_bytes() == before.read_bytes()

    saved = load_map_and_units(before)
    expected_units, expected_triggers = _units(saved), _trigger_strings(saved)
    old, new = saved.decompressed_body, result.loaded.decompressed_body
    assert old[:4] + old[8 : saved.units_block_offset] == new[:4] + new[8 : result.loaded.units_block_offset]
    assert _units(result.loaded) == expected_units
    assert _trigger_strings(result.loaded) == expected_triggers
    assert any("Renamed before converting" in s and "added before converting" in s for s in expected_triggers)
    assert sum(map(len, expected_units)) == 3, "one moved, one removed, one added"


def test_a_failure_after_the_159_reload_restores_the_158_vocabulary(tmp_path, monkeypatch) -> None:
    """Without scenario_io.reactivate() the library's condition defaults stay
    1.59's, so a condition added to the still-open 1.58 document gains
    allow_in_fog. The save afterwards shows the document still saves as 1.58."""
    import AoE2ScenarioParser.datasets.conditions as condition_dataset

    loaded = load_map_and_units(TRIGGERS)
    triggers = TriggerEditModel(loaded)

    def refuse(*args, **kwargs):
        raise WriteBlockedError("forced")

    monkeypatch.setattr(scenario_convert, "verify_conversion", refuse)
    with pytest.raises(WriteBlockedError):
        convert_scenario(loaded, triggers=triggers)
    assert "allow_in_fog" not in condition_dataset.default_attributes[0]

    triggers.structural_edit(lambda m: m.add_trigger("After a failed conversion"))
    triggers.manager().triggers[-1]._add_condition(27)
    triggers.mark_dirty(triggers.trigger_count - 1)
    out = tmp_path / "still_158.aoe2scenario"
    write_scenario(loaded, out, backup=False, triggers=triggers)
    back = load_map_and_units(out)
    assert back.scenario_version == "1.58"
    assert parse_triggers(back).triggers[-1].conditions[0].condition_type == 27


# -- refusals and the gate --------------------------------------------------------


def test_refusals_name_their_reason() -> None:
    assert scenario_convert.source_refusal("1.58") is None
    assert scenario_convert.source_refusal("1.59") == "This scenario is already version 1.59."
    assert scenario_convert.source_refusal("1.57") == "Only version 1.58 scenarios can be converted; this one is 1.57."
    with pytest.raises(ConversionRefusedError, match=r"already version 1\.59"):
        convert_scenario(load_map_and_units(V159))


def test_a_158_file_whose_trigger_version_reads_5_is_refused() -> None:
    loaded = load_map_and_units(TRIGGERS)
    body = bytearray(loaded.decompressed_body)
    body[loaded.units_section_end : loaded.units_section_end + 8] = struct.pack("<d", 5.0)
    patched = scenario_io.load_map_and_units_from_bytes(
        loaded.header_bytes + _compress_bytes(bytes(body)), TRIGGERS
    )
    with pytest.raises(ConversionRefusedError, match=r"trigger version does not read 4\.9"):
        convert_scenario(patched)


@pytest.mark.parametrize(
    ("path", "offset"),
    # units_120x120's unit 200 is garrisoned, so a byte shifted into garrisoned_in_id is not FF. triggers_120x120
    # has no garrisoned unit (28 and 25 give identical bytes there), so it takes 24.
    [(UNITS, 28), (UNITS, 25), (TRIGGERS, 24)],
    ids=["units-28", "units-25", "triggers-24"],
)
def test_the_gate_catches_a_wrong_unit_offset(monkeypatch, path, offset) -> None:
    monkeypatch.setattr(scenario_convert, "unit_insert_offset", lambda step: offset)
    with pytest.raises(WriteBlockedError, match="changed in conversion"):
        convert_scenario(load_map_and_units(path))


def test_the_gate_catches_a_skipped_count_overwrite(monkeypatch) -> None:
    """A step whose count overwrite writes the old count: the field is inserted, its count left at 33."""
    monkeypatch.setitem(scenario_convert.STEPS, "1.58", dataclasses.replace(STEP, condition_counts=(33, 33)))
    with pytest.raises(WriteBlockedError, match=r"without the 1\.59 fields"):
        convert_scenario(load_map_and_units(TRIGGERS))


# -- the paste step table and the label --------------------------------------------


def test_paste_upgrade_fields_covers_equal_versions_and_the_one_step() -> None:
    versions = library_compat.vocabulary_versions()
    for a in versions:
        for b in versions:
            fields = scenario_convert.paste_upgrade_fields(a, b)
            if a == b:
                assert fields == ()
            elif (a, b) == ("1.58", "1.59"):
                assert fields == ("allow_in_fog",)
            else:
                assert fields is None, (a, b)


def test_the_label_tracks_the_target() -> None:
    labels = {key: label for key, label, _default in settings.REBINDABLE_ACTIONS}
    assert TARGET_VERSION in labels["file_convert"]
    assert labels["file_convert"] == scenario_convert.action_label()


# -- corpus ------------------------------------------------------------------------


def _converts(path) -> None:
    try:
        source = load_map_and_units(path)
    except (scenario_io.UnsupportedStructureVersion, ValueError, EndOfFileError) as e:
        pytest.skip(f"{path.name} does not open: {type(e).__name__}")
    if source.scenario_version != "1.58":
        pytest.skip(f"{path.name} is {source.scenario_version}")
    units_before, triggers_before = _units(source), _trigger_strings(source)
    result = convert_scenario(source)
    assert result.loaded.scenario_version == TARGET_VERSION
    assert _units(result.loaded) == units_before
    assert _trigger_strings(result.loaded) == triggers_before


@pytest.mark.corpus
def test_every_158_corpus_file_converts(scenario_path) -> None:
    _converts(scenario_path)


@pytest.mark.corpus
def test_atilla_converts() -> None:
    """Named directly: CORPUS_SUBSET_NAMES has no 1.58 file."""
    if not ATILLA.is_file():
        pytest.skip(f"missing {ATILLA}")
    _converts(ATILLA)
