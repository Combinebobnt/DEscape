"""The Map Options write path: descape/options_model.py's OptionsEditModel,
the options branch in descape/scenario_write.py, and the one row that does not
go through either of them.

Both halves live here rather than split across this file and
tests/test_trigger_write_path.py, even though the trigger execution-order flag
is written by TriggerEditModel: from a user's point of view it is a Map Options
row like any other, the bug it is guarded against is a Map Options bug, and
`-k option` has to select the whole surface or the filter reads as a clean run
while missing half of it.

The claims, in the order the risk runs:

1. **Containment.** A document whose options were only browsed writes exactly
   the bytes it wrote before this panel existed. The failure is silent.
2. **Locality.** One option edit changes exactly that field's bytes.
3. **Exec-order reaches the file at all.** It dirties no blob and changes no
   structure, so TriggerEditModel.has_edits has to name it explicitly or the
   verbatim-splice branch writes the original byte back with no error
   anywhere. This is the test that would report *zero* changed bytes rather
   than a wrong one, which is why it is written as an exact byte count.
4. **Round trip.** Write, reload through the real loader, read the value back.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from AoE2ScenarioParser.scenarios.aoe2_scenario import _decompress_bytes

from descape import option_fields
from descape.options_model import (
    OptionEditsUnavailableError,
    OptionsEditModel,
    field_offsets,
    options_write_supported,
    pack_struct_for,
)
from descape.scenario_io import load_map_and_units, parse_triggers
from descape.scenario_write import WriteBlockedError, write_scenario
from descape.trigger_model import TriggerEditModel, exec_order_value

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"
BLANK_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden_blank_120x120.aoe2scenario"


def _written_body(path: Path) -> bytes:
    raw = path.read_bytes()
    loaded = load_map_and_units(path)
    return _decompress_bytes(raw[len(loaded.header_bytes) :])


def _differing_ranges(before: bytes, after: bytes) -> list[tuple[int, int]]:
    """Maximal [start, end) spans where two equal-length buffers differ. Same
    helper tests/test_trigger_write_path.py uses, kept local rather than
    imported so neither file's imports depend on the other's layout."""
    assert len(before) == len(after)
    ranges: list[tuple[int, int]] = []
    start = None
    for i, (a, b) in enumerate(zip(before, after)):
        if a != b and start is None:
            start = i
        elif a == b and start is not None:
            ranges.append((start, i))
            start = None
    if start is not None:
        ranges.append((start, len(before)))
    return ranges


def _loaded(path: Path = FIXTURE_PATH):
    loaded = load_map_and_units(path)
    parse_triggers(loaded)
    return loaded


def _a_flag_spec(loaded):
    """A checkbox row this file stores as a real 0/1 flag. Picked from the
    file rather than named, so this does not silently start testing a row the
    spec list dropped.

    Pinned to lock_teams -- moving the Diplomacy-group specs to
    DiplomacyPanel only tagged them with OptionFieldSpec.panel="diplomacy",
    it did not remove them from option_fields._SPECS, so this still resolves
    to the same row it always has -- OptionsEditModel writes lock_teams
    exactly as before, regardless of which panel renders it. The assert
    below is what would catch it if that ever stopped being true."""
    for spec in option_fields.specs_for(loaded):
        if spec.kind == option_fields.CHECKBOX and spec.section != "Triggers":
            if option_fields.current_value(loaded, spec) in (0, 1):
                assert spec.field_id == "lock_teams", (
                    f"_a_flag_spec silently re-resolved to {spec.field_id!r}"
                )
                return spec
    raise AssertionError("no writable flag row on this file")


# -- 1. the packer ----------------------------------------------------------


@pytest.mark.parametrize("path", [FIXTURE_PATH, BLANK_FIXTURE], ids=["triggers", "blank"])
def test_every_writable_option_repacks_its_own_stored_bytes(path: Path) -> None:
    """The discriminator for the whole format table: a packer that reproduces
    the file's own bytes for the value already in the file has the right width
    *and* the right signedness. Neither is declared per spec -- both come from
    the retriever the file was parsed with -- so this is what would catch a
    drift between the offset walk and the packer.
    """
    loaded = _loaded(path)
    specs = option_fields.specs_for(loaded)
    offsets, _ = field_offsets(loaded, specs)
    assert offsets, "no offsets resolved -- this test would pass vacuously"
    for spec, _fo in offsets.items():
        retriever = loaded._scenario.sections[spec.section].retriever_map[spec.retriever]
        packer = pack_struct_for(retriever)
        assert packer is not None, f"{spec.field_id} has no packer"
        stored = option_fields.current_value(loaded, spec)
        assert packer.pack(stored) == retriever.get_data_as_bytes(), spec.field_id


def test_a_retriever_whose_width_disagrees_with_its_datatype_has_no_packer() -> None:
    """The width check in pack_struct_for() is not redundant with the lookup:
    `datatype.var` comes from the structure definition and the length from the
    parse, and a disagreement means the two are describing different fields."""

    class _Datatype:
        type = "u"  # not "struct", so retriever_length() measures the bytes
        var = "u32"

    class _Retriever:
        datatype = _Datatype()

        def get_data_as_bytes(self):
            return b"\x00"  # one byte, while the datatype claims four

    assert pack_struct_for(_Retriever()) is None


# -- 2. containment ----------------------------------------------------------


def test_a_model_with_no_option_edits_writes_the_same_file_as_no_model_at_all(
    tmp_path: Path,
) -> None:
    """The strongest containment check, and the one that fails silently:
    constructing an OptionsEditModel resolves offsets and verifies them, and
    must not change a single byte of the output."""
    loaded = _loaded()
    without = tmp_path / "without.aoe2scenario"
    write_scenario(loaded, without)

    model = OptionsEditModel(loaded)
    assert not model.has_edits
    assert model.serialize_patches() == []
    with_model = tmp_path / "with.aoe2scenario"
    write_scenario(loaded, with_model, options=model)

    assert with_model.read_bytes() == without.read_bytes()
    assert _written_body(without) == loaded.decompressed_body


def test_an_option_set_back_to_its_stored_value_leaves_the_document_clean(
    tmp_path: Path,
) -> None:
    """Flipping a flag and flipping it back must return to the verbatim path,
    not merely produce identical bytes through the patch path -- otherwise a
    fully-undone document still reads as edited everywhere else."""
    loaded = _loaded()
    spec = _a_flag_spec(loaded)
    stored = option_fields.current_value(loaded, spec)

    model = OptionsEditModel(loaded)
    model.set_value(spec.field_id, 1 - stored)
    assert model.has_edits
    model.set_value(spec.field_id, stored)
    assert not model.has_edits
    assert model.serialize_patches() == []


# -- 3. locality -------------------------------------------------------------


def test_one_option_edit_changes_exactly_that_fields_bytes(tmp_path: Path) -> None:
    loaded = _loaded()
    spec = _a_flag_spec(loaded)
    stored = option_fields.current_value(loaded, spec)
    offsets, _ = field_offsets(loaded, option_fields.specs_for(loaded))
    expected = offsets[spec]

    base = tmp_path / "base.aoe2scenario"
    write_scenario(loaded, base)

    model = OptionsEditModel(loaded)
    model.set_value(spec.field_id, 1 - stored)
    edited = tmp_path / "edited.aoe2scenario"
    write_scenario(loaded, edited, options=model)

    ranges = _differing_ranges(_written_body(base), _written_body(edited))
    assert ranges == [(expected.offset, expected.offset + expected.length)]


def test_a_victory_condition_edit_changes_exactly_that_fields_bytes(tmp_path: Path) -> None:
    """The Global Victory group's acceptance gate: victory_condition is a COMBO
    over GlobalVictory.mode, not a checkbox, and the section sits well before
    Units -- this is the shallowest possible offset, but it goes through the
    same body-level diff as every other field, not a hand-derived byte offset.
    """
    loaded = _loaded()
    spec = next(s for s in option_fields.specs_for(loaded) if s.field_id == "victory_condition")
    stored = option_fields.current_value(loaded, spec)
    new_value = next(choice for choice, _label in spec.choices if choice != stored)
    offsets, _ = field_offsets(loaded, option_fields.specs_for(loaded))
    expected = offsets[spec]

    base = tmp_path / "base.aoe2scenario"
    write_scenario(loaded, base)

    model = OptionsEditModel(loaded)
    model.set_value(spec.field_id, new_value)
    edited = tmp_path / "edited.aoe2scenario"
    write_scenario(loaded, edited, options=model)

    # Containment, not equality: every VictoryCondition value is < 256, so a
    # u32 field only ever changes its low byte here -- a real diff range
    # narrower than the field's full 4-byte span, not a bug in the walk.
    # What matters is that no byte outside this field's own span moved.
    ranges = _differing_ranges(_written_body(base), _written_body(edited))
    assert ranges, "no bytes changed"
    for start, end in ranges:
        assert expected.offset <= start and end <= expected.offset + expected.length, (
            f"changed range ({start}, {end}) escapes victory_condition's own "
            f"span ({expected.offset}, {expected.offset + expected.length})"
        )


def test_two_option_edits_change_exactly_two_fields_bytes(tmp_path: Path) -> None:
    """Patches are independent: each is fixed-width, so neither shifts the
    other's offset."""
    loaded = _loaded()
    specs = [
        spec
        for spec in option_fields.specs_for(loaded)
        if spec.kind == option_fields.CHECKBOX
        and spec.section != "Triggers"
        and option_fields.current_value(loaded, spec) in (0, 1)
    ][:2]
    assert len(specs) == 2
    offsets, _ = field_offsets(loaded, option_fields.specs_for(loaded))

    base = tmp_path / "base.aoe2scenario"
    write_scenario(loaded, base)

    model = OptionsEditModel(loaded)
    for spec in specs:
        model.set_value(spec.field_id, 1 - option_fields.current_value(loaded, spec))
    edited = tmp_path / "edited.aoe2scenario"
    write_scenario(loaded, edited, options=model)

    # Byte sets, not ranges: two one-byte flags in the same section can be
    # adjacent, and _differing_ranges() would then merge them into one span
    # that no per-field expectation matches.
    expected = {
        offset
        for spec in specs
        for offset in range(offsets[spec].offset, offsets[spec].offset + offsets[spec].length)
    }
    changed = {
        offset
        for start, end in _differing_ranges(_written_body(base), _written_body(edited))
        for offset in range(start, end)
    }
    assert changed == expected


# -- 4. exec-order, the row that does not take the options path ---------------


def test_flipping_only_exec_order_changes_exactly_one_byte(tmp_path: Path) -> None:
    """The gate the whole exec-order wiring exists for.

    legacy_exec_order dirties no trigger blob and changes no structure, so
    TriggerEditModel.has_edits has to name it explicitly. Without that term
    write_scenario() takes the verbatim-splice branch, writes the original byte
    back, and reports nothing -- so the failure this catches is *zero* changed
    bytes, not a wrong one. Asserted as an exact range list for that reason.
    """
    base_loaded = _loaded()
    base = tmp_path / "base.aoe2scenario"
    write_scenario(base_loaded, base)

    loaded = _loaded()
    model = TriggerEditModel(loaded)
    assert model.exec_order_supported
    stored = model.exec_order
    assert not model.has_edits
    model.set_exec_order(1 - stored)
    assert model.has_edits, "flipping exec-order left the model reading as clean"

    edited = tmp_path / "edited.aoe2scenario"
    write_scenario(loaded, edited, triggers=model)

    ranges = _differing_ranges(_written_body(base), _written_body(edited))
    assert len(ranges) == 1, ranges
    start, end = ranges[0]
    assert end - start == 1
    assert loaded.units_section_end + model.regions.exec_order_offset == start


def test_exec_order_flipped_and_flipped_back_leaves_the_document_clean(
    tmp_path: Path,
) -> None:
    loaded = _loaded()
    model = TriggerEditModel(loaded)
    stored = model.exec_order
    model.set_exec_order(1 - stored)
    assert model.has_edits
    model.set_exec_order(stored)
    assert not model.has_edits

    base_loaded = _loaded()
    base = tmp_path / "base.aoe2scenario"
    write_scenario(base_loaded, base)
    out = tmp_path / "out.aoe2scenario"
    write_scenario(loaded, out, triggers=model)
    assert out.read_bytes() == base.read_bytes()


def test_exec_order_refuses_a_value_that_is_not_a_flag() -> None:
    model = TriggerEditModel(_loaded())
    with pytest.raises(ValueError):
        model.set_exec_order(2)


def test_exec_order_is_read_from_the_forward_walk_not_the_sections_last_byte() -> None:
    """A file whose trigger version is below 4.5 stores no such byte at all,
    and its section's last byte belongs to `redacted`. The offset therefore has
    to come from the walk; on a file that does store it, the walk and "one
    before the end" agree, and that agreement is what this pins."""
    loaded = _loaded()
    model = TriggerEditModel(loaded)
    offset = model.regions.exec_order_offset
    assert offset is not None
    assert offset == model.regions.section_end - 1
    assert offset >= model.regions.variables_end


def test_exec_order_value_reads_without_building_a_model() -> None:
    """The Triggers-mode status line's read path: answerable from a bare
    parse, module-level like exec_order_write_supported(), and specifically
    NOT routed through TriggerEditModel -- test_trigger_panel.py's
    test_browsing_every_row_saves_byte_identically pins that browsing must
    not build one."""
    loaded = _loaded()
    assert exec_order_value(loaded) == TriggerEditModel(loaded).exec_order


def test_exec_order_value_is_none_before_triggers_are_parsed() -> None:
    loaded = load_map_and_units(FIXTURE_PATH)
    assert "Triggers" not in loaded._scenario.sections
    assert exec_order_value(loaded) is None


# -- 5. both write paths in one save -----------------------------------------


def test_an_option_edit_and_a_trigger_edit_both_land_in_one_save(tmp_path: Path) -> None:
    """The options patch runs before the trigger splice, which is the only one
    of the two that changes a section's length."""
    loaded = _loaded()
    spec = _a_flag_spec(loaded)
    stored = option_fields.current_value(loaded, spec)

    options = OptionsEditModel(loaded)
    options.set_value(spec.field_id, 1 - stored)
    triggers = TriggerEditModel(loaded)
    triggers.set_exec_order(1 - triggers.exec_order)

    out = tmp_path / "both.aoe2scenario"
    write_scenario(loaded, out, triggers=triggers, options=options)

    reloaded = load_map_and_units(out)
    parse_triggers(reloaded)
    assert option_fields.current_value(reloaded, spec) == 1 - stored
    assert TriggerEditModel(reloaded).exec_order == 1 - TriggerEditModel(loaded).exec_order


# -- 6. round trip -----------------------------------------------------------


def test_an_option_edit_reads_back_after_a_reload(tmp_path: Path) -> None:
    loaded = _loaded()
    spec = _a_flag_spec(loaded)
    stored = option_fields.current_value(loaded, spec)

    model = OptionsEditModel(loaded)
    model.set_value(spec.field_id, 1 - stored)
    out = tmp_path / "out.aoe2scenario"
    write_scenario(loaded, out, options=model)

    reloaded = load_map_and_units(out)
    assert option_fields.current_value(reloaded, spec) == 1 - stored
    # And the file is still fully verifiable, i.e. the patch did not desync the
    # offsets from what the parser reads.
    assert options_write_supported(reloaded, option_fields.specs_for(reloaded))


def test_an_exec_order_edit_reads_back_after_a_reload(tmp_path: Path) -> None:
    loaded = _loaded()
    model = TriggerEditModel(loaded)
    stored = model.exec_order
    model.set_exec_order(1 - stored)
    out = tmp_path / "out.aoe2scenario"
    write_scenario(loaded, out, triggers=model)

    reloaded = _loaded(out)
    assert TriggerEditModel(reloaded).exec_order == 1 - stored


# -- 7. the gates ------------------------------------------------------------


def test_set_value_refuses_a_field_this_file_cannot_write() -> None:
    model = OptionsEditModel(_loaded())
    with pytest.raises(KeyError):
        model.set_value("not_a_field", 1)


def test_set_value_refuses_a_value_that_does_not_fit_the_field() -> None:
    """Raises rather than clamping: every caller's widget range is already the
    field's own, so a value that will not pack means the spec and the file's
    datatype disagree."""
    loaded = _loaded()
    model = OptionsEditModel(loaded)
    with pytest.raises(ValueError):
        model.set_value("lock_teams", 999)


def test_an_unwritable_file_refuses_to_build_a_model(monkeypatch) -> None:
    """Fail closed at construction, mirroring TriggerEditModel."""
    loaded = _loaded()
    monkeypatch.setattr(loaded, "terrain_block_offset", -1)
    with pytest.raises(OptionEditsUnavailableError):
        OptionsEditModel(loaded)


def test_the_write_path_re_gates_rather_than_trusting_construction(
    tmp_path: Path, monkeypatch
) -> None:
    """The model already refused to exist for a file that failed the gate, so
    reaching this means something changed underneath it. Same shape the trigger
    path uses -- refuse rather than patch at an offset that no longer holds."""
    loaded = _loaded()
    model = OptionsEditModel(loaded)
    model.set_value("lock_teams", 1 - model.original_value("lock_teams"))
    monkeypatch.setattr(loaded, "terrain_block_offset", -1)
    with pytest.raises(WriteBlockedError):
        write_scenario(loaded, tmp_path / "out.aoe2scenario", options=model)


def test_a_patch_overlapping_the_trigger_counter_is_refused(tmp_path: Path) -> None:
    """Options.number_of_triggers is the 4 bytes ending at options_section_end
    and is written by the trigger splice. No mapped field reaches them -- the
    backward walk subtracts that retriever first -- so this pins the guard
    rather than a live case, by forging a patch that does."""
    loaded = _loaded()
    model = OptionsEditModel(loaded)
    model.set_value("lock_teams", 1 - model.original_value("lock_teams"))
    counter_start = loaded.options_section_end - 4
    model._offsets["lock_teams"] = type(model._offsets["lock_teams"])(counter_start, 1)
    with pytest.raises(WriteBlockedError):
        write_scenario(loaded, tmp_path / "out.aoe2scenario", options=model)


# -- 8. corpus ---------------------------------------------------------------


@pytest.mark.corpus
def test_every_writable_option_repacks_its_stored_bytes_across_the_corpus(
    scenario_path,
) -> None:
    """The version spread the two shipped fixtures cannot give: both are
    1.58/trigger-4.9, so the absent-field and differently-typed branches have
    no default-tier coverage at all."""
    loaded = load_map_and_units(scenario_path)
    parse_triggers(loaded)
    specs = option_fields.specs_for(loaded)
    offsets, _ = field_offsets(loaded, specs)
    for spec, _fo in offsets.items():
        retriever = loaded._scenario.sections[spec.section].retriever_map[spec.retriever]
        packer = pack_struct_for(retriever)
        if packer is None:
            continue  # options_write_supported() already fails this file closed
        assert packer.pack(option_fields.current_value(loaded, spec)) == (
            retriever.get_data_as_bytes()
        ), f"{scenario_path.name}: {spec.field_id}"


@pytest.mark.corpus
def test_a_browsed_options_model_saves_byte_identically_across_the_corpus(
    scenario_path, tmp_path: Path
) -> None:
    loaded = load_map_and_units(scenario_path)
    parse_triggers(loaded)
    if not loaded.terrain_write_supported:
        pytest.skip("terrain block failed verification, so no save path at all")
    if not options_write_supported(loaded, option_fields.specs_for(loaded)):
        pytest.skip("map options fail closed on this file, as intended")

    without = tmp_path / "without.aoe2scenario"
    write_scenario(loaded, without)
    model = OptionsEditModel(loaded)
    with_model = tmp_path / "with.aoe2scenario"
    write_scenario(loaded, with_model, options=model)
    assert with_model.read_bytes() == without.read_bytes()


@pytest.mark.corpus
def test_exec_order_survives_an_unrelated_option_save_across_the_corpus(
    scenario_path, tmp_path: Path
) -> None:
    """The round-trip check the trigger-reordering item asked for before
    anything started writing this flag: six corpus files ship with it set to 1,
    and a save that never touches it must leave it there."""
    loaded = load_map_and_units(scenario_path)
    parse_triggers(loaded)
    if not loaded.terrain_write_supported:
        pytest.skip("terrain block failed verification, so no save path at all")
    section = loaded._scenario.sections.get("Triggers")
    if section is None or "legacy_exec_order" not in section.retriever_map:
        pytest.skip("this file stores no trigger execution-order flag")
    before = section.retriever_map["legacy_exec_order"].data

    out = tmp_path / "out.aoe2scenario"
    write_scenario(loaded, out)
    reloaded = load_map_and_units(out)
    parse_triggers(reloaded)
    after = reloaded._scenario.sections["Triggers"].retriever_map["legacy_exec_order"].data
    assert after == before, scenario_path.name
