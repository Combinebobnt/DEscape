"""TASK-333: the water definition write path. OptionsEditModel's
water_definition entry plus scenario_write._patch_map_strings(), which
splices the Map string front (first marker through map_color_mood) after
_patch_script_name() and before _patch_disables().

Every shipped fixture stores an empty water_definition; the stored-name
shapes come from test_map_strings._variant(), a fixture body with the raw
payload swapped and reloaded through the real loader."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from AoE2ScenarioParser.exceptions.asp_exceptions import EndOfFileError
from test_map_strings import TRIGGER_FIXTURE, _item, _variant

from descape import ai_scripts, disables_fields, map_strings, option_fields, player_fields, scenario_write
from descape.messages_model import MessagesEditModel
from descape.options_model import (
    SCRIPT_NAME_FIELD_ID,
    WATER_DEFINITION_FIELD_ID,
    OptionEditsUnavailableError,
    OptionsEditModel,
    map_strings_write_supported,
)
from descape.scenario_io import (
    BLANK_TEMPLATE_PATH,
    UnsupportedStructureVersion,
    load_map_and_units,
    parse_triggers,
    xs_attachment,
)
from descape.scenario_resize import Anchor, resize_scenario
from descape.scenario_write import WriteBlockedError, build_patched, write_scenario
from descape.trigger_model import TriggerEditModel
from descape.unit_model import UnitEditModel

WATER = WATER_DEFINITION_FIELD_ID


def _save(loaded, out: Path, **kwargs):
    write_scenario(loaded, out, backup=False, **kwargs)
    return load_map_and_units(out)


def _water(loaded) -> str:
    return map_strings.current_value(loaded, map_strings.WATER_DEFINITION)


def _terrain(loaded) -> list[tuple[int, int]]:
    return [(t.terrain_id, t.elevation) for t in loaded.map_manager.terrain]


def _original_file_bytes(loaded) -> bytes:
    return loaded.header_bytes + loaded.original_compressed_body


# -- model ---------------------------------------------------------------------


def test_the_model_seeds_the_files_own_value() -> None:
    model = OptionsEditModel(load_map_and_units(TRIGGER_FIXTURE))
    assert model.map_strings_supported
    assert model.original_value(WATER) == ""
    assert not model.has_map_string_edits


def test_a_nul_variant_seeds_the_decoded_name() -> None:
    model = OptionsEditModel(_variant(b"Preset_Main\x00"))
    assert model.original_value(WATER) == "Preset_Main"


def test_the_file_value_is_accepted_and_writes_nothing() -> None:
    """Selecting the decoded name of a NUL-trailing file is the file's own
    value, so it keeps its NUL rather than being rewritten without it."""
    model = OptionsEditModel(_variant(b"Preset_Main\x00"))
    model.set_value(WATER, "Preset_Main")
    assert not model.has_edits
    assert model.serialize_map_strings_resize() is None


def test_a_water_edit_never_reaches_serialize_patches() -> None:
    model = OptionsEditModel(load_map_and_units(TRIGGER_FIXTURE))
    model.set_value(WATER, "Arctic")
    assert model.pending_values() == {WATER: "Arctic"}
    assert model.serialize_patches() == []
    assert WATER not in {s.field_id for s in model.specs}


@pytest.mark.parametrize(
    "value", [None, 5, b"Arctic", "a\x00b", "x" * (map_strings.STR16_MAX_PAYLOAD + 1)], ids=["none", "int", "bytes", "nul", "too-long"]
)
def test_validation_refuses(value) -> None:
    model = OptionsEditModel(load_map_and_units(TRIGGER_FIXTURE))
    with pytest.raises(ValueError):
        model.set_value(WATER, value)
    assert not model.has_edits


def test_any_storable_string_is_accepted_by_the_model() -> None:
    """The panel offers the list; the model stays a codec."""
    model = OptionsEditModel(load_map_and_units(TRIGGER_FIXTURE))
    model.set_value(WATER, "A data mod's water")
    assert model.current_value(WATER) == "A data mod's water"


def test_a_failed_gate_leaves_the_row_unwritable_and_the_scalars_writable() -> None:
    model = OptionsEditModel(_variant(b"X\x00\x00"))
    assert not model.map_strings_supported
    with pytest.raises(KeyError):
        model.set_value(WATER, "Arctic")
    model.set_value("collide_and_correct", 1 - model.original_value("collide_and_correct"))
    assert model.has_edits


# -- round trips -----------------------------------------------------------------


def test_empty_to_arctic_grows_the_body_by_six_at_the_span(tmp_path: Path) -> None:
    loaded = load_map_and_units(TRIGGER_FIXTURE)
    model = OptionsEditModel(loaded)
    model.set_value(WATER, "Arctic")

    patched = build_patched(loaded, options=model)
    body = loaded.decompressed_body
    water = _item(loaded, map_strings.WATER_DEFINITION)
    assert patched.body == body[: water.offset] + b"\x06\x00Arctic" + body[water.offset + water.length :]
    assert patched.terrain_offset == loaded.terrain_block_offset + 6

    reloaded = _save(loaded, tmp_path / "out.aoe2scenario", options=model)
    assert _water(reloaded) == "Arctic"
    assert reloaded.terrain_write_supported and reloaded.units_write_supported
    assert parse_triggers(reloaded) is not None and reloaded.trigger_write_supported
    assert map_strings_write_supported(reloaded)
    assert _terrain(reloaded) == _terrain(loaded)


@pytest.mark.parametrize(
    ("payload", "after", "delta"),
    [(b"Default", "Morning", 0), (b"Preset_Main\x00", "Noon", -8), (b"", "Red Tide", 8)],
    ids=["same-length", "shrinking", "with-a-space"],
)
def test_edits_of_every_length_read_back(payload, after, delta, tmp_path: Path) -> None:
    loaded = _variant(payload)
    model = OptionsEditModel(loaded)
    model.set_value(WATER, after)
    patched = build_patched(loaded, options=model)
    assert len(patched.body) == len(loaded.decompressed_body) + delta
    assert patched.terrain_offset == loaded.terrain_block_offset + delta
    reloaded = _save(loaded, tmp_path / "out.aoe2scenario", options=model)
    assert _water(reloaded) == after
    assert _item(reloaded, map_strings.WATER_DEFINITION).length == 2 + len(after.encode())
    assert _terrain(reloaded) == _terrain(loaded)
    assert OptionsEditModel(reloaded).original_value(WATER) == after


@pytest.mark.parametrize("payload", [b"", b"Preset_Main\x00"], ids=["empty", "preset-main-nul"])
def test_set_and_set_back_saves_byte_identically(payload, tmp_path: Path) -> None:
    loaded = _variant(payload)
    model = OptionsEditModel(loaded)
    original = model.original_value(WATER)
    model.set_value(WATER, "Arctic")
    model.set_value(WATER, original)
    assert not model.has_edits
    out = tmp_path / "out.aoe2scenario"
    write_scenario(loaded, out, options=model, backup=False)
    assert out.read_bytes() == _original_file_bytes(loaded)


def test_a_browse_only_save_of_the_nul_variant_is_byte_identical(tmp_path: Path) -> None:
    loaded = _variant(b"Preset_Main\x00")
    out = tmp_path / "out.aoe2scenario"
    write_scenario(loaded, out, options=OptionsEditModel(loaded), backup=False)
    assert out.read_bytes() == _original_file_bytes(loaded)


def test_every_splice_in_one_save_reads_back(tmp_path: Path) -> None:
    """The water splice beside every other resizing step, the script name
    splice just above it included, plus scalars and a per-player Map target
    that sit after it and shift with it. A splice on the wrong side of a
    neighbour reads back as garbage or trips an assert."""
    loaded = load_map_and_units(TRIGGER_FIXTURE)
    parse_triggers(loaded)
    triggers = TriggerEditModel(loaded)
    triggers.manager().triggers[0].name = "water ordering probe"
    triggers.mark_dirty(0)
    units = UnitEditModel(loaded)
    new_ref = units.add(player=2, unit_const=83, x=15.5, y=16.5).reference_id
    options = OptionsEditModel(loaded)
    messages = MessagesEditModel(loaded)

    options.set_value(WATER, "Red Tide")
    options.set_value(SCRIPT_NAME_FIELD_ID, "ordering_probe")
    flag = 1 - options.original_value("collide_and_correct")
    options.set_value("collide_and_correct", flag)
    pop_field = player_fields.player_field_id("pop_limit", 2)
    pop_after = options.original_value(pop_field) + 25
    options.set_value(pop_field, pop_after)
    options.set_value(disables_fields.disables_field_id("buildings", 2), (72, 621))
    civ_field = player_fields.player_field_id("civilization", 3)
    civ_after = next(v for v, _ in player_fields.civilization_choices(loaded) if v != options.current_value(civ_field))
    options.set_value(civ_field, civ_after)
    lib = ((b"Foo.per\x00", b"(defrule)"),)
    options.set_personality(3, ai_scripts.AiChoice("test:Foo.ai", "Foo.ai", ai_scripts.AI_TYPE_CUSTOM, "test",
                                                   text=b"(defrule)", library=lib, resolved=True))
    messages.set_value("instructions", "Water ordering probe.")

    reloaded = _save(
        loaded, tmp_path / "out.aoe2scenario", options=options, messages=messages, triggers=triggers, units=units
    )
    parse_triggers(reloaded)
    after = OptionsEditModel(reloaded)
    assert _water(reloaded) == "Red Tide"
    assert xs_attachment(reloaded) == ("ordering_probe", 0)
    assert after.original_value("collide_and_correct") == flag
    assert after.original_value(pop_field) == pop_after
    assert after.original_value(civ_field) == civ_after
    assert disables_fields.current_ids(reloaded, "buildings", 2) == (72, 621)
    assert MessagesEditModel(reloaded).current_value("instructions") == "Water ordering probe."
    assert parse_triggers(reloaded).triggers[0].name == "water ordering probe"
    got_new = next(u for u in reloaded.unit_manager.units[2] if u.reference_id == new_ref)
    assert (got_new.x, got_new.y, got_new.unit_const) == (15.5, 16.5, 83)
    assert _terrain(reloaded) == _terrain(loaded)
    assert map_strings_write_supported(reloaded)


def test_a_map_resize_carries_a_pending_water_edit_and_clamps_cameras(tmp_path: Path) -> None:
    """The shrink clamps a camera that sits after the water splice, so a
    wrong terrain-offset shift would clamp the wrong bytes."""
    from test_scenario_resize import _with_camera

    loaded = _with_camera(110)
    options = OptionsEditModel(loaded)
    options.set_value(WATER, "Arctic")
    result = resize_scenario(loaded, 100, 100, Anchor.TOP_LEFT, options=options)
    reloaded = _save(result.loaded, tmp_path / "resized.aoe2scenario", units=result.unit_edits, triggers=result.trigger_edits)
    assert (reloaded.map_manager.map_width, reloaded.map_manager.map_height) == (100, 100)
    assert _water(reloaded) == "Arctic"
    fields = player_fields.camera_fields(reloaded)
    pov_x = [f.value for f in fields if f.target.codec == "s32" and f.axis == "x"]
    assert pov_x[1] == 99
    assert all(f.value < 100 for f in fields), [(f.axis, f.value) for f in fields]


# -- fail-closed -------------------------------------------------------------------


def test_the_write_path_re_gates(tmp_path: Path, monkeypatch) -> None:
    loaded = load_map_and_units(TRIGGER_FIXTURE)
    model = OptionsEditModel(loaded)
    model.set_value(WATER, "Arctic")
    monkeypatch.setattr(scenario_write, "map_strings_write_supported", lambda *_: False)
    with pytest.raises(WriteBlockedError, match="water definition"):
        write_scenario(loaded, tmp_path / "out.aoe2scenario", options=model, backup=False)
    assert not (tmp_path / "out.aoe2scenario").exists()


def test_a_splice_outside_map_is_refused(tmp_path: Path, monkeypatch) -> None:
    loaded = load_map_and_units(TRIGGER_FIXTURE)
    model = OptionsEditModel(loaded)
    model.set_value(WATER, "Arctic")
    _start, end, replacement = model.serialize_map_strings_resize()
    monkeypatch.setattr(model, "serialize_map_strings_resize", lambda: (loaded.options_section_end - 1, end, replacement))
    with pytest.raises(WriteBlockedError, match="Map region"):
        write_scenario(loaded, tmp_path / "out.aoe2scenario", options=model, backup=False)


# -- corpus ------------------------------------------------------------------------


def _load_or_skip(path: Path):
    """Only files DEscape can open count: an unsupported version or a library parse failure skips."""
    try:
        loaded = load_map_and_units(path)
    except (UnsupportedStructureVersion, EndOfFileError, ValueError) as e:
        pytest.skip(f"{path.name} does not load ({type(e).__name__})")
    if map_strings.WATER_DEFINITION not in loaded._scenario.sections["Map"].retriever_map:
        pytest.skip(f"{path.name} has no water_definition")
    return loaded


def _model_or_skip(loaded, path: Path) -> OptionsEditModel:
    try:
        return OptionsEditModel(loaded)
    except OptionEditsUnavailableError as e:
        pytest.skip(f"{path.name}: the scalar gate fails, so no model: {e}")


@pytest.mark.corpus
def test_the_gate_passes_on_every_corpus_file(scenario_path) -> None:
    loaded = _load_or_skip(scenario_path)
    assert map_strings.write_supported(loaded), scenario_path.name
    spec = next(s for s in option_fields.specs_for(loaded) if s.field_id == WATER)
    model = _model_or_skip(loaded, scenario_path)
    assert option_fields.current_value(loaded, spec) == model.original_value(WATER)


@pytest.mark.corpus
def test_a_browsed_corpus_file_saves_byte_identically(scenario_path, tmp_path: Path) -> None:
    loaded = _load_or_skip(scenario_path)
    model = _model_or_skip(loaded, scenario_path)
    model.set_value(WATER, "Arctic" if model.original_value(WATER) != "Arctic" else "Noon")
    model.set_value(WATER, model.original_value(WATER))
    out = tmp_path / "out.aoe2scenario"
    write_scenario(loaded, out, options=model, backup=False)
    assert out.read_bytes() == scenario_path.read_bytes()


@pytest.mark.corpus
def test_an_arctic_edit_reads_back_on_every_corpus_file(scenario_path, tmp_path: Path) -> None:
    loaded = _load_or_skip(scenario_path)
    if not loaded.terrain_write_supported:
        pytest.skip(f"{scenario_path.name}: the terrain gate fails")
    model = _model_or_skip(loaded, scenario_path)
    target = "Arctic" if model.original_value(WATER) != "Arctic" else "Noon"
    model.set_value(WATER, target)
    patched = build_patched(loaded, options=model)
    mm = loaded.map_manager
    size = loaded.terrain_struct_size * mm.map_width * mm.map_height
    original = loaded.decompressed_body[loaded.terrain_block_offset : loaded.terrain_block_offset + size]
    assert patched.body[patched.terrain_offset : patched.terrain_offset + size] == original
    reloaded = _save(loaded, tmp_path / "out.aoe2scenario", options=model)
    assert _water(reloaded) == target
    assert _terrain(reloaded) == _terrain(loaded)


@pytest.mark.corpus
def test_the_water_list_matches_the_install() -> None:
    """WATER_DEFINITIONS is hardcoded in the game's order; re-measured here."""
    install = os.environ.get("AOE2DE_INSTALL_PATH")
    if not install:
        pytest.skip("no AoE2:DE install visible: set AOE2DE_INSTALL_PATH")
    path = Path(install) / "resources" / "_common" / "terrain" / "water_json" / "water_def.json"
    if not path.is_file():
        pytest.skip(f"no {path}")
    names = [entry["water_type_name"] for entry in json.loads(path.read_text(encoding="utf-8-sig"))["water_types"]]
    assert tuple(names) == option_fields.WATER_DEFINITIONS


def test_the_spec_offers_the_21_names_as_stored() -> None:
    """Default tier: the spec offers the 21 names as (name, name) pairs."""
    loaded = load_map_and_units(BLANK_TEMPLATE_PATH)
    spec = next(s for s in option_fields.specs_for(loaded) if s.field_id == WATER)
    assert spec.choices == tuple((n, n) for n in option_fields.WATER_DEFINITIONS)
    assert len(spec.choices) == 21
