"""TASK-238: the colour mood write path. OptionsEditModel's map_color_mood
entry, spliced with the water definition by scenario_write._patch_map_strings()
under its own gate. The splice's order and the resize are pinned in
test_water_definition_write_path.py, whose every-splice and resize tests carry
a pending mood too."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from AoE2ScenarioParser.exceptions.asp_exceptions import EndOfFileError
from test_map_strings import TRIGGER_FIXTURE, _item, _variant

from descape import colour_mood, map_strings, option_fields, scenario_write
from descape.options_model import (
    COLOUR_MOOD_FIELD_ID,
    SCRIPT_NAME_FIELD_ID,
    WATER_DEFINITION_FIELD_ID,
    OptionEditsUnavailableError,
    OptionsEditModel,
)
from descape.scenario_io import UnsupportedStructureVersion, load_map_and_units
from descape.scenario_write import WriteBlockedError, build_patched, write_scenario

MOOD = COLOUR_MOOD_FIELD_ID
WATER = WATER_DEFINITION_FIELD_ID


def _save(loaded, out: Path, **kwargs):
    write_scenario(loaded, out, backup=False, **kwargs)
    return load_map_and_units(out)


def _mood(loaded) -> str:
    return map_strings.current_value(loaded, map_strings.COLOUR_MOOD)


def _terrain(loaded) -> list[tuple[int, int]]:
    return [(t.terrain_id, t.elevation) for t in loaded.map_manager.terrain]


def _original_file_bytes(loaded) -> bytes:
    return loaded.header_bytes + loaded.original_compressed_body


def _mood_variant(payload: bytes):
    return _variant(payload, name=map_strings.COLOUR_MOOD)


# -- round trips -----------------------------------------------------------------


def test_empty_to_night_grows_the_body_by_five_at_the_span(tmp_path: Path) -> None:
    loaded = load_map_and_units(TRIGGER_FIXTURE)
    model = OptionsEditModel(loaded)
    model.set_value(MOOD, "Night")

    patched = build_patched(loaded, options=model)
    body = loaded.decompressed_body
    mood = _item(loaded, map_strings.COLOUR_MOOD)
    assert patched.body == body[: mood.offset] + b"\x05\x00Night" + body[mood.offset + mood.length :]
    assert patched.terrain_offset == loaded.terrain_block_offset + 5

    reloaded = _save(loaded, tmp_path / "out.aoe2scenario", options=model)
    assert _mood(reloaded) == "Night"
    assert reloaded.terrain_write_supported and reloaded.units_write_supported
    assert map_strings.write_supported(reloaded, map_strings.COLOUR_MOOD)
    assert _terrain(reloaded) == _terrain(loaded)


@pytest.mark.parametrize(
    ("payload", "after"), [(b"Empty", "Winter"), (b"Default", "Darkness"), ("默认".encode(), "Night")],
    ids=["empty-name", "default", "localized"],
)
def test_an_edit_of_a_stored_name_reads_back(payload, after, tmp_path: Path) -> None:
    loaded = _mood_variant(payload)
    model = OptionsEditModel(loaded)
    model.set_value(MOOD, after)
    patched = build_patched(loaded, options=model)
    assert len(patched.body) == len(loaded.decompressed_body) + len(after) - len(payload)
    reloaded = _save(loaded, tmp_path / "out.aoe2scenario", options=model)
    assert _mood(reloaded) == after
    assert _item(reloaded, map_strings.COLOUR_MOOD).length == 2 + len(after)
    assert _terrain(reloaded) == _terrain(loaded)


@pytest.mark.parametrize("payload", [b"", b"Empty", b"Default", "默认".encode()], ids=["empty", "empty-name", "default", "localized"])
def test_set_and_set_back_saves_byte_identically(payload, tmp_path: Path) -> None:
    loaded = _mood_variant(payload)
    model = OptionsEditModel(loaded)
    original = model.original_value(MOOD)
    model.set_value(MOOD, "Night")
    model.set_value(MOOD, original)
    assert not model.has_edits
    out = tmp_path / "out.aoe2scenario"
    write_scenario(loaded, out, options=model, backup=False)
    assert out.read_bytes() == _original_file_bytes(loaded)


def test_water_mood_and_script_name_read_back_from_one_save(tmp_path: Path) -> None:
    """Three strings, one splice for two of them and GH #216's for the third."""
    loaded = load_map_and_units(TRIGGER_FIXTURE)
    model = OptionsEditModel(loaded)
    model.set_value(WATER, "Red Tide")
    model.set_value(MOOD, "Twilight")
    model.set_value(SCRIPT_NAME_FIELD_ID, "three_strings")
    assert model.serialize_patches() == []
    reloaded = _save(loaded, tmp_path / "out.aoe2scenario", options=model)
    assert map_strings.current_value(reloaded, map_strings.WATER_DEFINITION) == "Red Tide"
    assert _mood(reloaded) == "Twilight"
    assert reloaded._scenario.sections["Map"].retriever_map["script_name"].data == "three_strings"
    assert _terrain(reloaded) == _terrain(loaded)


# -- gates -------------------------------------------------------------------------


def test_a_failed_mood_gate_leaves_water_writable(tmp_path: Path) -> None:
    loaded = _mood_variant(b"X\x00\x00")
    model = OptionsEditModel(loaded)
    assert not model.map_string_supported(MOOD)
    assert model.map_string_supported(WATER)
    with pytest.raises(KeyError):
        model.set_value(MOOD, "Night")
    model.set_value(WATER, "Arctic")
    reloaded = _save(loaded, tmp_path / "out.aoe2scenario", options=model)
    assert map_strings.current_value(reloaded, map_strings.WATER_DEFINITION) == "Arctic"
    assert _mood(reloaded) == "X\x00"


def test_a_failed_water_gate_leaves_the_mood_writable(tmp_path: Path) -> None:
    loaded = _variant(b"X\x00\x00")
    model = OptionsEditModel(loaded)
    assert not model.map_string_supported(WATER)
    model.set_value(MOOD, "Night")
    reloaded = _save(loaded, tmp_path / "out.aoe2scenario", options=model)
    assert _mood(reloaded) == "Night"


@pytest.mark.parametrize(
    ("failed", "refused", "row"),
    [(map_strings.COLOUR_MOOD, MOOD, "colour mood"), (map_strings.WATER_DEFINITION, WATER, "water definition")],
    ids=["mood", "water"],
)
def test_the_write_path_re_gates_each_pending_name_on_its_own(failed, refused, row, tmp_path: Path, monkeypatch) -> None:
    loaded = load_map_and_units(TRIGGER_FIXTURE)
    model = OptionsEditModel(loaded)
    model.set_value(refused, "Arctic")
    monkeypatch.setattr(scenario_write, "map_strings_write_supported", lambda _loaded, name: name != failed)
    with pytest.raises(WriteBlockedError, match=row):
        write_scenario(loaded, tmp_path / "out.aoe2scenario", options=model, backup=False)
    assert not (tmp_path / "out.aoe2scenario").exists()


def test_a_pending_name_with_a_passing_gate_writes_past_the_other_ones_failure(tmp_path: Path, monkeypatch) -> None:
    loaded = load_map_and_units(TRIGGER_FIXTURE)
    model = OptionsEditModel(loaded)
    model.set_value(WATER, "Arctic")
    monkeypatch.setattr(scenario_write, "map_strings_write_supported", lambda _loaded, name: name != map_strings.COLOUR_MOOD)
    write_scenario(loaded, tmp_path / "out.aoe2scenario", options=model, backup=False)


# -- corpus ------------------------------------------------------------------------


def _load_or_skip(path: Path):
    """Only files DEscape can open count, and a file with the field."""
    try:
        loaded = load_map_and_units(path)
    except (UnsupportedStructureVersion, EndOfFileError, ValueError) as e:
        pytest.skip(f"{path.name} does not load ({type(e).__name__})")
    if map_strings.COLOUR_MOOD not in loaded._scenario.sections["Map"].retriever_map:
        pytest.skip(f"{path.name} has no map_color_mood")
    return loaded


@pytest.mark.corpus
def test_the_mood_gate_passes_on_every_corpus_file(scenario_path) -> None:
    assert map_strings.write_supported(_load_or_skip(scenario_path), map_strings.COLOUR_MOOD), scenario_path.name


@pytest.mark.corpus
def test_a_night_edit_reads_back_on_every_corpus_file(scenario_path, tmp_path: Path) -> None:
    loaded = _load_or_skip(scenario_path)
    if not loaded.terrain_write_supported:
        pytest.skip(f"{scenario_path.name}: the terrain gate fails")
    try:
        model = OptionsEditModel(loaded)
    except OptionEditsUnavailableError as e:
        pytest.skip(f"{scenario_path.name}: the scalar gate fails, so no model: {e}")
    target = "Night" if model.original_value(MOOD) != "Night" else "Winter"
    model.set_value(MOOD, target)
    patched = build_patched(loaded, options=model)
    mm = loaded.map_manager
    size = loaded.terrain_struct_size * mm.map_width * mm.map_height
    original = loaded.decompressed_body[loaded.terrain_block_offset : loaded.terrain_block_offset + size]
    assert patched.body[patched.terrain_offset : patched.terrain_offset + size] == original
    reloaded = _save(loaded, tmp_path / "out.aoe2scenario", options=model)
    assert _mood(reloaded) == target
    assert _terrain(reloaded) == _terrain(loaded)


@pytest.mark.corpus
def test_the_mood_table_matches_the_install() -> None:
    """COLOUR_MOODS and MOOD_PARAMS are hardcoded; re-measured here."""
    install = os.environ.get("AOE2DE_INSTALL_PATH")
    if not install:
        pytest.skip("no AoE2:DE install visible: set AOE2DE_INSTALL_PATH")
    path = Path(install) / "resources" / "_common" / "terrain" / "colorcorrection_json" / "colorcorrection.json"
    if not path.is_file():
        pytest.skip(f"no {path}")
    moods = json.loads(path.read_text(encoding="utf-8-sig"))["correction_types"]
    assert tuple(mood["name"] for mood in moods) == option_fields.COLOUR_MOODS
    for mood in moods:
        params = colour_mood.MOOD_PARAMS[mood["name"]]
        assert params.brightness == mood["brightness"], mood["name"]
        assert list(params.terrain_tint_color) == mood["terrain_tint_color"], mood["name"]
        assert list(params.tint_color) == mood["tint_color"], mood["name"]
        assert params.tint_factor == mood["tint_factor"], mood["name"]
