"""GH #216: the Script filename write path. OptionsEditModel's script_name
entry plus scenario_write's two splices: Files.script_file_path (right after
_assemble_body(), translated by its shift) and Map.script_name (after the
downstream splices, before the disables).

The oracle is tests/fixtures/probe216_xs.aoe2scenario, saved by the 1.59
editor with Script Filename "probe" and a real probe.xs present: Map stores
`probe` with no NUL, Files stores `probe.xs` plus one NUL, and nothing is
embedded."""

from __future__ import annotations

from pathlib import Path

import pytest

from descape import ai_scripts, disables_fields, options_model, scenario_write
from descape import player_fields as pf
from descape.messages_model import MessagesEditModel
from descape.options_model import (
    SCRIPT_NAME_FIELD_ID,
    SCRIPT_NAME_MAX_BYTES,
    OptionsEditModel,
    script_name_layout,
    script_name_write_supported,
)
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units, parse_triggers, xs_attachment
from descape.scenario_write import WriteBlockedError, write_scenario
from descape.trigger_model import TriggerEditModel
from descape.unit_model import UnitEditModel

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURE_PATH = FIXTURES / "triggers_120x120.aoe2scenario"
REFERENCE_PATH = FIXTURES / "probe216_xs.aoe2scenario"


def _save(loaded, out: Path, **kwargs):
    write_scenario(loaded, out, backup=False, **kwargs)
    reloaded = load_map_and_units(out)
    parse_triggers(reloaded)
    return reloaded


def _spans(loaded) -> tuple[bytes, bytes]:
    layout = script_name_layout(loaded)
    assert layout is not None
    body = loaded.decompressed_body
    return tuple(body[s.offset : s.offset + s.length] for s in (layout.name, layout.path))


def _script_file_path(loaded) -> str:
    return loaded._scenario.sections["Files"].retriever_map["script_file_path"].data


# -- gate --------------------------------------------------------------------


@pytest.mark.parametrize("path", [BLANK_TEMPLATE_PATH, FIXTURE_PATH, REFERENCE_PATH], ids=["blank", "triggers", "probe"])
def test_the_gate_verifies_on_real_files(path) -> None:
    assert script_name_write_supported(load_map_and_units(path)) is True


def test_the_reference_file_stores_what_the_editor_wrote() -> None:
    loaded = load_map_and_units(REFERENCE_PATH)
    assert _spans(loaded) == (b"\x05\x00probe", b"\x09\x00probe.xs\x00")
    assert xs_attachment(loaded) == ("probe", 0)
    assert _script_file_path(loaded) == "probe.xs"


def test_the_gate_fails_without_the_map_retriever() -> None:
    """The pre-1.40 shape: no Map.script_name at all."""
    loaded = load_map_and_units(FIXTURE_PATH)
    del loaded._scenario.sections["Map"].retriever_map["script_name"]
    assert script_name_write_supported(loaded) is False
    model = OptionsEditModel(loaded)
    assert not model.script_name_supported
    with pytest.raises(KeyError):
        model.set_value(SCRIPT_NAME_FIELD_ID, "probe")


def test_the_gate_fails_when_files_does_not_verify(monkeypatch) -> None:
    loaded = load_map_and_units(FIXTURE_PATH)
    monkeypatch.setattr(pf, "files_layout", lambda _loaded: None)
    assert script_name_write_supported(loaded) is False


def test_the_gate_fails_when_the_map_bytes_disagree() -> None:
    """A parsed name that is not what the walked span holds (here: forged)."""
    loaded = load_map_and_units(FIXTURE_PATH)
    loaded._scenario.sections["Map"].retriever_map["script_name"].data = "x"
    assert script_name_write_supported(loaded) is False


def test_the_gate_fails_when_files_embeds_a_script() -> None:
    """A new name would leave the old script's text under it (no corpus file
    embeds one, so what the game does then is unknown)."""
    loaded = load_map_and_units(REFERENCE_PATH)
    parse_triggers(loaded)
    loaded._scenario.sections["Files"].retriever_map["script_file_content"].data = "void main() {}"
    assert script_name_write_supported(loaded) is False


def test_the_gate_does_not_run_at_model_construction() -> None:
    """It forces the trigger parse, which a first scalar edit must not pay."""
    loaded = load_map_and_units(FIXTURE_PATH)
    OptionsEditModel(loaded)
    assert "Files" not in loaded._scenario.sections


# -- model -------------------------------------------------------------------


def test_set_and_revert() -> None:
    model = OptionsEditModel(load_map_and_units(FIXTURE_PATH))
    assert model.current_value(SCRIPT_NAME_FIELD_ID) == ""
    model.set_value(SCRIPT_NAME_FIELD_ID, "probe")
    assert model.has_edits and model.has_script_name_edit
    assert model.pending_values() == {SCRIPT_NAME_FIELD_ID: "probe"}
    assert model.serialize_patches() == []
    model.set_value(SCRIPT_NAME_FIELD_ID, "")
    assert not model.has_edits
    assert model.serialize_script_name_resize() is None
    assert model.serialize_script_path_resize() is None


@pytest.mark.parametrize(
    "value",
    [
        "a\x00b",
        "dir/probe",
        "dir\\probe",
        "a:b",
        "a*b",
        'a"b',
        "a<b>",
        "a|b",
        "a?b",
        "a\tb",
        "probe ",
        " probe",
        "x" * (SCRIPT_NAME_MAX_BYTES + 1),
        5,
        None,
    ],
    ids=[
        "nul",
        "slash",
        "backslash",
        "colon",
        "star",
        "quote",
        "angles",
        "pipe",
        "question",
        "control",
        "trailing-space",
        "leading-space",
        "too-long",
        "int",
        "none",
    ],
)
def test_validation_refuses(value) -> None:
    model = OptionsEditModel(load_map_and_units(FIXTURE_PATH))
    with pytest.raises(ValueError):
        model.set_value(SCRIPT_NAME_FIELD_ID, value)
    assert not model.has_edits


def test_the_longest_name_still_fits_the_path() -> None:
    model = OptionsEditModel(load_map_and_units(FIXTURE_PATH))
    model.set_value(SCRIPT_NAME_FIELD_ID, "x" * SCRIPT_NAME_MAX_BYTES)
    _start, _end, path = model.serialize_script_path_resize()
    assert len(path) == 2 + 0x7FFF


# -- round trips -------------------------------------------------------------


def test_a_name_round_trips(tmp_path: Path) -> None:
    loaded = load_map_and_units(FIXTURE_PATH)
    model = OptionsEditModel(loaded)
    model.set_value(SCRIPT_NAME_FIELD_ID, "probe")
    reloaded = _save(loaded, tmp_path / "out.aoe2scenario", options=model)
    assert xs_attachment(reloaded) == ("probe", 0)
    assert _script_file_path(reloaded) == "probe.xs"
    assert script_name_write_supported(reloaded) is True
    assert [t.terrain_id for t in reloaded.map_manager.terrain] == [t.terrain_id for t in loaded.map_manager.terrain]


def test_output_matches_the_editor_save(tmp_path: Path) -> None:
    loaded = load_map_and_units(FIXTURE_PATH)
    model = OptionsEditModel(loaded)
    model.set_value(SCRIPT_NAME_FIELD_ID, "probe")
    reloaded = _save(loaded, tmp_path / "out.aoe2scenario", options=model)
    assert _spans(reloaded) == _spans(load_map_and_units(REFERENCE_PATH))


def test_a_unicode_name_round_trips(tmp_path: Path) -> None:
    loaded = load_map_and_units(FIXTURE_PATH)
    model = OptionsEditModel(loaded)
    model.set_value(SCRIPT_NAME_FIELD_ID, "Lechfeld 奥格斯堡")
    reloaded = _save(loaded, tmp_path / "out.aoe2scenario", options=model)
    assert xs_attachment(reloaded)[0] == "Lechfeld 奥格斯堡"
    assert _script_file_path(reloaded) == "Lechfeld 奥格斯堡.xs"


def test_clearing_the_name_restores_the_unnamed_bytes(tmp_path: Path) -> None:
    """Set then cleared in two saves lands on the original body exactly: an
    empty path is the bare `\\x00\\x00` every unnamed corpus file stores."""
    loaded = load_map_and_units(FIXTURE_PATH)
    model = OptionsEditModel(loaded)
    model.set_value(SCRIPT_NAME_FIELD_ID, "probe")
    named = _save(loaded, tmp_path / "named.aoe2scenario", options=model)
    model = OptionsEditModel(named)
    model.set_value(SCRIPT_NAME_FIELD_ID, "")
    cleared = _save(named, tmp_path / "cleared.aoe2scenario", options=model)
    assert xs_attachment(cleared) == ("", 0)
    assert cleared.decompressed_body == loaded.decompressed_body


def test_clearing_the_editor_save(tmp_path: Path) -> None:
    loaded = load_map_and_units(REFERENCE_PATH)
    model = OptionsEditModel(loaded)
    model.set_value(SCRIPT_NAME_FIELD_ID, "")
    cleared = _save(loaded, tmp_path / "out.aoe2scenario", options=model)
    assert _spans(cleared) == (b"\x00\x00", b"\x00\x00")


@pytest.mark.parametrize("path", [BLANK_TEMPLATE_PATH, FIXTURE_PATH, REFERENCE_PATH], ids=["blank", "triggers", "probe"])
def test_a_browse_only_save_is_byte_identical(path, tmp_path: Path) -> None:
    """A model whose name was set and set back splices nothing."""
    loaded = load_map_and_units(path)
    model = OptionsEditModel(loaded)
    original = model.current_value(SCRIPT_NAME_FIELD_ID)
    model.set_value(SCRIPT_NAME_FIELD_ID, original + "x")
    model.set_value(SCRIPT_NAME_FIELD_ID, original)
    out = tmp_path / "out.aoe2scenario"
    write_scenario(loaded, out, options=model, backup=False)
    assert out.read_bytes() == path.read_bytes()


def test_every_splice_in_one_save_reads_back(tmp_path: Path) -> None:
    """The script splices beside every other resizing step: units and
    triggers (_assemble_body()), the Files library, disables, personality,
    Messages and player_data_1. One on the wrong side of a neighbour reads
    back as garbage or trips the terrain-offset assert."""
    loaded = load_map_and_units(FIXTURE_PATH)
    parse_triggers(loaded)
    triggers = TriggerEditModel(loaded)
    triggers.manager().triggers[0].name = "script name ordering probe"
    triggers.mark_dirty(0)
    units = UnitEditModel(loaded)
    new_ref = units.add(player=2, unit_const=83, x=15.5, y=16.5).reference_id
    options = OptionsEditModel(loaded)
    messages = MessagesEditModel(loaded)
    options.set_value(SCRIPT_NAME_FIELD_ID, "ordering_probe")
    options.set_value(disables_fields.disables_field_id("buildings", 2), (72, 621))
    civ_field = pf.player_field_id("civilization", 3)
    civ_after = next(v for v, _ in pf.civilization_choices(loaded) if v != options.current_value(civ_field))
    options.set_value(civ_field, civ_after)
    lib = ((b"Foo.per\x00", b"(defrule)"),)
    options.set_personality(3, ai_scripts.AiChoice("test:Foo.ai", "Foo.ai", ai_scripts.AI_TYPE_CUSTOM, "test",
                                                   text=b"(defrule)", library=lib, resolved=True))
    messages.set_value("instructions", "Script name ordering probe.")

    reloaded = _save(
        loaded, tmp_path / "out.aoe2scenario", options=options, messages=messages, triggers=triggers, units=units
    )
    assert xs_attachment(reloaded) == ("ordering_probe", 0)
    assert _script_file_path(reloaded) == "ordering_probe.xs"
    assert parse_triggers(reloaded).triggers[0].name == "script name ordering probe"
    got_new = next(u for u in reloaded.unit_manager.units[2] if u.reference_id == new_ref)
    assert (got_new.x, got_new.y, got_new.unit_const) == (15.5, 16.5, 83)
    assert disables_fields.current_ids(reloaded, "buildings", 2) == (72, 621)
    assert MessagesEditModel(reloaded).current_value("instructions") == "Script name ordering probe."
    assert OptionsEditModel(reloaded).current_value(civ_field) == civ_after
    assert (b"Foo.per\x00", b"(defrule)") in [
        (e.name, pf.ai_library_entry_text(reloaded.decompressed_body, e)) for e in pf.ai_library_layout(reloaded).entries
    ]
    assert [t.terrain_id for t in reloaded.map_manager.terrain] == [t.terrain_id for t in loaded.map_manager.terrain]


# -- fail-closed ---------------------------------------------------------------


def test_the_write_path_re_gates(tmp_path: Path, monkeypatch) -> None:
    loaded = load_map_and_units(FIXTURE_PATH)
    model = OptionsEditModel(loaded)
    model.set_value(SCRIPT_NAME_FIELD_ID, "probe")
    monkeypatch.setattr(scenario_write, "script_name_write_supported", lambda *_: False)
    with pytest.raises(WriteBlockedError, match="script filename"):
        write_scenario(loaded, tmp_path / "out.aoe2scenario", options=model, backup=False)


def test_a_name_splice_outside_map_is_refused(tmp_path: Path, monkeypatch) -> None:
    loaded = load_map_and_units(FIXTURE_PATH)
    model = OptionsEditModel(loaded)
    model.set_value(SCRIPT_NAME_FIELD_ID, "probe")
    _start, end, replacement = model.serialize_script_name_resize()
    monkeypatch.setattr(model, "serialize_script_name_resize", lambda: (0, end, replacement))
    with pytest.raises(WriteBlockedError, match="Map region"):
        write_scenario(loaded, tmp_path / "out.aoe2scenario", options=model, backup=False)


def test_a_path_splice_outside_files_is_refused(tmp_path: Path, monkeypatch) -> None:
    loaded = load_map_and_units(FIXTURE_PATH)
    model = OptionsEditModel(loaded)
    model.set_value(SCRIPT_NAME_FIELD_ID, "probe")
    _start, end, replacement = model.serialize_script_path_resize()
    monkeypatch.setattr(model, "serialize_script_path_resize", lambda: (loaded.files_section_start - 1, end, replacement))
    with pytest.raises(WriteBlockedError, match="Files region"):
        write_scenario(loaded, tmp_path / "out.aoe2scenario", options=model, backup=False)


def test_the_layout_is_read_through_the_module(monkeypatch) -> None:
    """The model resolves its gate through options_model.script_name_layout(),
    so a failing layout leaves the row unwritable rather than half-set."""
    loaded = load_map_and_units(FIXTURE_PATH)
    monkeypatch.setattr(options_model, "script_name_layout", lambda _loaded: None)
    model = OptionsEditModel(loaded)
    with pytest.raises(KeyError):
        model.set_value(SCRIPT_NAME_FIELD_ID, "probe")


# -- info panel ----------------------------------------------------------------


def test_the_info_line_shows_a_pending_name() -> None:
    from descape.viewer import _xs_info_lines

    loaded = load_map_and_units(FIXTURE_PATH)
    assert _xs_info_lines(loaded)[0] == "XS script file: (none)"
    assert _xs_info_lines(loaded, "probe")[0] == "XS script file: probe"
    assert _xs_info_lines(load_map_and_units(REFERENCE_PATH), "")[0] == "XS script file: (none)"
