"""Messages mode's write path: descape/messages_model.py's serialize()/
header_patch() plus the messages branches in descape/scenario_write.py.

The load-bearing claim: Messages is the first section this codebase splices
that sits *upstream* of Units/Triggers (section 3 of 13), so an edit here has
to shift every downstream byte while every other write-path patch stays at
its original, pre-splice offset. See scenario_write.py's module docstring
for the ordering argument this test suite pins.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from AoE2ScenarioParser.scenarios.aoe2_scenario import _decompress_bytes

from descape import option_fields
from descape.messages_model import MessagesEditModel
from descape.options_model import OptionsEditModel
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units, parse_triggers
from descape.scenario_write import write_scenario
from descape.trigger_model import TriggerEditModel

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"


def _written_body(path: Path) -> bytes:
    raw = path.read_bytes()
    loaded = load_map_and_units(path)
    return _decompress_bytes(raw[len(loaded.header_bytes) :])


def _check_zero_edit_identity(path: Path, tmp_dir: Path, *, with_model: bool) -> tuple[bool, str]:
    s = load_map_and_units(path)
    out = tmp_dir / f"{path.stem}.zero_edit{path.suffix}"
    model = MessagesEditModel(s) if with_model else None
    write_scenario(s, out, messages=model)
    written = out.read_bytes()
    if written[: len(s.header_bytes)] != s.header_bytes:
        return False, "header bytes changed"
    body = _decompress_bytes(written[len(s.header_bytes) :])
    if body != s.decompressed_body:
        n = min(len(body), len(s.decompressed_body))
        i = next((k for k in range(n) if body[k] != s.decompressed_body[k]), n)
        return False, f"body diverged at byte {i} (lens {len(body)} vs {len(s.decompressed_body)})"
    return True, "OK (byte-identical)"


def test_zero_edit_save_with_no_model_is_unchanged(tmp_path: Path) -> None:
    ok, detail = _check_zero_edit_identity(BLANK_TEMPLATE_PATH, tmp_path, with_model=False)
    assert ok, detail


def test_zero_edit_save_with_a_clean_model_is_unchanged(tmp_path: Path) -> None:
    ok, detail = _check_zero_edit_identity(BLANK_TEMPLATE_PATH, tmp_path, with_model=True)
    assert ok, detail


@pytest.mark.corpus
def test_zero_edit_save_with_a_clean_model_is_unchanged_corpus(scenario_path, tmp_path: Path) -> None:
    loaded = load_map_and_units(scenario_path)
    if not loaded.messages_write_supported:
        pytest.skip(f"{scenario_path.name}: messages_write_supported is False")
    ok, detail = _check_zero_edit_identity(scenario_path, tmp_path, with_model=True)
    assert ok, detail


def test_editing_one_field_leaves_every_other_section_unchanged(tmp_path: Path) -> None:
    loaded = load_map_and_units(FIXTURE_PATH)
    parse_triggers(loaded)
    model = MessagesEditModel(loaded)
    model.set_value("hints", "a brand new hint")

    out = tmp_path / "edited.aoe2scenario"
    write_scenario(loaded, out, messages=model)

    reloaded = load_map_and_units(out)
    assert reloaded._scenario.sections["Messages"].retriever_map["ascii_hints"].data == "a brand new hint"
    assert len(reloaded.map_manager.terrain) == len(loaded.map_manager.terrain)
    assert [t.terrain_id for t in reloaded.map_manager.terrain] == [
        t.terrain_id for t in loaded.map_manager.terrain
    ]
    assert reloaded.number_of_unit_sections == loaded.number_of_unit_sections
    reloaded_triggers = parse_triggers(reloaded)
    original_triggers = parse_triggers(loaded)
    assert len(reloaded_triggers.triggers) == len(original_triggers.triggers)
    assert reloaded.messages_write_supported


def _a_flag_spec(loaded):
    for spec in option_fields.specs_for(loaded):
        if spec.kind == option_fields.CHECKBOX and spec.section != "Triggers":
            if option_fields.current_value(loaded, spec) in (0, 1):
                return spec
    raise AssertionError("no writable flag row on this file")


def test_multi_domain_save_lands_all_four_edits(tmp_path: Path) -> None:
    """Pins the splice ordering: a single save carrying a terrain edit, an
    option scalar, a trigger edit, and a Messages field all at once, then
    reload and assert every one landed. A Messages-only save has nothing to
    collide with (any splice order would pass it) -- this is what actually
    exercises "Messages splices last, after Units/Triggers already used
    their pre-splice offsets" (scenario_write.py's module docstring).

    Verified by hand during development, not re-checked by CI: temporarily
    moving the Messages splice ahead of _assemble_body() in
    scenario_write.write_scenario() makes this test fail (the option/
    trigger/terrain patches then land inside what becomes stale offsets
    once Messages' length change is applied first) -- confirming this test
    does pin the ordering rather than passing regardless of it.
    """
    loaded = load_map_and_units(FIXTURE_PATH)
    parse_triggers(loaded)

    # 1. terrain: flip tile 0's terrain id.
    tile = loaded.map_manager.terrain[0]
    new_terrain_id = 15 if tile.terrain_id == 2 else 2
    tile.terrain_id = new_terrain_id

    # 2. a map-option scalar.
    option_spec = _a_flag_spec(loaded)
    options_model = OptionsEditModel(loaded)
    original_flag = options_model.current_value(option_spec.field_id)
    options_model.set_value(option_spec.field_id, 1 - original_flag)

    # 3. a trigger.
    trigger_model = TriggerEditModel(loaded)
    trigger_model.manager().triggers[0].name = "Multi-domain save: trigger 0 (edited)"
    trigger_model.mark_dirty(0)

    # 4. a Messages field.
    messages_model = MessagesEditModel(loaded)
    messages_model.set_value("victory", "multi-domain victory text")

    out = tmp_path / "multi_domain.aoe2scenario"
    write_scenario(
        loaded, out, triggers=trigger_model, options=options_model, units=None, messages=messages_model
    )

    reloaded = load_map_and_units(out)
    assert reloaded.map_manager.terrain[0].terrain_id == new_terrain_id
    assert option_fields.current_value(reloaded, option_spec) == 1 - original_flag
    reloaded_triggers = parse_triggers(reloaded)
    assert reloaded_triggers.triggers[0].name == "Multi-domain save: trigger 0 (edited)"
    assert (
        reloaded._scenario.sections["Messages"].retriever_map["ascii_victory"].data
        == "multi-domain victory text"
    )
