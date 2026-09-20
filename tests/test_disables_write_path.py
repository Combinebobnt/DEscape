"""The per-player disable lists' write path: descape/disables_fields.py's
splice plus scenario_write._patch_disables().

The load-bearing claim, and the reason this file exists beside
tests/test_messages_write_path.py rather than inside it: this is the first
resizing step that lands in the *middle* of the write path's descending-offset
ordering rule. The Options region sits below Units/Triggers and above
Messages, so a save that edits all three at once is the only thing that
actually proves the ordering -- test_a_disables_edit_combined_with_messages_
and_players_all_read_back below.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from AoE2ScenarioParser.scenarios.aoe2_scenario import _decompress_bytes

from descape import disables_fields, player_fields, scenario_write
from descape.messages_model import MessagesEditModel
from descape.options_model import OptionsEditModel
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from descape.scenario_write import WriteBlockedError, write_scenario

import conftest

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"

_BUILDINGS_P2 = disables_fields.disables_field_id("buildings", 2)
_TECHS_P5 = disables_fields.disables_field_id("techs", 5)


def _written_body(path: Path, header_len: int) -> bytes:
    return _decompress_bytes(path.read_bytes()[header_len:])


def _save(loaded, out: Path, **kwargs):
    write_scenario(loaded, out, **kwargs)
    return _written_body(out, len(loaded.header_bytes))


# -- 1. zero-edit identity ---------------------------------------------------


@pytest.mark.parametrize("path", [BLANK_TEMPLATE_PATH, FIXTURE_PATH], ids=["blank", "triggers"])
@pytest.mark.parametrize("with_model", [False, True], ids=["no-model", "clean-model"])
def test_zero_edit_save_is_byte_identical(path, with_model: bool, tmp_path: Path) -> None:
    """_patch_disables() runs unconditionally whenever an OptionsEditModel is
    passed, so "a clean model splices nothing" is the containment every other
    model in this codebase gives and has to be pinned here too."""
    loaded = load_map_and_units(path)
    model = OptionsEditModel(loaded) if with_model else None
    body = _save(loaded, tmp_path / "out.aoe2scenario", options=model)
    assert body == loaded.decompressed_body


@pytest.mark.corpus
def test_zero_edit_save_with_a_clean_model_is_byte_identical_corpus(
    scenario_path, tmp_path: Path
) -> None:
    loaded = load_map_and_units(scenario_path)
    if not loaded.terrain_write_supported:
        pytest.skip(f"{scenario_path.name}: terrain_write_supported is False")
    body = _save(loaded, tmp_path / "out.aoe2scenario", options=OptionsEditModel(loaded))
    assert body == loaded.decompressed_body


# -- 2. an edit reads back, and nothing else moves ---------------------------


def test_an_edit_reads_back_after_a_reload(tmp_path: Path) -> None:
    loaded = load_map_and_units(FIXTURE_PATH)
    model = OptionsEditModel(loaded)
    model.set_value(_BUILDINGS_P2, (72, 621))
    model.set_value(_TECHS_P5, (22,))

    out = tmp_path / "out.aoe2scenario"
    write_scenario(loaded, out, options=model)

    reloaded = load_map_and_units(out)
    assert disables_fields.current_ids(reloaded, "buildings", 2) == (72, 621)
    assert disables_fields.current_ids(reloaded, "techs", 5) == (22,)
    # Every other list stays empty, and the counts agree with the lists.
    assert disables_fields.current_ids(reloaded, "buildings", 1) == ()
    assert disables_fields.current_counts(reloaded, "buildings")[:8] == (0, 2, 0, 0, 0, 0, 0, 0)
    assert disables_fields.current_counts(reloaded, "techs")[:8] == (0, 0, 0, 0, 1, 0, 0, 0)
    # And the file the splice produced still passes its own gate.
    assert disables_fields.verify_disables_block(reloaded) is True


def test_a_growing_edit_shifts_only_what_follows_the_region(tmp_path: Path) -> None:
    """Locality for the resizing case, which differing_ranges() cannot
    express (the two bodies are different lengths): everything in front of
    the region is untouched, and everything after it is the original bytes
    moved along by exactly the four the one added id costs."""
    loaded = load_map_and_units(FIXTURE_PATH)
    model = OptionsEditModel(loaded)
    model.set_value(_BUILDINGS_P2, (72,))
    body = _save(loaded, tmp_path / "out.aoe2scenario", options=model)

    start, end = disables_fields.disables_region_span(loaded)
    assert len(body) == len(loaded.decompressed_body) + 4
    assert body[:start] == loaded.decompressed_body[:start]
    assert body[end + 4 :] == loaded.decompressed_body[end:]


def test_a_same_length_edit_changes_only_bytes_inside_the_region(tmp_path: Path) -> None:
    """The equal-length case, so locality can be asserted byte-range-wise:
    swap one disabled id for another and nothing outside the region may
    differ."""
    loaded = load_map_and_units(FIXTURE_PATH)
    staged_model = OptionsEditModel(loaded)
    staged_model.set_value(_BUILDINGS_P2, (72,))
    stage = tmp_path / "stage.aoe2scenario"
    write_scenario(loaded, stage, options=staged_model)

    staged = load_map_and_units(stage)
    model = OptionsEditModel(staged)
    model.set_value(_BUILDINGS_P2, (621,))
    body = _save(staged, tmp_path / "out.aoe2scenario", options=model)

    start, end = disables_fields.disables_region_span(staged)
    assert len(body) == len(staged.decompressed_body)
    ranges = conftest.differing_ranges(staged.decompressed_body, body)
    assert ranges, "the id swap should have changed something"
    assert all(start <= lo and hi <= end for lo, hi in ranges), ranges


def test_an_out_of_enum_id_survives_a_round_trip(tmp_path: Path) -> None:
    """621 and 35 are real corpus values the library enums do not carry --
    the write path must not filter them."""
    loaded = load_map_and_units(FIXTURE_PATH)
    model = OptionsEditModel(loaded)
    model.set_value(disables_fields.disables_field_id("buildings", 1), (621,))
    model.set_value(disables_fields.disables_field_id("units", 1), (35,))
    out = tmp_path / "out.aoe2scenario"
    write_scenario(loaded, out, options=model)

    reloaded = load_map_and_units(out)
    assert disables_fields.current_ids(reloaded, "buildings", 1) == (621,)
    assert disables_fields.current_ids(reloaded, "units", 1) == (35,)


def test_clearing_a_list_shrinks_the_region(tmp_path: Path) -> None:
    """The other direction: a save must be able to make the region smaller,
    which is the case a fixed-length patch path could never express."""
    loaded = load_map_and_units(FIXTURE_PATH)
    first = OptionsEditModel(loaded)
    first.set_value(_BUILDINGS_P2, (72, 621, 10))
    stage = tmp_path / "stage.aoe2scenario"
    write_scenario(loaded, stage, options=first)

    staged = load_map_and_units(stage)
    assert disables_fields.current_ids(staged, "buildings", 2) == (72, 621, 10)
    second = OptionsEditModel(staged)
    second.set_value(_BUILDINGS_P2, ())
    out = tmp_path / "out.aoe2scenario"
    write_scenario(staged, out, options=second)

    reloaded = load_map_and_units(out)
    assert disables_fields.current_ids(reloaded, "buildings", 2) == ()
    assert disables_fields.current_counts(reloaded, "buildings")[:8] == (0,) * 8


# -- 3. the ordering proof ---------------------------------------------------


def test_a_disables_edit_combined_with_messages_and_players_all_read_back(
    tmp_path: Path,
) -> None:
    """Three splices and a fixed-offset patch in one save: Units/Triggers is
    untouched here, but Messages (upstream of Options) and player_data_1
    (offset 0) both resize, and a per-player scalar patches in place. If
    _patch_disables() ran on the wrong side of the Messages splice, one of
    these three reads back as garbage.
    """
    loaded = load_map_and_units(FIXTURE_PATH)
    options = OptionsEditModel(loaded)
    messages = MessagesEditModel(loaded)

    options.set_value(_BUILDINGS_P2, (72, 621))
    # civilization on this 1.56+ fixture is a str16, so it rides
    # _patch_player_data_1() -- the splice at offset 0, the far end of the
    # ordering rule from Messages.
    civ_field = player_fields.player_field_id("civilization", 3)
    civ_before = options.current_value(civ_field)
    civ_after = next(
        value for value, _label in player_fields.civilization_choices(loaded) if value != civ_before
    )
    options.set_value(civ_field, civ_after)

    messages.set_value("instructions", "Disables ordering probe.")

    out = tmp_path / "out.aoe2scenario"
    write_scenario(loaded, out, options=options, messages=messages)

    reloaded = load_map_and_units(out)
    assert disables_fields.current_ids(reloaded, "buildings", 2) == (72, 621)
    assert disables_fields.verify_disables_block(reloaded) is True
    assert MessagesEditModel(reloaded).current_value("instructions") == (
        "Disables ordering probe."
    )
    reloaded_options = OptionsEditModel(reloaded)
    assert reloaded_options.current_value(civ_field) == civ_after


# -- 4. fail-closed ----------------------------------------------------------


def test_the_write_path_re_gates_rather_than_trusting_construction(
    tmp_path: Path, monkeypatch
) -> None:
    """The model already refused to seed these ids for a file that failed the
    gate, so reaching this means something changed underneath it."""
    loaded = load_map_and_units(FIXTURE_PATH)
    model = OptionsEditModel(loaded)
    model.set_value(_BUILDINGS_P2, (72,))
    monkeypatch.setattr(scenario_write, "disables_write_supported", lambda *_: False)
    with pytest.raises(WriteBlockedError):
        write_scenario(loaded, tmp_path / "out.aoe2scenario", options=model)


def test_a_splice_outside_the_options_region_is_refused(tmp_path: Path, monkeypatch) -> None:
    """The boundary guard, pinned by forging a span the codec would never
    produce -- the same shape test_options_write_path.py's trigger-counter
    guard test uses."""
    loaded = load_map_and_units(FIXTURE_PATH)
    model = OptionsEditModel(loaded)
    model.set_value(_BUILDINGS_P2, (72,))
    real = model.serialize_disables_resize()
    monkeypatch.setattr(
        model, "serialize_disables_resize", lambda: (0, real[1], real[2]), raising=False
    )
    with pytest.raises(WriteBlockedError):
        write_scenario(loaded, tmp_path / "out.aoe2scenario", options=model)
