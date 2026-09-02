"""Players mode step 3b: the write path, no UI. descape/options_model.py's
OptionsEditModel extended with per-player fields (an additive entry set
under synthetic ids, "player:food:3"), players_write_supported() as its own
gate, and the descape/scenario_write.py branch that re-checks it. Still no
panel -- every edit here goes through OptionsEditModel.set_value() directly,
the same way tests/test_diplomacy_write_path.py exercises the Diplomacy grid.

The claims, mirroring that file's own list:

1. Containment: a model with no player edits writes exactly the bytes it
   wrote before this work existed.
2. Locality: one field edit changes only that field's own target span(s).
3. Mirror correctness: editing a mirrored field patches every mirror too.
4. Round trip: write, reload through the real loader, read back.
5. The gates: players_write_supported() is independent of
   options_write_supported()/diplomacy_write_supported().
6. encode_target()'s f32-exactness refusal surfaces through set_value().
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
from AoE2ScenarioParser.scenarios.aoe2_scenario import _decompress_bytes

from descape import option_fields, player_fields
from descape.options_model import (
    OptionsEditModel,
    diplomacy_write_supported,
    options_write_supported,
    players_write_supported,
)
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from descape.scenario_write import WriteBlockedError, write_scenario


def _loaded(path: Path = BLANK_TEMPLATE_PATH):
    return load_map_and_units(path)


def _written_body(path: Path) -> bytes:
    raw = path.read_bytes()
    loaded = load_map_and_units(path)
    return _decompress_bytes(raw[len(loaded.header_bytes) :])


def _differing_ranges(before: bytes, after: bytes) -> list[tuple[int, int]]:
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


# -- construction -------------------------------------------------------------


def test_player_fields_are_present_in_a_writable_model() -> None:
    loaded = _loaded()
    assert players_write_supported(loaded)
    model = OptionsEditModel(loaded)
    assert model.original_value("player:food:1") == player_fields.current_value(
        loaded, next(s for s in player_fields.specs_for(loaded) if s.field_id == "food"), 1
    )


def test_tribe_name_is_absent_from_the_model_until_step_3d() -> None:
    """c256 is a string codec; _original/_pending stay int-only until step
    3d widens them, so tribe_name must not appear as a settable id yet even
    though player_fields.write_targets() resolves it."""
    model = OptionsEditModel(_loaded())
    assert not any(key.startswith("player:tribe_name:") for key in model._player_targets)
    with pytest.raises(KeyError):
        model.set_value("player:tribe_name:1", 5)


def test_player_fields_are_absent_when_the_player_gate_fails() -> None:
    """An independent gate: corrupting the PlayerDataTwo anchor leaves the
    scalar rows and the Diplomacy grid constructible, and simply omits
    every per-player id rather than raising."""
    loaded = _loaded()
    loaded.player_data_two_section_end += 4
    assert not players_write_supported(loaded)
    assert options_write_supported(loaded, option_fields.specs_for(loaded))
    assert diplomacy_write_supported(loaded)

    model = OptionsEditModel(loaded)
    assert not model.has_player_edits
    with pytest.raises(KeyError):
        model.set_value("player:food:1", 500)
    model.set_value("lock_teams", 1 - model.original_value("lock_teams"))
    assert model.has_edits


# -- 1. containment -------------------------------------------------------


def test_a_model_with_no_player_edits_writes_the_same_file_as_no_model_at_all(
    tmp_path: Path,
) -> None:
    loaded = _loaded()
    without = tmp_path / "without.aoe2scenario"
    write_scenario(loaded, without)

    model = OptionsEditModel(loaded)
    assert not model.has_edits
    with_model = tmp_path / "with.aoe2scenario"
    write_scenario(loaded, with_model, options=model)

    assert with_model.read_bytes() == without.read_bytes()


def test_a_player_field_set_back_to_its_stored_value_leaves_the_document_clean() -> None:
    loaded = _loaded()
    model = OptionsEditModel(loaded)
    stored = model.original_value("player:base_priority:2")
    model.set_value("player:base_priority:2", stored + 1)
    assert model.has_edits
    assert model.has_player_edits
    model.set_value("player:base_priority:2", stored)
    assert not model.has_edits
    assert not model.has_player_edits


# -- 2. locality ------------------------------------------------------------


def test_editing_an_unmirrored_field_changes_only_its_own_span(tmp_path: Path) -> None:
    loaded = _loaded()
    targets = player_fields.write_targets(loaded)
    (target,) = targets["player:base_priority:2"]

    base = tmp_path / "base.aoe2scenario"
    write_scenario(loaded, base)

    model = OptionsEditModel(loaded)
    stored = model.original_value("player:base_priority:2")
    model.set_value("player:base_priority:2", (stored + 1) % 256)
    edited = tmp_path / "edited.aoe2scenario"
    write_scenario(loaded, edited, options=model)

    ranges = _differing_ranges(_written_body(base), _written_body(edited))
    assert ranges, "no bytes changed"
    for start, end in ranges:
        assert target.offset <= start and end <= target.offset + target.length


def test_editing_one_players_food_leaves_every_other_player_untouched(tmp_path: Path) -> None:
    loaded = _loaded()
    model = OptionsEditModel(loaded)
    stored = model.original_value("player:food:3")
    model.set_value("player:food:3", stored + 111)

    out = tmp_path / "out.aoe2scenario"
    write_scenario(loaded, out, options=model)
    reloaded = load_map_and_units(out)
    specs_by_id = {s.field_id: s for s in player_fields.specs_for(reloaded)}
    for player_id in range(1, 9):
        expected = stored + 111 if player_id == 3 else model.original_value(f"player:food:{player_id}")
        assert player_fields.current_value(reloaded, specs_by_id["food"], player_id) == expected


def test_editing_food_touches_only_the_primary_and_mirror_spans_gaia_untouched(
    tmp_path: Path,
) -> None:
    """Plan verification item 4, the byte-level half: edit P3's food and
    confirm the written body's diff is confined to exactly the primary and
    mirror spans. Covers two things test_editing_one_players_food_... above
    doesn't: the GAIA sentinel verify_player_block() leans on to detect a
    shift, and the neighbouring resources/player_data_4 fields (ore_x_unused,
    trade_goods, player_color, the *_duplicate siblings) that a food edit
    must never touch -- any of those changing would show up as a diff range
    outside the two 4-byte spans and fail the assertion below."""
    loaded = _loaded()
    targets = player_fields.write_targets(loaded)
    primary, mirror = targets["player:food:3"]
    gaia_resources = player_fields._array_target(loaded, "PlayerDataTwo", "resources", None, 8)
    assert gaia_resources is not None

    base = tmp_path / "base.aoe2scenario"
    write_scenario(loaded, base)

    model = OptionsEditModel(loaded)
    stored = model.original_value("player:food:3")
    model.set_value("player:food:3", stored + 111)
    edited = tmp_path / "edited.aoe2scenario"
    write_scenario(loaded, edited, options=model)

    before = _written_body(base)
    after = _written_body(edited)
    ranges = _differing_ranges(before, after)
    assert ranges, "no bytes changed"
    spans = (primary, mirror)
    for start, end in ranges:
        assert any(t.offset <= start and end <= t.offset + t.length for t in spans), (
            f"byte range ({start}, {end}) falls outside both the primary and mirror spans"
        )
    gaia_slice = slice(gaia_resources.target.offset, gaia_resources.target.offset + gaia_resources.target.length)
    assert before[gaia_slice] == after[gaia_slice]


# -- 3. mirror correctness ---------------------------------------------------


def test_editing_food_patches_both_the_primary_and_the_player_data_4_mirror(
    tmp_path: Path,
) -> None:
    loaded = _loaded()
    targets = player_fields.write_targets(loaded)
    primary, mirror = targets["player:food:5"]
    assert primary.codec == "s32"
    assert mirror.codec == "f32"

    model = OptionsEditModel(loaded)
    stored = model.original_value("player:food:5")
    model.set_value("player:food:5", stored + 42)
    patches = dict(model.serialize_patches())
    assert primary.offset in patches
    assert mirror.offset in patches
    assert patches[primary.offset] == struct.pack("<i", stored + 42)
    assert patches[mirror.offset] == struct.pack("<f", float(stored + 42))


def test_editing_color_patches_the_player_data_3_mirror_too(tmp_path: Path) -> None:
    loaded = _loaded()
    targets = player_fields.write_targets(loaded)
    primary, mirror = targets["player:color:2"]

    model = OptionsEditModel(loaded)
    stored = model.original_value("player:color:2")
    other = next(v for v in range(9) if v != stored)
    model.set_value("player:color:2", other)
    out = tmp_path / "out.aoe2scenario"
    write_scenario(loaded, out, options=model)

    reloaded = load_map_and_units(out)
    body = reloaded.decompressed_body
    (primary_value,) = struct.unpack_from("<i", body, primary.offset)
    (mirror_value,) = struct.unpack_from("<I", body, mirror.offset)
    assert primary_value == other
    assert mirror_value == other


# -- 4. round trip ------------------------------------------------------------


def test_a_player_field_edit_reads_back_after_a_reload(tmp_path: Path) -> None:
    loaded = _loaded()
    model = OptionsEditModel(loaded)
    stored = model.original_value("player:starting_age:4")
    other = next(v for v in range(6) if v != stored)
    model.set_value("player:starting_age:4", other)
    out = tmp_path / "out.aoe2scenario"
    write_scenario(loaded, out, options=model)

    reloaded = load_map_and_units(out)
    spec = next(s for s in player_fields.specs_for(reloaded) if s.field_id == "starting_age")
    assert player_fields.current_value(reloaded, spec, 4) == other
    assert players_write_supported(reloaded)


# -- 5. the gates -------------------------------------------------------------


def test_set_value_refuses_an_unrecognized_player_field() -> None:
    model = OptionsEditModel(_loaded())
    with pytest.raises(KeyError):
        model.set_value("player:food:9", 500)  # no player 9


def test_the_write_path_re_gates_players_rather_than_trusting_construction(
    tmp_path: Path,
) -> None:
    loaded = _loaded()
    model = OptionsEditModel(loaded)
    stored = model.original_value("player:food:1")
    model.set_value("player:food:1", stored + 5)

    loaded.player_data_two_section_end += 4
    assert options_write_supported(loaded, model.specs)
    assert not players_write_supported(loaded)

    with pytest.raises(WriteBlockedError):
        write_scenario(loaded, tmp_path / "out.aoe2scenario", options=model)


def test_a_players_only_failure_does_not_block_an_unrelated_scalar_save(
    tmp_path: Path,
) -> None:
    loaded = _loaded()
    model = OptionsEditModel(loaded)
    stored = model.original_value("lock_teams")
    model.set_value("lock_teams", 1 - stored)

    loaded.player_data_two_section_end += 4
    assert not players_write_supported(loaded)

    out = tmp_path / "out.aoe2scenario"
    write_scenario(loaded, out, options=model)

    reloaded = load_map_and_units(out)
    spec = next(s for s in option_fields.specs_for(reloaded) if s.field_id == "lock_teams")
    assert option_fields.current_value(reloaded, spec) == 1 - stored


# -- 6. f32 exactness ---------------------------------------------------------


def test_set_value_refuses_a_food_value_that_is_not_f32_exact() -> None:
    """food's mirror (player_data_4.food_duplicate) is f32; an integer above
    2**24 would silently desync the two stored copies."""
    model = OptionsEditModel(_loaded())
    with pytest.raises(ValueError):
        model.set_value("player:food:1", 2**24 + 1)
    model.set_value("player:food:1", 2**24)  # exact, must not raise


# -- 7. corpus ----------------------------------------------------------------


@pytest.mark.corpus
def test_a_browsed_player_model_saves_byte_identically_across_the_corpus(
    scenario_path, tmp_path: Path
) -> None:
    loaded = load_map_and_units(scenario_path)
    if not loaded.terrain_write_supported:
        pytest.skip("terrain block failed verification, so no save path at all")
    if not players_write_supported(loaded):
        pytest.skip("player block fails closed on this file, as intended")

    without = tmp_path / "without.aoe2scenario"
    write_scenario(loaded, without)
    model = OptionsEditModel(loaded)
    with_model = tmp_path / "with.aoe2scenario"
    write_scenario(loaded, with_model, options=model)
    assert with_model.read_bytes() == without.read_bytes()
