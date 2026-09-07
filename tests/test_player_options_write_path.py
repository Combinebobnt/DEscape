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
    player_count_write_supported,
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


def test_tribe_name_is_settable_as_of_step_3d() -> None:
    """c256 is a string codec; _original/_pending are int | str (step 3d),
    so tribe_name rides the same additive player-field set as every other
    Tier-1 spec."""
    model = OptionsEditModel(_loaded())
    assert any(key.startswith("player:tribe_name:") for key in model._player_targets)
    assert isinstance(model.original_value("player:tribe_name:1"), str)
    model.set_value("player:tribe_name:1", "Franks")
    assert model.current_value("player:tribe_name:1") == "Franks"


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


def test_a_tribe_name_edit_reads_back_after_a_reload(tmp_path: Path) -> None:
    loaded = _loaded()
    model = OptionsEditModel(loaded)
    model.set_value("player:tribe_name:6", "Mongols")
    out = tmp_path / "out.aoe2scenario"
    write_scenario(loaded, out, options=model)

    reloaded = load_map_and_units(out)
    spec = next(s for s in player_fields.specs_for(reloaded) if s.field_id == "tribe_name")
    assert player_fields.current_value(reloaded, spec, 6) == "Mongols"
    assert players_write_supported(reloaded)


# -- civ/architecture Step B: the resizing splice, reparse round trip --------


def _all_player_values(loaded) -> dict[tuple[str, int], int | str]:
    """Every (field_id, player_id) -> current_value(), for asserting that a
    resizing splice on player_data_1 leaves every *other* field on every
    *other* player reading byte-identically to before -- not just that the
    one edited field reads the new value."""
    return {
        (spec.field_id, player_id): player_fields.current_value(loaded, spec, player_id)
        for spec in player_fields.specs_for(loaded)
        for player_id in range(1, 9)
    }


@pytest.mark.parametrize(
    "new_value",
    [
        pytest.param("HUN-CIV", id="shrink"),  # RANDOM-CIV (10) -> HUN-CIV (7)
        pytest.param("MIRROR-RANDOM-CIV", id="grow"),  # RANDOM-CIV (10) -> 17
        pytest.param("BRITON-CIV", id="same-length"),  # 10 -> 10, the delta-zero case
    ],
)
def test_a_civilization_edit_reads_back_after_a_reload(new_value: str, tmp_path: Path) -> None:
    """BLANK_TEMPLATE_PATH is 1.58 (str16), so this exercises Step B's
    resizing splice, not Step A's plain byte patch. Confirms the edited
    field's new value, that every other field on every player is
    byte-for-byte unaffected, and that players_write_supported() still
    holds post-reload -- the reparse round trip the maintainer plan's
    verification section calls for, for a shrink, a grow, and the
    delta-zero same-length case in one parametrization."""
    loaded = _loaded()
    before = _all_player_values(loaded)
    assert before[("civilization", 1)] != new_value

    model = OptionsEditModel(loaded)
    model.set_value("player:civilization:1", new_value)
    out = tmp_path / "out.aoe2scenario"
    write_scenario(loaded, out, options=model)

    reloaded = load_map_and_units(out)
    spec = next(s for s in player_fields.specs_for(reloaded) if s.field_id == "civilization")
    assert player_fields.current_value(reloaded, spec, 1) == new_value
    assert players_write_supported(reloaded)

    after = _all_player_values(reloaded)
    for key, value in before.items():
        if key == ("civilization", 1):
            continue
        assert after[key] == value, f"{key} changed: {value!r} -> {after[key]!r}"


def test_civilization_and_architecture_are_independent_in_one_save(tmp_path: Path) -> None:
    """Editing P1's civilization must not touch P2's architecture, and vice
    versa -- both fields share player_data_1_splice()'s single combined
    region, so this is the test that would catch them clobbering each
    other's span."""
    loaded = _loaded()
    model = OptionsEditModel(loaded)
    civ_before = model.original_value("player:civilization:1")
    arch_before = model.original_value("player:architecture:2")
    model.set_value("player:civilization:1", "HUN-CIV")
    model.set_value("player:architecture:2", "MIRROR-RANDOM-CIV")
    assert civ_before != "HUN-CIV"
    assert arch_before != "MIRROR-RANDOM-CIV"

    out = tmp_path / "out.aoe2scenario"
    write_scenario(loaded, out, options=model)

    reloaded = load_map_and_units(out)
    specs = {s.field_id: s for s in player_fields.specs_for(reloaded)}
    assert player_fields.current_value(reloaded, specs["civilization"], 1) == "HUN-CIV"
    assert player_fields.current_value(reloaded, specs["architecture"], 2) == "MIRROR-RANDOM-CIV"
    # P1's architecture and P2's civilization must be untouched.
    assert player_fields.current_value(reloaded, specs["architecture"], 1) != "MIRROR-RANDOM-CIV"
    assert player_fields.current_value(reloaded, specs["civilization"], 2) != "HUN-CIV"
    assert players_write_supported(reloaded)


def test_a_civilization_edit_combined_with_a_messages_and_a_map_options_edit(
    tmp_path: Path,
) -> None:
    """The case _patch_player_data_1()'s ordering (after _patch_messages())
    exists for: a civ resize plus a Messages edit plus a Map Options scalar
    in one write_scenario() call. This is the one that would silently
    corrupt if that ordering regressed -- Messages sits between
    player_data_1 and everything _patch_options() addresses, so a wrong
    order would shift one or the other's offsets."""
    from descape.messages_model import MessagesEditModel

    loaded = _loaded()
    model = OptionsEditModel(loaded)
    model.set_value("player:civilization:1", "MIRROR-RANDOM-CIV")  # a grow
    before_years = model.original_value("victory_years")
    model.set_value("victory_years", before_years + 5)

    messages = MessagesEditModel(loaded)
    messages.set_value("instructions", "X" * 500)  # long enough to shift the body

    out = tmp_path / "out.aoe2scenario"
    write_scenario(loaded, out, options=model, messages=messages)

    reloaded = load_map_and_units(out)
    specs = {s.field_id: s for s in player_fields.specs_for(reloaded)}
    assert player_fields.current_value(reloaded, specs["civilization"], 1) == "MIRROR-RANDOM-CIV"
    assert option_fields.current_value(
        reloaded, next(s for s in option_fields.specs_for(reloaded) if s.field_id == "victory_years")
    ) == before_years + 5
    reloaded_messages = MessagesEditModel(reloaded)
    assert reloaded_messages.current_value("instructions") == "X" * 500
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


def test_set_value_refuses_tribe_name_text_that_overflows_its_slot() -> None:
    """tribe_name's DataHeader.tribe_names slot is a fixed 256-byte c256;
    text longer than the slot can hold must raise rather than silently
    overflowing into the next field."""
    model = OptionsEditModel(_loaded())
    with pytest.raises(ValueError):
        model.set_value("player:tribe_name:1", "x" * 257)
    model.set_value("player:tribe_name:1", "x" * 255)  # leaves room for the NUL, must not raise


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


# -- Number of Players (step 3e) ----------------------------------------------
#
# The one edit in this model that reaches two buffers: eight `active` flags
# in decompressed_body plus FileHeader.player_count in header_bytes. It has
# its own gate (player_count_write_supported()) because those two halves can
# fail independently of every per-player row.

_STR16_CIV_PATH = Path(__file__).resolve().parent.parent / "examples" / "ring75_v0_scx_resaved.aoe2scenario"


def _header_count(path: Path) -> int:
    loaded = load_map_and_units(path)
    start, end = loaded.header_player_count_span
    return int.from_bytes(loaded.header_bytes[start:end], "little")


def test_player_count_is_settable_and_normalises_back_out_of_pending() -> None:
    loaded = _loaded()
    model = OptionsEditModel(loaded)
    original = model.original_value(player_fields.PLAYER_COUNT_FIELD_ID)
    assert original == player_fields.defined_player_count(loaded)
    model.set_value(player_fields.PLAYER_COUNT_FIELD_ID, 5)
    assert model.has_player_count_edit and model.has_player_edits
    model.set_value(player_fields.PLAYER_COUNT_FIELD_ID, original)
    assert not model.has_player_count_edit and not model.has_edits


def test_setting_an_out_of_range_player_count_raises() -> None:
    model = OptionsEditModel(_loaded())
    with pytest.raises(ValueError):
        model.set_value(player_fields.PLAYER_COUNT_FIELD_ID, 9)
    with pytest.raises(ValueError):
        model.set_value(player_fields.PLAYER_COUNT_FIELD_ID, 0)


def test_the_player_count_is_absent_when_its_own_gate_fails() -> None:
    """Independent of players_write_supported(): breaking only the header
    span leaves every per-player row settable and takes just this one
    away."""
    loaded = _loaded()
    loaded.header_player_count_span = (-1, -1)
    assert not player_count_write_supported(loaded)
    assert players_write_supported(loaded)

    model = OptionsEditModel(loaded)
    with pytest.raises(KeyError):
        model.set_value(player_fields.PLAYER_COUNT_FIELD_ID, 5)
    model.set_value("player:food:1", 500)
    assert model.has_player_edits


def test_the_player_count_survives_a_per_player_gate_failure() -> None:
    """The other direction: a PlayerDataTwo failure takes away the
    per-player rows but not this one, since `active` lives in DataHeader
    and the header count lives outside the body entirely."""
    loaded = _loaded()
    loaded.player_data_two_section_end += 4
    assert not players_write_supported(loaded)
    assert player_count_write_supported(loaded)
    model = OptionsEditModel(loaded)
    model.set_value(player_fields.PLAYER_COUNT_FIELD_ID, 5)
    assert model.has_player_count_edit


def test_a_clean_model_emits_no_header_patch() -> None:
    assert OptionsEditModel(_loaded()).header_patch() is None


def test_the_header_patch_is_the_same_width_as_the_field_it_replaces() -> None:
    loaded = _loaded()
    model = OptionsEditModel(loaded)
    model.set_value(player_fields.PLAYER_COUNT_FIELD_ID, 5)
    start, end, payload = model.header_patch()
    assert (start, end) == loaded.header_player_count_span
    assert len(payload) == end - start
    assert int.from_bytes(payload, "little") == 5


def test_one_count_edit_patches_exactly_the_eight_active_flags(tmp_path: Path) -> None:
    """Locality, the same shape the mirrored-field tests above use: nothing
    outside the eight `active` spans moves, and GAIA's own slot (index 8)
    is left alone."""
    loaded = _loaded()
    model = OptionsEditModel(loaded)
    model.set_value(player_fields.PLAYER_COUNT_FIELD_ID, 5)

    targets = player_fields.player_count_targets(loaded)
    expected = {(t.offset, t.offset + t.length) for t in targets}
    patches = model.serialize_patches()
    assert {(o, o + len(d)) for o, d in patches} == expected

    out = tmp_path / "count.aoe2scenario"
    write_scenario(loaded, out, options=model, backup=False)
    changed = set(_differing_ranges(loaded.decompressed_body, _written_body(out)))
    # Only flags whose value actually changed differ, so this is a subset of
    # the eight spans, never a superset.
    assert changed
    for start, end in changed:
        assert any(lo <= start and end <= hi for lo, hi in expected), (start, end)

    gaia = player_fields._player_data_1_variable_target(loaded, "active", 0).target
    assert not any(
        start < gaia.offset + gaia.length and gaia.offset < end for start, end in changed
    )


def test_a_count_edit_round_trips_through_both_buffers(tmp_path: Path) -> None:
    loaded = _loaded()
    model = OptionsEditModel(loaded)
    model.set_value(player_fields.PLAYER_COUNT_FIELD_ID, 6)
    out = tmp_path / "count.aoe2scenario"
    write_scenario(loaded, out, options=model, backup=False)

    reloaded = load_map_and_units(out)
    assert player_fields.defined_player_ids(reloaded) == [1, 2, 3, 4, 5, 6]
    assert _header_count(out) == 6


def test_lowering_the_count_deactivates_the_top_players(tmp_path: Path) -> None:
    """Not just a same-or-growing check: the corpus's own counts run 2..8,
    and a writer that only ever set flags to 1 would pass a grow-only
    test."""
    loaded = load_map_and_units(_STR16_CIV_PATH)
    assert player_fields.defined_player_count(loaded) == 4
    model = OptionsEditModel(loaded)
    model.set_value(player_fields.PLAYER_COUNT_FIELD_ID, 2)
    out = tmp_path / "count.aoe2scenario"
    write_scenario(loaded, out, options=model, backup=False)

    reloaded = load_map_and_units(out)
    assert player_fields.defined_player_ids(reloaded) == [1, 2]
    assert _header_count(out) == 2


def test_a_count_edit_and_a_str16_civ_edit_in_one_save_both_land(tmp_path: Path) -> None:
    """The collision the civ/architecture plan's Step B ordering does NOT
    give for free: player_data_1_splice() rebuilds the whole array, and
    reading the *original* body there silently discards the `active` patch
    made in the same save. Confirmed empirically -- with the splice reading
    the original body, this file comes back with 4 players in the body and
    2 in the header.
    """
    loaded = load_map_and_units(_STR16_CIV_PATH)
    civ_key = player_fields.player_field_id("civilization", 3)
    model = OptionsEditModel(loaded)
    assert model._player_targets[civ_key][0].codec == "str16"
    model.set_value(civ_key, "HUN-CIV")
    model.set_value(player_fields.PLAYER_COUNT_FIELD_ID, 2)
    out = tmp_path / "both.aoe2scenario"
    write_scenario(loaded, out, options=model, backup=False)

    reloaded = load_map_and_units(out)
    civ_spec = next(s for s in player_fields.specs_for(reloaded) if s.field_id == "civilization")
    assert player_fields.current_value(reloaded, civ_spec, 3) == "HUN-CIV"
    assert player_fields.defined_player_ids(reloaded) == [1, 2]
    assert _header_count(out) == 2


def test_a_count_edit_is_refused_when_its_gate_stops_verifying(tmp_path: Path) -> None:
    """write_scenario()'s own re-gate, matching the Diplomacy/Players ones:
    the model already refused to offer this on a bad file, so reaching here
    means something changed under it."""
    loaded = _loaded()
    model = OptionsEditModel(loaded)
    model.set_value(player_fields.PLAYER_COUNT_FIELD_ID, 5)
    loaded.header_player_count_span = (-1, -1)
    with pytest.raises(WriteBlockedError, match="Number of Players"):
        write_scenario(loaded, tmp_path / "refused.aoe2scenario", options=model, backup=False)


def test_a_count_edit_leaves_the_rest_of_the_header_alone(tmp_path: Path) -> None:
    loaded = _loaded()
    model = OptionsEditModel(loaded)
    model.set_value(player_fields.PLAYER_COUNT_FIELD_ID, 5)
    out = tmp_path / "count.aoe2scenario"
    write_scenario(loaded, out, options=model, backup=False)

    written = load_map_and_units(out).header_bytes
    start, end = loaded.header_player_count_span
    assert len(written) == len(loaded.header_bytes)
    assert written[:start] == loaded.header_bytes[:start]
    assert written[end:] == loaded.header_bytes[end:]
