"""Players mode write path, step 3a: descape/player_fields.py's offsets,
write_targets(), verify_player_block() and encode_target(). Qt-free -- no
model wiring, no panel. Mirrors tests/test_diplomacy_write_path.py's shape:
real-load based rather than stub based, since offset math against a real
parse is exactly what needs validating.

The claims, in the order the maintainer plan's own verification section
lists them:

1. Every offset derivation verifies byte-for-byte (verify_player_block()).
2. Mutation-verify each guard: corrupting the GAIA sentinel, a mirror byte,
   or a trusted anchor must flip verify_player_block() to False.
3. encode_target()'s f32-exactness and c256-overflow refusals.
4. The corpus: 20/20 examples/ files verify, at whatever field count their
   own scenario version actually stores.
"""

from __future__ import annotations

import pytest

from descape import player_fields as pf
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units


def _loaded(path=BLANK_TEMPLATE_PATH):
    return load_map_and_units(path)


# -- 1. every offset verifies byte-for-byte ----------------------------------


def test_verify_player_block_passes_on_a_clean_load() -> None:
    assert pf.verify_player_block(_loaded())


def test_write_targets_is_empty_until_verify_passes() -> None:
    """write_targets() is only meaningful once verify_player_block() has
    returned True -- a caller (options_model.py, step 3b) is expected to
    gate on that first, the same shape diplomacy_write_supported() is
    gated ahead of stance_offsets()."""
    loaded = _loaded()
    assert pf.verify_player_block(loaded)
    targets = pf.write_targets(loaded)
    assert targets
    # 14 Tier-1 field_ids * 8 players, on a file new enough to carry all of
    # them (the blank template is 1.58).
    field_ids = {key.split(":")[1] for key in targets}
    assert field_ids == {
        "tribe_name", "player_name_string_id", "lock_civilization", "lock_personality",
        "starting_age", "food", "wood", "stone", "gold", "pop_limit", "color", "base_priority",
        "initial_view_x", "initial_view_y",
    }
    assert all(key.split(":")[2] in {str(p) for p in range(1, 9)} for key in targets)


def test_write_targets_never_includes_player_type_or_tier_2() -> None:
    targets = pf.write_targets(_loaded())
    field_ids = {key.split(":")[1] for key in targets}
    assert field_ids.isdisjoint({"player_type", "civilization", "architecture", "personality"})


def test_pop_limit_target_is_primary_then_map_mirror() -> None:
    targets = pf.write_targets(_loaded())
    primary, mirror = targets["player:pop_limit:1"]
    assert primary.codec == "f32"
    assert mirror.codec in ("u32", "s32")


def test_food_target_is_primary_then_player_data_4_mirror() -> None:
    targets = pf.write_targets(_loaded())
    primary, mirror = targets["player:food:1"]
    assert primary.codec == "s32"
    assert mirror.codec == "f32"


def test_color_target_is_primary_then_player_data_3_mirror() -> None:
    targets = pf.write_targets(_loaded())
    primary, mirror = targets["player:color:1"]
    assert primary.codec == "s32"
    assert mirror.codec == "u32"


def test_every_players_target_offset_is_within_the_body_and_disjoint_per_player() -> None:
    loaded = _loaded()
    targets = pf.write_targets(loaded)
    body = loaded.decompressed_body
    spans = []
    for key, entries in targets.items():
        for t in entries:
            assert 0 <= t.offset and t.offset + t.length <= len(body), key
            spans.append((t.offset, t.offset + t.length))
    spans.sort()
    for (a_start, a_end), (b_start, b_end) in zip(spans, spans[1:]):
        assert a_end <= b_start, "two targets overlap"


# -- 2. mutation-verify each guard -------------------------------------------


def test_corrupting_the_gaia_sentinel_flips_verify_to_false() -> None:
    """The GAIA (index 8) slot is deliberately never written, but is
    checked for every P1_TO_P8_THEN_GAIA spec -- the cheapest real shift
    detector, since P1..P8 alone can compare equal at a shifted offset."""
    loaded = _loaded()
    spec = next(s for s in pf.specs_for(loaded) if s.field_id == "starting_age")
    gaia = pf._array_target(loaded, spec.section, spec.retriever, spec.struct_field, 8)
    body = bytearray(loaded.decompressed_body)
    body[gaia.target.offset] ^= 0xFF
    loaded.decompressed_body = bytes(body)
    assert not pf.verify_player_block(loaded)


def test_corrupting_a_mirror_byte_flips_verify_to_false() -> None:
    loaded = _loaded()
    mirror = pf._resolve_all(loaded)["player:food:1"][1].target
    body = bytearray(loaded.decompressed_body)
    body[mirror.offset] ^= 0xFF
    loaded.decompressed_body = bytes(body)
    assert not pf.verify_player_block(loaded)


def test_corrupting_the_player_data_3_color_mirror_flips_verify_to_false() -> None:
    loaded = _loaded()
    color_mirror = pf._resolve_all(loaded)["player:color:2"][1].target
    body = bytearray(loaded.decompressed_body)
    body[color_mirror.offset] ^= 0xFF
    loaded.decompressed_body = bytes(body)
    assert not pf.verify_player_block(loaded)


def test_corrupting_the_player_data_two_anchor_flips_verify_to_false() -> None:
    loaded = _loaded()
    loaded.player_data_two_section_end += 4
    assert not pf.verify_player_block(loaded)


def test_units_write_supported_false_blocks_the_whole_block() -> None:
    loaded = _loaded()
    loaded.units_write_supported = False
    loaded.units_block_offset = -1
    assert not pf.verify_player_block(loaded)


# -- 3. encode_target() refusals ----------------------------------------------


def test_encode_target_refuses_a_non_f32_exact_value() -> None:
    target = pf.PlayerWriteTarget(0, 4, "f32")
    with pytest.raises(ValueError):
        pf.encode_target(target, 2**24 + 1)
    pf.encode_target(target, 2**24)  # exact, must not raise


def test_encode_target_refuses_text_that_overflows_a_c256_slot() -> None:
    target = pf.PlayerWriteTarget(0, 8, "c256")
    with pytest.raises(ValueError):
        pf.encode_target(target, "waytoolongforthisslot")


def test_encode_target_pads_short_text_with_nul() -> None:
    target = pf.PlayerWriteTarget(0, 8, "c256")
    assert pf.encode_target(target, "hi") == b"hi\x00\x00\x00\x00\x00\x00"


def test_encode_target_round_trips_every_numeric_codec() -> None:
    for codec, value in (("u8", 200), ("s32", -5), ("u32", 5), ("f32", 12.0)):
        target = pf.PlayerWriteTarget(0, 4, codec)
        encoded = pf.encode_target(target, value)
        (decoded,) = pf._CODEC_STRUCT[codec].unpack(encoded)
        assert decoded == value


# -- 4. corpus ----------------------------------------------------------------


@pytest.mark.corpus
def test_verify_player_block_across_the_corpus(scenario_path) -> None:
    loaded = load_map_and_units(scenario_path)
    if not loaded.units_write_supported:
        pytest.skip(f"{scenario_path.name}: units block fails verification, so no player write path")
    assert pf.verify_player_block(loaded), scenario_path.name


@pytest.mark.corpus
def test_write_targets_field_count_matches_specs_for_across_the_corpus(scenario_path) -> None:
    """Every Tier-1 spec this file's version actually stores gets a target
    for every player 1..8, and nothing else -- pins the version-gated field
    count (e.g. 1.37/1.41 lack per_player_lock_personality/base_priority's
    later siblings) against specs_for()'s own presence check."""
    loaded = load_map_and_units(scenario_path)
    if not loaded.units_write_supported:
        pytest.skip(f"{scenario_path.name}: units block fails verification, so no player write path")
    if not pf.verify_player_block(loaded):
        pytest.skip(f"{scenario_path.name}: player block fails verification on this file")
    targets = pf.write_targets(loaded)
    tier1_specs = {
        s.field_id for s in pf.specs_for(loaded) if s.field_id not in pf._NEVER_WRITABLE
    }
    field_ids = {key.split(":")[1] for key in targets}
    assert field_ids == tier1_specs, scenario_path.name
    for field_id in field_ids:
        players = {key.split(":")[2] for key in targets if key.split(":")[1] == field_id}
        assert players == {str(p) for p in range(1, 9)}, (scenario_path.name, field_id)
