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

from itertools import pairwise
from pathlib import Path

import pytest

from descape import player_fields as pf
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units

# 1.41 -- pre-1.56, so civilization/architecture are the plain u32 form Step
# A makes writable. Both blank donors below are 1.56+ (str16), so neither can
# exercise this half of the write path at all.
_PRE_1_56_PATH = Path(__file__).resolve().parent.parent / "examples" / "C2_ElCid_coop_1_v0_16.aoe2scenario"
_V158_DONOR_PATH = Path(__file__).resolve().parent / "fixtures" / "blank_120x120_v158.aoe2scenario"


@pytest.fixture(params=[BLANK_TEMPLATE_PATH, _V158_DONOR_PATH], ids=["template-1.59", "donor-1.58"])
def _loaded(request):
    """A fresh load per call, on each blank donor in turn: the shipped 1.59
    template and the real 1.58 export it is generated from."""
    return lambda: load_map_and_units(request.param)


# -- 1. every offset verifies byte-for-byte ----------------------------------


def test_verify_player_block_passes_on_a_clean_load(_loaded) -> None:
    assert pf.verify_player_block(_loaded())


def test_write_targets_is_empty_until_verify_passes(_loaded) -> None:
    """write_targets() is only meaningful once verify_player_block() has
    returned True -- a caller (options_model.py, step 3b) is expected to
    gate on that first, the same shape diplomacy_write_supported() is
    gated ahead of stance_offsets()."""
    loaded = _loaded()
    assert pf.verify_player_block(loaded)
    targets = pf.write_targets(loaded)
    assert targets
    # 17 Tier-1 field_ids * 8 players, on a file new enough to carry all of
    # them (both blank donors are 1.56+) -- civilization/architecture
    # included as of Step B, str16-coded on this file but still writable
    # through the resizing splice. GAIA (0) only for GAIA_FIELDS (GH #187).
    field_ids = {key.split(":")[1] for key in targets}
    assert field_ids == {
        "tribe_name", "player_name_string_id", "lock_civilization", "lock_personality",
        "civilization", "architecture", "starting_age", "food", "wood", "stone", "gold",
        "pop_limit", "color", "human", "base_priority", "initial_view_x", "initial_view_y",
    }
    for field_id in field_ids:
        players = {int(key.split(":")[2]) for key in targets if key.split(":")[1] == field_id}
        expected = set(range(9)) if field_id in pf.GAIA_FIELDS else set(range(1, 9))
        assert players == expected, field_id


def test_write_targets_never_includes_player_type_or_personality(_loaded) -> None:
    targets = pf.write_targets(_loaded())
    field_ids = {key.split(":")[1] for key in targets}
    assert field_ids.isdisjoint({"player_type", "personality"})


def test_write_targets_civilization_is_str16_coded_on_a_1_56_plus_file(_loaded) -> None:
    targets = pf.write_targets(_loaded())  # both blank donors are 1.56+
    (primary,) = targets["player:civilization:1"]
    assert primary.codec == "str16"


def test_pop_limit_target_is_primary_then_map_mirror(_loaded) -> None:
    targets = pf.write_targets(_loaded())
    primary, mirror = targets["player:pop_limit:1"]
    assert primary.codec == "f32"
    assert mirror.codec in ("u32", "s32")


def test_food_target_is_primary_then_player_data_4_mirror(_loaded) -> None:
    targets = pf.write_targets(_loaded())
    primary, mirror = targets["player:food:1"]
    assert primary.codec == "s32"
    assert mirror.codec == "f32"


def test_color_target_is_primary_then_player_data_3_mirror(_loaded) -> None:
    targets = pf.write_targets(_loaded())
    primary, mirror = targets["player:color:1"]
    assert primary.codec == "s32"
    assert mirror.codec == "u32"


def test_every_players_target_offset_is_within_the_body_and_disjoint_per_player(_loaded) -> None:
    loaded = _loaded()
    targets = pf.write_targets(loaded)
    body = loaded.decompressed_body
    spans = []
    for key, entries in targets.items():
        for t in entries:
            assert t.offset >= 0 and t.offset + t.length <= len(body), key
            spans.append((t.offset, t.offset + t.length))
    spans.sort()
    for (_a_start, a_end), (b_start, _b_end) in pairwise(spans):
        assert a_end <= b_start, "two targets overlap"


# -- 2. mutation-verify each guard -------------------------------------------


def test_corrupting_the_gaia_sentinel_flips_verify_to_false(_loaded) -> None:
    """The GAIA (index 8) slot is checked for every P1_TO_P8_THEN_GAIA spec
    -- the cheapest real shift detector, since P1..P8 alone can compare
    equal at a shifted offset. A written target since GH #187."""
    loaded = _loaded()
    spec = next(s for s in pf.specs_for(loaded) if s.field_id == "starting_age")
    gaia = pf._array_target(loaded, spec.section, spec.retriever, spec.struct_field, 8)
    body = bytearray(loaded.decompressed_body)
    body[gaia.target.offset] ^= 0xFF
    loaded.decompressed_body = bytes(body)
    assert not pf.verify_player_block(loaded)


def test_corrupting_the_civilization_gaia_sentinel_flips_verify_to_false(_loaded) -> None:
    """Step B: the str16 GAIA sentinel, reached via
    _player_data_1_variable_target() rather than _array_target() -- both
    blank donors are 1.56+, so civilization's own GAIA-slot check now
    exercises the variable-stride locator, not the fixed-stride one every
    other P1_TO_P8_THEN_GAIA spec above uses."""
    loaded = _loaded()
    gaia = pf._player_data_1_variable_target(loaded, "civilization", 0)
    assert gaia is not None
    body = bytearray(loaded.decompressed_body)
    body[gaia.target.offset] ^= 0xFF  # corrupts the length prefix's low byte
    loaded.decompressed_body = bytes(body)
    assert not pf.verify_player_block(loaded)


def test_corrupting_a_civilization_length_prefix_flips_verify_to_false(_loaded) -> None:
    """A wrong entry offset in a variable-stride array lands mid-string
    rather than on a plausible integer -- the maintainer plan's own claim
    for why this is a *stronger* shift detector than the fixed-width GAIA
    sentinel. Corrupts P3's own civilization field rather than GAIA's."""
    loaded = _loaded()
    target = pf.write_targets(loaded)["player:civilization:3"][0]
    body = bytearray(loaded.decompressed_body)
    body[target.offset] ^= 0xFF
    loaded.decompressed_body = bytes(body)
    assert not pf.verify_player_block(loaded)


def test_player_data_1_base_span_check_fails_closed_on_a_wrong_byte_length(_loaded) -> None:
    """_data_header_bases()'s span check (_forward_offsets()'s summed
    length vs. DataHeader's own trusted byte_length) is what everything in
    Step B hangs off -- corrupt that trusted figure directly (nothing in
    decompressed_body can desync it, so this is the only way to exercise
    the guard) and confirm every str16-dependent path fails closed rather
    than trusting a stale offset."""
    loaded = _loaded()
    section = loaded._scenario.sections["DataHeader"]
    section.byte_length += 4
    assert pf._data_header_bases(loaded) is None
    assert pf._player_data_1_variable_target(loaded, "civilization", 1) is None
    assert not pf.verify_player_block(loaded)
    assert pf.player_data_1_splice(loaded, [(1, "civilization", "HUN-CIV")]) is None


def test_corrupting_a_mirror_byte_flips_verify_to_false(_loaded) -> None:
    loaded = _loaded()
    mirror = pf._resolve_all(loaded)["player:food:1"][1].target
    body = bytearray(loaded.decompressed_body)
    body[mirror.offset] ^= 0xFF
    loaded.decompressed_body = bytes(body)
    assert not pf.verify_player_block(loaded)


def test_corrupting_the_player_data_3_color_mirror_flips_verify_to_false(_loaded) -> None:
    loaded = _loaded()
    color_mirror = pf._resolve_all(loaded)["player:color:2"][1].target
    body = bytearray(loaded.decompressed_body)
    body[color_mirror.offset] ^= 0xFF
    loaded.decompressed_body = bytes(body)
    assert not pf.verify_player_block(loaded)


def test_corrupting_the_player_data_two_anchor_flips_verify_to_false(_loaded) -> None:
    loaded = _loaded()
    loaded.player_data_two_section_end += 4
    assert not pf.verify_player_block(loaded)


def test_units_write_supported_false_blocks_the_whole_block(_loaded) -> None:
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


def test_encode_target_round_trips_str16() -> None:
    """Byte-for-byte against AoE2ScenarioParser's own parse_val_to_bytes()
    -- confirmed 2026-09-02 for HUN-CIV, BRITON-CIV, and the ungrounded
    MIRROR-RANDOM-CIV (17 bytes, the longest member)."""
    target = pf.PlayerWriteTarget(0, 8, "str16")
    assert pf.encode_target(target, "HUN-CIV") == b"\x07\x00HUN-CIV"
    assert pf.encode_target(target, "BRITON-CIV") == b"\n\x00BRITON-CIV"
    assert pf.encode_target(target, "MIRROR-RANDOM-CIV") == b"\x11\x00MIRROR-RANDOM-CIV"


def test_encode_target_refuses_a_non_civilization_str16_value() -> None:
    target = pf.PlayerWriteTarget(0, 8, "str16")
    with pytest.raises(ValueError):
        pf.encode_target(target, "NOT-A-REAL-CIV")
    with pytest.raises(ValueError):
        pf.encode_target(target, 5)  # not a str at all


# -- Step A: civilization/architecture become writable below 1.56 ------------


@pytest.mark.corpus
def test_civilization_and_architecture_are_writable_below_1_56() -> None:
    """The plan's headline claim for Step A: pre-1.56 files need no new
    write machinery, only the _NEVER_WRITABLE relaxation -- pin it against
    a real file rather than trusting write_targets()'s field-count check
    alone. Also the regression test for player_data_1 needing its own
    entry in _DATA_HEADER_WANTED: without it, _array_target() returns None
    for civilization/architecture on every file regardless of version,
    since it looks up bases["player_data_1"], not bases["civilization"]."""
    if not _PRE_1_56_PATH.exists():
        pytest.skip("examples/ corpus not present")
    loaded = load_map_and_units(_PRE_1_56_PATH)
    assert float(loaded.scenario_version) < 1.56, loaded.scenario_version
    assert pf.verify_player_block(loaded)
    targets = pf.write_targets(loaded)
    for field_id in ("civilization", "architecture"):
        target = targets[f"player:{field_id}:1"][0]
        assert target.codec == "u32"


@pytest.mark.corpus
def test_a_civilization_edit_below_1_56_round_trips_through_write_scenario(tmp_path) -> None:
    """The AGENTS.md before/after-artifact pattern, applied in-test rather
    than to a build/ file: a real edit through the real write path
    (OptionsEditModel + write_scenario()), reloaded and read back -- not
    just the offset math test_civilization_and_architecture_are_writable_
    below_1_56() already covers."""
    from descape.options_model import OptionsEditModel
    from descape.player_fields import specs_for
    from descape.scenario_write import write_scenario

    if not _PRE_1_56_PATH.exists():
        pytest.skip("examples/ corpus not present")
    loaded = load_map_and_units(_PRE_1_56_PATH)
    model = OptionsEditModel(loaded)
    before = model.original_value("player:civilization:1")
    after = 5  # CivilizationOld.BRITONS -- deliberately different from before
    assert before != after
    model.set_value("player:civilization:1", after)

    out = tmp_path / "out.aoe2scenario"
    write_scenario(loaded, out, options=model)

    reloaded = load_map_and_units(out)
    spec = next(s for s in specs_for(reloaded) if s.field_id == "civilization")
    assert pf.current_value(reloaded, spec, 1) == after
    assert pf.verify_player_block(reloaded)


# -- GH #126 follow-ups: Player Type (player_data_1.human) --------------------


def mixed_length_player_data_1(tmp_path: Path) -> Path:
    """The blank template with P2's civ set to HUN-CIV through the write path:
    player_data_1's entries then differ in length (P2's is 3 bytes shorter), so
    _array_target()'s single stride is wrong there and only the variable route
    reaches `human`. The template itself has all 16 entries at one length."""
    from descape.options_model import OptionsEditModel
    from descape.scenario_write import write_scenario

    loaded = load_map_and_units(BLANK_TEMPLATE_PATH)
    model = OptionsEditModel(loaded)
    model.set_value("player:civilization:2", "HUN-CIV")
    out = tmp_path / "mixed_player_data_1.aoe2scenario"
    write_scenario(loaded, out, options=model, backup=False)
    entries = load_map_and_units(out)._scenario.sections["DataHeader"].retriever_map["player_data_1"].data
    assert len({e.byte_length for e in entries}) > 1, "the fixture's entries all have one length"
    return out


def test_human_targets_sit_right_after_active_on_a_mixed_length_file(tmp_path: Path) -> None:
    """`human` is the u32 after `active`, whose eight targets Number of Players
    already trusts; the bytes there decode to the parsed value for P1..P8."""
    loaded = load_map_and_units(mixed_length_player_data_1(tmp_path))
    spec = next(s for s in pf.specs_for(loaded) if s.field_id == "human")
    assert pf._array_target(loaded, spec.section, spec.retriever, spec.struct_field, 2) is None
    assert pf.verify_player_block(loaded)
    targets = pf.write_targets(loaded)
    active = pf.player_count_targets(loaded)
    body = loaded.decompressed_body
    for player_id in range(1, 9):
        (target,) = targets[f"player:human:{player_id}"]
        assert target.codec == "u32"
        assert target.offset == active[player_id - 1].offset + 4, player_id
        stored = pf.current_value(loaded, spec, player_id)
        assert body[target.offset : target.offset + 4] == stored.to_bytes(4, "little"), player_id
    assert "player:human:0" not in targets


def test_corrupting_the_human_gaia_sentinel_flips_verify_to_false(tmp_path: Path) -> None:
    loaded = load_map_and_units(mixed_length_player_data_1(tmp_path))
    gaia = pf._player_data_1_variable_target(loaded, "human", 0)
    body = bytearray(loaded.decompressed_body)
    body[gaia.target.offset] ^= 0xFF
    loaded.decompressed_body = bytes(body)
    assert not pf.verify_player_block(loaded)


@pytest.mark.corpus
def test_human_is_a_u32_target_below_1_56() -> None:
    if not _PRE_1_56_PATH.exists():
        pytest.skip("examples/ corpus not present")
    loaded = load_map_and_units(_PRE_1_56_PATH)
    assert float(loaded.scenario_version) < 1.56, loaded.scenario_version
    assert pf.verify_player_block(loaded)
    targets = pf.write_targets(loaded)
    active = pf.player_count_targets(loaded)
    for player_id in range(1, 9):
        (target,) = targets[f"player:human:{player_id}"]
        assert target.codec == "u32"
        assert target.offset == active[player_id - 1].offset + 4, player_id


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
    tier1_specs = {s.field_id for s in pf.specs_for(loaded) if s.field_id not in pf._NOT_BYTE_PATCHED}
    field_ids = {key.split(":")[1] for key in targets}
    assert field_ids == tier1_specs, scenario_path.name
    gaia_ids = {s.field_id for s in pf.specs_for_player(loaded, 0)}
    for field_id in field_ids:
        players = {int(key.split(":")[2]) for key in targets if key.split(":")[1] == field_id}
        expected = set(range(9)) if field_id in gaia_ids else set(range(1, 9))
        assert players == expected, (scenario_path.name, field_id)


@pytest.mark.corpus
def test_reencoding_the_stored_civ_value_is_byte_identical_across_the_corpus(scenario_path) -> None:
    """The maintainer plan's own "cheap correctness guard, worth a test on
    its own": re-encode()ing every 1.56+ entry's *existing* civilization/
    architecture value must reproduce the exact bytes already on disk --
    catches prefix width, signedness, encoding, and NUL-trail in one
    assertion, before this write path is ever trusted to write a *changed*
    value. verify_player_block()'s own _matches() already does exactly
    this per-target internally; this test pins it explicitly and by name
    so a future refactor that weakens the check gets caught here too, not
    just via the corpus-wide verify() pass."""
    loaded = load_map_and_units(scenario_path)
    if not loaded.units_write_supported:
        pytest.skip(f"{scenario_path.name}: units block fails verification")
    targets = pf.write_targets(loaded)
    specs_by_id = {s.field_id: s for s in pf.specs_for(loaded)}
    body = loaded.decompressed_body
    for field_id in ("civilization", "architecture"):
        for player_id in range(1, 9):
            target = targets.get(f"player:{field_id}:{player_id}")
            if target is None or target[0].codec != "str16":
                continue
            (primary,) = target
            stored_value = pf.current_value(loaded, specs_by_id[field_id], player_id)
            reencoded = pf.encode_target(primary, stored_value)
            assert body[primary.offset : primary.offset + primary.length] == reencoded, (
                scenario_path.name, field_id, player_id,
            )


# -- GAIA (GH #187) -----------------------------------------------------------
#
# GAIA_FIELDS write GAIA's own slot: index 8 of every P1_TO_P8_THEN_GAIA
# array, and Map.per_player_population_cap[8] for Pop Limit. No mirror.

# A distinct value per field; None toggles a checkbox. HUN-CIV is shorter than
# the template's RANDOM-CIV (a real resize); BRITON-CIV is the same length.
_GAIA_EDITS = {
    "civilization": "BRITON-CIV", "architecture": "HUN-CIV", "lock_civilization": None,
    "lock_personality": None, "starting_age": 5, "food": 4321, "wood": 1234, "stone": 555,
    "gold": 666, "color": 3, "pop_limit": 123,
}


def _written_body(path: Path) -> bytes:
    from AoE2ScenarioParser.scenarios.aoe2_scenario import _decompress_bytes

    raw = path.read_bytes()
    return _decompress_bytes(raw[len(load_map_and_units(path).header_bytes) :])


def _gaia_new_value(model, field_id: str):
    stored = model.original_value(pf.player_field_id(field_id, 0))
    value = _GAIA_EDITS[field_id]
    if value is None:
        value = 1 - stored
    assert value != stored, field_id
    return value


def test_gaia_fields_get_one_target_at_gaias_slot_and_no_mirror(_loaded) -> None:
    loaded = _loaded()
    resolved = pf._resolve_all(loaded)
    specs = {s.field_id: s for s in pf.specs_for(loaded)}
    for field_id in pf.GAIA_FIELDS:
        (gaia,) = resolved[pf.player_field_id(field_id, 0)]
        spec = specs[field_id]
        if field_id == "pop_limit":
            expected = pf._array_target(loaded, "Map", "per_player_population_cap", None, 8)
        elif field_id in ("civilization", "architecture"):
            expected = pf._player_data_1_variable_target(loaded, spec.struct_field, 0)
        else:
            expected = pf._array_target(loaded, spec.section, spec.retriever, spec.struct_field, 8)
        assert gaia.target == expected.target, field_id
        assert gaia.parsed_value == pf.current_value(loaded, spec, 0), field_id
    others = {s.field_id for s in pf.specs_for(loaded)} - pf.GAIA_FIELDS
    assert not any(pf.player_field_id(f, 0) in resolved for f in others)


def test_gaia_pop_limit_targets_the_map_array_not_player_data_4(_loaded) -> None:
    loaded = _loaded()
    (target,) = pf.write_targets(loaded)["player:pop_limit:0"]
    map_base = pf._map_bases(loaded)["per_player_population_cap"]
    assert target.codec == "u32"
    assert target.offset == map_base.offset + 8 * target.length
    pd4 = pf._units_bases(loaded)["player_data_4"]
    assert not pd4.offset <= target.offset < pd4.offset + pd4.length


@pytest.mark.parametrize("field_id", sorted(pf.GAIA_FIELDS))
def test_a_gaia_edit_changes_only_gaias_slot(field_id: str, tmp_path: Path, _loaded) -> None:
    """Every byte outside GAIA's own target is unchanged, which covers P1..P8,
    Units.player_data_4 and Map's other pop-cap slots; then a reload reads it back."""
    from descape.options_model import OptionsEditModel
    from descape.scenario_write import write_scenario

    loaded = _loaded()
    key = pf.player_field_id(field_id, 0)
    (target,) = pf.write_targets(loaded)[key]
    base = tmp_path / "base.aoe2scenario"
    write_scenario(loaded, base)

    model = OptionsEditModel(loaded)
    value = _gaia_new_value(model, field_id)
    model.set_value(key, value)
    edited = tmp_path / "edited.aoe2scenario"
    write_scenario(loaded, edited, options=model)

    before, after = _written_body(base), _written_body(edited)
    assert before == loaded.decompressed_body
    new_bytes = pf.encode_target(target, value)
    end = target.offset + target.length
    assert after[: target.offset] == before[: target.offset]
    assert after[target.offset : target.offset + len(new_bytes)] == new_bytes
    assert after[target.offset + len(new_bytes) :] == before[end:]
    if target.codec != "str16":
        assert len(new_bytes) == target.length

    reloaded = load_map_and_units(edited)
    assert pf.verify_player_block(reloaded)
    spec = next(s for s in pf.specs_for(reloaded) if s.field_id == field_id)
    assert pf.current_value(reloaded, spec, 0) == value
    for player_id in range(1, 9):
        assert pf.current_value(reloaded, spec, player_id) == model.original_value(
            pf.player_field_id(field_id, player_id)
        ), player_id
    delta = len(after) - len(before)
    pd4 = pf._units_bases(loaded)["player_data_4"]
    assert after[pd4.offset + delta : pd4.offset + delta + pd4.length] == before[pd4.offset : pd4.offset + pd4.length]


def test_corrupting_gaias_pop_limit_slot_flips_verify_to_false(_loaded) -> None:
    loaded = _loaded()
    (target,) = pf.write_targets(loaded)["player:pop_limit:0"]
    body = bytearray(loaded.decompressed_body)
    body[target.offset] ^= 0xFF
    loaded.decompressed_body = bytes(body)
    assert not pf.verify_player_block(loaded)


def test_verify_player_block_catches_a_shift_only_gaias_slot_can_see(monkeypatch, _loaded) -> None:
    """Every slot but GAIA's holds Dark Age, and so do the 4 bytes before the
    array, so a walk shifted back one slot still matches P1..P8. Only GAIA's
    own check can fail it."""
    loaded = _loaded()
    ages = loaded._scenario.sections["Options"].retriever_map["per_player_starting_age"]
    base = pf._options_bases(loaded)["per_player_starting_age"]
    stride = base.length // len(ages.data)
    codec = pf.write_targets(loaded)["player:starting_age:1"][0].codec
    dark, imperial = 2, 5
    assert set(ages.data) == {dark}
    body = bytearray(loaded.decompressed_body)
    body[base.offset - stride : base.offset] = pf.encode_target(pf.PlayerWriteTarget(0, stride, codec), dark)
    gaia_offset = base.offset + 8 * stride
    body[gaia_offset : gaia_offset + stride] = pf.encode_target(pf.PlayerWriteTarget(0, stride, codec), imperial)
    loaded.decompressed_body = bytes(body)
    ages.data[8] = imperial
    assert pf.verify_player_block(loaded), "the crafted file must verify unshifted"

    real_bases = pf._options_bases

    def shifted(arg):
        bases = dict(real_bases(arg))
        ages_base = bases["per_player_starting_age"]
        bases["per_player_starting_age"] = pf.FieldOffset(ages_base.offset - stride, ages_base.length)
        return bases

    monkeypatch.setitem(pf._SECTION_BASES, "Options", shifted)
    resolved = pf._resolve_all(loaded)
    for player_id in range(1, 9):
        (item,) = resolved[f"player:starting_age:{player_id}"]
        t = item.target
        assert loaded.decompressed_body[t.offset : t.offset + t.length] == pf.encode_target(t, item.parsed_value)
    assert not pf.verify_player_block(loaded)


def test_edit_gaia_save_edit_again_save_again(tmp_path: Path, _loaded) -> None:
    """Both saves patch a copy, so the open document keeps verifying and the
    second save carries the second values (a str16 resize each time)."""
    from descape.options_model import OptionsEditModel
    from descape.scenario_write import write_scenario

    loaded = _loaded()
    model = OptionsEditModel(loaded)
    model.set_value("player:civilization:0", "BRITON-CIV")
    model.set_value("player:food:0", 4321)
    first = tmp_path / "first.aoe2scenario"
    write_scenario(loaded, first, options=model)
    assert pf.verify_player_block(loaded)

    model.set_value("player:civilization:0", "HUN-CIV")
    model.set_value("player:food:0", 999)
    model.set_value("player:pop_limit:0", 123)
    second = tmp_path / "second.aoe2scenario"
    write_scenario(loaded, second, options=model)
    assert pf.verify_player_block(loaded)

    for path, expected in (
        (first, {"civilization": "BRITON-CIV", "food": 4321, "pop_limit": 200}),
        (second, {"civilization": "HUN-CIV", "food": 999, "pop_limit": 123}),
    ):
        reloaded = load_map_and_units(path)
        assert pf.verify_player_block(reloaded), path.name
        specs = {s.field_id: s for s in pf.specs_for(reloaded)}
        for field_id, value in expected.items():
            assert pf.current_value(reloaded, specs[field_id], 0) == value, (path.name, field_id)
            assert pf.current_value(reloaded, specs[field_id], 1) == model.original_value(f"player:{field_id}:1")


@pytest.mark.corpus
@pytest.mark.parametrize("new_civ", ["BRITON-CIV", "HUN-CIV"])
def test_a_gaia_civilization_splice_on_a_real_1_59_file(new_civ: str, tmp_path: Path) -> None:
    """8tp3w9j's GAIA is ROMANS-CIV. BRITON-CIV is the same length; HUN-CIV is
    3 bytes shorter, so every later offset moves."""
    from descape.options_model import OptionsEditModel
    from descape.scenario_write import write_scenario

    path = Path(__file__).resolve().parent.parent / "examples" / "8tp3w9j.aoe2scenario"
    if not path.exists():
        pytest.skip("examples/ corpus not present")
    loaded = load_map_and_units(path)
    (target,) = pf.write_targets(loaded)["player:civilization:0"]
    assert target.codec == "str16"
    model = OptionsEditModel(loaded)
    assert model.original_value("player:civilization:0") == "ROMANS-CIV"
    model.set_value("player:civilization:0", new_civ)
    out = tmp_path / "out.aoe2scenario"
    write_scenario(loaded, out, options=model)

    after = _written_body(out)
    before = loaded.decompressed_body
    new_bytes = pf.encode_target(target, new_civ)
    assert (len(new_bytes) == target.length) == (new_civ == "BRITON-CIV")
    assert after[: target.offset] == before[: target.offset]
    assert after[target.offset : target.offset + len(new_bytes)] == new_bytes
    assert after[target.offset + len(new_bytes) :] == before[target.offset + target.length :]
    reloaded = load_map_and_units(out)
    assert pf.verify_player_block(reloaded)
    spec = next(s for s in pf.specs_for(reloaded) if s.field_id == "civilization")
    assert pf.current_value(reloaded, spec, 0) == new_civ
    assert [pf.current_value(reloaded, spec, p) for p in range(1, 9)] == [
        model.original_value(f"player:civilization:{p}") for p in range(1, 9)
    ]


# -- Number of Players (step 3e) ---------------------------------------------
#
# Its own gate, separate from verify_player_block() above: it needs
# FileHeader.player_count's offset as well as the eight `active` flags, and
# that header walk fails on a real corpus file. Same mutation discipline as
# every guard above -- each check is broken deliberately and confirmed to
# flip the gate red, since a guard that never fires is not a guard.


def _shift_header_span(loaded, delta: int) -> None:
    start, end = loaded.header_player_count_span
    loaded.header_player_count_span = (start + delta, end + delta)


def test_verify_player_count_block_passes_on_a_clean_load(_loaded) -> None:
    assert pf.verify_player_count_block(_loaded())


def test_player_count_targets_are_eight_u32_active_flags_in_player_order(_loaded) -> None:
    """`active` is player_data_1's first retriever, so P(n+1)'s flag always
    sits after P(n)'s -- ascending offsets pin that the tuple is in player
    order and not, say, reversed by the variable-stride walk."""
    loaded = _loaded()
    targets = pf.player_count_targets(loaded)
    assert targets is not None
    assert len(targets) == pf.NUM_PLAYERS
    assert {t.codec for t in targets} == {"u32"}
    assert [t.offset for t in targets] == sorted(t.offset for t in targets)


def test_encode_player_count_writes_a_prefix_of_ones(_loaded) -> None:
    loaded = _loaded()
    targets = pf.player_count_targets(loaded)
    encoded = pf.encode_player_count(targets, 3)
    assert [int.from_bytes(b, "little") for b in encoded] == [1, 1, 1, 0, 0, 0, 0, 0]


def test_a_corrupted_active_flag_flips_the_count_gate_to_false(_loaded) -> None:
    loaded = _loaded()
    target = pf.player_count_targets(loaded)[0]
    body = bytearray(loaded.decompressed_body)
    body[target.offset] ^= 0xFF
    loaded.decompressed_body = bytes(body)
    assert not pf.verify_player_count_block(loaded)


def test_an_unresolvable_header_span_flips_the_count_gate_to_false(_loaded) -> None:
    """The real failure mode, not a synthetic one: the FileHeader forward
    walk does not reconcile on a scenario version 1.37 corpus file, which
    is what leaves Number of Players read-only there while every other
    player row on that same file stays editable."""
    loaded = _loaded()
    loaded.header_player_count_span = (-1, -1)
    assert not pf.verify_player_count_block(loaded)


def test_a_header_span_shifted_by_one_byte_flips_the_count_gate_to_false(_loaded) -> None:
    """The plan's own "break the offset by one byte" check for the one
    guard that had no coverage at all before this step."""
    loaded = _loaded()
    _shift_header_span(loaded, 1)
    assert not pf.verify_player_count_block(loaded)


def test_a_header_count_disagreeing_with_the_active_flags_flips_the_gate(_loaded) -> None:
    """The coupling is the premise the write rests on (one value, two
    buffers). A file whose two stored copies already disagreed would have
    one of them silently "corrected" by any edit here, so it is refused."""
    loaded = _loaded()
    start, end = loaded.header_player_count_span
    header = bytearray(loaded.header_bytes)
    header[start:end] = (7).to_bytes(end - start, "little")
    loaded.header_bytes = bytes(header)
    assert pf.defined_player_count(loaded) != 7
    assert not pf.verify_player_count_block(loaded)


def test_defined_player_count_matches_the_known_blank_template_default(_loaded) -> None:
    """Moved here from tests/test_diplomacy_fields.py with the function
    itself in step 3e."""
    assert pf.defined_player_count(_loaded()) == 2
    assert pf.defined_player_ids(_loaded()) == [1, 2]


def test_a_pending_count_overrides_the_stored_flags_and_yields_a_prefix(_loaded) -> None:
    """The pending branch's own guarantee, which deliberately differs from
    the stored one: a count is a count, so committing one rewrites the
    flags into a contiguous prefix."""
    loaded = _loaded()
    pending = {pf.PLAYER_COUNT_FIELD_ID: 5}
    assert pf.defined_player_ids(loaded, pending) == [1, 2, 3, 4, 5]
    assert pf.defined_player_count(loaded, pending) == 5


def test_pending_edits_that_are_not_the_count_leave_the_stored_read_alone(_loaded) -> None:
    loaded = _loaded()
    pending = {pf.player_field_id("food", 3): 500, "stance:1:2": 0}
    assert pf.defined_player_ids(loaded, pending) == [1, 2]


@pytest.mark.corpus
def test_verify_player_count_block_across_the_corpus(corpus_files) -> None:
    """Not-vacuous on two axes at once: the gate must actually refuse at
    least one real file (the 1.37 one, whose header walk does not
    reconcile), and the counts it does resolve must genuinely vary rather
    than all landing on 8."""
    passed, refused, counts = 0, 0, set()
    for path in corpus_files:
        loaded = load_map_and_units(str(path))
        if pf.verify_player_count_block(loaded):
            passed += 1
            count = pf.defined_player_count(loaded)
            assert 2 <= count <= pf.NUM_PLAYERS, path.name
            counts.add(count)
        else:
            refused += 1
    assert passed, "no corpus file passed the Number of Players gate"
    assert refused, "the Number of Players gate never fired on the corpus"
    assert len(counts) > 1, f"every corpus file resolved to the same count: {counts}"
