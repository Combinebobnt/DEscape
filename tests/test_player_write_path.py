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

from pathlib import Path

import pytest

from descape import player_fields as pf
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units

# 1.41 -- pre-1.56, so civilization/architecture are the plain u32 form Step
# A makes writable. BLANK_TEMPLATE_PATH itself is 1.58 (str16), so it cannot
# exercise this half of the write path at all.
_PRE_1_56_PATH = Path(__file__).resolve().parent.parent / "examples" / "C2_ElCid_coop_1_v0_16.aoe2scenario"


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
    # 16 Tier-1 field_ids * 8 players, on a file new enough to carry all of
    # them (the blank template is 1.58) -- civilization/architecture
    # included as of Step B, str16-coded on this file but still writable
    # through the resizing splice.
    field_ids = {key.split(":")[1] for key in targets}
    assert field_ids == {
        "tribe_name", "player_name_string_id", "lock_civilization", "lock_personality",
        "civilization", "architecture", "starting_age", "food", "wood", "stone", "gold",
        "pop_limit", "color", "base_priority", "initial_view_x", "initial_view_y",
    }
    assert all(key.split(":")[2] in {str(p) for p in range(1, 9)} for key in targets)


def test_write_targets_never_includes_player_type_or_personality() -> None:
    targets = pf.write_targets(_loaded())
    field_ids = {key.split(":")[1] for key in targets}
    assert field_ids.isdisjoint({"player_type", "personality"})


def test_write_targets_civilization_is_str16_coded_on_a_1_56_plus_file() -> None:
    targets = pf.write_targets(_loaded())  # BLANK_TEMPLATE_PATH is 1.58
    (primary,) = targets["player:civilization:1"]
    assert primary.codec == "str16"


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


def test_corrupting_the_civilization_gaia_sentinel_flips_verify_to_false() -> None:
    """Step B: the str16 GAIA sentinel, reached via
    _player_data_1_variable_target() rather than _array_target() -- BLANK_
    TEMPLATE_PATH is 1.58, so civilization's own GAIA-slot check now
    exercises the variable-stride locator, not the fixed-stride one every
    other P1_TO_P8_THEN_GAIA spec above uses."""
    loaded = _loaded()
    gaia = pf._player_data_1_variable_target(loaded, "civilization", 0)
    assert gaia is not None
    body = bytearray(loaded.decompressed_body)
    body[gaia.target.offset] ^= 0xFF  # corrupts the length prefix's low byte
    loaded.decompressed_body = bytes(body)
    assert not pf.verify_player_block(loaded)


def test_corrupting_a_civilization_length_prefix_flips_verify_to_false() -> None:
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


def test_player_data_1_base_span_check_fails_closed_on_a_wrong_byte_length() -> None:
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
    tier1_specs = {s.field_id for s in pf.specs_for(loaded) if s.field_id not in pf._NEVER_WRITABLE}
    field_ids = {key.split(":")[1] for key in targets}
    assert field_ids == tier1_specs, scenario_path.name
    for field_id in field_ids:
        players = {key.split(":")[2] for key in targets if key.split(":")[1] == field_id}
        assert players == {str(p) for p in range(1, 9)}, (scenario_path.name, field_id)


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


def test_verify_player_count_block_passes_on_a_clean_load() -> None:
    assert pf.verify_player_count_block(_loaded())


def test_player_count_targets_are_eight_u32_active_flags_in_player_order() -> None:
    """`active` is player_data_1's first retriever, so P(n+1)'s flag always
    sits after P(n)'s -- ascending offsets pin that the tuple is in player
    order and not, say, reversed by the variable-stride walk."""
    loaded = _loaded()
    targets = pf.player_count_targets(loaded)
    assert targets is not None
    assert len(targets) == pf.NUM_PLAYERS
    assert {t.codec for t in targets} == {"u32"}
    assert [t.offset for t in targets] == sorted(t.offset for t in targets)


def test_encode_player_count_writes_a_prefix_of_ones() -> None:
    loaded = _loaded()
    targets = pf.player_count_targets(loaded)
    encoded = pf.encode_player_count(targets, 3)
    assert [int.from_bytes(b, "little") for b in encoded] == [1, 1, 1, 0, 0, 0, 0, 0]


def test_a_corrupted_active_flag_flips_the_count_gate_to_false() -> None:
    loaded = _loaded()
    target = pf.player_count_targets(loaded)[0]
    body = bytearray(loaded.decompressed_body)
    body[target.offset] ^= 0xFF
    loaded.decompressed_body = bytes(body)
    assert not pf.verify_player_count_block(loaded)


def test_an_unresolvable_header_span_flips_the_count_gate_to_false() -> None:
    """The real failure mode, not a synthetic one: the FileHeader forward
    walk does not reconcile on a scenario version 1.37 corpus file, which
    is what leaves Number of Players read-only there while every other
    player row on that same file stays editable."""
    loaded = _loaded()
    loaded.header_player_count_span = (-1, -1)
    assert not pf.verify_player_count_block(loaded)


def test_a_header_span_shifted_by_one_byte_flips_the_count_gate_to_false() -> None:
    """The plan's own "break the offset by one byte" check for the one
    guard that had no coverage at all before this step."""
    loaded = _loaded()
    _shift_header_span(loaded, 1)
    assert not pf.verify_player_count_block(loaded)


def test_a_header_count_disagreeing_with_the_active_flags_flips_the_gate() -> None:
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


def test_defined_player_count_matches_the_known_blank_template_default() -> None:
    """Moved here from tests/test_diplomacy_fields.py with the function
    itself in step 3e."""
    assert pf.defined_player_count(_loaded()) == 2
    assert pf.defined_player_ids(_loaded()) == [1, 2]


def test_a_pending_count_overrides_the_stored_flags_and_yields_a_prefix() -> None:
    """The pending branch's own guarantee, which deliberately differs from
    the stored one: a count is a count, so committing one rewrites the
    flags into a contiguous prefix."""
    loaded = _loaded()
    pending = {pf.PLAYER_COUNT_FIELD_ID: 5}
    assert pf.defined_player_ids(loaded, pending) == [1, 2, 3, 4, 5]
    assert pf.defined_player_count(loaded, pending) == 5


def test_pending_edits_that_are_not_the_count_leave_the_stored_read_alone() -> None:
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
