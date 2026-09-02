"""Covers descape/options_model.py: the backward offset walk and the
write-support verification gate for the Map Options panel's byte-patched
scalars.

test_a_single_byte_corruption_flips_verify_to_false is the load-bearing
test here: "a gate that never
fires is not a gate." Everything else pins the offsets themselves against a
real file, which is what caught a real bug during implementation: Map's
anchor (terrain_block_offset) marks where terrain_data *starts*, not the
section's end, and a first version of the backward walk that didn't account
for that landed at offsets tens of thousands of bytes negative.
"""

from __future__ import annotations

import pytest

from descape import option_fields, scenario_io
from descape.options_model import field_offsets, verify_options_block
from descape.scenario_io import retriever_length


def _loaded():
    return scenario_io.load_map_and_units(scenario_io.BLANK_TEMPLATE_PATH)


# -- offsets, against a real file ---------------------------------------------


def test_every_mapped_field_verifies_on_the_blank_template() -> None:
    loaded = _loaded()
    specs = option_fields.specs_for(loaded)
    offsets, all_available = field_offsets(loaded, specs)

    assert all_available
    assert len(offsets) == len(specs)
    assert verify_options_block(loaded, specs)


def test_every_offset_reproduces_its_retriever_s_own_bytes() -> None:
    """The check verify_options_block() does internally, asserted directly
    per field so a failure names which one instead of just "False"."""
    loaded = _loaded()
    specs = option_fields.specs_for(loaded)
    offsets, _ = field_offsets(loaded, specs)
    body = loaded.decompressed_body

    for spec, fo in offsets.items():
        retriever = loaded._scenario.sections[spec.section].retriever_map[spec.retriever]
        assert body[fo.offset : fo.offset + fo.length] == retriever.get_data_as_bytes(), spec.field_id


def test_the_global_victory_anchor_walks_across_the_deferred_custom_fields() -> None:
    """The three live Global Victory specs (mode, required_score_..., and
    time_for_timed_game_...) sit at the very end of the section -- the
    shallowest possible walk. The four still-deferred custom-victory fields
    sit earlier, and nothing in specs_for() exercises reaching past them.

    Declared locally rather than added to option_fields._SPECS, because the
    custom-victory deferral is deliberate and a spec added "just for a test"
    would put those rows back in the panel by accident.
    """
    from descape.option_fields import COMBO, SPINBOX, OptionFieldSpec

    # Two fields at opposite ends of the section, so the walk has to traverse
    # it rather than land on its last retriever twice.
    specs = (
        OptionFieldSpec(
            "gv_mode", "Victory condition", "Global Victory",
            "GlobalVictory", "mode", COMBO, choices=((0, "Standard"),),
        ),
        OptionFieldSpec(
            "gv_all_custom", "Require all custom", "Global Victory",
            "GlobalVictory", "all_custom_conditions_required", SPINBOX,
            minimum=0, maximum=1,
        ),
    )
    loaded = _loaded()
    offsets, all_available = field_offsets(loaded, specs)

    assert all_available
    assert len(offsets) == 2
    assert verify_options_block(loaded, specs)
    body = loaded.decompressed_body
    for spec, fo in offsets.items():
        retriever = loaded._scenario.sections["GlobalVictory"].retriever_map[spec.retriever]
        assert body[fo.offset : fo.offset + fo.length] == retriever.get_data_as_bytes(), spec.field_id
    # And they really are different offsets, so this is a walk and not two
    # reads of the same anchor.
    assert len({fo.offset for fo in offsets.values()}) == 2


def test_offsets_within_a_section_are_distinct_and_dont_overlap() -> None:
    loaded = _loaded()
    specs = option_fields.specs_for(loaded)
    offsets, _ = field_offsets(loaded, specs)

    by_section: dict[str, list] = {}
    for spec, fo in offsets.items():
        by_section.setdefault(spec.section, []).append((fo.offset, fo.length))

    for section, spans in by_section.items():
        spans.sort()
        for (off_a, len_a), (off_b, _len_b) in zip(spans, spans[1:]):
            assert off_a + len_a <= off_b, section


def test_legacy_exec_order_is_never_offset_by_this_module() -> None:
    """Its read/write path is TriggerEditModel, not options_model.py -- see
    that field's own design decision. field_offsets()
    must silently skip it rather than raising on a section with no anchor."""
    loaded = _loaded()
    scenario_io.parse_triggers(loaded)
    specs = option_fields.specs_for(loaded)
    assert any(s.field_id == "legacy_exec_order" for s in specs)  # sanity: it's in scope

    offsets, all_available = field_offsets(loaded, specs)
    assert all_available  # its absence from _ANCHOR_ATTR must not count against this
    assert not any(spec.field_id == "legacy_exec_order" for spec in offsets)


# -- the write-support gate, mutation-tested -----------------------------------


def test_a_single_byte_corruption_flips_verify_to_false() -> None:
    """Proof the gate is load-bearing, not vacuous: an unmutated load
    verifies True, and flipping exactly one byte inside one mapped field's
    own range flips it False."""
    loaded = _loaded()
    specs = option_fields.specs_for(loaded)
    assert verify_options_block(loaded, specs)

    offsets, _ = field_offsets(loaded, specs)
    target_offset = next(iter(offsets.values())).offset
    body = bytearray(loaded.decompressed_body)
    body[target_offset] ^= 0xFF
    loaded.decompressed_body = bytes(body)

    assert not verify_options_block(loaded, specs)


def test_an_unavailable_map_anchor_fails_the_whole_gate_closed() -> None:
    """terrain_write_supported False (a corrupted/unrecognised terrain
    block) already sets terrain_block_offset to -1 in scenario_io.py --
    field_offsets() must report all_available=False rather than silently
    verifying only the sections whose anchors happened to be fine."""
    loaded = _loaded()
    specs = option_fields.specs_for(loaded)
    loaded.terrain_block_offset = -1

    offsets, all_available = field_offsets(loaded, specs)
    assert not all_available
    assert not verify_options_block(loaded, specs)
    # Sections with an unrelated, still-valid anchor are simply not walked --
    # this isn't a partial success, the whole gate reports closed above.
    assert not any(spec.section == "Map" for spec in offsets)


# -- independent cross-check: does the section's full retriever-length sum ---
# -- match its independently-captured span? --------------------------------


def _options_section_length_matches(loaded) -> bool:
    """Sums scenario_io.retriever_length() across *every* Options retriever
    -- including the huge, genuinely variable-length disabled-id arrays this
    module never walks -- and compares against
    options_section_end - diplomacy_section_end, both captured independently
    by data_igen.progress during the original load walk, with no dependency
    on retriever_length() at all.

    This is what settles whether a field missing from specs_for() (e.g.
    ai_map_type on a 1.41 file) is genuinely absent from the file's
    retriever_map -- a real, version-dependent construction difference, not
    a byte-arithmetic near-miss where some other field's length is wrong
    and only happens to cancel out over the fields this module maps.
    """
    rm = loaded._scenario.sections["Options"].retriever_map
    total = sum(retriever_length(r) for r in rm.values())
    return total == loaded.options_section_end - loaded.diplomacy_section_end


def test_options_section_length_is_self_consistent_on_the_blank_template() -> None:
    assert _options_section_length_matches(_loaded())


@pytest.mark.corpus
def test_ai_map_type_is_genuinely_absent_on_a_1_41_file_not_mis_lengthed() -> None:
    """The concrete case that motivated the cross-check above: ai_map_type
    is missing from specs_for() on this file. Confirms it's because the key
    itself is absent from retriever_map (a real per-version construction
    difference in AoE2ScenarioParser, not reproduced in this repo), and that
    every other Options field around it is still lengthed correctly."""
    from tests.conftest import ROOT

    path = ROOT / "examples" / "C2_ElCid_coop_1_v0_16.aoe2scenario"
    if not path.is_file():
        pytest.skip(f"corpus file not present: {path}")
    loaded = scenario_io.load_map_and_units(path)
    rm = loaded._scenario.sections["Options"].retriever_map
    assert "ai_map_type" not in rm
    assert _options_section_length_matches(loaded)
    assert not any(s.field_id == "ai_map_type" for s in option_fields.specs_for(loaded))


# -- corpus: every file maps and verifies, or is honestly explained -----------


@pytest.mark.corpus
def test_field_offsets_verify_across_the_corpus(scenario_path) -> None:
    loaded = scenario_io.load_map_and_units(scenario_path)
    specs = option_fields.specs_for(loaded)
    assert verify_options_block(loaded, specs), scenario_path.name


@pytest.mark.corpus
def test_options_section_length_is_self_consistent_across_the_corpus(scenario_path) -> None:
    loaded = scenario_io.load_map_and_units(scenario_path)
    assert _options_section_length_matches(loaded), scenario_path.name
