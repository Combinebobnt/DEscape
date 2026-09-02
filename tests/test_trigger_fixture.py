"""Guards tests/fixtures/triggers_120x120.aoe2scenario, the default tier's
only trigger-bearing scenario.

Everything the phase 4b write path claims is about what happens to triggers
that *were not* edited, and a file with zero triggers passes every one of those
claims trivially. Until this fixture existed, every file this repo ships had
zero triggers, so the whole per-trigger splice model could only be exercised
against the untracked examples/ corpus.

These tests are about the fixture itself: that its generator is deterministic,
that the file on disk still matches what the generator produces, and that it
carries the specific content the write-path tests depend on. Anything testing
the write path *through* the fixture lives in tests/test_trigger_write_path.py.
"""

from __future__ import annotations

from pathlib import Path

import conftest

from descape.scenario_io import load_map_and_units, parse_triggers
from descape.trigger_model import SECTION_HEADER_SIZE

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"

# ACTIVATE_TRIGGER / DEACTIVATE_TRIGGER, the two effects whose trigger_id a
# reorder or delete has to remap.
_REFERENCE_EFFECT_TYPES = (8, 9)


def _generator():
    return conftest.load_verify_module("gen_trigger_fixture")


def test_fixture_is_tracked_on_disk() -> None:
    assert FIXTURE_PATH.is_file(), (
        f"{FIXTURE_PATH} is missing -- regenerate it with "
        "`.venv/bin/python3 tools/gen_trigger_fixture.py`"
    )


def test_generator_is_deterministic() -> None:
    """Two runs produce identical bytes. Not a given: the generator goes
    through the library's own serializer, and anything order-dependent in
    there would make the committed fixture drift on every regeneration."""
    gen = _generator()
    assert gen.build_fixture_bytes() == gen.build_fixture_bytes()


def test_committed_fixture_matches_its_generator() -> None:
    """The file in git is what the generator produces now. A failure here
    means either the generator changed or the fixture was edited by hand;
    either way, regenerate rather than adjusting this assertion."""
    assert FIXTURE_PATH.read_bytes() == _generator().build_fixture_bytes()


def test_fixture_passes_its_own_verification() -> None:
    """The generator's reload-through-the-real-loader check, run against the
    committed file rather than against a freshly written one."""
    _generator().verify_fixture(FIXTURE_PATH)


def test_fixture_carries_the_content_the_write_path_tests_need() -> None:
    """Each assertion corresponds to a case the write-path tests would
    silently stop covering if the fixture's contents were trimmed."""
    gen = _generator()
    loaded = load_map_and_units(FIXTURE_PATH)
    manager = parse_triggers(loaded)
    assert manager is not None
    assert loaded.trigger_write_supported

    names = [trigger.name for trigger in manager.triggers]
    assert len(names) == gen.TRIGGER_COUNT

    reference_ids = [
        effect.trigger_id
        for trigger in manager.triggers
        for effect in trigger.effects
        if effect.effect_type in _REFERENCE_EFFECT_TYPES
    ]
    assert len(reference_ids) >= 2
    assert all(0 <= trigger_id < len(names) for trigger_id in reference_ids)

    # An effect whose quantity is bit-split across _quantity_int /
    # _quantity_float / variable. Plan finding 8's int-to-f32 retype lives in
    # that slot and has nowhere to show up without one.
    split_effects = [
        effect
        for trigger in manager.triggers
        for effect in trigger.effects
        if getattr(effect, "armour_attack_class", None) is not None
    ]
    assert split_effects, "no armour/attack effect: the quantity bit-split is untested"

    # A non-empty variable block, so the region after unknown_bytes isn't
    # trivially zero-length.
    assert len(manager.variables) == 1
    assert manager.variables[0].name == gen.VARIABLE_NAME

    # Non-empty str16 fields, the source of the NUL-padding drift the byte-blob
    # model exists to contain.
    assert any(trigger.description for trigger in manager.triggers)


def _parsed_slices(section_bytes: bytes, entries) -> list[bytes]:
    slices = []
    offset = SECTION_HEADER_SIZE
    for entry in entries:
        slices.append(section_bytes[offset : offset + entry.byte_length])
        offset += entry.byte_length
    return slices


def test_every_trigger_still_drifts_after_a_commit() -> None:
    """The fixture reproduces, in the default tier, the drift measured on the
    real corpus. This is the premise of the whole byte-blob design, and the one
    property that makes the fixture worth shipping.

    **The commit is the entire point of this test.** descape/trigger_model.py's
    serialize() commits the manager before reading any trigger's bytes back,
    and on a file that came out of AoE2ScenarioParser's own serializer that
    commit *repairs* the drift: the first version of this fixture drifted on 3
    of 4 triggers before a commit and on 0 of 4 after one. Measuring
    pre-commit drift therefore proves nothing about the write path -- it passes
    just as happily on a fixture that cannot test anything.

    If this ever starts asserting byte-identity instead, either the library's
    behaviour changed or the generator stopped writing empty strings the game's
    way (tools/gen_trigger_fixture.py's _game_style_bytes). Re-measure against
    the real corpus before anyone concludes the blobs are unnecessary.
    """
    loaded = load_map_and_units(FIXTURE_PATH)
    manager = parse_triggers(loaded)
    assert manager is not None
    section = loaded._scenario.sections["Triggers"]
    parsed = loaded.decompressed_body[loaded.units_section_end : loaded.triggers_section_end]
    slices = _parsed_slices(parsed, section.retriever_map["trigger_data"].data)

    manager.commit()

    assert section.get_data_as_bytes() != parsed
    entries = section.retriever_map["trigger_data"].data
    clean = [
        index
        for index, (slice_, entry) in enumerate(zip(slices, entries))
        if slice_ == entry.get_data_as_bytes()
    ]
    assert not clean, (
        f"triggers {clean} re-serialize to their own parsed bytes after a commit, so the "
        "fixture cannot tell the byte-blob write path apart from whole-section "
        "re-serialization -- see tools/gen_trigger_fixture.py's _game_style_bytes()"
    )
