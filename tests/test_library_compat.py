"""Covers descape/library_compat.py and scenario_io.py's trigger read path
(phase 4a). Nothing here writes a file: 4a is read-only by construction, and
trigger_tail still splices back verbatim on save. Unit joined the poisoned
classes for the Units write path (phase 3.5a); its own load-time depoison
call is covered here for the same process-global reason the trigger classes
are.

The load-bearing test is the cross-version de-poison regression. AoE2ScenarioParser
disables version-unsupported fields by overwriting them on the *class*, permanently
and with the then-current version baked into the error message, so loading a 1.41
file and then a 1.55 file makes the 1.55 file's own fields raise "supported since:
1.55. Current version: 1.41". The library has a standing Todo admitting this
(retriever_object_link.py:81). It is a process-global hazard of exactly the kind
tests/README.md's "verify modules share one interpreter" note describes, which is
why the fix is asserted against real corpus files and not just synthetically.

Most tests here need no scenario at all: the vocabulary is read from the library's
own per-version JSON, which is what lets the trigger UI populate a picker before
any file is open.
"""

from __future__ import annotations

import struct

import pytest

from descape import library_compat
from descape.scenario_io import load_map_and_units, parse_triggers


def test_pristine_snapshot_covers_every_poisoned_class() -> None:
    assert set(library_compat.PRISTINE_CLASS_STATE) == set(library_compat.POISONED_CLASSES)
    for cls in library_compat.POISONED_CLASSES:
        assert library_compat.PRISTINE_CLASS_STATE[cls], f"{cls.__name__} snapshot is empty"


def test_depoison_leaves_every_class_clean() -> None:
    """The invariant stated as depoison()'s postcondition rather than as
    "clean at import": test order is not guaranteed, and any earlier test that
    parsed triggers would leave the classes poisoned."""
    library_compat.depoison()
    for cls in library_compat.POISONED_CLASSES:
        added, overwritten = library_compat.class_state_delta(cls)
        assert added == [], f"{cls.__name__} still carries added names: {added}"
        assert overwritten == [], f"{cls.__name__} still carries overwritten: {overwritten}"


def test_map_units_load_does_not_poison_trigger_classes() -> None:
    """Poisoning happens when trigger objects are constructed, not on any load.
    Recorded because it is what makes the lazy parse safe: opening a file for
    terrain editing leaves the trigger classes pristine.

    Unit is deliberately excluded from this assertion: unlike
    Trigger/Condition/Effect/Variable, UnitManager.construct() runs inside
    every load_map_and_units() call, so this same file's own Units parse is
    expected to poison Unit by this file's version -- that is normal, not the
    bug. What matters is that Unit starts *clean* going into that parse; see
    test_map_units_load_depoisons_unit_before_parsing below.
    """
    from AoE2ScenarioParser.objects.data_objects.condition import Condition
    from AoE2ScenarioParser.objects.data_objects.effect import Effect
    from AoE2ScenarioParser.objects.data_objects.trigger import Trigger
    from AoE2ScenarioParser.objects.data_objects.variable import Variable

    library_compat.depoison()
    load_map_and_units("tests/fixtures/golden_blank_120x120.aoe2scenario")
    for cls in (Trigger, Condition, Effect, Variable):
        assert library_compat.class_state_delta(cls) == ([], [])


def test_map_units_load_depoisons_unit_before_parsing() -> None:
    """Unit is parsed eagerly, so depoisoning it *after* a load is a no-op on
    objects already parsed while poisoned. Synthetically poisons Unit first, the same
    way a prior file's load would have, then asserts the next load starts
    from a clean class regardless -- proving the fix is the depoison() call
    sitting before scenario_io._load_map_and_units()'s own section walk, not
    merely somewhere in the module.
    """
    from AoE2ScenarioParser.objects.data_objects.unit import Unit

    library_compat.depoison()
    Unit.a_name_the_library_would_have_added = property(lambda self_: None)
    added, _overwritten = library_compat.class_state_delta(Unit)
    assert "a_name_the_library_would_have_added" in added, "the synthetic poisoning did not take"

    load_map_and_units("tests/fixtures/golden_blank_120x120.aoe2scenario")
    assert not hasattr(Unit, "a_name_the_library_would_have_added"), (
        "load_map_and_units() must depoison Unit before its own Units parse"
    )


def test_depoison_reverts_a_synthetic_poisoning() -> None:
    """Exercises all three of depoison()'s steps without needing a scenario:
    an added name, an overwritten value, and a flipped `disabled`."""
    from AoE2ScenarioParser.objects.data_objects.effect import Effect

    library_compat.depoison()
    # _quantity_float is the real case finding 6 records: a property Effect
    # defines itself, which the library overwrites. Deleting added names alone
    # would leave it poisoned, which is why depoison() restores values too.
    original_quantity_float = Effect._quantity_float
    links = list(library_compat._iter_links(Effect._link_list))
    assert links, "Effect._link_list flattened to nothing"

    Effect.a_name_the_library_would_have_added = property(lambda self_: None)
    Effect._quantity_float = property(lambda self_: "poisoned")
    links[0].disabled = True

    added, overwritten = library_compat.class_state_delta(Effect)
    assert "a_name_the_library_would_have_added" in added
    assert "_quantity_float" in overwritten

    library_compat.depoison()

    assert library_compat.class_state_delta(Effect) == ([], [])
    assert Effect._quantity_float is original_quantity_float
    assert not hasattr(Effect, "a_name_the_library_would_have_added")
    assert links[0].disabled is False


def test_iter_links_flattens_nested_groups() -> None:
    """_link_list entries can be a RetrieverObjectLinkGroup whose real links
    live on `.group`; only the leaves carry support/disabled."""
    from AoE2ScenarioParser.objects.data_objects.condition import Condition

    flat = list(library_compat._iter_links(Condition._link_list))
    assert len(flat) > len(Condition._link_list)
    for link in flat:
        assert hasattr(link, "disabled")


def test_trigger_version_reads_the_leading_f64() -> None:
    assert library_compat.trigger_version(struct.pack("<d", 4.9) + b"junk") == 4.9


def test_trigger_version_rejects_a_short_tail() -> None:
    with pytest.raises(ValueError):
        library_compat.trigger_version(b"\x00\x00\x00")


def test_vocabulary_is_available_discriminates() -> None:
    assert library_compat.vocabulary_is_available("1.55")
    assert not library_compat.vocabulary_is_available("1.21")


def test_load_vocabulary_has_the_documented_shape() -> None:
    vocab = library_compat.load_vocabulary("1.55")
    # docs/INGAME_EDITOR_REFERENCE.md's vocabulary: 40 conditions, 100 effects.
    # Asserted as lower bounds so a newer structure version adding types passes.
    assert len(vocab.conditions) >= 40
    assert len(vocab.effects) >= 100
    assert -1 not in vocab.conditions, "the -1 presentation pseudo-entry leaked in as a type"
    assert -1 not in vocab.effects

    bring_object = vocab.conditions[1]
    assert bring_object.name == "bring_object_to_area"
    assert "area_x1" in bring_object.attributes
    # attributes is what a type displays; default_attributes is every field.
    assert set(bring_object.attributes) <= set(bring_object.default_attributes)
    assert vocab.condition_presentation["source_player"] == "PlayerId"


def test_load_vocabulary_is_cached() -> None:
    assert library_compat.load_vocabulary("1.55") is library_compat.load_vocabulary("1.55")


def test_load_vocabulary_rejects_an_unknown_version() -> None:
    with pytest.raises(FileNotFoundError):
        library_compat.load_vocabulary("1.21")


@pytest.mark.corpus
def test_trigger_read_and_alignment_gate(scenario_path) -> None:
    """The alignment gate: a full walk of Triggers and every section after it
    must consume exactly len(trigger_tail).

    v1.36/1.37 are expected to fail it -- the library models no Files section
    there, leaving a large unconsumed remainder -- and that exclusion falls out
    of the byte accounting rather than being special-cased.
    """
    loaded = load_map_and_units(scenario_path)
    assert loaded.decompressed_body[loaded.units_section_end :] == loaded.trigger_tail
    assert loaded.options_section_end > 0
    assert loaded.trigger_read_supported is None, "parse must be lazy, not eager"

    manager = parse_triggers(loaded)
    if manager is None:
        # The 1.54/trigger-3.9 set. Must degrade, never fail the open.
        assert loaded.trigger_read_supported is False
        assert loaded.trigger_write_supported is False
        assert loaded.scenario_version == "1.54"
        return

    assert loaded.trigger_read_supported is True
    assert parse_triggers(loaded) is manager, "parse must be memoized"
    assert loaded.triggers_section_end > loaded.units_section_end
    assert loaded.trigger_version > 0

    expected_gate = float(loaded.scenario_version) >= 1.40
    assert loaded.trigger_write_supported is expected_gate, (
        f"alignment gate {loaded.trigger_write_supported} for scenario version "
        f"{loaded.scenario_version}; expected {expected_gate}"
    )


@pytest.mark.corpus
def test_cross_version_depoison_regression(corpus_files) -> None:
    """Two different-version files parsed in one process, then every
    version-gated field read on the second. This is the case that fails
    without depoison(); monkeypatching it away below proves the test can see
    the failure rather than passing vacuously.
    """
    by_version: dict[str, object] = {}
    for path in corpus_files:
        loaded = load_map_and_units(path)
        if parse_triggers(loaded) is not None:
            by_version.setdefault(loaded.scenario_version, loaded)
    if len(by_version) < 2:
        pytest.skip(f"need two parseable scenario versions, got {sorted(by_version)}")

    versions = sorted(by_version)
    older, newer = by_version[versions[0]], by_version[versions[-1]]

    for loaded in (older, newer):
        manager = parse_triggers(loaded)
        for trigger in manager.triggers:
            for condition in trigger.conditions:
                condition.condition_type, condition.quantity
            for effect in trigger.effects:
                effect.effect_type, effect.quantity


@pytest.mark.corpus
def test_re_parsing_makes_an_earlier_manager_readable_again(corpus_files) -> None:
    """Parsing scenario B poisons the classes that a manager for scenario A
    reads through, because the gating is class-level and process-global.

    Re-calling parse_triggers(A) is the documented repair, and it has to work
    on a *memo hit* -- the bug this pins is an early return that handed back
    the cached manager without restoring class state, so re-calling fixed
    nothing.
    """
    from AoE2ScenarioParser.exceptions.asp_exceptions import UnsupportedAttributeError

    newer_path = older_path = None
    for path in corpus_files:
        loaded = load_map_and_units(path)
        if parse_triggers(loaded) is None:
            continue
        version = float(loaded.scenario_version)
        if version >= 1.55 and newer_path is None:
            newer_path = path
        if version <= 1.41 and older_path is None:
            older_path = path
    if newer_path is None or older_path is None:
        pytest.skip("need one <=1.41 and one >=1.55 parseable file in the corpus")

    # Fresh loads: the poisoning only happens on a real parse, and a memoized
    # parse_triggers() now depoisons instead, which is the repair under test.
    newer = load_map_and_units(newer_path)
    older = load_map_and_units(older_path)

    newer_manager = parse_triggers(newer)
    conditions = [c for t in newer_manager.triggers for c in t.conditions]
    assert conditions, "the >=1.55 file has no conditions to read"

    parse_triggers(older)  # first parse of this object: poisons the 1.55-only fields
    with pytest.raises(UnsupportedAttributeError):
        for condition in conditions:
            condition.local_technology

    assert parse_triggers(newer) is newer_manager, "repair must not re-parse"
    for condition in conditions:
        condition.local_technology


@pytest.mark.corpus
def test_reading_a_newer_only_field_needs_depoison(corpus_files, monkeypatch) -> None:
    """The negative half of the regression above. With depoison() stubbed out,
    reading a 1.55-only Condition field after an older load raises with the
    *older* version quoted back."""
    from AoE2ScenarioParser.exceptions.asp_exceptions import UnsupportedAttributeError

    older = newer = None
    for path in corpus_files:
        loaded = load_map_and_units(path)
        if parse_triggers(loaded) is None:
            continue
        version = float(loaded.scenario_version)
        if version <= 1.41 and older is None:
            older = loaded
        if version >= 1.55 and newer is None:
            newer = loaded
    if older is None or newer is None:
        pytest.skip("need one <=1.41 and one >=1.55 parseable file in the corpus")

    try:
        monkeypatch.setattr(library_compat, "depoison", lambda: None)
        poisoned_older = load_map_and_units(older.path)
        poisoned_newer = load_map_and_units(newer.path)
        parse_triggers(poisoned_older)
        manager = parse_triggers(poisoned_newer)
        assert manager is not None
        conditions = [c for t in manager.triggers for c in t.conditions]
        assert conditions, "no conditions to read; the raise below would be vacuous"
        with pytest.raises(UnsupportedAttributeError):
            for condition in conditions:
                condition.local_technology
    finally:
        monkeypatch.undo()
        library_compat.depoison()
