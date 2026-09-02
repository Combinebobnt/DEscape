"""Covers descape/option_fields.py: the Qt-free field-spec module for the
Map Options panel.

The stub-based tests below need no scenario file and no Qt -- they pin
specs_for()'s presence-gate logic (missing section / missing retriever /
present-but-zero-length, each a real corpus shape) directly. The fixture-based
tests confirm that logic against a real load.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from descape import option_fields, scenario_io
from descape.option_fields import current_value, specs_for


class _StubRetriever:
    def __init__(self, raw_bytes: bytes, data=0):
        self._raw = raw_bytes
        self.data = data
        self.datatype = SimpleNamespace(type="u32")

    def get_data_as_bytes(self) -> bytes:
        return self._raw


def _stub_loaded(overrides: dict[tuple[str, str], "_StubRetriever"] = None):
    """A LoadedScenario stand-in exposing only what specs_for()/current_value()
    touch: `_scenario.sections[name].retriever_map[name]`. Every spec in
    option_fields._SPECS gets a present, non-empty (4-byte) stub by default;
    `overrides` replaces or removes individual (section, retriever) entries
    -- pass None as the value to simulate a missing retriever."""
    overrides = overrides or {}
    sections: dict[str, dict[str, _StubRetriever]] = {}
    for spec in option_fields._SPECS:
        sections.setdefault(spec.section, {})
        key = (spec.section, spec.retriever)
        if key in overrides:
            value = overrides[key]
            if value is not None:
                sections[spec.section][spec.retriever] = value
        else:
            sections[spec.section][spec.retriever] = _StubRetriever(b"\x00\x00\x00\x00")

    # Drop a section entirely if every one of its retrievers was overridden away.
    section_objs = {
        name: SimpleNamespace(retriever_map=retrievers)
        for name, retrievers in sections.items()
        if retrievers
    }
    return SimpleNamespace(_scenario=SimpleNamespace(sections=section_objs))


# -- specs_for(): presence gate -----------------------------------------------


def test_every_spec_present_by_default() -> None:
    loaded = _stub_loaded()
    assert {s.field_id for s in specs_for(loaded)} == {s.field_id for s in option_fields._SPECS}


def test_a_missing_section_drops_every_spec_in_it() -> None:
    """Mirrors a file where Triggers hasn't been parsed yet -- the section
    key itself is absent from loaded._scenario.sections."""
    overrides = {
        (s.section, s.retriever): None for s in option_fields._SPECS if s.section == "Triggers"
    }
    loaded = _stub_loaded(overrides)
    ids = {s.field_id for s in specs_for(loaded)}
    assert "legacy_exec_order" not in ids
    assert "collide_and_correct" in ids  # an unrelated section is untouched


def test_a_missing_retriever_drops_only_that_spec() -> None:
    """A section present but missing one named retriever -- the shape a
    structure-version change would produce for a single field."""
    loaded = _stub_loaded({("Map", "secondary_game_modes"): None})
    ids = {s.field_id for s in specs_for(loaded)}
    assert "secondary_game_modes" not in ids
    assert "collide_and_correct" in ids
    assert "no_waves_on_shore" in ids


def test_a_zero_length_retriever_is_treated_as_absent() -> None:
    """The real corpus shape for legacy_exec_order on a file whose
    trigger_version is below 4.5: the retriever exists in the map but its
    SET_REPEAT eval produced zero bytes, not a missing key."""
    loaded = _stub_loaded({("Options", "ai_map_type"): _StubRetriever(b"")})
    ids = {s.field_id for s in specs_for(loaded)}
    assert "ai_map_type" not in ids
    assert "all_techs" in ids


def test_current_value_reads_the_stub_data() -> None:
    loaded = _stub_loaded({("Map", "collide_and_correct"): _StubRetriever(b"\x01", data=1)})
    spec = next(s for s in specs_for(loaded) if s.field_id == "collide_and_correct")
    assert current_value(loaded, spec) == 1


def test_current_value_raises_rather_than_return_a_list() -> None:
    """The guard a future repeat-driven spec (e.g. a Players-mode
    per_player_starting_age) needs: current_value() must fail loudly, not
    hand a list to a widget expecting a scalar."""
    loaded = _stub_loaded({("Map", "collide_and_correct"): _StubRetriever(b"\x01\x01", data=[1, 1])})
    spec = next(s for s in specs_for(loaded) if s.field_id == "collide_and_correct")
    with pytest.raises(TypeError):
        current_value(loaded, spec)


# -- against a real load -------------------------------------------------------


def test_specs_for_a_real_file_before_and_after_trigger_parse() -> None:
    loaded = scenario_io.load_map_and_units(scenario_io.BLANK_TEMPLATE_PATH)

    before = {s.field_id for s in specs_for(loaded)}
    assert "legacy_exec_order" not in before  # Triggers not parsed yet
    assert "collide_and_correct" in before
    assert "lock_teams" in before
    assert len(before) == len(option_fields._SPECS) - 1

    scenario_io.parse_triggers(loaded)
    after = {s.field_id for s in specs_for(loaded)}
    assert after == {s.field_id for s in option_fields._SPECS}


def test_every_spec_resolves_to_a_scalar_int() -> None:
    """Pins the invariant current_value() relies on: nothing in _SPECS today
    points at a repeat-driven retriever. If a future spec (e.g. a
    Players-mode field) breaks this, it should fail here -- a plain,
    fast, default-tier sweep -- rather than surface as a mysterious list
    reaching a Qt widget in the panel."""
    loaded = scenario_io.load_map_and_units(scenario_io.BLANK_TEMPLATE_PATH)
    scenario_io.parse_triggers(loaded)
    for spec in specs_for(loaded):
        assert isinstance(current_value(loaded, spec), int), spec.field_id


def test_current_value_matches_known_blank_template_defaults() -> None:
    """Pins a few values against what the shipped donor actually contains,
    not just "some int came back" -- a wrong section/retriever pairing in
    _SPECS would otherwise read a plausible-looking but wrong field."""
    loaded = scenario_io.load_map_and_units(scenario_io.BLANK_TEMPLATE_PATH)
    specs = {s.field_id: s for s in specs_for(loaded)}

    assert current_value(loaded, specs["max_number_of_teams"]) == 4
    assert current_value(loaded, specs["allow_players_choose_teams"]) == 1
    assert current_value(loaded, specs["lock_teams"]) == 0
    assert current_value(loaded, specs["collide_and_correct"]) == 0
    assert current_value(loaded, specs["all_techs"]) == 0


# -- widget-range sanity ------------------------------------------------------


def test_every_spinbox_spec_declares_bounds_a_qspinbox_can_hold() -> None:
    """The Map Options panel range-checks a stored value against these bounds
    before choosing a widget, and a QSpinBox is int32-bounded. A spec with a
    missing or out-of-int32 bound would either raise TypeError in that
    comparison or OverflowError inside Qt -- neither of which any default-tier
    test would otherwise reach, since the check only runs with a Qt widget in
    hand. Qt-free here on purpose.
    """
    int32 = range(-2_147_483_648, 2_147_483_648)
    for spec in option_fields._SPECS:
        if spec.kind != option_fields.SPINBOX:
            continue
        assert spec.minimum is not None, f"{spec.field_id} has no minimum"
        assert spec.maximum is not None, f"{spec.field_id} has no maximum"
        assert spec.minimum <= spec.maximum, f"{spec.field_id}'s bounds are inverted"
        assert spec.minimum in int32, f"{spec.field_id}'s minimum is outside int32"
        assert spec.maximum in int32, f"{spec.field_id}'s maximum is outside int32"


def test_every_spec_declares_sentinel_and_sentinel_display_together() -> None:
    """A `sentinel` with no `sentinel_display` reaches _is_representable()
    with `value = None` and raises TypeError -- both or neither. When both
    are set, sentinel_display must itself land inside [minimum, maximum], or
    substitution would just swap one unrepresentable value for another.
    """
    for spec in option_fields._SPECS:
        has_sentinel = spec.sentinel is not None
        has_display = spec.sentinel_display is not None
        assert has_sentinel == has_display, (
            f"{spec.field_id} sets sentinel/sentinel_display independently"
        )
        if has_sentinel:
            assert spec.minimum <= spec.sentinel_display <= spec.maximum, (
                f"{spec.field_id}'s sentinel_display is outside its own bounds"
            )
