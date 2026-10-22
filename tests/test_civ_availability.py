"""GH #230: descape/civ_availability.py, whether a player's civ can have an id.

Qt-free. Reads the committed object_catalog.json's `civs` and `gives` keys
and builds triggers on the real triggers fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from descape import civ_availability, disables_fields, game_dataset, player_fields
from descape.civ_availability import NO_OVERRIDES, Overrides
from descape.command_buttons import TECH, UNIT
from descape.scenario_io import load_map_and_units, parse_triggers

TRIGGERS_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"

_SPANISH, _ARMENIANS = 14, 44
_MILITIA, _KNIGHT, _MULE_CART, _CONQUISTADOR = 74, 38, 1808, 771
_ELITE_KAMAYUK_TECH, _ELITE_KAMAYUK, _LOOM = 509, 881, 22


def test_has_reads_the_catalogs_civs() -> None:
    assert not civ_availability.has(_SPANISH, _MULE_CART, UNIT)
    assert civ_availability.has(_ARMENIANS, _MULE_CART, UNIT)
    assert civ_availability.has(_SPANISH, _CONQUISTADOR, UNIT)
    assert civ_availability.has(_SPANISH, _MILITIA, UNIT), "no civs key: every civ"
    assert civ_availability.has(_SPANISH, _LOOM, TECH)
    assert civ_availability.civs_of(_MILITIA, UNIT) is None
    assert civ_availability.civs_of(_MULE_CART, UNIT) == (44, 45)


def test_a_civ_of_none_filters_nothing() -> None:
    assert civ_availability.has(None, _MULE_CART, UNIT)
    assert civ_availability.available(None, NO_OVERRIDES, _MULE_CART, UNIT)


def test_the_file_overrides_the_civ() -> None:
    enabled = Overrides(enabled=frozenset({(_MULE_CART, UNIT)}))
    assert civ_availability.available(_SPANISH, enabled, _MULE_CART, UNIT), "measured: an enabled Mule Cart shows"
    listed = Overrides(listed=frozenset({(_MILITIA, UNIT)}))
    assert not civ_availability.available(None, listed, _MILITIA, UNIT)
    both = Overrides(enabled=frozenset({(_MILITIA, UNIT)}), listed=frozenset({(_MILITIA, UNIT)}))
    assert civ_availability.available(_SPANISH, both, _MILITIA, UNIT), "a trigger enables what Disables listed"
    keep = civ_availability.keeper(_SPANISH, NO_OVERRIDES)
    assert keep(_CONQUISTADOR, UNIT) and not keep(_MULE_CART, UNIT)


@pytest.mark.parametrize(
    ("value", "civ", "name"),
    [(_SPANISH, _SPANISH, "Spanish"), ("HUN-CIV", 17, "Huns"), ("RANDOM-CIV", None, "Random")],
)
def test_player_civ_reads_the_civilization(monkeypatch, value, civ, name) -> None:
    loaded = load_map_and_units(TRIGGERS_FIXTURE)
    monkeypatch.setattr(player_fields, "current_value", lambda _loaded, _spec, _player: value)
    assert civ_availability.player_civ(loaded, 1) == civ
    assert civ_availability.civ_name(loaded, 1) == name


def test_a_return_of_rome_file_reads_its_own_civ_tokens(monkeypatch) -> None:
    """PERSIAN-CIV is civ 9 in Return of Rome's .dat, 8 in AoE2's."""
    loaded = load_map_and_units(TRIGGERS_FIXTURE)
    monkeypatch.setattr(player_fields, "current_value", lambda _loaded, _spec, _player: "PERSIAN-CIV")
    assert civ_availability.player_civ(loaded, 1) == 8
    loaded.game_variant = game_dataset.ROR_VARIANT
    assert civ_availability.player_civ(loaded, 1) == 9


def test_gaia_and_unset_players_have_no_civ() -> None:
    loaded = load_map_and_units(TRIGGERS_FIXTURE)
    assert civ_availability.player_civ(loaded, 0) is None
    assert civ_availability.player_civ(loaded, -1) is None
    assert civ_availability.civ_name(loaded, -1) == ""


def test_file_overrides_reads_enable_disable_effects_and_the_disables_list(monkeypatch) -> None:
    loaded = load_map_and_units(TRIGGERS_FIXTURE)
    manager = parse_triggers(loaded)
    trigger = manager.add_trigger("gh230 overrides")
    trigger.new_effect.enable_disable_object(object_list_unit_id=_MULE_CART, source_player=1, enabled=1)
    trigger.new_effect.enable_disable_object(object_list_unit_id=_KNIGHT, source_player=1, enabled=0)
    trigger.new_effect.enable_disable_object(object_list_unit_id=_CONQUISTADOR, source_player=2, enabled=1)
    trigger.new_effect.enable_disable_technology(technology=_ELITE_KAMAYUK_TECH, source_player=1, enabled=1)
    everyone = trigger.new_effect.enable_disable_object(object_list_unit_id=1, source_player=1, enabled=1)
    everyone.source_player = -1
    monkeypatch.setattr(
        disables_fields,
        "current_ids",
        lambda _loaded, category, player: (_MILITIA,) if (category, player) == ("units", 1) else (),
    )

    overrides = civ_availability.file_overrides(manager, loaded, 1)
    assert (_MULE_CART, UNIT) in overrides.enabled
    assert (_ELITE_KAMAYUK_TECH, TECH) in overrides.enabled
    assert (_ELITE_KAMAYUK, UNIT) in overrides.enabled, "the tech gives the unit"
    assert (1, UNIT) in overrides.enabled, "source_player -1 is every player"
    assert (_CONQUISTADOR, UNIT) not in overrides.enabled, "player 2's"
    assert overrides.disabled == frozenset({(_KNIGHT, UNIT)})
    assert overrides.listed == frozenset({(_MILITIA, UNIT)})
    assert overrides.turned_off((_KNIGHT, UNIT)) and overrides.turned_off((_MILITIA, UNIT))


def test_file_overrides_without_triggers_or_a_file() -> None:
    assert civ_availability.file_overrides(None, None, 1) == NO_OVERRIDES


@pytest.mark.parametrize(
    ("start", "researched"),
    [(2, set()), (3, {22}), (4, {22, 435}), (5, {22, 435, 207}), (6, {22, 435, 207, 47})],
)
def test_a_later_start_already_has_the_earlier_ages_generic_techs(start, researched) -> None:
    """Measured in game 2026-10-16 (Spanish): Loom at Feudal, Bloodlines at Castle..., Chemistry only at Post-Imperial."""
    probes = {22, 435, 207, 47}
    assert {t for t in probes if civ_availability.researched_at_start(start, t)} == researched
    assert not civ_availability.researched_at_start(None, 22)


def test_unique_techs_are_left_out_of_the_start_age_rule() -> None:
    assert not civ_availability.researched_at_start(6, 494), "Pavise (Italians) was not measured"


def test_missing_names_the_reason() -> None:
    listed = Overrides(listed=frozenset({(_MILITIA, UNIT)}))
    assert civ_availability.missing(_SPANISH, NO_OVERRIDES, _CONQUISTADOR, UNIT) == ""
    assert civ_availability.missing(_SPANISH, NO_OVERRIDES, _MULE_CART, UNIT) == civ_availability.MISSING_CIV
    assert civ_availability.missing(None, listed, _MILITIA, UNIT) == civ_availability.MISSING_LISTED
    assert civ_availability.missing(None, NO_OVERRIDES, _LOOM, TECH, 5) == civ_availability.MISSING_START
    assert not civ_availability.keeper(None, NO_OVERRIDES, 5)(_LOOM, TECH)


@pytest.mark.parametrize(("value", "age"), [(5, 5), (6, 6), (0, None), (4294967295, None)])
def test_player_start_age_reads_starting_age(monkeypatch, value, age) -> None:
    loaded = load_map_and_units(TRIGGERS_FIXTURE)
    monkeypatch.setattr(player_fields, "current_value", lambda _loaded, _spec, _player: value)
    assert civ_availability.player_start_age(loaded, 1) == age
