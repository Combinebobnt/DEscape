"""GH #169: descape/command_buttons.py, what a building's command panel holds.

Qt-free. Reads the committed object_catalog.json, so the ids below are the
.dat's own.
"""

from __future__ import annotations

import pytest

from descape import command_buttons
from descape.command_buttons import ECONOMIC, MILITARY, TECH, UNIT

_BARRACKS, _DOCK, _CASTLE, _TOWN_CENTER = 12, 45, 82, 109
_MILITIA, _SWORDSMAN, _HOUSE, _MARKET = 74, 75, 70, 84


@pytest.mark.parametrize(
    ("value", "slot", "label"),
    [
        (0, None, "no button"),
        (1, (1, 1, 1), "row 1, col 1"),
        (7, (1, 2, 2), "row 2, col 2"),
        (15, (1, 3, 5), "row 3, col 5"),
        (16, None, "no button"),
        (21, (2, 1, 1), "page 2, row 1, col 1"),
        (24, (2, 1, 4), "page 2, row 1, col 4"),
        (35, (2, 3, 5), "page 2, row 3, col 5"),
        (36, None, "no button"),
        (-1, None, "no button"),
    ],
)
def test_slot_and_its_label(value: int, slot, label: str) -> None:
    assert command_buttons.slot(value) == slot
    assert command_buttons.slot_label(value) == label
    if slot is not None:
        assert command_buttons.button_value(*slot) == value


def test_button_values_are_both_pages() -> None:
    assert command_buttons.button_values() == (*range(1, 16), *range(21, 36))


def test_barracks_holds_units_and_techs_in_one_panel() -> None:
    units = command_buttons.occupants(_BARRACKS, UNIT)
    techs = command_buttons.occupants(_BARRACKS, TECH)
    assert set(units) == {1, 2, 3, 4}
    assert {_MILITIA, _SWORDSMAN} <= set(units[1])
    assert not set(units) & set(techs), "a unit and a tech never share a button"
    assert command_buttons.base_of(units[1], UNIT) == _MILITIA


def test_castle_button_1_holds_every_civs_unique_unit() -> None:
    assert len(command_buttons.occupants(_CASTLE, UNIT)[1]) > 50


def test_the_dock_has_a_second_page() -> None:
    units = command_buttons.occupants(_DOCK, UNIT)
    techs = command_buttons.occupants(_DOCK, TECH)
    assert 24 in units and 27 in techs
    assert command_buttons.slot(24)[0] == 2


def test_base_of_skips_heroes_and_hidden() -> None:
    """Dock page 2 holds only heroes, so the base falls back to the lowest of them."""
    rows = command_buttons._rows()[UNIT]
    ids = command_buttons.occupants(_BARRACKS, UNIT)[1]
    ordered = command_buttons.ordered(ids, UNIT)
    plain = [i for i in ordered if not rows[i].get("hidden") and not rows[i].get("hero_mode", 0) & 1]
    assert ordered[: len(plain)] == tuple(plain), "heroes and hidden ones last"
    heroes = command_buttons.occupants(_DOCK, UNIT)[24]
    assert command_buttons.base_of(heroes, UNIT) == min(heroes)
    assert command_buttons.base_of((), UNIT) is None


def test_a_tech_for_every_civ_comes_before_a_civ_only_one() -> None:
    techs = command_buttons.occupants(_BARRACKS, TECH)[6]
    base = command_buttons.base_of(techs, TECH)
    assert command_buttons._rows()[TECH][base]["civ"] == -1


def test_the_builder_menus_split_by_interface_kind() -> None:
    builder = command_buttons.BUILDER_UNIT
    economic = command_buttons.occupants(builder, UNIT, ECONOMIC)
    military = command_buttons.occupants(builder, UNIT, MILITARY)
    assert _HOUSE in economic[1] and _HOUSE not in military.get(1, ())
    assert _BARRACKS in military[1] and _BARRACKS not in economic.get(1, ())
    assert _MARKET in economic[8]
    assert command_buttons.villager_page(_HOUSE) == ECONOMIC
    assert command_buttons.villager_page(_BARRACKS) == MILITARY
    assert command_buttons.villager_page(_MILITIA) is None
    assert command_buttons.has_buttons(builder)


@pytest.mark.parametrize("villager", sorted(command_buttons.VILLAGER_IDS))
def test_a_villager_is_not_the_builder(villager: int) -> None:
    """In game a trigger naming Villager 83 or 293 moves nothing (2026-10-15)."""
    assert not command_buttons.is_builder(villager)
    assert not command_buttons.has_buttons(villager)
    assert command_buttons.occupants(villager, UNIT) == {}


def test_an_unknown_building_has_no_buttons() -> None:
    assert command_buttons.occupants(9999, UNIT) == {}
    assert command_buttons.occupants(_MILITIA, UNIT) == {}
    assert not command_buttons.has_buttons(9999)
    assert command_buttons.has_buttons(_TOWN_CENTER)


def test_the_lower_id_shows_in_a_shared_slot() -> None:
    """Measured in game: Knight 38 over Spearman 93, Spearman 93 over Scout
    Cavalry 448, and in the builder menus House 70 over Market 84, Market 84
    over Blacksmith 103, Monastery 104 over University 209, Mill 68 over House
    70, Farm 50 over Mining Camp 584."""
    spearman_line = command_buttons.occupants(_BARRACKS, UNIT)[2]
    assert command_buttons.shown_in_slot(38, spearman_line, UNIT)
    assert not command_buttons.shown_in_slot(448, spearman_line, UNIT)
    economic = command_buttons.occupants(command_buttons.BUILDER_UNIT, UNIT, ECONOMIC)
    for mover, button, shown in ((_MARKET, 1, False), (_MARKET, 7, True), (209, 9, False), (68, 1, True), (50, 3, True)):
        assert command_buttons.shown_in_slot(mover, economic[button], UNIT) is shown, (mover, button)
    assert command_buttons.shown_in_slot(448, (), UNIT), "an empty slot shows anything"
    assert command_buttons.shown_in_slot(435, (215,), TECH), "Bloodlines and Squires take turns in game"


def test_blocker_names_the_unit_that_hides_a_button() -> None:
    builder = command_buttons.BUILDER_UNIT
    assert command_buttons.blocker(_MARKET, builder, 1) == _HOUSE
    assert command_buttons.blocker(448, _BARRACKS, 2) == 93
    assert command_buttons.blocker(38, _BARRACKS, 2) is None, "the knight's lower id shows"
    assert command_buttons.blocker(448, _BARRACKS, 5) is None, "a free slot"
    assert command_buttons.blocker(_HOUSE, builder, 1) is None, "already there"
    assert command_buttons.blocker(1264, builder, 9) is None, "a building on neither menu"
    assert command_buttons.building_name(builder, ECONOMIC) == "Villager build menu (economic)"


def test_slot_values_follow_the_building() -> None:
    assert command_buttons.slot_values(command_buttons.BUILDER_UNIT) == tuple(range(14))
    assert command_buttons.slot_values(_DOCK) == (0, *range(1, 16), *range(21, 36))
    assert command_buttons.slot_values(-1) == command_buttons.slot_values(_DOCK), "unset offers everything"


def test_only_the_builder_menus_keep_14_and_15() -> None:
    builder = command_buttons.BUILDER_UNIT
    assert [command_buttons.reserved_label(builder, v) for v in (13, 14, 15)] == ["", "More", "Cancel"]
    assert command_buttons.reserved_label(_DOCK, 15) == ""
    military = command_buttons.occupants(builder, UNIT, MILITARY)
    economic = command_buttons.occupants(builder, UNIT, ECONOMIC)
    assert not {14, 15} & (set(military) | set(economic)), "the .dat puts nothing there"


def test_home_slots_say_where_it_is_trained_today() -> None:
    assert command_buttons.home_slots(_MILITIA, UNIT) == ((_BARRACKS, 1),)
    assert command_buttons.home_slots(22, TECH) == ((_TOWN_CENTER, 6),)
    assert (command_buttons.BUILDER_UNIT, 1) in command_buttons.home_slots(_HOUSE, UNIT)
    assert command_buttons.home_slots(9999, UNIT) == ()


# -- GH #230: the civ filter -----------------------------------------------------


def _spanish():
    from descape import civ_availability

    return civ_availability.keeper(14, civ_availability.NO_OVERRIDES)


def test_occupants_keeps_only_what_the_predicate_accepts() -> None:
    castle = command_buttons.occupants(82, UNIT, keep=_spanish())
    assert set(castle[1]) <= {771, 773} and 771 in castle[1]
    assert command_buttons.occupants(82, UNIT)[1] != castle[1]


def test_a_blocker_the_civ_lacks_does_not_hide_the_button() -> None:
    """Cavalry Archer 39 sits at Archery Range button 3; Spanish have none."""
    assert command_buttons.blocker(74, 87, 3) == 39
    assert command_buttons.blocker(74, 87, 3, _spanish()) is None


def test_a_unit_covers_the_techs_in_its_slot() -> None:
    covered = command_buttons.covered_techs(12, 6)
    assert 222 in covered, "Man-At-Arms at Barracks button 6"
    assert command_buttons.covered_techs(12, 8) == ()
    assert command_buttons.covered_techs(118, 1) == (), "the build menus hold no techs"


def test_a_tech_is_hidden_by_a_unit_in_its_slot() -> None:
    assert command_buttons.covering_unit(22, 12, 1) == 74, "Loom at Barracks button 1 sits under the Militia"
    assert command_buttons.covering_unit(22, 109, 6) is None, "Loom already sits there"
    assert command_buttons.covering_unit(22, 12, 8) is None
