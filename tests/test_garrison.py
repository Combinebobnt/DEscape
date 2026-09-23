"""Verifies descape.garrison against hand-written ground truth, plus the
corpus measurement the eligibility rule was derived from.

**Why the ground truth is hand-written**, following test_unit_stats_table.py's
own reasoning: an oracle built out of garrison_table.json would restate the
table it is meant to be checking and would pass just as happily on a
class-to-bit mapping that admits a mangonel into a tower. Every expectation
below is what the game itself does, written out by hand.
"""

from __future__ import annotations

import pytest

from descape import garrison
from descape.scenario_io import load_map_and_units

# host const -> base capacity, read from empires2_x2_p1.dat directly.
CAPACITY_GROUND_TRUTH = {
    79: ("Watch Tower", 5),
    109: ("Town Center", 15),
    82: ("Castle", 20),
    545: ("Transport Ship", 20),
    45: ("Dock", 5),
    70: ("House", 0),
}

# (host, occupant, accepted) -- each row is a thing the game itself does or
# refuses, not a restatement of the table.
ACCEPTS_GROUND_TRUTH = (
    (79, 4, True),  # Archer into a Watch Tower: a foot archer garrisons a tower
    (79, 83, True),  # Villager into a Watch Tower
    (79, 125, True),  # Monk into a Watch Tower
    (79, 38, False),  # Knight: a tower takes no mounted unit
    (79, 280, False),  # Mangonel: no building garrisons siege
    (109, 83, True),  # Villager into a Town Center
    (109, 38, False),  # Knight into a Town Center
    (82, 38, True),  # Knight into a Castle -- the bit a castle adds over a tower
    (82, 4, True),  # Archer into a Castle
    (82, 280, False),  # Mangonel into a Castle
    (45, 539, True),  # Galley into a Dock
    (45, 4, False),  # Archer into a Dock
    (545, 4, True),  # Archer onto a Transport Ship
    (545, 280, True),  # Mangonel onto a Transport Ship: transports carry siege
    (545, 539, False),  # Galley onto a Transport Ship
    (84, 83, False),  # Market: garrison_capacity 10, but its mask holds nothing
    (70, 83, False),  # House: no capacity at all
)


def test_capacity_matches_the_game_table() -> None:
    for host, (name, cap) in CAPACITY_GROUND_TRUTH.items():
        assert garrison.capacity(host) == cap, name


def test_capacity_of_an_unknown_const_is_zero() -> None:
    assert garrison.capacity(999999) == 0


@pytest.mark.parametrize(("host", "occupant", "accepted"), ACCEPTS_GROUND_TRUTH)
def test_accepts_matches_the_game(host: int, occupant: int, accepted: bool) -> None:
    assert garrison.accepts(host, occupant) is accepted


def test_a_host_with_no_building_section_takes_every_land_class() -> None:
    """A Transport Ship has garrison_capacity but no `building` section, so
    it has no mask of its own to read -- see the module docstring."""
    assert garrison.garrison_mask(545) == garrison._LAND_MASK
    assert garrison.garrison_mask(79) == 11


def test_eligible_consts_agrees_with_accepts() -> None:
    for host in (79, 82, 45, 545):
        eligible = garrison.eligible_consts(host)
        assert eligible
        for occupant in (4, 38, 83, 125, 280, 539):
            assert (occupant in eligible) is garrison.accepts(host, occupant)


def test_eligible_consts_is_empty_for_a_host_that_holds_nothing() -> None:
    assert garrison.eligible_consts(70) == frozenset()
    assert garrison.eligible_consts(84) == frozenset()


def test_can_hold_separates_a_real_host_from_a_capacity_only_row() -> None:
    assert garrison.can_hold(79)
    assert garrison.can_hold(545)
    assert not garrison.can_hold(84)  # capacity 10, mask 0
    assert not garrison.can_hold(70)


@pytest.mark.corpus
def test_every_corpus_occupant_is_accepted_by_its_host(scenario_path) -> None:
    """The measurement the class-to-bit mapping was derived from (2026-09-21:
    55 occupants over four files, all at their host's exact point, zero
    dangling links). A counterexample here means the mapping is wrong, not
    that the file is."""
    loaded = load_map_and_units(scenario_path)
    by_ref = {unit.reference_id: unit for units in loaded.unit_manager.units for unit in units}
    for units in loaded.unit_manager.units:
        for unit in units:
            host_id = getattr(unit, "garrisoned_in_id", -1)
            if host_id in (-1, unit.reference_id):
                continue
            host = by_ref.get(host_id)
            if host is None:  # a dangling link is a file fact, not a rule fact
                continue
            assert garrison.accepts(host.unit_const, unit.unit_const), (
                f"{scenario_path.name}: const {unit.unit_const} is garrisoned in const "
                f"{host.unit_const}, which this module refuses"
            )
            assert garrison.capacity(host.unit_const) > 0, host.unit_const
