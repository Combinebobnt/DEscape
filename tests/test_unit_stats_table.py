"""Verifies descape.unit_stats_table.unit_stats against ground truth read off
the real game data independently of tools/gen_unit_stats.py.

**Why this file exists separately**, following tests/test_unit_footprints.py's
own reasoning: a consistency oracle built from unit_stats.json itself would
just restate the data it is supposed to be checking, and would pass just as
happily on a wrong-field read (e.g. reading attack off `Unit.creatable`
instead of `Unit.type_50`) as on a correct one. Everything here is a
hand-written table, not derived from the generator or its output.
"""

from __future__ import annotations

from descape.unit_stats_table import line_of_sight, unit_stats

# unit_const -> (name, hp, attack, melee_armour, pierce_armour, range,
# max_range), read from empires2_x2_p1.dat directly. Hardcoded on purpose --
# see module docstring.
GROUND_TRUTH = {
    4: ("Archer", 30, 4, 0, 0, 4.0, 4.0),
    83: ("Villager", 25, 3, 0, 0, 0.0, 0.0),
    109: ("Town Center", 2400, 5, 3, 5, 6.0, 6.0),
    70: ("House", 550, 0, -2, 7, 0.0, 0.0),
    48: ("Wolf", 75, 7, 0, 0, 0.0, 0.0),
}


def test_ground_truth_units_match_real_stats():
    for unit_const, (name, hp, attack, melee, pierce, range_, max_range) in GROUND_TRUTH.items():
        stats = unit_stats(unit_const)
        assert stats == {
            "hp": hp,
            "attack": attack,
            "melee_armour": melee,
            "pierce_armour": pierce,
            "range": range_,
            "max_range": max_range,
        }, name


def test_poenari_castle_displays_no_range_but_reaches_eight():
    """GH #111: the .dat's displayed_range is 0.0 for const 445, while its
    max_range is 8.0 with Castle arrows (projectile 746), read off the .dat."""
    stats = unit_stats(445)
    assert (stats["range"], stats["max_range"]) == (0.0, 8.0)


def test_house_melee_armour_is_really_negative_two():
    """Not a bug to "fix" to 0 -- House's displayed_melee_armour is real data
    that agrees exactly with its own armours[class 4] entry in the .dat."""
    assert unit_stats(70)["melee_armour"] == -2


def test_a_tree_is_hp_only():
    """Const 349 (Oak tree) has hp=20 and neither a type_50 nor a creatable
    block -- absence is structural, not a zero-filled sentinel, so combat
    keys must be entirely absent rather than present at 0."""
    assert unit_stats(349) == {"hp": 20}


def test_an_unknown_const_returns_an_empty_dict():
    assert unit_stats(999999) == {}


def test_map_revealers_see_four_ten_and_twenty_tiles():
    """GH #153: Map Revealer / Medium / Giant (837 / 1774 / 1775), read off
    the .dat's line_of_sight, identical across every civ."""
    assert [line_of_sight(c) for c in (837, 1774, 1775)] == [4.0, 10.0, 20.0]


def test_invisible_object_a_and_an_unknown_const_have_no_line_of_sight():
    """Invisible Object A (1291) carries LOS 0, which the table omits."""
    assert line_of_sight(1291) is None
    assert line_of_sight(999999) is None


def test_line_of_sight_stays_out_of_the_stats_entries():
    """A LOS key in "stats" would give every revealer a stats grid on its card."""
    assert unit_stats(837) == {"hp": 1}
