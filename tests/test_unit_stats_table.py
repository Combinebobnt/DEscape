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

from descape.unit_stats_table import unit_stats

# unit_const -> (name, hp, attack, melee_armour, pierce_armour, range), read
# from empires2_x2_p1.dat directly. Hardcoded on purpose -- see module
# docstring.
GROUND_TRUTH = {
    4: ("Archer", 30, 4, 0, 0, 4.0),
    83: ("Villager", 25, 3, 0, 0, 0.0),
    109: ("Town Center", 2400, 5, 3, 5, 6.0),
    70: ("House", 550, 0, -2, 7, 0.0),
    48: ("Wolf", 75, 7, 0, 0, 0.0),
}


def test_ground_truth_units_match_real_stats():
    for unit_const, (name, hp, attack, melee, pierce, range_) in GROUND_TRUTH.items():
        stats = unit_stats(unit_const)
        assert stats == {
            "hp": hp,
            "attack": attack,
            "melee_armour": melee,
            "pierce_armour": pierce,
            "range": range_,
        }, name


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
