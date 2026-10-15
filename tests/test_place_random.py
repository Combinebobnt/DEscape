"""descape/place_random.py (GH #164): the "Villager (Random)" pseudo pick.
Qt-free and install-free."""

from __future__ import annotations

import random

import pytest

from descape import object_catalog, place_random, unit_rotation
from descape.place_random import RANDOM_VILLAGER

MALE, FEMALE = 83, 293


def test_the_sentinel_is_negative_and_no_real_object_id() -> None:
    assert RANDOM_VILLAGER < 0
    assert RANDOM_VILLAGER not in {entry.id for entry in object_catalog.objects()}
    assert all(value < 0 for value in place_random.RANDOM_PICKS)


def test_the_two_villagers_share_a_frame_grid_so_a_facing_carries_over() -> None:
    for const in (MALE, FEMALE):
        assert unit_rotation.rotation_is_angle(const)
        assert unit_rotation.angle_count_for(const) == 16


def test_choices_is_the_const_itself_for_a_real_pick_and_the_mix_for_a_random_one() -> None:
    assert place_random.choices(4) == (4,)
    assert place_random.choices(RANDOM_VILLAGER) == (MALE, FEMALE)
    assert place_random.is_random(RANDOM_VILLAGER)
    assert not place_random.is_random(MALE)
    assert not place_random.is_random(None)


def test_a_seeded_roll_gives_both_villagers_and_only_them() -> None:
    rng = random.Random(164)
    rolls = [place_random.roll(RANDOM_VILLAGER, rng) for _ in range(40)]
    assert set(rolls) == {MALE, FEMALE}


def test_a_real_const_rolls_to_itself_without_drawing() -> None:
    class NoDraw:
        def choice(self, _options):
            raise AssertionError("a real const must not draw")

    assert place_random.roll(4, NoDraw()) == 4


def test_label_and_preview_const() -> None:
    assert place_random.label(RANDOM_VILLAGER) == "Villager (Random)"
    assert place_random.preview_const(RANDOM_VILLAGER) == MALE
    with pytest.raises(KeyError):
        place_random.label(MALE)


def test_catalog_entries_are_units_rows_under_the_villagers_class() -> None:
    """GH #68's browser lists them; found by villager, random, male and female."""
    (entry,) = place_random.catalog_entries()
    villager = object_catalog.entry(object_catalog.objects(), MALE)
    assert (entry.name, entry.id, entry.category, entry.hidden) == ("Villager (Random)", RANDOM_VILLAGER, "Units", False)
    assert entry.class_id == villager.class_id is not None
    for needle in ("villager", "random", "male", "female"):
        assert needle in entry.search_text
