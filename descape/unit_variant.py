"""Whether a unit's graphic variant may be cycled, and the arithmetic of doing it.

A tree's `rotation` is not an angle but a graphic-variant index: oak (349) has
42 variants, pine (350) 27. This is the edit-path counterpart to
unit_rotation.py (angles) and gate_orientation.py (gate orientation consts),
with the same shape: a small Qt-free module, so the default test tier covers it.

Measured over the 20-file examples/ corpus (2026-09-11 plan,
tools/scan_variant_cycling.py re-derives it): every one of 132,118 placements on
a cyclable const stores an exact literal integer, never the `k*2pi/n` radian
form walls use. So writes are always a plain literal integer.

Walls, cliffs and gates are excluded BY CONST SET, not by `unit.class_`: the
game re-derives their stored index from neighbours (and a gate's orientation
lives in its const). Class 27 is 36 consts of which only 8 are walls, so a
class exclusion would also drop Aqueduct (231) and Mole (2421).
"""

from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path

from descape import gate_orientation, unit_rotation, unit_sprites

_GRAPHIC_MAP_PATH = Path(__file__).resolve().parent / "unit_graphic_map.json"


@lru_cache(maxsize=1)
def excluded_consts() -> frozenset[int]:
    """Walls, cliffs and every gate: variant-indexed, but not the user's to cycle.

    A function rather than a module constant because the gate half reads two
    JSON tables, the same reason unit_sprites.wall_connector_consts() is one.
    """
    gates = frozenset(const for group in gate_orientation.groups().values() for const in group)
    return unit_sprites._ROTATION_VARIANT_CONSTS | unit_sprites._CLIFF_VARIANT_CONSTS | gates


@lru_cache(maxsize=1)
def _variant_counts() -> dict[int, int]:
    """unit_const -> the committed `variant_count` (.sld frames // frame_count)."""
    data = json.loads(_GRAPHIC_MAP_PATH.read_text())["graphics"]
    return {int(const): int(entry["variant_count"]) for const, entry in data.items() if "variant_count" in entry}


def variant_count_for(unit_const: int) -> int:
    """How many graphic variants this const's file really holds.

    Falls back to angle_count where the .sld was missing at generation time,
    mirroring unit_rotation.angle_count_for(). Render keeps reading the file
    itself (unit_sprites.sld_frame_count()); this is only the install-free
    modulus the edit path needs.
    """
    return _variant_counts().get(unit_const, unit_rotation.angle_count_for(unit_const))


def is_cyclable(unit_const: int) -> bool:
    """The write-path predicate for Cycle Variant."""
    return (
        unit_rotation.semantics_for(unit_const) == unit_rotation.VARIANT
        and unit_const not in excluded_consts()
        and variant_count_for(unit_const) >= 2
    )


def variant_of(rotation: float, angle_count: int, variant_count: int) -> int:
    """The currently selected variant. Delegates so there stays exactly one reader."""
    return unit_sprites.variant_index(rotation, angle_count, variant_count)


def cycle_step(rotation: float, angle_count: int, variant_count: int, steps: int) -> float:
    """The variant `steps` along from the current one, as a literal integer."""
    if variant_count <= 1:
        raise ValueError(f"variant_count must be > 1 to cycle, got {variant_count}")
    return float((variant_of(rotation, angle_count, variant_count) + steps) % variant_count)


def random_variant(rotation: float, angle_count: int, variant_count: int, rng: random.Random) -> float:
    """A uniformly random variant other than the current one, so a press always changes something."""
    if variant_count <= 1:
        raise ValueError(f"variant_count must be > 1 to randomize, got {variant_count}")
    current = variant_of(rotation, angle_count, variant_count)
    pick = rng.randrange(variant_count - 1)
    return float(pick + 1 if pick >= current else pick)
