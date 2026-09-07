"""Whether a unit's `rotation` field is an angle the user may turn, or a
graphic-variant index that must be passed through verbatim.

Deliberately separate from unit_sprites.rotation_is_variant(), which stays
untouched: the two answer different questions -- "which frame do I draw" vs.
"may the user change this field" -- and conflating them is exactly the trap
render.py's `rotation = 0.0 if player_id == 0` shortcut sets. That shortcut is
a render rule, not an edit-time one (player-owned walls are common), so nothing
here reads player id.

The whitelist is measured, not judged. Cross-tabulating the .dat's own
`unit.type` (committed in object_catalog.json) against rotation form over the
20-file example corpus, restricted to units whose graphic has angle_count > 1:

    type 70 (creatable)   5551 angle-like vs 256 integer-in-range, and every
                          one of those 256 is exactly 0.0 -- index 0 under
                          either convention, so evidence for neither
    type 10 (eye candy)   13 vs 135467          -> variant index
    type 80 (building)    1515 vs 6896          -> mixed; walls dominate
    type 20 / 30          0 vs 143              -> variant / inert

type 70 has zero counterexamples in 5808 placements. That is the whitelist,
and tests/test_unit_rotation.py's corpus-marked test is what keeps it pinned to
data rather than to one session's judgement.

Gates are INERT rather than excluded by name: all 24 visible gate consts (6
families x 4 orientations) have angle_count == 1, so there is no second frame
for a rotation to select. A gate's orientation lives in its unit_const, not its
rotation -- the graphic file names are `..._ne_closed_x1` / `_se_` / `_e_` /
`_n_`, one const each. Walls are VARIANT for the reason AGENTS.md's hard rule
already records: their stored frames are shapes, and the correct value is a
function of the wall's neighbours, so "rotate this wall" is not a meaningful
user operation.

Reads the two committed JSON tables directly -- stdlib only, no library import
and no configured install -- so the default test tier covers it.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

# semantics_for()'s three answers.
ANGLE = "angle"
VARIANT = "variant"
INERT = "inert"

_CATALOG_PATH = Path(__file__).resolve().parent / "object_catalog.json"
_GRAPHIC_MAP_PATH = Path(__file__).resolve().parent / "unit_graphic_map.json"

# object_catalog.json's `objects[<const>]["type"]` -- the .dat's own unit.type.
_CREATABLE_TYPE = 70

# Trebuchet, packed and unpacked: mobile units the .dat types as buildings
# (80), angle_count 32. Keyed on the exact unit_const, not the shared code
# name. Corpus: 42 is 8 angle-like vs 1 integer-in-range, 331 is 22 vs 1 -- 30
# of 33, the same signature as the type-70 whitelist. 1690/1691 share TREBU/
# PTREB's code name, class and angle_count but have zero corpus placements, so
# they are included on that structural match alone, unconfirmed by data.
#
# Mule Cart (1808) was checked and dropped: both its corpus placements are
# integer-in-range at angle_count 16, the variant signature, not the angle one.
# "It's a mobile unit" was the wrong prior.
_EXTRA_ANGLE_CONSTS: frozenset[int] = frozenset({42, 331, 1690, 1691})


@lru_cache(maxsize=1)
def _object_types() -> dict[int, int]:
    """unit_const -> the .dat's `unit.type`, from the committed catalog."""
    data = json.loads(_CATALOG_PATH.read_text())["objects"]
    return {int(const): entry["type"] for const, entry in data.items() if "type" in entry}


@lru_cache(maxsize=1)
def _angle_counts() -> dict[int, int]:
    """unit_const -> its standing graphic's angle_count. Same table
    unit_sprites.graphic_map() reads, loaded independently to keep this module
    free of that one's install-dependent imports."""
    data = json.loads(_GRAPHIC_MAP_PATH.read_text())["graphics"]
    return {int(const): int(entry["angle_count"]) for const, entry in data.items()}


def angle_count_for(unit_const: int) -> int:
    """How many frames this const's graphic stores, or 1 for a const the
    graphic table doesn't cover (which is INERT either way)."""
    return max(1, _angle_counts().get(unit_const, 1))


def semantics_for(unit_const: int) -> str:
    """ANGLE, VARIANT or INERT -- what this const's `rotation` field means.

    INERT wins over everything when angle_count <= 1: there is nothing to
    select, so neither reading applies and there is nothing to show a user.
    """
    if angle_count_for(unit_const) <= 1:
        return INERT
    if unit_const in _EXTRA_ANGLE_CONSTS:
        return ANGLE
    if _object_types().get(unit_const) == _CREATABLE_TYPE:
        return ANGLE
    return VARIANT


def rotation_is_angle(unit_const: int) -> bool:
    """The write-path predicate: may a Rotate action touch this const?

    Named as a rule string in unit_fields.UnitFieldSpec.conditional, so the
    inspector's per-const editability resolves through here rather than
    duplicating the check.
    """
    return semantics_for(unit_const) == ANGLE


def quarter_turn_steps(angle_count: int) -> int:
    """Frames closest to a quarter turn, never fewer than one.

    Rounds to a whole number of frames rather than writing an exact 90
    degrees: that is 4 steps at angle_count 16 and 2 at 8, but 1.5 at 6 (two
    catalog consts), and writing the raw quarter-turn there would put the
    stored value off the graphic's own grid for no visible gain. At
    angle_count 2 the closest available step is a half turn, not a quarter --
    the floor of 1 is what the name has to give up when the graphic stores
    nothing finer.

    floor(x + 0.5), not round(), for the reason unit_sprites.angle_index()
    already records -- round() is banker's rounding and breaks .5 ties toward
    even, i.e. inconsistently.
    """
    return max(1, math.floor(angle_count / 4 + 0.5))


def rotate_step(rotation: float, angle_count: int, steps: int) -> float:
    """`rotation` advanced by `steps` whole stored frames, wrapped into
    [0, 2*pi).

    ANGLE_ZERO_OFFSET_DEG is deliberately not applied, and that is not an
    oversight: it is a constant shift between the field and the stored angle
    set, so advancing the field by one whole step advances the drawn frame by
    exactly one step regardless. Per unit_sprites' own measurement, AoE2
    rotation grows clockwise on screen, so positive `steps` rotates clockwise.

    Also the normalizer for a typed inspector value -- steps=0 wraps a stored
    junk value (7.0 appears 574 times in the corpus) into range without moving
    it off its own frame.
    """
    if angle_count <= 1:
        raise ValueError(f"angle_count must be > 1 to rotate, got {angle_count}")
    turn = 2 * math.pi
    return (rotation + steps * turn / angle_count) % turn
