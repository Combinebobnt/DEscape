"""Which four unit_consts are one gate's four orientations, and how a cycle
step moves between them.

A gate carries no rotation: every visible gate const has angle_count == 1
(AGENTS.md's gate hard rule), so there is no second frame for a rotation to
select and the orientation lives in the unit_const instead. "Rotating" a gate
means swapping the const among its four siblings, which changes the footprint:
(4, 1) running +x, (1, 4) running +y, and two sparse 4x4 diagonals.

The groups are derived from the committed tables, not hand-kept. Every
`class == 39` object_catalog entry whose .dat `code` parses as
`<family>GT<orientation><state><digits>` joins the group keyed by (family,
state), with the middle letter as the orientation. That yields exactly 24
complete groups of 4 over 96 consts, with zero disagreements against
unit_graphic_map.json's own `_ne_`/`_se_`/`_e_`/`_n_` file_name token wherever
one exists (measured 2026-09-08 against the catalog and 300 corpus gate
placements; pinned by tests/test_gate_orientation.py so it is re-measured
rather than trusted).

Two data traps, both handled by the derivation rather than by a hand-kept
exclusion list. `code` is not injective (const 1192 also carries GTAC2,
colliding with 81), and `class == 39` is wider than "gates", catching
corner-pillar consts too. A const with no unit_graphic_map.json entry is
skipped, which is exactly what drops 1192: it has no graphic entry and no
corpus placement, while 81 has both. A group whose four slots do not fill
cleanly is dropped rather than half-built, so a future catalog change can
never install the wrong sibling; the 24/96 test is the loud half of that.

Cycle order is A -> C -> B -> D, not the alphabetical A -> B -> C -> D: by run
direction those are 0, 45, 90 and 135 degrees, so one step is 45 degrees and a
quarter turn is two.

Reads the two committed JSON tables directly (stdlib only, no library import
and no configured install), so the default test tier covers it, the same
shape unit_rotation.py uses for the same reason.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

_CATALOG_PATH = Path(__file__).resolve().parent / "object_catalog.json"
_GRAPHIC_MAP_PATH = Path(__file__).resolve().parent / "unit_graphic_map.json"

# object_catalog.json's `objects[<const>]["class"]`, the .dat's unit.class_.
_GATE_CLASS = 39

# <family><GT><orientation><state><digits>: family is the empty string for the
# stone/fortified pair (which carry a trailing digit instead), state is A
# closed / B open / C the 1x1 corner / X the span corner.
_CODE_RE = re.compile(r"^([PSCF]?)GT([ABCD])([ABCX])(\d*)$")

# Cycle order, by run direction rather than alphabetically: A (+x) -> C (+x+y)
# -> B (+y) -> D (+x-y), 45 degrees per step.
_ORIENTATIONS = ("A", "C", "B", "D")

# The coarse Rotate multiplier, unit_rotation.quarter_turn_steps()' gate
# counterpart: a step is 45 degrees, so a quarter turn is two of them.
QUARTER_TURN_STEPS = 2


@lru_cache(maxsize=1)
def groups() -> dict[tuple[str, str], tuple[int, ...]]:
    """(family, state) -> that group's four consts, in cycle order.

    `family` is the code's prefix letter plus its trailing digits: "2" stone,
    "3" fortified, "P" palisade, "S" sea, "C" city, "F" fort. `state` is the
    code's own third letter: A closed, B open, C the 1x1 corner, X the span
    corner.
    """
    objects = json.loads(_CATALOG_PATH.read_text())["objects"]
    graphics = json.loads(_GRAPHIC_MAP_PATH.read_text())["graphics"]
    slots: dict[tuple[str, str], dict[str, list[int]]] = {}
    for const, entry in objects.items():
        if entry.get("class") != _GATE_CLASS or const not in graphics:
            continue
        match = _CODE_RE.match(str(entry.get("code") or ""))
        if match is None:
            continue
        prefix, orientation, state, digits = match.groups()
        slots.setdefault((prefix + digits, state), {}).setdefault(orientation, []).append(int(const))
    return {
        key: tuple(found[o][0] for o in _ORIENTATIONS)
        for key, found in slots.items()
        if all(len(found.get(o, ())) == 1 for o in _ORIENTATIONS)
    }


@lru_cache(maxsize=1)
def _siblings_by_const() -> dict[int, tuple[int, ...]]:
    return {const: consts for consts in groups().values() for const in consts}


def orientation_siblings(unit_const: int) -> tuple[int, ...] | None:
    """`unit_const`'s whole group in cycle order, or None if it is not a gate.

    Includes `unit_const` itself, so a caller validating a proposed swap can
    test membership directly (UnitEditModel.set_unit_const's guard does).
    """
    return _siblings_by_const().get(unit_const)


def is_gate(unit_const: int) -> bool:
    """Whether this const is one of the 96 consts with orientation siblings."""
    return unit_const in _siblings_by_const()


def cycle_const(unit_const: int, steps: int) -> int:
    """`unit_const` advanced `steps` orientations around its own 4-cycle.

    45 degrees per step, positive in the same screen direction as a positive
    unit_rotation.rotate_step() (clockwise), so one Rotate keypress turns a
    gate and an archer the same way.

    Raises for a const with no siblings, never a silent return of the same
    const, mirroring rotate_step()'s loud-refusal contract: a caller that
    thought it had cycled a wall should hear about it.
    """
    siblings = orientation_siblings(unit_const)
    if siblings is None:
        raise ValueError(
            f"unit_const {unit_const} is not a gate: it has no orientation siblings to cycle "
            f"through, and its orientation is not stored in its const"
        )
    return siblings[(siblings.index(unit_const) + steps) % len(siblings)]
