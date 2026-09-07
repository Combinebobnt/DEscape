"""Qt-free specs for the Units-mode inspector's per-field editability --
default-tier testable without a QApplication, mirroring player_fields.py's
shape. viewer.py's inline grid loop
reads FIELDS instead of a bare caption tuple, so which of
ViewerWindow._build_unit_inspector()'s rows gets a real editor -- and what
kind -- has one source of truth.

X, Y, Z and Owner are editable unconditionally (D5's decision); Name is
derived, and Unit ID/Reference ID/Garrisoned in stay plain read-only labels.
Rotation is editable only for the consts unit_rotation classifies ANGLE -- see
`conditional` below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

FLOAT = "float"
PLAYER = "player"
TEXT = "text"


@dataclass(frozen=True)
class UnitFieldSpec:
    field_id: str
    label: str
    kind: str  # FLOAT | PLAYER | TEXT
    editable: bool
    minimum: float | None = None  # FLOAT only
    maximum: float | None = None  # FLOAT only
    decimals: int = 2  # FLOAT only
    # Names a per-const rule the viewer resolves through unit_rotation before
    # it lets this field's editor accept input; None means "editable whenever
    # `editable` is". A string key rather than a callable on purpose: it keeps
    # this module data-only and Qt-free, and keeps the rule itself unit-
    # testable in isolation.
    conditional: str | None = None


# Wide sentinel bounds, not derived from map_width/map_height: nothing in the
# file format keeps a unit's stored x/y/z inside the map, and the in-game
# editor itself allows dragging a unit just off the edge.
_COORD_MIN = -(2**15)
_COORD_MAX = 2**15

FIELDS: tuple[UnitFieldSpec, ...] = (
    UnitFieldSpec("name", "Name", TEXT, editable=False),
    UnitFieldSpec("unit_const", "Unit ID", TEXT, editable=False),
    UnitFieldSpec("player", "Owner", PLAYER, editable=True),
    UnitFieldSpec("x", "X", FLOAT, editable=True, minimum=_COORD_MIN, maximum=_COORD_MAX),
    UnitFieldSpec("y", "Y", FLOAT, editable=True, minimum=_COORD_MIN, maximum=_COORD_MAX),
    UnitFieldSpec("z", "Z", FLOAT, editable=True, minimum=_COORD_MIN, maximum=_COORD_MAX),
    # Raw radians, never degree-formatted -- the value actually stored, so the
    # round-trip stays obvious. Editable only where `rotation` genuinely is an
    # angle: for ~65% of GAIA objects and for every wall it is a shape-variant
    # index (AGENTS.md's hard rule), and for a gate there is only one stored
    # frame. Four decimals, not the default two: one stored frame at
    # angle_count 16 is 0.3927 rad, and two decimals would round every
    # arrow-click off the graphic's own grid. Arrow-stepping is not the
    # frame-accurate path regardless -- the Rotate actions are.
    UnitFieldSpec(
        "rotation", "Rotation", FLOAT, editable=True,
        minimum=0.0, maximum=2 * math.pi, decimals=4,
        conditional="rotation_is_angle",
    ),
    UnitFieldSpec("reference_id", "Reference ID", TEXT, editable=False),
    UnitFieldSpec("garrisoned_in_id", "Garrisoned in", TEXT, editable=False),
)

FIELDS_BY_ID: dict[str, UnitFieldSpec] = {spec.field_id: spec for spec in FIELDS}
