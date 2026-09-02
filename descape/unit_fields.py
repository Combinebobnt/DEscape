"""Qt-free specs for the Units-mode inspector's per-field editability --
default-tier testable without a QApplication, mirroring player_fields.py's
shape (descape-units-edit-ui.md's D5/1.2). viewer.py's inline grid loop
reads FIELDS instead of a bare caption tuple, so which of
ViewerWindow._build_unit_inspector()'s rows gets a real editor -- and what
kind -- has one source of truth.

Only X, Y, Z and Owner are editable (D5's decision, and AGENTS.md's hard
rule on Rotation): Name is derived, and Unit ID/Reference ID/Garrisoned in/
Rotation stay plain read-only labels.
"""

from __future__ import annotations

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
    # Raw, never degree-formatted, and never editable -- AGENTS.md's hard
    # rule: for ~65% of GAIA objects and for every wall/gate, this is a
    # shape-variant index, not an angle.
    UnitFieldSpec("rotation", "Rotation", TEXT, editable=False),
    UnitFieldSpec("reference_id", "Reference ID", TEXT, editable=False),
    UnitFieldSpec("garrisoned_in_id", "Garrisoned in", TEXT, editable=False),
)

FIELDS_BY_ID: dict[str, UnitFieldSpec] = {spec.field_id: spec for spec in FIELDS}
