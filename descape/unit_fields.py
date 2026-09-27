"""Qt-free specs for the Units-mode inspector's per-field editability --
default-tier testable without a QApplication, mirroring player_fields.py's
shape. viewer.py's inline grid loop
reads FIELDS instead of a bare caption tuple, so which of
ViewerWindow._build_unit_inspector()'s rows gets a real editor -- and what
kind -- has one source of truth.

X, Y, Z and Owner are editable unconditionally (D5's decision); Name is
derived, and Unit ID/Reference ID/Garrisoned in stay plain read-only labels.
Rotation is editable only where Rotate may change it (GH #123): as a whole
facing for the consts unit_rotation classifies ANGLE, and as a variant number
for the ones unit_variant calls cyclable -- see `conditional` below.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from descape.iso_geometry import MAX_ELEVATION

FLOAT = "float"
FACING = "facing"
PLAYER = "player"
TEXT = "text"


@dataclass(frozen=True)
class UnitFieldSpec:
    field_id: str
    label: str
    kind: str  # FLOAT | FACING | PLAYER | TEXT
    editable: bool
    minimum: float | None = None  # FLOAT only
    maximum: float | None = None  # FLOAT only
    decimals: int = 2  # FLOAT only
    # Names a per-const rule the panel resolves (units_panel._FIELD_CONDITIONALS) before
    # it lets this field's editor accept input; None means "editable whenever
    # `editable` is". A string key rather than a callable on purpose: it keeps
    # this module data-only and Qt-free, and keeps the rule itself unit-
    # testable in isolation.
    conditional: str | None = None


# The no-scenario fallback only. With a map loaded, the panel narrows X/Y/Z to
# coordinate_bounds() (GH #118), widened over any stored off-map value.
_COORD_MIN = -(2**15)
_COORD_MAX = 2**15

FIELDS: tuple[UnitFieldSpec, ...] = (
    UnitFieldSpec("name", "Name", TEXT, editable=False),
    UnitFieldSpec("unit_const", "Unit ID", TEXT, editable=False),
    UnitFieldSpec("player", "Owner", PLAYER, editable=True),
    UnitFieldSpec("x", "X", FLOAT, editable=True, minimum=_COORD_MIN, maximum=_COORD_MAX),
    UnitFieldSpec("y", "Y", FLOAT, editable=True, minimum=_COORD_MIN, maximum=_COORD_MAX),
    UnitFieldSpec("z", "Z", FLOAT, editable=True, minimum=_COORD_MIN, maximum=_COORD_MAX),
    # A whole facing (GH #61), 0 to angle_count - 1, on ANGLE consts, or a
    # variant number, 0 to variant_count - 1, on cyclable ones (GH #123); the
    # range is per-const, so the panel sets it at populate time. Walls, cliffs,
    # gates and single-frame consts keep a raw read-only label (AGENTS.md).
    UnitFieldSpec(
        "rotation", "Rotation", FACING, editable=True,
        conditional="rotation_is_editable",
    ),
    UnitFieldSpec("reference_id", "Reference ID", TEXT, editable=False),
    UnitFieldSpec("garrisoned_in_id", "Garrisoned in", TEXT, editable=False),
)

FIELDS_BY_ID: dict[str, UnitFieldSpec] = {spec.field_id: spec for spec in FIELDS}


def coordinate_bounds(field_id: str, width: int, height: int) -> tuple[float, float]:
    """The in-map editor range for X, Y or Z (GH #118).

    X uses `width` and Y `height`, each ending one display step short of the
    size: render.unit_tile_bounds is half-open, so an anchor at exactly `size`
    is off the map. Z is elevation in levels, `0..MAX_ELEVATION` on any map
    (the corpus maximum is exactly 15, on maps whose own elevation tops at 4).
    """
    if field_id == "z":
        return 0.0, float(MAX_ELEVATION)
    size = {"x": width, "y": height}[field_id]
    return 0.0, size - 10 ** -FIELDS_BY_ID[field_id].decimals


def widen(bounds: tuple[float, float], values: Iterable[float]) -> tuple[float, float]:
    """`bounds` extended just far enough to include every value, so a stored
    out-of-range value is shown as it is rather than silently clamped."""
    lo, hi = bounds
    values = list(values)
    return min([lo, *values]), max([hi, *values])
