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

Capture Flag and Unit State (GH #186) are CHOICE rows: a fixed list of raw
values with the in-game editor's own labels. Unit State is in every file,
Capture Flag only where the file stores it: see `requires_codec_field`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from descape.iso_geometry import MAX_ELEVATION

FLOAT = "float"
FACING = "facing"
PLAYER = "player"
TEXT = "text"
CHOICE = "choice"

# Raw values are AoE2ScenarioParser's CaptureFlag and ObjectState enums
# (tests/test_unit_fields.py pins the two equal); labels are the game's own,
# key-value-strings ids 31346-31350 and 31354-31362. A stored value outside a
# table is kept and shown as "Other (N)", never rewritten.
CAPTURE_FLAG_CHOICES: tuple[tuple[int, str], ...] = (
    (-1, "Default"),
    (0, "Convert Never"),
    (1, "Convert Once"),
    (2, "Convert Multiple Times"),
    (3, "Convert Never (Gaia Aggressive)"),
)
# The game's list also starts with a "Default" entry (id 31353). Its stored
# byte has not been measured, so it is not offered here.
UNIT_STATE_CHOICES: tuple[tuple[int, str], ...] = (
    (0, "Foundation"),
    (1, "Almost Alive"),
    (2, "Alive"),
    (3, "Dead"),
    (4, "Almost Dead"),
    (5, "Really Dead"),
    (6, "Undead"),
    (7, "Remove"),
    (8, "Going"),
)


@dataclass(frozen=True)
class UnitFieldSpec:
    field_id: str
    label: str
    kind: str  # FLOAT | FACING | PLAYER | TEXT | CHOICE
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
    choices: tuple[tuple[int, str], ...] = ()  # CHOICE only: (raw value, label)
    # Names a Unit field the loaded FILE must store (scenario_io.
    # unit_field_supported) for this row to be live; the panel disables the
    # editor otherwise. Per file, so deliberately not a `conditional`, which
    # is per const.
    requires_codec_field: str | None = None


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
    # GH #186, in the in-game editor's own order. Capture Flag is stored from 1.59 only.
    UnitFieldSpec(
        "capture_flag", "Capture Flag", CHOICE, editable=True,
        choices=CAPTURE_FLAG_CHOICES, requires_codec_field="capture_flag",
    ),
    UnitFieldSpec("status", "Unit State", CHOICE, editable=True, choices=UNIT_STATE_CHOICES),
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
