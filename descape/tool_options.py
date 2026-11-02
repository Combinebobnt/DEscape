"""The Tool Options row's schema: every option's name, kind, legal values and
schema default. Qt-free and a leaf (it imports only brush and iso_geometry),
so settings.py can import it.

Three tiers, as in daubED's tool defaults:

- **schema default**: `ToolOption.default`, fixed for the build.
- **user default**: what Settings > Tools stored, under the config key
  `tool_options`, read by `settings.get_tool_option_defaults()`.
- **session value**: the row's own widget, from launch until DEscape closes.
  A toolbar edit writes nothing.

**Only overrides are stored.** A file holding every value would pin today's
default for anyone who had run DEscape once, so a later build that changed a
schema default would never reach them. `prune_to_overrides()` drops every
entry equal to its schema default; it lives beside the schema and the
settings writer calls it, so the storage layer never has to know what a
default is.

**One value per option, not per tool.** One widget serves several tools (one
brush size for every brush tool), so the store is a flat `{option: value}`.

The two data-driven ids (Cliff family and piece) default to None, meaning
"the row's first entry": their first entry comes from game data, so a
literal default would pin one dataset's. `normalize()` checks shape only;
whether an id exists in this session's game data is the viewer's check
(`descape/tool_option_bindings.py`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from descape import brush, iso_geometry

BOOL = "bool"
INT = "int"
CHOICE = "choice"
ID = "id"
SET = "set"
KINDS = (BOOL, INT, CHOICE, ID, SET)


@dataclass(frozen=True)
class ToolOption:
    name: str
    label: str
    group: str
    kind: str
    default: object
    minimum: int | None = None  # INT only
    maximum: int | None = None  # INT only; None is unbounded (Cliff frame's range is per piece)
    choices: tuple = ()  # CHOICE only
    members: tuple = ()  # SET only, in the order the canonical list keeps
    id_type: type | None = None  # ID only
    nullable: bool = False  # ID only: whether None is a value
    # The legal values in words, for config.example.yaml, where the kind alone can't say (ids).
    note: str = ""


# (group id, title), in the Settings > Tools tab's order.
GROUPS: tuple[tuple[str, str], ...] = (
    ("brush", "Brush"),
    ("terrain", "Terrain painting"),
    ("elevation", "Set Elevation"),
    ("rect", "Draw Rectangle"),
    ("paste", "Paste Region"),
    ("placement", "Placement"),
    ("cliff", "Cliff"),
    ("unit_brush", "Unit Brush"),
    ("convert", "Convert"),
    ("eraser", "Eraser"),
    ("trigger_area", "Trigger areas"),
)

# The Convert Sources menu's player ids, GAIA (0) to player 8. unit_filter's
# GAIA_PLAYER_ID/MAX_PLAYER_ID are not imported: unit_filter is not a leaf.
CONVERT_SOURCES = tuple(range(9))
# eraser.TARGET_LABELS' names and EraseTargets()' defaults, as literals for the
# same reason (eraser reaches settings through unit_pick). Pinned by tests.
ERASER_TARGETS = ("units", "buildings", "trees", "overlay", "invisible", "gaia", "cliffs")
ERASER_DEFAULT_TARGETS = ("units", "buildings")
# Stone Wall in AoE2, Medium Wall in Return of Rome. Pinned against wall_family_consts() by tests.
DEFAULT_WALL_CONST = 117

# In restore order: Cliff family before piece before frame, since each re-ranges the next.
OPTIONS: tuple[ToolOption, ...] = (
    ToolOption(
        "brush_size", "Brush size", "brush", INT, brush.BRUSH_SIZE_MIN,
        minimum=brush.BRUSH_SIZE_MIN, maximum=brush.BRUSH_SIZE_MAX,
    ),
    ToolOption("brush_shape", "Brush shape", "brush", CHOICE, brush.BRUSH_SHAPE_SQUARE, choices=brush.BRUSH_SHAPES),
    ToolOption(
        "brush_thickness", "Outline thickness", "brush", INT, brush.BRUSH_THICKNESS_MIN,
        minimum=brush.BRUSH_THICKNESS_MIN, maximum=brush.BRUSH_THICKNESS_MAX,
    ),
    ToolOption("auto_beach", "Auto beach", "terrain", BOOL, False),
    ToolOption(
        "beach_terrain", "Beach terrain", "terrain", ID, None, id_type=int, nullable=True,
        note="null for Auto, or a beach terrain id",
    ),
    ToolOption("beach_width", "Beach width", "terrain", INT, 1, minimum=1, maximum=3),
    ToolOption("paint_trees", "Trees", "terrain", BOOL, True),
    ToolOption("paint_eye_candy", "Eye candy", "terrain", BOOL, False),
    ToolOption(
        "elevation_level", "Elevation level", "elevation", INT, 0, minimum=0, maximum=iso_geometry.MAX_ELEVATION,
    ),
    ToolOption("rect_filled", "Filled rectangle", "rect", BOOL, True),
    ToolOption("paste_terrain", "Paste terrain", "paste", BOOL, True),
    ToolOption("paste_elevation", "Paste elevation", "paste", BOOL, True),
    ToolOption("paste_units", "Paste units", "paste", BOOL, True),
    ToolOption("free_placement", "Free placement", "placement", BOOL, False),
    ToolOption("off_grid", "Off-grid", "placement", BOOL, False),
    ToolOption("walls_skip_occupied", "Walls skip occupied tiles", "placement", BOOL, False),
    ToolOption("gate_walls", "Moving a gate updates walls", "placement", BOOL, False),
    ToolOption(
        "last_wall", "Wall", "placement", ID, DEFAULT_WALL_CONST, id_type=int,
        note="a wall unit id",
    ),
    ToolOption(
        "cliff_family", "Cliff family", "cliff", ID, None, id_type=str, nullable=True,
        note="null for the first family, or a cliff graphic name",
    ),
    ToolOption(
        "cliff_piece", "Cliff piece", "cliff", ID, None, id_type=int, nullable=True,
        note="null for the family's first piece, or a cliff unit id",
    ),
    ToolOption("cliff_frame", "Cliff frame", "cliff", INT, 0, minimum=0),
    ToolOption("unit_brush_density", "Unit Brush density", "unit_brush", INT, 100, minimum=1, maximum=100),
    ToolOption("unit_brush_random", "Random facing/variant", "unit_brush", BOOL, True),
    ToolOption(
        "convert_sources", "Convert sources", "convert", SET, list(CONVERT_SOURCES), members=CONVERT_SOURCES,
    ),
    ToolOption(
        "eraser_targets", "Eraser targets", "eraser", SET, list(ERASER_DEFAULT_TARGETS), members=ERASER_TARGETS,
    ),
    ToolOption("area_edit", "Edit Area", "trigger_area", BOOL, False),
    ToolOption("area_move", "Move areas and locations", "trigger_area", BOOL, True),
    ToolOption("area_resize", "Resize areas", "trigger_area", BOOL, True),
    ToolOption("area_tile_borders", "Tile borders", "trigger_area", BOOL, False),
)

BY_NAME: dict[str, ToolOption] = {option.name: option for option in OPTIONS}


def option(name: str) -> ToolOption:
    """The schema entry, or ValueError for a name the schema doesn't have."""
    try:
        return BY_NAME[name]
    except KeyError:
        raise ValueError(f"unknown tool option {name!r}") from None


def _whole(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _same(a: object, b: object) -> bool:
    """Equal and the same type, so True never matches 1 and 1.0 never matches 1."""
    return type(a) is type(b) and a == b


def normalize(name: str, raw: object, *, lenient: bool = False) -> object:
    """`raw` in canonical form, or ValueError. Never coerces: a bool must be a
    real bool, an int a real int (never a bool) in range, a choice one of the
    choices, an id its id_type (or None where nullable). A set's canonical form
    is a list in the schema's member order (yaml writes a Python set as !!set);
    its input must be a list or tuple. `lenient`, for the loader only, drops a
    set's illegal members instead of raising."""
    spec = option(name)
    if spec.kind == BOOL:
        if isinstance(raw, bool):
            return raw
    elif spec.kind == INT:
        if _whole(raw) and (spec.minimum is None or raw >= spec.minimum) and (
            spec.maximum is None or raw <= spec.maximum
        ):
            return raw
    elif spec.kind == CHOICE:
        for choice in spec.choices:
            if _same(raw, choice):
                return choice
    elif spec.kind == ID:
        if raw is None:
            legal = spec.nullable
        elif spec.id_type is int:
            legal = _whole(raw)
        else:
            legal = isinstance(raw, spec.id_type)
        if legal:
            return raw
    elif spec.kind == SET and isinstance(raw, (list, tuple)):
        picked = []
        for member in raw:
            if not any(_same(member, allowed) for allowed in spec.members):
                if lenient:
                    continue
                raise ValueError(f"{name}: {member!r} is not one of {list(spec.members)}")
            picked.append(member)
        return [allowed for allowed in spec.members if any(_same(allowed, member) for member in picked)]
    shown = "NaN" if isinstance(raw, float) and math.isnan(raw) else repr(raw)
    raise ValueError(f"{name}: {shown} is not a legal {spec.kind} value")


def default(name: str) -> object:
    """The schema default, a fresh copy for a set."""
    value = option(name).default
    return list(value) if isinstance(value, list) else value


def defaults() -> dict[str, object]:
    return {spec.name: default(spec.name) for spec in OPTIONS}


def is_default(name: str, value: object) -> bool:
    """Whether a canonical `value` equals `name`'s schema default (a set in any order)."""
    spec = option(name)
    if spec.kind == SET:
        return isinstance(value, list) and normalize(name, value) == spec.default
    return _same(value, spec.default)


def prune_to_overrides(values: dict[str, object]) -> dict[str, object]:
    """Only the entries worth storing, in schema order: unknown names and every
    value equal to its schema default are dropped. Raises ValueError on an
    illegal value; the settings setters validate before anything reaches here."""
    kept: dict[str, object] = {}
    for spec in OPTIONS:
        if spec.name in values:
            value = normalize(spec.name, values[spec.name])
            if not is_default(spec.name, value):
                kept[spec.name] = value
    return kept
