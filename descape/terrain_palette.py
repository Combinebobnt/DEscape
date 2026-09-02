"""terrain_id -> RGB color, for rendering the map grid without real texture blitting.

When a real AoE2DE install is configured (see asset_source.py), colors are the
actual average color of that terrain's real .dds texture -- authoritative, not
guessed. Otherwise falls back to a hand-picked palette built by keyword-matching
AoE2ScenarioParser's TerrainId enum names, which is approximate by nature (it
can't distinguish e.g. FOREST_OAK from FOREST_PINE, both just match "FOREST").
"""

from __future__ import annotations

import json
from pathlib import Path

from AoE2ScenarioParser.datasets.terrains import TerrainId

from descape import asset_source

_KEYWORD_COLORS: list[tuple[str, tuple[int, int, int]]] = [
    ("DEEP_WATER", (28, 62, 130)),
    ("WATER", (52, 104, 178)),
    ("BEACH", (222, 200, 140)),
    ("DESERT", (214, 186, 121)),
    ("DIRT", (150, 111, 66)),
    ("GRAVEL", (140, 130, 116)),
    ("FOREST", (52, 96, 44)),
    ("JUNGLE", (36, 92, 40)),
    ("RAINFOREST", (30, 84, 36)),
    ("SNOW", (235, 240, 245)),
    ("ICE", (200, 225, 235)),
    ("ROAD", (120, 108, 92)),
    ("ROCK", (110, 108, 104)),
    ("CLIFF", (98, 92, 84)),
    ("MOUNTAIN", (120, 116, 110)),
    ("SHALLOW", (94, 148, 196)),
    ("GRASS", (86, 138, 62)),
]

_DEFAULT_COLOR = (100, 100, 100)

_ID_TO_NAME: dict[int, str] = {t.value: t.name for t in TerrainId}


def _color_for_name(name: str) -> tuple[int, int, int]:
    for keyword, color in _KEYWORD_COLORS:
        if keyword in name:
            return color
    return _DEFAULT_COLOR


_TERRAIN_COLOR_CACHE: dict[int, tuple[int, int, int]] = {
    tid: _color_for_name(name) for tid, name in _ID_TO_NAME.items()
}


def color_for_terrain_id(terrain_id: int) -> tuple[int, int, int]:
    real = asset_source.get_terrain_average_color(terrain_id)
    if real is not None:
        return real
    return _TERRAIN_COLOR_CACHE.get(terrain_id, _DEFAULT_COLOR)


def name_for_terrain_id(terrain_id: int) -> str:
    return _ID_TO_NAME.get(terrain_id, f"UNKNOWN_{terrain_id}")


# unit_const ids for tree/tree-like GAIA flora -- see tools/gen_tree_unit_ids.py.
# Rendered dark green regardless of owner, matching AoE2:DE's own minimap
# (which hardcodes tree coloring rather than reading it from unit data --
# see that script's docstring).
_TREE_UNIT_IDS_PATH = Path(__file__).resolve().parent / "tree_unit_ids.json"
TREE_UNIT_IDS: frozenset[int] = frozenset(
    int(uid) for uid in json.loads(_TREE_UNIT_IDS_PATH.read_text())["ids"]
)
TREE_COLOR: tuple[int, int, int] = (35, 65, 30)

# unit_const -> (span_x, span_y) footprint size in whole tiles (buildings) /
# real minimap RGB (resources) -- see tools/gen_unit_render_data.py. Any
# unit_const not present in either dict is a non-building, non-resource-special
# object: 1-tile dot, owner color.
#
# Spans, not radii. The name says so because every consumer of the older
# BUILDING_FOOTPRINTS read its values as a radius about a centre tile, which
# can only express odd sizes -- the bug that made every even-footprint
# building render a tile too large per axis.
_UNIT_RENDER_DATA = json.loads(
    (Path(__file__).resolve().parent / "unit_render_data.json").read_text()
)
BUILDING_TILE_SPANS: dict[int, tuple[int, int]] = {
    int(uid): (sx, sy) for uid, (sx, sy) in _UNIT_RENDER_DATA["buildings"].items()
}
RESOURCE_COLORS: dict[int, tuple[int, int, int]] = {
    int(uid): tuple(rgb) for uid, rgb in _UNIT_RENDER_DATA["resource_colors"].items()
}

# unit_const -> terrain_id, for every building whose .dat entry declares a
# foundation terrain (`building.foundation_terrain_id >= 0`). A raw mirror of
# the field, not a "renders as terrain" policy -- most buildings here also
# have a real .sld and must keep drawing as their sprite; render.py's
# _terrain_overlay_for is what decides which consts actually use this.
FOUNDATION_TERRAIN: dict[int, int] = {
    int(uid): tid for uid, tid in _UNIT_RENDER_DATA["foundation_terrain"].items()
}

# Distinct colors for GAIA (index 0) + up to 8 players, for unit dots. This is
# the IDENTITY fallback -- player N's color when no valid stored override
# exists -- not the general case: a scenario author picks each player's color
# independently in the in-game editor, and resolve_player_colors() below is
# what actually renders that choice.
PLAYER_COLORS: list[tuple[int, int, int]] = [
    (90, 90, 90),      # GAIA
    (60, 110, 220),    # P1 blue
    (220, 60, 60),     # P2 red
    (70, 200, 90),     # P3 green
    (230, 210, 40),    # P4 yellow
    (60, 200, 200),    # P5 cyan
    (225, 55, 205),    # P6 magenta
    (250, 130, 15),    # P7 orange
    (230, 230, 230),   # P8 white/gray
]

# ColorId-indexed (0..7), for a unit's REAL stored color override. Reuses
# PLAYER_COLORS' hand-tuned RGBs, but the last two are swapped from their
# PLAYER_COLORS position: the game's own ColorId enum orders GRAY (6) before
# ORANGE (7), the reverse of PLAYER_COLORS' "P7 orange, P8 white/gray"
# labeling, which assumed (wrongly) that player N always gets color N.
PLAYER_COLOR_BY_ID: list[tuple[int, int, int]] = [
    PLAYER_COLORS[1],  # ColorId 0 BLUE
    PLAYER_COLORS[2],  # ColorId 1 RED
    PLAYER_COLORS[3],  # ColorId 2 GREEN
    PLAYER_COLORS[4],  # ColorId 3 YELLOW
    PLAYER_COLORS[5],  # ColorId 4 AQUA
    PLAYER_COLORS[6],  # ColorId 5 PURPLE
    PLAYER_COLORS[8],  # ColorId 6 GRAY
    PLAYER_COLORS[7],  # ColorId 7 ORANGE
]


def resolve_player_colors(
    color_ids: list[int] | tuple[int, ...],
) -> tuple[tuple[tuple[int, int, int], ...], tuple[int, ...]]:
    """The 8 stored per-player ColorId overrides (P1..P8, as read from
    PlayerDataTwo -- see scenario_io._read_player_colors()) resolved to two
    9-tuples, both indexed by player_id (0 = GAIA):

    - dots: the RGB each player renders with -- a drop-in for the
      PLAYER_COLORS[player_id % len(PLAYER_COLORS)] expression it replaces.
    - team_indices: the TEAM_COLORS/unit_sprites tint index for each
      player's real sprite -- color_id + 1, since TEAM_COLORS is GAIA-first
      while color_ids is not. Using color_id directly here would tint every
      BLUE (id 0) player untinted (TEAM_COLORS[0] is GAIA's white).

    GAIA is never overridden -- its own PlayerDataTwo slot is known junk,
    not a real color (see the maintainer doc). An out-of-range id (never
    observed across this project's corpus, but not guaranteed) falls back to
    that player's identity default, matching PLAYER_COLORS' own fallback
    role.
    """
    dots: list[tuple[int, int, int]] = [PLAYER_COLORS[0]]
    team_indices: list[int] = [0]
    for player_id in range(1, 9):
        color_id = color_ids[player_id - 1]
        if 0 <= color_id < len(PLAYER_COLOR_BY_ID):
            dots.append(PLAYER_COLOR_BY_ID[color_id])
            team_indices.append(color_id + 1)
        else:
            dots.append(PLAYER_COLORS[player_id])
            team_indices.append(player_id)
    return tuple(dots), tuple(team_indices)
