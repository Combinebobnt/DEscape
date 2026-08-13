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

# unit_const -> (radius_x, radius_y) footprint (buildings) / real minimap RGB
# (resources) -- see tools/gen_unit_render_data.py. Any unit_const not present
# in either dict is a non-building, non-resource-special object: 1-tile dot,
# owner color.
_UNIT_RENDER_DATA = json.loads(
    (Path(__file__).resolve().parent / "unit_render_data.json").read_text()
)
BUILDING_FOOTPRINTS: dict[int, tuple[int, int]] = {
    int(uid): (rx, ry) for uid, (rx, ry) in _UNIT_RENDER_DATA["buildings"].items()
}
RESOURCE_COLORS: dict[int, tuple[int, int, int]] = {
    int(uid): tuple(rgb) for uid, rgb in _UNIT_RENDER_DATA["resource_colors"].items()
}

# Distinct colors for GAIA (index 0) + up to 8 players, for unit dots.
PLAYER_COLORS: list[tuple[int, int, int]] = [
    (90, 90, 90),      # GAIA
    (60, 110, 220),    # P1 blue
    (220, 60, 60),     # P2 red
    (70, 200, 90),     # P3 green
    (230, 210, 40),    # P4 yellow
    (60, 200, 200),    # P5 cyan
    (220, 120, 220),   # P6 magenta/pink
    (140, 90, 40),     # P7 orange/brown
    (230, 230, 230),   # P8 white/gray
]
