"""Which of a tile's two terrains a terrain tool writes: its base
`terrain_id`, or the layered terrain drawn over it (`TerrainTile.layer`,
-1 = none). The in-game editor calls the second one Layering Mode.

Qt-free, and imports only batch_api: fill_tools and the viewer both write
tiles through write_tile(), so the three write rules live in one place.
batch_api must not import this module back.
"""

from __future__ import annotations

from descape.batch_api import set_layer, set_terrain

TERRAIN, LAYER, CLEAR = "terrain", "layer", "clear"
TARGETS = (TERRAIN, LAYER, CLEAR)

# History labels off the Terrain target. settings.ToolDef.stroke_label holds
# one label per tool, so the per-target ones live here.
LAYER_STROKE_LABELS: dict[tuple[str, str], str] = {
    ("draw", LAYER): "Paint layer",
    ("draw", CLEAR): "Clear layer",
    ("fill", LAYER): "Fill layer",
    ("fill", CLEAR): "Clear layer (fill)",
    ("draw_line", LAYER): "Draw layer line",
    ("draw_line", CLEAR): "Clear layer (line)",
    ("draw_rect", LAYER): "Draw layer rectangle",
    ("draw_rect", CLEAR): "Clear layer (rectangle)",
}


def write_tile(tile, target: str, terrain_id: int) -> None:
    """One tile's write for `target`. TERRAIN sets terrain_id and clears the
    layer (batch_api.set_terrain); LAYER sets the layer to terrain_id and
    leaves the base alone (a layer equal to the base is written as none);
    CLEAR removes the layer and ignores terrain_id."""
    if target == TERRAIN:
        set_terrain(tile, terrain_id)
    elif target == LAYER:
        set_layer(tile, terrain_id)
    elif target == CLEAR:
        set_layer(tile, None)
    else:
        raise ValueError(f"unknown terrain target {target!r}")


def stroke_label(tool_id: str, target: str, base_label: str) -> str:
    """The History label for one gesture: the tool's own `base_label` on the
    Terrain target (or for a tool with no per-target label), else the
    LAYER_STROKE_LABELS entry."""
    if target == TERRAIN:
        return base_label
    return LAYER_STROKE_LABELS.get((tool_id, target), base_label)
