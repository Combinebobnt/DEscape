"""A synthetic data mod's tables (GH #58): the AoE2 tables with one sentinel per table.

Shared by the data mod tests: tests/test_data_mod_tables.py reads each
sentinel back through its public accessor, and tests/test_units_panel.py
lists the mod's objects in the Units catalog (GH #265).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from descape import dat_derive, game_dataset

DESCAPE = Path(game_dataset.__file__).resolve().parent
ARCHER, SPEARMAN, TOWN_CENTER, CASTLE = 4, 93, 109, 82
NEW_UNIT = 9990  # no install table has it
NEW_TREE = 9991


def load(folder: Path, name: str) -> dict:
    return json.loads((folder / name).read_text())


def save(folder: Path, name: str, data: dict) -> None:
    (folder / name).write_text(json.dumps(data))


def mod_tables(tmp_path: Path) -> game_dataset.DataModTables:
    """A copy of the AoE2 tables in tmp_path/tables with one sentinel per
    table; its mod folder is tmp_path/mod (not created) and NEW_UNIT its only mod-only id."""
    folder = tmp_path / "tables"
    folder.mkdir()
    for name in dat_derive.TABLE_NAMES:
        shutil.copy(DESCAPE / name, folder / name)
    stats = load(folder, "unit_stats.json")
    stats["stats"][str(SPEARMAN)]["hp"] = 4242
    save(folder, "unit_stats.json", stats)
    catalog = load(folder, "object_catalog.json")
    catalog["objects"][str(NEW_UNIT)] = {**catalog["objects"][str(SPEARMAN)], "code": "MODUNIT", "string_id": -1}
    save(folder, "object_catalog.json", catalog)
    graphics = load(folder, "unit_graphic_map.json")
    graphics["graphics"][str(SPEARMAN)]["file_name"] = "sentinel_x1"
    graphics["graphics"][str(NEW_UNIT)] = graphics["graphics"][str(SPEARMAN)]
    save(folder, "unit_graphic_map.json", graphics)
    render_data = load(folder, "unit_render_data.json")
    render_data["buildings"][str(TOWN_CENTER)] = [5, 5]
    save(folder, "unit_render_data.json", render_data)
    trees = load(folder, "tree_unit_ids.json")
    trees["ids"][str(NEW_TREE)] = "MODTREE"
    save(folder, "tree_unit_ids.json", trees)
    hosts = load(folder, "garrison_table.json")
    hosts["hosts"][str(CASTLE)]["cap"] = 77
    save(folder, "garrison_table.json", hosts)
    classes = load(folder, "terrain_classes.json")
    classes["classes"]["0"]["family"] = "sentinel"
    save(folder, "terrain_classes.json", classes)
    textures = load(folder, "terrain_texture_map.json")
    textures["mapping"]["0"] = "sentinel.dds"
    save(folder, "terrain_texture_map.json", textures)
    blend = load(folder, "terrain_blend_map.json")
    blend["terrains"]["0"]["priority"] = 4321
    save(folder, "terrain_blend_map.json", blend)
    art = load(folder, "building_art_map.json")
    art["civ_names"]["1"] = "Sentinelians"
    save(folder, "building_art_map.json", art)
    units = load(folder, "terrain_unit_map.json")
    first = next(iter(units["terrains"]))
    units["terrains"][first] = [{**units["terrains"][first][0], "id": NEW_TREE}]
    save(folder, "terrain_unit_map.json", units)
    return game_dataset.DataModTables("sentinel", folder, tmp_path / "mod", "Sentinel Mod", frozenset({NEW_UNIT}))
