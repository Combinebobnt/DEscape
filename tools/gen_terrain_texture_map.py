#!/usr/bin/env python3
"""
Regenerates descape/terrain_texture_map.json: a terrain_id -> AoE2DE terrain
texture filename table.

This is pure factual id/filename correspondence data extracted from the game's
own terrain table (via genieutils-py, which parses empires2_x2_p1.dat) -- not
game asset content itself. The .dds texture files it points at are never
bundled in this repo; see descape/asset_source.py for how they get loaded from
a configured AoE2DE install at runtime, with a graceful fallback when one isn't
configured.

Needs genieutils-py (`pip install -r requirements-dev.txt`) and a real AoE2DE
install -- neither of which this repo depends on for normal use, only for
regenerating this file if the game updates its terrain table.

Normally run through tools/regen_game_data.py, which also restamps
tools/game_data_stamp.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _game_root

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _texture_mapping(terrains, available: set[str]) -> tuple[dict[str, str], list[tuple[int, str, str]]]:
    """terrain_id -> "<name_2>.dds" for each enabled, named terrain whose
    texture the install has, and (id, name, name_2) for those it lacks. A
    lenient run (a data mod's derive, GH #272) keeps those too: the data
    mod's art mod ships them, asset_source asks Mod Art first, and a texture
    nothing ships still draws the flat colour."""
    mapping: dict[str, str] = {}
    missing: list[tuple[int, str, str]] = []
    for i, t in enumerate(terrains):
        if not t.enabled or not t.name_2:
            continue
        stem = t.name_2.strip()
        if stem not in available:
            missing.append((i, t.name, stem))
        if stem in available or _game_root.lenient():
            mapping[str(i)] = f"{stem}.dds"
    return mapping, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "aoe2de_root",
        type=Path,
        help="Path to the AoE2DE install root (contains resources/_common/...)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "descape",
        help="Where to write terrain_texture_map.json (default: descape/)",
    )
    _game_root.add_dataset_argument(parser)
    args = parser.parse_args()
    _game_root.set_lenient(args.lenient)

    from genieutils.datfile import DatFile

    dat_path = _game_root.dat_path(args.aoe2de_root, args.dataset, args.dat)
    tex_dir = _game_root.folder(args.aoe2de_root, args.dataset, "resources/_common/terrain/textures/2x")
    if not dat_path.is_file():
        raise SystemExit(f"Not found: {dat_path}")
    if not tex_dir.is_dir():
        raise SystemExit(f"Not found: {tex_dir}")

    data = DatFile.parse(str(dat_path))
    terrains = data.terrain_block.terrains
    available = {f.stem for f in tex_dir.glob("*.dds")}
    mapping, missing = _texture_mapping(terrains, available)

    if missing:
        kept = " (kept: a data mod's art mod may ship them)" if _game_root.lenient() else ""
        print(f"WARNING: {len(missing)} enabled terrain(s) had no matching .dds file{kept}:")
        for i, name, stem in missing:
            print(f"  id={i} name={name!r} name_2={stem!r}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "terrain_texture_map.json"
    out = {
        "_comment": (
            "terrain_id -> AoE2DE terrain texture filename "
            "(resources/_common/terrain/textures/2x/). Pure factual id/filename "
            "correspondence extracted from the game's own data table via "
            "genieutils-py, not game asset content itself -- the .dds files "
            "are never bundled here. Regenerate with tools/gen_terrain_texture_map.py."
        ),
        "mapping": mapping,
    }
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(mapping)} entries to {out_path}")


if __name__ == "__main__":
    main()
