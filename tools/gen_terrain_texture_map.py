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

Under --dataset aoc (the 1999 game) or aoe1 (the 1997 one) a terrain's texture
is its `slp` id in the install's terrain.drs, named `slp_<id>.png`: the file
DEscape's derived root holds for it, built from that SLP's diamond tiles. Under
aoe1 a terrain with no SLP of its own takes its `terrain_to_draw` terrain's:
most 1997 terrains draw as one of four.

Under --dataset hd (HD Edition) a terrain's texture is its `name_2` PNG,
`resources/_common/terrain/textures/<name_2>_00_color.png` in the install (the
suffix in either case), named `<name_2>_00_color.png`, the file DEscape's
derived root holds for it. A terrain whose `name_2` is unset ("None": the
forests, Water Bridge, Ice 2, Snow Foundation, Ice Beach) takes its
`terrain_to_draw` terrain's, one step.

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


# Classic datasets whose terrain without an SLP draws as its terrain_to_draw. The 1999 game
# would gain 14 textures by it too; that is TASK-430.04's terrain measurement, not adopted yet.
_DRAWS_THROUGH = frozenset({_game_root.AOE1})


def _drawn_slp(terrains, terrain, through: bool) -> int:
    """A classic terrain's SLP: its own, else (`through`) its terrain_to_draw terrain's, one step, else -1."""
    if terrain.slp >= 0 or not through:
        return terrain.slp
    drawn = terrain.terrain_to_draw
    return terrains[drawn].slp if 0 <= drawn < len(terrains) else -1


def _classic_texture_mapping(
    terrains, slp_ids: set[int], through: bool = False
) -> tuple[dict[str, str], list[tuple[int, str, str]]]:
    """_texture_mapping() for a classic game: "slp_<slp>.png" for each enabled terrain whose drawn SLP terrain.drs holds."""
    mapping: dict[str, str] = {}
    missing: list[tuple[int, str, str]] = []
    for i, t in enumerate(terrains):
        slp = _drawn_slp(terrains, t, through) if t.enabled else -1
        if slp < 0:
            continue
        if slp in slp_ids:
            mapping[str(i)] = f"slp_{slp}.png"
        else:
            missing.append((i, t.name, f"slp_{slp}"))
    return mapping, missing


# HD Edition's texture PNGs, the suffix matched case-insensitively (four spell it _00_COLOR.png).
_HD_TEXTURE_SUFFIX = "_00_color.png"


def _hd_stem(terrains, terrain) -> str:
    """An HD terrain's texture stem: its own name_2, else its terrain_to_draw terrain's, one step, else ''."""
    def own(t) -> str:
        stem = (t.name_2 or "").strip()
        return "" if stem == "None" else stem

    if own(terrain) or not 0 <= terrain.terrain_to_draw < len(terrains):
        return own(terrain)
    return own(terrains[terrain.terrain_to_draw])


def _hd_texture_mapping(terrains, available: set[str]) -> tuple[dict[str, str], list[tuple[int, str, str]]]:
    """_texture_mapping() for HD Edition: "<name_2>_00_color.png" for each enabled terrain whose
    drawn stem (casefolded in `available`) the install has a PNG for."""
    mapping: dict[str, str] = {}
    missing: list[tuple[int, str, str]] = []
    for i, t in enumerate(terrains):
        stem = _hd_stem(terrains, t) if t.enabled else ""
        if not stem:
            continue
        if stem.casefold() in available:
            mapping[str(i)] = f"{stem}{_HD_TEXTURE_SUFFIX}"
        else:
            missing.append((i, t.name, stem))
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

    dat_path = _game_root.dat_path(args.aoe2de_root, args.dataset, args.dat)
    if args.dataset in _game_root.CLASSIC:
        try:
            tex_dir = _game_root.classic_archives(args.aoe2de_root, args.dataset, "terrain.drs")
        except FileNotFoundError as exc:
            raise SystemExit(f"Not found: {exc}") from None
        found = True
    elif args.dataset == _game_root.HD:
        tex_dir = args.aoe2de_root / "resources/_common/terrain/textures"
        found = tex_dir.is_dir()
    else:
        tex_dir = _game_root.folder(args.aoe2de_root, args.dataset, "resources/_common/terrain/textures/2x")
        found = tex_dir.is_dir()
    if not dat_path.is_file():
        raise SystemExit(f"Not found: {dat_path}")
    if not found:
        raise SystemExit(f"Not found: {tex_dir}")

    data = _game_root.parse_dat(dat_path)
    terrains = data.terrain_block.terrains
    if args.dataset in _game_root.CLASSIC:
        mapping, missing = _classic_texture_mapping(
            terrains, set(tex_dir.ids("slp")), through=args.dataset in _DRAWS_THROUGH
        )
    elif args.dataset == _game_root.HD:
        names = (f.name.casefold() for f in tex_dir.iterdir())
        available = {name[: -len(_HD_TEXTURE_SUFFIX)] for name in names if name.endswith(_HD_TEXTURE_SUFFIX)}
        mapping, missing = _hd_texture_mapping(terrains, available)
    else:
        mapping, missing = _texture_mapping(terrains, {f.stem for f in tex_dir.glob("*.dds")})

    if missing:
        kept = " (kept: a data mod's art mod may ship them)" if _game_root.lenient() else ""
        print(f"WARNING: {len(missing)} enabled terrain(s) had no matching texture{kept}:")
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
