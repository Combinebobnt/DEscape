#!/usr/bin/env python3
"""Writes the before/after files for confirming, in the old games themselves,
that a pre-DE (.scn/.scx) scenario saved back by DEscape still opens: one
set per band (Age of Kings 1.20, The Conquerors 1.22, HD Edition 1.26), the
check nothing in-session can drive.

Per source: the untouched original, one file per edit family through the
real edit models and scenario_write.write_scenario(), and one with all four:

- terrain: an L of one terrain along the upper-left edge, near the west tip
  (tile (0, 0)), its long leg on that edge and its short leg going inward;
- elevation: a hill of up to two levels near the south tip, through elevation_tools;
- messages: Hints set to text with cp1252-only characters, Instructions
  to a text of a new length, which also rewrites the header's copy, and one
  field stored empty (no trailing NUL) to text, which is then saved without one;
- diplomacy: player 1's stance toward player 2 only, so a mirrored write shows.

A band's families stay closed in DEscape (legacy_write.ALLOWED) until its pass;
this tool widens them for its own writes only (legacy_write._allow_for_oracle()).
Each file is reloaded and its edit read back before README.txt says what to check.

Writes into build/legacy_oracle/<band>/, which is gitignored.
"""

from __future__ import annotations

import argparse
import shutil
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import batch_api, diplomacy_fields, elevation_tools, legacy_write, messages_fields
from descape.messages_model import MessagesEditModel
from descape.options_model import OptionsEditModel, diplomacy_write_supported
from descape.scenario_io import is_workshop_subscribed
from descape.scenario_write import WriteBlockedError, write_scenario
from descape.terrain_palette import name_for_terrain_id

EXAMPLES = ROOT / "examples"
SOURCES = (EXAMPLES / "wallace_5.scn", EXAMPLES / "ring75_v0.scx", EXAMPLES / "real_world_bohemia.scx")
TERRAIN_CHOICES = (14, 6, 11)  # Desert, Dirt 1, Dirt 2: every band has them; the first the L's tiles lack
HILL = 2  # levels, capped at the band's highest
HINTS = "Résumé – “test” €"
INSTRUCTIONS = "Oracle check: these instructions were saved by DEscape, and are longer than before."
WAS_EMPTY = "Oracle check: this field was empty, and is now saved with no trailing NUL."
ENEMY, NEUTRAL = 3, 1
FAMILIES = (legacy_write.TERRAIN, legacy_write.ELEVATION, legacy_write.MESSAGES, legacy_write.DIPLOMACY)


def _l_tiles(width: int) -> list[tuple[int, int]]:
    return [(x, 0) for x in range(2, min(12, width))] + [(2, y) for y in range(1, 4)]


def _edit(loaded, families) -> tuple[dict, list[str]]:
    """Applies each family's edit to `loaded`; returns write_scenario()'s model kwargs and what to look for."""
    mm = loaded.map_manager
    kwargs: dict = {}
    notes: list[str] = []
    if legacy_write.TERRAIN in families:
        tiles = _l_tiles(mm.map_width)
        present = {mm.get_tile(x, y).terrain_id for x, y in tiles}
        terrain = next(t for t in TERRAIN_CHOICES if t not in present and t in legacy_write.legal_terrain(loaded))
        for x, y in tiles:
            mm.get_tile(x, y).terrain_id = terrain
        notes.append(
            f"terrain: an L of {name_for_terrain_id(terrain)} (id {terrain}) at the map's west (left) tip, "
            "10 tiles along the upper-left edge from 2 tiles in, plus 3 tiles inward at its near end"
        )
    if legacy_write.ELEVATION in families:
        x, y = 4, mm.map_height - 5
        target = min(mm.get_tile(x, y).elevation + HILL, legacy_write.max_elevation(loaded, 0))
        if target <= mm.get_tile(x, y).elevation:
            raise WriteBlockedError(f"tile ({x}, {y}) is already at this band's highest elevation")
        elevation_tools.set_tile_elevation(mm, x, y, target)
        notes.append(f"elevation: a hill up to level {target} 4 tiles in from the map's south (bottom) tip, tile ({x}, {y})")
    if legacy_write.MESSAGES in families:
        if not loaded.messages_write_supported:
            raise WriteBlockedError(f"{loaded.path.name}: its Messages section did not verify")
        messages = kwargs["messages"] = MessagesEditModel(loaded)
        messages.set_value("hints", HINTS)
        messages.set_value("instructions", INSTRUCTIONS)
        where = "Instructions, and the scenario description, read" if messages._header_sync_verified else "Instructions reads"
        notes.append(f"messages: Hints reads '{HINTS}'; {where} '{INSTRUCTIONS}'")
        if empty := _empty_field(loaded, messages):
            messages.set_value(empty.field_id, WAS_EMPTY)
            notes.append(f"messages: {empty.label}, stored empty before, reads '{WAS_EMPTY}'")
    if legacy_write.DIPLOMACY in families:
        if not diplomacy_write_supported(loaded):
            raise WriteBlockedError(f"{loaded.path.name}: its Diplomacy section did not verify")
        options = kwargs["options"] = OptionsEditModel(loaded)
        before = diplomacy_fields.stance_value(loaded, 1, 2)
        stance = NEUTRAL if before == ENEMY else ENEMY
        options.set_value(diplomacy_fields.stance_cell_id(1, 2), stance)
        back = diplomacy_fields.stance_value(loaded, 2, 1)
        notes.append(
            f"diplomacy: player 1 is {_stance(stance)} toward player 2 (was {_stance(before)}); "
            f"player 2 toward player 1 stays {_stance(back)}"
        )
    return kwargs, notes


def _empty_field(loaded, messages):
    """A Messages text field other than Hints and Instructions stored empty with no NUL, else None."""
    fields = messages_fields.present_fields(loaded._scenario.sections["Messages"].retriever_map)
    return next(
        (f for f in fields if f.field_id not in ("hints", "instructions") and messages._raw[f.field_id] == b""), None
    )


def _stance(value: int) -> str:
    return {0: "Ally", 1: "Neutral", 3: "Enemy"}.get(value, f"stance {value}")


def _read_back(path: Path, families, source) -> list[str]:
    """Why the written file does not read back with its edit, or []."""
    back = batch_api.load(path)
    problems = []
    if not (back.terrain_write_supported and back.units_write_supported and back.messages_write_supported):
        problems.append("a load-time verifier failed")
    if struct.unpack_from("<I", back.header_bytes, 4)[0] != len(back.header_bytes) - 8:
        problems.append("header_length does not match the header")
    if legacy_write.TERRAIN in families:
        x, y = _l_tiles(back.map_manager.map_width)[0]
        if back.map_manager.get_tile(x, y).terrain_id == source.map_manager.get_tile(x, y).terrain_id:
            problems.append("the terrain L did not read back")
    if legacy_write.ELEVATION in families:
        x, y = 4, back.map_manager.map_height - 5
        if back.map_manager.get_tile(x, y).elevation <= source.map_manager.get_tile(x, y).elevation:
            problems.append("the hill did not read back")
    if legacy_write.MESSAGES in families:
        model = MessagesEditModel(back)
        if (model.original_value("hints"), model.original_value("instructions")) != (HINTS, INSTRUCTIONS):
            problems.append("the Messages text did not read back")
        if not model._header_sync_verified and MessagesEditModel(source)._header_sync_verified:
            problems.append("the header's Instructions copy disagrees with the Messages copy")
        if (empty := _empty_field(source, MessagesEditModel(source))) and model.original_value(empty.field_id) != WAS_EMPTY:
            problems.append(f"the {empty.label} text did not read back")
    if legacy_write.DIPLOMACY in families and diplomacy_fields.stance_value(back, 1, 2) == diplomacy_fields.stance_value(
        source, 1, 2
    ):
        problems.append("the stance did not read back")
    return problems


def _band(source_path: Path, out_root: Path) -> list[str]:
    source = batch_api.load(source_path)
    fmt = source.source_format
    if not fmt.legacy or fmt.data_version not in legacy_write.ALLOWED:
        raise WriteBlockedError(f"{source_path.name}: not a pre-DE band this tool covers ({fmt.data_version})")
    out = out_root / fmt.data_version
    out.mkdir(parents=True, exist_ok=True)
    suffix = legacy_write.save_suffix(source)
    stem = source_path.stem[:16].replace(" ", "_")
    original = out / f"original_{stem}{suffix}"
    shutil.copyfile(source_path, original)
    lines = [f"{fmt.display_name} (data {fmt.data_version}): {source_path.name}", "", f"{original.name}: untouched"]
    for name, families in (*((f, (f,)) for f in FAMILIES), ("combined", FAMILIES)):
        loaded = batch_api.load(source_path)
        try:
            kwargs, notes = _edit(loaded, families)
        except WriteBlockedError as e:
            lines.append(f"{name}: skipped, {e}")
            continue
        path = out / f"{name}_{stem}{suffix}"
        with legacy_write._allow_for_oracle(fmt.data_version):
            write_scenario(loaded, path, backup=False, **kwargs)
        problems = _read_back(path, families, source)
        if problems:
            raise WriteBlockedError(f"{path}: {'; '.join(problems)}")
        lines.append(f"{path.name}:")
        lines += [f"  {note}" for note in notes]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="*", type=Path, default=list(SOURCES), help="Pre-DE scenarios, one per band")
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "legacy_oracle", help="Output directory")
    args = parser.parse_args()
    if is_workshop_subscribed(args.out):
        raise WriteBlockedError(f"Refusing to write into a subscribed-mods (mods/subscribed/) folder: {args.out}")
    missing = [p for p in args.sources if not p.is_file()]
    if missing:
        print(f"missing source: {', '.join(map(str, missing))}", file=sys.stderr)
        return 1
    lines = [
        "For each file below: copy it into the game's scenario folder, open it in that game's editor",
        "(it opens, and the edit shows where it says), press Test (the scenario starts), then save it",
        "from the old editor and open that save in DEscape (it loads). For a diplomacy file, check the",
        "stance on the in-game diplomacy screen too, not only in the editor.",
        "",
    ]
    for source in args.sources:
        lines += [*_band(source, args.out), ""]
    readme = args.out / "README.txt"
    readme.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
