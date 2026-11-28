"""Writes the pre-DE trigger vocabularies under descape/versions/legacy/v<data version>/.

Each is the pinned AoE2ScenarioParser's newest DE conditions.json/effects.json,
cut to the ids its game used, its attributes cut to the fields a trigger-1.6
entry stores, and renamed as the old game's editor named them. Run from the repo
root with `.venv/bin/python3 tools/gen_legacy_vocabulary.py`; `--check` exits 1
if a committed file differs. The names are literals below, so the game install
is not needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from descape import library_compat

# (string id, name): Age of Kings' language.dll; The Conquerors' language_x1.dll adds effects 24-29.
AOK_CONDITIONS = {
    0: (10500, "None"),
    1: (10501, "Bring Object to Area"),
    2: (10502, "Bring Object to Object"),
    3: (10503, "Own Objects"),
    4: (10504, "Own Fewer Objects"),
    5: (10505, "Objects in Area"),
    6: (10506, "Destroy Object"),
    7: (10507, "Capture Object"),
    8: (10508, "Accumulate Attribute"),
    9: (10509, "Research Technology"),
    10: (10510, "Timer"),
    11: (10511, "Object Selected"),
    12: (10512, "AI Signal"),
    13: (10513, "Player Defeated"),
    14: (10514, "Object Has Target"),
    15: (10515, "Object Visible"),
    16: (10516, "Object Not Visible"),
    17: (10517, "Researching Tech"),
    18: (10518, "Units Garrisoned"),
    19: (10519, "Difficulty Level"),
}
AOK_EFFECTS = {
    0: (10550, "None"),
    1: (10551, "Change Diplomacy"),
    2: (10552, "Research Technology"),
    3: (10553, "Send Chat"),
    4: (10554, "Play Sound"),
    5: (10555, "Tribute"),
    6: (10556, "Unlock Gate"),
    7: (10557, "Lock Gate"),
    8: (10558, "Activate Trigger"),
    9: (10559, "Deactivate Trigger"),
    10: (10560, "AI Script Goal"),
    11: (10561, "Create Object"),
    12: (10562, "Task Object"),
    13: (10563, "Declare Victory"),
    14: (10564, "Kill Object"),
    15: (10565, "Remove Object"),
    16: (10566, "Change View"),
    17: (10567, "Unload"),
    18: (10568, "Change Ownership"),
    19: (10569, "Patrol"),
    20: (10570, "Display Instructions"),
    21: (10571, "Clear Instructions"),
    22: (10572, "Freeze Unit"),
    23: (10573, "Use Advanced Buttons"),
}
AOC_EFFECTS = {
    **AOK_EFFECTS,
    24: (10574, "Damage Object"),
    25: (10575, "Place Foundation"),
    26: (10576, "Change Object Name"),
    27: (10577, "Change Object HP"),
    28: (10578, "Change Object Attack"),
    29: (10579, "Stop Unit"),
    # 30-33 are HD Edition / UserPatch ids with no Conquerors string: DE's names.
    30: (None, None),
    31: (None, None),
    32: (None, None),
    33: (None, None),
}

# The library attribute names a trigger-1.6 entry can fill: its stored fields
# (legacy_location_object_reference reads as location_object_reference, item_id
# is derived from them) plus the type field.
EFFECT_ATTRIBUTES = frozenset(
    [
        "effect_type", "ai_script_goal", "quantity", "tribute_list", "diplomacy", "number_of_units_selected",
        "location_object_reference", "item_id", "object_list_unit_id", "source_player", "target_player",
        "technology", "string_id", "display_time", "trigger_id", "location_x", "location_y", "area_x1",
        "area_y1", "area_x2", "area_y2", "object_group", "object_type", "instruction_panel_position",
        "message", "sound_name", "selected_object_ids",
    ]
)
CONDITION_ATTRIBUTES = frozenset(
    [
        "condition_type", "quantity", "attribute", "unit_object", "next_object", "object_list",
        "source_player", "technology", "timer", "area_x1", "area_y1", "area_x2", "area_y2", "object_group",
        "object_type", "ai_signal",
    ]
)
# Fields an old game's entry uses that DE's own vocabulary no longer lists for it,
# by kind and type id: found by tests/test_legacy_oracle.py on Attila (2026-11-26).
_EXTRA_ATTRIBUTES = {"effects": {10: ("source_player",)}, "conditions": {}}
# At trigger 1.6 the library splits quantity by 256 into this pair, so a negative value would read as garbage.
_ARMOUR_PAIR = ("armour_attack_quantity", "armour_attack_class")

BANDS = {
    "1.20": (AOK_CONDITIONS, AOK_EFFECTS),  # Age of Kings
    "1.22": (AOK_CONDITIONS, AOC_EFFECTS),  # The Conquerors
    **dict.fromkeys(("1.21", "1.23", "1.24", "1.26"), (AOK_CONDITIONS, AOC_EFFECTS)),  # HD Edition / UserPatch
}


def _identifier(name: str) -> str:
    return name.lower().replace(" ", "_")


def _cut(raw: dict, names: dict, allowed: frozenset[str], extra: dict) -> dict:
    out = {"-1": raw["-1"]}
    for type_id, (_string_id, name) in names.items():
        entry = dict(raw[str(type_id)])
        if name is not None and _identifier(name) != entry["name"]:
            entry["legacy_name"] = _identifier(name)  # shown; `name` stays DE's, which rule tables key on
        attributes = []
        for listed in entry.get("attributes", []):
            attribute = "quantity" if listed in _ARMOUR_PAIR else listed
            if attribute in allowed and attribute not in attributes:
                attributes.append(attribute)
        attributes += [a for a in extra.get(type_id, ()) if a not in attributes]
        entry["attributes"] = attributes
        defaults = entry.get("default_attributes", {})
        if "quantity" in attributes and isinstance(defaults.get("quantity"), list):  # 28/31 ship [] there
            entry["default_attributes"] = {**defaults, "quantity": -1}
        out[str(type_id)] = entry
    return out


def _newest_library_dir() -> Path:
    versions = [d for d in library_compat.VERSIONS_DIR.glob("v*") if (d / "effects.json").is_file()]
    return max(versions, key=lambda d: tuple(int(p) for p in d.name[1:].split(".")))


def build() -> dict[Path, str]:
    """{path: JSON text} for every legacy vocabulary file."""
    source = _newest_library_dir()
    conditions = json.loads((source / "conditions.json").read_text(encoding="utf-8"))
    effects = json.loads((source / "effects.json").read_text(encoding="utf-8"))
    out = {}
    for band, (condition_names, effect_names) in BANDS.items():
        folder = library_compat.LEGACY_VERSIONS_DIR / f"v{band}"
        for kind, raw, names, allowed in (
            ("conditions", conditions, condition_names, CONDITION_ATTRIBUTES),
            ("effects", effects, effect_names, EFFECT_ATTRIBUTES),
        ):
            out[folder / f"{kind}.json"] = json.dumps(_cut(raw, names, allowed, _EXTRA_ATTRIBUTES[kind]), indent=2) + "\n"
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if a committed file differs")
    args = parser.parse_args()
    stale = []
    for path, text in build().items():
        if path.is_file() and path.read_text(encoding="utf-8") == text:
            continue
        stale.append(path)
        if not args.check:
            path.write_text(text, encoding="utf-8")
    for path in stale:
        print(("stale: " if args.check else "wrote ") + str(path.relative_to(REPO)))
    return 1 if args.check and stale else 0


if __name__ == "__main__":
    sys.exit(main())
