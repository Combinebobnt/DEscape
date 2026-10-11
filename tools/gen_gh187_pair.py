#!/usr/bin/env python3
"""Writes a before/after .aoe2scenario pair asking the real AoE2:DE editor
whether GAIA's Players-tab settings read what DEscape writes (GH #187).

"before" is a zero-edit write_scenario() of the source, "after" sets GAIA's
rows through OptionsEditModel then write_scenario(), the code Players mode
runs on Save:

- Civilization BRITON-CIV (ROMANS-CIV on 8tp3w9j, the same length, so it
  rides the player_data_1 splice without resizing it).
- Architecture HUN-CIV, 3 bytes shorter than ROMANS-CIV: a real str16 resize
  that moves every later offset.
- Starting Age Imperial, Food 4321, Pop Limit 123 (Map.per_player_population_cap
  slot 8, the only GAIA copy; the H pass is what confirms the editor reads it).

The probe then checks P1..P8's bytes, Units.player_data_4 and the other Map
pop-cap slots are unchanged, and reads every edit back. `--probe` reruns only
that on an existing pair.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import player_fields as pf
from descape.options_model import OptionsEditModel
from descape.scenario_io import is_workshop_subscribed, load_map_and_units
from descape.scenario_write import WriteBlockedError, write_scenario

SOURCE = ROOT / "examples" / "8tp3w9j.aoe2scenario"
OUT = ROOT / "build" / "gh187"
GAIA = 0
EDITS = {
    "civilization": "BRITON-CIV",
    "architecture": "HUN-CIV",
    "starting_age": 5,
    "food": 4321,
    "pop_limit": 123,
}

EXPECTED = """Expected in the AoE2:DE editor, Players tab, GAIA selected:
  Civilization Britons, Architecture Huns, Starting Age Imperial, Food 4321,
  Pop Limit 123. Wood/Stone/Gold, Color and both locks as in before.
  Player 1..8 unchanged. The editor saves the file without an error.
  Pop Limit 200 instead of 123 means GAIA's Map slot is not what the editor reads."""


def _span(body: bytes, target: pf.PlayerWriteTarget) -> bytes:
    return body[target.offset : target.offset + target.length]


def probe(before_path: Path, after_path: Path) -> int:
    before = load_map_and_units(before_path)
    after = load_map_and_units(after_path)
    failures: list[str] = []

    def check(ok: bool, what: str) -> None:
        print(f"  [{'ok' if ok else 'FAIL'}] {what}")
        if not ok:
            failures.append(what)

    print(f"Probe {after_path} against {before_path}:")
    check(pf.verify_player_block(after), "after's per-player block verifies")
    b_targets, a_targets = pf.write_targets(before), pf.write_targets(after)
    player_keys = [key for key in b_targets if pf.parse_player_field_id(key)[1] != GAIA]
    changed = [
        key for key in player_keys
        if [_span(before.decompressed_body, t) for t in b_targets[key]]
        != [_span(after.decompressed_body, t) for t in a_targets[key]]
    ]
    check(not changed, f"P1..P8 bytes unchanged at every target and mirror ({len(player_keys)} keys){changed or ''}")

    b_pd4, a_pd4 = pf._units_bases(before)["player_data_4"], pf._units_bases(after)["player_data_4"]
    check(
        before.decompressed_body[b_pd4.offset : b_pd4.offset + b_pd4.length]
        == after.decompressed_body[a_pd4.offset : a_pd4.offset + a_pd4.length],
        f"Units.player_data_4 byte-identical ({b_pd4.length} bytes)",
    )
    b_caps = list(before._scenario.sections["Map"].retriever_map["per_player_population_cap"].data)
    a_caps = list(after._scenario.sections["Map"].retriever_map["per_player_population_cap"].data)
    check(
        b_caps[:8] + b_caps[9:] == a_caps[:8] + a_caps[9:],
        f"Map pop-cap slots 0..7 and 9..15 unchanged (slot 8: {b_caps[8]} -> {a_caps[8]})",
    )

    b_pd1 = before._scenario.sections["DataHeader"].retriever_map["player_data_1"].data
    a_pd1 = after._scenario.sections["DataHeader"].retriever_map["player_data_1"].data
    delta = sum(e.byte_length for e in a_pd1) - sum(e.byte_length for e in b_pd1)
    check(
        [e.byte_length for e in b_pd1[:8]] == [e.byte_length for e in a_pd1[:8]],
        f"player_data_1 entries P1..P8 keep their lengths (array delta {delta:+d} bytes, all in GAIA's entry)",
    )

    print(f"\n  {'field':18s} {'GAIA before':>11} -> {'after':<11} {'P1 before':>12} -> after")
    specs_b = {s.field_id: s for s in pf.specs_for_player(before, GAIA)}
    specs_a = {s.field_id: s for s in pf.specs_for_player(after, GAIA)}
    for field_id in sorted(pf.GAIA_FIELDS):
        if field_id not in specs_b or field_id not in specs_a:
            check(False, f"{field_id} has a GAIA row in both files")
            continue
        g0 = pf.current_value(before, specs_b[field_id], GAIA)
        g1 = pf.current_value(after, specs_a[field_id], GAIA)
        p0 = pf.current_value(before, specs_b[field_id], 1)
        p1 = pf.current_value(after, specs_a[field_id], 1)
        print(f"  {field_id:18s} {g0!s:>11} -> {g1!s:<11} {p0!s:>12} -> {p1!s}")
        expected = EDITS.get(field_id, g0)
        check(g1 == expected, f"GAIA {field_id} reads back {expected!r}")
        check(p0 == p1, f"P1 {field_id} unchanged")
    print(f"\n{len(failures)} failed.")
    return 1 if failures else 0


def write_pair(source: Path, out: Path) -> int:
    if is_workshop_subscribed(out):
        raise WriteBlockedError(f"Refusing to write into a subscribed-mods (mods/subscribed/) folder: {out}")
    if not source.is_file():
        print(f"missing source: {source}", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    before, after = out / "before.aoe2scenario", out / "after.aoe2scenario"

    loaded = load_map_and_units(source)
    write_scenario(loaded, before, backup=False)
    model = OptionsEditModel(loaded)
    for field_id, value in EDITS.items():
        model.set_value(pf.player_field_id(field_id, GAIA), value)
    write_scenario(loaded, after, backup=False, options=model)

    print(f"wrote {before}\nwrote {after}\n")
    print(EXPECTED + "\n")
    return probe(before, after)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=SOURCE, help="The scenario to edit GAIA in")
    parser.add_argument("--out", type=Path, default=OUT, help="Output directory")
    parser.add_argument("--probe", action="store_true", help="Only probe the existing pair in --out")
    args = parser.parse_args()
    if args.probe:
        return probe(args.out / "before.aoe2scenario", args.out / "after.aoe2scenario")
    return write_pair(args.source, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
