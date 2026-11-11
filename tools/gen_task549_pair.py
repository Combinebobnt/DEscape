#!/usr/bin/env python3
"""Writes a before/after .aoe2scenario pair for Place Unit into a garrison
host (TASK-549), from examples/blank_map (two players) through the real
write path: UnitEditModel.add() then scenario_write.write_scenario(), the
code the GUI runs, never a hand-crafted byte patch.

"before" holds four empty player-1 hosts, placed as Place Unit places them.
"after" is the same hosts with occupants added exactly as Place Unit's click
on a host adds them (ViewerWindow._place_into_garrison()): at the host's
point, with `garrisoned_in_id` = the host's reference_id, the Owner combo's
player (GAIA for a GAIA-only const) and the placement rotation from
place_rotation.rotation_for(). Each host holds a different count, so a
swapped or dropped link reads back wrong:

- Castle (P1): an Archer (P1), a Knight (P1) and an Archer (P2), the mixed
  owner the user chose (2026-11-02).
- Watch Tower (P1): one Villager (P1).
- Monastery (P1): one Relic, which goes in as GAIA.
- Battering Ram (P1): two Militia (P1).

Writes into build/task549_pair/, which is gitignored, then reads "after"
back and checks every link.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import garrison, place_rotation, unit_kind
from descape.edit_history import EditHistory
from descape.scenario_io import is_workshop_subscribed, load_map_and_units
from descape.scenario_write import WriteBlockedError, write_scenario
from descape.unit_model import UnitEditModel

SOURCE = ROOT / "examples" / "blank_map.aoe2scenario"
OUT = ROOT / "build" / "task549_pair"
CASTLE, TOWER, MONASTERY, RAM = 82, 79, 104, 1258
ARCHER, KNIGHT, VILLAGER, RELIC, MILITIA = 4, 38, 83, 285, 74

# (label, host const, x, y, [(occupant const, Owner combo player), ...])
HOSTS = (
    ("Castle", CASTLE, 50.0, 60.0, [(ARCHER, 1), (KNIGHT, 1), (ARCHER, 2)]),
    ("Watch Tower", TOWER, 60.5, 60.5, [(VILLAGER, 1)]),
    ("Monastery", MONASTERY, 66.5, 60.5, [(RELIC, 1)]),
    ("Battering Ram", RAM, 56.5, 66.5, [(MILITIA, 1), (MILITIA, 1)]),
)


def _hosts(loaded):
    model = UnitEditModel(loaded)
    model.begin_unit_edit([1])
    hosts = [model.add(1, const, x, y) for _label, const, x, y, _occupants in HOSTS]
    model.commit_unit_edit("Place hosts", EditHistory())
    return model, hosts


def _write(out: Path, with_occupants: bool) -> list:
    loaded = load_map_and_units(SOURCE)
    model, hosts = _hosts(loaded)
    if with_occupants:
        model.begin_unit_edit([0, 1, 2])  # the Relic goes to GAIA
        state = place_rotation.PlaceRotation()
        for host, (_label, host_const, _x, _y, occupants) in zip(hosts, HOSTS, strict=True):
            consts = [const for const, _chosen in occupants]
            assert garrison.refusal(host_const, str(host_const), 0, consts) is None, (host_const, consts)
            for const, chosen in occupants:
                player = unit_kind.placement_owner(const, chosen)
                model.add(
                    player,
                    const,
                    host.x,
                    host.y,
                    getattr(host, "z", 0.0),
                    rotation=place_rotation.rotation_for(state, const, player),
                    garrisoned_in_id=host.reference_id,
                )
        model.commit_unit_edit("Place unit", EditHistory())
    write_scenario(loaded, out, backup=False, units=model)
    return [(h.reference_id, h.x, h.y) for h in hosts]


def _check(after: Path, hosts) -> None:
    units = load_map_and_units(after).unit_manager.units
    for (ref, x, y), (label, _const, _x, _y, occupants) in zip(hosts, HOSTS, strict=True):
        inside = sorted(
            (unit.unit_const, player)
            for player, player_units in enumerate(units)
            for unit in player_units
            if unit.garrisoned_in_id == ref
        )
        want = sorted((const, unit_kind.placement_owner(const, chosen)) for const, chosen in occupants)
        assert inside == want, f"{label}: read back {inside}, wanted {want}"
        stray = [
            u for pu in units for u in pu if u.garrisoned_in_id == ref and (u.x, u.y) != (x, y)
        ]
        assert not stray, f"{label}: an occupant is off the host's point"
        names = ", ".join(f"{const} P{player}" for const, player in want)
        print(f"  {label} (ref {ref}, at {x:g}, {y:g}): {names}")


def main() -> int:
    if is_workshop_subscribed(OUT):
        raise WriteBlockedError(f"Refusing to write into a subscribed-mods (mods/subscribed/) folder: {OUT}")
    if not SOURCE.is_file():
        print(f"missing source: {SOURCE}", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    before, after = OUT / "before.aoe2scenario", OUT / "after.aoe2scenario"
    _write(before, with_occupants=False)
    hosts = _write(after, with_occupants=True)
    print(f"wrote {before.relative_to(ROOT)}\nwrote {after.relative_to(ROOT)}\nRead back, each host holds:")
    _check(after, hosts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
