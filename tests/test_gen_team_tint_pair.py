"""tools/gen_team_tint_pair.py writes its pair through the real write path and
reads every unit, colour and civ back (GH #82's in-game probe)."""

from __future__ import annotations

import itertools
import json

from descape.scenario_io import BLANK_TEMPLATE_PATH

import conftest


def test_the_pair_reads_back_with_one_row_per_owner(tmp_path) -> None:
    module = conftest.load_verify_module("gen_team_tint_pair")

    assert module.write_pair(BLANK_TEMPLATE_PATH, tmp_path) == 0  # 1 on any readback problem

    units = json.loads((tmp_path / "layout.json").read_text())["units"]
    assert [u["player"] for u in units if u["column"] == 0] == list(range(9))  # GAIA first, then P1-8
    assert all(u["team_index"] == u["player"] for u in units)  # ColorId = slot - 1
    assert len({(u["x"], u["y"]) for u in units}) == len(units) == 27


def test_rows_clear_the_house_and_read_left_to_right() -> None:
    module = conftest.load_verify_module("gen_team_tint_pair")
    units = module.layout(0, 0)
    # Screen y in half-tile widths is (y - x) / 2. The x1 House frame is 120 px
    # tall and a half-tile width 48 px, so stacked Houses need 2.5 apart.
    houses = sorted((u["y"] - u["x"]) / 2 for u in units if u["const"] == module.HOUSE)
    assert min(b - a for a, b in itertools.pairwise(houses)) > 120 / 48
    per_row: dict[int, list[float]] = {}
    for u in units:
        per_row.setdefault(u["row"], []).append(u["x"] + u["y"])
    assert all(sorted(xs) == xs for xs in per_row.values())  # Knight, Militia, House left to right
