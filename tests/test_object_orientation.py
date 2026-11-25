"""object_orientation's derived sibling groups (TASK-329) and
mirror_tools.reorient_object_const(), the const a bridge, chain, longhouse or
blocker takes under a turn or mirror.

The pure cases run in the default tier: object_orientation reads only the
committed JSON tables. The typed 7-row table below is the independent oracle
for the top-ward constants (it and the corpus-marked test at the bottom are
the only checks on them); the composition check is plumbing only, since it
holds for any choice of constants.
"""

from __future__ import annotations

import itertools
import re

import pytest

from descape import game_dataset, gate_orientation, object_orientation, render_units, unit_sprites
from descape.mirror_tools import TRANSFORMS, AngularElement, compose, image_const, reorient_object_const, swaps_axes
from descape.terrain_palette import tile_span

D4 = list(TRANSFORMS)
MOLES = {2421, 2422, 2521, 2522}
DE_BRIDGE_PIECES = {1840, 1841, 1844, 1845, 1993, 1994, 1996, 1997, 1999, 2000, 2039, 2041}
A_TOP, A_MIDDLE, A_BOTTOM, B_TOP, B_MIDDLE, B_BOTTOM = 605, 606, 607, 608, 609, 610
ARCHER = 4
_BRIDGE_CODE_RE = re.compile(r"^BG([A-F])([A-F])$")


def _non_square_candidates() -> set[int]:
    """Every non-square, non-gate, non-cliff const, from tile_span()'s own tables."""
    from descape import terrain_palette

    consts = set()
    for table in (
        terrain_palette.BUILDING_TILE_SPANS,
        terrain_palette.BLOCKER_TILE_SPANS,
        terrain_palette.FISH_TILE_SPANS,
        terrain_palette.DOODAD_TILE_SPANS,
        terrain_palette.STATIC_TILE_SPANS,
    ):
        consts.update(c for c, (sx, sy) in table.items() if sx != sy)
    return {c for c in consts if not gate_orientation.is_gate(c) and c not in unit_sprites.cliff_consts()}


def _counts() -> tuple[int, int, int]:
    found = object_orientation.groups()
    return (
        sum(1 for g in found if len(g) == 4),
        sum(1 for g in found if len(g) == 2),
        sum(len(g) for g in found),
    )


def test_aoe2_has_six_four_groups_and_ten_pairs_and_leaves_out_only_mole_and_de_pieces():
    assert _counts() == (6, 10, 44)
    grouped = {c for g in object_orientation.groups() for c in g}
    candidates = _non_square_candidates()
    assert len(candidates) == 60
    assert candidates - grouped == MOLES | DE_BRIDGE_PIECES


def test_return_of_rome_has_six_four_groups_and_seven_pairs():
    with game_dataset.activated(game_dataset.ROR):
        assert _counts() == (6, 7, 38)
        assert not any(object_orientation.has_siblings(c) for c in (2423, 2424, 2429, 2430, 2559, 2561))
    assert object_orientation.has_siblings(2559)


def test_the_bridge_codes_agree_with_every_bridge_group():
    """BG<X><P>: X pairs A/B, C/D, E/F (the two axes of one family) with the
    same P; P pairs A/C top/bottom and E/F broken top/bottom, B and D (middle,
    cracked) have no end."""
    axes = {"A": "B", "C": "D", "E": "F"}
    ends = {"A": "C", "E": "F"}
    codes = _catalog_codes()
    # The code is not injective: the square wooden bridges (904-913, 1309-1318) reuse BGAA..BGBC.
    candidates = _non_square_candidates()
    bridge_consts = {c for c, code in codes.items() if _BRIDGE_CODE_RE.match(code) and c in candidates}
    assert len(bridge_consts) == 36
    seen = set()
    for group in object_orientation.groups():
        parsed = [_BRIDGE_CODE_RE.match(codes.get(c, "")) for c in group]
        if not any(parsed):
            continue
        assert all(parsed), group
        x0, p0 = parsed[0].groups()
        if len(group) == 4:
            assert [m.groups() for m in parsed] == [(x0, p0), (x0, ends[p0]), (axes[x0], p0), (axes[x0], ends[p0])]
        else:
            assert p0 in "BD" and [m.groups() for m in parsed] == [(x0, p0), (axes[x0], p0)]
        seen.update(group)
    assert seen == bridge_consts


def _catalog_codes() -> dict[int, str]:
    import json
    from pathlib import Path

    path = Path(object_orientation.__file__).resolve().parent / "object_catalog.json"
    return {int(c): str(e.get("code") or "") for c, e in json.loads(path.read_text())["objects"].items()}


def test_every_member_is_in_its_own_group_and_spans_are_the_two_transposes():
    for group in object_orientation.groups():
        for const in group:
            assert object_orientation.orientation_siblings(const) == group
            axis, end = object_orientation.group_state(const)
            assert object_orientation.sibling(const, axis, end) == const
        spans = [tile_span(c, render_units.NON_BUILDING_SPAN) for c in group]
        first, other = spans[0], spans[-1]
        assert first != other and other == first[::-1], group
        assert first[0] < first[1], "axis 0 is the member narrower along x"
        assert spans == ([first, first, other, other] if len(group) == 4 else [first, other])


# Where each D4 element sends an `a` top piece (605) and a `b` top piece (608):
# measured from the corpus's bridge runs and typed here rather than derived.
_TOP_IMAGES = {
    "id": (A_TOP, B_TOP),
    "r": (B_BOTTOM, A_TOP),
    "r2": (A_BOTTOM, B_BOTTOM),
    "r3": (B_TOP, A_BOTTOM),
    "mx": (A_BOTTOM, B_TOP),
    "my": (A_TOP, B_BOTTOM),
    "d": (B_BOTTOM, A_BOTTOM),
    "a": (B_TOP, A_TOP),
}


@pytest.mark.parametrize("element", D4)
def test_a_bridge_top_goes_where_the_measured_table_says(element):
    assert (reorient_object_const(A_TOP, element), reorient_object_const(B_TOP, element)) == _TOP_IMAGES[element]
    middle = reorient_object_const(A_MIDDLE, element)
    assert middle == (B_MIDDLE if swaps_axes(element) else A_MIDDLE)


def test_the_image_of_a_composition_is_the_image_of_the_image():
    for group in object_orientation.groups():
        for const, g, h in itertools.product(group, D4, D4):
            assert reorient_object_const(const, compose(g, h)) == reorient_object_const(
                reorient_object_const(const, h), g
            ), (const, g, h)


_ANGULAR = [AngularElement(sixths, reflect) for sixths in range(6) for reflect in (False, True)]


def test_every_angular_element_resolves_every_four_group_member_without_a_tie():
    for group in object_orientation.groups():
        for const, element in itertools.product(group, _ANGULAR):
            image = reorient_object_const(const, element)
            assert image in group
    axis, _end = object_orientation.group_state(A_TOP)
    assert object_orientation.group_state(reorient_object_const(A_TOP, AngularElement(1, False)))[0] == 1 - axis


@pytest.mark.parametrize("const", [2421, 1840, ARCHER])
def test_a_const_with_no_siblings_is_refused_loudly(const):
    with pytest.raises(ValueError):
        reorient_object_const(const, "r")
    assert image_const(const, "r") == const


def test_image_const_routes_gates_objects_and_everything_else():
    from descape.mirror_tools import reorient_gate_const

    assert image_const(64, "d") == reorient_gate_const(64, "d")
    assert image_const(2423, "r") == 2424
    assert image_const(1397, "d") == 1396
    assert image_const(2559, "a") == 2561
    assert image_const(2559, "mx") == 2559


def _bridge_runs(units) -> list[list[tuple[float, int]]]:
    """Classic, Asian and European bridge pieces of one family and axis on one
    line, stepping 2 along it: each run as (projection on TOP_WARD, const)."""
    lines: dict[tuple[int, int, float], list[tuple[float, int]]] = {}
    codes = _catalog_codes()
    for unit in units:
        match = _BRIDGE_CODE_RE.match(codes.get(unit.unit_const, ""))
        if match is None or not object_orientation.has_siblings(unit.unit_const):
            continue  # a square wooden bridge reuses the codes
        axis, _end = object_orientation.group_state(unit.unit_const)
        tx, ty = object_orientation.TOP_WARD[axis]
        across = unit.y if axis == 0 else unit.x
        family = "ABCDEF".index(match.group(1)) // 2
        lines.setdefault((family, axis, round(across, 3)), []).append((unit.x * tx + unit.y * ty, unit.unit_const))
    runs = []
    for pieces in lines.values():
        pieces.sort()
        run = [pieces[0]]
        for piece in pieces[1:]:
            if abs(piece[0] - run[-1][0] - 2.0) < 1e-3:
                run.append(piece)
            else:
                runs.append(run)
                run = [piece]
        runs.append(run)
    return runs


@pytest.mark.corpus
def test_every_corpus_bridge_run_has_its_top_piece_at_the_top_ward_end(corpus_files):
    """Pins TOP_WARD: in every run, a top piece sits only at the end TOP_WARD
    points to and a bottom piece only at the other."""
    from descape import scenario_io

    checked = 0
    wrong = []
    for path in corpus_files:
        loaded = scenario_io.load_map_and_units(path)
        units = [u for player in loaded.unit_manager.units for u in player]
        for run in _bridge_runs(units):
            if len(run) < 2:
                continue
            for index, (_p, const) in enumerate(run):
                _axis, end = object_orientation.group_state(const)
                if (end == 0 and index != len(run) - 1) or (end == 1 and index != 0):
                    wrong.append((path.name, run))
            ends = {object_orientation.group_state(c)[1] for _p, c in run}
            checked += {0, 1} <= ends
    assert checked > 0, "no corpus bridge run with both ends, so this measured nothing"
    assert not wrong
