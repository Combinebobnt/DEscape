"""object_orientation's derived sibling groups (TASK-329, TASK-706, TASK-705),
mirror_tools.reorient_object_const(), the const a bridge, chain, longhouse or
blocker takes under a turn or mirror, and mirror_tools.reorient_piece_frame(),
a DE bridge piece's frame.

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

from descape import bridge_piece_frames, game_dataset, gate_orientation, object_orientation, render_units, unit_sprites
from descape.mirror_tools import (
    TRANSFORMS,
    AngularElement,
    _linear_part,
    compose,
    image_const,
    reorient_object_const,
    reorient_piece_frame,
    swaps_axes,
)
from descape.terrain_palette import tile_span

D4 = list(TRANSFORMS)
MOLES = {2421, 2422, 2521, 2522}
DE_BRIDGE_PIECES = {1840, 1841, 1844, 1845, 1993, 1994, 1996, 1997, 1999, 2000, 2039, 2041}
A_TOP, A_MIDDLE, A_BOTTOM, B_TOP, B_MIDDLE, B_BOTTOM = 605, 606, 607, 608, 609, 610
ARCHER = 4
WOODEN_1X1 = set(range(904, 910)) | set(range(1309, 1315))
WOODEN_2X2 = {911, 912, 913, 1316, 1317, 1318}
IMPALED_CORPSE = 910
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


def test_aoe2_has_eight_four_groups_and_eighteen_pairs_and_leaves_out_only_the_mole():
    assert _counts() == (8, 18, 68)
    grouped = {c for g in object_orientation.groups() for c in g}
    candidates = _non_square_candidates()
    assert len(candidates) == 60
    assert candidates - grouped == MOLES
    square = {c for c in grouped if len(set(tile_span(c, render_units.NON_BUILDING_SPAN))) == 1}
    assert square == WOODEN_1X1


def test_only_the_2x2_wooden_bridges_have_an_unturned_axis():
    found = {c for c in object_orientation._candidates() if object_orientation.has_unturned_axis(c)}
    assert found == WOODEN_2X2


def test_the_de_bridge_pieces_pair_into_six_two_groups_b_on_axis_0():
    de_groups = [g for g in object_orientation.groups() if set(g) & DE_BRIDGE_PIECES]
    assert len(de_groups) == 6 and {c for g in de_groups for c in g} == DE_BRIDGE_PIECES
    for group in de_groups:
        assert len(group) == 2 and tile_span(group[0], render_units.NON_BUILDING_SPAN) == (1, 2)


def test_a_de_bridge_piece_without_a_matching_frame_row_is_not_a_candidate(monkeypatch):
    rows = dict(bridge_piece_frames._TABLE["aoe2"])
    del rows[1840]
    monkeypatch.setitem(bridge_piece_frames._TABLE, "aoe2", rows)
    object_orientation.groups.cache_clear()
    object_orientation._group_by_const.cache_clear()
    try:
        assert 1840 not in object_orientation._candidates()
        assert 1841 in object_orientation._candidates() and not object_orientation.has_siblings(1841)
        # Its partner alone keeps const and frame under every element, and counts as kept under a swap.
        from descape import region_transform

        for element in D4:
            assert image_const(1841, element) == 1841
            assert _image_rotation_of(1841, 0.0, element) == 0.0
            assert region_transform._keeps_orientation(1841, element) == swaps_axes(element)
    finally:
        object_orientation.groups.cache_clear()
        object_orientation._group_by_const.cache_clear()


def _image_rotation_of(const: int, rotation: float, element: str) -> float:
    from types import SimpleNamespace

    from descape.mirror_tools import _image_rotation

    return _image_rotation(SimpleNamespace(unit_const=const, rotation=rotation), element, False)


def test_return_of_rome_has_eight_four_groups_and_fifteen_pairs():
    with game_dataset.activated(game_dataset.ROR):
        assert _counts() == (8, 15, 62)
        assert not any(object_orientation.has_siblings(c) for c in (2423, 2424, 2429, 2430, 2559, 2561))
    assert object_orientation.has_siblings(2559)


def test_the_bridge_codes_agree_with_every_bridge_group():
    """BG<X><P>: X pairs A/B, C/D, E/F (the two axes of one family) with the
    same P; P pairs A/C top/bottom and E/F broken top/bottom, B and D (middle,
    cracked) have no end."""
    axes = {"A": "B", "C": "D", "E": "F"}
    ends = {"A": "C", "E": "F"}
    codes = _catalog_codes()
    candidates = _non_square_candidates() | WOODEN_1X1
    bridge_consts = {c for c, code in codes.items() if _BRIDGE_CODE_RE.match(code) and c in candidates}
    assert len(bridge_consts) == 48
    ungrouped = {c for c, code in codes.items() if _BRIDGE_CODE_RE.match(code)} - bridge_consts
    assert ungrouped == WOODEN_2X2
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


def test_every_member_is_in_its_own_group_and_spans_are_the_two_transposes_or_one_square():
    for group in object_orientation.groups():
        for const in group:
            assert object_orientation.orientation_siblings(const) == group
            axis, end = object_orientation.group_state(const)
            assert object_orientation.sibling(const, axis, end) == const
        spans = [tile_span(c, render_units.NON_BUILDING_SPAN) for c in group]
        first, other = spans[0], spans[-1]
        if first[0] == first[1]:
            assert set(spans) == {first} and set(group) <= WOODEN_1X1, group
            continue
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


# The wooden pieces (904-909) take the same table, substituted for 605-610.
_WOODEN = dict(zip((A_TOP, A_MIDDLE, A_BOTTOM, B_TOP, B_MIDDLE, B_BOTTOM), range(904, 910), strict=True))


@pytest.mark.parametrize("element", D4)
@pytest.mark.parametrize("family", ["classic", "wooden"])
def test_a_bridge_top_goes_where_the_measured_table_says(element, family):
    to = (lambda c: c) if family == "classic" else _WOODEN.__getitem__
    tops = (reorient_object_const(to(A_TOP), element), reorient_object_const(to(B_TOP), element))
    assert tops == tuple(to(c) for c in _TOP_IMAGES[element])
    middle = reorient_object_const(to(A_MIDDLE), element)
    assert middle == to(B_MIDDLE if swaps_axes(element) else A_MIDDLE)


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


@pytest.mark.parametrize("const", [2421, ARCHER, 911, IMPALED_CORPSE])
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


DE_ENDS = (1840, 1841, 1993, 1994, 2039, 2041)


def _inward(const: int, frame: int) -> tuple[int, int]:
    kind, vector = bridge_piece_frames.frames(const)[frame]
    assert kind == bridge_piece_frames.END
    return vector


def test_a_de_end_frame_turns_its_inward_vector_with_the_element():
    for const, element in itertools.product(DE_ENDS, D4):
        new_const = image_const(const, element)
        for frame in range(2):
            image = int(reorient_piece_frame(const, float(frame), element))
            assert _inward(new_const, image) == _linear_part(element)(*_inward(const, frame)), (const, frame, element)


def test_the_frame_image_of_a_composition_is_the_image_of_the_image():
    for group in object_orientation.groups():
        for const in group:
            if not bridge_piece_frames.covers(const):
                continue
            for frame, g, h in itertools.product(range(len(bridge_piece_frames.frames(const))), D4, D4):
                once = int(reorient_piece_frame(const, float(frame), compose(g, h)))
                mid_const, mid = image_const(const, h), int(reorient_piece_frame(const, float(frame), h))
                assert once == int(reorient_piece_frame(mid_const, float(mid), g)), (const, frame, g, h)


# Where each D4 element sends 1840 frame 0 (inward -y), typed from the measured frames rather than derived.
_DE_END_IMAGES = {
    "r": (1841, 0),
    "r2": (1840, 1),
    "r3": (1841, 1),
    "mx": (1840, 0),
    "my": (1840, 1),
    "d": (1841, 1),
    "a": (1841, 0),
}


@pytest.mark.parametrize("element", list(_DE_END_IMAGES))
def test_a_de_end_goes_where_the_typed_table_says(element):
    assert (image_const(1840, element), int(reorient_piece_frame(1840, 0.0, element))) == _DE_END_IMAGES[element]


def test_a_cosmetic_frame_keeps_its_kind_across_an_axis_swap():
    assert image_const(1844, "r") == 1845 and reorient_piece_frame(1844, 0.0, "r") == 1.0
    for const, element in itertools.product(DE_BRIDGE_PIECES - set(DE_ENDS), D4):
        row, new_const = bridge_piece_frames.frames(const), image_const(const, element)
        for frame in range(len(row)):
            image = int(reorient_piece_frame(const, float(frame), element))
            assert bridge_piece_frames.frames(new_const)[image] == row[frame]


def test_every_angular_element_resolves_every_de_end_frame_without_a_tie():
    for const, element, frame in itertools.product(DE_ENDS, _ANGULAR, range(2)):
        assert 0 <= reorient_piece_frame(const, float(frame), element) < 2


def test_an_unchanged_de_frame_on_an_unchanged_const_comes_back_as_stored():
    stored = 1.0000001  # reads as frame 1, kept as stored rather than rewritten as 1.0
    assert reorient_piece_frame(1840, stored, "mx") is stored
    assert reorient_piece_frame(1840, stored, "r2") == 0.0


def _bridge_runs(units) -> list[list[tuple[float, int]]]:
    """Classic, Asian and European bridge pieces of one family and axis on one
    line, stepping 2 along it: each run as (projection on TOP_WARD, const)."""
    lines: dict[tuple[int, int, float], list[tuple[float, int]]] = {}
    codes = _catalog_codes()
    for unit in units:
        match = _BRIDGE_CODE_RE.match(codes.get(unit.unit_const, ""))
        if match is None or unit.unit_const in WOODEN_1X1 or not object_orientation.has_siblings(unit.unit_const):
            continue  # a square wooden bridge reuses the codes; _wooden_runs() takes those
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


def _wooden_runs(units) -> list[list[tuple[float, int]]]:
    """Isolated straight 1x1 wooden runs: an 8-connected cluster of wooden
    pieces that is one line of one axis and one `hidden` block, stepping 1.
    The wide decorative clusters (rows of end pieces, mixed axes) have no
    single run direction and are skipped."""
    stacked: dict[tuple[int, int], list] = {}
    for u in units:
        if u.unit_const in WOODEN_1X1:
            stacked.setdefault((int(u.x), int(u.y)), []).append(u)
    tiles = {tile: found[0] for tile, found in stacked.items()}
    seen: set[tuple[int, int]] = set()
    runs = []
    for start in tiles:
        if start in seen:
            continue
        cluster, stack = [], [start]
        seen.add(start)
        while stack:
            tx, ty = stack.pop()
            cluster.append(tiles[(tx, ty)])
            for dx, dy in itertools.product((-1, 0, 1), repeat=2):
                near = (tx + dx, ty + dy)
                if near in tiles and near not in seen:
                    seen.add(near)
                    stack.append(near)
        axes = {object_orientation.group_state(u.unit_const)[0] for u in cluster}
        if any(len(stacked[(int(u.x), int(u.y))]) > 1 for u in cluster):
            continue  # a stacked tile is not a plain run
        if len(cluster) < 2 or len(axes) != 1 or len({u.unit_const < 1000 for u in cluster}) != 1:
            continue
        axis = axes.pop()
        tx, ty = object_orientation.TOP_WARD[axis]
        if len({round(u.y if axis == 0 else u.x, 3) for u in cluster}) != 1:
            continue
        run = sorted((u.x * tx + u.y * ty, u.unit_const) for u in cluster)
        if all(abs(b[0] - a[0] - 1.0) < 1e-3 for a, b in itertools.pairwise(run)):
            runs.append(run)
    return runs


def _misplaced_ends(run) -> bool:
    for index, (_p, const) in enumerate(run):
        _axis, end = object_orientation.group_state(const)
        if (end == 0 and index != len(run) - 1) or (end == 1 and index != 0):
            return True
    return False


@pytest.mark.corpus
def test_every_corpus_bridge_run_has_its_top_piece_at_the_top_ward_end(corpus_files):
    """Pins TOP_WARD: in every run, classic and isolated wooden alike, a top
    piece sits only at the end TOP_WARD points to and a bottom piece only at
    the other."""
    from descape import scenario_io

    checked = wooden_checked = 0
    wrong = []
    for path in corpus_files:
        loaded = scenario_io.load_map_and_units(path)
        units = [u for player in loaded.unit_manager.units for u in player]
        for run in _bridge_runs(units):
            if len(run) < 2:
                continue
            if _misplaced_ends(run):
                wrong.append((path.name, run))
            ends = {object_orientation.group_state(c)[1] for _p, c in run}
            checked += {0, 1} <= ends
        for run in _wooden_runs(units):
            if _misplaced_ends(run):
                wrong.append((path.name, run))
            ends = {object_orientation.group_state(c)[1] for _p, c in run}
            wooden_checked += {0, 1} <= ends
    assert checked > 0, "no corpus bridge run with both ends, so this measured nothing"
    if any("Dos Pilas" in path.name for path in corpus_files):
        assert wooden_checked >= 2, "Dos Pilas holds 2 both-ended wooden runs"
    assert not wrong
