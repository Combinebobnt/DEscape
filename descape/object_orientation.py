"""Which unit_consts are one non-square object's orientations, and which of
them are the two ends of a directional run (TASK-329).

A bridge, the harbour chain, the longhouse and the two blocker pairs have a
footprint that is not square and no rotation that turns it: each orientation
is its own const with the transposed span (a classic bridge's `a` pieces are
(2, 3), its `b` pieces (3, 2)). So turning one, like turning a gate
(gate_orientation.py), means taking a sibling const. A classic, Asian or
European bridge is also directional along its run: a "top" piece, a "bottom"
piece and the pieces between, so a turn that reverses the run swaps top and
bottom (and broken top and broken bottom) even when the span stays.

The groups are derived from the committed tables, not hand-kept. A
candidate is a const with a non-square span that is neither a gate nor a
cliff. Its key is its unit_graphic_map.json `file_name` minus `_x1`, split
on `_`, or, with no graphic, its catalog `code` split on spaces. Two
candidates of one `class` whose spans are each other's transpose and whose
keys differ in exactly one token are an orientation pair; two of one class
and span whose keys differ only by `top` against `bottom` are an end pair,
linked only between consts that both have an orientation pair. A group is
four consts (two orientations times two ends) or two (orientations only),
and one that does not close cleanly, or a const that would pair twice, is
dropped rather than half-built, as gate_orientation drops a half-filled gate.
Measured 2026-10-28: AoE2 6 four-groups and 10 two-groups over 44 consts,
Return of Rome 6 and 7 over 38 (pinned by tests/test_object_orientation.py).

Left out, so they keep their const and are reported as kept: the mole (no
transposed partner, so no measured run axis) and, by one named family rule,
the twelve DE bridge pieces (`s_bridge_piece_*`). A DE piece's two `end`
frames are its two ends, so its run sense lives in its rotation field, and
re-orienting it would need that frame remapped: a VARIANT write on a new
unit, which no write rule allows yet. The longhouse's three frames are roof
styles that match on A and B, so its index stays verbatim.

Which end is "top" is the one measured fact the tables cannot give. In the
corpus an `a` bridge runs along x with its top piece at high x and a `b`
bridge along y with its top at low y: both tops at the end nearer the screen
top. Asian and European bridges were checked by their rendered frames only.
TOP_WARD holds that fact per axis, pinned by a corpus test.

Reads the committed JSON tables directly (stdlib only), like gate_orientation.
"""

from __future__ import annotations

import json
from pathlib import Path

from descape import game_dataset, gate_orientation

_CATALOG_PATH = Path(__file__).resolve().parent / "object_catalog.json"
_GRAPHIC_MAP_PATH = Path(__file__).resolve().parent / "unit_graphic_map.json"
_RENDER_DATA_PATH = Path(__file__).resolve().parent / "unit_render_data.json"

# unit_render_data.json's span tables; "object_spans" is the cliffs', which are left out.
_SPAN_TABLES = ("buildings", "blocker_spans", "fish_spans", "doodad_spans", "static_spans")
# The DE bridge pieces, whose run sense is a rotation frame rather than a const (module docstring).
_EXCLUDED_FAMILY_PREFIX = "s_bridge_piece_"
_TOP, _BOTTOM = "top", "bottom"

# The direction from a piece toward its run's "top" end, per axis. Axis 0 is the
# orientation whose span is narrower along x ((2, 3) for a bridge), axis 1 the other.
TOP_WARD = ((1, 0), (0, -1))


def _candidates() -> dict[int, tuple[object, tuple[int, int], tuple[str, ...]]]:
    """const -> (class, span, key) for every non-square, non-gate, non-cliff const."""
    objects = json.loads(game_dataset.table_path(_CATALOG_PATH).read_text())["objects"]
    graphics = json.loads(game_dataset.table_path(_GRAPHIC_MAP_PATH).read_text())["graphics"]
    data = json.loads(game_dataset.table_path(_RENDER_DATA_PATH).read_text())
    spans: dict[int, tuple[int, int]] = {}
    for table in _SPAN_TABLES:
        for const, (span_x, span_y) in data.get(table, {}).items():
            spans.setdefault(int(const), (span_x, span_y))
    found = {}
    for const, span in spans.items():
        if span[0] == span[1] or gate_orientation.is_gate(const):
            continue
        graphic = graphics.get(str(const))
        file_name = graphic.get("file_name") if isinstance(graphic, dict) else None
        if file_name:
            if file_name.startswith(_EXCLUDED_FAMILY_PREFIX):
                continue
            key = tuple(file_name.removesuffix("_x1").split("_"))
        else:
            key = tuple(str((objects.get(str(const)) or {}).get("code") or "").split())
        if key:
            found[const] = ((objects.get(str(const)) or {}).get("class"), span, key)
    return found


def _one_token_apart(a: tuple[str, ...], b: tuple[str, ...]) -> tuple[str, str] | None:
    if len(a) != len(b):
        return None
    diffs = [(x, y) for x, y in zip(a, b, strict=True) if x != y]
    return diffs[0] if len(diffs) == 1 else None


def _unique_partners(pairs: list[tuple[int, int]]) -> dict[int, int]:
    """Each const's partner, keeping only consts that pair exactly once."""
    partners: dict[int, list[int]] = {}
    for a, b in pairs:
        partners.setdefault(a, []).append(b)
        partners.setdefault(b, []).append(a)
    return {c: found[0] for c, found in partners.items() if len(found) == 1 and len(partners[found[0]]) == 1}


@game_dataset.cache(maxsize=1)
def groups() -> tuple[tuple[int, ...], ...]:
    """Every orientation group: (axis-0 top, axis-0 bottom, axis-1 top,
    axis-1 bottom) for a four-group, (axis 0, axis 1) for a two-group."""
    found = _candidates()
    consts = sorted(found)
    orient_pairs, end_pairs = [], []
    for i, a in enumerate(consts):
        class_a, span_a, key_a = found[a]
        for b in consts[i + 1:]:
            class_b, span_b, key_b = found[b]
            if class_a != class_b:
                continue
            diff = _one_token_apart(key_a, key_b)
            if diff is None:
                continue
            if span_b == span_a[::-1]:
                orient_pairs.append((a, b))
            elif span_b == span_a and set(diff) == {_TOP, _BOTTOM}:
                end_pairs.append((a, b) if diff[0] == _TOP else (b, a))
    orient = _unique_partners(orient_pairs)
    ends = _unique_partners([(t, b) for t, b in end_pairs if t in orient and b in orient])
    tops = {t for t, _b in end_pairs}
    result = []
    for const in sorted(orient):
        if found[const][1][0] >= found[const][1][1]:
            continue  # each group once, from its axis-0 member
        other = orient[const]
        if const not in ends and other not in ends:
            result.append((const, other))
        elif const in ends and const in tops and other in ends:
            bottom = ends[const]
            if ends[other] == orient.get(bottom) and other in tops:
                result.append((const, bottom, other, ends[other]))
    return tuple(result)


@game_dataset.cache(maxsize=1)
def _group_by_const() -> dict[int, tuple[int, ...]]:
    return {const: group for group in groups() for const in group}


def orientation_siblings(unit_const: int) -> tuple[int, ...] | None:
    """`unit_const`'s whole group (itself included), or None if it has none."""
    return _group_by_const().get(unit_const)


def has_siblings(unit_const: int) -> bool:
    return unit_const in _group_by_const()


def group_state(unit_const: int) -> tuple[int, int | None]:
    """(axis, end) of a group member: axis 0 or 1 (TOP_WARD's index), end 0
    top or 1 bottom, or None in a two-group. Raises for a const with no group."""
    group = orientation_siblings(unit_const)
    if group is None:
        raise ValueError(f"unit_const {unit_const} has no orientation siblings")
    index = group.index(unit_const)
    return (index // 2, index % 2) if len(group) == 4 else (index, None)


def sibling(unit_const: int, axis: int, end: int | None) -> int:
    """The member of `unit_const`'s group at (axis, end), group_state()'s inverse."""
    group = orientation_siblings(unit_const)
    if group is None:
        raise ValueError(f"unit_const {unit_const} has no orientation siblings")
    if len(group) == 4:
        if end is None:
            raise ValueError(f"unit_const {unit_const}'s group has two ends: an end is needed")
        return group[2 * axis + end]
    return group[axis]
