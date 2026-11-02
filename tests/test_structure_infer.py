"""tools/structure_infer.py: the localize-and-repair search on small synthetic
structures built in the test, and one end-to-end run on the 1.59 fixture."""

from __future__ import annotations

import itertools
import json
import struct

import numpy as np
import pytest

from descape import structure_walk as sw
from tools import structure_infer as si

import conftest

FIXTURES = conftest.ROOT / "tests" / "fixtures"
TERRAIN_IDS = {int(k) for k in json.loads((conftest.ROOT / "descape" / "terrain_classes.json").read_text(encoding="utf-8"))["classes"]}

_INTS = {"u8": "B", "s8": "b", "u16": "H", "s16": "h", "u32": "I", "s32": "i"}
_LIMITS = {"B": 250, "b": 120, "H": 60000, "h": 30000, "I": 4_000_000, "i": 2_000_000}


def _repeat_by(target: str) -> dict:
    return {"on_refresh": {"action": "SET_REPEAT", "target": f"self:{target}"}, "on_construct": {"action": "REFRESH_SELF"}}


def base_structure() -> dict:
    """One section: a count, that many Item structs, a trailing string."""
    item = {
        "retrievers": {
            "a": {"type": "s32"},
            "b": {"type": "f32"},
            "c": {"type": "s32"},
            "d": {"type": "u16"},
            "e": {"type": "s32"},
            "label": {"type": "str16"},
        }
    }
    return {
        "S": {
            "retrievers": {
                "n_items": {"type": "u32"},
                "items": {"type": "struct:Item", "dependencies": _repeat_by("n_items")},
                "tail": {"type": "str16"},
            },
            "structs": {"Item": item},
        }
    }


def encode(structure: dict, counts: dict[str, int], seed: int = 0) -> bytes:
    """Bytes for `structure` with distinctive values (never 0, 1 or -1);
    each `n_*` field takes its count from `counts`."""
    tick = itertools.count(seed + 2)

    def one(t: str, name: str) -> bytes:
        k = next(tick)
        if name.startswith("n_"):
            return struct.pack("<" + _INTS[t], counts[name])
        if t in _INTS:
            fmt = _INTS[t]
            low = 70_000 if fmt in "iI" else 2  # a 4-byte value's high half nonzero, so it can't read as u16 + empty str16
            return struct.pack("<" + fmt, low + k % _LIMITS[fmt])
        if t == "f32":
            return struct.pack("<f", 1.5 + k)
        if t in ("str16", "str32"):
            text = f"v{k}".encode()
            return struct.pack("<H" if t == "str16" else "<I", len(text)) + text
        if t.isdigit():
            return bytes((k + j) % 200 + 2 for j in range(int(t)))
        raise ValueError(t)

    def block(node: dict, structs: dict) -> bytes:
        out = bytearray()
        for name, d in node["retrievers"].items():
            deps = d.get("dependencies") or {}
            target = (deps.get("on_refresh") or {}).get("target")
            n = counts[target.removeprefix("self:")] if target else d.get("repeat", 1)
            for _ in range(n):
                t = d["type"]
                if t.startswith("struct:"):
                    out += block(structs[t[7:]], structs)
                else:
                    out += one(t, name)
        return bytes(out)

    return b"".join(block(sec, sec.get("structs", {})) for sec in structure.values())


def reference_for(truth: dict, data: bytes) -> si.Reference:
    walk = sw.walk(truth, data, 0)
    return si.Reference({p: c.value_list() for p, c in walk.columns.items()})


def search(base: dict, data: bytes, ref: si.Reference | None, budget: si.Budget = si.DEFAULT_BUDGET) -> si.SegmentOutcome:
    inp = si.SegmentInput("synthetic", data, 0, si._end_anchor(len(data)), si.FilterContext(), ref)
    return si.SegmentSearch("S", ["S"], [inp], si.Names("9.9"), budget, [0]).run(base)


def same_layout(a: dict, b: dict) -> bool:
    def sig(s: dict) -> tuple:
        return tuple((sec, si.layout_signature(s[sec]["retrievers"])) for sec in s) + tuple(
            (sec, name, si.layout_signature(node["retrievers"])) for sec in s for name, node in s[sec].get("structs", {}).items()
        )

    return sig(a) == sig(b)


ITEM = ("S", "Item")
EDITS = [
    si.Edit(ITEM, 2, 0, [("new", {"type": "s8"})], "insert s8"),
    si.Edit(ITEM, 2, 0, [("new", {"type": "s16"})], "insert s16"),
    si.Edit(ITEM, 3, 0, [("new", {"type": "s32"})], "insert s32"),
    si.Edit(ITEM, 2, 0, [("new", {"type": "8"})], "insert 8"),
    si.Edit(ITEM, 2, 0, [("new", {"type": "3"})], "insert raw 3"),
    si.Edit(ITEM, 2, 0, [("new", {"type": "str16"})], "insert str16"),
    si.Edit(ITEM, 2, 0, [("new", {"type": "str32"})], "insert str32"),
    si.Edit(ITEM, 2, 0, [("new", {"type": "u16"}), ("new2", {"type": "str16"})], "insert u16 + str16"),
    si.Edit(ITEM, 3, 1, [], "delete 1"),
    si.Edit(ITEM, 2, 1, [("c", {"type": "s16"})], "retype s32 -> s16"),
    si.Edit(("S",), 2, 0, [("new", {"type": "s32"})], "insert s32 in the section"),
]


@pytest.mark.parametrize("edit", EDITS, ids=[e.kind for e in EDITS])
def test_each_edit_kind_is_found_from_a_reference(edit: si.Edit) -> None:
    base = base_structure()
    truth = si.apply_edit(base, edit)
    data = encode(truth, {"n_items": 6})
    out = search(base, data, reference_for(truth, data))
    assert out.refused is None
    assert len(out.edits) == 1
    choice = out.edits[0][1]
    assert not choice.ambiguous, choice.note
    assert same_layout(out.structure, truth)


def test_an_s32_anywhere_in_a_run_of_s32s_is_one_layout() -> None:
    block = {"retrievers": {"a": {"type": "s32"}, "b": {"type": "s32"}, "c": {"type": "u8"}}}
    s = {"S": block}
    at0 = si.apply_edit(s, si.Edit(("S",), 0, 0, [("new", {"type": "s32"})], "insert s32"))
    at1 = si.apply_edit(s, si.Edit(("S",), 1, 0, [("new", {"type": "s32"})], "insert s32"))
    at3 = si.apply_edit(s, si.Edit(("S",), 3, 0, [("new", {"type": "s32"})], "insert s32"))
    assert si.layout_signature(at0["S"]["retrievers"]) == si.layout_signature(at1["S"]["retrievers"])
    assert si.layout_signature(at0["S"]["retrievers"]) != si.layout_signature(at3["S"]["retrievers"])


def test_without_a_reference_an_insert_among_identical_layouts_is_ambiguous_with_the_truth_kept_or_listed() -> None:
    base = base_structure()
    edit = si.Edit(ITEM, 3, 0, [("new", {"type": "s32"})], "insert s32")
    truth = si.apply_edit(base, edit)
    data = encode(truth, {"n_items": 6})
    out = search(base, data, None)
    assert out.refused is None
    ((kept, choice, _),) = out.edits
    assert choice.ambiguous
    assert any(same_layout(si.apply_edit(base, e), truth) for e in [kept, *choice.rivals])


def test_two_edits_in_one_segment_are_found_one_at_a_time() -> None:
    base = base_structure()
    truth = si.apply_edit(base, si.Edit(ITEM, 2, 0, [("x", {"type": "s8"})], "insert s8"))
    truth = si.apply_edit(truth, si.Edit(("S",), 2, 0, [("y", {"type": "s16"})], "insert s16"))
    data = encode(truth, {"n_items": 6})
    out = search(base, data, reference_for(truth, data))
    assert out.refused is None
    assert len(out.edits) == 2
    assert same_layout(out.structure, truth)


def test_the_per_block_budget_refuses() -> None:
    base = base_structure()
    truth = si.apply_edit(base, si.Edit(ITEM, 1, 0, [("x", {"type": "s8"})], "insert s8"))
    truth = si.apply_edit(truth, si.Edit(ITEM, 5, 0, [("y", {"type": "s16"})], "insert s16"))
    data = encode(truth, {"n_items": 6})
    out = search(base, data, reference_for(truth, data), si.Budget(edits_per_block=1))
    assert out.refused is not None and "edit budget of 1 for S/Item" in out.refused


def test_a_dependent_repeat_is_refused_not_mis_repaired() -> None:
    base = base_structure()
    item = base["S"]["structs"]["Item"]["retrievers"]
    items = list(item.items())
    items[2:2] = [("n_sub", {"type": "u8"}), ("sub", {"type": "u16", "dependencies": _repeat_by("n_sub")})]
    truth = json.loads(json.dumps(base))
    truth["S"]["structs"]["Item"]["retrievers"] = dict(items)
    # Varying counts per item: no fixed-size insert fits every instance.
    chunks = [encode({"S": {"retrievers": {"n_items": {"type": "u32"}}}}, {"n_items": 5})]
    for i, n in enumerate((0, 3, 1, 4, 2)):
        chunks.append(encode({"I": truth["S"]["structs"]["Item"]}, {"n_sub": n}, seed=100 * i))
    chunks.append(encode({"T": {"retrievers": {"tail": {"type": "str16"}}}}, {}, seed=900))
    data = b"".join(chunks)
    out = search(base, data, reference_for(truth, data))
    assert out.refused is not None
    assert not same_layout(out.structure, truth)


def _never_wrong(out: si.SegmentOutcome, base: dict, truth: dict) -> None:
    """The plan's WRONG: a result with no ambiguous edit whose layout isn't the truth."""
    if out.refused is None and not any(c.ambiguous for _, c, _ in out.edits):
        assert same_layout(out.structure, truth), [e.describe() for e, _, _ in out.edits]


def test_two_s32_inserts_around_a_field_never_come_out_wrong() -> None:
    # A misframing whose garbage reads as small values (0, 1) must still count as a break.
    base = base_structure()
    truth = si.apply_edit(base, si.Edit(ITEM, 1, 0, [("x", {"type": "s32"})], "insert s32"))
    truth = si.apply_edit(truth, si.Edit(ITEM, 5, 0, [("y", {"type": "s32"})], "insert s32"))
    data = encode(truth, {"n_items": 6})
    _never_wrong(search(base, data, reference_for(truth, data)), base, truth)


def test_one_differing_value_decides_nothing() -> None:
    # The new field copies c in every item but the first, where the re-save rewrote c:
    # inserting before c or after it differs in that one value only.
    base = base_structure()
    truth = si.apply_edit(base, si.Edit(ITEM, 3, 0, [("n", {"type": "s32"})], "insert s32"))
    out = bytearray(struct.pack("<I", 200))
    for i in range(200):
        label = f"L{i}".encode()
        out += struct.pack("<ifiiHi", 70000 + 3 * i, 2.5 + i, 90009 if i == 0 else 90007, 90007, 300 + i, 80000 + i)
        out += struct.pack("<H", len(label)) + label
    data = bytes(out + struct.pack("<H", 4) + b"tail")
    walk = sw.walk(truth, data, 0)
    values = {p: c.value_list() for p, c in walk.columns.items()}
    values[("S", "items", "c")][0] = 90007  # the source's value before the re-save
    result = search(base, data, si.Reference(values))
    _never_wrong(result, base, truth)
    assert result.refused is None and result.edits[0][1].ambiguous


def test_a_reference_from_another_file_refuses_without_crashing() -> None:
    base = base_structure()
    data = encode(base, {"n_items": 6})
    other = encode(base, {"n_items": 7}, seed=5000)
    assert search(base, data, reference_for(base, other)).refused is not None


def test_pair_distance_counts_an_empty_column_as_missing() -> None:
    ref = si.Reference({("S", "x"): [5, 6, 7]})
    full = sw.walk({"S": {"retrievers": {"x": {"type": "s32", "repeat": 3}}}}, struct.pack("<3i", 5, 6, 7), 0)
    empty = sw.Walk(columns={("S", "x"): sw.Column()})
    assert si.pair_distance(full, empty, ref, ["S"]) == si.pair_distance(full, sw.Walk(), ref, ["S"])


def _frozen_pair(**kwargs) -> si.InferResult:
    return si.infer(
        "1.58",
        [conftest.TEST_MAP_RESAVE_PATH],
        reference=(conftest.TEST_MAP_RESAVE_SOURCE_PATH, conftest.TEST_MAP_RESAVE_PATH),
        terrain_ids=TERRAIN_IDS,
        library_check=None,
        **kwargs,
    )


@pytest.mark.test_map
def test_the_frozen_resave_pair_keeps_the_truth_and_leaves_only_default_positions_open() -> None:
    result = _frozen_pair()
    assert result.status == si.STATUS_AMBIGUOUS, result.report_lines
    by_block = {r.edit.block: r for r in result.edits}
    unit = by_block[("Units", "PlayerUnitsStruct", "UnitStruct")]
    condition = by_block[("Triggers", "TriggerStruct", "ConditionStruct")]
    assert len(result.edits) == 2
    assert unit.edit.index == 9 and not unit.ambiguous  # capture_flag, decided by the reference despite the game's rewrites
    assert condition.edit.index == 35 and condition.ambiguous  # allow_in_fog
    # Open: positions that read every source condition identically (33, 34) or differ from the
    # truth in fewer than PAIR_MIN values (31, 32); the reference rules out 18..30.
    assert sorted(e.index for e in [condition.edit, *condition.rivals]) == [31, 32, 33, 34, 35]


@pytest.mark.test_map
def test_a_false_reference_break_refuses_rather_than_accepting_junk(monkeypatch: pytest.MonkeyPatch) -> None:
    # With realignment off, the game's rewrite of player_data_3[0] reads as a layout break that no
    # edit clears. Before the guard, the "gets furthest" fallback stacked junk inserts on it.
    monkeypatch.setattr(si, "_realigns", lambda events: np.zeros(len(events.matched), dtype=bool))
    result = _frozen_pair()
    assert result.status == si.STATUS_REFUSED
    assert result.edits == []
    assert "player_data_3.editor_camera_x" in result.refusal


def _events(matched: list[bool], distinctive: list[bool] | None = None) -> si.Events:
    n = len(matched)
    d = distinctive if distinctive is not None else [True] * n
    z = np.zeros(n, dtype=np.int64)
    return si.Events(np.arange(n, dtype=np.int64) * 4, z + 4, np.asarray(matched), np.asarray(d), [("S", "x")], z)


def test_a_rewrite_cluster_the_stream_realigns_after_is_not_a_break() -> None:
    # 9 of 16 rewritten, no 3 in a row: as dense as the game's rewrite of player_data_3[0]
    rewrite = [False, False, True, False, False, True, False, False, True, False, False, True, False, True, True, True]
    after = [True] * 20
    shift = [False] * 200
    events = _events(rewrite + after + shift)
    assert si.first_break(events) == len(rewrite) + len(after)
    assert si.first_break(_events(rewrite + shift)) == 0


def test_a_shift_a_string_resyncs_is_still_a_break() -> None:
    # Solid garbage for 9 values, then the stream happens to match again: a shift, not a rewrite.
    matched = [True] * 4 + [False] * 9 + [True] * 40 + [False] * 200
    assert si.first_break(_events(matched)) == 4
    # Small garbage values (non-distinctive) still count toward the solid run.
    distinctive = [True] * 4 + [False, True, False, False, True, False, False, True, False] + [True] * 240
    assert si.first_break(_events(matched, distinctive)) == 4


def test_infer_on_the_159_fixture_from_158() -> None:
    sample = FIXTURES / "v159_units_triggers.aoe2scenario"
    truth = si.load_structure("1.59")
    walked = sw.walk_file(truth, sample.read_bytes())
    values = {p: c.value_list() for w in (walked.header, walked.body) for p, c in w.columns.items()}
    referenced = si.infer("1.58", [sample], reference_values=values, terrain_ids=TERRAIN_IDS, library_check=None)
    # Units is decided; the fixture's one non-default condition is below PAIR_MIN, so Conditions stays open.
    assert referenced.status == si.STATUS_AMBIGUOUS, referenced.report_lines
    assert [r.ambiguous for r in referenced.edits] == [False, True]
    blind = si.infer("1.58", [sample], terrain_ids=TERRAIN_IDS, library_check=None)
    assert blind.status == si.STATUS_AMBIGUOUS

    unit = ("Units", "PlayerUnitsStruct", "UnitStruct")
    condition = ("Triggers", "TriggerStruct", "ConditionStruct")
    for result in (referenced, blind):
        kept = {r.edit.block: r.edit.index for r in result.edits}
        assert kept == {unit: 9, condition: 35}  # capture_flag and allow_in_fog's true positions

    s = referenced.structure
    capture = si.block_dict(s, unit)["retrievers"]["inferred_v1_59_0"]
    fog = si.block_dict(s, condition)["retrievers"]["inferred_v1_59_1"]
    assert capture["type"] == "s8" and fog["type"] == "s32"
    assert capture["comment"].startswith("Inferred for 1.59 from 1.58")
    assert capture["default"] == si._most_common(values[("Units", "players_units", "units", "capture_flag")])
    assert si.block_dict(s, condition)["retrievers"]["static_value_33"]["default"] == 34
    assert s["FileHeader"]["retrievers"]["version"]["default"] == values[("FileHeader", "version")][0]
    assert s["Triggers"]["retrievers"]["trigger_version"]["default"] == values[("Triggers", "trigger_version")][0]
