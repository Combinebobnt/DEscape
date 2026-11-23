"""tools/structure_infer.py: the localize-and-repair search on small synthetic
structures built in the test, and one end-to-end run on the 1.59 fixture."""

from __future__ import annotations

import itertools
import json
import random
import re
import struct
import subprocess
import sys

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


def _run_struct(n_fields: int) -> dict:
    rets = {f"f{i}": {"type": "s32"} for i in range(n_fields)}
    rets["label"] = {"type": "str16"}
    s = base_structure()
    s["S"]["structs"]["Item"] = {"retrievers": rets}
    return s


def _encode_run(structure: dict, values: dict[str, int | None], n_items: int) -> bytes:
    """Each s32 field holds its constant from `values`, or a distinct value per item for None."""
    names = list(structure["S"]["structs"]["Item"]["retrievers"])
    out = bytearray(struct.pack("<I", n_items))
    for i in range(n_items):
        for k, name in enumerate(names):
            if name == "label":
                text = f"L{i}".encode()
                out += struct.pack("<H", len(text)) + text
            else:
                v = values[name]
                out += struct.pack("<i", 70_000 + 1000 * k + i if v is None else v)
    return bytes(out + struct.pack("<H", 4) + b"tail")


def _fuzz_case(seed: int) -> tuple[dict, dict, bytes]:
    """Two s32 inserts (or one) into a struct of mostly constant 4-byte fields: (base, truth, data)."""
    rng = random.Random(seed)
    n = rng.randint(4, 8)
    base = _run_struct(n)
    values = {f"f{i}": rng.choice([0, -1, 1, 2, None]) for i in range(n)}
    truth = base
    for name in ("x", "y")[: rng.choice([1, 2])]:
        index = rng.randint(0, len(truth["S"]["structs"]["Item"]["retrievers"]) - 1)
        truth = si.apply_edit(truth, si.Edit(ITEM, index, 0, [(name, {"type": "s32"})], "insert s32"))
        values[name] = rng.choice([0, -1, 1, 2, None])
    return base, truth, _encode_run(truth, values, si.WHOLE_MIN + 4)


def _fuzz(seeds: range, blind: bool = False) -> tuple[list, int]:
    """(wrong results, refused count). With a reference, wrong is any truth value read differently;
    blind, it is `_never_wrong`'s (an unambiguous wrong layout)."""
    wrong, refused = [], 0
    for seed in seeds:
        base, truth, data = _fuzz_case(seed)
        out = search(base, data, None if blind else reference_for(truth, data))
        if out.refused is not None:
            refused += 1
            continue
        edits = [e.describe() for e, _, _ in out.edits]
        if blind:
            if not any(c.ambiguous for _, c, _ in out.edits) and not same_layout(out.structure, truth):
                wrong.append((seed, edits))
            continue
        got, want = sw.walk(out.structure, data, 0), sw.walk(truth, data, 0)
        if any(p in got.columns and c.value_list() != got.columns[p].value_list() for p, c in want.columns.items()):
            wrong.append((seed, edits))
    return wrong, refused


# Refused counts measured 2026-11-21: a rise means retries stopped recovering what they did.
FUZZ_REFUSED = {(40, False): 8, (40, True): 0, (300, False): 48, (300, True): 0}  # greedy: 15, 0, 94, 0
TRIGGER_RECOVERED = 19  # of _trigger_case's 30 seeds, by a retry (greedy refuses all 30)


def test_two_inserts_in_a_repeated_struct_never_come_out_wrong() -> None:
    # Before whole-path breaks, one 8-byte insert at the first position "fit" (the fields between
    # read each other's bytes in every item, too few values for a stream break) and was reported
    # INFERRED: 72 of 300 seeds.
    wrong, refused = _fuzz(range(40))
    assert not wrong, wrong
    assert refused <= FUZZ_REFUSED[(40, False)]


def test_two_inserts_blind_never_come_out_wrong() -> None:
    wrong, refused = _fuzz(range(40), blind=True)
    assert not wrong, wrong
    assert refused <= FUZZ_REFUSED[(40, True)]


@pytest.mark.slow
@pytest.mark.parametrize("blind", [False, True], ids=["reference", "blind"])
def test_two_inserts_fuzz_300_seeds_never_wrong(blind: bool) -> None:
    wrong, refused = _fuzz(range(300), blind=blind)
    assert not wrong, wrong
    assert refused <= FUZZ_REFUSED[(300, blind)]


def _record(out: si.SegmentOutcome) -> list:
    return [(e.describe(), e.new_names(), c.note, [r.describe() for r in c.rivals], c.ambiguous) for e, c, _ in out.edits]


def test_retrying_changes_nothing_where_the_greedy_search_succeeds() -> None:
    # Retries engage only after a dead end, so every greedy result is kept edit for edit. (Blind, this
    # fuzz never dead-ends, so only the referenced runs can tell.)
    compared = 0
    for seed in range(40):
        base, truth, data = _fuzz_case(seed)
        ref = reference_for(truth, data)
        greedy = search(base, data, ref, si.Budget(backtracks=0))
        if greedy.refused is not None:
            continue
        compared += 1
        assert _record(search(base, data, ref)) == _record(greedy), seed
    assert compared >= 40 - 15


Fields = tuple[tuple[str, str], ...]


def _two_conversions(before: Fields = (("a", "s32"),), after: Fields = (("c", "s32"),)) -> tuple[dict, dict, bytes]:
    """1.56's shape: two adjacent u32 fields p, q of a repeated struct each become u16 + str16."""
    base = base_structure()
    fields = (*before, ("p", "u32"), ("q", "u32"), *after, ("label", "str16"))
    base["S"]["structs"]["Item"] = {"retrievers": {n: {"type": t} for n, t in fields}}
    truth = base
    for name in ("p", "q"):
        index = list(truth["S"]["structs"]["Item"]["retrievers"]).index(name)
        truth = si.apply_edit(truth, si.Edit(ITEM, index, 1, [(f"new_{name}", {"type": "u16"}), (name, {"type": "str16"})], "retype"))
    return base, truth, encode(truth, {"n_items": 6})


def test_two_conversions_in_one_repeated_struct_recover_by_retrying() -> None:
    base, truth, data = _two_conversions()
    ref = reference_for(truth, data)
    greedy = search(base, data, ref, si.Budget(backtracks=0))
    assert greedy.refused is not None and greedy.edits, "the fixture must reproduce the greedy dead end"
    out = search(base, data, ref)
    assert out.refused is None and same_layout(out.structure, truth)
    choice = out.edits[0][1]
    assert re.match(r"retry \d at round 1 after a dead end \(no single edit", choice.note), choice.note
    assert choice.abandoned
    # The greedy path's edit took inferred_v9_9_0; the kept path's names start there again, contiguous.
    assert greedy.edits[0][0].new_names() == ["inferred_v9_9_0"]
    assert [n for e, _, _ in out.edits for n in e.new_names()] == ["inferred_v9_9_0", "inferred_v9_9_1"]


def test_a_refusal_after_retries_reports_the_first_dead_end() -> None:
    # With a float before the conversions and nothing after, no retried layout fits either.
    base, truth, data = _two_conversions((("a", "s32"), ("f", "f32")), ())
    ref = reference_for(truth, data)
    greedy = search(base, data, ref, si.Budget(backtracks=0))
    assert greedy.refused is not None and greedy.edits
    for n in (1, 2, si.DEFAULT_BUDGET.backtracks):
        out = search(base, data, ref, si.Budget(backtracks=n))
        assert out.refused is not None
        spent = int(out.refused.removeprefix(greedy.refused + "; ").split()[0])
        assert out.refused == f"{greedy.refused}; {si._retries(spent)} after the dead end found no fit"
        assert 1 <= spent <= n
        assert _record(out) == _record(greedy)


def _trigger_case(seed: int) -> tuple[dict, dict, bytes]:
    """Triggers holding effects and conditions with static_value counts, one s32 inserted into
    each struct: the count fields say both grew, and neither k-insert fits alone."""
    rng = random.Random(seed)
    eff = {"effect_type": "s32", "static_value_x": "s32", **{f"e{i}": "s32" for i in range(rng.randint(3, 6))}, "message": "str16"}
    cond = {"condition_type": "s32", "static_value_y": "s32", **{f"c{i}": "s32" for i in range(rng.randint(3, 6))}, "xs": "str16"}
    trig = {
        "t_a": {"type": "s32"},
        "name": {"type": "str16"},
        "n_eff": {"type": "u32"},
        "effects": {"type": "struct:EffectStruct", "dependencies": _repeat_by("n_eff")},
        "n_cond": {"type": "u32"},
        "conditions": {"type": "struct:ConditionStruct", "dependencies": _repeat_by("n_cond")},
    }
    pairs = (("EffectStruct", eff), ("ConditionStruct", cond))
    structs = {name: {"retrievers": {n: {"type": t} for n, t in rets.items()}} for name, rets in pairs}
    top = {
        "n_trig": {"type": "u32"},
        "trigger_data": {"type": "struct:TriggerStruct", "dependencies": _repeat_by("n_trig")},
        "tail": {"type": "str16"},
    }
    base = {"Triggers": {"retrievers": top, "structs": {"TriggerStruct": {"retrievers": trig, "structs": structs}}}}
    truth = base
    for name, sub, size in (("x", "EffectStruct", len(eff)), ("y", "ConditionStruct", len(cond))):
        block = ("Triggers", "TriggerStruct", sub)
        truth = si.apply_edit(truth, si.Edit(block, rng.randint(2, size - 1), 0, [(name, {"type": "s32"})], "insert s32"))
    tick = itertools.count(1000)

    def text() -> bytes:
        s = f"s{next(tick)}".encode()
        return struct.pack("<H", len(s)) + s

    def item(rets: dict) -> bytes:
        out = b""
        for name, d in rets.items():
            if name.startswith("static_value"):
                out += struct.pack("<i", sum(1 for e in rets.values() if e["type"] == "s32") - si.COUNT_FIELD_BIAS)
            elif d["type"] == "str16":
                out += text()
            else:
                out += struct.pack("<i", rng.choice([0, -1, 1, 2, 70_000 + next(tick)]))
        return out

    t_structs = truth["Triggers"]["structs"]["TriggerStruct"]["structs"]
    data = struct.pack("<I", 12)
    for _ in range(12):
        n_eff, n_cond = rng.randint(0, 3), rng.randint(0, 2)
        data += struct.pack("<i", 70_000 + next(tick)) + text() + struct.pack("<I", n_eff)
        data += b"".join(item(t_structs["EffectStruct"]["retrievers"]) for _ in range(n_eff))
        data += struct.pack("<I", n_cond) + b"".join(item(t_structs["ConditionStruct"]["retrievers"]) for _ in range(n_cond))
    return base, truth, data + text()


def _layouts(structure: dict) -> list:
    out = []

    def visit(node: dict, path: tuple) -> None:
        out.append((path, si.layout_signature(node["retrievers"])))
        for name, child in node.get("structs", {}).items():
            visit(child, (*path, name))

    for name, node in structure.items():
        visit(node, (name,))
    return out


def _blind_triggers(data: bytes, base: dict, budget: si.Budget = si.DEFAULT_BUDGET) -> si.SegmentOutcome:
    inp = si.SegmentInput("synthetic", data, 0, si._end_anchor(len(data)), si.FilterContext(), None)
    return si.SegmentSearch("T", ["Triggers"], [inp], si.Names("9.9"), budget, [0]).run(base)


def test_a_retry_takes_the_count_backed_insert_before_further_reaching_raw_inserts() -> None:
    # 1.42's shape, blind: a raw insert into the parent gets furthest, then dead-ends.
    base, truth, data = _trigger_case(6)
    greedy = _blind_triggers(data, base, si.Budget(backtracks=0))
    assert greedy.refused is not None and greedy.edits[0][0].kind.startswith("insert raw"), _record(greedy)
    out = _blind_triggers(data, base)
    assert out.refused is None and _layouts(out.structure) == _layouts(truth), _record(out)
    first, choice, _ = out.edits[0]
    assert choice.note.startswith("retry 1 at round 1 "), choice.note
    assert first.kind == "insert 1 x s32" and first.block[-1] == "EffectStruct"


def test_a_whole_path_break_needs_nearly_every_value_of_a_path_with_enough_instances() -> None:
    n = si.WHOLE_MIN
    z = np.zeros(2 * n, dtype=np.int64)
    paths = [("S", "items", "a"), ("S", "players", "b")]

    def events(a_miss: int, b_miss: int, b_count: int) -> si.Events:
        index = np.asarray([0] * n + [1] * b_count)
        matched = np.asarray([True] * (n - a_miss) + [False] * a_miss + [True] * (b_count - b_miss) + [False] * b_miss)
        offs = np.arange(len(index), dtype=np.int64) * 4
        return si.Events(offs, z[: len(index)] + 4, matched, matched, paths, index)

    assert si.whole_path_break(events(n, 0, 8)) == 0  # every one of a's values
    assert si.whole_path_break(events(int(n * si.WHOLE_RATIO) - 1, 0, 8)) is None
    assert si.whole_path_break(events(0, 8, 8)) is None  # all of a per-player path's 8: the re-save rewrites those


def test_a_field_the_walk_never_reached_has_an_unknown_multiplicity() -> None:
    # The walk fails in the first item: the field it failed on was read in no item, and how many
    # items the file holds is unknown, so no anchor delta may be divided by a count of 1.
    base = base_structure()
    data = encode(base, {"n_items": 1})[:25]  # n_items, a..e, label's length, 1 byte of its text
    with pytest.raises(sw.WalkError) as e:
        sw.walk(base, data, 0)
    compiled = sw.compile_structure(base)
    found = {(p.block, p.index): p for p in si.positions(compiled, e.value.partial, 0, len(data))}
    assert found[(ITEM, 5)].multiplicity == 0  # label
    assert found[(ITEM, 4)].multiplicity == 1  # e, read once
    edits = si.candidates(base, compiled, [found[(ITEM, 5)]], 30, si.Names("9.9"))
    assert any(e.kind == "insert raw 5" for e in edits)  # the full sweep, not 30 // 1


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


def test_clean_check_runs_the_library_on_the_inferred_structure_in_a_worker(tmp_path) -> None:
    from tools import map_structure_divergence as msd

    sample = FIXTURES / "v159_units_triggers.aoe2scenario"
    for version in ("1.59", "1.58"):
        (tmp_path / f"{version}.json").write_text(json.dumps(si.load_structure(version)), encoding="utf-8")
    good = msd.clean_check(sample, tmp_path / "1.59.json")
    assert (good["status"], good["remainder"]) == ("CLEAN", 0), good
    bad = msd.clean_check(sample, tmp_path / "1.58.json")
    assert (bad["status"], bad["section"]) == ("DIVERGE", "Units"), bad


@pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS behaviour checked on Linux only")
def test_the_worker_address_cap_lowers_an_unlimited_limit_and_never_raises_one() -> None:
    # RLIM_INFINITY is -1 on Linux: a plain min() would "cap" at unlimited.
    code = (
        "import resource\n"
        "from tools import map_structure_divergence as msd\n"
        "msd._cap_address_space(1 << 34)\n"
        "print(resource.getrlimit(resource.RLIMIT_AS)[0])\n"
        "msd._cap_address_space(1 << 35)\n"
        "print(resource.getrlimit(resource.RLIMIT_AS)[0])\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=conftest.ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.split() == [str(1 << 34)] * 2


@pytest.mark.slow
def test_blind_retries_on_trigger_structs_never_come_out_wrong() -> None:
    # The run-struct fuzz never dead-ends blind, so this is the evidence-ranked retries' fuzz: greedy
    # refuses nearly every seed here, and a retried path must fit or refuse, never fit wrongly.
    wrong, recovered = [], 0
    for seed in range(30):
        base, truth, data = _trigger_case(seed)
        out = _blind_triggers(data, base)
        if out.refused is None:
            recovered += any(c.abandoned for _, c, _ in out.edits)
            if not any(c.ambiguous for _, c, _ in out.edits) and _layouts(out.structure) != _layouts(truth):
                wrong.append((seed, _record(out)))
    assert not wrong, wrong
    assert recovered >= TRIGGER_RECOVERED


def test_the_report_lists_the_path_a_retried_edit_abandoned() -> None:
    kept, given_up = EDITS[0], EDITS[1]
    record = si.EditRecord("S", kept, "evidence", "retry 1 at round 1 after a dead end (x); y", [], False, [given_up])
    lines = si.render_report(si.InferResult(si.STATUS_INFERRED, "9.9", "9.8", {}, [record]))
    assert f"  - abandoned after a dead end: {given_up.describe()}" in lines
    # The note names an input file; the emitted structure, which may ship, must not.
    record.note = "retry 1 at round 1 after a dead end (private.aoe2scenario @ 40: ...); the only candidate of its layout"
    result = si.InferResult(si.STATUS_INFERRED, "9.9", "9.8", None, [record])
    emitted = si.emit(si.apply_edit(base_structure(), kept), result, [], "9.9", "9.8")
    comment = emitted["S"]["structs"]["Item"]["retrievers"]["new"]["comment"]
    assert comment == "Inferred for 9.9 from 9.8 (insert s8; retried after a dead end, see INFERRED.md)"
