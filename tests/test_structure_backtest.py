"""tools/structure_backtest.py: hiding a version the library knows and checking what
tools/structure_infer.py infers for it. Zero WRONG is the hard line everywhere: a refusal
or an ambiguity the report names is the designed fallback, a confident misread is not."""

from __future__ import annotations

import hashlib
import json
from itertools import pairwise

import pytest

from descape import structure_walk as sw
from tools import structure_backtest as bt
from tools import structure_infer as si

import conftest

FIXTURES = conftest.ROOT / "tests" / "fixtures"
SAMPLE = FIXTURES / "v159_units_triggers.aoe2scenario"
TERRAIN_IDS = {int(k) for k in json.loads((conftest.ROOT / "descape" / "terrain_classes.json").read_text(encoding="utf-8"))["classes"]}
UNIT = ("Units", "PlayerUnitsStruct", "UnitStruct")
CONDITION = ("Triggers", "TriggerStruct", "ConditionStruct")

# Every library bump with an edit the search refuses by design (decision 7), by class.
OUT_OF_VOCABULARY = {
    "1.37->1.40": ["new_section"],  # Files
    "1.46->1.47": ["insert_dependent"],  # unknown_bytes2, repeat by trigger_version
    "1.51->1.53": ["insert_dependent", "repeat_rule_change"],  # ai_error; diplomacy_for_ai_system and four victory_version rules
    "1.53->1.54": ["insert_dependent"],  # the static_value-dependent effect and condition fillers
    "1.54->1.55": ["insert_dependent"],  # legacy_exec_order, repeat by trigger_version
}


def test_every_bump_in_history_is_in_the_vocabulary_or_listed() -> None:
    history = bt.history()
    assert len(history) == 19
    out = {pair: sorted({k for _, k, _ in edits if k not in bt.IN_VOCABULARY}) for pair, edits in history.items()}
    assert {pair: kinds for pair, kinds in out.items() if kinds} == OUT_OF_VOCABULARY
    # No bump hides its size changes from a segment-length anchor by cancelling them out.
    assert {pair: bt.cancelling_segments(edits) for pair, edits in history.items() if bt.cancelling_segments(edits)} == {}
    # A delete counts by its size: review #9's delete plus an insert of the same size cancels.
    before = {"Map": {"retrievers": {"a": {"type": "s32"}, "b": {"type": "u8"}}}}
    deleted = bt.history_edits(before, {"Map": {"retrievers": {"b": {"type": "u8"}}}})
    assert deleted == [("Map", "delete", "a (4 B)")]
    assert bt.cancelling_segments([*deleted, ("Map", "insert_fixed", "c (4 B)")]) == [si.SEG_PREGRID]


def test_the_comparator_tells_exact_shape_and_wrong_apart() -> None:
    raw = SAMPLE.read_bytes()
    truth, base = si.load_structure("1.59"), si.load_structure("1.58")
    assert bt.compare_files(truth, truth, [raw], base).verdict == bt.EXACT
    # capture_flag read as raw bytes: the same bytes, unnamed and untyped.
    raw_flag = si.apply_edit(base, si.Edit(UNIT, 9, 0, [("inferred_v1_59_0", {"type": "1"})], "insert raw"))
    raw_flag = si.apply_edit(raw_flag, si.Edit(CONDITION, 35, 0, [("inferred_v1_59_1", {"type": "s32"})], "insert"))
    shaped = bt.compare_files(raw_flag, truth, [raw], base)
    assert (shaped.verdict, shaped.reshaped) == (bt.SHAPE, [("Units", "players_units", "units", "capture_flag")])
    # One position early: rotation's bytes no longer read as rotation.
    early = si.apply_edit(base, si.Edit(UNIT, 6, 0, [("inferred_v1_59_0", {"type": "s8"})], "insert"))
    early = si.apply_edit(early, si.Edit(CONDITION, 35, 0, [("inferred_v1_59_1", {"type": "s32"})], "insert"))
    wrong = bt.compare_files(early, truth, [raw], base)
    assert wrong.verdict == bt.WRONG and ("Units", "players_units", "units", "rotation") in wrong.misread
    # The base itself can't walk a 1.59 file at all.
    assert bt.compare_files(base, truth, [raw], base).misread[0][0] == "<walk>"


def _renamed(structure: dict, block: tuple[str, ...], old: str, new: str) -> dict:
    names = list(si.block_dict(structure, block)["retrievers"])
    definition = si.block_dict(structure, block)["retrievers"][old]
    return si.apply_edit(structure, si.Edit(block, names.index(old), 1, [(new, definition)], "rename"))


def test_a_base_name_frames_a_true_field_only_as_its_one_to_one_rename_at_the_same_prefix() -> None:
    """Each layout reads every byte the truth reads with equal values; only which name frames
    which true field tells the misplaced edit from the right one."""
    raw = SAMPLE.read_bytes()
    truth = si.load_structure("1.59")
    flag = ("Units", "players_units", "units", "capture_flag")
    # The truth deleted a base `ghost` before capture_flag; deleting capture_flag instead keeps
    # `ghost`, a name the truth lacks, reading capture_flag's bytes.
    base = si.apply_edit(truth, si.Edit(UNIT, 9, 0, [("ghost", {"type": "s8"})], "insert"))
    misdeleted = si.apply_edit(base, si.Edit(UNIT, 10, 1, [], "delete"))
    c = bt.compare_files(misdeleted, truth, [raw], base)
    assert c.verdict == bt.WRONG and flag in c.misread, c

    # The truth renamed object_type2_old and inserted allow_in_fog after it. Inserting one early
    # leaves the renamed field reading allow_in_fog's bytes; the right position is EXACT.
    base = _renamed(si.load_structure("1.58"), CONDITION, "object_type2", "object_type2_old")
    pairs = bt.rename_pairs(base, truth)
    assert pairs == {("Triggers", "trigger_data", "condition_data"): frozenset({("object_type2_old", "object_type2")})}
    s = si.apply_edit(base, si.Edit(UNIT, 9, 0, [("inferred_v1_59_0", {"type": "s8"})], "insert"))
    at = list(si.block_dict(base, CONDITION)["retrievers"]).index("object_type2_old")
    verdicts = {}
    for index in (at, at + 1):
        inferred = si.apply_edit(s, si.Edit(CONDITION, index, 0, [("inferred_v1_59_1", {"type": "s32"})], "insert"))
        verdicts[index] = bt.compare_files(inferred, truth, [raw], base)
    assert verdicts[at].verdict == bt.WRONG and ("Triggers", "trigger_data", "condition_data", "allow_in_fog") in verdicts[at].misread
    assert verdicts[at + 1].verdict == bt.EXACT, verdicts[at + 1]

    # Messages' first field read as a new field at the end of DataHeader: same bytes, wrong section.
    base = si.apply_edit(truth, si.Edit(("Messages",), 0, 1, [], "delete"))
    instructions = truth["Messages"]["retrievers"]["instructions"]
    end = len(base["DataHeader"]["retrievers"])
    shifted = si.apply_edit(base, si.Edit(("DataHeader",), end, 0, [("inferred_v1_59_0", instructions)], "insert"))
    c = bt.compare_files(shifted, truth, [raw], base)
    assert c.verdict == bt.WRONG and ("Messages", "instructions") in c.misread, c


@pytest.mark.parametrize(("index", "kind"), [(31, "s32"), (29, "s32"), (34, "4")])
def test_a_condition_insert_at_the_wrong_position_is_wrong_though_every_byte_matches(index, kind) -> None:
    """The fixture's conditions are default from 29 to 35, so an insert anywhere there reads the
    same values; only the names tell a base field framed on a true field's bytes from the truth."""
    raw = SAMPLE.read_bytes()
    truth, base = si.load_structure("1.59"), si.load_structure("1.58")
    s = si.apply_edit(base, si.Edit(UNIT, 9, 0, [("inferred_v1_59_0", {"type": "s8"})], "insert"))
    s = si.apply_edit(s, si.Edit(CONDITION, index, 0, [("inferred_v1_59_1", {"type": kind})], "insert"))
    c = bt.compare_files(s, truth, [raw], base)
    assert c.verdict == bt.WRONG and ("Triggers", "trigger_data", "condition_data", "allow_in_fog") in c.misread


def test_infer_on_the_159_fixture_from_158(tmp_path) -> None:
    truth, base = si.load_structure("1.59"), si.load_structure("1.58")
    values = bt.simulated_reference(truth, SAMPLE.read_bytes(), base)
    referenced = si.infer("1.58", [SAMPLE], reference_values=values, terrain_ids=TERRAIN_IDS, library_check=None)
    # Units is decided; the fixture's one non-default condition is below PAIR_MIN, so Conditions stays open.
    assert referenced.status == si.STATUS_AMBIGUOUS, referenced.report_lines
    assert [r.ambiguous for r in referenced.edits] == [False, True]
    blind = si.infer("1.58", [SAMPLE], terrain_ids=TERRAIN_IDS, library_check=None)
    assert blind.status == si.STATUS_AMBIGUOUS

    for result in (referenced, blind):
        kept = {r.edit.block: r.edit.index for r in result.edits}
        assert kept == {UNIT: 9, CONDITION: 35}  # capture_flag and allow_in_fog's true positions
        outcome = bt.classify(result, truth, [SAMPLE.read_bytes()], base)
        assert (outcome.kind, outcome.comparison.verdict) == (bt.AMBIGUOUS, bt.EXACT), outcome.detail

    s = referenced.structure
    capture = si.block_dict(s, UNIT)["retrievers"]["inferred_v1_59_0"]
    fog = si.block_dict(s, CONDITION)["retrievers"]["inferred_v1_59_1"]
    assert capture["type"] == "s8" and fog["type"] == "s32"
    assert capture["comment"].startswith("Inferred for 1.59 from 1.58")
    flags = sw.walk_file(truth, SAMPLE.read_bytes()).body.columns[("Units", "players_units", "units", "capture_flag")]
    assert capture["default"] == si._most_common(flags.value_list())
    assert si.block_dict(s, CONDITION)["retrievers"]["static_value_33"]["default"] == 34
    assert s["FileHeader"]["retrievers"]["version"]["default"] == values[("FileHeader", "version")][0]
    assert s["Triggers"]["retrievers"]["trigger_version"]["default"] == values[("Triggers", "trigger_version")][0]

    # The write side: DEscape loads, saves and edits the file through the inferred structure.
    judged = bt.classify(blind, truth, [SAMPLE.read_bytes()], base).comparison
    assert bt.write_side(judged.structure, judged.names, SAMPLE, "1.59", tmp_path) == []


@pytest.mark.parametrize(("unit_at", "condition_at", "says"), [(8, 35, "units.garrisoned_in_id"), (9, 34, "conditions.allow_in_fog")])
def test_the_write_side_fails_a_named_layout_that_misplaces_a_field(tmp_path, unit_at, condition_at, says) -> None:
    """Control: the write side is worth what it can catch. Each layout carries the true names, so
    it loads and round-trips; one field sits one position early, before its neighbour."""
    base = si.load_structure("1.58")
    s = si.apply_edit(base, si.Edit(UNIT, unit_at, 0, [("capture_flag", {"type": "s8", "default": -1})], "insert"))
    s = si.apply_edit(s, si.Edit(CONDITION, condition_at, 0, [("allow_in_fog", {"type": "s32", "default": -1})], "insert"))
    problems = bt.write_side(s, {}, SAMPLE, "1.59", tmp_path)
    assert any(says in p for p in problems), problems


@pytest.mark.test_map
def test_the_frozen_resave_pair_keeps_the_truth_and_leaves_only_default_positions_open() -> None:
    """The real re-save backtest: the game's own re-save of the test map, base 1.58."""
    result = si.infer(
        "1.58",
        [conftest.TEST_MAP_RESAVE_PATH],
        reference=(conftest.TEST_MAP_RESAVE_SOURCE_PATH, conftest.TEST_MAP_RESAVE_PATH),
        terrain_ids=TERRAIN_IDS,
        library_check=None,
    )
    assert result.status == si.STATUS_AMBIGUOUS, result.report_lines
    by_block = {r.edit.block: r for r in result.edits}
    unit = by_block[UNIT]
    condition = by_block[CONDITION]
    assert len(result.edits) == 2
    assert unit.edit.index == 9 and not unit.ambiguous  # capture_flag, decided by the reference despite the game's rewrites
    assert condition.edit.index == 35 and condition.ambiguous  # allow_in_fog
    # Open: positions that read every source condition identically (33, 34) or differ from the
    # truth in fewer than PAIR_MIN values (31, 32); the reference rules out 18..30.
    assert sorted(e.index for e in [condition.edit, *condition.rivals]) == [31, 32, 33, 34, 35]
    raw = conftest.TEST_MAP_RESAVE_PATH.read_bytes()
    outcome = bt.classify(result, si.load_structure("1.59"), [raw], si.load_structure("1.58"))
    assert (outcome.kind, outcome.comparison.verdict) == (bt.AMBIGUOUS, bt.EXACT), outcome.detail


def test_an_ambiguous_set_without_the_truth_is_wrong() -> None:
    truth, base = si.load_structure("1.59"), si.load_structure("1.58")
    wrong = si.Edit(UNIT, 6, 0, [("inferred_v1_59_0", {"type": "s8"})], "insert")
    also_wrong = si.Edit(UNIT, 5, 0, [("inferred_v1_59_0", {"type": "s8"})], "insert")
    fog = si.Edit(CONDITION, 35, 0, [("inferred_v1_59_1", {"type": "s32"})], "insert")
    structure = bt.rebuild(base, [wrong, fog])
    record = si.EditRecord(si.SEG_UNITS, wrong, "", "", [also_wrong], True)
    records = [record, si.EditRecord(si.SEG_TRIGGERS, fog, "", "", [], False)]
    result = si.InferResult(si.STATUS_AMBIGUOUS, "1.59", "1.58", structure, records)
    assert bt.classify(result, truth, [SAMPLE.read_bytes()], base).kind == bt.WRONG
    # With the truth among the rivals, the same set is AMBIGUOUS.
    record.rivals.append(si.Edit(UNIT, 9, 0, [("inferred_v1_59_0", {"type": "s8"})], "insert"))
    assert bt.classify(result, truth, [SAMPLE.read_bytes()], base).kind == bt.AMBIGUOUS


def test_a_simulated_reference_holds_only_the_base_paths() -> None:
    truth, base = si.load_structure("1.59"), si.load_structure("1.58")
    values = bt.simulated_reference(truth, SAMPLE.read_bytes(), base)
    assert ("Units", "players_units", "units", "capture_flag") not in values
    assert ("Units", "players_units", "units", "x") in values
    assert set(values) <= bt.base_paths(base)


# The corpus coverage table (2026-11-16, examples/ plus the maintainer corpus): per transition,
# the number of distinct corpus files of the new version, then the blind and simulated-reference outcomes.
# REFUSED in-vocabulary pairs: 1.41->1.42 (an Effect and a Condition insert in one segment) and
# 1.55->1.56 (two u16+str16 conversions in one struct), where the greedy partial step stacks a raw
# insert and dead-ends. 1.36->1.37 refuses for another reason: the library has no Files section
# before 1.40, so even the true 1.37 structure leaves the body's tail unread (8 bytes, or 13 MB of
# embedded AI files) and no layout meets the end anchor. A 1.40+ base can't hit that.
CORPUS_COVERAGE = {
    "1.36->1.37": (3, bt.REFUSED, bt.REFUSED),
    "1.37->1.40": (5, bt.REFUSED, bt.REFUSED),
    "1.40->1.41": (43, bt.AMBIGUOUS, bt.AMBIGUOUS),
    "1.41->1.42": (6, bt.REFUSED, bt.REFUSED),
    "1.42->1.43": (3, bt.EXACT, bt.EXACT),
    "1.44->1.45": (31, bt.AMBIGUOUS, bt.AMBIGUOUS),
    "1.46->1.47": (2, bt.REFUSED, bt.REFUSED),
    "1.47->1.48": (2, bt.EXACT, bt.EXACT),
    "1.48->1.49": (3, bt.SHAPE, bt.SHAPE),
    "1.49->1.51": (2, bt.SHAPE, bt.SHAPE),
    "1.51->1.53": (1, bt.REFUSED, bt.REFUSED),
    "1.53->1.54": (10, bt.REFUSED, bt.REFUSED),
    "1.54->1.55": (4, bt.REFUSED, bt.REFUSED),
    "1.55->1.56": (8, bt.REFUSED, bt.REFUSED),
    "1.56->1.57": (2, bt.AMBIGUOUS, bt.EXACT),
    "1.57->1.58": (15, bt.AMBIGUOUS, bt.AMBIGUOUS),
    "1.58->1.59": (1, bt.AMBIGUOUS, bt.AMBIGUOUS),
}


def _unique(paths: list) -> list:
    """The first of each set of byte-identical files (examples/ and the private corpus share some)."""
    seen: set[bytes] = set()
    out = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).digest()
        if digest not in seen:
            seen.add(digest)
            out.append(path)
    return out


def _corpus_transitions() -> list[tuple[str, str, list]]:
    from test_structure_walk import _corpus_by_version

    by_version = _corpus_by_version()
    versions = bt.library_versions()
    return [(a, b, sorted(_unique(by_version[b]), key=lambda p: p.stat().st_size)) for a, b in pairwise(versions) if b in by_version]


@pytest.mark.corpus
@pytest.mark.parametrize("pair", sorted(CORPUS_COVERAGE))
def test_the_corpus_backtest_is_never_wrong_and_keeps_its_coverage(pair, tmp_path) -> None:
    """Up to three of the smallest distinct files of the new version are the samples (the first
    gives the simulated reference); every distinct corpus file of that version is compared. The
    pin needs the private corpus; with it, a corpus of another size than the table's fails, so the
    pin is re-measured rather than silently skipped."""
    from test_structure_walk import PRIVATE_CORPUS

    found = {f"{a}->{b}": (a, b, files) for a, b, files in _corpus_transitions()}
    if pair not in found:
        pytest.skip(f"no corpus file for {pair}")
    base, version, files = found[pair]
    outcomes = {}
    for mode in ("blind", "simulated"):
        outcome = bt.run(base, version, files[:3], compare=files, mode=mode, terrain_ids=TERRAIN_IDS)
        print(f"{pair} {mode}: {outcome.kind} in {outcome.seconds:.1f}s {outcome.detail[:200]}")
        assert outcome.kind != bt.WRONG, outcome.detail
        outcomes[mode] = outcome
    if version == "1.59":
        # The write side on every real 1.59 file, through the layout the blind result was judged on.
        judged = outcomes["blind"].comparison
        for i, path in enumerate(files):
            work = tmp_path / str(i)
            work.mkdir()
            assert bt.write_side(judged.structure, judged.names, path, version, work) == [], path.name
    got = (len(files), outcomes["blind"].kind, outcomes["simulated"].kind)
    if not PRIVATE_CORPUS.is_dir():
        pytest.skip(f"never WRONG on {got[0]} examples/ files; the coverage pin needs the private corpus at {PRIVATE_CORPUS}")
    assert got == CORPUS_COVERAGE[pair], f"re-pin CORPUS_COVERAGE[{pair!r}] to {got} if the corpus or the search changed on purpose"
