"""descape/structure_walk.py against AoE2ScenarioParser: same section spans
and same value for every field, plus the guards that make a misaligned
structure fail fast instead of allocating."""

from __future__ import annotations

import contextlib
import io
import json
import math
import struct
import time
import tracemalloc
import uuid
import zlib
from pathlib import Path

import pytest
from AoE2ScenarioParser import settings
from AoE2ScenarioParser.helper.incremental_generator import IncrementalGenerator
from AoE2ScenarioParser.sections.aoe2_file_section import AoE2FileSection

from descape import library_compat
from descape import structure_walk as sw

import conftest

FIXTURES = conftest.ROOT / "tests" / "fixtures"
SMALL_FIXTURES = [
    FIXTURES / name
    for name in (
        "blank_120x120_v158.aoe2scenario",
        "find_triggers_120x120.aoe2scenario",
        "golden_blank_120x120.aoe2scenario",
        "probe216_xs.aoe2scenario",
        "real_blank_120x120_v159.aoe2scenario",
        "real_blank_240x240.aoe2scenario",
        "real_blank_480x480.aoe2scenario",
        "triggers_120x120.aoe2scenario",
        "units_120x120.aoe2scenario",
        "v159_units_triggers.aoe2scenario",
    )
]
TEMPLATES = sorted((conftest.ROOT / "descape" / "templates").glob("*.aoe2scenario"))
LIBRARY_DEFAULTS = [library_compat.VERSIONS_DIR / f"v1.{n}" / "default.aoe2scenario" for n in range(54, 60)]
PRIVATE_CORPUS = conftest.ROOT / "maintainer" / "scenarios_for_descape"


def structure_for(version: str) -> dict:
    for base in (library_compat.VERSIONS_DIR, library_compat.REPO_VERSIONS_DIR):
        path = base / f"v{version}" / "structure.json"
        if path.is_file():
            structure = json.loads(path.read_text(encoding="utf-8"))
            library_compat.fix_structure(structure, version)
            return structure
    raise LookupError(version)


def _flatten(section, path: tuple[str, ...], out: dict) -> None:
    for name, retriever in section.retriever_map.items():
        key = (*path, name)
        if retriever.datatype.type == "struct":
            out.setdefault(key, []).append(len(retriever.data))
            for sub in retriever.data:
                _flatten(sub, key, out)
        else:
            out.setdefault(key, []).append(retriever.data)


def library_parse(structure: dict, raw: bytes) -> tuple[dict, dict]:
    """Every field's values in read order, and each section's span, from the
    library's own section parser. Sections only: no AoE2Scenario object, so
    no class-level link state is touched (tests/README.md, Process-global
    hazard)."""
    settings.PRINT_STATUS_UPDATES = False
    values: dict = {}
    spans: dict = {}
    scenario_id = uuid.uuid4()
    with contextlib.redirect_stdout(io.StringIO()):
        header = IncrementalGenerator("header", raw)
        section = AoE2FileSection.from_structure("FileHeader", structure["FileHeader"], scenario_id)
        section.set_data_from_generator(header)
        spans["FileHeader"] = (0, header.progress)
        _flatten(section, ("FileHeader",), values)
        body = IncrementalGenerator("body", zlib.decompress(raw[header.progress :], -zlib.MAX_WBITS))
        for name, block in structure.items():
            if name == "FileHeader":
                continue
            start = body.progress
            section = AoE2FileSection.from_structure(name, block, scenario_id)
            section.set_data_from_generator(body)
            spans[name] = (start, body.progress)
            _flatten(section, (name,), values)
    return values, spans


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    return type(a) is type(b) and a == b


def parity_problems(path: Path, *, library_may_fail: bool = False) -> list[str] | None:
    """Every disagreement, or None when the library itself can't parse the
    file and `library_may_fail` (some corpus files are refused by both)."""
    raw = path.read_bytes()
    structure = structure_for(raw[:4].decode("ascii"))
    try:
        expected, spans = library_parse(structure, raw)
    except Exception:
        if library_may_fail:
            return None
        raise
    walked = sw.walk_file(structure, raw)
    got = {key: col.value_list() for w in (walked.header, walked.body) for key, col in w.columns.items() if key[-1] != "#"}
    problems = [f"path only on one side: {'.'.join(key)}" for key in set(expected) ^ set(got)]
    for key, values in expected.items():
        mine = got.get(key)
        if mine is not None and not _same(values, mine):
            index = next((i for i, (x, y) in enumerate(zip(values, mine, strict=False)) if not _same(x, y)), min(len(values), len(mine)))
            problems.append(f"{'.'.join(key)} #{index}: library {values[index : index + 1]!r}, walker {mine[index : index + 1]!r}")
    if {**walked.header.sections, **walked.body.sections} != spans:
        problems.append(f"spans differ: {walked.header.sections | walked.body.sections} vs {spans}")
    return problems


@pytest.mark.parametrize("path", SMALL_FIXTURES + TEMPLATES + LIBRARY_DEFAULTS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_walker_matches_the_library_field_by_field(path: Path) -> None:
    assert parity_problems(path) == []


def _corpus_by_version() -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    for base in (conftest.ROOT / "examples", PRIVATE_CORPUS):
        for path in sorted(base.glob("*.aoe2scenario")) if base.is_dir() else []:
            version = path.read_bytes()[:4].decode("ascii", "replace")
            try:
                structure_for(version)
            except LookupError:
                continue
            found.setdefault(version, []).append(path)
    return found


@pytest.mark.corpus
def test_walker_matches_the_library_on_one_corpus_file_per_version() -> None:
    """The first file of each version the library parses; a file the library
    refuses must be refused by the walker too. Reports every version with
    no library-parseable file (1.21 on this corpus: the library fails all of
    them under the repo structure)."""
    by_version = _corpus_by_version()
    if not by_version:
        pytest.skip("no corpus files")
    failures: dict[str, list[str]] = {}
    unparsed = []
    for version, paths in sorted(by_version.items()):
        for path in paths:
            problems = parity_problems(path, library_may_fail=True)
            if problems is None:
                raw = path.read_bytes()
                with pytest.raises(sw.WalkError):
                    sw.walk_file(structure_for(version), raw)
                continue
            if problems:
                failures[path.name] = problems[:3]
            break
        else:
            unparsed.append(version)
    assert failures == {}
    assert set(unparsed) <= {"1.21"}, unparsed


def _field(walk: sw.Walk, *path: str) -> tuple[int, int]:
    col = walk.column(*path)
    return col.offsets[0], col.sizes[0]


def _assert_fails_fast(structure: dict, body: bytes, expect_path: str, expect_reason: str) -> None:
    tracemalloc.start()
    started = time.perf_counter()
    try:
        with pytest.raises(sw.WalkError) as raised:
            sw.walk(structure, body, 0, [n for n in structure if n != "FileHeader"])
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert expect_path in raised.value.path
    assert expect_reason in raised.value.reason
    assert elapsed < 1.0
    assert peak < 100 * 1024 * 1024


def test_a_garbage_grid_size_fails_before_allocating() -> None:
    raw = (FIXTURES / "v159_units_triggers.aoe2scenario").read_bytes()
    structure = structure_for("1.59")
    walked = sw.walk_file(structure, raw)
    body = bytearray(walked.body_bytes)
    width_at, _ = _field(walked.body, "Map", "map_width")
    height_at, _ = _field(walked.body, "Map", "map_height")
    struct.pack_into("<I", body, width_at, 46341)  # 46341^2 is just over 2^31
    struct.pack_into("<I", body, height_at, 46341)
    _assert_fails_fast(structure, bytes(body), "Map.terrain_data", "needs at least")


def test_a_garbage_struct_count_fails_on_the_size_guard() -> None:
    raw = (FIXTURES / "v159_units_triggers.aoe2scenario").read_bytes()
    structure = structure_for("1.59")
    walked = sw.walk_file(structure, raw)
    body = bytearray(walked.body_bytes)
    count_at, _ = _field(walked.body, "Triggers", "number_of_triggers")
    struct.pack_into("<i", body, count_at, 1_000_000)
    _assert_fails_fast(structure, bytes(body), "Triggers.trigger_data", "needs at least")


def test_a_garbage_string_length_fails_before_allocating() -> None:
    raw = (FIXTURES / "v159_units_triggers.aoe2scenario").read_bytes()
    structure = structure_for("1.59")
    walked = sw.walk_file(structure, raw)
    body = bytearray(walked.body_bytes)
    message_at, _ = _field(walked.body, "Triggers", "trigger_data", "effect_data", "message")
    struct.pack_into("<i", body, message_at, 0x7FFFFFFF)
    _assert_fails_fast(structure, bytes(body), "effect_data[0].message", "string length")


def test_the_read_budget_stops_a_walk() -> None:
    raw = (FIXTURES / "v159_units_triggers.aoe2scenario").read_bytes()
    with pytest.raises(sw.WalkError, match="budget"):
        sw.walk_file(structure_for("1.59"), raw, budget=1000)


def _every_structure() -> list[Path]:
    found = sorted(library_compat.VERSIONS_DIR.glob("v*/structure.json"))
    return found + sorted(library_compat.REPO_VERSIONS_DIR.glob("v*/structure.json"))


def test_the_eval_whitelist_accepts_every_read_path_expression() -> None:
    codes = {code for path in _every_structure() for code in sw.read_path_evals(json.loads(path.read_text(encoding="utf-8")))}
    assert len(codes) > 10
    for code in codes:
        sw.compile_eval(code)
    for path in _every_structure():
        sw.compile_structure(json.loads(path.read_text(encoding="utf-8")))


@pytest.mark.parametrize(
    "code",
    ["__import__('os')", "math.sqrt(x)", "open('f')", "x.real", "[1]", "(lambda: 1)()", "[y for y in x]", "len(x, y=1)", "x := 1"],
)
def test_the_eval_whitelist_refuses_anything_else(code: str) -> None:
    with pytest.raises(sw.StructureError):
        sw.compile_eval(code)


def test_eval_semantics_match_python_on_the_library_shapes() -> None:
    env = {"victory_version": 2.0, "unknown_5": [], "number_of_ai_files": [], "trigger_version": 3.5, "w": 3, "h": 4}
    cases = {
        "1 if round(victory_version, 2) >= 2 else 0": 1,
        "0 if type(unknown_5) is list else unknown_5": 0,
        "number_of_ai_files if number_of_ai_files != [] else 0": 0,
        "1 if trigger_version == 3.5 else 0": 1,
        "w * h": 12,
    }
    for code, expected in cases.items():
        assert sw._run(sw.compile_eval(code), env) == expected, code
