"""descape/scenario_compat.py: the compatibility view's analysis and proxy
(TASK-057). A newer version is simulated by reading a known file under the
version before it (or a file's c4 rewritten to 1.60 under 1.59); "truth" is
the normal load of the same file under its own version, and every area the
analysis trusts must agree with it."""

from __future__ import annotations

import struct
import time
import zlib
from pathlib import Path

import pytest

from descape import format_scan as fs
from descape import library_compat, scenario_compat, scenario_io, scenario_new
from descape import structure_walk as sw

import conftest

FIXTURES = conftest.ROOT / "tests" / "fixtures"
V159 = FIXTURES / "v159_units_triggers.aoe2scenario"
BLANK_V158 = FIXTURES / "blank_120x120_v158.aoe2scenario"
UNIT_FIELDS = (
    "x", "y", "z", "reference_id", "unit_const", "status", "rotation",
    "initial_animation_frame", "garrisoned_in_id", "caption_string_id",
)  # fmt: skip


def _deflate(data: bytes) -> bytes:
    packer = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
    return packer.compress(data) + packer.flush()


def c4(raw: bytes, version: str = "1.60") -> bytes:
    return version.encode("ascii") + raw[4:]


def _rebuilt(raw: bytes, edit) -> bytes:
    """`raw` with its body replaced by edit(body, grid)."""
    plan = scenario_compat.analyse(raw)
    assert isinstance(plan, scenario_compat.CompatPlan), plan
    return raw[: plan.body.start] + _deflate(edit(bytearray(plan.body.data), plan.grid))


def _plain(value):
    if hasattr(value, "retriever_map"):
        return {k: _plain(r.data) for k, r in value.retriever_map.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def _retrievers(loaded, section: str) -> dict:
    return {k: _plain(r.data) for k, r in loaded._scenario.sections[section].retriever_map.items()}


def disagreements(proxy, truth) -> list[str]:
    """Every trusted area of `proxy` that differs from `truth`."""
    plan = proxy.compat
    bad = []
    tiles = [(t.terrain_id, t.elevation, t.layer) for t in proxy.map_manager.terrain]
    if tiles != [(t.terrain_id, t.elevation, t.layer) for t in truth.map_manager.terrain]:
        bad.append("terrain")
    if plan.settings_trusted:
        names = list(truth._scenario.structure)
        for section in ["FileHeader", *names[names.index("DataHeader") : names.index("Map")]]:
            got, want = _retrievers(proxy, section), _retrievers(truth, section)
            # FileHeader.version is the c4 itself, the file's own in a proxy's header (compat.file_version).
            bad += [f"{section}.{k}" for k in got.keys() & want.keys() if got[k] != want[k] and (section, k) != ("FileHeader", "version")]
        got, want = _retrievers(proxy, "Map"), _retrievers(truth, "Map")
        pre = list(want)[: list(want).index("map_width")]
        bad += [f"Map.{k}" for k in pre if k in got and got[k] != want[k]]
    if plan.units_trusted:
        for pid, (a, b) in enumerate(zip(proxy.unit_manager.units, truth.unit_manager.units, strict=True)):
            rows = [[tuple(getattr(u, f, None) for f in UNIT_FIELDS) for u in units] for units in (a, b)]
            if rows[0] != rows[1]:
                bad.append(f"units of player {pid}")
    return bad


def open_ok(raw: bytes, base: str, donor: bytes | None = None):
    loaded = scenario_compat.open_compat(raw, "compat.aoe2scenario", base, donor)
    assert not isinstance(loaded, scenario_compat.CompatRefusal), loaded
    return loaded


def truth_of(raw: bytes):
    return scenario_io.load_map_and_units_from_bytes(raw, "truth.aoe2scenario")


def verdict(loaded) -> tuple[bool, bool]:
    return loaded.compat.settings_trusted, loaded.compat.units_trusted


# -- the gate and the newest version ------------------------------------------


def test_newest_structure_version_agrees_with_the_tools_lists() -> None:
    from tools import structure_backtest as bt
    from tools import structure_infer as si

    newest = scenario_io.newest_structure_version()
    assert newest == si.known_versions()[-1]
    assert scenario_io._version_key(newest) >= scenario_io._version_key(bt.library_versions()[-1])


def test_newest_structure_version_follows_a_higher_repo_version(tmp_path, monkeypatch) -> None:
    (tmp_path / "v9.99").mkdir()
    (tmp_path / "v9.99" / "structure.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(library_compat, "REPO_VERSIONS_DIR", tmp_path)
    assert scenario_io.newest_structure_version() == "9.99"


def test_applies_only_to_versions_newer_than_every_structure() -> None:
    newest = scenario_io.newest_structure_version()
    major, minor = newest.split(".")
    assert scenario_compat.applies(f"{major}.{int(minor) + 1}")
    assert not scenario_compat.applies(newest)
    assert not scenario_compat.applies("1.32")
    assert not scenario_compat.applies("junk")


def test_the_shipped_templates_are_the_newest_version() -> None:
    newest = scenario_io.newest_structure_version()
    for game in ("aoe2", "ror"):
        path = scenario_new.blank_template_path_for(game)
        assert path.read_bytes()[:4].decode("ascii") == newest, f"regenerate {path.name} at {newest}"


# -- trusted areas agree with the truth --------------------------------------


def test_v159_read_as_158_trusts_settings_not_units() -> None:
    raw = V159.read_bytes()
    proxy = open_ok(raw, "1.58", BLANK_V158.read_bytes())
    assert verdict(proxy) == (True, False)  # 1.59 inserted capture_flag into UnitStruct
    assert disagreements(proxy, truth_of(raw)) == []
    assert sum(map(len, proxy.unit_manager.units)) == 0
    assert proxy.trigger_version == -1.0 and proxy.trigger_read_supported is False


def test_v159_as_160_under_159_trusts_everything() -> None:
    raw = V159.read_bytes()
    proxy = open_ok(c4(raw), "1.59")
    assert verdict(proxy) == (True, True)
    assert proxy.compat.file_version == "1.60" and proxy.scenario_version == "1.59"
    truth = truth_of(raw)
    assert disagreements(proxy, truth) == []
    assert sum(map(len, proxy.unit_manager.units)) == sum(map(len, truth.unit_manager.units)) > 0
    assert proxy.trigger_version == truth.trigger_version


@pytest.mark.test_map
def test_the_game_resave_under_158_and_as_160() -> None:
    raw = conftest.TEST_MAP_RESAVE_PATH.read_bytes()
    truth = truth_of(raw)
    proxy = open_ok(raw, "1.58", (FIXTURES / "real_blank_240x240.aoe2scenario").read_bytes())
    assert verdict(proxy) == (True, False)
    assert disagreements(proxy, truth) == []
    proxy = open_ok(c4(raw), "1.59")
    assert verdict(proxy) == (True, True)
    assert disagreements(proxy, truth) == []
    assert sum(map(len, proxy.unit_manager.units)) == sum(map(len, truth.unit_manager.units)) > 1000


LIBRARY_DEFAULTS = {"1.55": (True, True), "1.56": (False, True), "1.57": (True, True), "1.58": (True, True), "1.59": (True, True)}


@pytest.mark.parametrize("version", sorted(LIBRARY_DEFAULTS))
def test_library_defaults_read_as_the_version_before(version) -> None:
    """1.56 via 1.55 is the case that reads DataHeader..Diplomacy at wrong
    offsets without raising; the anchor must leave Settings Unknown."""
    previous = f"1.{int(version.split('.')[1]) - 1}"
    raw = (library_compat.VERSIONS_DIR / f"v{version}" / "default.aoe2scenario").read_bytes()
    donor = (library_compat.VERSIONS_DIR / f"v{previous}" / "default.aoe2scenario").read_bytes()
    proxy = open_ok(raw, previous, donor)
    assert verdict(proxy) == LIBRARY_DEFAULTS[version]
    assert disagreements(proxy, truth_of(raw)) == []


# -- untrusted units ----------------------------------------------------------


def _units_walk(body: bytes, grid: fs.GridMatch, base: str = "1.59") -> sw.Walk:
    return sw.walk(scenario_compat.base_structure(base), body, grid.end, ["Units"])


def test_a_byte_inserted_in_every_unit_makes_units_unknown() -> None:
    def insert(body, grid):
        col = _units_walk(bytes(body), grid).column(*fs.UNITS_PREFIX, "garrisoned_in_id")
        assert len(col) > 0
        for off, size in sorted(zip(col.offsets, col.sizes, strict=True), reverse=True):
            body[off + size : off + size] = b"\x00"
        return bytes(body)

    plan = scenario_compat.analyse(_rebuilt(c4(V159.read_bytes()), insert), "1.59")
    assert plan.settings_trusted and not plan.units_trusted, plan.reasons


def test_a_huge_unit_count_makes_units_unknown_fast() -> None:
    def huge(body, grid):
        off = _units_walk(bytes(body), grid).column("Units", "players_units", "unit_count").offsets[0]
        body[off : off + 4] = struct.pack("<I", 0x7FFFFFF0)
        return bytes(body)

    raw = _rebuilt(c4(V159.read_bytes()), huge)
    t0 = time.perf_counter()
    plan = scenario_compat.analyse(raw, "1.59")
    assert time.perf_counter() - t0 < 1.0
    assert plan.settings_trusted and not plan.units_trusted, plan.reasons


def test_units_ending_on_an_implausible_trigger_version_are_unknown() -> None:
    def garble(body, grid):
        end = _units_walk(bytes(body), grid).end
        body[end : end + 8] = struct.pack("<d", 123.456)
        return bytes(body)

    plan = scenario_compat.analyse(_rebuilt(c4(V159.read_bytes()), garble), "1.59")
    assert plan.settings_trusted and not plan.units_trusted
    assert "trigger version" in plan.reasons["units"]


# -- refusals -----------------------------------------------------------------


def _refusal(raw: bytes, base: str = "1.59", donor: bytes | None = None) -> str:
    result = scenario_compat.open_compat(raw, "x.aoe2scenario", base, donor)
    assert isinstance(result, scenario_compat.CompatRefusal), result
    return result.reason


def test_a_corrupted_grid_refuses() -> None:
    def corrupt(body, grid):
        body[grid.tiles_start + 1] = 200  # an elevation no map has
        return bytes(body)

    assert "terrain could not be found" in _refusal(_rebuilt(c4(V159.read_bytes()), corrupt))


def test_a_second_grid_shaped_match_refuses() -> None:
    def twice(body, grid):
        return bytes(body) + bytes(body[grid.offset : grid.end + 4])

    assert "2 grid-shaped matches" in _refusal(_rebuilt(c4(V159.read_bytes()), twice))


def test_a_donor_of_another_version_refuses() -> None:
    reason = _refusal(c4(V159.read_bytes()), "1.59", BLANK_V158.read_bytes())
    assert "templates are version 1.58" in reason


def test_garbage_refuses_without_raising() -> None:
    assert "data could not be found" in _refusal(b"1.60" + bytes(4096))


def test_the_proxy_is_never_written(tmp_path) -> None:
    from descape import scenario_write

    proxy = open_ok(c4(V159.read_bytes()), "1.59")
    with pytest.raises(scenario_write.CompatReadOnly):
        scenario_write.write_scenario(proxy, tmp_path / "out.aoe2scenario")
    with pytest.raises(scenario_write.CompatReadOnly):
        scenario_write.build_patched_body(proxy)
    assert not (tmp_path / "out.aoe2scenario").exists()


def test_load_for_viewer_opens_a_newer_file_and_refuses_with_the_reason(tmp_path, monkeypatch) -> None:
    path = tmp_path / "newer.aoe2scenario"
    path.write_bytes(c4(V159.read_bytes()))
    loaded = scenario_compat.load_for_viewer(path, scenario_io.load_map_and_units)
    assert loaded.compat is not None and loaded.compat.file_version == "1.60"
    monkeypatch.setattr(scenario_compat, "TEST_DONOR", BLANK_V158.read_bytes())
    with pytest.raises(scenario_io.UnsupportedStructureVersion, match="Compatibility mode could not read it either"):
        scenario_compat.load_for_viewer(path, scenario_io.load_map_and_units)
    old = tmp_path / "old.aoe2scenario"
    old.write_bytes(c4(V159.read_bytes(), "1.32"))
    with pytest.raises(scenario_io.UnsupportedStructureVersion) as e:
        scenario_compat.load_for_viewer(old, scenario_io.load_map_and_units)
    assert "Compatibility" not in str(e.value)


# -- corpus tier --------------------------------------------------------------

# Per consecutive corpus pair "base->version": (distinct files, files with
# Settings trusted, files with Units trusted). Measured on examples/ plus the
# private corpus; re-pin when either changes on purpose.
COMPAT_COVERAGE: dict[str, tuple[int, int, int]] = {
    "1.36->1.37": (3, 0, 0),
    "1.37->1.40": (5, 0, 5),
    "1.40->1.41": (43, 0, 43),
    "1.41->1.42": (6, 0, 6),
    "1.42->1.43": (3, 3, 3),
    "1.44->1.45": (31, 0, 0),
    "1.46->1.47": (2, 0, 0),
    "1.47->1.48": (2, 2, 2),
    "1.48->1.49": (3, 3, 3),
    "1.49->1.51": (2, 2, 2),
    "1.51->1.53": (1, 0, 0),
    "1.53->1.54": (10, 10, 0),
    "1.54->1.55": (4, 4, 0),
    "1.55->1.56": (8, 0, 8),
    "1.56->1.57": (2, 2, 2),
    "1.57->1.58": (15, 15, 15),
    "1.58->1.59": (1, 1, 0),
}


def _corpus_pairs() -> list[tuple[str, str, list[Path]]]:
    from test_structure_backtest import _corpus_transitions

    return _corpus_transitions()


def _walker_truth_agrees(plan: scenario_compat.CompatPlan, raw: bytes, version: str) -> list[str]:
    """Below 1.54 there is no donor: the verdicts are checked against a walk
    of the truth under `version` over the same bytes."""
    if not (plan.settings_trusted or plan.units_trusted):
        return []
    # Up to Units only: the walker can't read every old file's Triggers.
    compiled = sw.compile_structure(scenario_compat.base_structure(version))
    names = list(compiled.sections)
    try:
        truth = sw.walk(compiled, plan.body.data, 0, names[1 : names.index("Units") + 1])
    except sw.WalkError as e:
        return [f"trusted, but the truth walk under {version} fails: {e}"]
    bad = []
    if plan.settings_trusted and truth.sections["Map"][1] != plan.grid.end:
        bad.append("settings")
    if plan.units_trusted:
        base = sw.walk(scenario_compat.base_structure(plan.base), plan.body.data, plan.grid.end, ["Units"])
        if base.end != truth.end:
            bad.append("units end")
        for name in UNIT_FIELDS:
            path = (*fs.UNITS_PREFIX, name)
            if path in base.columns and base.values(*path) != truth.values(*path):
                bad.append(f"units.{name}")
    return bad


@pytest.mark.corpus
def test_the_corpus_backtest_trusts_nothing_wrong() -> None:
    from test_structure_walk import PRIVATE_CORPUS

    coverage, wrong, timing = {}, [], []
    for base, version, files in _corpus_pairs():
        donor_path = library_compat.VERSIONS_DIR / f"v{base}" / "default.aoe2scenario"
        settings = units = 0
        for path in files:
            raw = path.read_bytes()
            t0 = time.perf_counter()
            plan = scenario_compat.analyse(raw, base)
            if isinstance(plan, scenario_compat.CompatRefusal):
                wrong.append(f"{path.name}: refused: {plan.reason}")
                continue
            settings += plan.settings_trusted
            units += plan.units_trusted
            if donor_path.is_file():
                proxy = scenario_compat.open_compat(raw, path, base, donor_path.read_bytes())
                if isinstance(proxy, scenario_compat.CompatRefusal):
                    wrong.append(f"{path.name}: proxy refused: {proxy.reason}")
                    continue
                timing.append((time.perf_counter() - t0, path.name))
                bad = disagreements(proxy, truth_of(raw))
            else:
                bad = _walker_truth_agrees(plan, raw, version)
            wrong += [f"{base}->{version} {path.name}: {b}" for b in bad]
        coverage[f"{base}->{version}"] = (len(files), settings, units)
    for seconds, name in sorted(timing)[-3:]:
        print(f"compat open {seconds:.2f}s {name}")
    largest = max((p for _, _, files in _corpus_pairs() for p in files), key=lambda p: p.stat().st_size)
    t0 = time.perf_counter()
    assert isinstance(fs.find_body(largest.read_bytes(), min_bytes=fs.COMPAT_MIN_BODY_BYTES), fs.Body)
    print(f"find_body() fallback {time.perf_counter() - t0:.2f}s {largest.name}")
    print(f"coverage {coverage}")
    assert wrong == []
    assert coverage.get("1.55->1.56", (0, 0, 0))[1] == 0, "1.56 via 1.55 misreads Settings; the anchor must refuse it"
    if not PRIVATE_CORPUS.is_dir():
        pytest.skip(f"no trusted area disagrees; the coverage pin needs the private corpus at {PRIVATE_CORPUS}")
    assert coverage == COMPAT_COVERAGE, f"re-pin COMPAT_COVERAGE to {coverage} if the corpus changed on purpose"
