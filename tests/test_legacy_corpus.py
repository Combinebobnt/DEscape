"""TASK-429's corpus tier: every pre-DE `.scn`/`.scx` under the scenario dir
loads through its legacy structure, with byte-level checks the structure
cannot satisfy by accident. Point it at a folder of old scenarios:

    pytest -m corpus tests/test_legacy_corpus.py --scenario-dir <dir>

The folder is searched recursively (old installs nest their Scenario dirs).
A header-3 file (HD's own `.aoe2scenario` format) and a header-2 band with
no structure yet each skip, naming why, never fail.

What a pass proves: the byte total through Units (the verifiers, the
fast walks and the trigger tail) and the GlobalVictory, Diplomacy and Map
separators, which pin the section boundaries between them. What it cannot:
which of two same-size fields is which. test_legacy_oracle.py's panel
comparison covers that.

Every file loads in this process, like test_v121_alignment.py. The class
poisoning hazard in tests/README.md does not split these files: each is
pre-1.54 and the loader depoisons before every walk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from descape import library_compat, scenario_format, scenario_io
from descape.scenario_io import UnsupportedStructureVersion, load_map_and_units

pytestmark = pytest.mark.corpus

_SUFFIXES = {".scn", ".scx"}
_SEPARATOR = b"\x9d\xff\xff\xff"
_TRIGGER_VERSIONS = {1.3, 1.6}  # every loaded file in the 2026-10-28 census
_MAX_TRIGGERS = 100_000


def _legacy_candidates(config: pytest.Config) -> tuple[Path, list[Path]]:
    override = config.getoption("--scenario-dir")
    root = Path(override) if override else Path(__file__).resolve().parent.parent / "examples"
    if not root.is_dir():
        return root, []
    return root, sorted(p for p in root.rglob("*") if p.suffix.lower() in _SUFFIXES and p.is_file())


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "legacy_path" not in metafunc.fixturenames:
        return
    root, files = _legacy_candidates(metafunc.config)
    if not files:
        metafunc.parametrize(
            "legacy_path", [pytest.param(None, marks=pytest.mark.skip(reason="no .scn/.scx files"), id="none")]
        )
        return
    ids = [str(p.relative_to(root)) for p in files]
    metafunc.parametrize("legacy_path", files, ids=ids)


def _load_or_skip(path: Path) -> scenario_io.LoadedScenario:
    fmt = scenario_format.detect_format(path.read_bytes())
    if not fmt.legacy:
        pytest.skip(f"header {fmt.header_version}, not a pre-DE format")
    if library_compat.legacy_structure_path(fmt.data_version) is None:
        # Only a band with no structure skips; a supported band's failure fails.
        with pytest.raises(UnsupportedStructureVersion):
            load_map_and_units(path)
        pytest.skip(f"data {fmt.data_version}: no legacy structure for this band")
    return load_map_and_units(path)


def test_a_legacy_file_loads_with_its_byte_walks_verified(legacy_path: Path) -> None:
    loaded = _load_or_skip(legacy_path)
    assert loaded.structure_source == "legacy" and loaded.source_format.legacy
    assert loaded.terrain_write_supported, "the terrain verifier disagrees with the parse"
    assert loaded.units_write_supported, "the units verifier disagrees with the parse"
    assert loaded.trigger_read_supported is False and scenario_io.parse_triggers(loaded) is None

    tail = loaded.trigger_tail
    assert round(loaded.trigger_version, 2) in _TRIGGER_VERSIONS, loaded.trigger_version
    assert len(tail) >= 13
    count = int.from_bytes(tail[9:13], "little", signed=True)
    assert 0 <= count <= _MAX_TRIGGERS, count

    body = loaded.decompressed_body
    sections = loaded._scenario.sections
    # GlobalVictory opens on the separator, right where PlayerDataTwo ends.
    assert body[loaded.player_data_two_section_end :][:4] == _SEPARATOR
    assert sections["GlobalVictory"].retriever_map["separator"].data == 0xFFFFFF9D
    assert sections["Diplomacy"].retriever_map["separator"].data == 0xFFFFFF9D
    # Map.separator, read off the raw body where Options ends.
    assert body[loaded.options_section_end :][:4] == _SEPARATOR


def test_a_legacy_file_fast_terrain_matches_the_library_walk(legacy_path: Path) -> None:
    from test_scenario_io import _assert_paths_agree

    _load_or_skip(legacy_path)
    _assert_paths_agree(load_map_and_units(legacy_path, fast_terrain=False), load_map_and_units(legacy_path))


def test_a_legacy_file_fast_units_match_the_library_walk(legacy_path: Path, monkeypatch) -> None:
    import test_fast_units

    _load_or_skip(legacy_path)
    spy = test_fast_units._Spy()
    original = scenario_io._load_units_fast

    def wrapped(scenario, data_igen):
        result = original(scenario, data_igen)
        spy.results.append(result)
        return result

    monkeypatch.setattr(scenario_io, "_load_units_fast", wrapped)
    test_fast_units._assert_units_agree(*test_fast_units._load_both(spy, load_map_and_units, legacy_path))
