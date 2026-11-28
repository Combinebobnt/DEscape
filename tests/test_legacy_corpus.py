"""TASK-429's corpus tier: every pre-DE `.scn`/`.scx`/`.scx2` under the scenario dir
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

from descape import legacy_import, library_compat, scenario_format, scenario_io
from descape.scenario_io import UnsupportedStructureVersion, load_map_and_units

pytestmark = pytest.mark.corpus

_SUFFIXES = {".scn", ".scx", ".scx2"}
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
            "legacy_path", [pytest.param(None, marks=pytest.mark.skip(reason="no .scn/.scx/.scx2 files"), id="none")]
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
    if scenario_format.is_aoe1_era(fmt):
        pytest.skip("Age of Empires (1997) band: no triggers, no library walk, view-only (tests/test_aoe1_corpus.py)")
    return load_map_and_units(path)


def test_a_legacy_file_loads_with_its_byte_walks_verified(legacy_path: Path) -> None:
    loaded = _load_or_skip(legacy_path)
    assert loaded.structure_source == "legacy" and loaded.source_format.legacy
    assert loaded.terrain_write_supported, "the terrain verifier disagrees with the parse"
    assert loaded.units_write_supported, "the units verifier disagrees with the parse"
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


def test_a_legacy_files_triggers_read_or_name_their_format(legacy_path: Path) -> None:
    """GH #316: a trigger-1.6 section normalizes with exact consumption and parses to its
    stored trigger count, read-only; any other trigger version stays unread and skips."""
    from descape import legacy_triggers

    loaded = _load_or_skip(legacy_path)
    if loaded.trigger_version != legacy_triggers.TRIGGER_VERSION:
        assert loaded.trigger_read_supported is False and scenario_io.parse_triggers(loaded) is None
        pytest.skip(f"trigger format {loaded.trigger_version:g} isn't read")
    legacy_triggers.normalize(loaded.trigger_tail)  # raises unless the walk ends exactly at the tail's end
    manager = scenario_io.parse_triggers(loaded)
    assert manager is not None
    assert len(manager.triggers) == int.from_bytes(loaded.trigger_tail[9:13], "little", signed=True)
    assert loaded.trigger_write_supported is False


def test_a_legacy_file_fast_terrain_matches_the_library_walk(legacy_path: Path) -> None:
    from test_scenario_io import _assert_paths_agree

    _load_or_skip(legacy_path)
    _assert_paths_agree(load_map_and_units(legacy_path, fast_terrain=False), load_map_and_units(legacy_path))


def test_a_legacy_file_fast_units_match_the_library_walk(legacy_path: Path, monkeypatch) -> None:
    import test_fast_units

    _load_or_skip(legacy_path)
    spy = test_fast_units._Spy()
    original = scenario_io._load_units_fast

    def wrapped(scenario, data_igen, absent):
        result = original(scenario, data_igen, absent)
        spy.results.append(result)
        return result

    monkeypatch.setattr(scenario_io, "_load_units_fast", wrapped)
    test_fast_units._assert_units_agree(*test_fast_units._load_both(spy, load_map_and_units, legacy_path))


def test_a_legacy_file_stores_only_old_tree_variants(legacy_path: Path) -> None:
    """legacy_import.tree_variant()'s premise (TASK-685): every legacy tree stores an integer in 0..13
    whose 3v+2 stays inside its DE graphic's variant count (the corpus max is 3 for bamboo, 8 for the
    pines, 12 for palm). A nonzero count names its const and band, which then stays out of
    MEASURED_BANDS, and an out-of-count value reopens how DEscape draws it."""
    from descape import unit_sprites

    loaded = _load_or_skip(legacy_path)
    graphics = unit_sprites.graphic_map()
    bad: dict[int, int] = {}
    for units in loaded.unit_manager.units:
        for unit in units:
            if unit.unit_const not in legacy_import.TREE_CONSTS:
                continue
            drawn = legacy_import.tree_variant(unit.rotation, 2)  # the run's top value
            if drawn is None or drawn >= graphics[unit.unit_const]["angle_count"]:
                bad[unit.unit_const] = bad.get(unit.unit_const, 0) + 1
    assert not bad, f"{loaded.source_format.game}: {bad}"


def test_a_legacy_file_converts_to_a_writable_de_document(legacy_path: Path) -> None:
    """GH #315: every loaded legacy file converts, reloads with every write gate open,
    and keeps every unit the report does not name as dropped."""
    import random

    from descape import legacy_convert

    loaded = _load_or_skip(legacy_path)
    new, report = legacy_convert.load_converted(loaded, rng=random.Random(0))
    assert not new.source_format.legacy
    assert new.terrain_write_supported and new.units_write_supported and new.messages_write_supported
    before = sum(map(len, loaded.unit_manager.units))
    assert sum(map(len, new.unit_manager.units)) == before - sum(report.dropped_consts.values())
    ids = [u.reference_id for units in new.unit_manager.units for u in units]
    assert len(set(ids)) == len(ids) and min(ids, default=0) >= 0


# -- TASK-520: saving back to the file's own format --------------------------------


def test_a_legacy_file_saves_back_byte_identical_and_its_gates_open(legacy_path: Path) -> None:
    """A no-edit save is the file's own bytes; the Messages walk, the header's Instructions copy and
    (except 1.23's unread team bytes, fail-closed) the Diplomacy grid all verify."""
    from descape import legacy_write, messages_fields
    from descape.messages_model import MessagesEditModel
    from descape.options_model import OptionsEditModel, diplomacy_write_supported
    from descape.scenario_write import build_patched

    loaded = _load_or_skip(legacy_path)
    patched = build_patched(loaded)
    assert (patched.header, patched.body) == (loaded.header_bytes, loaded.decompressed_body)
    assert loaded.messages_write_supported, "the Messages walk does not reconcile"
    start, end = loaded.header_instructions_span
    assert start >= 0, "the header walk does not reconcile"
    # Some files leave the header's copy empty; an Instructions edit then splices the Messages copy alone.
    assert MessagesEditModel(loaded)._header_sync_verified or start == end, "the header's Instructions copy disagrees"
    OptionsEditModel(loaded)
    assert diplomacy_write_supported(loaded) is (loaded.source_format.data_version != "1.23")
    for field_id, raw in messages_fields.raw_payloads(loaded).items():
        text, nul = (raw[:-1], b"\x00") if raw.endswith(b"\x00") else (raw, b"")
        assert legacy_write.encode(legacy_write.decode(text)) + nul == raw, field_id


def _diff_positions(a: bytes, b: bytes) -> list[int]:
    assert len(a) == len(b)
    return [i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y]


@pytest.mark.parametrize("family", ["terrain", "elevation", "messages", "diplomacy"])
def test_a_legacy_file_saves_each_family_back_touching_only_its_own_bytes(legacy_path: Path, family, tmp_path) -> None:
    """Inside _allow_for_oracle(), one synthetic edit per family saves, reloads with every verifier
    true, reads back, keeps header_length right and leaves every byte outside its own spans."""
    import struct

    from descape import diplomacy_fields, legacy_write
    from descape.messages_model import MessagesEditModel
    from descape.options_model import OptionsEditModel, diplomacy_write_supported
    from descape.scenario_write import write_scenario

    loaded = _load_or_skip(legacy_path)
    band = loaded.source_format.data_version
    if band not in legacy_write.ALLOWED:
        pytest.skip(f"data {band}: not a band ALLOWED keys (TASK-675)")
    original = loaded.decompressed_body
    kwargs = {}
    mm = loaded.map_manager
    stride, start, n = loaded.terrain_struct_size, loaded.terrain_block_offset, len(mm.terrain)
    if family == "terrain":
        tile = mm.terrain[n // 2]
        tile.terrain_id = next(t for t in sorted(legacy_write.legal_terrain(loaded)) if t != tile.terrain_id)
        allowed_span = range(start, start + stride * n)
    elif family == "elevation":
        tile = mm.terrain[n // 2]
        tile.elevation = tile.elevation + 1 if tile.elevation < legacy_write.BAND_MAX_ELEVATION[band] else tile.elevation - 1
        allowed_span = range(start, start + stride * n)  # no neighbour check here: bytes only, never opened in-game
    elif family == "messages":
        messages = kwargs["messages"] = MessagesEditModel(loaded)
        messages.set_value("hints", "Résumé – “test” €")
        messages.set_value("instructions", "Corpus round trip, a longer text than before.")
    else:
        if not diplomacy_write_supported(loaded):
            pytest.skip(f"data {band}: Diplomacy fails closed")
        options = kwargs["options"] = OptionsEditModel(loaded)
        options.set_value(diplomacy_fields.stance_cell_id(1, 2), 3 if diplomacy_fields.stance_value(loaded, 1, 2) != 3 else 1)
        cell = diplomacy_fields.stance_offsets(loaded)[diplomacy_fields.stance_cell_id(1, 2)]
        allowed_span = range(cell.offset, cell.offset + cell.length)
    out = tmp_path / f"out{legacy_write.save_suffix(loaded)}"
    with legacy_write._allow_for_oracle(band):
        assert write_scenario(loaded, out, backup=False, **kwargs).wrote
    back = load_map_and_units(out)
    assert back.terrain_write_supported and back.units_write_supported and back.messages_write_supported
    assert struct.unpack_from("<I", back.header_bytes, 4)[0] == len(back.header_bytes) - 8
    body = back.decompressed_body
    if family == "messages":
        model = MessagesEditModel(back)
        assert model.original_value("hints") == "Résumé – “test” €"
        if not messages._header_sync_verified:  # an empty header copy beside a non-empty Messages one
            assert back.header_bytes == loaded.header_bytes, "an unsynced header copy stays as it was"
        else:
            assert model._header_sync_verified and model.original_value("instructions").startswith("Corpus round trip")
        s, e = loaded.messages_section_start, loaded.messages_section_end
        assert body[:s] == original[:s] and body[len(body) - (len(original) - e) :] == original[e:]
        assert back.header_bytes[-8:] == loaded.header_bytes[-8:]
        return
    assert all(i in allowed_span for i in _diff_positions(original, body))
    assert back.header_bytes == loaded.header_bytes
    if family == "diplomacy":
        assert diplomacy_fields.verify_diplomacy_block(back)
        assert diplomacy_fields.stance_value(back, 1, 2) != diplomacy_fields.stance_value(loaded_again := _load_or_skip(legacy_path), 1, 2)
        assert diplomacy_fields.stance_value(back, 2, 1) == diplomacy_fields.stance_value(loaded_again, 2, 1)
    else:
        assert tile_state_of(back, n // 2) == (tile.terrain_id, tile.elevation)


def tile_state_of(loaded, index: int) -> tuple[int, int]:
    tile = loaded.map_manager.terrain[index]
    return tile.terrain_id, tile.elevation
