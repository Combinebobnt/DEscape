"""Default-tier coverage for the v1.21 load-path wiring (Step 6 of the v1.21
structure work): the structure-derived offsets and gates in
descape/scenario_io.py and descape/scenario_write.py that let a file whose
layout differs from DE's load and save without any version-keyed table.

Nothing here needs a v1.21 file: the real 25-file oracle is
tests/test_v121_alignment.py (corpus tier, this machine only). These tests
pin the same derivations against the shipped DE fixtures, where old and new
must agree, and against synthetic v1.21-shaped inputs, where they must not.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from descape import scenario_io
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from descape.scenario_write import _patch_terrain_block

TRIGGERS_FIXTURE = "tests/fixtures/triggers_120x120.aoe2scenario"


def _tile(terrain_id: int, elevation: int, layer: int = -1) -> SimpleNamespace:
    return SimpleNamespace(terrain_id=terrain_id, elevation=elevation, layer=layer)


def _stride3_body(tiles, prefix: bytes = b"\xaa\xbb", unused: int = 0x5C) -> bytes:
    return prefix + b"".join(bytes([t.terrain_id, t.elevation, unused]) for t in tiles) + b"\xee"


class _FakeStructRetriever:
    """Just enough of a struct retriever for scenario_io.retriever_length()."""

    def __init__(self, entries) -> None:
        self.datatype = SimpleNamespace(type="struct")
        self.data = entries


def _fake_map_section(n_tiles: int, entry_length: int, fields: tuple[str, ...]):
    entries = [
        SimpleNamespace(byte_length=entry_length, retriever_map=dict.fromkeys(fields)) for _ in range(n_tiles)
    ]
    return SimpleNamespace(retriever_map={"terrain_data": _FakeStructRetriever(entries)})


# -- 6b: terrain stride and layer presence -----------------------------------


@pytest.mark.parametrize("fixture", [str(BLANK_TEMPLATE_PATH), TRIGGERS_FIXTURE])
def test_de_fixture_reads_a_seven_byte_stride_with_layer(fixture: str) -> None:
    loaded = load_map_and_units(fixture)
    assert loaded.terrain_struct_size == scenario_io.TERRAIN_STRUCT_SIZE == 7
    assert loaded.terrain_has_layer is True
    assert loaded.terrain_write_supported


@pytest.mark.corpus
def test_de_corpus_reads_a_seven_byte_stride_with_layer(scenario_path) -> None:
    """Old (constant 7) equals new (read off the parse) on every examples/ file."""
    loaded = load_map_and_units(scenario_path)
    assert loaded.terrain_struct_size == 7
    assert loaded.terrain_has_layer is True


def test_terrain_layout_reads_stride_and_layer_off_the_parse() -> None:
    offset, stride, has_layer = scenario_io._terrain_layout(
        _fake_map_section(6, 3, ("terrain_id", "elevation", "unused")), 100, 3, 2
    )
    assert (offset, stride, has_layer) == (100 - 18, 3, False)
    offset, stride, has_layer = scenario_io._terrain_layout(
        _fake_map_section(4, 7, ("terrain_id", "elevation", "unused", "layer")), 50, 2, 2
    )
    assert (offset, stride, has_layer) == (50 - 28, 7, True)


def test_terrain_layout_fails_closed_on_a_remainder() -> None:
    """A parsed length that is not a multiple of w*h is a parse that is not
    what it claims: stride 0, never a rounded one."""
    _offset, stride, _layer = scenario_io._terrain_layout(
        _fake_map_section(1, 7, ("terrain_id", "elevation")), 100, 2, 2
    )
    assert stride == 0


def test_verify_terrain_block_walks_a_three_byte_stride() -> None:
    tiles = [_tile(1, 2), _tile(3, 4), _tile(5, 6)]
    body = _stride3_body(tiles)
    assert scenario_io._verify_terrain_block(body, 2, tiles, 3, False)
    # A layer check on a 3-byte struct has no target: refused, not read.
    assert not scenario_io._verify_terrain_block(body, 2, tiles, 3, True)
    # Wrong stride and wrong anchor both fail.
    assert not scenario_io._verify_terrain_block(body, 2, tiles, 7, False)
    assert not scenario_io._verify_terrain_block(body, 1, tiles, 3, False)
    assert not scenario_io._verify_terrain_block(body, 2, tiles, 0, False)


def test_patch_terrain_block_at_stride_three_never_writes_a_layer() -> None:
    tiles = [_tile(1, 2), _tile(3, 4), _tile(5, 6)]
    body = _stride3_body(tiles)
    scenario = SimpleNamespace(
        decompressed_body=body,
        terrain_block_offset=2,
        terrain_struct_size=3,
        terrain_has_layer=False,
        map_manager=SimpleNamespace(terrain=tiles),
    )
    tiles[1].terrain_id, tiles[1].elevation = 9, 1
    patched = _patch_terrain_block(scenario)
    assert len(patched) == len(body)
    expected = bytearray(body)
    expected[2 + 3 : 2 + 5] = bytes([9, 1])
    # The unused byte, the neighbouring tiles, and both ends are untouched:
    # the in-memory layer -1 never reached disk.
    assert patched == bytes(expected)


def test_patch_terrain_block_at_stride_seven_still_writes_layer() -> None:
    loaded = load_map_and_units(TRIGGERS_FIXTURE)
    tile = loaded.map_manager.terrain[5]
    tile.layer = 42
    patched = _patch_terrain_block(loaded)
    o = loaded.terrain_block_offset + 7 * 5 + 5
    assert scenario_io._LAYER_STRUCT.unpack_from(patched, o)[0] == 42


# -- 6c: forward-derived units offset, and the trailer -----------------------


class _FakeFixedRetriever:
    def __init__(self, length: int) -> None:
        self.datatype = SimpleNamespace(type="u32")
        self._length = length

    def get_data_as_bytes(self) -> bytes:
        return b"\0" * self._length


def _fake_units_section(order: tuple[str, ...], lengths: dict[str, int]):
    return SimpleNamespace(retriever_map={name: _FakeFixedRetriever(lengths[name]) for name in order})


_UNITS_LENGTHS = {
    "number_of_unit_sections": 4,
    "player_data_4": 40,
    "players_units": 100,
    "number_of_players": 4,
    "player_data_3": 30,
}


def _backward_units_block_offset(loaded) -> int:
    players_units = loaded._scenario.sections["Units"].retriever_map["players_units"].data
    return loaded.units_section_end - sum(s.byte_length for s in players_units)


@pytest.mark.parametrize("fixture", [str(BLANK_TEMPLATE_PATH), TRIGGERS_FIXTURE, "tests/fixtures/units_120x120.aoe2scenario"])
def test_forward_units_walk_equals_the_old_backward_derivation(fixture: str) -> None:
    loaded = load_map_and_units(fixture)
    assert loaded.units_write_supported
    assert loaded.units_block_offset == _backward_units_block_offset(loaded)
    assert loaded.players_units_end == loaded.units_section_end


@pytest.mark.corpus
def test_forward_units_walk_equals_the_old_backward_derivation_corpus(scenario_path) -> None:
    loaded = load_map_and_units(scenario_path)
    if not loaded.units_write_supported:
        pytest.skip(f"{scenario_path.name}: units block not trusted")
    assert loaded.units_block_offset == _backward_units_block_offset(loaded)
    assert loaded.players_units_end == loaded.units_section_end


def test_units_layout_on_a_de_ordered_section_has_no_trailer() -> None:
    order = ("number_of_unit_sections", "player_data_4", "number_of_players", "player_data_3", "players_units")
    assert scenario_io._units_layout(_fake_units_section(order, _UNITS_LENGTHS), 1000) == (1078, 1178, 0)


def test_units_layout_on_a_v121_ordered_section_carries_the_trailer() -> None:
    order = ("number_of_unit_sections", "player_data_4", "players_units", "number_of_players", "player_data_3")
    assert scenario_io._units_layout(_fake_units_section(order, _UNITS_LENGTHS), 1000) == (1044, 1144, 34)


def _fake_players_units(counts):
    return [
        SimpleNamespace(retriever_map={"unit_count": SimpleNamespace(data=c)}, byte_length=4 + 10 * c)
        for c in counts
    ]


def test_verify_units_block_reconciles_the_whole_section() -> None:
    counts = (0, 2, 1)
    array = b"".join(c.to_bytes(4, "little") + b"u" * 10 * c for c in counts)
    body = b"HEAD" + array + b"TRAILER" + b"triggers"
    pu = _fake_players_units(counts)
    section_end = 4 + len(array) + 7
    assert scenario_io._verify_units_block(body, 4, pu, 7, section_end)
    # The old check only asked for o <= len(body); a wrong trailer length or
    # section end must now fail.
    assert not scenario_io._verify_units_block(body, 4, pu, 0, section_end)
    assert not scenario_io._verify_units_block(body, 4, pu, 7, section_end + 1)
    assert not scenario_io._verify_units_block(body, 3, pu, 7, section_end)


def test_assemble_body_keeps_the_units_trailer() -> None:
    from descape.scenario_write import _assemble_body

    base = b"HEAD" + b"OLDARRAY" + b"TRAILER" + b"triggers"
    scenario = SimpleNamespace(
        decompressed_body=base,
        units_block_offset=4,
        players_units_end=12,
        units_section_end=19,
    )
    units = SimpleNamespace(has_edits=True, serialize=lambda: b"NEW")
    assert _assemble_body(scenario, base, units, None) == b"HEAD" + b"NEW" + b"TRAILER" + b"triggers"
    # DE shape: empty trailer, so the splice is exactly what it was before.
    scenario.players_units_end = scenario.units_section_end = 12
    base_de = b"HEAD" + b"OLDARRAY" + b"triggers"
    scenario.decompressed_body = base_de
    assert _assemble_body(scenario, base_de, units, None) == b"HEAD" + b"NEW" + b"triggers"


# -- 6d: header span walk -----------------------------------------------------


class _FakeBytesRetriever:
    def __init__(self, raw: bytes) -> None:
        self.datatype = SimpleNamespace(type="u32")
        self._raw = raw

    def get_data_as_bytes(self) -> bytes:
        return self._raw


def _hand_packed_v121_header(with_individual_victories: bool = True):
    """v1.21's 11 FileHeader fields, packed literally here rather than
    through the library, so the test is independent of the code it checks."""
    import struct

    instructions = b"hello"
    fields = [
        ("version", b"1.21"),
        ("header_length", struct.pack("<I", 0)),
        ("savable", struct.pack("<i", 0)),
        ("timestamp_of_last_save", struct.pack("<I", 0)),
        ("scenario_instructions", struct.pack("<I", len(instructions)) + instructions),
        ("individual_victories_used", struct.pack("<I", 0)),
        ("player_count", struct.pack("<I", 4)),
        ("unknown_value", struct.pack("<I", 1000)),
        ("unknown_value_2", struct.pack("<I", 1)),
        ("amount_of_unknown_numbers", struct.pack("<I", 2)),
        ("unknown_numbers", struct.pack("<II", 7, 8)),
    ]
    if not with_individual_victories:
        fields = [f for f in fields if f[0] != "individual_victories_used"]
    header = b"".join(raw for _name, raw in fields)
    return header, {name: _FakeBytesRetriever(raw) for name, raw in fields}


def test_header_walk_on_v121_degrades_to_empty_without_raising() -> None:
    """The real v1.21 header: no creator_name/trigger_count, and an
    individual_victories_used the walk does not know, so it cannot reconcile.
    Header editing is off for v1.21, by design, not a crash."""
    header, retriever_map = _hand_packed_v121_header()
    assert len(retriever_map) == 11
    assert "creator_name" not in retriever_map and "trigger_count" not in retriever_map
    assert scenario_io._header_field_spans(header, retriever_map) == {}
    assert scenario_io._header_instructions_span(header, retriever_map) == (-1, -1)
    assert scenario_io._header_player_count_span(header, retriever_map) == (-1, -1)


def test_header_walk_skips_absent_names_and_still_reconciles() -> None:
    """Without the unknown field the same walk reconciles: absent names are
    skipped rather than indexed, which is what used to raise."""
    header, retriever_map = _hand_packed_v121_header(with_individual_victories=False)
    spans = scenario_io._header_field_spans(header, retriever_map)
    assert "creator_name" not in spans and "trigger_count" not in spans
    start, end = spans["scenario_instructions"]
    assert header[start:end] == b"hello"
    start, end = spans["player_count"]
    assert header[start:end] == (4).to_bytes(4, "little")


def test_header_walk_refuses_a_str32_prefix_past_the_end() -> None:
    header, retriever_map = _hand_packed_v121_header(with_individual_victories=False)
    assert scenario_io._header_field_spans(header[:14], retriever_map) == {}


# -- 6e: trigger-count patches ------------------------------------------------


@pytest.mark.parametrize("fixture", [str(BLANK_TEMPLATE_PATH), TRIGGERS_FIXTURE])
def test_de_fixture_has_trigger_counters(fixture: str) -> None:
    assert load_map_and_units(fixture).has_trigger_counters is True


@pytest.mark.corpus
def test_de_corpus_has_trigger_counters(scenario_path) -> None:
    assert load_map_and_units(scenario_path).has_trigger_counters is True


def _without_counters():
    import dataclasses

    return dataclasses.replace(load_map_and_units(TRIGGERS_FIXTURE), has_trigger_counters=False)


def test_trigger_count_patches_raise_without_counters() -> None:
    from descape.scenario_write import WriteBlockedError, _patch_header_trigger_count, _patch_trigger_count

    loaded = _without_counters()
    with pytest.raises(WriteBlockedError):
        _patch_trigger_count(loaded.decompressed_body, loaded, 3)
    with pytest.raises(WriteBlockedError):
        _patch_header_trigger_count(loaded.header_bytes, loaded, 3)


def test_trigger_count_patches_still_write_with_counters() -> None:
    import struct

    from descape.scenario_write import _patch_header_trigger_count, _patch_trigger_count

    loaded = load_map_and_units(TRIGGERS_FIXTURE)
    body = _patch_trigger_count(loaded.decompressed_body, loaded, 77)
    assert struct.unpack_from("<I", body, loaded.options_section_end - 4)[0] == 77
    header = _patch_header_trigger_count(loaded.header_bytes, loaded, 78)
    assert struct.unpack_from("<I", header, len(header) - 4)[0] == 78


def test_patch_options_reservation_is_zero_width_without_counters() -> None:
    """On v1.21 the 4 bytes ending at options_section_end are player 16's
    starting age, not a trigger counter: a patch there is ordinary content."""
    from descape.scenario_write import WriteBlockedError, _patch_options

    loaded = _without_counters()
    at = loaded.options_section_end - 4
    options = SimpleNamespace(serialize_patches=lambda: [(at, b"\x07\x00\x00\x00")])
    patched = _patch_options(loaded.decompressed_body, loaded, options)
    assert patched[at : at + 4] == b"\x07\x00\x00\x00"

    with_counters = load_map_and_units(TRIGGERS_FIXTURE)
    with pytest.raises(WriteBlockedError):
        _patch_options(with_counters.decompressed_body, with_counters, options)


# -- 6f: structure resolution and the trigger guard ---------------------------


def test_structure_is_available_from_library_or_repo() -> None:
    assert scenario_io.structure_is_available("1.58")
    assert scenario_io.structure_is_available("1.21")
    assert not scenario_io.structure_is_available("9.99")
    assert scenario_io._repo_structure_path("1.58") is None
    assert scenario_io._repo_structure_path("1.21") == scenario_io.REPO_VERSIONS_DIR / "v1.21" / "structure.json"


def test_library_structure_wins_over_a_repo_one(tmp_path, monkeypatch) -> None:
    """A repo file for a version the library also ships is never read: this
    one is not even a valid structure, so reading it would fail the load."""
    (tmp_path / "v1.58").mkdir()
    (tmp_path / "v1.58" / "structure.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(scenario_io, "REPO_VERSIONS_DIR", tmp_path)
    assert scenario_io._repo_structure_path("1.58") is None
    assert load_map_and_units(TRIGGERS_FIXTURE).structure_source == "library"


def _load_via_repo_structure(tmp_path, monkeypatch):
    """TRIGGERS_FIXTURE (1.58) loaded as if the library shipped nothing for it
    and this repo did: the library's own structure copied into a fake repo
    dir. The v1.21 path, minus bytes this repo can't commit."""
    import shutil

    from descape import library_compat

    repo = tmp_path / "repo"
    (repo / "v1.58").mkdir(parents=True)
    shutil.copy(library_compat.VERSIONS_DIR / "v1.58" / "structure.json", repo / "v1.58" / "structure.json")
    monkeypatch.setattr(library_compat, "VERSIONS_DIR", tmp_path / "no_library")
    monkeypatch.setattr(scenario_io, "REPO_VERSIONS_DIR", repo)

    def _no_vocabulary(*_args):
        raise AssertionError("repo-structure load initialised a trigger vocabulary")

    monkeypatch.setattr(scenario_io, "_initialise_version_dependencies", _no_vocabulary)
    return load_map_and_units(TRIGGERS_FIXTURE)


def test_repo_structure_branch_loads_and_saves_byte_identically(tmp_path, monkeypatch) -> None:
    from pathlib import Path

    from descape.scenario_write import write_scenario

    loaded = _load_via_repo_structure(tmp_path, monkeypatch)
    assert loaded.structure_source == "repo"
    assert loaded.terrain_write_supported and loaded.units_write_supported
    out = tmp_path / "same.aoe2scenario"
    write_scenario(loaded, out, backup=False)
    assert out.read_bytes() == Path(TRIGGERS_FIXTURE).read_bytes()


def test_parse_triggers_refuses_a_repo_structure_file_without_parsing(tmp_path, monkeypatch) -> None:
    loaded = _load_via_repo_structure(tmp_path, monkeypatch)

    def _no_parse(*_args, **_kwargs):
        raise AssertionError("parse_triggers() parsed a repo-structure file")

    monkeypatch.setattr(scenario_io.TriggerManager, "construct", _no_parse)
    monkeypatch.setattr(loaded._scenario, "_create_and_load_section", _no_parse)
    assert scenario_io.parse_triggers(loaded) is None
    assert loaded.trigger_read_supported is False
    assert loaded._trigger_manager is None


def test_trigger_panel_explains_why_triggers_are_unreadable() -> None:
    """One text per cause: the 1.54/3.9 re-save advice would be false about a
    v1.21 file, whose map and units come from this repo's own structure."""
    import dataclasses

    conftest = pytest.importorskip("conftest")
    if not conftest.PYQT5_AVAILABLE:
        pytest.skip("PyQt5 not importable")
    window = conftest.shown_window(1500, 900)
    try:
        window.load_scenario(BLANK_TEMPLATE_PATH)
        window.mode_combo.setCurrentText("Triggers")
        panel = window.trigger_panel

        panel.show_scenario(dataclasses.replace(
                window.scenario, structure_source="repo", scenario_version="1.21", _trigger_manager=None
            ))
        text = panel.status.text()
        assert "Scenario version 1.21 is older" in text and "unit editing" in text
        assert "1.54" not in text and "re-save" not in text

        panel.show_scenario(dataclasses.replace(window.scenario, trigger_read_supported=False, _trigger_manager=None))
        assert panel.status.text() == panel._UNSUPPORTED["library"]
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- steps 0/7: naming the cause for a version with no structure at all -------


def test_unsupported_version_sentence_measures_older_against_the_library() -> None:
    """The older-than-everything phrasing is only used when it is true: 1.32
    is below everything AoE2ScenarioParser ships, a future version is not."""
    assert "older than any version" in scenario_io.unsupported_version_sentence("1.32")
    assert "1.32" in scenario_io.unsupported_version_sentence("1.32")
    newer = scenario_io.unsupported_version_sentence("9.99")
    assert "older" not in newer and "9.99 is not a version" in newer
    # A non-numeric version (viewer.py's File > New sentinel) must not raise.
    assert "older" not in scenario_io.unsupported_version_sentence("untitled")


def test_unsupported_structure_message_names_the_repo_version() -> None:
    text = scenario_io.unsupported_structure_message("1.35")
    assert "1.35 is older" in text and "version 1.21, but not for this one" in text
    assert "UnknownScenarioStructureError" not in text and ":(" not in text


def test_load_raises_the_named_error_when_neither_side_ships_a_structure(tmp_path, monkeypatch) -> None:
    """TRIGGERS_FIXTURE (1.58) with both structure directories emptied: the
    v1.32/v1.35 situation, without bytes this repo can't commit."""
    from descape import library_compat

    monkeypatch.setattr(library_compat, "VERSIONS_DIR", tmp_path / "no_library")
    monkeypatch.setattr(scenario_io, "REPO_VERSIONS_DIR", tmp_path / "no_repo")

    def _no_structure_load(*_args, **_kwargs):
        raise AssertionError("_load_structure() ran for an unsupported version")

    monkeypatch.setattr(scenario_io.AoE2DEScenario, "_load_structure", _no_structure_load)
    with pytest.raises(scenario_io.UnsupportedStructureVersion) as excinfo:
        load_map_and_units(TRIGGERS_FIXTURE)
    assert excinfo.value.scenario_version == "1.58"
    assert "DEscape can't open this file." in str(excinfo.value)


def test_load_failure_modal_explains_an_unsupported_version() -> None:
    """The modal names the cause instead of showing the raw library text, in
    the same wording the trigger panel uses for a repo-structure file."""
    from pathlib import Path

    conftest = pytest.importorskip("conftest")
    if not conftest.PYQT5_AVAILABLE:
        pytest.skip("PyQt5 not importable")
    from descape import viewer as viewer_module

    shown: list[tuple[str, str]] = []
    window = conftest.shown_window(1200, 800)
    try:
        window.load_scenario(BLANK_TEMPLATE_PATH)
        open_document = window.scenario
        real_critical = viewer_module.QMessageBox.critical

        def _capture(_parent, title, text, *_args, **_kwargs):
            shown.append((title, text))

        viewer_module.QMessageBox.critical = staticmethod(_capture)
        real_load = viewer_module.load_map_and_units

        def _unsupported(_path):
            raise scenario_io.UnsupportedStructureVersion("1.32")

        viewer_module.load_map_and_units = _unsupported
        try:
            window.load_scenario(Path("tests/fixtures/does_not_matter.aoe2scenario"))
        finally:
            viewer_module.QMessageBox.critical = real_critical
            viewer_module.load_map_and_units = real_load

        assert len(shown) == 1
        title, text = shown[0]
        assert title == "Failed to load"
        assert "Scenario version 1.32 is older" in text
        assert "UnknownScenarioStructureError" not in text and ":(" not in text
        assert window.scenario is open_document  # the open document survives
    finally:
        window.edit_history.mark_saved()
        window.close()
