"""The shipped blank template, descape/templates/blank_120x120.aoe2scenario,
and the tool that generates it (tools/gen_blank_template.py).

File > New Map splices every new map from that one file, so its scenario
version is every new map's. It sat at 1.58 after 1.59 support landed, and
everything gated on 1.59 (Capture Flag) was off in a new map. Three things
keep that from recurring:

- The oracle. tests/fixtures/real_blank_120x120_v159.aoe2scenario is a blank
  120x120 map saved by the 1.59 game editor. The template must equal it
  except in three fields a template inherits anyway. It is a byte oracle:
  never regenerate it, only replace it with a newer game save.
- The negative control. The same comparison on the 1.58 donor fails and names
  the three version carriers, which is the proof the oracle test can fail.
- The tripwire. The template's version must be the newest one DEscape has a
  trigger vocabulary for, so a library pin that brings a new version fails
  here until the template is refreshed.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

from descape import library_compat, scenario_io
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from descape.scenario_new import load_blank_scenario
from descape.scenario_write import write_scenario
from descape.unit_model import UnitEditModel
from testkit.zlib_runtime import BYTES_DIFFER, NOT_STOCK_REASON, STOCK_ZLIB, scenario_parts

import conftest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ORACLE_PATH = FIXTURES / "real_blank_120x120_v159.aoe2scenario"
V158_DONOR_PATH = FIXTURES / "blank_120x120_v158.aoe2scenario"

_CAMERA_X = "Units.player_data_3[0].editor_camera_x"
_CAMERA_Y = "Units.player_data_3[0].editor_camera_y"
# What a template inherits from its own save rather than from the format.
INHERITED_FIELDS = {"FileHeader.timestamp_of_last_save", "DataHeader.filename", _CAMERA_X, _CAMERA_Y}
VERSION_CARRIERS = {"FileHeader.version", "DataHeader.version", "Triggers.trigger_version"}
_TIMESTAMP = slice(12, 16)  # FileHeader.timestamp_of_last_save, after three fixed-width fields
_TRIGGER_VERSION_SIZE = struct.calcsize("<d")

_ARCHER = 4


def _gen():
    return conftest.load_verify_module("gen_blank_template")


def _fields(node, prefix: str):
    """(path, value) for every retriever under a section or struct entry."""
    for name, retriever in node.retriever_map.items():
        path = f"{prefix}.{name}"
        if retriever.datatype.type != "struct":
            yield path, retriever.data
            continue
        entries = retriever.data or []
        yield f"{path} (count)", len(entries)
        for i, entry in enumerate(entries):
            yield from _fields(entry, f"{path}[{i}]")


def _parsed_fields(loaded) -> dict:
    return {path: value for name, section in loaded._scenario.sections.items() for path, value in _fields(section, name)}


def _terrain_block(loaded) -> bytes:
    mm = loaded.map_manager
    start = loaded.terrain_block_offset
    return loaded.decompressed_body[start : start + loaded.terrain_struct_size * mm.map_width * mm.map_height]


def _differences(candidate, oracle) -> set[str]:
    """Where `candidate` differs from `oracle`: parsed fields by path, plus
    the byte ranges the field walk does not reach (the fast terrain path
    leaves terrain_data empty, and nothing after Units is parsed at load)."""
    ours, theirs = _parsed_fields(candidate), _parsed_fields(oracle)
    assert len(ours) > 600, "the field walk found almost nothing to compare"
    found = {path for path in ours.keys() | theirs.keys() if path not in ours or path not in theirs or ours[path] != theirs[path]}
    if _terrain_block(candidate) != _terrain_block(oracle):
        found.add("terrain block")
    our_tail = candidate.decompressed_body[candidate.units_section_end :]
    their_tail = oracle.decompressed_body[oracle.units_section_end :]
    if our_tail[:_TRIGGER_VERSION_SIZE] != their_tail[:_TRIGGER_VERSION_SIZE]:
        found.add("Triggers.trigger_version")
    if our_tail[_TRIGGER_VERSION_SIZE:] != their_tail[_TRIGGER_VERSION_SIZE:]:
        found.add("bytes after the trigger version")
    return found


def _assert_matches_the_oracle(path: Path) -> None:
    """No literal body offsets: the oracle's filename is longer, so everything after it sits 8 bytes later."""
    candidate = load_map_and_units(path, fast_units=False)
    oracle = load_map_and_units(ORACLE_PATH, fast_units=False)
    wrong = _differences(candidate, oracle) ^ INHERITED_FIELDS
    assert not wrong, f"{path.name} against the game-saved oracle: unexpected or missing differences in {sorted(wrong)}"
    assert candidate.header_bytes[: _TIMESTAMP.start] == oracle.header_bytes[: _TIMESTAMP.start]
    assert candidate.header_bytes[_TIMESTAMP.stop :] == oracle.header_bytes[_TIMESTAMP.stop :]


# -- the generator -------------------------------------------------------------


def test_the_template_matches_its_generator() -> None:
    gen = _gen()
    built = gen.build_template_bytes()
    assert scenario_parts(built) == scenario_parts(BLANK_TEMPLATE_PATH.read_bytes())
    assert gen.build_template_bytes() == built, "two builds differ"
    gen.verify_template(built)


@pytest.mark.skipif(not STOCK_ZLIB, reason=NOT_STOCK_REASON)
def test_the_template_matches_its_generator_byte_identical() -> None:
    assert _gen().build_template_bytes() == BLANK_TEMPLATE_PATH.read_bytes(), BYTES_DIFFER


def test_the_generator_refuses_a_source_that_is_not_the_1_58_donor() -> None:
    """Each carrier must read the source value first: the shipped template is already past it."""
    gen = _gen()
    with pytest.raises(gen.GenerationVerificationError, match=r"FileHeader\.version"):
        gen.build_template_bytes(BLANK_TEMPLATE_PATH)


def test_the_generators_verify_refuses_a_file_at_another_version() -> None:
    gen = _gen()
    with pytest.raises(gen.GenerationVerificationError, match=r"reloaded as 1\.58"):
        gen.verify_template(V158_DONOR_PATH.read_bytes())


def _run_the_generator(tmp_path: Path, monkeypatch) -> Path:
    out = tmp_path / "templates" / "blank.aoe2scenario"
    monkeypatch.setattr(sys, "argv", ["gen_blank_template", "--out", str(out)])
    _gen().main()
    return out


def test_the_generator_writes_the_template_and_leaves_no_temp_file(tmp_path: Path, monkeypatch) -> None:
    out = _run_the_generator(tmp_path, monkeypatch)
    assert scenario_parts(out.read_bytes()) == scenario_parts(BLANK_TEMPLATE_PATH.read_bytes())
    assert [p.name for p in out.parent.iterdir()] == [out.name]


@pytest.mark.skipif(not STOCK_ZLIB, reason=NOT_STOCK_REASON)
def test_the_generator_writes_the_template_and_leaves_no_temp_file_byte_identical(tmp_path: Path, monkeypatch) -> None:
    assert _run_the_generator(tmp_path, monkeypatch).read_bytes() == BLANK_TEMPLATE_PATH.read_bytes(), BYTES_DIFFER


# -- against a game-saved 1.59 blank -------------------------------------------


def test_the_template_equals_a_game_saved_1_59_blank_except_inherited_fields() -> None:
    _assert_matches_the_oracle(BLANK_TEMPLATE_PATH)


def test_the_oracle_really_differs_where_the_template_inherits() -> None:
    """A replacement oracle saved with a centred camera would shrink the
    inherited set silently; this makes that a deliberate edit here."""
    oracle = _parsed_fields(load_map_and_units(ORACLE_PATH, fast_units=False))
    template = _parsed_fields(load_map_and_units(BLANK_TEMPLATE_PATH, fast_units=False))
    assert (template[_CAMERA_X], template[_CAMERA_Y]) == (60.0, 60.0)
    assert (oracle[_CAMERA_X], oracle[_CAMERA_Y]) == pytest.approx((49.61, 70.39), abs=0.005)
    assert (template["DataHeader.filename"], oracle["DataHeader.filename"]) == (
        "blank120.aoe2scenario",
        "gh126_a_baseline.aoe2scenario",
    )


def test_the_oracle_comparison_fails_on_the_1_58_donor_and_names_the_version_carriers() -> None:
    """The permanent negative control: the file the template used to be."""
    with pytest.raises(AssertionError) as caught:
        _assert_matches_the_oracle(V158_DONOR_PATH)
    message = str(caught.value)
    for carrier in sorted(VERSION_CARRIERS):
        assert carrier in message, message
    donor = load_map_and_units(V158_DONOR_PATH, fast_units=False)
    oracle = load_map_and_units(ORACLE_PATH, fast_units=False)
    assert _differences(donor, oracle) == INHERITED_FIELDS | VERSION_CARRIERS


# -- the tripwire ---------------------------------------------------------------


def test_the_template_is_the_newest_supported_scenario_version() -> None:
    template = load_map_and_units(BLANK_TEMPLATE_PATH).scenario_version
    newest = library_compat.vocabulary_versions()[-1]
    assert template == newest, (
        f"blank template is {template}, the newest supported scenario version is {newest}: "
        "refresh it with tools/gen_blank_template.py"
    )


# -- the symptom: a new map takes the 1.59-only fields --------------------------


def test_a_new_map_stores_a_capture_flag(tmp_path: Path) -> None:
    gen = _gen()
    loaded = load_blank_scenario(120)
    assert (loaded.scenario_version, loaded.trigger_version) == (gen.SCENARIO_VERSION, gen.TRIGGER_VERSION)
    assert scenario_io.unit_field_supported(loaded, "capture_flag")

    model = UnitEditModel(loaded)
    archer = model.add(player=1, unit_const=_ARCHER, x=10.5, y=10.5)
    model.set_capture_flag(archer, 2)
    out = tmp_path / "new.aoe2scenario"
    write_scenario(loaded, out, backup=False, units=model)

    # The library walk: read_capture_flags() reads the parsed slots.
    reloaded = load_map_and_units(out, fast_units=False)
    assert reloaded.scenario_version == gen.SCENARIO_VERSION
    assert conftest.load_verify_module("gen_v159_fixture").read_capture_flags(reloaded) == {archer.reference_id: 2}


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_file_new_map_enables_capture_flag_on_a_placed_unit() -> None:
    conftest.ensure_qapp()
    from PyQt5.QtCore import QPointF, Qt

    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    try:
        window.new_action.trigger()
        assert window.scenario is not None
        window.iso_action.setChecked(False)
        window.terrain_style_combo.setCurrentText("Flat")
        window.mode_combo.setCurrentText("Units")
        window.units_panel.select_object(_ARCHER)
        assert window.units_panel.select_owner(1)
        window.place_unit_action.setChecked(True)
        tp = window.map_view._tile_pixels
        window.on_unit_place(QPointF(10.5 * tp, 10.5 * tp), Qt.NoModifier)

        (key,) = window._selection
        archer = window.map_view._unit_index.entry_for_key(key).unit
        combo = window.units_panel.unit_field_editors["capture_flag"]
        assert combo.isEnabled()
        combo.setCurrentIndex(combo.findData(1))
        assert archer.capture_flag == 1
    finally:
        window.edit_history.mark_saved()
        window.close()
