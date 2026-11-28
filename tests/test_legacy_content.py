"""TASK-430 stage 2: the 1999 game's derived root, built from a synthetic install.

descape/legacy_content.py and drs_archive.py on archives, a string-table
DLL and a palette packed here with literal `struct.pack` calls (D6), and what
the root changes: the content root versus the DE install (D5), Auto with and
without an install (D4) and the viewer's build before activating.
"""

from __future__ import annotations

import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_drs import drs as tools_drs
from test_drs import pack_drs, pack_slp

from descape import asset_source, drs_archive, game_dataset, games, legacy_content

import conftest

PALETTE = [(i, 255 - i, i // 2) for i in range(256)]
PALETTE_TEXT = ("JASC-PAL\r\n0100\r\n256\r\n" + "".join(f"{r} {g} {b}\r\n" for r, g, b in PALETTE)).encode()
STOCK, PATCHED, OTHER = pack_slp([(4, 4, 2, 2)], b"stock"), pack_slp([(5, 5, 2, 2)], b"patched"), pack_slp([(3, 3, 1, 1)])
ARCHER = 5083  # block 318, index 11
LANG_EN, LANG_DE = 0x0409, 0x0407


def pack_pe(blocks: dict[int, list[str]], lang: int = LANG_EN) -> bytes:
    """A PE32 image whose one section holds RT_STRING blocks of up to 16 length-prefixed UTF-16 strings."""
    rva, raw = 0x1000, 0x200
    count = len(blocks)

    def directory(entries):
        return struct.pack("<IIHHHH", 0, 0, 0, 0, 0, len(entries)) + b"".join(struct.pack("<II", *e) for e in entries)

    types_at = 16 + 8
    langs_at = types_at + 16 + 8 * count
    leaves_at = langs_at + 24 * count
    data_at = leaves_at + 16 * count
    payloads = [b"".join(struct.pack("<H", len(s)) + s.encode("utf-16-le") for s in (texts + [""] * 16)[:16]) for texts in blocks.values()]
    section = directory([(6, types_at | 0x80000000)])
    section += directory([(block, (langs_at + 24 * i) | 0x80000000) for i, block in enumerate(blocks)])
    section += b"".join(directory([(lang, leaves_at + 16 * i)]) for i in range(count))
    offset = data_at
    for payload in payloads:
        section += struct.pack("<IIII", rva + offset, len(payload), 0, 0)
        offset += len(payload)
    section += b"".join(payloads)
    coff = struct.pack("<HHIIIHH", 0x14C, 1, 0, 0, 0, 224, 0x2102)
    optional = struct.pack("<H", 0x10B) + bytes(94 + 16) + struct.pack("<II", rva, len(section)) + bytes(104)
    table = b".rsrc\0\0\0" + struct.pack("<IIII", len(section), rva, len(section), raw) + bytes(16)
    head = b"MZ" + bytes(58) + struct.pack("<I", 0x40) + b"PE\0\0" + coff + optional + table
    assert len(head) <= raw
    return head.ljust(raw, b"\0") + section


def _strings(at: dict[int, str]) -> dict[int, list[str]]:
    blocks: dict[int, list[str]] = {}
    for string_id, text in at.items():
        block = blocks.setdefault(string_id // 16 + 1, [""] * 16)
        block[string_id % 16] = text
    return blocks


def make_install(root: Path, lang: int = LANG_EN, palette: bool = True, palette_text: bytes = PALETTE_TEXT) -> Path:
    """A Conquerors install: Graphics.drs in a copied install's own case, one override archive, the rest beside it."""
    data = root / "Data"
    data.mkdir(parents=True)
    pack_drs(data / "Graphics.drs", {b" pls": {2: OTHER, 5168: STOCK}})
    pack_drs(data / "gamedata_x1_p1.drs", {b" pls": {5168: PATCHED}})
    pack_drs(data / "terrain.drs", {b" pls": {15000: b"grass"}})
    pack_drs(data / "interfac.drs", {b"anib": {50500: palette_text} if palette else {50501: b"x"}})
    (data / "empires2_x1_p1.dat").write_bytes(b"read at regen time only")
    (data / "blendomatic.dat").write_bytes(b"blend")
    (root / "language.dll").write_bytes(pack_pe(_strings({ARCHER: "Archer", ARCHER + 1: 'Say "hi"\r\nthere'}), lang))
    (root / "language_x1.dll").write_bytes(pack_pe(_strings({ARCHER: "Archer (x1)"}), lang))
    return root


@pytest.fixture
def install(tmp_path, monkeypatch) -> Path:
    """A valid install, configured through AOC_INSTALL_PATH (conftest unsets it and redirects CACHE_ROOT)."""
    path = make_install(tmp_path / "aoc")
    monkeypatch.setenv(legacy_content.INSTALL_ENV, str(path))
    asset_source.clear_install_caches(None)
    yield path
    asset_source.clear_install_caches(None)


def test_drs_archive_reads_what_tools_drs_reads(tmp_path):
    path = pack_drs(tmp_path / "a.drs", {b" pls": {2: OTHER, 9: STOCK}, b"anib": {50500: PALETTE_TEXT}, b" vaw": {7: b"w"}})
    runtime, tools = drs_archive.DrsArchive(path), tools_drs.DrsArchive(path)
    assert runtime.members == tools.members
    for kind in ("slp", "bina", "wav"):
        assert [runtime.read(kind, i) for i in runtime.ids(kind)] == [tools.read(kind, i) for i in tools.ids(kind)]
    assert drs_archive.AOC_GRAPHICS_ORDER == tools_drs.AOC_GRAPHICS_ORDER
    with pytest.raises(ValueError, match="version"):
        drs_archive.DrsArchive(pack_drs(tmp_path / "v.drs", {b" pls": {}}, version=b"2.00"))


def test_a_root_mirrors_des_layout(tmp_path):
    root = legacy_content.ensure_root(make_install(tmp_path / "aoc"))
    assert root.parent == legacy_content.CACHE_ROOT
    graphics = root / legacy_content.GRAPHICS_SUBPATH
    assert sorted(p.name for p in graphics.iterdir()) == ["slp_2.slp", "slp_5168.slp"]
    assert (graphics / "slp_5168.slp").read_bytes() == PATCHED, "the override archive wins"
    assert (graphics / "slp_2.slp").read_bytes() == OTHER
    assert (root / legacy_content.BLENDOMATIC_SUBPATH).read_bytes() == b"blend"
    palettes = root / legacy_content.PALETTE_SUBPATH
    assert (palettes / legacy_content.PALETTE_NAME).read_bytes() == PALETTE_TEXT
    assert (palettes / legacy_content.PALETTES_CONF).read_text() == f"50500,{legacy_content.PALETTE_NAME}\n"
    ramp = tools_drs.read_jasc_palette((palettes / legacy_content.WHITE_PLAYER_PALETTE).read_bytes())
    assert ramp == [PALETTE[128 + i % 8] for i in range(256)], "the grey player's eight shades, repeated"
    lines = (root / legacy_content.STRINGS_SUBPATH.format(lang="en")).read_text(encoding="utf-8").splitlines()
    assert lines == [f'{ARCHER} "Archer (x1)"', f'{ARCHER + 1} "Say \\"hi\\"\\nthere"'], "a later DLL wins"


def test_a_current_root_is_reused_and_a_changed_install_keys_a_new_one(tmp_path, monkeypatch):
    path = make_install(tmp_path / "aoc")
    builds = []
    real = legacy_content.build_root
    monkeypatch.setattr(legacy_content, "build_root", lambda *a: builds.append(a) or real(*a))
    first = legacy_content.ensure_root(path)
    assert legacy_content.ensure_root(path) == first and len(builds) == 1
    (path / "Data" / "blendomatic.dat").write_bytes(b"a longer blend")
    assert legacy_content.built_root(path) is None, "a changed source file is another key"
    second = legacy_content.ensure_root(path)
    assert second != first and len(builds) == 2 and first.is_dir(), "nothing deletes the old root"
    assert (second / legacy_content.BLENDOMATIC_SUBPATH).read_bytes() == b"a longer blend"


def test_an_install_is_validated_before_anything_builds(tmp_path):
    path = make_install(tmp_path / "aoc")
    assert legacy_content.validate_install(path)[0]
    (path / "Data" / "empires2_x1_p1.dat").rename(path / "Data" / "empires2.dat")
    ok, message = legacy_content.validate_install(path)
    assert not ok and "Age of Kings-only" in message
    (path / "language.dll").unlink()
    ok, message = legacy_content.validate_install(path)
    assert not ok and "language.dll" in message
    assert legacy_content.ensure_root(path) is None and not legacy_content.CACHE_ROOT.exists()
    with pytest.raises(ValueError, match="not a PE"):
        legacy_content.read_string_table(b"not a dll")


def test_the_names_show_in_any_language(tmp_path, monkeypatch):
    """Game.strings_any_language: a German install's one table names units whatever DEscape's language is."""
    path = make_install(tmp_path / "aoc", lang=LANG_DE)
    monkeypatch.setenv(legacy_content.INSTALL_ENV, str(path))
    root = legacy_content.ensure_root()
    assert (root / legacy_content.STRINGS_SUBPATH.format(lang="de")).is_file()
    asset_source.clear_install_caches(None)
    game_dataset.set_active(game_dataset.AOC)
    assert asset_source.resource_string(ARCHER, "en") == asset_source.resource_string(ARCHER, "fr") == "Archer (x1)"
    assert asset_source.resource_string(ARCHER + 1, "de") == 'Say "hi"\\nthere', "a line break as DE's own strings keep it"
    folders = {lang: legacy_content.language_folder(lang) for lang in (0x0404, 0x0C0C, 0x0816, 0x0436)}
    assert folders == {0x0404: "tw", 0x0C0C: "fr", 0x0816: "br", 0x0436: "en"}


def test_the_content_root_is_the_active_games_and_the_de_install_stays_des(install, fake_install, tmp_path):
    """D5: get_install_path() follows the active game; the DE install and its override are DE-only."""
    de = asset_source.get_de_install_path()
    assert de is not None and asset_source.get_install_path() == de
    root = legacy_content.ensure_root()
    game_dataset.set_active(game_dataset.AOC)
    assert asset_source.get_install_path() == root and asset_source.get_de_install_path() == de
    assert game_dataset.install_root(games.AOE2) == de and game_dataset.install_root(games.AOC) == root
    other = tmp_path / "other_de"
    other.mkdir()
    asset_source.set_install_path_override(other)
    assert asset_source.get_de_install_path() == other and asset_source.get_install_path() == root, "the override is DE's"


def _window(scenario):
    from descape import viewer_file

    class Window(viewer_file._ViewerFileMixin):
        pass

    window = Window()
    window.scenario = scenario
    window.logged = []
    window.progress = []
    window._log_status = lambda text, level=None: window.logged.append(text)
    window.load_progress = SimpleNamespace(
        start=lambda owner, caption: window.progress.append(("start", owner)),
        step=lambda owner, percent, caption: window.progress.append(("step", percent)),
        finish=lambda owner: window.progress.append(("finish", owner)),
    )
    return window


def test_auto_shows_a_1999_file_with_its_own_data_once_the_install_is_set(tmp_path, monkeypatch):
    """D4: the stand-in without an install, the 1999 data with one, before its root is even built."""
    window = _window(conftest.legacy_loaded())
    assert window._shown_game_dataset() == "aoe2" and not game_dataset.configured(games.AOC)
    monkeypatch.setenv(legacy_content.INSTALL_ENV, str(make_install(tmp_path / "aoc")))
    asset_source.clear_install_caches(None)
    assert game_dataset.configured(games.AOC) and window._shown_game_dataset() == "aoc"
    assert game_dataset.file_game(window.scenario) is games.AOC
    assert game_dataset.install_root(games.AOC) is None, "configured() never builds"


def test_the_viewer_builds_the_root_before_activating(qapp, install):
    from descape import viewer_game_root

    window = _window(conftest.legacy_loaded())
    assert viewer_game_root.ready_shown_game(window) == "aoc"
    assert legacy_content.built_root(install) is not None
    assert window.progress[0] == ("start", viewer_game_root.PROGRESS_OWNER) and window.progress[-1][0] == "finish"
    assert any("unpacked" in line for line in window.logged), window.logged
    game_dataset.set_active(game_dataset.AOC)
    assert asset_source.get_install_path() == legacy_content.built_root(install)
    assert viewer_game_root.status_lines(window) == [], "no stand-in line, no missing-art line"


def test_a_failed_build_shows_the_file_with_the_stand_ins_data(qapp, tmp_path, monkeypatch):
    from descape import viewer_game_root

    broken = make_install(tmp_path / "aoc", palette=False)
    monkeypatch.setenv(legacy_content.INSTALL_ENV, str(broken))
    asset_source.clear_install_caches(None)
    window = _window(conftest.legacy_loaded())
    assert viewer_game_root.ready_shown_game(window) == "aoe2"
    assert any("Could not unpack" in line for line in window.logged), window.logged
    assert legacy_content.build_failed(broken) and not game_dataset.configured(games.AOC)
    assert any(line.startswith("Shown with Age of Empires II data") for line in viewer_game_root.status_lines(window))
    assert viewer_game_root.ready_shown_game(window) == "aoe2" and window.progress.count(("finish", "game_root")) == 1, "no retry"
    legacy_content.forget_failures()
    asset_source.clear_install_caches(None)
    assert game_dataset.configured(games.AOC), "a changed install setting tries again"


def test_a_1999_game_without_a_root_says_its_art_is_missing(install):
    from descape import viewer_game_root

    window = _window(conftest.legacy_loaded())
    game_dataset.set_active(game_dataset.AOC)
    lines = viewer_game_root.status_lines(window)
    assert any("art is not available" in line for line in lines), lines


def test_the_settings_row_saves_only_a_valid_install_and_clears_it(qapp, tmp_path):
    from PyQt5.QtWidgets import QVBoxLayout, QWidget

    from descape import settings_game_install

    calls = []
    window = SimpleNamespace(
        _log_status=lambda text, level=None: calls.append(text),
        apply_game_data=lambda: calls.append("apply"),
        _refresh_game_assets=lambda: calls.append("refresh"),
    )
    dialog = QWidget()
    dialog._window = window
    row = settings_game_install.InstallRow(dialog, QVBoxLayout(dialog), games.AOC)
    assert row.edit.text() == "" and not row.clear_button.isEnabled()
    row.edit.setText(str(tmp_path / "nowhere"))
    row.load()
    assert legacy_content.install_path() is None and "apply" not in calls and "Not a folder" in row.status.text()
    path = make_install(tmp_path / "aoc")
    row.edit.setText(str(path))
    row.load()
    assert legacy_content.install_path() == path and calls.count("apply") == 1
    assert row.clear_button.isEnabled() and "install found" in row.status.text()
    row.clear()
    assert legacy_content.install_path() is None and calls.count("apply") == 2 and row.edit.text() == ""


def test_a_bad_palette_fails_the_build_before_any_art_is_written(tmp_path):
    """Any error a user's files raise is a RootBuildError, recorded, and leaves no staging folder."""
    short = b"JASC-PAL\r\n0100\r\n16\r\n" + b"1 2 3\r\n" * 16
    path = make_install(tmp_path / "aoc", palette_text=short)
    with pytest.raises(legacy_content.RootBuildError, match="16 colours"):
        legacy_content.ensure_root(path)
    assert legacy_content.build_failed(path)
    assert not any(legacy_content.CACHE_ROOT.glob("*.partial-*")), "nothing staged before the failure"


def test_a_cyclic_or_huge_resource_directory_is_refused():
    pe = bytearray(pack_pe(_strings({ARCHER: "Archer"})))
    struct.pack_into("<II", pe, 0x200 + 16, 6, 0x80000000)  # the type entry points back at the root
    with pytest.raises(ValueError, match="repeats"):
        legacy_content.read_string_table(bytes(pe))
    pe = bytearray(pack_pe(_strings({ARCHER: "Archer"})))
    struct.pack_into("<H", pe, 0x200 + 14, 60000)
    with pytest.raises(ValueError, match="60000 entries"):
        legacy_content.read_string_table(bytes(pe))


def test_a_1999_file_with_its_own_data_reaches_the_family_check(install, tmp_path):
    """Decision A: aoc is view-only, but a pre-DE document's save is legacy_write's call, not the view-only refusal."""
    from descape import legacy_write, scenario_write

    legacy = conftest.legacy_loaded()
    assert game_dataset.file_game(legacy) is games.AOC and not games.AOC.writable
    scenario_write.check_destination(legacy, tmp_path / f"out{legacy_write.save_suffix(legacy)}")
    assert game_dataset.lock_reason("aoc", "aoc", legacy=True) is None


def test_a_script_load_builds_the_root_or_falls_back_to_the_stand_in(install, tmp_path, monkeypatch):
    from descape import batch_api

    legacy = conftest.legacy_loaded()
    batch_api._activate(legacy)
    assert game_dataset.active() == "aoc" and legacy_content.built_root(install) is not None
    game_dataset.set_active(game_dataset.AOE2)
    broken = make_install(tmp_path / "broken", palette=False)
    monkeypatch.setenv(legacy_content.INSTALL_ENV, str(broken))
    asset_source.clear_install_caches(None)
    batch_api._activate(legacy)
    assert game_dataset.active() == "aoe2" and legacy_content.build_failed(broken)


def test_de_only_callers_keep_the_de_install_with_aoc_active(install, fake_install):
    from descape import content_roots

    de_profile, de_roots = content_roots.profile_path(), [r.path for r in content_roots.content_roots()]
    legacy_content.ensure_root()
    game_dataset.set_active(game_dataset.AOC)
    assert content_roots.profile_path() == de_profile
    assert [r.path for r in content_roots.content_roots()] == de_roots
