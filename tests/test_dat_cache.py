"""GH #58: the data-mod table cache, its key, its refusals and the shared derivation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

from descape import dat_cache, dat_derive

DESCAPE = Path(dat_derive.__file__).resolve().parent


@pytest.fixture()
def install(tmp_path) -> Path:
    root = tmp_path / "install"
    (root / dat_cache.DAT_SUBPATH).parent.mkdir(parents=True)
    (root / dat_cache.DAT_SUBPATH).write_bytes(b"install dat")
    palette = root / "resources" / "_common" / "palettes" / "original.pal"
    palette.parent.mkdir(parents=True)
    palette.write_text("JASC-PAL")
    return root


@pytest.fixture()
def mod_dat(tmp_path) -> Path:
    path = tmp_path / "mods" / "local" / "M" / "resources" / "_common" / "dat" / "empires2_x2_p1.dat"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"mod dat")
    return path


# What fake_derive's derive() returns: one lenient guard's warning (GH #262).
FAKE_WARNINGS = ["gen_unit_graphic_map.py: unit_const 45 broke a rule (drawn alone)"]


@pytest.fixture()
def fake_derive(monkeypatch):
    """derive() writes every table, the .dat parse yields fixed id sets; counts calls."""
    calls = []

    def derive(dat, install, out_dir, *, tools=None, parsed=None):
        calls.append(dat)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in dat_derive.TABLE_NAMES:
            (out_dir / name).write_text("{}")
        return list(FAKE_WARNINGS)

    def ids(dat, parsed=None):
        return {1, 2, 2750} if Path(dat).name == "empires2_x2_p1.dat" and "mods" in Path(dat).parts else {1, 2}

    monkeypatch.setattr(dat_derive, "derive", derive)
    monkeypatch.setattr(dat_derive, "parse_unit_ids", ids)
    return calls


def _key(mod_dat, install) -> str:
    return dat_cache.entry_key(dat_cache.entry_inputs(mod_dat, install))


def test_a_build_is_then_a_hit(tmp_path, install, mod_dat, fake_derive):
    key = _key(mod_dat, install)
    assert dat_cache.lookup(key, tmp_path / "cache") is None
    entry = dat_cache.build_entry(mod_dat, install, tmp_path / "cache" / key)
    assert entry.mod_only_ids == {2750}
    assert dat_cache.lookup(key, tmp_path / "cache") == entry
    assert sorted(p.name for p in (tmp_path / "cache").iterdir()) == sorted([dat_cache.INSTALL_IDS_DIR, key]), "no staging left"
    meta = json.loads((entry.path / dat_cache.META_NAME).read_text())
    assert meta["mod_dat"] == str(mod_dat) and meta["source_digest"] == dat_derive.source_digest()


def test_the_derive_warnings_land_in_meta_and_the_entry(tmp_path, install, mod_dat, fake_derive):
    key = _key(mod_dat, install)
    entry = dat_cache.build_entry(mod_dat, install, tmp_path / "cache" / key)
    assert entry.warnings == tuple(FAKE_WARNINGS)
    assert json.loads((entry.path / dat_cache.META_NAME).read_text())["warnings"] == FAKE_WARNINGS
    assert dat_cache.lookup(key, tmp_path / "cache").warnings == tuple(FAKE_WARNINGS)


@pytest.mark.parametrize("warnings", [None, "not a list"], ids=["absent", "not_a_list"])
def test_an_entry_without_a_warnings_list_reads_as_none(tmp_path, install, mod_dat, fake_derive, warnings):
    key = _key(mod_dat, install)
    entry = dat_cache.build_entry(mod_dat, install, tmp_path / "cache" / key)
    meta = json.loads((entry.path / dat_cache.META_NAME).read_text())
    del meta["warnings"]
    if warnings is not None:
        meta["warnings"] = warnings
    (entry.path / dat_cache.META_NAME).write_text(json.dumps(meta))
    found = dat_cache.lookup(key, tmp_path / "cache")
    assert found is not None and found.warnings == ()


def test_an_entry_without_meta_is_a_miss(tmp_path, install, mod_dat, fake_derive):
    key = _key(mod_dat, install)
    entry = dat_cache.build_entry(mod_dat, install, tmp_path / "cache" / key)
    os.replace(entry.path / dat_cache.META_NAME, tmp_path / "moved")
    assert dat_cache.lookup(key, tmp_path / "cache") is None


def test_an_unreadable_dat_fails_before_any_folder_exists(tmp_path, install, mod_dat, fake_derive, monkeypatch):
    def unreadable(dat, parsed=None):
        raise AssertionError("genieutils could not read it")

    monkeypatch.setattr(dat_derive, "parse_unit_ids", unreadable)
    with pytest.raises(AssertionError):
        dat_cache.build_entry(mod_dat, install, tmp_path / "cache" / _key(mod_dat, install))
    assert not (tmp_path / "cache").exists() and fake_derive == []


def test_a_failed_derive_never_lands_under_the_key(tmp_path, install, mod_dat, fake_derive, monkeypatch):
    key = _key(mod_dat, install)

    def broken(dat, install, out_dir, *, tools=None, parsed=None):
        out_dir.mkdir(parents=True)
        (out_dir / "unit_stats.json").write_text("{")
        raise RuntimeError("gen_unit_stats.py exited with 1")

    monkeypatch.setattr(dat_derive, "derive", broken)
    with pytest.raises(RuntimeError):
        dat_cache.build_entry(mod_dat, install, tmp_path / "cache" / key)
    assert not (tmp_path / "cache" / key).exists() and dat_cache.lookup(key, tmp_path / "cache") is None


def test_the_second_of_two_builds_uses_the_first_entry(tmp_path, install, mod_dat, fake_derive, monkeypatch):
    """Two windows deriving one key: the loser's rename finds a complete entry and keeps it, untouched."""
    key = _key(mod_dat, install)
    first = dat_cache.build_entry(mod_dat, install, tmp_path / "cache" / key)
    before = {p.name: p.stat().st_mtime_ns for p in first.path.iterdir()}
    second = dat_cache.build_entry(mod_dat, install, tmp_path / "cache" / key)
    assert second == first
    assert {p.name: p.stat().st_mtime_ns for p in first.path.iterdir()} == before


def test_an_incomplete_folder_under_the_key_is_set_aside(tmp_path, install, mod_dat, fake_derive):
    key = _key(mod_dat, install)
    (tmp_path / "cache" / key).mkdir(parents=True)
    (tmp_path / "cache" / key / "unit_stats.json").write_text("{")
    entry = dat_cache.build_entry(mod_dat, install, tmp_path / "cache" / key)
    assert dat_cache.lookup(key, tmp_path / "cache") == entry
    assert any(p.name.startswith(f"{key}.stale-") for p in (tmp_path / "cache").iterdir())


def test_the_key_covers_every_input_but_the_mod_path(tmp_path, install, mod_dat, monkeypatch):
    key = _key(mod_dat, install)
    moved = tmp_path / "elsewhere.dat"
    moved.write_bytes(mod_dat.read_bytes())
    assert _key(moved, install) == key, "the same bytes elsewhere are the same entry"
    mod_dat.write_bytes(b"mod dat v2")
    assert _key(mod_dat, install) != key
    mod_dat.write_bytes(b"mod dat")
    palette = install / "resources" / "_common" / "palettes" / "original.pal"
    os.utime(palette, ns=(1, 1))
    assert _key(mod_dat, install) != key, "a game patch's palette change misses"
    patched = _key(mod_dat, install)
    monkeypatch.setattr(dat_derive, "source_digest", lambda tools=None: "edited generators")
    assert _key(mod_dat, install) != patched, "a generator edit misses"
    edited = _key(mod_dat, install)
    monkeypatch.setattr(dat_cache, "genieutils_version", lambda: "9.9.9")
    assert _key(mod_dat, install) != edited, "a parser bump misses"


def test_the_source_digest_covers_the_generators_and_helper():
    names = {path.name for path in dat_derive.source_files()}
    assert {script for script, _ in dat_derive.GENERATORS} <= names
    assert {dat_derive.GAME_ROOT_HELPER, "dat_derive.py"} <= names
    assert all(path.is_file() for path in dat_derive.source_files())


@pytest.mark.parametrize(
    "parts",
    [("steamapps", "compatdata", "813780", "cache"), ("Games", "Age of Empires 2 DE", "cache"), ("mods", "subscribed", "x")],
)
def test_a_destination_in_the_game_profile_is_refused(tmp_path, install, mod_dat, fake_derive, parts):
    out = tmp_path.joinpath(*parts)
    assert dat_cache.write_refusal(out) is not None
    with pytest.raises(ValueError, match="refusing"):
        dat_cache.build_entry(mod_dat, install, out)
    assert not out.exists() and fake_derive == []


def test_a_mod_read_from_the_game_profile_is_accepted(tmp_path, install, fake_derive):
    dat = tmp_path / "compatdata" / "813780" / "mods" / "subscribed" / "M" / "empires2_x2_p1.dat"
    dat.parent.mkdir(parents=True)
    dat.write_bytes(b"x")
    assert dat_cache.write_refusal(tmp_path / "cache") is None
    dat_cache.build_entry(dat, install, tmp_path / "cache" / _key(dat, install))


def test_the_derive_subprocess_exit_codes(tmp_path, monkeypatch):
    built = []
    monkeypatch.setattr(dat_cache, "build_entry", lambda dat, install, out: built.append((dat, install, out)))
    assert dat_derive.main(["a.dat", "inst", str(tmp_path / "k")]) == 0 and built == [(Path("a.dat"), Path("inst"), tmp_path / "k")]
    assert dat_derive.main(["only-one"]) == 2

    def fail(*args):
        raise AssertionError("VER 7.2")

    monkeypatch.setattr(dat_cache, "build_entry", fail)
    assert dat_derive.main(["a.dat", "inst", str(tmp_path / "k")]) == 1


def test_a_bare_parser_failure_names_the_dat_format(tmp_path):
    """genieutils fails a VER 7.2 .dat with an empty AssertionError; the viewer shows this line instead."""
    deflate = zlib.compressobj(wbits=-15)
    dat = tmp_path / "old.dat"
    dat.write_bytes(deflate.compress(b"VER 7.2\0" + bytes(64)) + deflate.flush())
    assert dat_derive.dat_version(dat) == "VER 7.2"
    assert dat_derive.failure_message(AssertionError(), dat) == (
        "genieutils-py cannot read this .dat (VER 7.2, AssertionError)"
    )
    assert dat_derive.failure_message(ValueError("bad tail"), dat) == "ValueError: bad tail"
    (tmp_path / "junk.dat").write_bytes(b"not deflate")
    assert "not a readable .dat" in dat_derive.failure_message(AssertionError(), tmp_path / "junk.dat")


# A stand-in for every generator: writes its tables, logs its argv, and the
# graphic map's trips one guard through _game_root.refuse() (GH #262).
_SPY_GENERATOR = """
import json, sys
from pathlib import Path
import _game_root
args = sys.argv
out = Path(args[args.index("--out-dir") + 1])
_game_root.set_lenient("--lenient" in args)
for name in {names!r}:
    (out / name).write_text("{{}}")
with (out / "argv.jsonl").open("a") as log:
    log.write(json.dumps(args) + "\\n")
if {script!r} == "gen_unit_graphic_map.py":
    _game_root.refuse("unit_const 45 broke a rule", "drawn alone")
    _game_root.refuse("unit_const 45 broke a rule", "drawn alone")
if {script!r} in ("gen_unit_graphic_map.py", "gen_unit_render_data.py"):
    _game_root.refuse("a shared rule broke", "left out")
"""


@pytest.fixture()
def spy_tools(tmp_path) -> Path:
    tools = tmp_path / "tools"
    tools.mkdir()
    shutil.copy(DESCAPE.parent / "tools" / dat_derive.GAME_ROOT_HELPER, tools)
    for script, names in dat_derive.GENERATORS:
        (tools / script).write_text(_SPY_GENERATOR.format(names=names, script=script))
    return tools


@pytest.fixture()
def no_cached_helper():
    """No _game_root cached, so the spy's copy is the one derive() leaves behind; the old state after."""
    saved = sys.modules.pop("_game_root", None)
    yield
    if saved is None:
        sys.modules.pop("_game_root", None)
    else:
        sys.modules["_game_root"] = saved


def _argvs(out: Path) -> list[list[str]]:
    return [json.loads(line) for line in (out / "argv.jsonl").read_text().splitlines()]


def test_a_lenient_derive_labels_each_warning_by_its_generator(tmp_path, spy_tools, monkeypatch):
    pytest.importorskip("genieutils")
    monkeypatch.syspath_prepend(str(DESCAPE.parent / "tools"))
    import _game_root  # the repo's copy, as a test importing a generator has it

    out = tmp_path / "out"
    warnings = dat_derive.derive(tmp_path / "mod.dat", tmp_path / "install", out, tools=spy_tools)
    assert warnings == [
        "gen_unit_graphic_map.py: unit_const 45 broke a rule (drawn alone)",
        "gen_unit_graphic_map.py: a shared rule broke (left out)",
        "gen_unit_render_data.py: a shared rule broke (left out)",
    ], "de-duplicated per generator run, so a later generator's repeat keeps its own label"
    assert all("--lenient" in argv for argv in _argvs(out))
    assert sys.modules["_game_root"] is _game_root, "the caller's copy is put back"
    assert not _game_root.lenient()


def test_a_strict_derive_fails_on_the_first_guard_and_stays_strict(tmp_path, spy_tools, no_cached_helper):
    pytest.importorskip("genieutils")
    out = tmp_path / "out"
    with pytest.raises(dat_derive.DeriveError, match=r"gen_unit_graphic_map\.py exited with unit_const 45 broke a rule"):
        dat_derive.derive(tmp_path / "mod.dat", tmp_path / "install", out, tools=spy_tools, lenient=False)
    assert not any("--lenient" in argv for argv in _argvs(out))
    helper = sys.modules["_game_root"]
    assert Path(helper.__file__).parent == spy_tools
    assert not helper.lenient() and helper.WARNINGS == []


def test_a_derive_that_fails_leaves_the_helper_strict(tmp_path, spy_tools, no_cached_helper):
    pytest.importorskip("genieutils")
    (spy_tools / "gen_unit_stats.py").write_text("raise ValueError('a reshaped table')")
    with pytest.raises(ValueError, match="a reshaped table"):
        dat_derive.derive(tmp_path / "mod.dat", tmp_path / "install", tmp_path / "out", tools=spy_tools)
    helper = sys.modules["_game_root"]
    assert not helper.lenient() and helper.WARNINGS == []


def test_the_regen_tool_uses_the_shared_registry():
    sys.path.insert(0, str(DESCAPE.parent / "tools"))
    try:
        import regen_game_data
    finally:
        sys.path.remove(str(DESCAPE.parent / "tools"))
    assert regen_game_data.GENERATORS is dat_derive.GENERATORS


@pytest.mark.corpus
@pytest.mark.slow
def test_derive_reproduces_the_committed_tables(tmp_path):
    """The Step 2 gate: one in-process derive of the install .dat, byte-identical to every committed table."""
    from descape import asset_source

    install = asset_source.get_install_path()
    if install is None:
        pytest.skip("no AoE2:DE install visible: set AOE2DE_INSTALL_PATH to one")
    pytest.importorskip("genieutils")
    code = (
        "import sys; from pathlib import Path; from descape import dat_derive; "
        "print('WARNINGS', dat_derive.derive(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])))"
    )
    run = subprocess.run(
        [sys.executable, "-c", code, str(install / dat_cache.DAT_SUBPATH), str(install), str(tmp_path)],
        capture_output=True, text=True, timeout=600, check=False, cwd=DESCAPE.parent,
    )
    assert run.returncode == 0, run.stderr
    differ = [n for n in dat_derive.TABLE_NAMES if (tmp_path / n).read_bytes() != (DESCAPE / n).read_bytes()]
    assert differ == []
    # Run lenient, as a data mod's is: the stock .dat must trip no guard at all (GH #262).
    assert run.stdout.strip().splitlines()[-1] == "WARNINGS []"


@pytest.mark.corpus
@pytest.mark.slow
def test_the_galactic_battlegrounds_dat_derives_with_warnings(tmp_path):
    """GH #262, local only: the Galactic Battlegrounds Test - Data mod's .dat
    (my-scenarios/, never committed; DESCAPE_GBG_DAT overrides the path) loads,
    building 45's ship-sail shell simplified rather than failing the derive."""
    from descape import asset_source

    dat = Path(os.environ.get("DESCAPE_GBG_DAT") or DESCAPE.parent / "my-scenarios" / "gbg_empires2_x2_p1.dat")
    if not dat.is_file():
        pytest.skip(f"no Galactic Battlegrounds .dat at {dat}: set DESCAPE_GBG_DAT to one")
    install = asset_source.get_install_path()
    if install is None:
        pytest.skip("no AoE2:DE install visible: set AOE2DE_INSTALL_PATH to one")
    pytest.importorskip("genieutils")
    code = (
        "import json, sys; from pathlib import Path; from descape import dat_derive; "
        "print(json.dumps(dat_derive.derive(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))))"
    )
    run = subprocess.run(
        [sys.executable, "-c", code, str(dat), str(install), str(tmp_path)],
        capture_output=True, text=True, timeout=600, check=False, cwd=DESCAPE.parent,
    )
    assert run.returncode == 0, run.stderr
    assert all((tmp_path / name).is_file() for name in dat_derive.TABLE_NAMES)
    warnings = json.loads(run.stdout.strip().splitlines()[-1])
    assert any(w.startswith("gen_unit_graphic_map.py: unit_const 45 (type 80)") for w in warnings), warnings[:5]
