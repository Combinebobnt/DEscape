"""tools/regen_game_data.py: the registry of .dat generators, the stamp that
records what the committed tables were generated from, and the wrapper's
modes and exit codes.

The synthetic cases build a scratch install, tables directory and stamp, and
a registry of trivial generator scripts, and pass every path explicitly: none
of them reads the developer's own install, so they behave the same with
AOE2DE_INSTALL_PATH set. The one real-install case is corpus + slow.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from descape.content_roots import AOE2DE_STEAM_APP_ID
from tools import regen_game_data as regen

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "regen_game_data.py"

# Top-level descape/*.json tables that no .dat generator writes.
NON_DAT_TABLES = {
    "cliff_connectivity.json",  # measured from scenario files by tools/gen_cliff_connectivity.py
    "trigger_requirements.json",  # hand-kept
}

FAKE_GENERATORS = (
    ("gen_alpha.py", ("alpha.json",)),
    ("gen_beta.py", ("beta.json", "gamma.json")),
)


def _write_generator(tools_dir: Path, script: str, tables: dict[str, str], *, exit_code: int = 0) -> None:
    """A stand-in generator: same command line as the real ones, each table's
    text followed by the .dat's length. A failing one still writes first."""
    lines = [
        "import argparse, pathlib, sys",
        "parser = argparse.ArgumentParser()",
        'parser.add_argument("install", type=pathlib.Path)',
        'parser.add_argument("--out-dir", type=pathlib.Path)',
        "args = parser.parse_args()",
        'dat = (args.install / "resources/_common/dat/empires2_x2_p1.dat").read_bytes()',
    ]
    for name, text in tables.items():
        lines.append(f'(args.out_dir / {name!r}).write_text({text!r} + " " + str(len(dat)) + "\\n")')
        lines.append(f'print("Wrote", {name!r})')
    if exit_code:
        lines.append(f'print("WARNING: this generator refuses", file=sys.stderr); sys.exit({exit_code})')
    (tools_dir / script).write_text("\n".join(lines) + "\n", encoding="utf-8")


class _World:
    def __init__(self, tmp_path: Path) -> None:
        self.steamapps = tmp_path / "steamapps"
        self.install = self.steamapps / "common" / "AoE2DE"
        self.dat = self.install / regen.DAT_SUBPATH
        self.dat.parent.mkdir(parents=True)
        self.dat.write_bytes(b"dat one")
        self.set_buildid("111")
        self.tools = tmp_path / "tools"
        self.tools.mkdir()
        _write_generator(self.tools, "gen_alpha.py", {"alpha.json": "alpha"})
        _write_generator(self.tools, "gen_beta.py", {"beta.json": "beta", "gamma.json": "gamma"})
        self.tables = tmp_path / "tables"
        self.stamp = tmp_path / "stamp.json"
        self.version: str | None = "9.9"
        self.day = "2026-01-02"
        self.lines: list[str] = []

    def set_buildid(self, buildid: str | None) -> None:
        manifest = self.steamapps / f"appmanifest_{AOE2DE_STEAM_APP_ID}.acf"
        text = '"AppState"\n{\n\t"appid"\t\t"813780"\n' + (f'\t"buildid"\t\t"{buildid}"\n' if buildid else "") + "}\n"
        manifest.write_text(text, encoding="utf-8")

    def context(self) -> regen.Context:
        return regen.Context(
            tools_dir=self.tools,
            generators=FAKE_GENERATORS,
            genieutils_version=lambda: self.version,
            today=lambda: self.day,
            out=self.lines.append,
        )

    def run(self, *argv: str, install: Path | None = None) -> int:
        self.lines.clear()
        paths = ["--install", str(install or self.install), "--tables-dir", str(self.tables), "--stamp", str(self.stamp)]
        return regen.main([*argv, *paths], ctx=self.context())

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def stamp_data(self) -> dict:
        return json.loads(self.stamp.read_text(encoding="utf-8"))

    def written(self) -> dict[str, bytes]:
        """Every file in the tables directory plus the stamp, by name."""
        files = {path.name: path.read_bytes() for path in sorted(self.tables.glob("*"))} if self.tables.is_dir() else {}
        if self.stamp.is_file():
            files["<stamp>"] = self.stamp.read_bytes()
        return files


@pytest.fixture()
def world(tmp_path):
    return _World(tmp_path)


@pytest.fixture()
def stamped(world):
    assert world.run() == regen.EXIT_OK, world.text
    return world


# --- the real registry, tables and stamp ---------------------------------


def test_the_registry_covers_every_dat_generator():
    parsing = {path.name for path in (ROOT / "tools").glob("gen_*.py") if "DatFile.parse(" in path.read_text(encoding="utf-8")}
    registered = {script for script, _tables in regen.GENERATORS}
    assert registered == parsing, (
        f"tools/regen_game_data.py's GENERATORS and the tools/gen_*.py that parse the .dat disagree: "
        f"unregistered {sorted(parsing - registered)}, registered but not parsing {sorted(registered - parsing)}"
    )


def test_every_top_level_table_is_a_registered_output_or_a_named_non_dat_table():
    outputs = regen.outputs_of(regen.GENERATORS)
    assert len(outputs) == len(set(outputs)), "two generators claim one table"
    assert not set(outputs) & NON_DAT_TABLES
    on_disk = {path.name for path in (ROOT / "descape").glob("*.json")}
    assert on_disk == set(outputs) | NON_DAT_TABLES, (
        f"descape/*.json and the registry disagree: on disk only {sorted(on_disk - set(outputs) - NON_DAT_TABLES)}, "
        f"listed only {sorted((set(outputs) | NON_DAT_TABLES) - on_disk)}. A table a .dat generator writes joins "
        f"GENERATORS in tools/regen_game_data.py; any other joins NON_DAT_TABLES here."
    )


def test_each_committed_table_matches_its_stamp_entry():
    stamp = regen.load_stamp(regen.STAMP_PATH)
    assert stamp is not None, "tools/game_data_stamp.json is missing or unreadable: run tools/regen_game_data.py"
    outputs = regen.outputs_of(regen.GENERATORS)
    assert sorted(stamp["tables"]) == sorted(outputs), "the stamp's tables are not the registry's: run tools/regen_game_data.py"
    for name in outputs:
        assert regen.table_sha256(regen.TABLES_DIR / name) == stamp["tables"][name], (
            f"{name} changed without its stamp: run tools/regen_game_data.py --only {regen.script_for(name)} "
            f"and commit tools/game_data_stamp.json with the table"
        )


def test_the_stamp_is_kept_out_of_the_bundled_data_and_carries_its_fields():
    # packaging/descape.spec bundles every descape/**/*.json, so the stamp must not live there.
    assert (ROOT / "descape") not in regen.STAMP_PATH.parents
    stamp = regen.load_stamp(regen.STAMP_PATH)
    assert sorted(stamp) == ["_comment", "dat", "genieutils", "refreshed", "steam_buildid", "tables"]
    assert sorted(stamp["dat"]) == ["sha256", "size"]
    assert regen.STAMP_PATH.read_text(encoding="utf-8") == regen.stamp_text(stamp), "the stamp was edited by hand"


# --- --status ------------------------------------------------------------


def test_status_is_current_right_after_a_regeneration(stamped):
    assert stamped.run("--status") == regen.EXIT_OK
    assert stamped.lines[-1] == "game data is current"


def test_status_names_a_table_that_differs_from_the_stamp(stamped):
    (stamped.tables / "beta.json").write_text("edited by hand\n", encoding="utf-8")
    assert stamped.run("--status") == regen.EXIT_STALE
    assert "STALE beta.json: differs from the stamp" in stamped.lines
    assert not any(line.startswith("STALE") and "beta.json" not in line for line in stamped.lines)
    assert stamped.lines[-1] == "game data is stale: run tools/regen_game_data.py"


def test_status_names_the_dat_hash_when_the_install_changed(stamped):
    old = regen.dat_fingerprint(stamped.install)["sha256"]
    stamped.dat.write_bytes(b"dat two")  # same length, so only the hash moves
    new = regen.dat_fingerprint(stamped.install)["sha256"]
    assert stamped.run("--status") == regen.EXIT_STALE
    assert [line for line in stamped.lines if line.startswith("STALE")] == [f"STALE dat.sha256: install {new}, stamp {old}"]


def test_status_names_the_dat_size_too_when_it_moved(stamped):
    stamped.dat.write_bytes(b"dat one, one byte longer and more")
    assert stamped.run("--status") == regen.EXIT_STALE
    assert any(line.startswith("STALE dat.sha256:") for line in stamped.lines)
    assert "STALE dat.size: install 33, stamp 7" in stamped.lines


def test_status_names_the_buildid_when_steam_updated(stamped):
    stamped.set_buildid("222")
    assert stamped.run("--status") == regen.EXIT_STALE
    assert [line for line in stamped.lines if line.startswith("STALE")] == ["STALE steam_buildid: install 222, stamp 111"]


def test_status_does_not_compare_a_buildid_one_side_lacks(stamped):
    stamped.set_buildid(None)
    assert stamped.run("--status") == regen.EXIT_OK, stamped.text


def test_status_is_stale_without_a_stamp(world):
    assert world.run("--status") == regen.EXIT_STALE
    assert any(line.startswith("STALE") and "no readable stamp" in line for line in world.lines)


def test_status_is_stale_when_the_stamp_and_the_registry_list_different_tables(stamped):
    data = stamped.stamp_data()
    data["tables"]["delta.json"] = data["tables"].pop("gamma.json")
    stamped.stamp.write_text(regen.stamp_text(data), encoding="utf-8")
    assert stamped.run("--status") == regen.EXIT_STALE
    assert "STALE gamma.json: not in the stamp" in stamped.lines
    assert "STALE delta.json: in the stamp, but no generator writes it" in stamped.lines


def test_status_folds_crlf_so_a_windows_checkout_still_matches(stamped):
    table = stamped.tables / "alpha.json"
    table.write_bytes(table.read_bytes().replace(b"\n", b"\r\n"))
    assert stamped.run("--status") == regen.EXIT_OK, stamped.text


def test_status_cannot_tell_without_an_install(stamped, monkeypatch):
    assert regen.status(None, stamped.context()) == regen.EXIT_CANNOT
    # main() with no --install asks DEscape's own lookup, which is all this patches.
    monkeypatch.setattr(regen.asset_source, "get_install_path", lambda: None)
    paths = ["--tables-dir", str(stamped.tables), "--stamp", str(stamped.stamp)]
    for mode in ("--status", "--check"):
        stamped.lines.clear()
        assert regen.main([mode, *paths], ctx=stamped.context()) == regen.EXIT_CANNOT
        assert "no AoE2:DE install visible" in stamped.text


def test_status_cannot_tell_when_the_given_install_has_no_dat(stamped, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    for mode in ("--status", "--check"):
        assert stamped.run(mode, install=empty) == regen.EXIT_CANNOT
        assert stamped.lines == [f"not found: {empty / regen.DAT_SUBPATH}"]


# --- --check -------------------------------------------------------------


def test_check_is_current_and_writes_nothing(stamped):
    before = stamped.written()
    assert stamped.run("--check") == regen.EXIT_OK, stamped.text
    assert [line for line in stamped.lines if line.endswith(": identical")] == [
        "  alpha.json: identical", "  beta.json: identical", "  gamma.json: identical",
    ]
    assert "Wrote alpha.json" in stamped.lines, "a generator's own output must be echoed"
    assert stamped.written() == before


def test_check_reads_identical_tables_even_before_the_first_stamp(stamped):
    stamped.stamp.rename(stamped.stamp.with_name("stamp.moved"))
    assert stamped.run("--check") == regen.EXIT_STALE
    assert len([line for line in stamped.lines if line.endswith(": identical")]) == 3
    assert len([line for line in stamped.lines if line.startswith("STALE")]) == 1


def test_check_names_a_table_a_fresh_regeneration_would_change(stamped):
    # The generator now writes something else, and the table and stamp still agree with each other.
    _write_generator(stamped.tools, "gen_alpha.py", {"alpha.json": "alpha, revised"})
    before = stamped.written()
    assert stamped.run("--check") == regen.EXIT_STALE
    assert [line for line in stamped.lines if line.startswith("STALE")] == ["STALE alpha.json: differs from a fresh regeneration"]
    assert "  alpha.json: differs" in stamped.lines
    assert stamped.written() == before


def test_check_is_stale_when_a_generator_fails_and_prints_its_output(stamped):
    _write_generator(stamped.tools, "gen_beta.py", {"beta.json": "beta", "gamma.json": "gamma"}, exit_code=3)
    before = stamped.written()
    assert stamped.run("--check") == regen.EXIT_STALE
    assert "WARNING: this generator refuses" in stamped.lines
    assert "gen_beta.py: exited 3" in stamped.lines
    assert any(line.startswith("STALE gen_beta.py:") for line in stamped.lines)
    assert "  alpha.json: identical" in stamped.lines, "the generators that did run are still compared"
    assert stamped.written() == before


def test_keep_leaves_the_generated_tables_where_it_says(stamped, tmp_path, monkeypatch):
    monkeypatch.setattr(regen.tempfile, "tempdir", str(tmp_path))
    assert stamped.run("--check", "--keep") == regen.EXIT_OK
    kept = [line for line in stamped.lines if line.startswith("kept the generated tables in ")]
    assert len(kept) == 1
    assert sorted(path.name for path in Path(kept[0].removeprefix("kept the generated tables in ")).iterdir()) == [
        "alpha.json", "beta.json", "gamma.json",
    ]


# --- regenerating --------------------------------------------------------


def test_a_full_regeneration_writes_every_table_and_the_stamp(world):
    assert world.run() == regen.EXIT_OK, world.text
    assert (world.tables / "alpha.json").read_text(encoding="utf-8") == "alpha 7\n"
    assert world.stamp_data() == {
        "_comment": regen.STAMP_COMMENT,
        "dat": regen.dat_fingerprint(world.install),
        "steam_buildid": "111",
        "genieutils": "9.9",
        "refreshed": "2026-01-02",
        "tables": {name: regen.table_sha256(world.tables / name) for name in ("alpha.json", "beta.json", "gamma.json")},
    }
    assert world.stamp.read_text(encoding="utf-8") == regen.stamp_text(world.stamp_data())
    assert sorted(world.written()) == ["<stamp>", "alpha.json", "beta.json", "gamma.json"], "no temp file left behind"


def test_refreshed_moves_only_when_the_game_build_did(stamped):
    stamped.day = "2026-02-03"
    assert stamped.run() == regen.EXIT_OK
    assert stamped.stamp_data()["refreshed"] == "2026-01-02", "the same .dat and buildid is not a refresh"
    assert f"  {stamped.stamp}: unchanged" in stamped.lines

    stamped.set_buildid("222")
    assert stamped.run() == regen.EXIT_OK
    assert (stamped.stamp_data()["refreshed"], stamped.stamp_data()["steam_buildid"]) == ("2026-02-03", "222")

    stamped.day = "2026-03-04"
    stamped.dat.write_bytes(b"dat two!")
    assert stamped.run() == regen.EXIT_OK
    assert stamped.stamp_data()["refreshed"] == "2026-03-04"
    assert (stamped.tables / "alpha.json").read_text(encoding="utf-8") == "alpha 8\n"
    assert stamped.run("--status") == regen.EXIT_OK


def test_a_failing_generator_leaves_the_tables_and_the_stamp_untouched(stamped):
    # Both generators would change their tables, and the second one fails after writing its own.
    _write_generator(stamped.tools, "gen_alpha.py", {"alpha.json": "alpha, revised"})
    _write_generator(stamped.tools, "gen_beta.py", {"beta.json": "beta, revised", "gamma.json": "gamma"}, exit_code=3)
    before = stamped.written()
    assert stamped.run() == regen.EXIT_STALE
    assert stamped.written() == before
    assert "nothing written: gen_beta.py failed (output above)" in stamped.lines


def test_a_generator_that_exits_0_without_its_table_counts_as_failed(stamped):
    _write_generator(stamped.tools, "gen_beta.py", {"beta.json": "beta, revised"})
    before = stamped.written()
    assert stamped.run() == regen.EXIT_STALE
    assert "gen_beta.py: exited 0 without writing gamma.json" in stamped.lines
    assert stamped.written() == before


def test_only_updates_its_own_entries_and_leaves_the_rest_of_the_stamp(stamped):
    _write_generator(stamped.tools, "gen_alpha.py", {"alpha.json": "alpha, revised"})
    _write_generator(stamped.tools, "gen_beta.py", {"beta.json": "beta, revised", "gamma.json": "gamma"})
    before = stamped.stamp_data()
    stamped.day, stamped.version = "2026-02-03", "10.0"
    assert stamped.run("--only", "tools/gen_alpha") == regen.EXIT_OK, stamped.text
    after = stamped.stamp_data()
    assert after["tables"]["alpha.json"] == regen.table_sha256(stamped.tables / "alpha.json") != before["tables"]["alpha.json"]
    assert (stamped.tables / "alpha.json").read_text(encoding="utf-8") == "alpha, revised 7\n"
    assert (stamped.tables / "beta.json").read_text(encoding="utf-8") == "beta 7\n", "gen_beta.py was not selected"
    del after["tables"]["alpha.json"], before["tables"]["alpha.json"]
    assert after == before, "every other entry, refreshed and genieutils included, stays as the full run left it"
    assert "== gen_beta.py" not in stamped.lines
    assert stamped.run("--status") == regen.EXIT_OK


def test_only_refuses_when_the_dat_is_not_the_one_the_stamp_records(stamped):
    old = regen.dat_fingerprint(stamped.install)["sha256"]
    stamped.dat.write_bytes(b"dat two")
    new = regen.dat_fingerprint(stamped.install)["sha256"]
    _write_generator(stamped.tools, "gen_alpha.py", {"alpha.json": "alpha, revised"})
    before = stamped.written()
    assert stamped.run("--only", "gen_alpha.py") == regen.EXIT_STALE
    assert f"refusing --only: the install's .dat sha256 is {new}, the stamp records {old}" in stamped.lines
    assert stamped.written() == before
    assert "== gen_alpha.py" not in stamped.lines, "a refused run starts no generator"


def test_only_refuses_when_the_buildid_is_not_the_one_the_stamp_records(stamped):
    stamped.set_buildid("222")
    _write_generator(stamped.tools, "gen_alpha.py", {"alpha.json": "alpha, revised"})
    before = stamped.written()
    assert stamped.run("--only", "gen_alpha.py") == regen.EXIT_STALE
    assert "refusing --only: the install's Steam buildid is 222, the stamp records 111" in stamped.lines
    assert stamped.written() == before


def test_only_refuses_without_a_stamp_and_rejects_an_unknown_generator(world):
    assert world.run("--only", "gen_alpha.py") == regen.EXIT_STALE
    assert world.written() == {}
    with pytest.raises(SystemExit) as exit_info:
        world.run("--only", "gen_nothing.py")
    assert exit_info.value.code == regen.EXIT_CANNOT


def test_regenerating_into_another_tables_dir_needs_its_own_stamp(tmp_path):
    with pytest.raises(SystemExit) as exit_info:
        regen.main(["--tables-dir", str(tmp_path / "tables")])
    assert exit_info.value.code == regen.EXIT_CANNOT
    assert not (tmp_path / "tables").exists()


def test_regenerating_refuses_a_subscribed_mods_folder(world, tmp_path):
    world.tables = tmp_path / "mods" / "subscribed" / "123_x" / "tables"
    assert world.run() == regen.EXIT_CANNOT
    assert not world.tables.exists() and not world.stamp.exists()
    assert world.lines == [f"refusing to write into a subscribed-mods (mods/subscribed/) folder: {world.tables}"]


# --- without genieutils-py -----------------------------------------------


def test_without_genieutils_status_still_works_and_nothing_generates(stamped, monkeypatch):
    def not_installed(name):
        raise regen.importlib.metadata.PackageNotFoundError(name)

    with monkeypatch.context() as patch:
        patch.setattr(regen.importlib.metadata, "version", not_installed)
        assert regen.genieutils_version() is None
    stamped.version = None
    before = stamped.written()
    assert stamped.run("--status") == regen.EXIT_OK, stamped.text
    for argv in (["--check"], [], ["--only", "gen_alpha.py"]):
        assert stamped.run(*argv) == regen.EXIT_CANNOT
        assert "genieutils-py is not installed" in stamped.lines[-1]
        assert not any(line.startswith("== ") for line in stamped.lines)
    assert stamped.written() == before


# --- the real install ----------------------------------------------------


@pytest.mark.corpus
@pytest.mark.slow
def test_every_generator_reproduces_its_committed_tables():
    """The oracle: all ten generators against the installed game, compared
    with the committed tables and the stamp. Same env-var-only install route
    as tests/test_unit_graphic_map.py's single-generator case."""
    from descape import asset_source

    install = asset_source.get_install_path()
    if install is None:
        pytest.skip("no AoE2:DE install visible: set AOE2DE_INSTALL_PATH to one")
    pytest.importorskip("genieutils")
    run = subprocess.run(
        [sys.executable, str(TOOL), "--check", "--install", str(install)],
        capture_output=True, text=True, timeout=600, check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
