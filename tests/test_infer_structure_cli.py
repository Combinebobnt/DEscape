"""tools/infer_structure.py: exit codes, last lines, refusals, and that nothing is written on failure."""

from __future__ import annotations

import json
import shutil

import pytest

from descape import games
from descape import structure_walk as sw
from tools import infer_structure as cli
from tools import structure_infer as si

import conftest

FIXTURES = conftest.ROOT / "tests" / "fixtures"
SAMPLE = FIXTURES / "v159_units_triggers.aoe2scenario"
TEMPLATES = conftest.ROOT / "descape" / "templates"


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = cli.main([str(a) for a in argv])
    out, err = capsys.readouterr()
    return code, out.strip().splitlines()[-1] if out.strip() else "", err.strip()


def test_ambiguous_exits_1_and_writes_nothing(capsys, tmp_path) -> None:
    out = tmp_path / "out"
    code, last, _ = _run(capsys, "infer", SAMPLE, "--base", "1.58", "--out", out)
    assert (code, last.split(":")[0]) == (1, "AMBIGUOUS")
    assert not out.exists()


def test_inferred_writes_both_files_once_the_library_parses_them(capsys, tmp_path) -> None:
    out = tmp_path / "out"
    code, last, _ = _run(capsys, "infer", SAMPLE, "--base", "1.58", "--out", out, "--accept-ambiguous")
    assert (code, last.split(":")[0]) == (0, "INFERRED"), last
    assert sorted(p.name for p in out.iterdir()) == sorted(cli.OUTPUT_FILES)  # no temp left behind
    structure = json.loads((out / "structure.json").read_text(encoding="utf-8"))
    assert "inferred_v1_59_0" in si.block_dict(structure, ("Units", "PlayerUnitsStruct", "UnitStruct"))["retrievers"]
    assert "Status: INFERRED" in (out / "INFERRED.md").read_text(encoding="utf-8")
    # A non-empty --out refuses before any search, and leaves the outputs alone.
    before = (out / "structure.json").stat().st_mtime_ns
    code, _, err = _run(capsys, "infer", SAMPLE, "--base", "1.58", "--out", out)
    assert code == 2 and "not empty" in err
    assert (out / "structure.json").stat().st_mtime_ns == before


def test_refused_exits_1_and_writes_nothing(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "terrain_ids", lambda game: {1})  # the fixture's grid is all terrain 0, which this table lacks
    out = tmp_path / "out"
    code, last, _ = _run(capsys, "infer", SAMPLE, "--base", "1.58", "--out", out)
    assert (code, last.split(":")[0]) == (1, "REFUSED")
    assert "terrain_classes.json" in last
    assert not out.exists()


@pytest.mark.parametrize(
    ("argv", "says"),
    [
        (["infer", "missing.aoe2scenario"], "missing.aoe2scenario"),
        (["infer"], "no sample files"),
        (["infer", SAMPLE, "--base", "9.99"], "9.99"),
        (["infer", SAMPLE, "--diff-expect-new"], "--diff-pair"),
        (["infer", TEMPLATES / "blank_120x120.aoe2scenario", TEMPLATES / "blank_ror_120x120.aoe2scenario"], "more than one game"),
        (["suggest-reference", "missing_dir"], "missing_dir"),
    ],
)
def test_could_not_run_exits_2(capsys, tmp_path, argv, says) -> None:
    code, _, err = _run(capsys, *argv, *([] if argv[0] == "suggest-reference" else ["--out", tmp_path / "out"]))
    assert code == 2 and says in err and err.startswith("could not run:"), err
    assert not (tmp_path / "out").exists()


def test_a_walk_or_structure_error_exits_2_not_1(capsys, tmp_path, monkeypatch) -> None:
    """Exit 1 means AMBIGUOUS or REFUSED; a reference OLD that doesn't walk under its own
    version, or a base structure that doesn't compile, is an input the tool can't use."""
    bad = tmp_path / "structure.json"
    bad.write_text(json.dumps({"FileHeader": {"retrievers": {"x": {"type": "bogus"}}}}), encoding="utf-8")
    code, _, err = _run(capsys, "infer", SAMPLE, "--base", bad, "--out", tmp_path / "out")
    assert code == 2 and err.startswith("could not run: StructureError"), err

    def unwalkable(*args, **kwargs):
        raise sw.WalkError("Triggers.trigger_version", 12, "past the end")

    monkeypatch.setattr(si, "infer", unwalkable)
    code, _, err = _run(capsys, "infer", SAMPLE, "--base", "1.58", "--out", tmp_path / "out")
    assert code == 2 and err.startswith("could not run: WalkError"), err
    assert not (tmp_path / "out").exists()


def test_any_other_exception_exits_2_with_its_traceback(capsys, tmp_path) -> None:
    """A base JSON without FileHeader raises KeyError deep in the search: still not exit 1."""
    headless = tmp_path / "headless.json"
    headless.write_text(json.dumps({"Map": {"retrievers": {"x": {"type": "u8"}}}}), encoding="utf-8")
    code, _, err = _run(capsys, "infer", SAMPLE, "--base", headless, "--out", tmp_path / "out")
    assert code == 2 and err.startswith("could not run: unexpected error") and "Traceback" in err, err
    assert not (tmp_path / "out").exists()


def test_a_failed_replace_never_leaves_a_stale_report_beside_a_new_structure(tmp_path, monkeypatch) -> None:
    out = tmp_path / "out"
    cli.write_outputs(out, {"structure.json": "old\n", "INFERRED.md": "old\n"})
    real = cli.os.replace

    def fail_structure(src, dst):
        if str(dst).endswith("structure.json"):
            raise OSError("disk full")
        real(src, dst)

    monkeypatch.setattr(cli.os, "replace", fail_structure)
    with pytest.raises(OSError):
        cli.write_outputs(out, {"structure.json": "new\n", "INFERRED.md": "new\n"})
    assert sorted(p.name for p in out.iterdir()) == ["INFERRED.md"]  # no structure.json at all, no temp


def test_an_out_folder_under_mods_subscribed_is_refused(capsys, tmp_path) -> None:
    out = tmp_path / "mods" / "Subscribed" / "x"
    code, _, err = _run(capsys, "infer", SAMPLE, "--base", "1.58", "--out", out)
    assert code == 2 and "mods/subscribed" in err
    assert not out.exists()


def test_the_terrain_table_is_the_samples_own_games() -> None:
    files = [si.load_file(TEMPLATES / "blank_ror_120x120.aoe2scenario")]
    game, notes = cli.game_of(files, si.load_structure("1.59"))
    assert (game.id, notes) == ("ror", [])
    assert cli.terrain_table(game).parent == game.tables_dir != cli.terrain_table(games.default()).parent
    assert cli.terrain_ids(game)


def test_suggest_reference_ranks_only_the_base_versions_files(capsys, tmp_path) -> None:
    for name in ("triggers_120x120.aoe2scenario", "blank_120x120_v158.aoe2scenario", "v159_units_triggers.aoe2scenario"):
        shutil.copy(FIXTURES / name, tmp_path / name)
    code, last, _ = _run(capsys, "suggest-reference", tmp_path, "--base", "1.58")
    assert code == 0 and last.startswith("Re-save triggers_120x120.aoe2scenario")
    assert _run(capsys, "suggest-reference", tmp_path, "--base", "1.57")[:2] == (1, f"no file of version 1.57 in {tmp_path}")
