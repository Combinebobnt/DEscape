"""descape/game_dataset.py (GH #67): the variant mapping, which copy of a
table or install file the active dataset reads, the per-dataset cache, and
the guard that every generated-table read goes through table_path()."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from AoE2ScenarioParser.datasets.scenario_variant import ScenarioVariant

from descape import game_dataset
from tools import regen_game_data as regen

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        (ScenarioVariant.ROR, game_dataset.ROR),
        (2, game_dataset.ROR),
        (ScenarioVariant.AOE2, game_dataset.AOE2),
        (ScenarioVariant.LEGACY, game_dataset.AOE2),
        (None, game_dataset.AOE2),
        ("junk", game_dataset.AOE2),
    ],
)
def test_from_variant(variant, expected):
    assert game_dataset.from_variant(variant) == expected


def test_aoe2_is_the_default_and_set_active_reports_a_change():
    assert game_dataset.active() == game_dataset.AOE2
    assert game_dataset.set_active(game_dataset.ROR) is True
    assert game_dataset.is_ror()
    assert game_dataset.set_active(game_dataset.ROR) is False
    with pytest.raises(ValueError, match="unknown game dataset"):
        game_dataset.set_active("aoe3")
    assert game_dataset.active() == game_dataset.ROR


def test_table_path_picks_the_ror_copy_but_not_a_shared_table():
    catalog = game_dataset.TABLES_DIR / "object_catalog.json"
    assert game_dataset.table_path(catalog) == catalog
    assert game_dataset.table_path("object_catalog.json") == catalog
    game_dataset.set_active(game_dataset.ROR)
    assert game_dataset.table_path(catalog) == game_dataset.ROR_TABLES_DIR / "object_catalog.json"
    for shared in game_dataset.SHARED_TABLES:
        assert game_dataset.table_path(shared) == game_dataset.TABLES_DIR / shared


def test_the_shared_tables_are_exactly_the_non_dat_ones():
    outputs = set(regen.outputs_of(regen.GENERATORS))
    on_disk = {path.name for path in game_dataset.TABLES_DIR.glob("*.json")}
    assert on_disk - outputs == game_dataset.SHARED_TABLES


def test_resource_path_prefers_the_ror_root_and_falls_back_to_the_install(tmp_path, monkeypatch):
    from descape import asset_source

    monkeypatch.setenv("AOE2DE_INSTALL_PATH", str(tmp_path))
    asset_source.clear_install_caches()
    (tmp_path / "resources/_common/x").mkdir(parents=True)
    (tmp_path / "resources/_common/x/both.txt").write_text("base")
    (tmp_path / "resources/_common/x/base.txt").write_text("base")
    ror = tmp_path / game_dataset.ROR_SUBPATH / "resources/_common/x"
    ror.mkdir(parents=True)
    (ror / "both.txt").write_text("ror")

    assert game_dataset.resource_path("resources/_common/x/both.txt").read_text() == "base"
    assert game_dataset.resource_dirs("resources/_common/x") == [tmp_path / "resources/_common/x"]
    game_dataset.set_active(game_dataset.ROR)
    assert game_dataset.resource_path("resources/_common/x/both.txt").read_text() == "ror"
    assert game_dataset.resource_path("resources/_common/x/base.txt").read_text() == "base"
    assert game_dataset.resource_path("resources/_common/x/none.txt") == tmp_path / "resources/_common/x/none.txt"
    assert game_dataset.resource_dirs("resources/_common/x") == [ror, tmp_path / "resources/_common/x"]


def test_resource_path_is_none_without_an_install(monkeypatch):
    from descape import asset_source

    monkeypatch.setattr(asset_source, "get_install_path", lambda: None)
    assert game_dataset.resource_path("resources") is None
    assert game_dataset.resource_dirs("resources") == []


def test_cache_keeps_one_result_per_dataset_and_clears_both():
    calls = []

    @game_dataset.cache(maxsize=1)
    def table():
        calls.append(game_dataset.active())
        return game_dataset.active()

    assert table() == game_dataset.AOE2
    game_dataset.set_active(game_dataset.ROR)
    assert table() == game_dataset.ROR
    game_dataset.set_active(game_dataset.AOE2)
    assert table() == game_dataset.AOE2
    assert calls == [game_dataset.AOE2, game_dataset.ROR], "switching back re-parses nothing"
    assert table.cache_info().currsize == 1
    table.cache_clear()
    assert table() == game_dataset.AOE2
    assert len(calls) == 3


def test_a_switch_runs_the_hooks_and_clears_the_install_caches(monkeypatch):
    from descape import asset_source

    ran, cleared = [], []
    monkeypatch.setattr(game_dataset, "_switch_hooks", [lambda: ran.append(game_dataset.active())])
    monkeypatch.setattr(asset_source, "clear_install_caches", lambda: cleared.append(True))
    game_dataset.set_active(game_dataset.ROR)
    game_dataset.set_active(game_dataset.ROR)
    assert ran == [game_dataset.ROR] and cleared == [True]


def test_activated_switches_for_the_block_and_back_even_on_a_raise(monkeypatch):
    ran = []
    monkeypatch.setattr(game_dataset, "_switch_hooks", [lambda: ran.append(game_dataset.active())])
    with pytest.raises(RuntimeError, match="inside"), game_dataset.activated(game_dataset.ROR):
        assert game_dataset.active() == game_dataset.ROR
        raise RuntimeError("inside")
    assert game_dataset.active() == game_dataset.AOE2
    assert ran == [game_dataset.ROR, game_dataset.AOE2]


def test_activated_does_not_switch_when_already_active(monkeypatch):
    ran = []
    monkeypatch.setattr(game_dataset, "_switch_hooks", [lambda: ran.append(game_dataset.active())])
    with game_dataset.activated(game_dataset.AOE2):
        assert game_dataset.active() == game_dataset.AOE2
    with pytest.raises(ValueError, match="unknown game dataset"), game_dataset.activated("aoe3"):
        pass
    assert ran == [] and game_dataset.active() == game_dataset.AOE2


# --- the guard ------------------------------------------------------------


def _generated_tables() -> set[str]:
    return set(regen.outputs_of(regen.GENERATORS))


def test_every_generated_table_read_goes_through_table_path():
    """A `.read_text()` on a path naming a generated table, in any descape/
    module, must be wrapped in game_dataset.table_path(): a bare read would
    serve the AoE2 table under Return of Rome or a data mod (GH #58). The path constants themselves
    stay AoE2's, since tests monkeypatch them."""
    tables = _generated_tables()
    offenders = []
    for path in sorted((ROOT / "descape").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        constants = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
            and any(isinstance(c, ast.Constant) and c.value in tables for c in ast.walk(node.value))
        }
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("read_text", "is_file")):
                continue
            receiver = node.func.value
            names_table = any(
                (isinstance(c, ast.Name) and c.id in constants) or (isinstance(c, ast.Constant) and c.value in tables)
                for c in ast.walk(receiver)
            )
            through = any(
                isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "table_path"
                for c in ast.walk(receiver)
            )
            if names_table and not through:
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"generated tables read without game_dataset.table_path(): {offenders}"

