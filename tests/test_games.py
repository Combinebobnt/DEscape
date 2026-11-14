"""descape/games.py: the registry's invariants, its agreement with the
generators' own game roots and with the per-game checks the descriptor will
replace, and that a game registered later shows everywhere."""

from __future__ import annotations

import ast
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from descape import asset_source, dat_derive, game_dataset, games, object_catalog, player_fields, scenario_io, settings, unit_sprites
from testkit import fake_game
from tools import _game_root
from tools import regen_game_data as regen

ROOT = Path(__file__).resolve().parent.parent


def _registry_problems() -> list[str]:
    """One line per way the registry is inconsistent; a new game's checklist."""
    problems = []
    ids = games.ids()
    problems += [f"{game_id}: registered twice" for game_id in sorted(set(ids)) if ids.count(game_id) > 1]
    defaults = [game.id for game in games.GAMES if game.variants is None]
    if defaults != [games.DEFAULT_ID]:
        problems.append(f"exactly the default game {games.DEFAULT_ID!r} may have variants=None, found {defaults}")
    # The default game is the install itself, with descape/'s tables and the stamp's top level.
    default = games.default()
    if (default.stamp_key, default.root_subpath, default.fallback) != (None, None, None):
        problems.append(f"{default.id}: the default game takes no stamp_key, root_subpath or fallback")
    if default.tables_dir != game_dataset.TABLES_DIR:
        problems.append(f"{default.id}: the default game's tables_dir must be {game_dataset.TABLES_DIR}")
    claimed: dict[int, str] = {}
    sections: dict[str, str] = {}
    for game in games.GAMES:
        for variant in sorted(game.variants or ()):
            if variant in claimed:
                problems.append(f"{game.id}: variant {variant} is already {claimed[variant]}'s")
            claimed.setdefault(variant, game.id)
        if game.id != games.DEFAULT_ID:
            # Without its own section a regeneration would overwrite the default game's stamp.
            if game.stamp_key is None:
                problems.append(f"{game.id}: needs a stamp_key")
            elif game.stamp_key in sections:
                problems.append(f"{game.id}: stamp_key {game.stamp_key!r} is already {sections[game.stamp_key]}'s")
            else:
                sections[game.stamp_key] = game.id
            if default.tables_dir not in game.tables_dir.parents:
                problems.append(f"{game.id}: tables_dir must sit under {default.tables_dir}")
        if game.install != games.AOE2DE_INSTALL:
            problems.append(f"{game.id}: install {game.install!r} is not implemented")
        try:
            games.chain(game)
        except ValueError as error:
            problems.append(f"{game.id}: fallback: {error}")
        missing = [name for name in dat_derive.TABLE_NAMES if not (game.tables_dir / name).is_file()]
        if missing:
            problems.append(f"{game.id}: {len(missing)} generated tables missing from {game.tables_dir}, first {missing[0]}")
        if game.blank_template is not None and not (scenario_io.TEMPLATE_DIR / game.blank_template).is_file():
            problems.append(f"{game.id}: no template {game.blank_template}")
        if game.reader not in games.IMPLEMENTED_READERS:
            problems.append(f"{game.id}: reader {game.reader!r} is not implemented")
        if game.art not in games.IMPLEMENTED_ART:
            problems.append(f"{game.id}: art {game.art!r} is not implemented")
    return problems


def test_the_registry_is_consistent():
    assert not _registry_problems()


def test_the_registry_check_names_every_kind_of_problem(tmp_path):
    broken = fake_game.fake(
        tmp_path,
        fallback="nowhere",
        variants=games.ROR.variants,
        blank_template="blank_fake.aoe2scenario",
        reader="scx",
        art="slp",
        install="hd",
        stamp_key=None,
    )
    second = fake_game.fake(tmp_path, id="second", variants=None, stamp_key=games.ROR.stamp_key)
    with fake_game.registered(broken), fake_game.registered(second):
        problems = _registry_problems()
    for expected in (
        "exactly the default game",
        "fake: needs a stamp_key",
        "second: stamp_key 'ror' is already ror's",
        "fake: tables_dir must sit under",
        "fake: install 'hd' is not implemented",
        "fake: variant 2 is already ror's",
        "fake: fallback: unknown game dataset 'nowhere'",
        "generated tables missing",
        "fake: no template blank_fake.aoe2scenario",
        "fake: reader 'scx' is not implemented",
        "fake: art 'slp' is not implemented",
    ):
        assert any(expected in problem for problem in problems), (expected, problems)
    assert not _registry_problems(), "both fake games are gone again"


def test_the_persisted_ids_and_their_order():
    # config.yaml and the stamp file carry these ids, and the order is the Settings and New Map order.
    assert games.ids() == (game_dataset.AOE2, game_dataset.ROR) == ("aoe2", "ror")
    assert games.default() is games.AOE2 and game_dataset.game() is games.AOE2


def test_register_refuses_a_duplicate_and_unregister_an_unknown_game():
    with pytest.raises(ValueError, match="already registered"):
        games.register(games.ROR)
    with pytest.raises(ValueError, match="unknown game dataset"):
        games.unregister("aoe3")
    assert games.GAMES == [games.AOE2, games.ROR]


def test_a_fallback_cycle_is_an_error_not_a_hang(tmp_path):
    looped = fake_game.fake(tmp_path, id="second", variants=None, fallback=fake_game.FAKE_ID)
    with (
        fake_game.registered(fake_game.fake(tmp_path, fallback="second")),
        fake_game.registered(looped),
        pytest.raises(ValueError, match="cycle"),
    ):
        games.chain(looped)
    assert games.chain(games.ROR) == (games.ROR, games.AOE2) and games.chain(games.AOE2) == (games.AOE2,)


def test_a_scenarios_own_game_does_not_follow_the_active_one():
    ror_file, aoe2_file = SimpleNamespace(game_variant=2), SimpleNamespace(game_variant=None)
    game_dataset.set_active(game_dataset.ROR)
    assert games.for_scenario(ror_file) is games.ROR and games.for_scenario(aoe2_file) is games.AOE2
    assert game_dataset.game() is games.ROR


def test_the_generators_game_roots_match_the_registry(tmp_path):
    """tools/_game_root.py imports nothing from descape (its bytes key every cached data mod), so it keeps its own table."""
    assert games.ids() == _game_root.DATASETS
    for game in games.GAMES:
        assert _game_root.game_root(tmp_path, game.id) == game.root(tmp_path)
        assert _game_root.dat_path(tmp_path, game.id) == tmp_path / regen.dat_subpath(game)


def test_the_descriptors_agree_with_the_paths_and_behaviour_they_drive():
    """The fields agree with the paths asset_source and scenario_io still export, and drive what they name."""
    assert asset_source.STRINGS_SUBPATH_TEMPLATE.endswith("/" + games.AOE2.strings_files[0])
    assert asset_source.ROR_STRINGS_SUBPATH_TEMPLATE.endswith("/" + games.ROR.strings_files[0])
    assert (games.AOE2.icon_materials, games.ROR.icon_materials) == (None, asset_source.ROR_MATERIALS_NAME)
    assert scenario_io.TEMPLATE_DIR / games.AOE2.blank_template == scenario_io.BLANK_TEMPLATE_PATH
    assert scenario_io.TEMPLATE_DIR / games.ROR.blank_template == scenario_io.BLANK_ROR_TEMPLATE_PATH
    mod = game_dataset.DataModTables("test-games", game_dataset.TABLES_DIR, game_dataset.TABLES_DIR, "a mod")
    game_dataset.set_data_mod(mod)
    zoomed_out = asset_source.TEXTURE_REFERENCE_TILE_PIXELS // 2
    for game in games.GAMES:
        game_dataset.set_active(game.id)
        shrinks = asset_source.terrain_texture_size(zoomed_out) != asset_source.LOADED_TEXTURE_SIZE
        assert shrinks == game.texture_scales_with_level, game.id
        assert (game_dataset.data_mod() is mod) == game.data_mods, game.id


def test_a_game_registered_later_shows_everywhere(tmp_path, monkeypatch):
    install = tmp_path / "install"
    # Real tables (a copy of RoR's), since activating a game refills the tables other modules imported.
    tables = tmp_path / "tables"
    shutil.copytree(games.ROR.tables_dir, tables)
    game = fake_game.fake(tables, blank_template=games.AOE2.blank_template)
    choices_before, new_map_before = settings.game_data_choices(), settings.new_map_game_data()
    with fake_game.registered(game):
        assert game_dataset.DATASETS == ("aoe2", "ror", "fake")
        assert game_dataset.DISPLAY_NAMES["fake"] == "Fake Game"
        assert game_dataset.from_variant(fake_game.FAKE_VARIANT) == "fake"
        assert settings.game_data_choices() == [*choices_before, ("fake", "Fake Game")]
        assert settings.new_map_game_data() == (*new_map_before, "fake")
        settings.set_game_data("fake")
        assert settings.get_game_data() == "fake"
        settings.set_new_map_choice({**settings.get_new_map_choice(), "game_data": "fake"})
        assert settings.get_new_map_choice()["game_data"] == "fake"

        assert game_dataset.set_active("fake") is True
        assert game_dataset.game() is game and game_dataset.active() != game_dataset.ROR
        assert game_dataset.table_path("object_catalog.json") == tables / "object_catalog.json"
        for shared in game_dataset.SHARED_TABLES:
            assert game_dataset.table_path(shared) == game_dataset.TABLES_DIR / shared
        assert game_dataset.table_path("object_catalog.json", "aoe2") == game_dataset.TABLES_DIR / "object_catalog.json"

        own, base = install / "modes/Fake/resources/x", install / "resources/x"
        own.mkdir(parents=True)
        base.mkdir(parents=True)
        (own / "own.txt").write_text("fake")
        (base / "base.txt").write_text("base")
        assert game_dataset.resource_path_in(install, "resources/x/own.txt") == own / "own.txt"
        assert game_dataset.resource_path_in(install, "resources/x/base.txt") == base / "base.txt"
        assert game_dataset.resource_path_in(install, "resources/x/none.txt") == base / "none.txt"
        monkeypatch.setenv("AOE2DE_INSTALL_PATH", str(install))
        asset_source.clear_install_caches()
        assert game_dataset.resource_dirs("resources/x") == [own, base]

    assert game_dataset.active() == game_dataset.AOE2, "registered() leaves no fake game active"
    assert game_dataset.DATASETS == ("aoe2", "ror") and settings.game_data_choices() == choices_before
    with pytest.raises(ValueError, match="unknown game dataset"):
        game_dataset.set_active("fake")
    with pytest.raises(ValueError, match="game_data must be one of"):
        settings.set_game_data("fake")


def test_a_game_registered_again_under_the_same_id_reads_its_own_tables(tmp_path):
    """game_dataset.cache() keys on the id, so a fake game's entries must go with it."""

    calls = []

    @game_dataset.cache()
    def catalog_folder():
        calls.append(game_dataset.cache_key())
        return game_dataset.table_path("object_catalog.json").parent

    for source in (games.ROR.tables_dir, games.AOE2.tables_dir):
        tables = tmp_path / source.name / "tables"
        tables.mkdir(parents=True)
        for name in dat_derive.TABLE_NAMES:
            shutil.copy(source / name, tables / name)
        with fake_game.registered(fake_game.fake(tables)):
            game_dataset.set_active(fake_game.FAKE_ID)
            assert catalog_folder() == tables

    # forget() drops a game's entries under a data mod too, and only that game's.
    mod = game_dataset.DataModTables("test-games", game_dataset.TABLES_DIR, game_dataset.TABLES_DIR, "a mod")
    for overlay in (mod, None, mod, None):
        game_dataset.set_data_mod(overlay)
        catalog_folder()
    game_dataset.set_active(game_dataset.ROR)
    catalog_folder()
    assert calls == ["fake", "fake", ("aoe2", mod), "aoe2", "ror"], "one call per key, cached after"
    game_dataset.forget(game_dataset.AOE2)
    catalog_folder()
    game_dataset.set_active(game_dataset.AOE2)
    for overlay in (mod, None):
        game_dataset.set_data_mod(overlay)
        catalog_folder()
    assert calls[5:] == [("aoe2", mod), "aoe2"], "Return of Rome's entry stayed"


def test_a_game_without_a_blank_template_stays_out_of_new_map(tmp_path):
    with fake_game.registered(fake_game.fake(tmp_path)):
        assert "fake" not in settings.new_map_game_data()
        assert ("fake", "Fake Game") in settings.game_data_choices()


# ----------------------------------------------------------------- per-game tables


def _per_game_calls() -> list[str]:
    """file:line of every PerGame(...) call in descape/."""
    found = []
    for path in sorted((ROOT / "descape").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                if name == "PerGame":
                    found.append(f"{path.name}:{node.lineno}")
    return found


def _per_game_problems() -> list[str]:
    """One line per PerGame entry a registered game lacks: a new game's measuring checklist."""
    fake_game.import_every_module()
    ids = set(games.ids())
    problems = []
    for table in games.TABLES:
        problems += [f"{game_id}: no entry in PerGame {table.what!r}" for game_id in sorted(ids - table.keys())]
        problems += [f"PerGame {table.what!r} has an entry for {game_id!r}, no registered game" for game_id in sorted(table.keys() - ids)]
    calls = _per_game_calls()
    if len(calls) != len(games.TABLES):
        problems.append(f"{len(calls)} PerGame() calls in descape/ ({calls}) but {len(games.TABLES)} tables built on import")
    return problems


def test_every_per_game_table_has_every_registered_game():
    assert not _per_game_problems()


def test_the_per_game_check_lists_what_a_new_game_must_measure(tmp_path):
    fake_game.import_every_module()
    games.register(fake_game.fake(tmp_path))
    try:
        problems = _per_game_problems()
    finally:
        games.unregister(fake_game.FAKE_ID)
    assert len(problems) == len(games.TABLES) > 0
    assert "fake: no entry in PerGame 'unit_sprites: cliff consts (.dat class 34)'" in problems


def test_a_per_game_table_reads_the_active_game_and_patches_per_game(monkeypatch):
    table = unit_sprites._CLIFF_VARIANT_BY_GAME
    assert unit_sprites.cliff_const_set() is table["aoe2"] is table.for_game(games.AOE2)
    game_dataset.set_active(game_dataset.ROR)
    assert unit_sprites.cliff_const_set() is table["ror"] is table.active()
    monkeypatch.setitem(table, "ror", frozenset({1}))
    assert unit_sprites.cliff_const_set() == {1}
    game_dataset.set_active(game_dataset.AOE2)
    assert unit_sprites.cliff_const_set() is table["aoe2"]


def test_a_return_of_rome_civ_token_encodes_while_aoe2_is_active():
    """A RoR file shown with its own data writes its tokens; the token set must not follow the active game."""
    assert game_dataset.game() is games.AOE2
    civ = "CARTHAGIAN-CIV"
    assert civ in player_fields.encodable_civilizations()
    assert player_fields._encode_str16(civ).endswith(civ.encode())


def test_a_data_mod_switch_keeps_a_pick_of_a_const_the_mod_left_alone(tmp_path):
    """same_object_in_aoe2() compares the table in effect with the default game's committed
    one, so a data mod, which has no fallback game, keeps a pick its .dat did not change."""
    tables = tmp_path / "tables"
    tables.mkdir()
    for name in dat_derive.TABLE_NAMES:
        shutil.copy(games.AOE2.tables_dir / name, tables / name)
    catalog = json.loads((tables / "object_catalog.json").read_text())
    kept, changed = 83, 93  # Villager, Spearman
    catalog["objects"][str(changed)]["code"] = "MODDED"
    (tables / "object_catalog.json").write_text(json.dumps(catalog))
    game_dataset.set_data_mod(game_dataset.DataModTables("test-pick", tables, tmp_path, "a mod"))
    assert object_catalog.same_object_in_aoe2(kept)
    assert not object_catalog.same_object_in_aoe2(changed)


# ----------------------------------------------------------------- no branching on which game

# Names of a game id constant: game_dataset.AOE2/ROR, games.DEFAULT_ID, and the games.AOE2/ROR descriptors.
_GAME_CONSTANT_NAMES = frozenset({"AOE2", "ROR", "DEFAULT_ID"})
_GAME_ACCESSORS = frozenset({"game", "for_scenario", "for_variant", "get"})


def _names_a_game(node: ast.expr) -> bool:
    """A game id literal or constant, or a tuple, list or set holding one."""
    if isinstance(node, ast.Tuple | ast.List | ast.Set):
        return any(_names_a_game(element) for element in node.elts)
    if isinstance(node, ast.Constant):
        return node.value in games.ids()
    if isinstance(node, ast.Name):
        return node.id in _GAME_CONSTANT_NAMES
    if isinstance(node, ast.Attribute):
        if node.attr == "id":
            return _names_a_game(node.value)
        return isinstance(node.value, ast.Name) and node.value.id in ("game_dataset", "games") and node.attr in _GAME_CONSTANT_NAMES
    return False


def _is_constant(node: ast.expr) -> bool:
    """A literal, an UPPER_CASE name or attribute, or a container: not a game-valued variable or call."""
    if isinstance(node, ast.Constant | ast.Tuple | ast.List | ast.Set):
        return True
    name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else ""
    return name.isupper()


def _accessor_id(node: ast.expr) -> bool:
    """`<game_dataset.game() | games.for_scenario() | ...>.id`."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "id"
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr in _GAME_ACCESSORS
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id in ("game_dataset", "games")
    )


def _game_branches(source: str) -> list[int]:
    """Lines comparing with a game id (==, !=, in, is), and lines comparing a game accessor's
    .id with a constant. Comparing two game-valued variables or calls stays allowed."""
    lines = []
    tree = ast.parse(source)
    per_game_values = {
        id(arg)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and (getattr(node.func, "attr", None) or getattr(node.func, "id", None)) == "PerGame"
        for arg in node.args
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict) and id(node) not in per_game_values and any(
            key is not None and _names_a_game(key) for key in node.keys
        ):
            lines.append(node.lineno)  # a dict keyed by game: a PerGame table the completeness test cannot see
        if isinstance(node, ast.MatchValue) and _names_a_game(node.value):
            lines.append(node.lineno)
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        names_a_game = any(_names_a_game(operand) for operand in operands)
        if names_a_game or (any(_accessor_id(operand) for operand in operands) and any(_is_constant(operand) for operand in operands)):
            lines.append(node.lineno)
    return lines


def test_the_game_branch_lint_tells_a_branch_from_a_comparison_of_two_games():
    flagged = [
        'game_dataset.active() == "ror"',
        "game_dataset.active() in (game_dataset.ROR,)",
        "games.for_scenario(loaded) is games.ROR",
        "game_dataset.game().id != games.DEFAULT_ID",
        "games.get(x).id in OTHER_GAMES",
        'kind not in ["aoe2"]',
        "HOSTS = {game_dataset.AOE2: 1, game_dataset.ROR: 2}",
        'match game_dataset.active():\n    case "ror":\n        pass',
    ]
    allowed = [
        'TABLE = games.PerGame("what", {"aoe2": 1, "ror": 2})',
        'DIRS = {"objects": 1}',
        "games.for_scenario(scenario).id != game_dataset.active()",
        "file_game != shown_game",
        "entry.id == wanted.id",
        'game_dataset.game().id == own_id',
        'section == "objects"',
    ]
    assert [source for source in flagged if not _game_branches(source)] == []
    assert [source for source in allowed if _game_branches(source)] == []


def test_no_descape_module_branches_on_which_game():
    """Branching on the game goes through a descriptor field or a PerGame table
    (docs/GAME_DATA.md, "Per-game tables"); only the registry and the resolver
    name game ids. tools/ is exempt: its generators import nothing from descape."""
    offenders = []
    for path in sorted((ROOT / "descape").rglob("*.py")):
        if path.name in ("games.py", "game_dataset.py"):
            continue
        offenders += [f"{path.name}:{line}" for line in _game_branches(path.read_text(encoding="utf-8"))]
    assert not offenders, f"comparisons with a game id, use a Game field or a PerGame table: {offenders}"


# ----------------------------------------------------------------- the generators' own per-game values

# Each _game_root dataset's values in the two generators that swap in hand-measured lists:
# None is the module's own (the default game), else the name of its dict of replacements.
_GENERATOR_VALUES = {_game_root.AOE2: None, _game_root.ROR: "_ROR_VALUES"}
_SELECTING_GENERATORS = ("gen_unit_graphic_map.py", "gen_unit_render_data.py")


def _private_generator(name: str, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location(f"_{name[:-3]}_select", ROOT / "tools" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # a private copy: _select_dataset() rewrites module globals
    return module


def test_the_generators_have_values_for_every_game_root(monkeypatch):
    """A game added to tools/_game_root.py also needs its hand-measured values in each
    generator and a _select_dataset() branch for them, then a _GENERATOR_VALUES entry."""
    assert set(_GENERATOR_VALUES) == set(_game_root.DATASETS)
    for name in _SELECTING_GENERATORS:
        for dataset, values_name in _GENERATOR_VALUES.items():
            module = _private_generator(name, monkeypatch)
            before = dict(vars(module))
            module._select_dataset(dataset)
            if values_name is None:
                changed = [key for key, value in vars(module).items() if before.get(key) is not value]
                assert not changed, (name, dataset, changed)
                continue
            values = getattr(module, values_name)
            assert values, (name, values_name)
            assert all(getattr(module, key) is value for key, value in values.items()), (name, dataset)
            # gen_unit_graphic_map also swaps its art holdouts outside the values dict.
            holdouts = values_name.replace("_VALUES", "_ART_INVARIANT_HOLDOUTS")
            if hasattr(module, holdouts):
                assert module._ART_INVARIANT_HOLDOUTS is getattr(module, holdouts), (name, dataset)
