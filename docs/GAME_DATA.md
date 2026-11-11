# Game data: which game a scenario is read with

DEscape reads Age of Empires II and Return of Rome scenarios (GH #67), and
each game has its own `.dat`, so the same unit const is a different unit in
each (93 is Spearman in AoE2 and Hoplite in Return of Rome). This file
describes how a game is declared. `descape/game_dataset.py` holds the active
game and resolves tables and install files from it.

## The descriptor

A game is one `Game` in `descape/games.py`: a frozen dataclass of plain
data, registered in `games.GAMES`. `games.py` imports nothing from
`descape/`, and its only state is that list.

| Field | Age of Empires II | Return of Rome | What it says |
|---|---|---|---|
| `id` | `aoe2` | `ror` | The game's name in code and in saved settings |
| `display_name` | Age of Empires II | Return of Rome | Shown in Settings, New Map and messages |
| `title_tag` | none | ` [Return of Rome]` | Appended to the window title |
| `install` | `aoe2de` | `aoe2de` | Which configured install holds the game |
| `root_subpath` | none | `modes/Pompeii` | The game's own root under the install |
| `fallback` | none | `aoe2` | The game whose files fill in what this root lacks |
| `variants` | none | `{2}` | The file header variants that are this game's |
| `tables_dir` | `descape/` | `descape/ror/` | The folder of its generated tables |
| `stamp_key` | none | `ror` | Its section of `tools/game_data_stamp.json` |
| `strings_files` | `key-value-strings-utf8.txt` | `key-value-pompeii-strings-utf8.txt` | Its strings files |
| `icon_materials` | none | `morematerials.json` | The materials file naming its own icons |
| `texture_scales_with_level` | no | yes | A zoomed-out level shrinks terrain textures |
| `library_enums` | yes | no | AoE2ScenarioParser's enums name its ids |
| `data_mods` | yes | no | File > Data Mod applies |
| `blank_template` | `blank_120x120.aoe2scenario` | `blank_ror_120x120.aoe2scenario` | New Map's donor in `descape/templates/` |
| `writable` | yes | yes | No makes the game view-only |
| `reader` | `aoe2scenarioparser` | `aoe2scenarioparser` | The scenario reader |
| `art` | `sld` | `sld` | The sprite format |

Notes on the fields:

- **`variants`.** Exactly one game has none: the default game
  (`games.DEFAULT_ID`), which takes every file no other game claims.
  `games.for_variant()` maps a header variant to its game, and
  `games.for_scenario()` does the same for a loaded scenario. That answers
  "whose file is this?", which is a different question from "which game's
  data is active?" (`game_dataset.game()`).
- **`root_subpath` and `fallback`.** A game's root mirrors the install's
  layout. A file is looked up along the chain `games.chain()` returns: the
  game's own root first, then each fallback's. Return of Rome's chain is
  `modes/Pompeii`, then the install.
- **The `.dat`** is `resources/_common/dat/empires2_x2_p1.dat` under the
  game's root. A game with another name or format adds a field for it.
- **`tables_dir`** holds every table `dat_derive.TABLE_NAMES` lists,
  generated from that game's `.dat` by `tools/regen_game_data.py`.
- **`reader`, `art` and `install`** have one implemented value each
  (`games.IMPLEMENTED_READERS`, `games.IMPLEMENTED_ART`,
  `games.AOE2DE_INSTALL`). They mark where another scenario reader, sprite
  format or install plugs in.
- **Format versions are not a field.** Every Definitive Edition scenario
  version is the same game, and `variants` is the only per-game marker in
  those headers.

Several fields are declared before the code that will read them:
`title_tag`, `strings_files`, `icon_materials`, `texture_scales_with_level`,
`library_enums`, `data_mods`, `writable`, `reader` and `art` still have
their own checks elsewhere in `descape/`. Until those move,
`tests/test_games.py` pins `strings_files`, `icon_materials`,
`texture_scales_with_level`, `data_mods` and `blank_template` to them.

## Ids are persisted

A game id is saved in `config.yaml` (`game_data`, and New Map's last
choice) and names a section of `tools/game_data_stamp.json`. Never rename
one. Registry order is the order of the Settings and New Map lists.

## Derived lists are computed per call

Everything derived from the registry is read through a function or computed
on access, never frozen at import: `game_dataset.DATASETS` and
`DISPLAY_NAMES`, `settings.game_data_choices()` and
`settings.new_map_game_data()`, and the `--dataset` choices of
`tools/regen_game_data.py`. A game registered later, as a test does with
`games.register()`, shows in all of them.

## The generators stay import-free

The table generators (`tools/gen_*.py`) and `tools/_game_root.py` import
nothing from `descape/`. They are shipped as data and hashed into the key
of every cached data mod (`dat_derive.source_digest()`), so an edit to any
of them re-derives every user's cached data mods once. `_game_root.py`
therefore keeps its own small table of game roots, and
`tests/test_games.py` checks it against the registry.

## A third game is not yet data alone

A descriptor is where a new game starts, and `tests/test_games.py` fails
for one whose tables, template, fallback, variants or stamp section are
missing or inconsistent. It is not yet enough: `descape/` still branches on
Return of Rome by name in many places, and the generators keep their
hand-measured per-game values for these two games only.
