# Game data: which game a scenario is read with

DEscape reads Age of Empires II and Return of Rome scenarios (GH #67), and
each game has its own `.dat`, so the same unit const is a different unit in
each (93 is Spearman in AoE2 and Hoplite in Return of Rome). This file
describes how a game is declared, how tables, strings and files resolve
for it and a data mod over it, which caches each change drops, where the
hand-measured per-game facts live, and how to add a game.
`descape/game_dataset.py` holds the active content stack and resolves
through it.

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

What reads each field outside `game_dataset`: the window title reads
`title_tag`, the portrait lookup `icon_materials`, the terrain texture size
`texture_scales_with_level`, File > Data Mod and the content stack
`data_mods`, New Map `blank_template`, and the lock and `write_scenario()`
`writable` (below). Sprites and portraits resolve along the root chain,
and the x2 art probe and the missing-art line check the game's own root,
so a game with its own root gets the same lookups Return of Rome does. A
game with a `blank_template` is listed in New Map even when it is
view-only, and its new map then opens locked; leave its template unset
until it can be written. `library_enums`, `reader`
and `art` are declared ahead of the code that will read them; until then
the per-game sets and AoE2ScenarioParser's enums are still chosen by game
elsewhere in `descape/`.

## Two questions: whose file, and which data

`game_dataset.game()` is the active game: the data that names, draws and
measures everything shown. `games.for_scenario(loaded)` is a file's own
game, from its header variant, whatever data is active. They differ when
Settings > Advanced > Game data shows a file with another game's data.

Code that asks about the file, not the display, reads `for_scenario()`:
the lock, the clipboard's stamp (a copied block carries the ids of its
file's game), the civ indexes of a file's players (`civ_art`,
`civ_availability`), New Map's donor and `batch_api.load()`, which makes
the file's own game active. Building art compares the two: a file shown
with another game's data draws the plain table art, since the active
tables do not have its civs.

## View-only games and the lock

`game_dataset.lock_reason(file_game, shown_game)` is the one rule for
whether a document may be written, Qt-free. A game whose descriptor says
`writable=False` is view-only: every one of its files is locked, whatever
data shows it. Otherwise a file is locked while another game's data shows
it. The window's `game_data_lock_reason()` shows that reason in its banner
and refuses every edit and save while it applies.

`scenario_write.write_scenario()` refuses a scenario whose own game is not
writable, so a batch script is refused as well, and the rule is enforced
rather than only shown. It does not reach the tools that write scenario
bytes themselves (`strip_units` and the template generators); the first
view-only game's own work covers those. Both games today are writable.

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

## The content stack

What DEscape reads is one `game_dataset.ContentStack`: the active game and
the data mod set over it (`wanted_mod`). Both are process state, since the
window holds one document. The stack is immutable, and `set_active()`,
`set_data_mod()` and `activated()` each build a new one and swap it in with
a single assignment, so no reader ever sees a game from one stack with the
data mod of another. A function that reads the stack twice can still see
two stacks across a switch; the generation checks below cover that.
`game_dataset.stack()` returns it.

**Data mods: one in effect, art mods stacked.** The game selects one data
mod in the lobby and its `.dat` replaces the game's own, so no scenario can
depend on two at once: File > Data Mod lists every cached one and puts one
in effect. A data mod applies only to a game with `data_mods` (Age of
Empires II). Under Return of Rome it stays set but inert, and it is in
effect again on the switch back. Cosmetic art mods are not on the stack:
`mod_graphics` stacks them by the game's mod priority, highest first.

How each kind of content resolves, bottom layer first:

| Content | Resolution |
|---|---|
| Generated tables (`table_path()`) | The whole set from the top provider: the data mod's derived folder, else the game's `tables_dir`. The tables no `.dat` derives (`SHARED_TABLES`) always come from `descape/`. |
| Strings (`ContentStack.strings_paths()`) | Merged key by key, bottom up: the bottom fallback's `strings_files`, then each game up to the active one's own, then the data mod's strings files. A later file wins on a shared key. |
| Install files: textures, masks, blendomatic, palettes, particles, UI colours (`resource_path()`) | The first that exists along the root chain (`ContentStack.roots()`): the game's own root, then each fallback's. When none has it, the bottom root's path. |
| Sprite layout: frame counts and draw guards | The root chain's file. |
| Sprite and terrain-texture pixels | `mod_graphics` over the root chain, pixels only. |

`game_dataset.has_own_strings(lang, game)` answers whether a game's root
ships its own strings files; a game without them shows its fallback's.

## Cache layers

A switch tells the caches which layers changed, as a set passed down
`asset_source.clear_install_caches(changed)`, `clear_art_caches()` and
`unit_sprites.clear_caches()`:

| Change | `changed` |
|---|---|
| A data-mod switch | `{TABLES}` (`game_dataset.DATA_MOD_SWITCH`) |
| A game switch | `{TABLES, ROOTS}` (`game_dataset.GAME_SWITCH`) |
| The install path, the profile, a test's reset | `None`: everything |

Mod Art toggled or a mod opted out calls `clear_art_caches()` alone, with
`None`: the art caches below drop, the strings and the install-keyed rest
stay.

A cache is cleared unless it sits in a kept block that lists every layer
in `changed` (`game_dataset.only_changed()`). The one kept block today holds
the file-, path- and install-keyed caches and lists `TABLES` alone
(`asset_source.FILE_KEYED_SURVIVES`): a data mod changes which file a const
or terrain id points at, not the files.

| Caches | Keyed on | Dropped by |
|---|---|---|
| A `game_dataset.cache()` loader | the stack's `tables_key`, one cache per key | only by a clear listed for it below; a loader no row names keeps every key's cache, so switching back re-reads nothing (a data mod's side files) |
| Tables other modules imported by name (`terrain_units`, `terrain_blend`, `terrain_palette`, `player_fields`) | the active tables | refilled in place by the `on_switch()` hooks, after the stack swap and the clear |
| Strings, the data mod's string sets, `object_catalog`'s catalogs and names | language or id, per `tables_key` | every clear |
| `has_own_strings()` | install, game and language | every clear |
| A terrain id's texture path and average colour, blended terrain blocks | terrain id | every clear |
| Sprite lookups made through the tables: wall connectors, building art, particle stems, a mod's reach | unit const | every clear |
| Decoded sprites: native, scaled, icon and `_x2` frames | file stem | any layer besides `TABLES`. Under `TABLES` alone only the draws naming a stem the install lacks, which only a data mod's own file can draw (every draw when the graphics folder will not list) |
| Sprite indexes and bytes, frame counts, particle art, markers | file | any layer besides `TABLES` |
| Loaded textures, thumbnails, shrunk textures, pending texture loads | file path | any layer besides `TABLES` |
| The install path, `mod_graphics`' scan, blendomatic and masks, icons, UI colours | install | any layer besides `TABLES` |
| The sprite memo, unit outlines, sprite prefetches | `game_dataset.generation()` among their keys | moves on every switch |
| Unit portraits (`get_unit_icon`), a data mod's and its art mods' portrait tables | icon id, or the portrait layers and install | every clear: a data mod's `widgetui/` and the art mods Mod Art draws supply portraits (GH #363) |
| Catalog thumbnails, the button grid's and instruction preview's portraits | `catalog_thumbs.art_generation()`: `asset_source.install_generation()` and `game_dataset.generation()` | either moving; a data-mod switch moves only the second |

**The rule for a new layer.** A change that moves something new (a data
mod's own folder joining art lookup, say) names a new layer in its
`changed` set. Every cache that layer reaches leaves any kept block that
would survive it: the kept block lists the layers its caches do not read,
so an unlisted layer clears the whole block until someone moves the
affected caches out and lists it. Clearing is the default; keeping is what
needs the argument.

## Per-game tables

Some facts about a game are not in its files: which consts are walls and
cliffs, which objects its editor lists for GAIA only, its civilizations and
starting ages, which hosts take relics. Each is measured by hand, in the
game's own editor or over its scenarios, and kept as a `games.PerGame`
table in the module that uses it: a dict from game id to that game's value
(a frozenset, a tuple, or a small function), built with a one-line
description of what it holds.

A table has no default and no inheritance: Return of Rome falls back to
AoE2's files one by one, but its walls, cliffs and civs are its own, and a
copied AoE2 value would be silently wrong. `table.active()` reads the
active game's value at call time; `table.for_game(game)` reads a given
game's, for a file's own game (`games.for_scenario()`, as `civ_art` does
for a document shown with the other game's data). A test patches one
game's entry with `monkeypatch.setitem(table, "aoe2", ...)`. The named
module constants (`unit_sprites._ROTATION_VARIANT_CONSTS` and the others)
stay as the tables' values, for the tests and tools that read them.

`tests/test_games.py` imports every `descape` module, then fails for any
table missing a registered game, listing each table's description: that
list is a new game's measuring checklist. It also counts the `PerGame(`
calls in `descape/` against the tables built, so a table in a module
nothing imports cannot pass by absence.

Two descriptor fields cover the rest. `library_enums` says whether
AoE2ScenarioParser's unit, building, hero and tech enums name the game's
ids; without them the object and tech pickers, trigger tech names,
`batch_api.is_building()` and the player statistics read the game's own
`.dat` table. `fallback` names the game whose committed tables the shared
rows compare against (Return of Rome's editor-hidden objects that are the
same object as AoE2's, and whether an install string fits a row when the
game ships no strings of its own). A data mod has no fallback game: what a
data mod's rows and a pick kept across a switch compare with is the
default game's committed table.

## No branching on which game

Outside `games.py` and `game_dataset.py`, no `descape/` module compares
anything with a game id: not with a literal such as `"ror"`, not with
`game_dataset.ROR`, `games.DEFAULT_ID` or the `games.ROR` descriptor, and
not inside a tuple, list or set (`in ("ror",)`), nor in a `match` case.
No dict is keyed by game id except a `PerGame` table's, which the
completeness test can see. Nor is a game accessor's
`.id` (`game_dataset.game()`, `games.for_scenario()`, `games.for_variant()`,
`games.get()`) compared with a constant. A branch on the game reads a
descriptor field or a `PerGame` table instead. Comparing two games stays
allowed: the file's own game against the one shown is how the lock and
the clipboard refusal work. `tests/test_games.py` enforces this with an AST
pass. `tools/` is exempt, since its generators import nothing from
`descape/`; a test there checks that both generators that swap in per-game
values (`_select_dataset()`) handle every game `tools/_game_root.py` lists.

## Adding a game

A game DEscape reads with its existing readers (a Definitive Edition game
mode, as Return of Rome is):

1. A descriptor in `descape/games.py`. `tests/test_games.py` then lists
   what is missing or inconsistent.
2. Its root in `tools/_game_root.py`, and its hand-measured values in each
   generator that keeps them (`_ROR_VALUES` in `gen_unit_graphic_map.py`
   and `gen_unit_render_data.py`, swapped in by `_select_dataset()`). Both
   are hashed into every cached data mod's key, so this re-derives every
   user's cached data mods once.
3. Its tables and stamp section: `tools/regen_game_data.py --dataset <id>`.
4. Its blank template under `descape/templates/`, if New Map should offer
   it.
5. Every `PerGame` entry the completeness test lists, each measured in the
   game's own editor or over its scenarios, never copied from another
   game. Its civ tokens join `player_fields`' encodable set at import, so
   the descriptor and its civ rows ship together, never registered later.
6. A corpus test like `tests/test_ror_corpus.py`, re-measuring the
   rotation, wall, cliff and gate rules on the game's own files, and its
   figures in `docs/WRITE_RULES.md`.

A game before the Definitive Edition also needs its own scenario reader
(`reader`), art format (`art`, SLP inside DRS archives) and `.dat` reader:
`genieutils-py` reads only the DE family. Each is that game's own work;
the fields mark where it plugs in.
