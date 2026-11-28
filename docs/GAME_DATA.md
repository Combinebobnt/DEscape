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
| `formats` | none | `aoe1-1.15` | The `scenario_format` labels that are this game's |
| `tables_dir` | `descape/` | `descape/ror/` | The folder of its generated tables |
| `stamp_key` | none | `ror` | Its section of `tools/game_data_stamp.json` |
| `strings_files` | `key-value-strings-utf8.txt`, `key-value-paphos-strings-utf8.txt` | `key-value-pompeii-strings-utf8.txt` | Its strings files, read in this order |
| `icon_materials` | none | `morematerials.json` | The materials file naming its own icons |
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
- **`reader`, `art` and `install`** take a value from
  `games.IMPLEMENTED_READERS`, `games.IMPLEMENTED_ART` and
  `games.IMPLEMENTED_INSTALLS`. They mark where another scenario reader,
  sprite format or install plugs in. `aoc_install` is the 1999 install
  and `hd_install` HD Edition's, each set in Settings > Advanced > Game data
  (`descape/legacy_content.py`, `descape/hd_content.py`); a game with
  `derived_root` resolves its files under the root built from it (below). `"slp"` art is registered ahead of its walker: until then a game
  with `"slp"` art draws marks.
- **`formats`** claims files by `scenario_format` label
  (`ScenarioFormat.game`), checked before `variants`: Age of Kings / The
  Conquerors claims the `scn-aok` and `scn-aoc` labels and `scn-up`, data
  version 1.23 (mostly UserPatch `.scx2`, which the bytes cannot tell from an
  HD Edition `.scx` of that band), and Return of Rome
  the `aoe1-1.15` label of Age of Empires (1997) and Rise of Rome files,
  which it shows until Age of Empires has data of its own. Every Definitive
  Edition scenario version is the same game, so the DE games claim no DE
  version. HD Edition (`hd`) claims `scn-hd`, its header-2 bands 1.21, 1.24
  and 1.26. Its header-3 `.aoe2scenario` carries `hd-aoe2scenario`, which no
  game claims, so it opens as an AoE2:DE file; `hd`'s `hand_pick_formats`
  names it, so while HD Edition is configured the status log says Settings
  > Advanced > Game data can show it with HD data, view-only.
- **`stand_in`** names the game whose data shows this game's files while its
  own install is not configured (`game_dataset.configured()`).

What reads each field outside `game_dataset`: the window title reads
`title_tag`, the portrait lookup `icon_materials`, File > Data Mod and the
content stack `data_mods`, New Map `blank_template`, and the lock and `write_scenario()`
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
measures everything shown. `game_dataset.file_game(loaded)` is a file's
game, whatever data is active: `games.for_scenario(loaded)` (the game
claiming its format, else its header variant's), or that game's `stand_in`
while its own install is not configured, so an Age of Kings file with no
1999 install is an Age of Empires II file in every respect. They differ when
Settings > Advanced > Game data shows a file with another game's data.

Code that asks about the file, not the display, reads `file_game()`: Auto,
the lock and `write_scenario()`'s refusal, the clipboard's stamp (a copied
block carries the ids of its file's game), the civ indexes of a file's
players (`civ_art`, `civ_availability`) and `batch_api.load()`, which makes
that game active. Only New Map's donor reads `for_scenario()` itself. Building art compares the two: a file shown
with another game's data draws the plain table art, since the active
tables do not have its civs.

## View-only games and the lock

`game_dataset.lock_reason(file_game, shown_game, legacy)` is the one rule
for whether a document may be written, Qt-free. A game whose descriptor says
`writable=False` is view-only: every one of its files is locked, whatever
data shows it, except a pre-DE (`legacy`) document, whose band's edit
families decide instead (below). Otherwise a file is locked while another
game's data shows it. The window's `game_data_lock_reason()` shows that
reason in its banner and refuses every edit and save while it applies.

`scenario_write.write_scenario()` refuses a scenario whose own game is not
writable (a pre-DE one excepted, as above), so a batch script is refused as
well, and the rule is enforced rather than only shown. It does not reach the
tools that write scenario bytes themselves (`strip_units` and the template
generators); the first view-only game's own work covers those. Age of Kings
/ The Conquerors is view-only (`writable=False`), and every one of its files
is pre-DE, so only its band's families ever save; while its install is not
configured its files are their stand-in's and lock as such. HD Edition is
view-only the same way; its header-2 files are pre-DE, and a header-3 HD
file stays AoE2's, so showing it with HD data locks it.

A pre-DE format (`scenario_format`, `LoadedScenario.source_format.legacy`)
is a second lock, at the format level, beside `writable`, and it is per
edit family: the window's `game_data_lock_reason(family)` returns its reason
first, whatever data shows the file, unless the band's
`legacy_write.ALLOWED` entry holds that family (a call naming no family is
always locked), and then still falls through to the data lock above, so a
legacy file shown under Return of Rome data stays locked. Save, Save As,
autosave and the Scripts menu ask whether any family is allowed.
`scenario_write.build_patched()` enforces the same table, raising
`LegacyEditNotAllowed` for any other edit (`docs/SCENARIO_IO.md`, "Saving
back to the old format"). `scenario_write.refuse_legacy()` still refuses
outright in resize, Convert to Version and the byte-writing tools
(`strip_units`, `gen_blank_template`). The
way out is File > Convert to .aoe2scenario (`legacy_convert.convert()`),
which reads the legacy document and builds a new one under AoE2 data from
the DE blank template.

## The 1999 and HD Edition installs and their derived roots

`asset_source.get_install_path()` is the active game's content root: the
AoE2:DE install for a game on it, and for a `derived_root` game the folder its
recipe (`legacy_content.recipe(game.install)`) built from its install, None until that
exists (it never builds). `game_dataset.install_root(game)` answers the same
for a game that is not active. The AoE2:DE install itself, and the Settings
install field's override that sets it, is `get_de_install_path()`, read by
what only DE has: the profile and its mods (`content_roots`, `mod_graphics`),
File > Data Mod and the game-data regeneration tool.

The root mirrors an AoE2:DE install's layout (each graphic SLP, the palette
with a neutral player-colour ramp, blendomatic, the string tables), under the
user cache's `game_roots/aoc/<key>/`, the key hashing a format version, the
install path and each source file's name, size and mtime. A root is staged
and renamed into place whole, and nothing deletes one. The viewer builds it
before activating the game, on a worker thread behind a progress line
(`viewer_game_root.py`), and `batch_api.load()` calls the
recipe's `ensure_root()` first. A build that fails leaves the game
unconfigured (`configured()`), so Auto shows the file with its stand-in's
data, until the install setting changes.

HD Edition's root is a second recipe of the same shape, `hd_content`, under
`game_roots/hd/<key>/`: its loose files are linked (a hardlink, else a
symlink, else a copy) rather than unpacked, the art folders in
`gamedata_x2`, `gamedata_x1`, `graphics` order, and its key hashes each art
and texture folder's matching file names (not their stats, nor the
folders' mtimes, which the scenarios and sounds saved beside the art move)
and Steam's build id from `appmanifest_221380.acf`.
`legacy_content.recipe(game.install)` gives the module for an install kind;
every caller reaches a root through it.

## Ids are persisted

A game id is saved in `config.yaml` (`game_data`, New Map's last choice,
and the keys of `units_favorites`, the Units catalog's favorites per game)
and names a section of `tools/game_data_stamp.json`. Never rename
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
| The 1999 install path, its root built or its build failed | `None`; the viewer's build (`viewer_game_root.build_root()`) clears with `None` again when it ends, so a repaint during the build cannot keep `get_install_path()` as None |

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
| Tables other modules imported by name (`terrain_units`, `terrain_blend`, `terrain_palette`, `player_fields`) | the active tables | refilled in place by the `on_switch()` hooks, after the stack swap and the clear; a hook that raises puts the old stack back and runs the clear and every hook again for it |
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
the fields mark where it plugs in. `_game_root.parse_dat()` picks the
reader by the `.dat`'s version string: `tools/_dat57.py` for the 1999
game's `VER 5.7`, `tools/_dat37.py` for Age of Empires' `VER 3.7`. Such a
game is a `CLASSIC` dataset in `_game_root.py`: its `.dat` and DRS archives
are found case-insensitively along its ordered art folders
(`classic_archives()`, `classic_graphics()`), and each generator's classic
branch serves every classic game. A dataset whose descriptor is not
registered yet waits in `_game_root.PENDING`, which `--dataset` accepts.
