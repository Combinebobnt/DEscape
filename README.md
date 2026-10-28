# DEscape

External viewer/editor for Age of Empires 2: Definitive Edition `.aoe2scenario`
files. Renders the terrain grid and unit placement, and supports terrain/elevation
editing with undo/redo and a scriptable batch-edit API, without needing the
in-game editor.

## Download

Prebuilt builds that need no Python install are attached to every
[release](https://github.com/Combinebobnt/DEscape/releases/latest). CI builds
and smoke-tests them on every `v*` tag. Pick the asset for your platform:

| Platform | Asset |
| --- | --- |
| Linux x86_64 | `DEscape-<version>-x86_64.AppImage` (single file), or `DEscape-<version>-linux-x86_64.tar.gz` |
| Windows x86_64 | `DEscape-<version>-windows-x86_64.zip` |
| macOS, Apple Silicon | `DEscape-<version>-macos-arm64.tar.gz` |

There is no Intel Mac build; run from source there (see Setup below). The
`.tar.gz` and `.zip` archives unpack to a `DEscape` folder: run `DEscape`
(`DEscape.exe` on Windows) inside it, and keep the rest of the folder beside it.

First-run notes:
- **Linux AppImage:** make it executable first (`chmod +x DEscape-*.AppImage`,
  or tick "Allow executing" in your file manager's Properties). It is type-2,
  so it also needs `libfuse2` (`sudo apt install libfuse2` on Debian/Ubuntu),
  or run it with `./DEscape-*.AppImage --appimage-extract-and-run` if FUSE
  isn't available.
- **Linux, both formats:** v0.8 and later builds need glibc 2.28 or newer
  (Ubuntu 20.04, Debian 10, RHEL 8 or later); the v0.7 builds need 2.35
  (Ubuntu 22.04, Debian 12, Fedora 36 or later). On an older distro they fail
  with a `GLIBC_2.xx not found` error; run from source there instead.
- **Linux file dialogs:** when xdg-desktop-portal is running (most KDE and
  GNOME desktops), Open, Save As and the folder pickers use the desktop's own
  dialog; otherwise Qt's. To keep Qt's dialog anyway, start DEscape with an
  empty `QT_QPA_PLATFORMTHEME=` (for example
  `QT_QPA_PLATFORMTHEME= ./LAUNCH_DEscape_LinuxMac.sh`).
- **Windows:** the build is not code-signed, so SmartScreen may show "Windows
  protected your PC". Choose More info, then Run anyway. A console window opens
  alongside the editor; that is expected.
- **macOS:** the build is not signed or notarized, so Gatekeeper blocks it on
  first launch. Allow it under System Settings > Privacy & Security > Open
  Anyway, or clear the download flag with
  `xattr -dr com.apple.quarantine DEscape` on the unpacked folder. (UNTESTED!)

## Setup

Everything in this section is for running from source instead of a download.

### Quickest way to run it

No command line needed beyond installing Python itself. Double-click:
- **Windows:** `LAUNCH_DEscape_Windows.bat` (UNTESTED!)
- **Linux:** `LAUNCH_DEscape_LinuxMac.sh` (from a terminal: `./LAUNCH_DEscape_LinuxMac.sh`)
- **macOS:** `LAUNCH_DEscape_LinuxMac.sh`, but run it from Terminal
  (`./LAUNCH_DEscape_LinuxMac.sh`) — Finder usually opens a plain `.sh` file in a
  text editor instead of running it on double-click; renaming a copy to
  `LAUNCH_DEscape_LinuxMac.command` makes Finder execute it instead, if you want
  double-click to work there too. (UNTESTED!)

The only prerequisite is Python 3 itself — get it from
[python.org](https://www.python.org/downloads/) if `python3 --version` (or,
on Windows, `python --version`) doesn't already work in a terminal (Windows:
check "Add python.exe to PATH" during install). The launcher hands off to
`bootstrap.py`, which handles everything downstream of that automatically,
every time you run it: creating the `.venv` virtual environment the first
time, installing/updating the dependencies listed in `requirements.txt`, and
then starting the editor. If something goes wrong, the window it opens stays
open with an error message instead of just vanishing.

Because a double-click launch has no terminal attached, `bootstrap.py` shows a
small "Loading dependencies..." status window (via `tkinter`, part of the
Python standard library) while it works, so it doesn't look like nothing is
happening — otherwise the first launch in particular (creating the venv and
downloading dependencies) can take a while with no visible sign of it. If
`tkinter` isn't available (Debian/Ubuntu can split it into a separate
`python3-tk` package, the same way they split `venv` — see below) it falls
back to a console instead, opening one itself if none is already attached.

### Faster renderer (optional C compiler)

Part of the map renderer is compiled from `descape/_composite_native.pyx`.
The downloads already include it. From source, `bootstrap.py` builds it on
the first launch if it finds a C compiler ("Building the fast renderer, first
launch only..."), and rebuilds it after an update changes it. Without a
compiler DEscape still runs, on a slower built-in renderer; Help > Debug Log
says which one is active. To get the fast one, install:
- **Linux (Debian/Ubuntu):** `sudo apt install build-essential python3-dev`
- **macOS:** the Xcode Command Line Tools (`xcode-select --install`)
- **Windows:** [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/),
  with the "Desktop development with C++" workload

then launch again. A build that started and then failed isn't retried on
later launches until an update or a new Python version; building by hand
retries it now, and is also the way to get it with the manual setup below:
`pip install -r packaging/requirements-native.txt` in the venv, then
`python tools/build_native.py --force`.

### Manual setup

Only needed if you'd rather not use the launcher script above, or it doesn't
work for you (that's worth reporting as a real gap, not user error) — same
end result either way, just done by hand. Built to be cross-platform (Windows
/ macOS / Linux) — `pathlib` throughout, every dependency (PyQt5, Pillow,
numpy, PyYAML, AoE2ScenarioParser) is itself cross-platform, and
`map_editor.py` detects its own venv layout per OS.

All runtime dependencies (including a pinned
[AoE2ScenarioParser](https://github.com/KSneijders/AoE2ScenarioParser)) are
tracked in `requirements.txt` — one install step, no manual version juggling:

**If `python3 -m venv` fails** with `ensurepip is not available` (a real error
hit while building this tool, on Debian): Debian/Ubuntu split the `venv`
module out of the base `python3` package into a separate one. The error
message itself names the exact package for your Python version — install that
with `sudo apt install python3.11-venv` (or whatever version it names) and
re-run the venv command. Not something to fix inside a venv, since this is
what's blocking one from being created. Other Linux distros (Fedora, Arch),
plus the standard macOS and Windows Python installers, all bundle `venv` by
default and shouldn't hit this.

**Linux / macOS:**
```bash
cd ~/source/DEscape
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
cd DEscape
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
If `Activate.ps1` is blocked by execution policy, that's a PowerShell default,
not this project — `Set-ExecutionPolicy -Scope Process RemoteSigned` fixes it
for just the current shell. Use `python`/`py -3`, not `python3` — the official
Windows installer doesn't add a `python3` command the way Linux/macOS do.

**Verification note:** confirmed working end-to-end via
`LAUNCH_DEscape_LinuxMac.sh`'s own first-run path — a fresh `.venv`
with no packages installed,
`pip install -r requirements.txt` pulling everything (including
`AoE2ScenarioParser`) from PyPI for real, and the editor launching
successfully from it. Only checked on Linux so far — that covers the manual
steps and the logic `LAUNCH_DEscape_Windows.bat` shares with the `.sh` script,
but the `.bat` file itself has not been run on a real Windows machine; the
Windows steps above
are believed to be correct but unverified in practice.

`AoE2ScenarioParser` is version-pinned rather than just listed, specifically
because `descape/scenario_io.py` reaches into several of its *private* methods
(see below) — an unpinned install could silently pick up a version where those
methods changed or were removed. Confirmed that the PyPI `0.9.3` sdist
matches git tag `v0.9.3` (commit `29b8a25e8a345f1b1add3318b4c41cbfe35faf97`)
file for file, apart from `version.py`'s substituted version string — no
version-placeholder issues on PyPI's side; that's purely an artifact of
installing from a raw git clone instead (see below).

The pin is deliberately not `0.9.4`. Its new 1.43/1.57 effect fillers
(upstream #133) are only linked for 1.54, so any newly built effect on a 1.43
or 1.57 file (New, Copy, Retype, Paste) writes extra bytes and the saved
Triggers section no longer parses. The library's own `from_file` /
`write_to_file` round trip fails the same way; `0.9.3` does not.

`0.9.3` ships scenario definitions for versions 1.36 to 1.59. DEscape supplies
its own structure for 1.21 under `descape/versions/DE/`. The library's own
definition always wins over a repo one.

<details>
<summary>Alternative: install AoE2ScenarioParser from a local clone instead</summary>

Worth doing if you want its source available locally to read or debug against
— e.g. while extending `scenario_io.py`, which depends on several of its
private methods staying the way they are. Not needed for normal use.

The checked-in `pyproject.toml` in the git repo (unlike the PyPI release) has
a literal `<VERSION_HERE>` placeholder that only gets substituted during their
PyPI release process, so a raw clone needs one local patch first — a plain
Python snippet rather than `sed`, since GNU sed (Linux) and BSD sed (macOS)
don't even agree with each other on `-i` syntax, and Windows has no `sed` at
all:

```bash
git clone https://github.com/KSneijders/AoE2ScenarioParser.git ~/source/AoE2ScenarioParser
git -C ~/source/AoE2ScenarioParser checkout v0.9.3
python3 -c "import pathlib; p = pathlib.Path.home() / 'source/AoE2ScenarioParser/pyproject.toml'; p.write_text(p.read_text().replace('<VERSION_HERE>', '0.9.3'))"
pip install -e ~/source/AoE2ScenarioParser
pip install numpy Pillow PyQt5 PyYAML   # everything else from requirements.txt
```

This is what this project's own development environment actually uses (proven
working, unlike the plain `pip install -r requirements.txt` path above). If
you update the clone past `v0.9.3`, re-run `tools/dump_scenario.py` against a
real scenario and check the output still looks sane before trusting it.

</details>

## Usage

The easiest way to start the GUI is `LAUNCH_DEscape_LinuxMac.sh`/
`LAUNCH_DEscape_Windows.bat` (see "Quickest way to run it" above) — it doesn't
need a venv already set up or
activated. Once one exists (via either setup path), it can also be run
directly:

```bash
# GUI viewer -- works from anywhere, with any python (system or venv): it
# re-execs itself under this project's venv automatically if needed.
python3 map_editor.py    # Linux/macOS
python map_editor.py      # Windows
python3 map_editor.py examples/2_Joan_coop_2_v0_15.aoe2scenario   # open a file immediately
# A .aoe2scenario dropped onto the open window opens it too.

# CLI summary + PNG export
.venv/bin/python3 tools/dump_scenario.py path/to/scenario.aoe2scenario                       # Linux/macOS
.venv\Scripts\python.exe tools\dump_scenario.py path\to\scenario.aoe2scenario                 # Windows
.venv/bin/python3 tools/dump_scenario.py path/to/scenario.aoe2scenario --png out.png --scale 3

# Round-trip / tail-completeness check (now a pytest suite -- see tests/README.md)
.venv/bin/python3 -m pytest tests/test_scenario_io.py -m "corpus or slow"
```

**Using the editor.** DEscape has a built-in guide: Help > DEscape Guide (F1
by default) opens the page for the mode you are in, and Help > Keyboard and
Mouse lists your current keys and mouse gestures. The same pages are plain
Markdown in [`descape/help/`](descape/help/), readable on GitHub:
[Getting Started](descape/help/getting-started.md) covers opening, saving,
backups and autosave, and each mode and the map-wide tools have their own
page.

For XS scripts in Script Call triggers, including scripts longer than the
in-game editor accepts, see `docs/XS_SCRIPTING.md`.

## Scriptable batch edits

No GUI required: `descape/batch_api.py` is a small Python API for one-off
bulk edits driven by a scenario's units and terrain together (e.g. "raise
the ground under every one of a player's buildings") — the one capability
category with no in-game editor equivalent at all, since the in-game editor
has no scripting surface. Built on the same `scenario_io`/`scenario_write`
object model and write path the viewer uses, plus a few helpers that model
didn't already have: `buildings_of`/`is_building` (which unit_consts are
buildings), `neighbors` (adjacent-tile lookup, off-map-safe), `is_water`
(terrain-family classification), and
`raise_elevation_under`/`raise_elevation_under_many`/`set_terrain` (the
correct elevation clamp, an order-safe bulk variant, and double-terrain-blend
reset, matching what the viewer's own tools already enforce). `set_layer`
sets or removes a tile's layered terrain (the second terrain drawn over its
base) and leaves the base alone; call it after `set_terrain`, which still
clears the layer. `resize_map(scenario, 168, 168, "CENTER")` does what
Map > Resize Map… does (square maps only, not undoable).

**Unit edits persist only through a `UnitEditModel`.** `batch_api.save()`
always writes Map terrain exactly like the viewer's Save As does. Units are
written only when edited through a `descape.unit_model.UnitEditModel` passed as
`save(..., units=model)`; mutating a `Unit` object returned by `buildings_of()`
directly (its player, position, etc.) and calling `save()` silently drops that
edit, on purpose. See `descape/batch_api.py`'s module docstring for the full
detail. The model's `set_status(unit, value)` and `set_capture_flag(unit,
value)` set a unit's Unit State and Capture Flag by raw value (a bare
`unit.status = ...` is not saved); `set_capture_flag` raises on a scenario saved
before version 1.59, which has no such field.

`descape/scatter.py` places units at random, reproducible positions across a
tile set you build yourself (e.g. every water tile, or one connected pond):
`scatter_units(scenario, units, tiles, unit_const, count=40, seed=7)`. It never
randomizes `rotation`, which is a graphic-variant index rather than an angle for
many GAIA objects; pass `rotation_choices=` with values you've confirmed if you
need variety.

The same thing from the GUI is **Edit > Scatter Units in Region…**: pick an
object (and an owner) in the Units panel, then give it an area: outline one
with the Select tool in Terrain mode (the dialog then opens from any mode), or
select units in Units mode. With no outlined region the area is the selected
units' bounding box, widened on every side by the dialog's **Grow by** tiles (2
by default); with both, the dialog starts on the region and its **Area** row
switches to the units. It filters the area to eligible tiles (any tile, water
only, or one terrain type), optionally skips tiles already under a unit, and
places a count or a density of randomized copies as one undo step. The live "N
eligible tiles" line is exactly what will be placed. GAIA-only objects
(resources, trees, wild animals and the Others the game's editor lists only for
GAIA) always go to GAIA, whatever the owner. With no area at all the
menu item is greyed, and its tooltip says how to get one.

Its other mode, **Move the selected units**, takes the units selected in Units
mode instead and moves each one to a random eligible tile of the area, one
per tile while there is room, as one undo step. It is the default when the area
came from the selection. Only `x`/`y` change; rotation
and every other field are kept, buildings snap to whole tiles with their own
footprint (one that cannot fit stays put), and a garrison moves with its host.
The same seed and selection give the same result. From a script, it is
`scatter_existing_units(scenario, units, tiles, seed=7)`, which returns the
new positions and leaves applying them to you.

`batch_scripts/` has runnable examples:

```bash
# Raise the tile elevation under every one of Player 3's buildings by 1
.venv/bin/python3 batch_scripts/raise_elevation_under_buildings.py \
    examples/2_Joan_coop_2_v0_15.aoe2scenario out.aoe2scenario --player 3 --amount 1

# Recolor every DIRT_2 tile that borders water to BEACH
.venv/bin/python3 batch_scripts/recolor_dirt_near_water.py \
    examples/2_Joan_coop_1_v0_13.aoe2scenario out.aoe2scenario

# Reassign every unit_const 4 unit from Player 1 to Player 2
.venv/bin/python3 batch_scripts/reassign_units_by_const.py \
    examples/2_Joan_coop_2_v0_15.aoe2scenario out.aoe2scenario \
    --unit-const 4 --from-player 1 --to-player 2

# Scatter 40 GAIA fish across water, reproducibly (--pond-at X Y for one pond)
.venv/bin/python3 batch_scripts/scatter_fish_in_water.py \
    examples/2_Joan_coop_1_v0_13.aoe2scenario out.aoe2scenario --count 40 --seed 7

# Verify the API itself against a directory of scenarios
.venv/bin/python3 tools/verify_batch_api.py examples/
```

A script that processes many scenario files in one Python process (rather
than one file per run, like the examples above) should read
`descape/batch_api.py`'s module docstring first — it documents a real
AoE2ScenarioParser bug around creating new units (`add_unit`/`clone_unit`)
after an older-format file has been loaded earlier in the same process.

## Real terrain colors (optional)

By default terrain renders in hand-guessed flat colors (keyword-matched against
terrain names — approximate; e.g. it can't tell forest floor is brown, not
green, without more info). Pointing this tool at a real AoE2DE install gets you
the *actual* average color of each terrain's real texture instead — no visual
difference in code, `descape/terrain_palette.py`'s `color_for_terrain_id()` just
starts returning better numbers.

Easiest way: use the "AoE2DE install path" field in Edit > Settings…, General tab (type
or Browse… to it, then Load) — it validates the path and writes it to
`config.yaml` for you. Equivalently, copy `config.example.yaml` to
`config.yaml` yourself and set `aoe2de_install` to your AoE2DE install root
(or set `AOE2DE_INSTALL_PATH` instead — checked first). `config.yaml` lives
at the OS-standard per-user config location, not in the repo checkout:
- **Linux:** `$XDG_CONFIG_HOME/DEscape/config.yaml`, falling back to
  `~/.config/DEscape/config.yaml`
- **macOS:** `~/Library/Application Support/DEscape/config.yaml`
- **Windows:** `%APPDATA%\DEscape\config.yaml`

With an install configured the map draws each terrain's real texture, blends
each tile into higher-priority neighbouring terrain through the game's own
transition masks, and draws a tile's second ("layered") terrain over it.
View > Layers > Terrain Textures and Terrain Blending turn these off. The
blending is a first pass and does not match the game exactly.

In Terrain mode, the Target option on the Tool Options row chooses which of
the two a terrain tool edits (the in-game editor's Layering Mode). Terrain
paints the base terrain and clears the tile's layered terrain, Layer paints
the layered terrain and leaves the base alone, and Clear layer removes it.
Draw, Draw Line, Draw Rectangle and Paint Can follow it; the Eyedropper picks
the layered terrain while it is on Layer. Hovering a tile names its layered
terrain, and Paste Region carries it. A layered terrain is saved either way,
but only drawn with an install configured and both View > Layers rows on.

Every other control on the Tool Options row (brush size and shape, Trees and
Eye candy, Free placement, the Eraser's targets and the rest) starts each
launch at the default set in Settings > Tools. A change on the toolbar lasts
until DEscape closes; Load From Toolbar on that tab makes the current values
the defaults. Target always starts on Terrain.

No install configured, or a specific terrain missing from it, both fall back
to the guessed palette automatically — this is optional, not a hard
dependency.

DEscape can also read content from your AoE2:DE **profile folder** (the one
holding a numeric folder per account, each with its own `resources/` and
`mods/`), including your enabled subscribed and local mods. It is optional
and content lookup only reads it: without it, only the install's own
content is offered. It is usually found on its own: on Windows at
`%USERPROFILE%\Games\Age of Empires 2 DE`, and on Linux in the Proton prefix
beside the install's Steam library
(`steamapps/compatdata/813780/pfx/drive_c/users/steamuser/Games/Age of Empires 2 DE`).
To point DEscape elsewhere, use the profile folder row in **Settings > Mods**
(Browse…, then Load; Use auto-detect goes back to finding it on its own),
set `aoe2de_profile` in `config.yaml`, or set the `AOE2DE_PROFILE_PATH`
environment variable (checked first, and it overrides the Settings row). The
same tab lists every folder DEscape scans and how many mods the game's mod
list has disabled. Players mode's Personality list is built from these
folders: the install's AIs, plus any in your profile and enabled mods.

Cosmetic mods draw too: the art your enabled mods ship (trees, buildings,
units, walls, terrain textures, `.sld` or `.smx`) replaces the game's own on
the map, in the game's mod order, the top mod winning where two ship the
same file. **View > Layers > Mod Art** turns it off, and the Mod art list in
Settings > Mods leaves single mods out. That list also shows the mods the
game's own mod list disables, off until ticked: a ticked one draws in DEscape
only, at its place in the game's mod order, and the game's list is not
changed. Mods are read once per launch; after enabling or disabling one in
the game, restart DEscape or reload the profile folder.

Data mods (a mod that ships its own `resources/_common/dat/empires2_x2_p1.dat`,
the kind the game's lobby selects rather than enables) are picked in
**File > Data Mod**. Its units, stats, footprints, names and every other
table read from the game's data then come from the mod's `.dat`, the
window title names it, and a unit only the mod defines lists under its usual
tab with "(data mod)" after its name. The scenario itself records no data
mod, so DEscape remembers the pick for each saved scenario and applies it
when that file opens. The first pick of a mod reads its `.dat` in the
background (about 10-25 s, "Parsing …" in the status bar) and caches the
result in DEscape's own cache folder, so later picks are instant. An older
mod whose `.dat` genieutils-py cannot read logs why when picked, and stays
greyed out with the reason until DEscape restarts. Data mods apply to Age
of Empires II data only, not Return of Rome.

Typical Steam install locations, if you need a starting point for Browse…:
- **Windows:** `C:\Program Files (x86)\Steam\steamapps\common\AoE2DE`
- **Linux:** `~/.local/share/Steam/steamapps/common/AoE2DE` (or wherever your
  Steam library actually lives — this project's own dev path is
  `~/games/steam/steamapps/common/AoE2DE`, a non-default library location)
- **macOS:** AoE2:DE only got native Mac support in May 2026 (Apple Silicon
  only, via a Feral Interactive port) — likely
  `~/Library/Application Support/Steam/steamapps/common/AoE2DE` for the Steam
  version, but this project hasn't been checked against a real Mac install, so
  treat that path as a guess, not a confirmed default. If it's wrong,
  `validate_install_path()`'s error message will say exactly what's missing.

**Proprietary assets are never bundled in this repo** — `descape/asset_source.py`
is the one place that reads real `.dds` files from your configured install at
runtime; nothing from that install gets copied into the repo or committed. The
things that *are* committed are small factual correspondence tables (not
asset content): the `descape/*.json` files that the `tools/gen_*.py`
generators extract from the game's own `empires2_x2_p1.dat`, parsed with
[genieutils-py](https://github.com/SiegeEngineers/genieutils-py) (in
`requirements.txt`). The viewer reads no `.dat` for these tables, so an
install is not needed for them; genieutils-py runs at runtime only for
**File > Data Mod**, which derives a data mod's tables with these same
generators. The tables only need regenerating when the game updates its data.

They regenerate together with `tools/regen_game_data.py`. It runs every
generator against your install, replaces the tables only if all of them
succeed, and records what it ran against (the `.dat`'s hash and size, the
Steam build id, each table's hash) in `tools/game_data_stamp.json`:

```bash
.venv/bin/python3 tools/regen_game_data.py            # regenerate every table and the stamp
.venv/bin/python3 tools/regen_game_data.py --check    # would a regeneration reproduce the committed tables?
.venv/bin/python3 tools/regen_game_data.py --status   # quick: install and tables against the stamp, no generator runs
```

One table is regenerated with `--only <script>` (e.g. `--only
gen_unit_stats.py`) and committed together with the stamp: the test suite
fails on a table whose hash the stamp does not carry, and `--only` refuses to
run against a different game build than the stamp records. `--check` reports
whether the tables still match an install: it exits 0 when a fresh
regeneration reproduces them and the install matches the stamp, 1 when the
game data is stale (a new game build counts, even with equal tables), and 2 when it
cannot tell (no install found, or genieutils-py missing).

The first of the tables is `descape/terrain_texture_map.json`: a plain factual
`terrain_id -> filename` table from `tools/gen_terrain_texture_map.py`, naming
a real `.dds` file for each enabled terrain and cross-checked against several
terrain IDs already known from real scenario data (e.g. `terrain_id=19` is
`FOREST_PINE` in both AoE2ScenarioParser's enum and the `.dat` table's own
terrain name).

`descape/terrain_classes.json` is the second, from the same `.dat` and the same
kind of generator (`tools/gen_terrain_classes.py`): each terrain's water/land
family and climate, which is what lets the Draw tool's "Auto beach" option pick
a shoreline terrain from the game's own classification rather than guessing from
enum names. That distinction is not cosmetic — `BEACH_NON_NAVIGABLE` is
classified as land, and `FOREST_MANGROVE` and the rice-farm terrains as
shallows, both of which a name match gets backwards.

`descape/terrain_blend_map.json` is the third (`tools/gen_terrain_blend_map.py`):
each terrain's blend priority, blend type and layer mask file name, which
terrain blending reads to decide which neighbour bleeds into which and through
which of the install's masks.

A related SLP-icon decoder (`descape/slp_decoder.py`) exists from investigating
whether UI graphics could enhance the viewer too, parked rather than wired in.

`descape/sld_decoder.py` reads AoE2:DE's own in-world sprite format (`.sld`)
from your install, groundwork for drawing real unit and building graphics
instead of colored marks. Nothing calls it yet, so it has no effect on what
the viewer draws today.

## Contributing

Report bugs and suggest features in
[GitHub issues](https://github.com/Combinebobnt/DEscape/issues). Pull requests
are welcome; read [`CONTRIBUTING.md`](CONTRIBUTING.md) first.

## License

Copyright (C) 2026 Combinebobnt

GPL-3.0-or-later: this program is free software, redistributable and
modifiable under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at your
option) any later version. See `LICENSE` for the full text.

GPL rather than a permissive license because both core runtime dependencies
are themselves GPL-3.0 - AoE2ScenarioParser (its PyPI "MIT" classifier is
stale metadata; the license shipped in the wheel is GPLv3) and PyQt5 - so any
distributed combination of this tool with its dependencies is GPL territory
regardless.

The committed game-derived data files (`descape/terrain_texture_map.json`,
`terrain_classes.json`, `terrain_blend_map.json`, `tree_unit_ids.json`, `unit_render_data.json`,
`unit_graphic_map.json`, `unit_stats.json`, `object_catalog.json`) are factual id/name/filename correspondence tables, not
game asset content. The seven committed blank
`.aoe2scenario` files hold default structural data containing no authored
scenario content. Four under `tests/fixtures/` (`blank_120x120_v158`,
`golden_blank_120x120`, `real_blank_240x240` and `real_blank_480x480`) are
unit-stripped exports of the in-game editor's own "Blank" map style, and
`real_blank_120x120_v159` there is a blank map saved by the 1.59 game editor.
`descape/templates/blank_120x120.aoe2scenario`, which every new map starts
from, is the `blank_120x120_v158` export with its three version fields set to
1.59 by `tools/gen_blank_template.py`.
`descape/templates/blank_ror_120x120.aoe2scenario`, which every new Return
of Rome map starts from, is a blank map saved by the 1.59 Return of Rome
editor, unit-stripped the same way and with its creator name set to
"DEscape". The exports and the game saves are kept
as real game files rather than generated ones because
`tests/test_scenario_new.py` and `tests/test_blank_template.py` use them as
byte oracles (see `tests/README.md`). Nothing in this repository
grants any rights over Microsoft's Age of Empires II content; use of game
assets from your own install is governed by Microsoft's Game Content Usage
Rules.

The generated test map, `tests/fixtures/test_map_240x240.aoe2scenario`, is
built by `tools/gen_test_map.py` from the shipped
`descape/templates/blank_120x120.aoe2scenario` template.

## Credits

- **Age of Empires II**, created by Ensemble Studios, and **Age of Empires
  II: Definitive Edition**, from the studios and publisher behind it —
  DEscape is an unofficial, fan-made tool with no affiliation to or
  endorsement by Microsoft or any of them. See the License section above
  for what this repository does and doesn't grant rights to.
- [AoE2ScenarioParser](https://github.com/KSneijders/AoE2ScenarioParser) by
  KSneijders and contributors — the library this entire tool is built on;
  `descape/scenario_io.py`/`scenario_write.py` work around one of its
  version-support gaps rather than replacing it (see
  `docs/SCENARIO_IO.md`).
- [genieutils-py](https://github.com/SiegeEngineers/genieutils-py) by the
  SiegeEngineers organization — parses the game's own `.dat` file to keep
  several committed `descape/*.json` tables accurate, including
  `descape/terrain_texture_map.json` and `descape/terrain_classes.json` (see
  "Real terrain colors" above).
- [genie](https://github.com/fredreichbier/genie) by fredreichbier — its
  documentation of the classic SLP sprite format is the basis for
  `descape/slp_decoder.py`'s reimplementation (no code reused; see that
  module's docstring).
- [openage](https://github.com/SFTtech/openage) by the SFTtech authors — its
  documentation and reference decoders for AoE2:DE's SLD and SMX sprite
  formats are the basis for `descape/sld_decoder.py`'s and
  `descape/smx_decoder.py`'s reimplementations (no code reused; see those
  modules' docstrings). GPL-3.0-or-later, the same license as this tool.
