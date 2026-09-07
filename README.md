# DEscape

External viewer/editor for Age of Empires 2: Definitive Edition `.aoe2scenario`
files. Renders the terrain grid and unit placement, and supports terrain/elevation
editing with undo/redo and a scriptable batch-edit API, without needing the
in-game editor.

## Setup

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
are believed correct but unverified in practice.

`AoE2ScenarioParser` is version-pinned rather than just listed, specifically
because `descape/scenario_io.py` reaches into several of its *private* methods
(see below) — an unpinned install could silently pick up a version where those
methods changed or were removed. Confirmed via PyPI's metadata API that the
`0.8.3` release corresponds exactly to git tag `v0.8.3` (commit
`b763e2e37006bedab50c7d349b3ce24b9c2497f6`) — no version-placeholder issues on
PyPI's side; that's purely an artifact of installing from a raw git clone
instead (see below).

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
git -C ~/source/AoE2ScenarioParser checkout v0.8.3
python3 -c "import pathlib; p = pathlib.Path.home() / 'source/AoE2ScenarioParser/pyproject.toml'; p.write_text(p.read_text().replace('<VERSION_HERE>', '0.8.3'))"
pip install -e ~/source/AoE2ScenarioParser
pip install numpy Pillow PyQt5 PyYAML   # everything else from requirements.txt
```

This is what this project's own development environment actually uses (proven
working, unlike the plain `pip install -r requirements.txt` path above). If
you update the clone past `v0.8.3`, re-run `tools/dump_scenario.py` against a
real scenario and check the output still looks sane before trusting it.

</details>

## Prebuilt builds

Each [tagged release](https://github.com/Combinebobnt/DEscape/releases) has
frozen, no-Python-required builds attached: a `.tar.gz` for Linux/macOS, a
`.zip` for Windows, and a Linux `.AppImage`. These are built and tested by CI
on every `v*` tag push; the setup steps above are only needed if you'd
rather run from source.

The AppImage is type-2, so it needs either a `libfuse2` package installed
(most distros ship this, or `sudo apt install libfuse2` on Debian/Ubuntu) or
running it with `./DEscape-*.AppImage --appimage-extract-and-run` if FUSE
isn't available.

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

# CLI summary + PNG export
.venv/bin/python3 tools/dump_scenario.py path/to/scenario.aoe2scenario                       # Linux/macOS
.venv\Scripts\python.exe tools\dump_scenario.py path\to\scenario.aoe2scenario                 # Windows
.venv/bin/python3 tools/dump_scenario.py path/to/scenario.aoe2scenario --png out.png --scale 3

# Round-trip / tail-completeness check (now a pytest suite -- see tests/README.md)
.venv/bin/python3 -m pytest tests/test_scenario_io.py -m "corpus or slow"
```

Saving (Ctrl+S or File > Save) leaves a `.bak` and, on the first save of a
given file, a one-time `.orig` snapshot beside it, so the write path always
has something to fall back to.

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
reset, matching what the viewer's own tools already enforce).

**Only terrain edits persist.** `batch_api.save()` writes Map terrain
exactly like the viewer's Save As does — it does not write Units or the
trigger tail at all, regardless of what's in memory. `buildings_of()` hands
back real, mutable `Unit` objects so a script can *read* unit data to decide
*where* to edit terrain (that's what every example below does), but editing
a unit itself (reassigning its player, moving it, etc.) and calling `save()`
silently drops that edit — confirmed directly, not just documented. See
`descape/batch_api.py`'s module docstring for the full detail.

`batch_scripts/` has two runnable examples:

```bash
# Raise the tile elevation under every one of Player 3's buildings by 1
.venv/bin/python3 batch_scripts/raise_elevation_under_buildings.py \
    examples/2_Joan_coop_2_v0_15.aoe2scenario out.aoe2scenario --player 3 --amount 1

# Recolor every DIRT_2 tile that borders water to BEACH
.venv/bin/python3 batch_scripts/recolor_dirt_near_water.py \
    examples/2_Joan_coop_1_v0_13.aoe2scenario out.aoe2scenario

# Verify the API itself against a directory of scenarios
.venv/bin/python3 tools/verify_batch_api.py examples/
```

A script that processes many scenario files in one Python process (rather
than one file per run, like the two examples above) should read
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

Easiest way: use the "AoE2DE install path" field in the viewer's sidebar (type
or Browse… to it, then Load) — it validates the path and writes it to
`config.yaml` for you. Equivalently, copy `config.example.yaml` to
`config.yaml` yourself and set `aoe2de_install` to your AoE2DE install root
(or set `AOE2DE_INSTALL_PATH` instead — checked first). `config.yaml` lives
at the OS-standard per-user config location, not in the repo checkout:
- **Linux:** `$XDG_CONFIG_HOME/DEscape/config.yaml`, falling back to
  `~/.config/DEscape/config.yaml`
- **macOS:** `~/Library/Application Support/DEscape/config.yaml`
- **Windows:** `%APPDATA%\DEscape\config.yaml`

No install configured, or a specific terrain missing from it, both fall back
to the guessed palette automatically — this is optional, not a hard
dependency.

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
one thing that *is* committed is `descape/terrain_texture_map.json`: a plain
factual `terrain_id -> filename` table (not asset content), extracted once via
`tools/gen_terrain_texture_map.py` using
[genieutils-py](https://github.com/SiegeEngineers/genieutils-py) (`pip install
-r requirements-dev.txt`) to parse the real terrain table out of the game's own
`empires2_x2_p1.dat` — confirmed to cover all 131 enabled terrain IDs with a
real matching `.dds` file, cross-checked against several terrain IDs already
known from real scenario data (e.g. `terrain_id=19` is `FOREST_PINE` in both
AoE2ScenarioParser's enum and the `.dat` table's own terrain name). Only needed
again if the game updates its terrain table; not a runtime dependency.

A related SLP-icon decoder (`descape/slp_decoder.py`) exists from investigating
whether UI graphics could enhance the viewer too, parked rather than wired in.

`descape/sld_decoder.py` reads AoE2:DE's own in-world sprite format (`.sld`)
from your install, groundwork for drawing real unit and building graphics
instead of colored marks. Nothing calls it yet, so it has no effect on what
the viewer draws today.

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
`tree_unit_ids.json`, `unit_render_data.json`) are factual id/name/filename
correspondence tables, not game asset content. The four committed
`.aoe2scenario` files (`descape/templates/blank_120x120.aoe2scenario` and the
three blanks under `tests/fixtures/`) are unit-stripped exports of the in-game
editor's own "Blank" map style - default structural data containing no authored
scenario content. They are kept as real game exports rather than generated
files because `tests/test_scenario_new.py` uses them as byte oracles (see
`tests/README.md`). Nothing in this repository
grants any rights over Microsoft's Age of Empires II content; use of game
assets from your own install is governed by Microsoft's Game Content Usage
Rules.

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
  `descape/terrain_texture_map.json` accurate (see "Real terrain colors"
  above).
- [genie](https://github.com/fredreichbier/genie) by fredreichbier — its
  documentation of the classic SLP sprite format is the basis for
  `descape/slp_decoder.py`'s reimplementation (no code reused; see that
  module's docstring).
- [openage](https://github.com/SFTtech/openage) by the SFTtech authors — its
  documentation and reference decoder for AoE2:DE's SLD sprite format are the
  basis for `descape/sld_decoder.py`'s reimplementation (no code reused; see
  that module's docstring). GPL-3.0-or-later, the same license as this tool.
