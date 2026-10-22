# tests/

pytest suite migrating `tools/verify_*.py`'s 44 checks into real pytest
tests. Currently mid-migration: Ordering steps 0-3 are done, step 4
onward is not started. See `tests/migration_manifest.py`'s own docstring
for the 44-check inventory and its `family`/`tier` bookkeeping.

## Running

```
./run_all_tests.sh                                  # default tier, in parallel -- ~2.5 min for ~7150 tests, no corpus needed (Linux/Mac)
run_all_tests.bat                                   # same, for Windows
./run_all_tests.sh -n 0                             # same tier, serially -- ~29 min
.venv/bin/python3 -m pytest                         # same tier, invoked directly: serial, see below
bash tools/ci_parity.sh [tag]                       # same tier plus the test map items, as CI sees them (corpus, install, examples/ hidden); run before every tag

./run_corpus_quick.sh                                # default tier plus the test map and slow items, in parallel; no examples/ needed
.venv/bin/python3 -m pytest -m "not corpus"          # same thing, invoked directly: serial
.venv/bin/python3 -m pytest -m test_map              # the test map items alone

./run_corpus_stress.sh                               # corpus/gui tier, the FULL corpus, plus the test map items -- ~27 min
.venv/bin/python3 -m pytest -m "corpus or slow or test_map" --corpus-full

.venv/bin/python3 -m pytest --scenario-dir=/path/to/other/examples -m "corpus or slow" [--corpus-full]
```

`run_all_tests.sh` / `run_all_tests.bat` (repo root) are thin wrappers
around the default tier only -- they exist so there's one obvious command
for "did I break anything," not a shortcut for the corpus tier, which
still needs the `-m` flag spelled out above.

**Parallel runs.** The wrappers run the tier under pytest-xdist
(`requirements-dev.txt`) with `-n auto --maxprocesses 8 --dist loadgroup`:
one worker per CPU, capped at 8 so the machine stays usable during a run.
`tools/ci_parity.sh` and CI cap at 16 instead. Arguments you pass come last
and win, so `-n 0` runs serially and `--maxprocesses 16` raises the cap.
Measured on a 32-thread, 61 GB machine (2026-09-25), all with the same 7093
passed, 50 skipped, 609 deselected:

| workers | wall time | extra memory in use |
|---|---|---|
| serial | 28:55 | 4.9 GB peak RSS |
| 8 (the wrappers' cap) | 3:56 | 5.9 GB |
| 16 (CI's cap) | 2:22 to 2:34 | 6.9 to 7.5 GB |
| 32 | 2:07 | 9.8 GB |

Past 16 workers the gain flattens (11% for 43% more memory), hence CI's cap.
`pytest.ini`'s addopts deliberately stays serial: `test_font_dpi_sweep.py`
and `test_collection_env.py` spawn child pytests that inherit addopts (and
`PYTEST_ADDOPTS`), so a `-n` there would fan out again in every child.
`--dist loadgroup` keeps the DPI sweep's seven cases on one worker so its
module fixture starts its seven subprocesses once, not once per worker.

A modal `QMessageBox` hang stalls one worker and with it the whole run, just
as silently as serially; wrap runs in `timeout 1800` when working near
trigger edits. To check no file depends on an earlier one having run, run
each alone in a fresh process (exit 5 is a file whose every test is
corpus/slow, so it is fine). From the repo root:

```
ls tests/test_*.py | xargs -P 16 -I{} sh -c '.venv/bin/python3 -m pytest --no-header -q -p no:cacheprovider {} > /dev/null 2>&1; s=$?; [ $s -eq 0 ] || [ $s -eq 5 ] || echo FAIL {}'
```

`test_lint.py` (default tier, runs `ruff check .`, see `ruff.toml`) fails
outright, not skips, if ruff isn't installed. `.venv/bin/python3 -m pip install
ruff` (`requirements-dev.txt`) before running the suite for the first time.
The selected ruleset is broad rather than syntax-only, so expect this test to
catch style and idiom findings (line length at 140, import order, bugbear,
security, datetime, modernization) as readily as real errors.

**Native composite kernel.** `test_native_composite.py` compares the
compiled kernel (`descape/_composite_native.pyx`) against the numpy code it
replaces, byte for byte, and skips when the kernel isn't built:
`.venv/bin/python3 tools/build_native.py` builds it (cython from
`requirements-dev.txt`). `DESCAPE_REQUIRE_NATIVE=1` turns that skip into a
failure (CI and `ci_parity.sh` set it, and build first).
`DESCAPE_COMPOSITE=numpy` or `native` forces the backend the rest of the
suite renders with, and decodes .sld sprites with. `test_sld_native.py` does
the same for the kernel's SLD walk and delta-chain decode against
`sld_decoder`'s numpy path. Run the tier once each way after a kernel change.

**Anything that rebuilds `.venv` drops the dev dependencies, pytest
included** -- `bootstrap.py` (what the `LAUNCH_DEscape_*` launchers run)
installs `requirements.txt` only, correctly, since it sets up an end-user
install rather than a development one. So a cold launcher run, or deleting
`.venv` by hand, leaves `.venv/bin/python3 -m pytest` reporting
`No module named pytest` until you reinstall:
`.venv/bin/python3 -m pip install -r requirements-dev.txt`. The app itself
works fine in that state, which is what makes it confusing.

Bare `pytest` (not `.venv/bin/python3 -m pytest`) is not the documented
invocation -- this repo has no `pyproject.toml` on purpose and isn't
pip-installable, so nothing guarantees `pytest` on PATH resolves to this
venv's install.

**The full corpus/gui tier (`--corpus-full`) takes about 27 minutes**
against the real 18-file `examples/` corpus (full-canvas render comparisons
on files up to 480x480, `~30` random rects per file per check, etc.) --
expensive enough that running it after every routine change was its own
problem. `conftest.py`'s `QUICK_CORPUS_NAMES` (`2_Joan_coop_1`,
`C2_ElCid_coop_1`, `F7_2_Dos Pilas` -- a small cross-section of real
map-size bands, including the 480x480 outlier that dominates full-corpus
runtime the most -- plus the 200x200 scenario-1.37 `0_June_Event_Scenario`,
the only file the Number of Players gate refuses) is what `-m "corpus or slow"` runs
when invoked directly with no flag. `F7_3_York` (220x220) was dropped from this set
2026-08-13; runtime dropped to about 11 minutes (measured: 11m02s), down
from the old ~15 min (13m40s) figure. Pass `--corpus-full` for the
complete corpus before a release or after touching rendering/write-path
internals specifically -- routine feature work only needs the quick
default. `run_corpus_quick` runs no corpus file at all: the `examples/` maps
run only in `run_corpus_stress`. When iterating on a change that only touches one or two scripts,
`-k` narrows further on top of either scope -- e.g. `-m "corpus or slow"
-k "viewer_pick or copy_paste"` re-verifies just the gui-tier tests against
the quick set in under a minute. Check which modules a change can possibly
affect first (e.g. `grep -l "import settings" descape/*.py`) rather than
assuming the full corpus needs re-running.

## Tiers

| Marker | Contents | Default |
|---|---|---|
| _(none)_ | synthetic + fixture-based tests | runs |
| `corpus` | needs the real `examples/` corpus | deselected; `-m "corpus or slow"` to opt in, restricted to `QUICK_CORPUS_NAMES` (~11 min) unless `--corpus-full` (~27 min) is also passed |
| `slow` | full-render comparisons, O(pixels) sweeps | deselected |
| `test_map` | runs on the generated test map or its game resave (see below) | deselected; `run_corpus_quick` and `run_corpus_stress` add it, `-m test_map` runs it alone with no `examples/` |
| `gui` | offscreen `ViewerWindow`, needs PyQt5 | runs if PyQt5 imports; several default-tier tests are `gui`-only, not paired with `corpus` (`test_lazy_viewport.py`, `test_new_map.py`) |
| `font_sensitive` | wrap/fit geometry checks on the Map Options and trigger property forms, and the Units, Players, Diplomacy and View info-page minimum-width checks | runs at the pinned 96 DPI; `test_font_dpi_sweep.py` also re-runs them at 72-168 DPI, one subprocess per DPI |

`test_lint.py` (`ruff check .`, see `ruff.toml`) runs in the default tier
alongside these, unmarked, since it's not a corpus/gui/slow concern. It stays
a fast static check despite the breadth of the ruleset: a cold `--no-cache`
run over all 348 linted files takes well under a second.

The default tier is meant to give the same result on any machine.
`testkit/qt_window.ensure_qapp()` pins the app font to the bundled
`testkit/fonts/DejaVuSans.ttf` at 12pt and hard-sets `QT_FONT_DPI` to 96, so
a shell's own `QT_FONT_DPI` has no effect; only `DESCAPE_TEST_FONT_DPI`
(which the sweep sets) overrides it. The session header prints the resolved
font, DPI and one reference string's advance. Two guard tests back this up:
`test_collection_env.py` collects the whole suite with `HOME`,
`XDG_CONFIG_HOME` and `AOE2DE_INSTALL_PATH` hidden and an empty
`--scenario-dir`, and fails on any collection error, and
`test_font_dpi_sweep.py` runs the `font_sensitive` checks across DPIs.
Pass `--scenario-dir` in its `=` form when the path is outside the repo and
no test path follows: otherwise the bare directory becomes pytest's rootdir
and the option goes unrecognised.

A `pytest_terminal_summary` hook in `conftest.py` prints how much of the
run was deselected/skipped, so an empty `examples/` or a default `-m` run
doesn't silently read as full coverage.

A per-file test (one that takes `scenario_path`) without a `corpus` mark
runs first on the generated test map and its game resave, as `[map240]`
and `[test_map_game_resave]` items marked `test_map`, then on each corpus
file as a `corpus`-marked item. So the default tier runs exactly what it
ran before the map existed, and `run_corpus_quick` and `run_corpus_stress`
add the map items. A named-file test's map sibling (see Fixtures), and every test in
`test_test_map_fixture.py` and `test_test_map_game_resave.py`, carries
`test_map` itself; `test_test_map_collection.py` fails on a test that reads
`conftest.TEST_MAP_PATH` or `TEST_MAP_RESAVE_PATH` without it. A test that keeps
`corpus` (a census of real files, a real-file baseline, an install-only
check, or one that would only skip on the map) gets the corpus files alone,
with a one-line comment saying why.

## Fixtures

Each generated file below is pinned twice against its generator, as is the
New Map splice against the real 240/480 exports: header plus decompressed
body, which holds on any zlib, and a `..._byte_identical`
sibling on the compressed bytes, which skips (`testkit/zlib_runtime.py`)
when the interpreter's zlib is zlib-ng, since its deflate output differs
from stock zlib's while inflating to the same body. A sibling failing on CI
while its partner passes means the file was regenerated off stock zlib:
regenerate it on stock zlib.

- `descape/templates/blank_120x120.aoe2scenario`
  (`descape/scenario_io.BLANK_TEMPLATE_SIZES`) -- tracked in git so the
  default tier needs no corpus. The one shipped donor template `File > New
  Map` generates every size from (`descape/scenario_new.py` splices a resized
  terrain array into it; see that module's docstring) -- not test-only, so it
  lives with the other package data files instead of under `tests/`. Square,
  scenario v1.59, `terrain_write_supported`, and **unit-free**. Derived, not
  exported: `tools/gen_blank_template.py` generates it from the 1.58 donor
  below by setting the three fields that carry a scenario's version
  (`FileHeader.version`, `DataHeader.version` and the Triggers section's
  `trigger_version`); regenerate with
  `.venv/bin/python3 tools/gen_blank_template.py` and commit the result.
  `tests/test_blank_template.py` pins it three ways: against a fresh
  generator run, against the game-saved 1.59 oracle below, and against the
  newest scenario version DEscape has a trigger vocabulary for, so a library
  pin that brings a new version fails there until the template is
  refreshed. `tests/fixtures/
  golden_blank_120x120.aoe2scenario` is the older, already-unit-free 120x120
  export that predates that discovery; kept as the strip tool's own no-op
  self-test golden reference (`tests/test_strip_units.py`).
  The six building-dependent `verify_batch_api` checks would return `None`
  ("no player has any on-map building") on this unit-free template, which
  the adapter maps to `pytest.skip`, so it is not their input. The generated test map
  (`tests/fixtures/test_map_240x240.aoe2scenario`) has buildings, water
  beside dirt and elevation, so all nine `verify_batch_api` path checks now
  run on it in the quick and stress tiers (`map240`).
- `descape/templates/blank_ror_120x120.aoe2scenario`
  (`descape/scenario_io.BLANK_ROR_TEMPLATE_PATH`): the Return of Rome
  donor every Return of Rome `File > New Map…` choice splices from (GH
  #243). A blank 120x120 map saved by the 1.59 Return of Rome editor,
  unit-stripped via `tools/strip_units.py`, with its creator name set to
  "DEscape". **Not generated**: refresh it by hand from a newer game save
  (the steps are on `BLANK_ROR_TEMPLATE_PATH`). `tests/test_scenario_new.py`
  pins its preconditions (RoR header, unit-free, blank terrain, the creator
  name), and `tests/test_blank_template.py` fails once a newer scenario
  version is supported until it is refreshed.
- `tests/fixtures/blank_120x120_v158.aoe2scenario`: the 1.58 donor, and the
  shipped template's own bytes until 1.59. A real in-game "Blank" map export,
  unit-stripped via `tools/strip_units.py`, since that map style actually
  seeds sparse `GRASS_GREEN` ground-scatter doodads rather than producing a
  literally empty map. It is the source `tools/gen_blank_template.py` reads,
  the donor of the three 1.58 fixtures below (`triggers_120x120`,
  `find_triggers_120x120`, `units_120x120`), and the donor
  `tests/test_scenario_new.py` reproduces the two real 1.58 exports from.
  **A real export, never regenerated.**
- `tests/fixtures/real_blank_120x120_v159.aoe2scenario`: a blank 120x120 map
  saved by the 1.59 game editor, untouched. **A byte oracle**:
  `tests/test_blank_template.py` requires the shipped template to equal it
  except in three fields a template inherits from its own save (the save
  timestamp, the stored filename and Player 1's editor camera, which was
  moved off the centre in this save). The same comparison run on the 1.58
  donor fails and names the three version fields, which is that test's
  permanent negative control. Replace it only with a newer game save, never
  with a generator run.
- `tests/fixtures/real_blank_{240x240,480x480}.aoe2scenario` -- real
  AoE2:DE 1.58 "Blank" map exports (unit-stripped the same way as the 1.58
  donor above), moved from `descape/templates/` once `descape/scenario_new.py`
  could reproduce them from the 120x120 donor. **Byte oracles, not
  regenerable assets** -- `tests/test_scenario_new.py`'s central test
  reproduces them from the 1.58 donor and compares byte-for-byte (header and
  decompressed body everywhere, compressed bytes on stock zlib); regenerating
  these files *from* the donor would make that comparison a tautology and
  silently stop testing anything. If they're ever replaced, it must be with
  a fresh real game export, never a generator run.
- `tests/fixtures/triggers_120x120.aoe2scenario` -- the default tier's only
  scenario that actually has triggers. Generated from the 1.58 donor above
  by `tools/gen_trigger_fixture.py`; regenerate with
  `.venv/bin/python3 tools/gen_trigger_fixture.py` and commit the result.
  **A test input, not a byte oracle** -- the opposite of `real_blank_*` below,
  and regenerating it is expected rather than forbidden.
  `tests/test_trigger_fixture.py` pins the committed bytes against a fresh
  generator run, so a drifting generator shows up as a failure rather than
  silently. Trigger 3 also carries the map coordinates
  `tests/test_trigger_geometry.py` reads: a condition area, a patrol (area +
  location), a location-only effect, a half-set area and a whole-map area.
  After those come the placed-unit references `tests/test_unit_references.py`
  resolves, backed by three Player 1 units (two archers and a house, ids
  500-502): a Destroy Object on the house, a Task Object whose selection and
  location object name the three, a Patrol with a selection and no location,
  and a Destroy Object naming id 999, which nothing carries.
  Four triggers, chosen for what each exercises rather than for volume:
  trigger-to-trigger references (so a reorder/delete has something to remap),
  an armour effect whose quantity is bit-split across
  `_quantity_int`/`_quantity_float`/`variable`, a named variable, and populated
  str32 fields. **Empty strings are written the game's way (length 0), not the
  library's (length 1 holding a NUL)** -- see the generator's
  `_game_style_bytes()`. Without that one difference the file is in
  AoE2ScenarioParser's own normal form, re-serializes through it perfectly, and
  the default tier cannot tell the phase 4b byte-blob write path apart from
  whole-section re-serialization (verified by mutation, both ways).
- `tests/fixtures/units_120x120.aoe2scenario` -- the default tier's only
  scenario with a non-trivial Units section, for the same reason the trigger
  fixture above exists: every other tracked fixture is unit-free (bar the
  trigger fixture's three, which only back its unit references), and a
  unit-free file passes every phase 3.5a write-path claim trivially.
  Generated from the 1.58 donor by `tools/gen_units_fixture.py`;
  regenerate with `.venv/bin/python3 tools/gen_units_fixture.py` and commit
  the result. **A test input, not a byte oracle**, same convention as
  `triggers_120x120.aoe2scenario`. `tests/test_units_fixture.py` pins the
  committed bytes against a fresh generator run.
  8 units: two GAIA trees and a GAIA wall covering both `rotation`
  variant-index encodings AGENTS.md names (a plain integer, and the same
  index as `k*2pi/5` radians -- neither is an angle), a real on-map
  building plus two more units for Player 1 (one garrisoned in that
  building), and two units for Player 2 (one with a non-empty caption --
  the only coverage anywhere for the caption normalizer's unmeasured
  non-empty branch, see `descape/unit_model.py`'s `_serialize_unit`
  docstring). Empty captions are written the game's way (length 0), not the
  library's (length 1 holding a NUL) -- same `_game_style_bytes()` helper
  the trigger fixture uses, now shared via `tools/_fixture_bytes.py`.
- `tests/fixtures/v159_units_triggers.aoe2scenario` -- the default tier's
  only scenario version 1.59 file with units and triggers, read through
  AoE2ScenarioParser's own v1.59 definitions. Generated from the shipped
  1.59 template by `tools/gen_v159_fixture.py` (units and triggers added
  through the library; the template already carries the version, trigger
  version included); regenerate with
  `.venv/bin/python3 tools/gen_v159_fixture.py` and commit the result. **A
  test input, not a byte oracle.** `tests/test_v159_load_path.py` pins it
  against a fresh generator run. 11 units on GAIA and Players 1-2 carrying
  every capture setting (-1..3), ordered so list neighbours always differ,
  and two triggers with an Object Visible Multiplayer condition, one with
  `allow_in_fog=1` between two timers. That ordering is what makes the
  slot-shift regression tests bite: they guarded `descape/unlinked_fields.py`
  on the 0.8.3 pin and now prove the library's own linking of both fields.
  `reference_id`s have a deliberate gap, and `next_unit_id_to_place` sits
  above every assigned id, mirroring every real corpus file (a naive
  `len(units)`-based add path fails on either).
- `tests/fixtures/test_map_240x240.aoe2scenario`: a 240x240 v1.59 map of
  every catalog object, terrain, trigger condition and effect plus bug-bait
  regions (map edges and corners, stacks, cliffs, walls, gates, garrisons,
  odd values), built through the app's own edit models and tools from the
  shipped template by `tools/gen_test_map.py`; regenerate with
  `.venv/bin/python3 tools/gen_test_map.py` and commit the result. **A test
  input, not a byte oracle**: every serialization change regenerates it.
  64,437 bytes. `tests/test_test_map_fixture.py` pins it against a fresh
  build (about 2.6 s), checks it holds every catalog and vocabulary entry,
  and holds each of the generator's `REGIONS` and `SECTIONS` to its one-line
  "why". Those checks are `test_map`, so the default tier does not notice a
  stale map (CI and `tools/ci_parity.sh` run them as a second step): after
  a serialization or `tools/gen_test_map.py` change, run
  `./run_corpus_quick.sh` (or `-m test_map` alone). The run also writes
  `build/test_map_bisect/` (each risky region or section alone, for an
  in-game check); `--no-bisect` skips that.
  **`REGIONS` is the contract for finding a feature**: a test that needs
  one (the walls, a gate, the elevation staircase) reads its rectangle from
  `conftest.load_verify_module("gen_test_map").REGIONS` rather than
  hard-coding tiles, and the inventory test keeps each region holding what
  its "why" says. Besides the `[map240]` items of per-file tests, a corpus
  test that names a real file or takes `corpus_files` may have a separate
  `test_map` sibling on the map next to it; the corpus original stays
  alongside it.
- `tests/fixtures/test_map_game_resave.aoe2scenario`: the test map opened
  and saved once in the AoE2:DE 1.59 editor. Never regenerated. Every
  per-file test that runs on the map also runs on it (param id
  `test_map_game_resave`), and `tests/test_test_map_game_resave.py`
  compares the two, listing each change the game makes on save. When a
  generator change moves one of those, update the expected drift there or
  refresh the resave with a new in-game save.
- `tests/fixtures/test_map_script.xs`: an empty-`main()` stub for the
  `script_name` the map names. DEscape never embeds script content, so to
  open the map in AoE2:DE without an "XS: Error 0354: failed to open file"
  dialog, copy it into the game profile's `resources/_common/xs/` first.
- `examples/` -- untracked, gitignored, opt-in. The real 18-file corpus
  (a mix of scenario versions, sizes up to 480x480). Absent on a fresh
  clone or worktree; the corpus tier reports why it skipped rather than
  silently passing. `conftest.py`'s `QUICK_CORPUS_NAMES` is a checked-in
  list of 3 filenames from within this corpus, not a separate directory or
  fixture -- these are third-party scenario files, not this repo's own
  content, so nothing under `examples/` itself is ever committed. The quick
  tier is "the full corpus, filtered by name" rather than its own asset.

## `config.yaml` isolation

Every test gets `descape.settings.CONFIG_PATH` (and `asset_source`'s own
copy) redirected to a per-test `tmp_path/config.yaml` via `conftest.py`'s
autouse `_isolated_settings` fixture, and every `settings.py` module-level
memoized global reset. That reset list is
`testkit.settings_isolation.MEMOIZED_GLOBALS`, shared with the `tools/gen_*`
capture scripts (which call its `isolate_settings()`, having no fixture to
lean on) and guarded by `test_settings_isolation.py`, which discovers the real
set from `settings.py` and fails when the list has drifted.
`CONFIG_PATH` now resolves to the OS-standard
per-user config location rather than a repo-relative path, which makes
this redirect more important, not less -- an unredirected write now lands
in the developer's real `~/.config/DEscape/` (or platform equivalent)
instead of a gitignored file inside the checkout. This is load-bearing,
not just hygiene: found by
tracing (not by reading) that `ViewerWindow.closeEvent()` unconditionally
calls `settings.set_window_size()` on every window close, which without
this redirect writes straight through to the developer's real
`config.yaml` on every gui-tier test. Confirmed via md5sum/mtime that
Phase 1's first two corpus runs already did this to the real file before
the fixture existed -- silently harmless only because the offscreen
window's size happened to round-trip to the same bytes already on disk.

Side effect worth knowing about, not yet a deliberate decision: because
the redirected config starts empty, `settings.get_graphics_quality()` and
`settings.get_elev_step_pct()` now resolve to their hardcoded defaults for
every test, rather than whatever the developer last saved via the Settings
dialog. That's a real determinism improvement (the migration plan's hazard
6 asks for exactly this -- render output pinned to a stated quality rather
than "whatever the last developer saved") but it's implicit today. Phase
3's render-touching migrations should pin `graphics_quality`/
`elev_step_pct` to an explicitly stated value in the test itself rather
than relying on "the redirected config happens to be empty."

## Process-global hazard: verify modules share one interpreter

All 11 `tools/verify_*.py` modules import into the *same* pytest process
now, where each used to be its own standalone script invocation.
`verify_batch_api.py`'s own module docstring (lines 43-60) documents that
AoE2ScenarioParser does a **class-level** property swap
(`Unit.caption_string_id`/`caption_string`) the first time any
older-format (pre-1.54) scenario loads in a process, and that swap is
never reverted for later scenarios in the same process. Under the
standalone scripts this was invisible (one process per script). Under
pytest, corpus-tier test *ordering* can now matter in a way it never did
before -- if a corpus-tier failure doesn't reproduce when run standalone
or in isolation (`-k` to just that test), suspect this before suspecting
the check itself.

## Adding a test

Test logic against synthetic data, or a file from `tests/fixtures/` (copied
to `tmp_path` before anything writes to it), in the default tier. Mark a test `gui` only for the part
that needs a live `ViewerWindow`, and `corpus` only when it needs the real
scenario corpus. Rule of thumb: a check
belongs in the default tier unless it needs the real corpus or takes more
than a second. A test that reads the test map is marked `test_map` whatever
its runtime, so it runs in the quick and stress tiers, never the default (see Tiers).
New `tools/verify_*.py` scripts are not the convention
going forward.

## Migration status

- Steps 0-3 done: pytest scaffolding, Phase 1 adapter (`conftest.py` +
  `test_legacy_adapter.py` + `migration_manifest.py`) wrapping all 44 (now
  46 -- phase 3.5a split one `verify_batch_api.py` check into two and
  `set_layer` added one, see `migration_manifest.py`'s own docstring) checks
  with zero edits to the
  verify scripts, new tests #1-3 (`test_private_api_guard.py`,
  `test_edit_history.py`, `test_settings.py`).
- Step 4 done: `generate_reference_pngs`/the two benches extracted into
  `tools/gen_iso_reference_pngs.py` / `tools/bench_incremental_latency.py` /
  `tools/bench_chunk_px.py`. All three source verify scripts
  (`verify_iso_render.py`, `verify_iso_incremental.py`, `verify_iso_chunks.py`)
  keep every `check_*` function untouched; only the extracted function and
  its `main()` wiring were removed. `migration_manifest.py`'s
  `generate_reference_pngs` entry now records `tool:tools/gen_iso_reference_pngs.py`
  as its `migrated_test_nodeid` -- the only entry filled in so far.
- Phase 3 (the real per-script test migrations, step 5 onward) is underway:
  2 of 11 scripts fully migrated and deleted (`verify_roundtrip.py`,
  `verify_write_path.py`), 9 not started. Each script migration is its own
  session's work, not a batch -- see the corpus-tier runtime note above for
  why. Per-script review tracking (commit hashes, review dates) isn't part
  of the public repo, a deliberate call; this section is the migration as
  a whole.
- `conftest.py`'s `load_verify_module` already tolerates a verify script
  being deleted (skips with a named reason rather than erroring) -- this
  covers the step 8/9 window where some scripts are gone and others
  aren't, so no further adapter change is needed once deletions start.
