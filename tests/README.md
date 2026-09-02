# tests/

pytest suite migrating `tools/verify_*.py`'s 44 checks into real pytest
tests. Currently mid-migration: Ordering steps 0-3 are done, step 4
onward is not started. See `tests/migration_manifest.py`'s own docstring
for the 44-check inventory and its `family`/`tier` bookkeeping.

## Running

```
./run_all_tests.sh                                  # default tier -- seconds, no corpus needed (Linux/Mac)
run_all_tests.bat                                   # same, for Windows
.venv/bin/python3 -m pytest                         # same thing, invoked directly

./run_corpus_quick.sh                                # corpus/gui tier, QUICK_CORPUS_NAMES only -- ~11 min, the default
.venv/bin/python3 -m pytest -m "corpus or slow"      # same thing, invoked directly (no flag needed)

./run_corpus_stress.sh                               # corpus/gui tier, the FULL corpus -- ~27 min
.venv/bin/python3 -m pytest -m "corpus or slow" --corpus-full

.venv/bin/python3 -m pytest --scenario-dir /path/to/other/examples -m "corpus or slow" [--corpus-full]
```

`run_all_tests.sh` / `run_all_tests.bat` (repo root) are thin wrappers
around the default tier only -- they exist so there's one obvious command
for "did I break anything," not a shortcut for the corpus tier, which
still needs the `-m` flag spelled out above.

`test_lint.py` (default tier -- runs ruff, see `ruff.toml`) fails outright,
not skips, if ruff isn't installed. `.venv/bin/python3 -m pip install ruff`
(`requirements-dev.txt`) before running the suite for the first time.

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
runtime the most) is what `-m "corpus or slow"` runs **by default now**,
with no flag needed. `F7_3_York` (220x220) was dropped from this set
2026-08-13; runtime dropped to about 11 minutes (measured: 11m02s), down
from the old ~15 min (13m40s) figure. Pass `--corpus-full` for the
complete corpus before a release or after touching rendering/write-path
internals specifically -- routine feature work only needs the quick
default. When iterating on a change that only touches one or two scripts,
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
| `slow` | full-render comparisons, O(pixels) sweeps | deselected (unused so far -- no Phase 1 entry needed it) |
| `gui` | offscreen `ViewerWindow`, needs PyQt5 | runs if PyQt5 imports; several default-tier tests are `gui`-only, not paired with `corpus` (`test_lazy_viewport.py`, `test_new_map.py`) |

`test_lint.py` (`ruff check .`, see `ruff.toml`) runs in the default tier
alongside these -- unmarked, since it's not a corpus/gui/slow concern, just
a fast static check.

A `pytest_terminal_summary` hook in `conftest.py` prints how much of the
run was deselected/skipped, so an empty `examples/` or a default `-m` run
doesn't silently read as full coverage.

`tests/test_scenario_new.py`'s one `corpus`-marked test
(`test_map_size_precedes_terrain_block_in_every_corpus_file`) is a cheap
per-file byte-offset assertion, not a render -- unlike most of this suite's
`corpus` tier, it doesn't meaningfully add to the ~27-minute runtime.

## Fixtures

- `descape/templates/blank_120x120.aoe2scenario`
  (`descape/scenario_io.BLANK_TEMPLATE_SIZES`) -- tracked in git so the
  default tier needs no corpus. The one shipped donor template `File > New
  Map` generates every size from (`descape/scenario_new.py` splices a resized
  terrain array into it; see that module's docstring) -- not test-only, so it
  lives with the other package data files instead of under `tests/`. Square,
  scenario v1.58, `terrain_write_supported`, and **unit-free** -- stripped
  from a real in-game "Blank" map export via `tools/strip_units.py`, since
  that map style actually seeds sparse `GRASS_GREEN` ground-scatter doodads
  rather than producing a literally empty map. `tests/fixtures/
  golden_blank_120x120.aoe2scenario` is the older, already-unit-free 120x120
  export that predates that discovery; kept as the strip tool's own no-op
  self-test golden reference (`tests/test_strip_units.py`).
  On the 120x120 donor specifically, every batch_api.py building-dependent
  check (6 of them:
  `check_raise_elevation_under_buildings`, `check_raise_elevation_clamp`,
  `check_raise_elevation_off_map`, `check_raise_elevation_under_many`,
  `check_raw_unit_mutation_does_not_persist`,
  `check_unit_edit_persists_through_the_model`) returns `None` ("no player
  has any on-map building") on this file, which the adapter maps to
  `pytest.skip`, not a pass. Those 6 show up as SKIPPED in the corpus-tier
  run against the real `examples/` corpus too, on this same file -- not a
  regression, a known gap. `tests/fixtures/units_120x120.aoe2scenario`
  below has a real building, but it does **not** close this gap:
  `tools/verify_batch_api.py`'s checks are `tier=("corpus",)` and
  parametrized only over `examples/*.aoe2scenario`
  (`tests/conftest.py`'s `pytest_generate_tests`), which never globs
  `tests/fixtures/`. A fixture *inside* `examples/` with real buildings/
  units/water/elevation variation would close it; see the plan's "Larger
  fixtures" section for the exact ask (not yet requested).
- `tests/fixtures/real_blank_{240x240,480x480}.aoe2scenario` -- real
  AoE2:DE "Blank" map exports (unit-stripped the same way as the donor
  above), moved from `descape/templates/` once `descape/scenario_new.py`
  could reproduce them from the 120x120 donor. **Byte oracles, not
  regenerable assets** -- `tests/test_scenario_new.py`'s central test
  reproduces them from the donor and compares byte-for-byte; regenerating
  these files *from* the donor would make that comparison a tautology and
  silently stop testing anything. If they're ever replaced, it must be with
  a fresh real game export, never a generator run.
- `tests/fixtures/triggers_120x120.aoe2scenario` -- the default tier's only
  scenario that actually has triggers. Generated from the 120x120 donor above
  by `tools/gen_trigger_fixture.py`; regenerate with
  `.venv/bin/python3 tools/gen_trigger_fixture.py` and commit the result.
  **A test input, not a byte oracle** -- the opposite of `real_blank_*` below,
  and regenerating it is expected rather than forbidden.
  `tests/test_trigger_fixture.py` pins the committed bytes against a fresh
  generator run, so a drifting generator shows up as a failure rather than
  silently.
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
  fixture above exists: every other tracked fixture is unit-free, and a
  unit-free file passes every phase 3.5a write-path claim trivially.
  Generated from the 120x120 donor by `tools/gen_units_fixture.py`;
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
  `reference_id`s have a deliberate gap, and `next_unit_id_to_place` sits
  above every assigned id, mirroring every real corpus file (a naive
  `len(units)`-based add path fails on either).
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
memoized global reset. `CONFIG_PATH` now resolves to the OS-standard
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

See the parent migration plan's "What a new feature's test looks like"
section for the per-TODO-version guidance. Rule of thumb: a check
belongs in the default tier unless it needs the real corpus or takes more
than a second. New `tools/verify_*.py` scripts are not the convention
going forward.

## Migration status

- Steps 0-3 done: pytest scaffolding, Phase 1 adapter (`conftest.py` +
  `test_legacy_adapter.py` + `migration_manifest.py`) wrapping all 44 (now
  45 -- phase 3.5a split one `verify_batch_api.py` check into two, see
  `migration_manifest.py`'s own docstring) checks with zero edits to the
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
