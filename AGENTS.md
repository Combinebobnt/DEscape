# AGENTS.md

External viewer/editor for Age of Empires 2: Definitive Edition
`.aoe2scenario` files. Start with `README.md` — it covers setup, build/test
commands, and architecture.

## Hard rules

Before changing any code that writes, reads or draws a unit's `rotation`,
`initial_animation_frame`, `unit_const` or `garrisoned_in_id`, or adding a
write path, read `docs/WRITE_RULES.md`: it holds each exception's exact
scope.

- Never write into a subscribed-mods `mods/subscribed/` folder: the game's
  mod manager re-syncs it, so an edit there is silently lost. Every path that writes a scenario
  file (Save, autosave, and the tools that write one directly) and every
  CSV, theme or settings export refuses an output path with the consecutive parts
  `mods`, `subscribed` (case-insensitive), as written or resolved through
  symlinks (`scenario_io.is_workshop_subscribed()`). A new one must too.
  The game profile itself (Proton `compatdata/` on Linux,
  `Games/Age of Empires 2 DE` on Windows) is writable, but Save and Save As
  there always write `.bak`/`.orig` (`scenario_io.is_in_game_profile()`),
  even with backups switched off, and sidecar autosave puts no slot beside
  a scenario there. Autosave's own `write_scenario(..., backup=False)`
  stays as is. The template tools (`strip_units` and the generators) need
  no backup there: they are developer tools, not part of the app.
- GAIA units' `rotation` field is not an angle for ~65% of GAIA objects — it's a
  tree/doodad graphic-variant index (integer values well outside `[0, 2π)`, e.g.
  7..53). Every write path must pass it through verbatim, never normalize it.
- **Per game dataset (GH #67).** Return of Rome scenarios (header variant 2)
  are edited with Return of Rome game data (`descape/game_dataset.py`), and
  the viewer refuses every edit and save of a document shown with the other
  game's data (`ViewerWindow.game_data_lock_reason()`), while the model
  guards below still raise in a batch script. Every const set the rules
  below name is the active dataset's (under AoE2, a loaded data mod's, GH
  #58), activated by the viewer on open and by a script through
  `batch_api.load()`, not a bare `scenario_io.load_map_and_units()`.
- **The exceptions to "verbatim".** This list is
  exhaustive: every other write path stays verbatim as above, and each
  exception that goes through a model method raises rather than silently
  no-op'ing outside its scope, which is what keeps the rule enforced rather
  than merely documented. Each entry: name, guard, pinning test.
  - *Rotate*: `UnitEditModel.set_rotation()`, ANGLE consts only (the .dat's
    creatable type 70 plus four trebuchets, `descape/unit_rotation.py`);
    `tests/test_unit_model.py`, the whitelist re-measured by
    `tests/test_unit_rotation.py`'s corpus test.
  - *Region turn's unit images* (GH #197): `region_transform.transform_block()`,
    new units only, the eight D4 elements only; `tests/test_region_transform.py`.
  - *Selection transform's in-place unit images* (TASK-256):
    `region_transform.plan_in_place()` plus the three setters' own guards;
    `tests/test_region_transform.py`.
  - *Map mirroring's unit images*: `mirror_tools.plan_mirror_units()`, new
    units only, no model guard; `tests/test_mirror_units.py`.
  - *Cycle Variant*: `UnitEditModel.set_variant()`, cyclable consts only
    (`descape/unit_variant.py`), never a wall, cliff or gate;
    `tests/test_unit_variant.py`, `tests/test_unit_model.py`.
  - *Wall shapes re-derived from neighbours*:
    `UnitEditModel.set_wall_variant()`, wall-family consts and an index in
    `0..4` only; `tests/test_wall_run.py`.
  - *Cliff drag's join rewrites* (GH #44, GH #247):
    `UnitEditModel.set_cliff_frame()`, cliff consts only, a frame inside the
    const's count; frames chosen by `cliff_chain`; `tests/test_wall_run.py`,
    `tests/test_cliff_chain.py`.
  - *Replace* (GH #144): `UnitEditModel.replace_type()`, by
    `unit_model.replaced_rotation()`; `tests/test_unit_replace_type.py`.
  - *Place Unit's placement rotation* (GH #172, and the Unit Brush, GH
    #170): `place_rotation.rotation_for()`; `tests/test_place_rotation.py`.
  - *Terrain planting's variant*: `terrain_units.variant_for()`;
    `tests/test_terrain_units.py`.
- Gates carry no rotation at all: every visible gate const has
  `angle_count == 1`, and a gate's orientation lives in its `unit_const`,
  one const per orientation, so "rotating" a gate means swapping the const
  among its four siblings, which changes the footprint. Top-down Flat's
  `unit_sprites.flat_art()` is draw-only, for gates and walls alike: no
  write path may use it.
- **The two exceptions to "a placed unit's `unit_const` never changes".**
  `UnitEditModel.set_unit_const()` and
  `UnitEditModel.replace_type()` are the only code anywhere that may change an
  existing unit's `unit_const`. Each raises outside its scope rather than
  silently no-op'ing, and that guard is what keeps this rule enforced rather
  than merely documented.
  - `replace_type()`: Edit > Find and Replace only (GH #144), refusals in
    `replace_refusal()`; `tests/test_unit_replace_type.py`.
  - `set_unit_const()`: a gate's four orientation siblings only
    (`descape/gate_orientation.py`), re-anchored on the footprint's low
    corner; `tests/test_unit_model.py`.
  - Not exceptions: Map mirroring and a pasted region's turn, which `add()`
    new gates.
- Walls carry shapes, not facings: a wall graphic's five stored frames
  (`angle_count == 5`) are SHAPES, so `rotation` selects a variant
  (confirmed in-game) and a wall's correct value is a function of its
  neighbours. Real files encode that index as a literal `0..4` or as
  `k*2π/5` radians; both must be read, neither may have an angular
  zero-point offset applied, and `unit_sprites.variant_index()` is the only
  correct reader (`angle_index()` mis-maps 3 and 4). A write path passes it
  through verbatim, with the exceptions above.
- **An existing unit's `garrisoned_in_id` changes only through
  `UnitEditModel.set_garrisoned_in()`** (GH #115), which splices the garrison
  reverse-map and raises outside its structural scope: never inside a
  `fields_only` edit, and when linking, never into itself, into a reference_id
  no live unit has, into a host that is transitively inside the unit (a
  cycle), or for a unit that holds a garrison of its own (no nesting). -1
  unloads and is always allowed. Type and capacity are not model rules: every
  UI path checks them through `garrison.refusal()`, so a batch script stays
  unvalidated, as with `add()`. Clipboard paste (a region or copied units)
  and `add()` set the field at construction only.
- A unit's `standing_graphic` is not always the thing you draw. For some consts
  it is a DECORATION — an animated flag — whose real body is a delta hanging
  off it, and whose own art covers only the one shape the decoration sits on.
  Resolving "the first graphic with a modern-looking file_name" picks the
  overlay and draws a pennant floating over nothing.
  `tools/gen_unit_graphic_map.py`'s `_BODY_GRAPHIC_OVERRIDES` records the
  confirmed cases.

A new exception adds its one-line entry here and its scope to
`docs/WRITE_RULES.md`.

## User guide

The in-app guide (Help > DEscape Guide) is the Markdown in `descape/help/`,
one topic per file, registered in `help_guide.TOPICS`. A change to a
behaviour the guide describes updates its topic in the same commit, and a
new menu item, mode or tool gets a line in its topic. Guide prose names a
command and where it lives, never its rebindable key (the Keyboard and Mouse
page shows the live keys); `tests/test_help_guide.py` lints the rest.

## Contributing

If you are working on a contribution for a pull request, follow
`CONTRIBUTING.md`. Run `./run_all_tests.sh` (`run_all_tests.bat` on Windows)
and report what it printed before calling a change done. The Hard rules above
are not negotiable. A contributed PR leaves `CHANGELOG.md` alone.

## Changelog

User-visible changes here mean behavior, rendering, the write path, or a new
tool. `CHANGELOG.md` uses `MAJOR.MINOR` versioning (no patch number). A
contributed PR's entry is written by the maintainer at merge.

## Private maintainer repo

If `maintainer/` is present alongside this repo on disk, it has its own
fuller `AGENTS.md` — check it for open-work tracking and deeper context
before starting non-trivial work. If it isn't present, it simply isn't part
of what you're working with.
