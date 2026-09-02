# Changelog

Notable changes to DEscape, newest first. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), except versioning:
`MAJOR.MINOR` only, no patch number. This project has no CI and cuts
releases by hand rather than continuously, so the thing a patch number
usually buys - "bugfix only, safe to upgrade blindly" - doesn't apply here;
every release, fix or feature alike, bumps the minor. `1.0` is reserved for
whenever the on-disk write path is trusted enough to not need Save As as a
safety net.

What belongs here: changes someone using or building DEscape would notice -
new tools, rendering/write-path changes, fixed bugs, anything that changes
how the app behaves or how you run it.

What does not: routine scaffolding, doc tweaks, refactors with no observable
effect, and work in progress. Git history already covers those, and a
changelog padded with them stops being read.

Each entry is one or two lines: what changed, for a human skimming the
file. Verification detail and rationale belong in the commit itself (git
history already keeps it); if an entry would otherwise restate a doc's
content, link the doc instead of summarizing it.

## [0.3] - 2026-09-01

### Added

- **`0`-`8` select a player** wherever the active mode has its own player
  selector: Units mode's Place/Convert owner combo, the Players panel, and
  the Diplomacy panel. Each selector is independent -- switching one never
  changes the others. Rebindable under Settings > Keybinds' new "Player
  Selection" section.
- **Keybinds for Units, Map Options, Players and Diplomacy mode**
  (Ctrl+U/M/P/D), plus every other mode's default moved to a consistent
  Ctrl+<letter> scheme (View: Ctrl+I, Terrain: Ctrl+E, Triggers: Ctrl+T).
  Isometric View's own default moved to Ctrl+Shift+I to free up Ctrl+I.
  Several existing toggles (New Map's default-size action, Show Sprites,
  Perf Trace, and the Filters menu's six commands) are now rebindable too,
  shipping unbound by default. All Settings > Keybinds.
- **A "Show editor-hidden objects" checkbox** in the trigger reference-field
  browse dialog, off by default: filters the `hide_in_editor` objects (trees,
  mines, and similar) out of the picker tree until opted into.
- **Name-based pickers for trigger reference fields**: `object_list`,
  `building_list`, `technology`, `trigger_id`, and `variable`/`variable2`
  fields in the trigger property form now show a searchable name (type-ahead,
  or a "..." browse dialog with a sprite preview) instead of a raw integer.
  Also fixes `object_list`-family fields silently showing no name at all for
  most real values (they can hold any object type, not just units). Optional:
  `tools/gen_object_catalog.py` plus a configured AoE2:DE install resolve real
  in-game display names (`config.yaml`'s new `language` key); without either,
  names still show in the library's own ALL-CAPS form.
- **A Units-section write path**: placing, removing, moving, and reassigning
  a unit's player now persists through `scenario_write.write_scenario()`,
  with undo/redo. Reachable via `descape.unit_model.UnitEditModel`, the
  scriptable batch-edit API (`batch_scripts/reassign_units_by_const.py`),
  and now a full in-editor UI (see below). Rotating an existing wall or gate
  is deliberately not supported yet (its stored `rotation` is a
  shape-variant index, not an angle, and the remap table between variants
  isn't derived).
- **Units mode is now editable**: a Place Unit tool (owner-selectable,
  snapped to tile centre), drag-to-move, arrow-key nudge, and Delete (no
  confirmation dialog -- Ctrl+Z is the safety net, same as every other edit
  tool here), plus an in-place inspector (X/Y/Z/Owner) for the selected
  unit. Multi-unit editing too: rubber-band marquee and click-to-add
  selection, group move (arrow-key nudge) and group delete as a single undo
  step, and a Convert brush that reassigns every unit it's dragged across to
  a new owner in one stroke. All with undo/redo alongside every other edit
  mode. Free (non-grid-snapped) placement and rotate are not supported yet.
- **File > Save** writes in place to the file you already have open, instead
  of always prompting for a path the way Save As does.
- **A new Messages mode** edits the scenario's prose fields (Instructions,
  Hints, Victory, Loss, History, Scouts), with undo/redo alongside terrain,
  trigger, Map Options, Players and Diplomacy edits. Warns when a field's
  in-game display is actually driven by a string-table ID rather than the
  stored text. The codebase's first length-changing write path.
- **A new Diplomacy mode** owns the whole in-game Diplomacy tab: a player
  selector, that player's stance (Ally/Neutral/Enemy) toward each other
  defined player, their allied-victory flag, and the four team-setup
  scalars (Lock teams, Players choose teams, Random start points, Max
  number of teams), moved here from Map Options mode. Editable with
  undo/redo alongside terrain, trigger, Map Options and Players edits. A
  self-stance stored as something other than the default is kept exactly
  as recorded and shown as a note, since a player has no in-game stance
  toward itself to edit.
- **Players mode is now editable**: Identity, Start and AI settings for
  P1..P8 (name string id, lock civilization/personality, starting age,
  food/wood/stone/gold, colour, population limit, base priority) can be
  changed, with undo/redo alongside terrain, trigger and Map Options edits.
  A field stored twice (the four resources, colour, population limit)
  writes both copies, matching what the in-game editor itself does on
  save. Tribe name, civilization, architecture and personality stay
  read-only for now (a text field this panel can't edit yet, or
  variable-length data no byte patch could reach); Player Type stays
  read-only permanently, since its meaning is unconfirmed.
- **The trigger browser shows each trigger's id**, in a narrow column ahead
  of the name, so a trigger referenced from an Activate/Deactivate Trigger
  effect can be found by that number. Read-only and cosmetic: filtering
  still matches on name only.
- **A Perf Trace toggle, Help > Perf Trace**: traces drag-paint latency by
  phase (tile pick, highlight, dirty scan, bbox, patch, repaint) to Help >
  Debug Log, one summary line per drag stroke. Off by default; also settable
  via the `DESCAPE_PERF_TRACE=1` environment variable for headless runs.
- **The trigger browser groups triggers into the author's own sections**,
  when a file uses the `--- Section ---`-style divider convention: sections
  are collapsible, and triggers before the first divider sit under a
  "(before the first section)" header. A **tag filter** (beside the
  existing text filter) narrows the list to triggers sharing a leading
  `[tag]`/`(tag)`/`<tag>` name prefix. Read-only: no new file bytes, no
  sidecar, nothing written differently. A file with no dividers or tags
  browses exactly as before.
- **Sloped view draws real unit sprites**, with Show sprites on: units now
  composite as their actual game art on ramps, the same as Stepped, anchored
  to the ground surface the compositor paints rather than a flat pad.
  Farms still keep their plain coloured mark in Sloped (their real crop
  terrain is Stepped-only for now).
- **`.bak` and `.orig` backup files on save**: saving now leaves a `.bak`
  (the state immediately before that save, refreshed every time) and a
  `.orig` (a one-time pristine snapshot, captured the first time a save
  would overwrite an existing file) beside the target. Unconditional, no
  Settings toggle yet. A save that would write byte-identical content is a
  no-op and touches neither backup nor the target file's own mtime.
- **Units mode works in Sloped view**: a unit is now placed on the ground
  the compositor actually paints, at its own sub-tile position rather than
  on a flat pad averaged from its tile's corners, and can be hovered,
  clicked and highlighted there. Sloped no longer forces Units mode back to
  View. Unit markers are still plain diamonds and do not follow the warped
  tile footprint on a ramp.
- **Farms render as their real crop terrain**, with Show sprites on: a
  placed Farm (and its RFarm/Pasture family) now paints the actual farmland
  texture across its footprint instead of a coloured mark, with a thin
  player-colour outline around the footprint so a placed unit stays
  distinguishable from hand-painted farm terrain. Stepped only, gated on
  the same Show sprites toggle as real unit sprites.
- **Trigger reordering**, in Triggers mode: a sort selector (Display order /
  File order) so the panel shows triggers in the order the game shows them
  by default, a read-only line reporting whether the file executes in
  display or legacy trigger-ID order, and Move Up/Down buttons to change
  display order. Available while sorted by Display order with no filter
  active; the file's execution-order mode itself is set from Map Options,
  not here.
- **Show All / Hide All** in the Filters menu: two shortcuts at the bottom
  that check or uncheck every entry at once - GAIA, Trees, and all 8
  players - instead of clearing each in turn.
- A **Ruler tool**, `R`: drag across the map, or click twice, to measure
  the straight-line distance between two tiles. Reads out in tiles plus
  signed X and Y components, both on the map and in the log. Available in
  every mode and every Elevation View, unlike the terrain tools, and it
  changes nothing. Escape, right-click, or switching tool or mode clears
  the measurement.
- **Distance ticks along the map edge**, View > Distance Ticks: a ruler strip
  of tick marks just outside the map border, on all four edges, with a tile
  number on every fourth one. Minor ticks land every 4 or 5 tiles, your
  choice, giving a numbered tile every 16 or 20, and the marks keep the same
  on-screen size at any zoom. Off by default; both the toggle and the
  interval are remembered.
- A **Map Options** mode, alongside View, Terrain, Units and Triggers: a
  scrolling form of the scenario's own settings - diplomacy, map flags, full
  tech tree, AI map type, victory condition, score to win, time limit, and the
  trigger execution-order mode. Editable, with undo/redo alongside terrain and
  trigger edits, and each setting written back as a minimal diff of its own
  bytes. The four custom-victory conditions (conquest / relics / explored % /
  require-all) are not in yet. A setting the file stores as a value no editor
  could show truthfully (a byte that was never a flag on an older scenario
  version) is listed as stored rather than shown as something else, and
  cannot be edited - so a save can never overwrite it. Score to win shows
  14000, the real editor's own default, on a file whose stored value is its
  "no score victory" sentinel - editable like any other value, and a save
  stays byte-identical until it is actually changed.
- **Trigger editing**, in Triggers mode: every field of a trigger and of its
  conditions and effects can now be changed, with full undo/redo alongside
  terrain edits. Triggers you do not touch keep their exact original bytes, so
  a save stays a minimal diff.
- **New, Copy and Delete** for triggers, and for the conditions and effects
  inside one. New opens a filtered list of every condition and effect the
  file's scenario version supports, and the entry it adds starts on that
  version's own defaults.
- **Trigger variables**, from a Variables button under the trigger panel: the
  scenario's variables with their ids, and buttons to add and remove one, with
  undo/redo like every other trigger edit. Names are shown read-only and there
  is no rename, deliberately - a rename would not be undoable. Removing a
  variable frees its id for the next one added, which then inherits any
  reference the scenario still holds to it.
- The trigger panel is now **two panes**: trigger names on top, and below them
  the selected trigger's conditions and effects with a property form for
  whichever is selected. Both panes scroll rather than truncating long names.
- A **Units** mode, alongside View, Terrain and Triggers: hover outlines the
  unit under the cursor and clicking selects it, both sized to the unit's
  real footprint rather than a single tile, so a building highlights as the
  whole slab it occupies. Works in Flat and Stepped; unavailable in Sloped,
  which has no unit hit-testing yet. Read-only - nothing here edits a unit.
- A **unit inspector** in the left panel, shown in Units mode: name, unit ID,
  owner, position, rotation, reference ID and garrison target for the
  selected unit. Rotation is shown raw and labelled, because for most GAIA
  objects it is a doodad graphic-variant index rather than an angle.
- A **Show sprites** toggle in the View menu: draws units as their real
  game sprites instead of coloured marks. Stepped Elevation View only for
  now, and it needs a configured AoE2DE install path. Off by default and
  resets on each launch, because switching it on decodes every sprite once
  and can take a few seconds.
- A **Filters** toolbar menu for hiding units by owner or kind: GAIA, trees,
  and a checkbox per player. A typical scenario is ~92% GAIA clutter, so
  hiding trees alone usually makes a map readable. Applies in every mode and
  every Elevation View, and resets to "show everything" on each launch.
  Hidden units can't be selected either.
- A **Triggers** mode, alongside View and Terrain (`Ctrl+T`): a read-only
  browser for a scenario's triggers, with each trigger's conditions and
  effects named and a filter box over the list. Read-only for now - editing
  triggers is not implemented yet, and a file opened in Triggers mode saves
  exactly as one that wasn't.
- `Sloped` mode now responds to the cursor: hovering a tile outlines it and
  reports its coordinates, where before the mode ignored the mouse entirely.
  The outline follows the tile's real warped silhouette, so it fits a ramp
  rather than sitting on it as a flat diamond.
- **Terrain editing works in `Sloped`.** Draw, Paint Can, Elevate, Set
  Elevation and Copy/Paste are no longer greyed out there, and each edit
  repaints just the tiles it touched rather than the whole map. `Units` mode
  is still unavailable in `Sloped` - it needs unit hit-testing, which is a
  separate piece of work.

### Changed

- **Tool params (Terrain type / Level / Object / Owner / Brush) now sit on
  their own toolbar row** below the tool buttons, instead of competing with
  them for width. The row holds its height even when empty, so switching to
  Pan or Ruler doesn't shift the layout.
- **Draw, Paint Can, Elevate and Set Elevation now hide from the toolbar
  outside Terrain mode**, instead of just greying out. A file that blocks
  terrain writes, or a non-square map, still greys them out visibly rather
  than hiding them.
- **Save As now retargets the open document to the file it just wrote**,
  for a normally-opened file as well as a new/untitled one - the usual
  editor convention. Previously this only happened for an untitled
  document; saving an already-open file to a new path left the window
  still "on" the original file.
- The **Elevate tool's default shortcut moved from `R` to `E`**, freeing
  `R` for the new Ruler. If your keybinds are already saved, an Elevate still
  set to `R` is moved to `E` for you the first time you run this version,
  including one you chose deliberately - it cannot be told apart from the old
  default. An Elevate you had bound to anything else is left alone.
- The **left panel now runs the full window height**, with the system log
  beside it under the map view instead of as a full-width strip under both.
  The log's height is draggable and remembered between sessions.
- View > **Show sprites** now defaults to on instead of off, so Stepped opens
  showing real unit sprites where an AoE2DE install is configured.
- `Sloped` now shapes its ramps the way the game itself does. A raised area's
  outline sits at full height and the ramp is the ring of tiles just outside
  it, instead of the transition straddling the outline two tiles wide. Read
  off in-game captures rather than guessed, which is what the previous
  behaviour was.
- The **Edit** mode is renamed **Terrain** (keybind moved `E` to `T`), and its
  **Terrain** tool is renamed **Draw** (keybind moved `T` to `D`) to remove
  the resulting name collision between the mode and the tool.
- The left panel and the map view are now separated by a draggable split
  whose position is remembered between sessions, instead of the panel being
  capped at a fixed width.
- `tools/gen_seam_eyeball.py`, for eyeballing the terrace-edge seam line at
  every "Stepped elevation height" setting and, for the first time, at the
  two coarsest mip levels (`tile_px` 8/16) - the zoom where terrain shape
  actually gets read.
- `config.yaml` now lives at the OS-standard per-user config location
  (`~/.config/DEscape/` on Linux, `~/Library/Application Support/DEscape/`
  on macOS, `%APPDATA%\DEscape\` on Windows) instead of inside the repo
  checkout. An existing repo-root `config.yaml` is copied there
  automatically the first time you launch this version.
- Terrain/sprite mip selection now picks the coarsest level that never
  exceeds native texture density, instead of the finest level that never
  falls short of it. Tiles render at or below native resolution everywhere
  in the zoom range rather than up to 2x magnified right before each level
  switch - visibly sharper through every zoom band, at the cost of always
  compositing one level finer than strictly necessary.
- **A live zoom readout in the bottom status bar**, shown as percent of
  fit-to-view (`Zoom: 100%`). Updates on every wheel zoom, on a viewport
  resize, and on View > Isometric, and reads `Zoom: --` with no map loaded.

### Fixed

- **Assigning a keybind that's already in use no longer silently kills both
  actions.** The Settings dialog's Keybinds tab now auto-clears the other
  action's binding and shows an inline warning explaining what happened,
  instead of leaving both shortcuts dead with the only diagnostic on a
  stderr stream a GUI user never sees. The same reconciliation now also
  runs once at startup against your saved keybinds, so an old custom
  binding that happens to collide with one of this release's new defaults
  self-heals instead of silently going dead.
- Selecting a long trigger condition/effect row (or a long trigger or picker
  entry) no longer yanks the horizontal scroll all the way to the right.
  Qt's default scroll-to-selection behavior was chasing a `ResizeToContents`
  column's full width; the trigger browser's trees now keep whatever
  horizontal position you had scrolled to.
- **Units now render with the color their owning player actually picked in
  the in-game editor**, instead of assuming player N always gets color N.
  17 of this project's 20 example files store a different assignment than
  that identity guess, from a simple two-player swap up to a near-full
  permutation - every one of them was rendering at least one player in the
  wrong color.
- **Player colors 6 and 7 are legible again**: player 6 rendered as a
  washed-out pink instead of magenta, and player 7 as brown instead of an
  orange separable from red and yellow.
- Terrain and elevation drawing stutter. Every stroke edit was rebuilding
  unit-derived render state (positions, building footprints, sprites) from
  scratch regardless of what actually changed, and with Show sprites on,
  every dirty tile was padded for the worst-case sprite reach whether or
  not a unit was anywhere near it. Both are gone: a plain terrain edit now
  skips that rebuild entirely, and the sprite padding only applies near an
  actual unit.
- Palisade Wall, Fortified Palisade Wall and Sea Wall now draw at all. Their
  entry in the graphic table pointed at the animated flag that sits on top of
  a palisade rather than at the wall itself, and that flag has art for only
  one of the five wall shapes - so a palisade drew a floating pennant at one
  rotation and nothing at the other four. The flag itself is not drawn for
  now; it needs the multi-piece compositing that gates and town centres also
  wait on.
- Wall pieces now draw the right shape. A wall graphic's five stored frames
  are shapes - two diagonal runs, a tower, a flatter run, a narrow column -
  not five facings, so the `rotation` field selects a variant rather than an
  angle. Reading it as an angle made lone posts draw the tall tower and drew
  a run where the narrow column belongs. Covers Stone, Palisade and Fortified
  Wall plus the scenario City Wall, Fence and Fortified Wall.
- Stables now draw their real sprite instead of a coloured mark, along with
  35 other placeable objects: the Elite Huskarl, the Elite Genoese Crossbowman,
  21 scenario flags, and six animals and props. Their `.sld` files use a second
  on-disk layout the decoder rejected outright; it differs from the usual one
  only by a two-byte header size, which also shifts every layer boundary in the
  file. 226 of the install's 7,372 sprite files use that layout, but most are
  per-civ architecture art the editor never picks, so stables are the only
  building affected.
- Zoom limits no longer go stale when the window or the map view is resized.
  Zooming out could stop short of the whole map (and zoom-in short of its real
  ceiling) until you toggled Isometric View to force a re-fit.
- `Sloped`'s slope shading was lit from the wrong side. Its light direction
  was inverted horizontally, so the flank the game shades was left almost
  unshaded and the flank the game lights was darkened - measurably worse than
  applying no shading at all. The light direction and strength are now
  measured against in-game reference captures rather than guessed, which also
  makes the lit side visible: it could previously only brighten a slope by
  4.6%.

- Multi-tile buildings now fill their whole footprint in `Stepped` and
  `Sloped`. They previously painted only the up-screen part of it - roughly
  half a Colosseum, three quarters of a House - because a building drew its
  entire slab at its own tile's turn, and every footprint tile drawn later
  then painted terrain back over it. `Flat` was never affected.

- Undoing an edit while `Sloped` mode is showing no longer repaints garbage.
  Undo/Redo were never gated the way `Sloped`'s Terrain tools then were, so
  undoing a `Stepped` edit after switching to `Sloped` repainted flat,
  grid-aligned squares onto the isometric canvas. `Sloped` now has a real
  incremental repaint path of its own.

- Buildings are no longer rendered, hit-tested and highlighted one tile too
  large in each axis. Houses and Mills are 2x2 rather than 3x3, Farms and
  Docks 3x3 rather than 5x5, Town Centres and Castles 4x4 rather than 5x5,
  gate segments 4x1 rather than 5x1, and the largest wonders 8x8 rather than
  9x9. Walls, towers and small decoratives are unchanged.

- `Sloped` mode no longer draws a fine lattice of black hairline seams across
  sloped ground. Neighbouring tiles now meet exactly at every shared screen
  column, and a slope's columns foreshorten with the terrain instead of
  keeping their flat length, so a ramp reads as a smooth surface rather than
  a mesh. Flat ground is unchanged, at any height.

- Contact shadows in `Stepped` mode now run the full length of a terrace
  edge instead of breaking into one dash per tile. The band tapers to
  nothing approaching each tile's apex, where the tile diagonally behind is
  what shows through, and no pass had ever drawn there; a terrace now reads
  as one continuous shadow. Also gives the tallest "Stepped elevation
  height" settings a faint up-screen cue where previously they had none.

- Tall hills in `Stepped` mode no longer read as a lattice of dark
  triangles on their up-screen half. The contact shadow still covers the
  whole strip of the neighbour tile a raised tile leaves visible, but its
  darkening now fades out within a quarter of a tile of the edge casting
  it, instead of being stretched across that whole strip - which meant a
  single-level step tinted up to 82% of the tile behind it, and tinted it
  more the shallower the "Stepped elevation height" setting was.
- Terrace edges in `Stepped` mode now read as continuous lines instead of
  dashed ones, because each height step draws a thin contour along the
  up-screen side of the higher tile. At the tallest "Stepped elevation
  height" setting this is the only up-screen cue there is - a hill's back
  half used to fade into flat ground with no boundary at all.

- An edit that fell back to a full map redraw (a Paint Can fill spanning a
  large fraction of the map, or an elevation edit pushed outside the cached
  range) no longer resets your zoom and pan back to fit-to-view. A plain
  settings-change re-render (e.g. newly configuring an install path) keeps
  your view too. Opening a file or switching Terrain Style still fits the
  whole map, as before.

## [0.2] - 2026-08-15

### Added

- A third Elevation View option, `Sloped`, showing real ramped transitions
  between differing elevations instead of Stepped's columnar terraces.
  View-only for now: clicking on the map, painting, and elevation editing
  are not yet wired up in this mode. Early preview - sloped ground still
  renders a fine lattice of unpainted seams along tile boundaries, so it
  does not yet read as smooth ramps; `Stepped` remains the mode to use.

- Brush size (1-9) and shape (square / circle) for the Terrain, Elevate and
  Set Elevation tools, with a hover preview of exactly the tiles a stroke
  will affect. Not persisted - resets to size 1, square, on launch. `]`/`[`
  now adjust brush size for any tool that has one, falling back to Set
  Elevation's target level otherwise.

### Fixed

- Terrain above elevation 7 no longer renders as corrupted pixels wrapped
  to the wrong part of the canvas. DE authors elevations up to 15, but the
  viewer sized its canvas for a ceiling of 7, so anything higher - loaded
  from a scenario, or raised with the Elevate tool, which has no upper
  clamp - was drawn outside that canvas and silently wrapped around.

- A first run interrupted partway through setup no longer leaves the
  launcher stuck on a half-built `.venv`. `python -m venv` creates the
  interpreter well before the environment is usable, so the old
  "interpreter exists, we're done" check trusted it and produced confusing
  pip/import errors; setup now marks completion explicitly and rebuilds the
  environment from scratch if that marker is missing.

- Contact shadows in `Stepped` mode no longer spill onto tiles at the same
  elevation as the tile casting them. A hill built with the Elevate tool
  rendered as a lattice of dark concentric lines; the shadow band is now a
  per-column wedge confined to the back neighbour whose height difference
  produced it, so each terrace reads as shaded relief instead. The band
  also covers the whole exposed sliver of that neighbour rather than a
  fixed-height strip, which is what makes low `elev_step` settings read as
  relief at all.

### Changed

- Settings > Appearance's "Stepped elevation height" slider now moves in
  eight fixed 25% stops (25%-200%) instead of any value in between, and its
  floor rises from 10% to 25%. Off-stop heights quietly reduced how much
  real detail the viewer could show when zoomed in; an existing setting
  below the new floor reads back as 25%.

- Fit-to-view on a large map now renders from a coarser mip level instead of
  the full-resolution canvas: on a 480x480 map it paints in about half the
  time and adds ~39 MB rather than ~241 MB over the loaded scenario. Zooming
  out is allowed correspondingly further than "half of fit-to-view" too,
  since the coarser levels take the aliasing that limit existed to avoid.
- Zooming in past roughly 2x now shows real added terrain detail instead of
  magnifying the same texels, where the map's size allows it. Big maps are
  rendered at a reduced tile resolution to bound memory, and the viewer can
  now composite from a finer level on demand once you zoom in far enough to
  see the difference. Every export path is unchanged.

## [0.1] - 2026-08-12

### Added

- A viewer with real per-tile terrain textures and two rendering modes:
  `Flat` and `Stepped` (per-tile elevation displacement).
- An Edit mode with Terrain/Elevate/Set Elevation tools, undo/redo, bounded
  incremental redraw, and single-slot copy/paste.
- A terrain+elevation write path (Save As only) and a scriptable batch-edit
  API (`descape/batch_api.py`) built on top of it, with runnable examples
  under `batch_scripts/`.
- File > New Map at all 7 real AoE2:DE sizes plus arbitrary custom square
  sizes 61x61 to 480x480.
- A Settings dialog (General/Appearance/Game Resources/Keybinds tabs,
  including a live-togglable dark theme).
- One-click setup/launch scripts (`LAUNCH_DESCAPE_*`) that bootstrap a
  `.venv` and install dependencies automatically.
- Licensed GPL-3.0 (`LICENSE`).
