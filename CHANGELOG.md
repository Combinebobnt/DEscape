# Changelog

Notable changes to DEscape, newest first. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), except versioning:
`MAJOR.MINOR` only, no patch number. Pushing a `v*` tag builds every
platform and drafts a GitHub Release, but the thing a patch number usually
buys - "bugfix only, safe to upgrade blindly" - still doesn't apply here;
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

## [Unreleased]

### Added

- **`tools/release_render_diff.py`: pixel-diff the map render against a
  release** (GH #87). Renders each scenario at `v0.6` and at this checkout in
  every view, sprites off and on, and fails on any pixel that differs outside
  the changes made on purpose since. `--frames-dir` writes side-by-side crops
  of those; `--inject` runs a control that must fail.

### Changed

- **Faster Stepped rendering: part of the renderer is now compiled.** The
  downloads include it. From source, the launcher builds it on first launch
  when a C compiler is present, and otherwise runs the slower built-in
  renderer; see README "Faster renderer (optional C compiler)".
- **Elevate and Set Elevation no longer lag with sprites on.** An elevation
  edit now updates only the units on the changed tiles instead of rebuilding
  every unit's sprite and footprint.

### Fixed

- **Help > Perf Trace now reports panning and zooming** on their own
  `perf view` lines. Before, those repaints and any hover time were folded
  into the next drag's line, overstating it, and Convert and Cliff strokes
  were never reported.
- **`tools/gen_iso_reference_pngs.py` no longer crashes when `--out-dir` is
  outside the repo.** It wrote the PNGs, then failed printing their paths.

## [0.8] - 2026-09-23

### Added

- **File > Recover from Autosave can be given a keyboard shortcut** in
  Settings > Keybinds > File. Ships unbound.
- **Copy and paste multiple triggers** (GH #27, #28). The trigger list is
  multi-select (Ctrl/Shift-click); Copy, Delete and Move Up/Down act on the
  whole selection as one undo step. Edit > Copy Triggers / Paste Triggers
  (Ctrl+C / Ctrl+V in Triggers mode) and a Paste button paste the block below
  the current trigger, with links inside the block pointing at the pasted
  copies. A right-click menu offers Select Section and Select Tag.
- **Wall Rectangle tool** in Units mode: drag to place an outline ring of the
  wall picked in the Units catalog, one undo step per drag. Shift makes a
  square.
- **Tooltips on every Map Options and Diplomacy row** and on the Elevation
  View combo; Block humanity team change's tooltip says it locks co-op
  alliances (GH #43, #55).
- **`tools/ci_parity.sh [tag]`**: a CI-equivalent local run (release-tag
  check, then the default test tier with the Workshop corpus, DE install and
  `examples/` hidden) to run before every tag.
- **Trigger tags and sections are editable.** Rename tag… renames a leading
  `[tag]` on every trigger carrying it (merging into an existing tag after a
  confirm), and Remove tag… strips it. New Section…, Rename Section… and Move
  to Section work on `--- Section ---` dividers, and a new header copies the
  divider style the scenario already uses. Each action is one undo step.
- **A Sections menu in Triggers mode** offers Collapse All, Expand All and a
  jump to any section. Collapsed sections stay collapsed across edits, and a
  filter opens sections that hold a match.
- **The unit inspector edits a multi-unit selection** (GH #71): Owner, X, Y, Z
  and Rotation show the shared value or "(mixed)", and a typed value applies
  to every selected unit in one undo step. Rotation skips trees, walls, gates
  and scenery.
- **Drag a multi-unit selection to move the whole group** (GH #75), with a
  preview of every unit, stopping at the map edge with its shape intact. New
  Edit > Select Whole Stack (Ctrl+K) selects everything stacked with a
  selected unit.
- **Filters > Show Invisible Objects** hides Invisible Objects, Map Revealers
  and Blockers (GH #53). With sprites on, they now draw as owner-coloured
  editor-only markers instead of a plain box; revealers use the game's eye
  icon.
- **Hero units draw with a thin gold outline**, like the game's hero glow, in
  Stepped and Sloped (GH #39). Toggle it with View > Layers > Hero Glow.
- **Mirror Map gains 3-way rotational, 6-way rotational and 6-way reflective
  modes** for 3- and 6-player layouts. Corner tiles with no in-map source are
  left untouched and counted, and Elevation starts unticked for these modes.
- **Create Objects tool** in Triggers mode (GH #59): pick a Create Object
  effect, then click the map to add one copy of it per tile under the brush,
  one undo step per click. Tiles already holding that object are skipped.
- **DE 1.21 scenarios open natively** (older Steam Workshop coop mods):
  terrain, elevation and units load, render and save. Header editing and
  Players mode stay read-only, and their triggers can't be read yet.
- **Garrison editing in the unit inspector** (GH #42): a Garrison block lists
  a host's occupants, with Add restricted to what the game admits, and Delete.
  Garrisoned units are hidden by default behind a Filters toggle, and a moved
  or deleted host carries them. Map Analysis flags units garrisoned in a host
  that isn't on the map.
- **Range ring around a selected building** (GH #49): View > Show Range Rings
  draws its attack range, measured from the footprint edge. Off by default;
  the colour lives in Settings > Appearance.
- **Edit > Scatter Units in Region…**: fill a Select-tool region with
  randomized copies of the Units panel's object, restricted by water or
  terrain and sized by count or density, with seed, jitter and spacing, as one
  undo step.
- **Edit > History…** (GH #30): a window listing every recorded edit, current
  in bold and undone entries greyed. Double-click an entry to undo or redo the
  whole span in one step.
- **Multi-select conditions and effects** in Triggers mode (GH #60):
  Ctrl/Shift-click several entries of one trigger and edit them as a group.
  Shared fields show their value, differing ones read "(differs)" and are only
  written once changed. Copy and Delete act on the whole selection as one undo
  step, and right-click > Select Same Type selects every entry of that type.
- **Point of View buttons and player camera markers** (GH #22): Set View, Go to
  View and Reset View in Players mode, and View > Show Player Cameras marks each
  player's starting camera in their colour. Hidden on DE 1.41 files.
- **Paste triggers into another scenario** of the same version (GH #3). Links
  to uncopied triggers and placed-object references are cleared, as the
  in-game copy does, and named variables carry their names. The status log
  says what was cleared.
- **Select All / Deselect in Triggers mode**: Ctrl+A selects every visible
  trigger, collapsed sections included, and Ctrl+Shift+A clears the selection.
- **Map Analysis marks each located finding on the map** while its results
  dialog is open, with a severity glyph (`!!`, `!`, `i`, plus `xN` for shared
  tiles). The selected row's marker gets a ring; the colour is in Settings >
  Appearance.
- **The selected trigger is drawn on the map** (GH #41): its areas are outlined
  and tinted, following the ground in Stepped and Sloped, its locations are
  marked, and an entry with both (Patrol, Attack Move and the like) gets a line
  labelled with the Ruler's distance. The selected condition or effect is drawn
  strongest. View > Show Trigger Overlay (on by default); colours in Settings >
  Appearance.
- **Trigger fields that name a placed unit show which unit it is**, with its
  name, player and tile (or that the id is not on the map), and a **Pick**
  button sets the field by clicking a unit on the map; for a unit list, click
  to add or remove, then Esc. With the trigger overlay on, referenced units are
  outlined, and a run line goes from an entry's units to its location object.

### Changed

- **Linux builds run on older distros**: they are now built on AlmaLinux 8 and
  target glibc 2.28 (Ubuntu 20.04, Debian 10, RHEL 8 or later), down from
  v0.7's 2.35.
- **Wall Run is folded into Place Unit** (GH #98): pick a wall in the Units
  catalog and drag to place a run. A single wall click gets its real shape
  and reshapes its neighbours, and GAIA walls are allowed. The saved Wall Run
  keybind is dropped.
- **Placed Pastures draw like the game's** (GH #66): the tent in the middle, a
  post on each corner and broken fences along every side. Each pasture keeps
  its own mix of post and fence shapes across save and reload.
- **The selection highlight takes each owner's player colour** over a dark
  under-stroke, with View > Colour Selection by Owner (on by default). GAIA
  and the marquee keep the configured Selection colour.
- **Show Eye Candy no longer hides Blockers**; they belong to Show Invisible
  Objects now (GH #53).
- **Messages boxes behave like the trigger text fields**: Tab now moves to the
  next box and saves the edit, and opening a right-click menu or switching
  windows no longer saves it early.
- **Deleting a multi-unit selection no longer stalls on large maps**: a
  1000-unit delete on an 11k-unit map drops from ~640ms to ~7ms of model work.
- **The unit inspector's Rotation is a whole facing number** (GH #61), 0 to one
  less than the unit's direction count, wrapping on the scroll wheel with the
  stored radians in its tooltip. Mixed-count groups edit on the finest grid.
- **View mode shows a per-player breakdown** behind a player drop-down (GH #5):
  placements, units, buildings, trees, eye candy and triggers naming that
  player, replacing the old per-player unit list. Tools > Map Analysis now
  refreshes it instead of leaving it stale.
- **A scenario version DEscape cannot read now names the version and the
  cause** in place of the parsing library's raw error.
- **The terrain picker lives in Terrain mode's sidebar** (GH #56), always
  visible, with grouped swatches, a filter, a preview and a one-line hover
  readout. The toolbar's Terrain type drop-down and "…" browse window are gone,
  and map statistics show in View mode only.
- **Elevation edits repaint a smaller region**, sized to how far the terrain
  shading actually reaches: about 45% less height per single-tile edit at
  default settings.
- **The Convert brush, and Move/Place/Delete of units in Flat, update the map
  faster**: only the edited units are redrawn instead of every unit on the
  map. A 20-unit Convert on a large map drops from about 220-990ms to 10-100ms.

### Fixed

- **Fast brush drags no longer leave gaps.** Draw, Elevate and Set Elevation
  fill in every tile the cursor crossed between mouse moves, so a quick drag
  paints a continuous stroke instead of a dotted one, most visibly at small
  brush sizes.
- **The unit list of nine trigger effects is a list editor again** (Task
  Object, Build Object, Change Object Caption and six more). It showed as a
  number box reading "(unset)" even when units were stored, and an edit there
  would have replaced the list with one number.
- **View > Grid no longer draws over buildings and units.** With Follow
  Terrain Elevation on (the default), and always in Flat, the grid is now part
  of the map image, drawn under unit sprites, and nearer raised ground hides
  the lines behind it. Grid thickness now scales with zoom, like the terrain.
  Follow Terrain Elevation off still draws the flat ground-level grid on top.
- **Palisade, Fortified Palisade and Sea Wall banners stand on top of their
  tower** instead of hanging across the middle of the wall (GH #51).
- **The Map Options and Triggers panels widen with the app font**, so at 150%
  display scaling the status line no longer crowds out the form and trigger
  fields no longer overflow into a horizontal scrollbar.
- **Switching a Modify Attribute effect to or from Armor/Attack no longer
  crashes the next save**, and neither does retyping onto one of those
  effects. On a float attribute (e.g. Work Rate) Quantity Float is editable.
- **Moving or renaming triggers in a sectioned list, or switching Display/File
  order, no longer drops an unsaved Execution order change** from the position
  header and status line.
- **Footprint Outlines update right away when Show sprites is toggled in
  Sloped**, so draped farm outlines follow the switch.
- **Mirror Map turns units with a real facing** (soldiers, villagers, siege,
  ships) to the mirrored direction instead of keeping the original's. Trees,
  rocks and other variant-indexed objects are still copied as they are.
- **A terrain edit that fails partway no longer makes every later edit fail**
  (for example Draw on a non-square map). What it had already painted is kept
  as one undo step.

## [0.7] - 2026-09-20

### Added

- **View > Layers > Small Trees.** Draws tree sprites at 60% size so the
  tiles behind and beside a tree stay readable, without hiding the forest the
  way Filters > Show Trees does. Session-only, default off, Stepped and Sloped
  only; the tree still stands on the same tile and is still picked by its whole
  footprint. Rebindable in Settings > Keybinds; ships unbound.
- **Disable buildings, units and technologies per player.** A Disabled
  Objects dialog off Players mode, matching the in-game editor's own
  control: a player selector, Buildings / Units / Techs tabs, and a
  two-pane mover each. One OK is one undo step. Rebindable in Settings >
  Keybinds; ships unbound. A scenario that already carries disable lists
  now shows them instead of passing them through unseen, and a file that
  is only browsed still saves byte-identically.
- **Change an existing condition or effect's type.** A Type… button in the
  trigger panel's condition/effect row retypes the selected entry in place,
  keeping its position and every field the new type shares; anything that did
  not carry over is named in the status log, and the whole change is one undo
  step.
- **A Wall Run tool, Units mode.** Drag to place a whole run of walls in one
  undo record, for any of the eight wall families. The run is 45-degree
  aligned like the game's own: an off-angle drag bends once, into one
  diagonal segment and one axis-aligned one, and Shift constrains it to a
  single straight segment. Each piece's shape is derived from its
  neighbours, and walls the run meets are reshaped to match in the same undo
  record. Rebindable in Settings > Keybinds; ships unbound.
- **Filters > Show Walls and Show Eye Candy.** Two per-session toggles that
  hide walls and gates, or decorative objects (grass, rocks, flowers, stumps,
  barrels), from the map render and from picking alike. Resources, cliffs and
  trees are unaffected - gold, stone and berry bushes stay visible. Both are
  rebindable in Settings > Keybinds.
- **A UI font family and size, Settings > Appearance.** App chrome only -
  menus, dialogs, panels and the status log - applied live and remembered
  across restarts, with a Reset button for the platform default. The map
  view's own overlay text (the Ruler readout, distance-tick numbers,
  stacked-unit badges) keeps its own fixed sizes and no longer picks up the
  chrome font's family.
- **View > Layers.** Two per-session toggles for what the map draws at all:
  Terrain Textures (flat per-terrain colour instead of the real terrain
  textures) and Farm Terrain Overlay (Stepped and Sloped, with sprites on -
  turning it off draws farms as ordinary coloured marks rather than their own
  crop terrain). Both are rebindable in Settings > Keybinds.
- **Smooth, rebindable arrow-key panning.** Holding Pan Up/Down/Left/Right
  (Settings > Keybinds, defaulting to the arrow keys) now scrolls the map
  continuously instead of one step per keypress, at a speed set by the new Pan
  speed slider on Settings > Appearance. In Units mode an arrow still nudges the
  selection; with nothing selected it pans.
- **`tools/gen_review_pack.py`: review packs for an agent to judge a render.**
  Writes framed captures plus a checklist fixed before the images exist, for a
  blind agent to review, with injected-defect runs that show whether the
  reviewer can actually see a defect. See `tools/REVIEW_PACK.md`.

- **A move preview while you drag a unit.** Dragging a unit now shows a
  translucent copy of it following the cursor - its real sprite where sprites
  are on, its coloured mark otherwise - snapped to the destination tile and
  sitting at that tile's elevation. Escape cancels a drag. The original stays
  drawn at its source tile until you release.

- **Clipboard history.** Copy Region now keeps the last 10 copies instead of
  one. Edit > Clipboard History opens a list with a terrain thumbnail for each,
  where you can pick which one Paste Region uses, rename entries, or clear
  them. Session-only, so nothing is written to disk.

- **Move a pasted region.** After pasting, drag from inside the selection to
  re-place it somewhere else. The whole paste-plus-moves gesture collapses to a
  single undo step.

- **Terrain Mode: Draw Line and Draw Rectangle.** Drag out a straight line or
  a rectangle of terrain instead of freehanding it; nothing is painted until
  you release, so dragging back over your own path leaves no residue. Hold
  Shift to snap a line to one of 16 directions, or to square a rectangle.
  Rectangles paint filled or outlined, and an outline thickens with the brush.

- **A visual terrain browser**, on the "…" button beside the Terrain type
  combo: pick terrain by sight from category-grouped swatches of the real
  texture, with a filter and a box for the unused/moddable terrains. Falls
  back to flat colour swatches with no AoE2DE install configured.

- **Draw's new "Auto beach" option** paints a beach shoreline around water as
  you lay it down, one to three tiles wide. Which beach comes from the game's
  own terrain table rather than from terrain names, so ice gets an ice beach,
  and existing water, shallows and the non-navigable beaches are left alone.

- **View > Grid** draws a line on every tile boundary, every fourth one
  stronger, with blend and thickness sliders in Settings > Appearance. Blend
  is a smooth slider that leaves the lines invisible in the middle, darkens
  the terrain under them towards black to the left and whitens it to the
  right, like the in-game grid. Its
  **Follow Terrain Elevation** half (on by default) drapes the grid onto raised
  ground in Stepped and Sloped instead of leaving it on the flat ground plane.

- **View > Footprint Outlines** outlines the tiles each unit occupies, so
  buildings sharing an edge read apart and a footprint stays visible under
  sprite art. Scoped to multi-tile buildings, all buildings or all units, in a
  colour you can change in Settings > Appearance.

- **The distance-tick ruler labels each edge with the axis it measures**, X or
  Y, drawn outboard of the tile numbers.

- **The Ruler's two measuring-point tiles now glow**, pulsing in the ruler's
  own colour, instead of sitting as static outlines.

- **Edit > Cycle Variant** changes the graphic variant of selected trees,
  plants, rocks and scenery, which is what their `rotation` field really
  stores: Previous (`;`), Next (`'`) and Random (`-`), the whole selection in
  one undo step. Walls, cliffs and gates are skipped, since the game derives
  their shape from their neighbours.
- **Mirror Map can mirror units**, not just terrain and elevation: tick Units
  and every other slice's units are rewritten from the source slice in the
  same undo step. Optionally the copies change owner, one player per slice
  along the scenario's own defined-player list. A wall's run direction swaps
  with the axis, and a gate becomes the orientation it is mirrored into.
  Buildings whose footprint straddles the symmetry axis are reported and the
  mirror is refused rather than half-applied.
- **Map Options: Custom victory settings.** Conquest, Exploration, Relics and
  the Any one / All switch are now editable in the Global Victory group,
  greyed (still showing the stored values) unless Victory condition is Custom.
- **The info panel shows a scenario's XS attachment**: the attached script
  file name and the size of any embedded script, both read-only. See
  `docs/XS_SCRIPTING.md`.
- **Script Call XS opens in a multi-line editor.** A Script Call effect's
  `message` or condition's `xs_function` now shows its script as lines in a
  monospace box, where the old one-line field displayed every line break as a
  space.
- **`tools/census_xs_usage.py`**: counts Script Call XS and file-level XS
  attachments across a folder of scenarios, read-only.
- **`tools/census_player_ai.py`**: censuses each player's AI mode byte against
  its AI script name and embedded script length across a folder of scenarios,
  read-only.
- **Stacked units are marked and reachable.** In Units mode a count badge
  sits over every spot where a unit is hidden under another, and clicking
  that spot again selects the next unit down. View > Show Stacked-Unit
  Badges turns the badges off.
- **Triggers: a position column** shows each trigger's place in display
  order, headed "Exec #" when that is the execution order and "Display #"
  when it isn't (legacy ID order, or not stored in the file).
- **Tools > Map Analysis** runs a read-only check pass over the open map
  (elevation seams, off-map units, broken trigger and unit references, empty
  players, units stranded on small land areas, missing instructions). Each
  check reports clean, findings or unavailable; double-click a finding to jump
  to it.
- **`descape/scatter.py`** places units at random, reproducible positions
  across a tile set for batch scripts, with a runnable
  `batch_scripts/scatter_fish_in_water.py` example.
- **`tools/gen_contact_shadow_eyeball.py`** renders A/B sheets of Stepped
  contact-shadow settings, with connectivity and coverage numbers per cell.
- **Place Unit's Free placement checkbox** drops a unit exactly under the
  cursor instead of at the centre of the tile it fell in, in all three render
  styles; hold Alt for a one-off free placement without changing the setting.
- **Autosave**, on by default, writes a recovery snapshot every few minutes to
  its own rotating slot: never the file you have open, and it never clears the
  unsaved-changes marker. File > Recover from Autosave… lists the slots and
  opens one as a new untitled document. Settings > Saving holds the interval,
  how many to keep, whether they go beside `config.yaml` or beside the
  scenario, and a switch for the `.bak`/`.orig` pair a real save writes.

### Fixed

- **Map Options group boxes no longer clip their last rows at larger UI font
  sizes.** A group whose rows wrap in a narrow pane reported the height it
  would need if nothing wrapped, so Global Victory drew its custom-victory
  rows outside its own box. It now measures each group at the width it is
  actually given, and scrolls instead of clipping.

- **The trigger editor's message boxes fit real dialogue** (GH #38). The
  trigger's own Description and Short Description, and the message on Display
  Instructions, Send Chat, Display Timer, Change Object Description and Change
  Technology Description, are now word-wrapping six-line boxes instead of
  one-line fields. Each keeps whichever line separator the file already used,
  so editing one does not rewrite the rest of the field.

- **Clicking a unit in a downloaded build no longer crashes** (GH #72). Three
  data files, including the one the unit stats panel reads, were missing from
  the packaged build. The build now bundles every `descape/*.json` it finds
  rather than a hand-kept list, and the bundle's own `--self-check` derives the
  files it verifies from the code that reads them.

- **View > Grid no longer draws a line across the middle of a raised tile** in
  Stepped with Follow Terrain Elevation on. A tile boundary's lower copy is now
  skipped where the nearer tile's slab would bury it, so every grid line sits on
  a tile edge.

- **A Players-tab colour edit reaches the map.** Changing a player's colour now
  repaints its unit marks and sprite tint immediately, and updates the P1..P8
  swatches in both Players and Diplomacy mode. Undo and redo reverse all three.

- **The region selection's marching ants trace its true outer edge**, instead
  of outlining every boundary tile's own diamond. No ant marks land on the
  seams between tiles inside the selection any more.
- **Palisade Walls, Fortified Palisades and Sea Walls draw their banner**,
  which is also the only piece of a palisade that carries player colour, and a
  Sea Wall now sits on its submerged rocky base instead of on nothing.
- **A gate or corner pillar no longer disappears entirely** when one of its
  pieces fails to decode: the pieces that did resolve still draw, matching what
  two of the gate consts already did.
- **The Terrain brush cursor outlines every tile edge.** Each tile in the brush
  footprint was drawn from an unclosed path, so one of its four edges never
  stroked; the highlight's own fill had been hiding it. The Mirror Map overlay
  had the same open path and now strokes all its edges too.
- **Units render at the position their file actually stores.** Any sub-tile
  position was previously rounded away to the tile centre, so the 0.1
  arrow-key nudge and the inspector's two-decimal X/Y boxes moved a unit
  without moving it on screen, and units saved off-centre by the in-game
  editor were drawn up to half a tile from where they stand.
- **The hover tile readout** no longer stops updating on non-square maps.
- **Stepped contact shadows** no longer draw small tick marks at tile corners
  along straight terrace edges, or break at the vertex where an edge turns an
  inside corner.
- **Paste Region now preserves garrison links inside the copied region.**
  Copying a building with units garrisoned inside it and pasting it used to
  drop them to loose units standing on the same tile; the pasted occupant is
  now still garrisoned in the pasted building.

### Changed

- **Trigger fields with 20 or more choices** (attributes, object classes,
  damage classes and the like) use a type-ahead box with a "…" browse list
  instead of a long dropdown.

- **The Convert brush recolours units while you drag**, not only when the
  mouse is released. Still one undo step per drag.

- **Stepped contact shadows scale with step height**: taller steps cast longer
  shadows, filling in as triangles at high elevation step settings, and low
  settings are lighter than before.

- **A unit standing inside a town centre paints between the building's
  pieces** rather than behind the whole building, and a pasture's corner posts
  depth-sort against nearby units individually.

- **Elevation edits repaint a smaller region**, so a drag over hilly terrain
  keeps up better: the redrawn area now covers only the heights a tile was
  actually drawn at, instead of the whole 0-15 range.

## [0.6] - 2026-09-16

### Added

- **File > Open Recent**, listing the 10 most-recently-opened scenarios (by
  filename, full path on hover), plus a Clear Recent Files entry. Updated on
  every successful open; an entry whose file has since disappeared is hidden
  rather than pruned, so it reappears if the file comes back (e.g. a
  removable drive reconnected).
- **The Units mode inspector shows a selected unit's base stats**: hit
  points, attack, melee and pierce armour, and range, read from the game's
  own unit table. Values are base only, with no civ bonuses, tech upgrades
  or trigger effects applied, and the whole block hides for a unit with no
  combat stats (a tree, say).
- **The Units mode catalog and owner selector now live in the Units panel
  itself**, in a resizable vertical split above the inspector, replacing the
  separate toolbar pair.
- **A "More Tools" button** replaces Qt's anonymous `>>` chevron for tool
  buttons that don't fit the toolbar at the current window width, naming
  the active tool when it's one of the overflowed ones.

### Changed

- **Terrain painting is another 20-30% faster in Stepped and Sloped modes.**
  Tiles that end up drawing nothing in the repainted area are now skipped
  before any texture work. A size-9 brush stroke with sprites on drops from
  about 62 ms to 51 ms a step in Sloped and 40 ms to 31 ms in Stepped on a
  large map. Rendering output is unchanged, pixel for pixel.
- **Mouse-wheel zoom steps once per whole wheel notch**, so a
  high-resolution wheel or trackpad no longer zooms on every small scroll
  event, and one hard flick crosses at most one zoom level.
- **The first zoom after a stroke, paste or undo no longer stalls for
  seconds.** Neighbouring zoom levels are rebuilt in the background after
  every edit, not only after opening a file.
- **The tile highlight holds steady instead of pulsing while a stroke is in
  progress**, leaving those frames to the edit itself.

### Fixed

- **A pure horizontal wheel tilt no longer zooms out one step.**
- **The first unit edit of a file shows a wait cursor** instead of appearing
  to hang while unit editing is set up.
- **Place Unit, Paint Can fill, and Save no longer crash on a scenario older
  than version 1.55, or on any scenario opened after an older one earlier in
  the same session.** A caption-field compatibility shim could leave the
  `Unit` class "poisoned" against captions in a way that made these paths
  raise instead of just omitting the unsupported field.
- **Farms in Sloped mode no longer render as a plain coloured mark on open.**
  A freshly-opened scenario, or toggling `View > Show sprites` on, skipped
  the real terrain drape and fell back to the plain mark until the next
  elevation edit; now both routes drape immediately.

- **Cycling a gate's orientation no longer reshapes its neighbouring walls
  inconsistently.** The wall/gate connector set used to be a hand-kept
  15-const list that only covered some gate families; a palisade gate's
  cycle could flip whether an adjacent wall drew a connector while a stone
  gate's cycle didn't. It's now generated from every gate family, so the
  behaviour is consistent across gate types.

## [0.5] - 2026-09-09

### Added

- **Draw and Paint Can can auto-place trees and eye candy.** Two new toolbar
  checkboxes, Trees (on by default) and Eye candy (off), mirror the in-game
  editor's own option: painting a forest terrain scatters its matching GAIA
  tree per tile, with a varied graphic, instead of leaving a bare texture.
  Painting a different terrain over a tile removes what was there. One undo
  step covers both the terrain and the trees/eye candy it triggers.
- **Rotate now turns gates.** A gate stores its orientation in its object
  type rather than in the rotation field, so Rotate cycles a selected gate
  through its four orientations (45 degrees a press, 90 for the quarter-turn
  action) and re-anchors it so the new footprint keeps the tiles it stood on.
  Gates and ordinary units can be rotated together in one undo step; a gate
  with no room for its new footprint at the map edge is skipped and reported.
- **Settings > Appearance: distance-tick label size.** The map-edge distance
  ruler's major-tick numbers had a fixed 12px font; it's now a spinbox next
  to the Ruler label size one, 8-24px.
- **A Select tool in Terrain mode (bound to `S`).** Drag a tile rectangle to
  select a region; `Ctrl+A` selects the whole map, `Ctrl+Shift+A` or Escape
  clears it. Copy Region/Paste Region (`Ctrl+C`/`Ctrl+V`) now capture and
  stamp back a whole region's terrain, elevation and units together, with a
  checkbox per category deciding what a paste actually writes -- replacing
  the old single-tile, tool-scoped copy/paste. A paste with more than one
  category checked is a single undo step.
- **Settings > Appearance: tool overlay colors and Ruler label size.** Every
  overlay DEscape draws on the map -- the edit-tool highlight, Pan's hover
  outline, Units mode's hover/selection cues, the Ruler's line/label and the
  Select tool's marching ants -- was a hardcoded color; each is now its own
  picker, grouped by tool. The Ruler's on-map label font size is now a
  spinbox next to them instead of a fixed 18px.

### Changed

- **Terrain painting is roughly twice as fast in Sloped mode.** The
  per-tile compositing kernel was reworked (cached slope shading, a
  skirt-shade lookup table, direct terrain indexing, a scalar bounds
  pre-check, and a bucketed scan for buildings overlapping the repainted
  area). A size-9 brush stroke with sprites on drops from about 125 ms to
  62 ms a step on a large map, and Stepped mode from about 57 ms to 37 ms.
  Rendering output is unchanged, pixel for pixel.

- **The load/New status line now reports the deferred first-paint
  composite.** A second line, `First paint composited in X.XXs (total
  X.XXs, mip M)`, follows `Loaded ...`/`Created ...` once the map has
  actually drawn - previously that cost was visible only under Help > Perf
  Trace, and on a big map it is around a third of the real wait.

- Copy/Paste no longer depend on which edit tool is active -- they act on
  the Select tool's own region and clipboard instead, the same way Rotate
  acts on the unit selection regardless of tool.

### Fixed

- **Unit sprites face the right way in Flat mode.** With `View > Isometric
  View` unticked, every unit icon was drawn an eighth of a turn out: the
  isometric camera's own 45-degree facing correction was still being applied
  to a top-down grid that has no such rotation. A unit at rotation 0 now
  reads as facing screen-right. Stepped, Sloped and Flat with Isometric View
  ticked are unchanged, as are walls, cliffs, gates and trees, whose stored
  frames are shapes rather than facings.
- **Sloped terrain: unit and farm markers no longer spill off a ramped
  tile.** A unit or farm on a slope used to paint a flat, unwarped mark
  that could leave part of its own tile bare or bleed past its edge. A
  single-tile unit's marker now conforms exactly to its tile's warped
  footprint, and a farm drapes as real terrain (matching Stepped) with a
  warped perimeter outline whenever `View > Show sprites` is on; with
  sprites off, a farm keeps its previous plain mark. Multi-tile buildings
  are unaffected.

## [0.4] - 2026-09-07

### Added

- **A Cliff tool in Terrain mode.** Click to place one cliff of the family,
  piece and frame you picked, with a live sprite preview. Drag to lay a
  connected run: pieces step along their own footprint, and each one's shape
  is chosen from its neighbours, so runs and corners join up instead of
  needing to be assembled by hand. The whole drag is a single undo step. All
  ten cliff families are available, including the ones the in-game editor
  hides.
- **An Eyedropper tool in Terrain mode.** Click a tile to load its terrain
  and elevation straight into the toolbar (bound to `I`), instead of hunting
  the Terrain type combo's 131 entries. One pick feeds both Draw/Paint Can
  and Set Elevation.
- **Crash reports.** An unhandled exception (or a hard crash) now writes a
  dump with the traceback, app/OS version, and recent debug-log lines to a
  file next to your config, and shows it in a dialog with buttons to copy
  it, open its folder, or save your work to a new file. If the app dies too
  hard to show that dialog, it's surfaced the next time you launch instead.
- **The first zoom into a new map no longer stutters.** After a file opens,
  DEscape resolves the neighbouring zoom levels' unit sprites in idle time, a
  few milliseconds at a time, so the window stays fully interactive
  throughout. On by default; turn it off under `Settings > Appearance >
  Preload neighbouring zoom levels`.
- **Panning into never-visited territory stutters less.** DEscape now warms a
  ring of chunks just outside the current viewport in idle time as you pan,
  re-targeted continuously and prioritizing the direction you're heading, so
  scrolling past the edge of what's already on screen has less to composite
  on demand. Same `Preload neighbouring zoom levels` setting as above.
- **Units can now be rotated**, via `Edit > Rotate Selection`, the two
  toolbar buttons, or the `,` / `.` keys (`<` / `>` for a quarter turn). The
  whole selection turns in one undo step, each unit about its own centre.
  Only units whose rotation is a real facing turn: for walls, gates and most
  GAIA objects that field stores a graphic-variant index instead, and those
  are skipped rather than corrupted. The inspector's Rotation field is
  editable on the same units, in raw radians. The toolbar buttons are hidden
  outside Units mode instead of sitting permanently greyed out.
- **Packaged Linux, macOS and Windows builds.** DEscape now ships as a
  self-contained PyInstaller build, no separate Python or `.venv` setup
  needed, with a Linux AppImage on top of it. CI builds and smoke-tests all
  three platforms on every change.
- **`View > Isometric View` in Flat mode is now a real isometric render**,
  not a view rotation applied on top of the flat canvas. Terrain paints as
  real iso diamond tiles and units as upright, ground-anchored sprites (the
  same look Stepped mode draws), all at elevation 0 regardless of the
  scenario's real terrain heights. The checkbox ships checked, so opening
  Flat shows this by default; untick it for the plain top-down grid with
  footprint-fitted icons instead.
- **`View > Show sprites` now works in Flat mode too**, so it is no longer
  greyed out in any terrain style. Flat is a square top-down grid rather
  than an isometric one, so each unit gets its real sprite fitted into its
  own footprint square (aspect preserved, centred) instead of the isometric
  sprite Stepped and Sloped draw.
- **Tribe name is now editable in Players mode**, the one Identity field
  0.3's Players mode shipped read-only (it needed a text widget the panel
  hadn't built yet). Undo/redo alongside every other Players mode edit.
- **Civilization and Architecture are now editable in Players mode, on
  every scenario version.** Below version 1.56 both are a fixed-width
  field; from 1.56 on they're stored as variable-length text, which now
  resizes correctly when a shorter or longer civilization/architecture
  name is chosen.
- **Number of players is now editable in Players mode**, from a spinbox
  above the player selector. Raising it activates the next players in
  order, lowering it deactivates the highest-numbered ones; either way the
  Diplomacy tab's stance grid resizes to match straight away. A handful of
  older files store their header in a layout DEscape can't resolve, and
  show this one setting read-only while everything else stays editable.

- **Map mirroring** (`Map > Mirror Map…`, terrain and elevation only). Pick
  one of nine symmetries (mirror, 180°/90° rotation, 4-way or 8-way) and
  which half/quadrant/octant is authoritative, and DEscape rewrites the rest
  of the map to match in one undo step, with a live preview of both the
  source slice and the resulting map before you commit. Units are not
  mirrored yet.

### Changed

- **New app icon**: a green D on a gray background, replacing the placeholder
  icon.

### Fixed

- **The Keybinds tab's Default button no longer silently clears another
  action's binding** when the default it would restore collides with one you
  set deliberately elsewhere. It now leaves your binding alone and shows an
  inline warning explaining why.
- **A genuine zero-edit save (open, touch nothing, Ctrl+S) now writes back
  byte-for-byte identical to the file on disk**, for every corpus file, not
  just files whose original compression happened to match Python's zlib
  output. The write path now reuses the original compressed bytes verbatim
  when nothing changed, instead of always recompressing the decompressed
  body (which doesn't reliably reproduce the original compressed stream).
  Previously, a habitual save on an unaffected file silently rewrote it and
  churned its `.bak`/`.orig` backups even with no edits made.
- **Gates now render as whole buildings** with sprites on, instead of just
  their middle span (a 4-tile gate used to read as a floating portcullis
  arch with no corner towers or flags).
- **Diagonal gates now claim their real 6-tile footprint, not their full 4x4
  bounding box.** Fixes wrong tile-claiming/selection-depth in Stepped and
  Sloped (a diagonal gate's sprite/pick could land on a tile it doesn't
  actually occupy); Flat mode is unaffected by design. In-app selection,
  right-click convert, and picking near a diagonal gate's bbox corners now
  correctly miss the four "gap" tiles that were never really part of it.
- **The "in X.XXs" status/log messages after a load or render used to
  understate real time**, since they only ever covered synchronous prep with
  the deferred compositing pass (and, for a file open, the parse itself)
  excluded from the number entirely. A load's message now reports total
  synchronous time with a parse/prepare breakdown; every re-render message
  that still only times prep now says "prepared in" rather than "in"; Paint
  Can and the sprite toggle say "applied in", matching what they actually
  time. The deferred composite cost itself is exposed separately, under
  Help > Perf Trace, once a load's first paint completes.
- **Wall and gate rotation now reads correctly on scenario files whose
  rotation values carry no shape information at all** (radian-encoded
  files such as El Cid and the June Event scenario, reported 2026-08-29):
  a run/corner/junction's shape is now derived from which of its
  neighbours are also walls or gates, in every render mode. A wall piece
  with no wall/gate neighbour, and gates themselves, are known remaining
  exceptions.
- **Trees, and every other non-creatable object with more than one stored
  shape (Aqueduct, Granary, statues, decorative doodads, ...), now draw
  their real variant instead of always the first one.** Oak, with 42
  possible crowns, used to draw the same one everywhere; a forest now shows
  real variety. Fixes an out-of-range stored index (mangrove and a few
  others) landing on the wrong frame instead of wrapping back into range.

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
