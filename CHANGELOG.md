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
