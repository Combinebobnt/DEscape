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

## [Unreleased]

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
