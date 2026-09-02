# AGENTS.md

External viewer/editor for Age of Empires 2: Definitive Edition
`.aoe2scenario` files. Start with `README.md` — it covers setup, build/test
commands, and architecture.

## Hard rules

- Never write to a path under a Proton `compatdata/` prefix — that's the user's
  only copy of Workshop-synced scenario files. v1 has no write path at all; a
  future one must refuse any output path containing `compatdata`.
- GAIA units' `rotation` field is not an angle for ~65% of GAIA objects — it's a
  tree/doodad graphic-variant index (integer values well outside `[0, 2π)`, e.g.
  7..53). Any future write path must pass it through verbatim, never normalize it.
- The same is true of walls, for a different reason, and it is confirmed
  in-game: a wall graphic's five stored frames are SHAPES (two diagonal runs, a
  tower, a flatter run, a narrow column), not five facings, so `rotation`
  selects a variant. A wall cannot be rotated into a tower. Real files encode
  that index two ways for the same graphic — a literal `0..4`, or the same
  index as `k*2π/5` radians — so both must be read, and neither may have an
  angular zero-point offset applied. `unit_sprites.variant_index()` is the only
  correct reader; `angle_index()` mis-maps 3 and 4. A write path must pass the
  field through verbatim here too.
- A unit's `standing_graphic` is not always the thing you draw. For some consts
  it is a DECORATION — an animated flag — whose real body is a delta hanging
  off it, and whose own art covers only the one shape the decoration sits on.
  Resolving "the first graphic with a modern-looking file_name" picks the
  overlay and draws a pennant floating over nothing.
  `tools/gen_unit_graphic_map.py`'s `_BODY_GRAPHIC_OVERRIDES` records the
  confirmed cases.

## Changelog

User-visible changes here mean behavior, rendering, the write path, or a new
tool. `CHANGELOG.md` uses `MAJOR.MINOR` versioning (no patch number).

## Private maintainer repo

If `maintainer/` is present alongside this repo on disk, it has its own
fuller `AGENTS.md` — check it for open-work tracking and deeper context
before starting non-trivial work. If it isn't present, it simply isn't part
of what you're working with.
