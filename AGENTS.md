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
  7..53). Every write path must pass it through verbatim, never normalize it.
- **The one exception to "verbatim", and its exact scope.** `descape/
  unit_rotation.py` classifies each `unit_const` as ANGLE, VARIANT or INERT,
  and the Rotate action is the only code anywhere that may transform an
  existing unit's `rotation`, and only on ANGLE consts. Every other write path,
  and every VARIANT or INERT const, stays verbatim as above.
  `UnitEditModel.set_rotation()` raises rather than silently no-op'ing on a
  non-ANGLE const, and that guard is what keeps the rule enforced rather than
  merely documented. The whitelist is the .dat's own `unit.type == 70`
  (creatable), which has zero counterexamples across 5808 corpus placements,
  plus four hand-verified trebuchet consts; it is re-measured by a
  corpus-marked test, not trusted.
- Gates carry no rotation at all: all 24 visible gate consts (6 families × 4
  orientations) have `angle_count == 1`, so there is no second frame for a
  rotation to select, and across 300 corpus gate placements the field is only
  ever `0.0` or the junk sentinel `7.0`. A gate's orientation lives in its
  `unit_const`. The graphic file names are `..._ne_closed_x1` / `_se_` / `_e_`
  / `_n_`, one const per orientation (stone: 64/88/659/667). "Rotating" a gate
  therefore means swapping the const among its four siblings, which changes the
  footprint; it is not a rotation and is not implemented.
- The same is true of walls, for a different reason, and it is confirmed
  in-game: a wall graphic's five stored frames are SHAPES (two diagonal runs, a
  tower, a flatter run, a narrow column), not five facings, so `rotation`
  selects a variant. A wall cannot be rotated into a tower. Real files encode
  that index two ways for the same graphic — a literal `0..4`, or the same
  index as `k*2π/5` radians — so both must be read, and neither may have an
  angular zero-point offset applied. `unit_sprites.variant_index()` is the only
  correct reader; `angle_index()` mis-maps 3 and 4. A write path must pass the
  field through verbatim here too. A wall's correct stored value is also a
  function of its neighbours (98.9%/99.1% agreement between neighbour mask and
  stored index across the corpus), so rotating one would write a value the game
  re-derives. Walls are VARIANT, and Rotate skips them.
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
