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
- **The exceptions to "verbatim", and their exact scope.** This list is
  exhaustive: every other write path stays verbatim as above, and each
  exception that goes through a model method raises rather than silently
  no-op'ing outside its scope, which is what keeps the rule enforced rather
  than merely documented.
  - *Rotate*, via `UnitEditModel.set_rotation()`, only on ANGLE consts.
    `descape/unit_rotation.py` classifies each `unit_const` as ANGLE, VARIANT
    or INERT. The ANGLE whitelist is the .dat's own `unit.type == 70`
    (creatable), which has zero counterexamples across 5808 corpus placements,
    plus four hand-verified trebuchet consts; it is re-measured by a
    corpus-marked test, not trusted.
  - *Map mirroring's unit images*, via `mirror_tools.plan_mirror_units()`.
    This one transforms nothing in place: it derives a NEW unit's rotation
    from its source's and `UnitEditModel.add()`s it, so there is no model
    guard here, only the plan's own rule. A wall's stored run-direction index
    (0 along x, 1 along y) swaps under the four axis-swapping D4 elements,
    and only in a file that encodes the index as a literal integer (a
    radian-encoded file carries no shape information there, measured). A
    real facing angle, only on the consts `unit_rotation.rotation_is_angle()`
    calls ANGLE (the predicate Rotate uses, never the player id), maps through
    `mirror_tools.facing_image()`, whose 2x2 matrix is `TRANSFORMS`' own
    linear part, so the facing cannot drift from the positions. Every other
    rotation, VARIANT and INERT alike, is copied verbatim.
    The angular modes (3-way, 6-way rotational, 6-way reflective: tile-space
    rotations by multiples of 60 degrees, not D4 elements) extend this with
    three accepted approximations, each derived from the element's own
    linear action rather than hand-typed. A building keeps its const's
    axis-aligned span: its footprint centre is rotated and the same span is
    re-anchored around it on whole tiles, since AoE2 has no rotated
    footprint. A gate's image takes the orientation sibling nearest the
    rotated run direction (always 15 degrees off, never a tie), still via
    `reorient_gate_const()`. A wall's run-direction index swaps when the
    element carries the x axis nearer the y axis (60 and 120 degrees and `a`
    do, 180 degrees and the other two reflections do not), under the same
    integer-encoded-file rule. A facing angle rotates through
    `facing_image()` with the angular element's own matrix.
  - *Cycle Variant*, via `UnitEditModel.set_variant()`, only on consts
    `descape/unit_variant.py` calls cyclable: VARIANT, at least two real
    variants, and not a wall, cliff or gate. Those three are excluded by const
    set (never by `unit.class_`, which would also drop Aqueduct and Mole)
    because the game re-derives their index from neighbours, so a written
    value would be overridden. It always writes a literal integer index: every
    corpus placement on a cyclable const stores one, never the radian form.
  - *Place Unit's wall-run junction rewrites*, via
    `UnitEditModel.set_wall_variant()`, only on the 8 wall consts
    `unit_sprites.rotation_variant_eligible()` accepts (`angle_count == 5`)
    and only for an index in `0..4`. Allowed for the same reason the bullet
    above forbids *cycling* a wall: the game re-derives a wall's index from
    its neighbours, so writing the value derived from those same neighbours
    converges with the game rather than diverging from it, where an
    author-picked one would just be overwritten. The index is always the
    literal integer, never the radian re-encoding, and
    `initial_animation_frame` is not touched (it is 0 on all 8193 corpus wall
    placements). Gates are outside the scope and stay there: they have
    `angle_count == 1` and their orientation lives in the const.
- Gates carry no rotation at all: all 24 visible gate consts (6 families × 4
  orientations) have `angle_count == 1`, so there is no second frame for a
  rotation to select, and across 300 corpus gate placements the field is only
  ever `0.0` or the junk sentinel `7.0`. A gate's orientation lives in its
  `unit_const`. The graphic file names are `..._ne_closed_x1` / `_se_` / `_e_`
  / `_n_`, one const per orientation (stone: 64/88/659/667). "Rotating" a gate
  therefore means swapping the const among its four siblings, which changes the
  footprint.
- **The one exception to "a placed unit's `unit_const` never changes", and its
  exact scope.** `UnitEditModel.set_unit_const()` is the only code anywhere
  that may change an existing unit's `unit_const`, and only to one of the four
  orientation siblings `descape/gate_orientation.py` derives for it; anything
  else raises rather than silently no-op'ing, and that guard is what keeps this
  rule enforced rather than merely documented. It must re-anchor `x`/`y` by
  preserving the footprint's low corner (`span_low_corner()` forward,
  `render.span_anchor()` back), because the four orientations have four
  different spans and every corpus placement sits at `tile + span/2` per axis.
  `rotation` and `z` pass through verbatim as above: a gate's stored rotation
  is `0.0` or the junk sentinel `7.0`, and every sibling has
  `angle_count == 1`, so there is nothing there to normalize.
  Map mirroring is not an exception to this rule: it never changes a placed
  gate's const, it `add()`s a *new* gate whose const comes from
  `mirror_tools.reorient_gate_const()` and whose anchor is re-derived from
  that sibling's own span.
- The same is true of walls, for a different reason, and it is confirmed
  in-game: a wall graphic's five stored frames are SHAPES (two diagonal runs, a
  tower, a flatter run, a narrow column), not five facings, so `rotation`
  selects a variant. A wall cannot be rotated into a tower. Real files encode
  that index two ways for the same graphic — a literal `0..4`, or the same
  index as `k*2π/5` radians — so both must be read, and neither may have an
  angular zero-point offset applied. `unit_sprites.variant_index()` is the only
  correct reader; `angle_index()` mis-maps 3 and 4. A write path must pass the
  field through verbatim here too, with the single exception listed above
  (`UnitEditModel.set_wall_variant()`, which writes the value derived from
  the same neighbours the game would read). A wall's correct stored value is also a
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
