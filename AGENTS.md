# AGENTS.md

External viewer/editor for Age of Empires 2: Definitive Edition
`.aoe2scenario` files. Start with `README.md` — it covers setup, build/test
commands, and architecture.

## Hard rules

- Never write to a path under a Proton `compatdata/` prefix — that's the user's
  only copy of Workshop-synced scenario files. Every path that writes a
  scenario file (Save, autosave, and the tools that write one directly)
  refuses an output path containing `compatdata`, as written or resolved
  through symlinks (`scenario_io.is_under_compatdata()`). A new one must too.
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
    corpus-marked test, not trusted. The Rotate actions and the Rotation
    field also step cyclable members (trees, scenery) through their variants,
    but only via `UnitEditModel.set_variant()`, i.e. the *Cycle Variant*
    exception below with its scope unchanged; `set_rotation()` stays
    ANGLE-only.
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
    set (never by `unit.class_`, which would also drop Mole)
    because the game re-derives their index from neighbours, so a written
    value would be overridden. It always writes a literal integer index: every
    corpus placement on a cyclable const stores one, never the radian form.
  - *Place Unit's wall-run junction rewrites*, and its rewrites of the walls
    beside a gate it places (GH #159, which also removes the walls under the
    gate's footprint), via
    `UnitEditModel.set_wall_variant()`, only on the 9 wall-family consts
    `unit_sprites.rotation_variant_eligible()` accepts (`angle_count == 5`)
    and only for an index in `0..4`. Those are the 8 walls plus Aqueduct
    (231), which the user decided on 2026-09-26 (GH #110) to treat as a full
    wall: same 1x1 footprint and five stored shapes. No corpus file has a
    wall beside an Aqueduct, so that junction rests on the decision, not a
    measurement. Allowed for the same reason the bullet
    above forbids *cycling* a wall: the game re-derives a wall's index from
    its neighbours, so writing the value derived from those same neighbours
    converges with the game rather than diverging from it, where an
    author-picked one would just be overwritten. The index is always the
    literal integer, never the radian re-encoding, and
    `initial_animation_frame` is not touched (it is 0 on all 8193 corpus wall
    placements). Gates are outside the scope and stay there: they have
    `angle_count == 1` and their orientation lives in the const.
  - *Replace* (Edit > Find and Replace, GH #144), via
    `UnitEditModel.replace_type()`, which changes a unit's const (see the
    const exceptions below) and so must decide what its `rotation` and
    `initial_animation_frame` mean under the new const. The two move together
    (they are always equal in the corpus for cliffs, trees and doodads), by
    `unit_model.replaced_rotation()`: verbatim when both consts are ANGLE,
    both are walls (the 9 `rotation_variant_eligible()` consts; the game
    re-derives the index from neighbours), or both are INERT (which keeps the
    `7.0` sentinel); verbatim for two cyclable consts when the old variant
    index exists in the new const, else `0.0`/`0`; `0.0`/`0` for every other
    pairing and for any target gate.
- Gates carry no rotation at all: all 24 visible gate consts (6 families × 4
  orientations) have `angle_count == 1`, so there is no second frame for a
  rotation to select, and across 300 corpus gate placements the field is only
  ever `0.0` or the junk sentinel `7.0`. A gate's orientation lives in its
  `unit_const`. The graphic file names are `..._ne_closed_x1` / `_se_` / `_e_`
  / `_n_`, one const per orientation (stone: 64/88/659/667). "Rotating" a gate
  therefore means swapping the const among its four siblings, which changes the
  footprint.
- **The two exceptions to "a placed unit's `unit_const` never changes", and
  their exact scope.** `UnitEditModel.set_unit_const()` and
  `UnitEditModel.replace_type()` are the only code anywhere that may change an
  existing unit's `unit_const`. Each raises outside its scope rather than
  silently no-op'ing, and that guard is what keeps this rule enforced rather
  than merely documented.
  - `replace_type()` is user-directed, from Edit > Find and Replace only (GH
    #144). It keeps `reference_id`, so triggers and garrison links that name
    the unit keep working, plus the list slot, owner, `z`, status, caption,
    `capture_flag` and `garrisoned_in_id`. `x`/`y` stay verbatim on an axis
    whose span is unchanged and are re-anchored on the footprint's low corner
    only on an axis whose span changes. It raises (`replace_refusal()`) for
    the same const, a const `object_catalog` does not know, a cliff on either
    side, two orientation siblings of one gate (that is `set_unit_const()`'s),
    an off-map unit or a new footprint leaving the map, and a garrison the
    new const cannot hold or a host that cannot hold the new const.
  - `set_unit_const()` changes a gate's const only to one of the four
    orientation siblings `descape/gate_orientation.py` derives for it. It
    must re-anchor `x`/`y` by preserving the footprint's low corner
    (`span_low_corner()` forward, `render.span_anchor()` back), because the
    four orientations have four different spans and every corpus placement
    sits at `tile + span/2` per axis. `rotation` and `z` pass through
    verbatim as above: a gate's stored rotation is `0.0` or the junk sentinel
    `7.0`, and every sibling has `angle_count == 1`, so there is nothing
    there to normalize.
  - Map mirroring is not an exception to this rule: it never changes a placed
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
  field through verbatim here too, with the two exceptions listed above:
  `UnitEditModel.set_wall_variant()`, which writes the value derived from
  the same neighbours the game would read, and *Replace*
  (`UnitEditModel.replace_type()`, by `unit_model.replaced_rotation()`),
  which keeps a wall's index verbatim when the new const is also a wall and
  writes `0.0` when a non-wall becomes a wall. A wall's correct stored value is also a
  function of its neighbours (98.9%/99.1% agreement between neighbour mask and
  stored index across the corpus), so rotating one would write a value the game
  re-derives. Walls are VARIANT, and Rotate skips them.
- **An existing unit's `garrisoned_in_id` changes only through
  `UnitEditModel.set_garrisoned_in()`** (GH #115), which splices the garrison
  reverse-map and raises outside its structural scope: never inside a
  `fields_only` edit, and when linking, never into itself, into a reference_id
  no live unit has, into a host that is transitively inside the unit (a
  cycle), or for a unit that holds a garrison of its own (no nesting). -1
  unloads and is always allowed. Type and capacity are not model rules: every
  UI path checks them through `garrison.refusal()`, so a batch script stays
  unvalidated, as with `add()`. Region paste and `add()` set the field at
  construction only.
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
