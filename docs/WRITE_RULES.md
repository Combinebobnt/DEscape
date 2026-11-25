# Write rules: each exception's exact scope

`AGENTS.md`'s Hard rules state every write-path rule and name each exception
to it, with its guard and the test that pins it. This file holds what those
one-line entries leave out: each exception's exact scope, and the corpus
figures behind the rules. Read it before changing any code that writes,
reads or draws a unit's `rotation`, `initial_animation_frame`, `unit_const`
or `garrisoned_in_id`, or adding a write path. A new exception adds its
one-line entry to `AGENTS.md` and its scope here. The `garrisoned_in_id`
rule has no exception scope beyond `AGENTS.md`'s bullet.

## Per game (GH #67)

A scenario is edited with its own game's data. Each game is a descriptor
in `descape/games.py`, and Return of Rome claims header variant 2
(`docs/GAME_DATA.md` has the registry). The viewer refuses every edit and
save of a document shown with another game's data
(`game_dataset.lock_reason()`, behind `ViewerWindow.game_data_lock_reason()`).
That lock is the viewer's; the model guards below still raise in a batch
script. The hand-measured sets below (wall and cliff consts, the GAIA-only
objects) are `games.PerGame` tables, one entry per game. A data
mod (GH #58, File > Data Mod) swaps the AoE2 dataset's tables for ones
derived from its own `.dat` (`game_dataset.set_data_mod()`): the
classifiers below then read the mod's tables, and no write rule changes.
Every const set the rules below name is the active dataset's, so each
exception's scope follows the file being edited only while its game is
active: the viewer activates it on open, a script through
`batch_api.load()` (a bare `scenario_io.load_map_and_units()` does not).
The AoE2 figures below come from the AoE2 corpus. The Return of Rome ones
come from 8 RoR scenarios (100,098 placements, 2026-10-17). The ANGLE,
wall, cliff, gate, Cycle Variant and farm rules (not the exact counts, and
the wall neighbour agreement only to at least 95%) are re-checked by
`tests/test_ror_corpus.py`, run as
`pytest -m corpus tests/test_ror_corpus.py --scenario-dir <RoR folder>
--corpus-full` (it skips where there are no RoR files). The
`_BODY_GRAPHIC_OVERRIDES` consts are pinned against the committed RoR table
by `tests/test_ror_editing.py`:

- ANGLE: the RoR `.dat`'s own type 70 plus the same four trebuchet consts
  (the same objects there, hidden in its editor); 1,697 placements, zero
  counterexamples.
- Walls: 7 walls plus Aqueduct (WALL5 2678 does not exist there); 1,548
  placements, all literal integers with `initial_animation_frame` 0,
  96.9% neighbour agreement.
- Cliffs: 60 cliff consts (no Short Marble 2190-2225); 2,110 placements,
  `rotation` equal to `initial_animation_frame` on each.
- Gates: 20 orientation groups (no Fort gate), 12 visible consts; 12
  placements, all `7.0`, all on their span anchor.
- Non-square object siblings (`descape/object_orientation.py`): 38 consts
  in 6 four-groups and 7 pairs (no longhouse or blockers), against AoE2's 44
  in 6 and 10, derived from each game's tables with no RoR placement
  measured; pinned by `tests/test_object_orientation.py`, not this corpus test.
- Cycle Variant: derived from the RoR table, so Farm 50 and RoR scenery
  are cyclable there though INERT in AoE2; 56,364 placements,
  all literal. A RoR farm stores 0-3 with `initial_animation_frame` 0.
- `_BODY_GRAPHIC_OVERRIDES`: 119 and 788 only (72 is a plain wall there).

## The exceptions to "verbatim"

Every write path not listed here passes a unit's `rotation` through
verbatim (`AGENTS.md`'s GAIA rule). The exceptions, in `AGENTS.md`'s order:

- *Rotate*, via `UnitEditModel.set_rotation()`, only on ANGLE consts.
  `descape/unit_rotation.py` classifies each `unit_const` as ANGLE, VARIANT
  or INERT. The ANGLE whitelist is the .dat's own `unit.type == 70`
  (creatable), which has zero counterexamples across 5808 corpus placements,
  plus four hand-verified trebuchet consts; it is re-measured by a
  corpus-marked test, not trusted. The Rotate actions and the Rotation
  field also step cyclable members (trees, scenery) through their variants,
  but only via `UnitEditModel.set_variant()`, i.e. the *Cycle Variant*
  exception below with its scope unchanged; `set_rotation()` stays
  ANGLE-only. Move is a caller too (GH #280): a Rotate input during a live
  drag only adds steps to a pending turn (`descape/move_turn.py`, the same
  per-const rules), and the release writes it through these same setters,
  in the move's one undo record. Esc, a refused drop and a garrison drop
  write nothing, and a turn that nets a whole cycle for a const leaves its
  stored value verbatim. A wheel row set to Rotate preview or selection,
  or to Brush size, rotate preview or selection (GH #356), is a
  caller as well: with a Units-mode selection and nothing ahead of it, a
  notch runs the Rotate actions' own `ViewerWindow.on_unit_rotate()`, the
  same setters and guards, never a committed region in place. On a just-pasted region (GH #197) the Rotate
  actions edit no unit at all: they go through the *Region turn* exception below. On a
  committed Select region with no paste armed (TASK-256) they go through
  the *Selection transform* exception below instead. A script's
  `editor.rotate()` (`descape/scripting.py`) is a caller too: it refuses a
  non-finite facing, snaps the facing to the const's own frame grid, and
  leaves a non-ANGLE const to `set_rotation()`'s raise.
- *Region turn's unit images* (GH #197: Rotate, Edit > Flip Selection
  and, since GH #245, a wheel bound to Rotate preview, or to Brush size or
  rotate preview where no brush applies, and since GH #356 their "or
  selection" forms where no Units-mode selection exists, on a
  just-pasted region,
  all through `ViewerWindow.on_region_transform()`), via
  `region_transform.transform_block()`.
  Scope: only a block being re-pasted in place of the paste it undoes,
  every unit a NEW one `UnitEditModel.add()`ed, and only the eight D4
  elements. The per-const rules are mirroring's (the *Map mirroring* bullet
  below) because it
  calls the same functions through `mirror_tools.unit_image()`: an ANGLE
  facing through `facing_image()`, a wall's shape index through
  `wall_index_image()` in an integer-encoded file only, a gate's
  orientation through `reorient_gate_const()`, a bridge's, chain's,
  longhouse's or blocker's through `reorient_object_const()`, and every other rotation,
  VARIANT and INERT alike, copied verbatim. `add()` is unvalidated, so the
  guard is `transform_block()`, pinned by `tests/test_region_transform.py`.
  After the re-paste each pasted wall takes its neighbour-derived shape
  (the *Wall shapes re-derived from neighbours* bullet's paste clause), so
  the image index survives where it agrees with the new neighbours.
- *Selection transform's in-place unit images* (TASK-256: Edit > Flip
  Selection and Rotate Selection's Rotate Group pair on a Units-mode
  selection, and Flip, Rotate Group and the Rotate actions on a committed
  Select region, all through `ViewerWindow.on_selection_transform()` or
  `_transform_region()`), planned by `region_transform.plan_in_place()`.
  Scope: existing units only (the selection plus what it holds, or the
  units Copy Region would capture plus what they hold, plus each gate's
  corner pieces wherever their anchor tile is), edited in place
  so each keeps its `reference_id`, list slot, owner and every other
  field, and only the eight D4 elements. Every value comes from the same
  functions as Map mirroring and the region turn (`mirror_tools.unit_image()`):
  an ANGLE facing through `facing_image()`, written by `set_rotation()`;
  a wall's shape index through `wall_index_image()`, in an
  integer-encoded file only and only when the element changes it,
  written by `set_wall_variant()`; a gate's orientation through
  `reorient_gate_const()`, and a non-square object's through
  `reorient_object_const()`, written by `set_unit_const()`; a gate's
  carried 1x1 corner piece (TASK-532) at its own image point, an end tile
  of its gate's image, as `gate_orientation.corner_piece()` of the gate's
  image const (`unit_images()`' `gate` index), written by
  `set_unit_const()`; and every other
  rotation (cliffs, VARIANT, INERT) left verbatim, with no write at all.
  Guards: the three setters' own, plus `plan_in_place()` pinned by
  `tests/test_region_transform.py`. Allowed for its own reason, not the
  *Wall shapes re-derived from neighbours* bullet's: the permuted value is the
  element's image of a shape or facing the game already accepted, the
  same reason Mirror Map and the region turn may write it. Each step
  re-plans from the first state with the element composed since, so a
  run of flips and turns cannot drift.
- *Map mirroring's unit images*, via `mirror_tools.plan_mirror_units()`.
  This one transforms nothing in place: it derives a NEW unit's rotation
  from its source's and `UnitEditModel.add()`s it, so there is no model
  guard here, only the plan's own rule. A wall's stored shape index (0 a
  run along x, 1 along y, 3 and 4 the two diagonal runs) goes through
  `mirror_tools.wall_index_image()`, which derives each D4 element's
  permutation from `unit_sprites.wall_variant_from_neighbours8()`: 0 and 1
  swap under the four axis-swapping elements (`r`, `r3`, `d`, `a`), 3 and 4
  swap under `r`, `r3`, `mx` and `my`, and 2 never changes. That happens
  only in a file that encodes the index as a literal integer (a
  radian-encoded file carries no shape information there, measured). A
  real facing angle, only on the consts `unit_rotation.rotation_is_angle()`
  calls ANGLE (the predicate Rotate uses, never the player id), maps through
  `mirror_tools.facing_image()`, whose 2x2 matrix is `TRANSFORMS`' own
  linear part, so the facing cannot drift from the positions. A gate's
  image takes its orientation sibling through `reorient_gate_const()`, and
  a bridge's, chain's, longhouse's or blocker's through
  `reorient_object_const()` (`descape/object_orientation.py`, TASK-329): the
  other axis's sibling under an axis-swapping element, and for a classic,
  Asian or European bridge's top or bottom piece the end the element's
  image of its run points to, so a turn or flip that reverses the run swaps
  top and bottom (and broken top and broken bottom) on an unchanged span.
  The mole and the DE bridge pieces (`s_bridge_piece_*`, whose ends are
  rotation frames) keep their const and are reported as kept. Every other
  rotation, VARIANT and INERT alike, is copied verbatim; a longhouse's
  index is a roof style its two orientations share.
  The angular modes (3-way, 6-way rotational, 6-way reflective: tile-space
  rotations by multiples of 60 degrees, not D4 elements) extend this with
  three accepted approximations, each derived from the element's own
  linear action rather than hand-typed. A building keeps its const's
  axis-aligned span: its footprint centre is rotated and the same span is
  re-anchored around it on whole tiles, since AoE2 has no rotated
  footprint. Two building images may therefore overlap each other
  (abutting farms, a farm beside a Town Center): the plan checks an image
  against source and kept units only, never against the other images, and
  that overlap is accepted, not repaired. A gate's image takes the
  orientation sibling nearest the
  rotated run direction (always 15 degrees off, never a tie), still via
  `reorient_gate_const()`. A non-square object takes the other axis by the
  wall rule's nearest axis below, and a bridge end the end whose run
  direction is nearer its rotated one (never a tie, since the axis choice
  rules out a perpendicular image), still via `reorient_object_const()`.
  A wall's axis-run index (0 or 1) swaps when the
  element carries the x axis nearer the y axis (60 and 120 degrees and `a`
  do, 180 degrees and the other two reflections do not), under the same
  integer-encoded-file rule; the diagonal runs 3 and 4 stay verbatim in
  the angular modes. A facing angle rotates through
  `facing_image()` with the angular element's own matrix.
- *Cycle Variant*, via `UnitEditModel.set_variant()`, only on consts
  `descape/unit_variant.py` calls cyclable: VARIANT, at least two real
  variants, and not a wall, cliff or gate. Those three are excluded by const
  set (never by `unit.class_`, which would also drop Mole)
  because the game re-derives their index from neighbours, so a written
  value would be overridden. It always writes a literal integer index: every
  corpus placement on a cyclable const stores one, never the radian form
  (one const outside the corpus breaks this, see *What the game rewrites on
  save* below). Under AoE2 it writes the same index to
  `initial_animation_frame` on a .dat type 10 const
  (`unit_variant.frame_follows_variant()`): every type 10 corpus placement
  stores frame equal to rotation, and the 1.59 editor's save sets it so. Every
  other const keeps its stored frame: type 80 stores 0, and types 20 and 30
  store frames unrelated to the variant. The rule is not measured for Return
  of Rome (no editor save, and its corpus was not checked for it), so there
  `set_variant()` and placement keep the stored frame.
  The particle waterfalls (896, 1635, 2598, GH #330) are cyclable by this
  same rule: their variants are their effect's `ImageAngles` runs (3, 3 and
  2), written as literal indices, and a stored literal at or past the run
  count wraps when drawn (an SWFALL at `2` draws run 0). What the game draws
  for that value is unchecked.
  Its callers are Cycle Variant, the Rotate actions, the Rotation field,
  Move's pending turn (GH #280) and the wheel's selected-units choices (GH
  #356), the last two per the *Rotate* bullet above, and a script's
  `editor.set_variant()`, which takes only an `int` inside the const's
  variant count and writes it as its literal float.
- *Wall shapes re-derived from neighbours*: Place Unit's wall-run
  junction rewrites, its rewrites of the walls
  beside a gate it places (GH #159, which also removes the walls under the
  gate's footprint), a gate move's rewrites of the walls beside its old
  and new site when "Moving a gate updates walls" is ticked (GH #194, which
  also removes the walls under the new footprint and fills the line the
  gate left with new walls, `add_many()`ed with the same neighbour-derived
  literal integer index and frame 0; the gate itself only changes
  position, unless a turn pending on the drop (GH #280) also swaps its
  const, when its old tiles are its stored footprint and its new tiles
  the turned one), and a move's rewrites (GH #266: a wall carried onto another
  tile by a drag, group drag, arrow nudge, the Inspector's X or Y, or
  Scatter's move mode takes the shape its new neighbours give it, and a
  wall that stays takes the one its neighbours give it when a tile beside
  it starts or stops holding a connector because a wall moved; a gate's
  own move reshapes walls only under the GH #194 option), and the rewrites
  of the walls beside a gate whose orientation cycle (Rotate, the Rotation
  field) or Replace into a gate const gives it a new footprint (GH #339,
  always on: the walls under the new footprint are removed first, except
  a referenced wall or a Replace row the pre-flight refused, and the tiles
  it leaves are not filled), and a removal's rewrites of the walls left
  beside a removed wall or gate (GH #340: Delete, Cut Units, Find's Delete
  and the Eraser brush; only the walls beside a tile no other wall or gate
  still covers), and Find and Replace's rewrites of the walls beside a tile
  it turns into or out of a wall or gate (the gate target is the GH #339
  clause above), where a unit replaced into a wall from anything else takes
  the shape its new neighbours give it, and a paste's rewrites (Paste Units
  and Paste Region, and the move and turn of a just-pasted region: each
  pasted wall takes the shape its new neighbours give it, and the walls
  beside a tile a pasted wall or gate newly covers re-derive). A wall a
  move, a gate move, a gate's new footprint, a removal, a Replace or a
  paste leaves with no neighbour takes the tower index 2, which all
  24 isolated walls in the integer-encoded corpus store and AoE2:DE draws
  for a lone wall. All of these go via
  `UnitEditModel.set_wall_variant()`, only on the wall-family consts (9 in
  AoE2, 8 in Return of Rome)
  `unit_sprites.rotation_variant_eligible()` accepts (`angle_count == 5`)
  and only for an index in `0..4`. Those are the walls plus Aqueduct
  (231): 8 walls in AoE2, 7 in Return of Rome. The user decided on
  2026-09-26 (GH #110) to treat Aqueduct as a full wall: same 1x1
  footprint and five stored shapes. No AoE2 corpus file has a
  wall beside an Aqueduct, so that junction rests on the decision, not a
  measurement. Allowed for the same reason the bullet
  above forbids *cycling* a wall: the game re-derives a wall's index from
  its neighbours, so writing the value derived from those same neighbours
  converges with the game rather than diverging from it, where an
  author-picked one would just be overwritten. The index is always the
  literal integer, never the radian re-encoding, and
  `initial_animation_frame` is not touched (it is 0 on all 8193 corpus wall
  placements, and on all 1,548 Return of Rome ones). Gates are outside
  the scope and stay there: they have `angle_count == 1` and their
  orientation lives in the const. `set_wall_variant()` has one other
  caller, the *Selection transform* bullet above, under that bullet's own
  rationale; the guard here is the same for both.
- *Cliff drag's join rewrites* (GH #44), via
  `UnitEditModel.set_cliff_frame()`, which raises outside its guard. The
  guard: only the cliff consts (`unit_sprites.cliff_consts()`; 96 in AoE2,
  60 in Return of Rome), always
  the literal integer, a frame inside the const's own frame count (and
  never 24, which no family has), with `rotation` and
  `initial_animation_frame` written together (they are equal on every
  corpus cliff); the const and `x`/`y` are never touched. The trigger:
  only a Cliff tool stroke (a drag, or a click, which is a one-tile
  stroke), only an existing end cap or lone piece (at most one cardinal
  neighbour) that gains exactly one cardinal neighbour from that stroke,
  only when the new piece is shaped for that side with a leg of the same
  type (tall or thin), and only the frame the measured shape table gives
  for its new neighbours at its own size
  (`cliff_connectivity.rewrite_frames()`, read by
  `cliff_chain.ChainStroke.rewrites()`), or, where the family lacks that
  frame, the row's next frame it has, alone (user decision 2026-10-19).
  Where that join would need a
  different-size piece the old cliff is left alone. The erase trigger
  (GH #247, GH #286): only a Cliff tool right-button stroke, an Eraser
  stroke with its Cliffs target ticked, or a Delete, Cut Units or Find
  and Replace Delete that removes a cliff; only a surviving cliff that
  was joined to a cliff that stroke or Delete removed and is left joined
  to exactly one and touching no other surviving cliff (a junction's
  centre is left as it is), "joined" meaning a shared edge with legs of the same
  type where each piece's stored frame is shaped for that side
  (`cliff_chain.shaped_arms()`), and only the frame
  `cliff_connectivity.rewrite_frames()` gives that one arm at its own
  size (`cliff_chain.erase_rewrites()`, through
  `erase_rewrites_for_units()`), hidden survivors included. Else it is
  left alone, and for a stroke none is rewritten if the map's units
  changed during its drag. Allowed for the wall
  bullet's reason: the value is derived from the neighbours the piece now
  has, where an author-picked one is not.
- *Replace* (Edit > Find and Replace, GH #144), via
  `UnitEditModel.replace_type()`, which changes a unit's const (see the
  const exceptions below) and so must decide what its `rotation` and
  `initial_animation_frame` mean under the new const. The two move together
  (they are always equal in the corpus for cliffs and for .dat type 10
  objects such as trees and eye candy, not for other cyclable consts), by
  `unit_model.replaced_rotation()`: verbatim when both consts are ANGLE,
  both are walls (the `rotation_variant_eligible()` consts; the game
  re-derives the index from neighbours), or both are INERT (which keeps the
  `7.0` sentinel); verbatim for two cyclable consts when the old variant
  index exists in the new const (the frame becoming that index when the new
  const is a type 10 one under AoE2, as *Cycle Variant* writes it), else
  `0.0`/`0`; `0.0`/`0` for every other
  pairing (a unit that becomes a wall then takes its neighbour-derived
  shape, and a wall kept a wall re-derives when a tile beside it changes,
  both by the wall bullet above; `replace_type()` itself stays as stated)
  and for any target gate.
- *Place Unit's placement rotation* (GH #172: the hover ghost turned by
  the wheel's rotate binding, Ctrl+wheel by default, or the Rotate actions
  before a click, or set by the Eyedropper's Units-mode pick,
  `ViewerWindow.pick_unit_value()`). The value a new unit
  is `add()`ed with, and the ghost drawn with, comes only from
  `place_rotation.rotation_for()`, and the owner never changes it. An ANGLE
  const gets a facing on its own frame grid, GAIA's included (GH #217: the
  game draws a GAIA creatable's facing). A cyclable const gets a literal
  integer variant index, inside the *Cycle Variant* scope above, and its
  `initial_animation_frame` comes only from `place_rotation.frame_for()`: the
  same index on a type 10 const under AoE2, as *Cycle Variant* writes it, else
  `add()`'s 0. A gate's
  orientation is a different catalog const, picked before the click, so
  this is a new unit and not a `set_unit_const()` case, the same as
  mirroring. Everything else (walls, cliffs, INERT) keeps `add()`'s `0.0`.
  The Unit Brush (GH #170) writes `rotation_for()`'s value too: of the
  pending state with Random facing/variant unticked, else of a state
  `place_rotation.random_rotation()` rolls per unit (a facing on the
  const's own grid for ANGLE, a variant index for a cyclable const, which
  stays inside the *Cycle Variant* scope; nothing for any other const). It
  refuses walls, gates and cliffs outright. The pending state is the same
  one, and is also turned over the brush's own hover ghost (GH #355).
  Unit Line and Unit Shape (GH #365) write it the same way, through the
  brush's own tail (`unit_brush.place_candidate()`, driven by
  `unit_shape.place_cells()`), with the pending state and the Random tick
  latched at the drag's press. A wall pick there is not a placement
  rotation case: it goes through `wall_run.plan_wall_run()` like Place
  Unit's run, and gates and cliffs are refused before the drag starts.
  `add()` itself stays unvalidated, as with mirroring and paste: the guard
  is `rotation_for()`, pinned by `tests/test_place_rotation.py` and, for
  the shape tools, `tests/test_unit_shape.py`. A script's
  `editor.add_unit()` takes its rotation from `rotation_for()` too, from a
  state built of its `facing` or `variant` argument (each refused on a
  const of the other kind), and sends a GAIA-only object to GAIA as Place
  Unit does; `tests/test_scripting.py`. Edit > Scatter Units in Region's
  Random facing/variant box (TASK-510,
  `scatter.plan_scatter_units(random_rotation=True)`, which
  `scatter_units()` forwards) writes `random_rotation()`'s value per copy
  too, inside the same ANGLE and *Cycle Variant* scopes, and nothing but
  `0.0` for any other const; a rolled type 10 variant's frame takes
  `frame_for()`'s value too. `add_many()` is unvalidated, so the guard is
  `random_rotation()`, pinned by `tests/test_place_rotation.py` and
  `tests/test_scatter.py`. With Place Unit's own Random facing/variant
  ticked (TASK-367), the pending state is `place_rotation.roll()`ed for the
  picked const after each placement and on each new pick, and the ghost
  and the click still read it through `rotation_for()`, inside the same
  ANGLE and *Cycle Variant* scopes (`roll()` gives every other const the
  empty state, so `0.0`); `random_rotation()` is `rotation_for()` of the
  same `roll()`. The Unit Brush and Unit Line / Unit Shape read the state
  unrolled; `tests/test_place_rotation.py`, `tests/test_place_preview.py`.
- *Terrain planting's variant* (the Terrain tool's Trees and Eye candy,
  and File > New Map…'s, which `scenario_new` and
  `tools/gen_blank_maps.py` also reach): each new GAIA unit is
  `add_many()`ed with `terrain_units.variant_for()`'s rolled index, the
  literal integer in both `rotation` and `initial_animation_frame`;
  `add_many()` is unvalidated, so the guard is `variant_for()`'s own
  `randrange(angle_count_for())`, a range equal to the const's
  `unit_variant.variant_count_for()` on every const either game plants
  (39 on 2026-10-29). That equality is re-measured per game by
  `tests/test_terrain_units.py`, not trusted.

## Gates

Gates carry no rotation at all: all 24 visible gate consts (6 families × 4
orientations; Return of Rome: 12, 3 families × 4, and its 12 placements
are all `7.0`) have `angle_count == 1`, so there is no second frame for a
rotation to select, and across 300 corpus gate placements the field is only
ever `0.0` or the junk sentinel `7.0`. A gate's orientation lives in its
`unit_const`. The graphic file names are `..._ne_closed_x1` / `_se_` / `_e_`
/ `_n_`, one const per orientation (stone: 64/88/659/667). "Rotating" a gate
therefore means swapping the const among its four siblings, which changes the
footprint. Top-down Flat (GH #151) draws a gate with its next orientation
sibling's art through `unit_sprites.flat_art()`, inside the placed const's
own footprint; that is draw-only, and no write path may use it. The building
art tables may likewise draw a stone gate with its Fortified Wall sibling's
image at a later starting age (`unit_sprites.resolve_entry()`); that is
draw-only too, and orientation, footprint and `set_unit_const()` stay on the
placed const.

## Walls

The same is true of walls, for a different reason, and it is confirmed
in-game: a wall graphic's five stored frames are SHAPES (two diagonal runs, a
tower, a flatter run, a narrow column), not five facings, so `rotation`
selects a variant. A wall cannot be rotated into a tower. Real files encode
that index two ways for the same graphic — a literal `0..4`, or the same
index as `k*2π/5` radians — so both must be read, and neither may have an
angular zero-point offset applied. `unit_sprites.variant_index()` is the only
correct reader; `angle_index()` mis-maps 3 and 4. A write path must pass the
field through verbatim here too, with the exceptions listed above:
`UnitEditModel.set_wall_variant()`, which writes the value derived from
the same neighbours the game would read (or, for a wall a move, a
removal, a Replace or a paste leaves with none, the tower index 2); *Replace*
(`UnitEditModel.replace_type()`, by `unit_model.replaced_rotation()`),
which keeps a wall's index verbatim when the new const is also a wall and
writes `0.0` when a non-wall becomes a wall; the viewer then writes that
wall's neighbour-derived shape through `set_wall_variant()`; and the new wall images of
Map mirroring and a *Region turn* (GH #197), whose index goes through
`mirror_tools.wall_index_image()`, the element's permutation of that same
neighbour table, in an integer-encoded file only. A wall's correct stored value is also a
function of its neighbours (98.9%/99.1% agreement between neighbour mask and
stored index across the corpus), so rotating one would write a value the game
re-derives. Walls are VARIANT, and Rotate skips them in a unit selection; a
Region turn re-pastes them as new walls instead, and Flip and Rotate Group
(the *Selection transform*, TASK-256) carry a selected wall's index through
`wall_index_image()`, written by `set_wall_variant()`. Top-down Flat (GH #151)
draws a wall's shape with the frame `unit_sprites.flat_art()` picks
for the square grid; that is draw-only, and no write path may use it. Every
view draws a wall with at most one neighbour as the tower whatever its stored
index, and every wall of a radian-encoded file by its neighbours
(`unit_sprites.wall_drawn_shape()`); that is draw-only too, and no write path
may use it.

## The two exceptions to "a placed unit's `unit_const` never changes"

`AGENTS.md` states the rule and its two writers. Their exact scope:

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
  orientation siblings `descape/gate_orientation.py` derives for it, and a
  non-square object's (a bridge, chain, longhouse or blocker, TASK-329)
  only to one of its own group's siblings from
  `descape/object_orientation.py`, for the gate rule's reason: the
  orientation lives in the const. Its
  callers are Rotate and the Rotation field (a selected gate cycles), Move
  (GH #280: a turn pending on a drag lands as the sibling, then the drop's
  position), and the *Selection transform* (TASK-256: a turned or flipped
  gate takes `reorient_gate_const()`'s sibling, and a turned or flipped
  object `reorient_object_const()`'s, then its planned position),
  all inside that same sibling-only scope. Rotate and the Rotation field
  still skip a selected bridge. A gate's 1x1 corner pieces (TASK-532) are
  callers within the same scope: every turn above that changes a gate's
  const carries its pieces, each to the matching end tile of the new
  footprint as `gate_orientation.corner_piece()` of the new const, one of
  the piece's own four siblings, through
  `placement_rules.companion_landing()` (Rotate, the Rotation field and a
  drag's turn), or through `unit_images()`' `gate` index (the *Selection
  transform*), then `set_position()` for its point. `companion_landing()`
  raises for a const outside the piece's own siblings
  (`tests/test_placement_rules.py`).
  `mirror_tools.reorient_gate_const()`'s verbatim corner rule applies only
  to a piece turned without its gate. It
  must re-anchor `x`/`y` by preserving the footprint's low corner
  (`span_low_corner()` forward, `render_units.span_anchor()` back), because the
  four orientations have four different spans and every corpus placement
  sits at `tile + span/2` per axis. `rotation` and `z` pass through
  verbatim as above: a gate's stored rotation is `0.0` or the junk sentinel
  `7.0`, and every sibling has `angle_count == 1`, so there is nothing
  there to normalize. A bridge, chain or blocker is INERT the same way
  (every corpus bridge stores `0.0` or `7.0`), and a longhouse's three
  frames are roof styles its two orientations share.
- Map mirroring is not an exception to this rule: it never changes a placed
  gate's or object's const, it `add()`s a *new* one whose const comes from
  `mirror_tools.reorient_gate_const()` or `reorient_object_const()` and
  whose anchor is re-derived from that sibling's own span.
- A pasted region's turn (GH #197) is not an exception either, for the
  same reason: the turned block is re-pasted as *new* units, and a gate's
  const comes from `reorient_gate_const()`, an object's from
  `reorient_object_const()`, through `mirror_tools.unit_image()`. A
  selection or committed region turned in place (TASK-256) does change a
  placed gate's or object's const, through `set_unit_const()` above.

## A pasted unit's `reference_id`

Every new unit's `reference_id` is minted by `UnitEditModel.add()` (or
`add_many()`). The one exception is a same-document paste of a Unit Cut
(Edit > Cut Units, then Paste Units): the cut block remembers each unit's
own id and the document it came from (`RegionUnit.cut_reference_id`,
`RegionBlock.cut_doc_id`, never saved with a block), and
`region_clipboard.reusable_cut_ids()` hands `add(reference_id=)` each one
that no live unit holds and that sits below the mint counter, each id once.
So the first paste after a Cut moves the units, and a later paste, a paste
after the Cut is undone, or a paste into another document mints, as a copy
does. `add()` raises for an id a live unit holds
(`reference_id_in_use()`), a negative one, or one at or past
`next_unit_id`: every later mint is at least the counter, and `restore()`
rewinds the lists and the counter together, so no paste can duplicate an
id. A script's `editor.add_unit()` passes no id and always mints. Pinned by
`tests/test_unit_model.py`, `tests/test_region_clipboard.py` and
`tests/test_unit_clipboard_viewer.py`.

## What the game rewrites on save

Measured by opening the generated test map
(`tests/fixtures/test_map_240x240.aoe2scenario`) in the AoE2:DE 1.59 editor
and saving it (`tests/fixtures/test_map_game_resave.aoe2scenario`,
2026-10-17), paired unit by unit with the map it was saved from. Each
rewrite is one of two kinds: the game's own normalization, which needs
nothing here, or a value DEscape wrote that the game replaces, which is
fixed, refused or explained in the app. A rewrite a later save shows that
is not listed here is a new finding.

- **Objects deleted.** 212 units, 211 of them player-owned: 209 wild
  animals, birds, relics and Others the editor lists for GAIA only, which
  Place Unit and every owner change already give to GAIA alone
  (`unit_kind.owner_refused()`), Ruins 345, now GAIA-only too, and Hidden
  Outlaw 158 (Player 1); plus FLARE3 332, owned by GAIA. 158 and 332 are
  editor-hidden objects, and the rule leaves them as they are.
- **Variant frames.** Every .dat type 10 variant placement got
  `initial_animation_frame` equal to its `rotation`; types 20, 30 and 80 kept
  their frame. DEscape now writes it that way (*Cycle Variant* above). Two
  Horses (1869, an ANGLE const) got frames 49 and 54: normalization.
- **Packed Town Center 444.** Its three rotations (0.0, 8.0, 15.0) came back
  as 4, 0 and 5 sixteenths of a turn in radians, not a function of the value
  written. A value the game replaces. That contradicts *Cycle Variant*'s
  "every corpus placement on a cyclable const stores a literal", which holds
  for the corpus only: 444 has no corpus placement.
- **Lone walls and towers, 0.0 to 2.0.** 9 lone walls (the tower index 2) and
  10 towers. Normalization: corpus towers store both 2.0 and 0.0.
- **Const 768 to 420.** One hidden Blue Tree came back as a Cannon Galleon.
  Normalization of a hidden object, reachable only with Show hidden.
- **New units.** Mole 2421 gains a 2422 and Siege Tower 2443 a hidden 2444,
  and hidden gates 487, 488 and 792 gain 64, 63 and 789, each at its source
  unit's position. The game's own companions, the same for any author.
- **Edge positions.** A unit on the map's edge line is moved to 0.21 tiles
  from the edge, on every edge. Whether DEscape should clamp or warn at
  placement is still open.
- **Garrisoned units** move onto their host. Normalization.
- **z.** Every unit's `z` is recalculated from the terrain and stacking,
  typed values included. The Inspector's Z caption and the guide say so; the
  field stays editable, since the value is the file's.
- **Terrain.** Ten editor-hidden and foundation terrains (33, 34, 36, 38,
  39, 43, 44, 61, 62, 103) become a listed terrain plus a layer, and a
  farm-type building paints 7 or 63 under itself. No corpus map uses the ten.
- **Messages.** Hints and scouts text with a string id set is replaced by
  the string table's text. The Messages panel already says the game shows
  the language-file string.
