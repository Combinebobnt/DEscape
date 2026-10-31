# Map Tools

These commands work on the whole map, or on many objects at once, from any mode. They live in the Map, Edit and Tools menus. Each one that changes the map is one undo step, except Resize Map, which cannot be undone.

## Resize Map

**Map > Resize Map…** grows or shrinks the open map, keeping any of nine anchors in place: a corner, the middle of an edge, or the centre. Terrain, elevation, objects, and trigger areas and locations move with the map. New ground is plain grass at the height of the old edge beside it, so no cliff-like jump appears. It opens on the anchor you last chose with Resize, until you quit DEscape.

The dialog's preview shows how the new map will sit over the old one, and follows the current view: a diamond in an isometric view, a square in top-down Flat. The solid outline is the new map, the dashed one the old map, and the small ring marks the anchor. Pale ground is new. The dark area is cut off, along with everything on it. Red points are player objects that will be deleted, and anything the new edge cuts through. Trees and other GAIA objects in the dark area get no mark of their own, so the dialog's count can be larger than the number of red points.

Objects that end up off the map, or that the new edge would cut through, are deleted together with anything garrisoned in them. The dialog says how many before you commit, and how many trigger references would then point at a deleted object. Trigger areas are shifted only in files whose triggers DEscape can write back; otherwise the dialog warns that they are left as they were.

A resize cannot be undone. Your unsaved edits are carried into the resized map, still unsaved, but the undo history is cleared. Only square maps for now. A script can do the same through the [batch editing API](https://github.com/Combinebobnt/DEscape#scriptable-batch-edits): `batch_api.resize_map(scenario, 168, 168, "CENTER")`.

In the in-game editor, this is the Map tab's Extend Map.

## Mirror Map

**Map > Mirror Map…** copies one part of the map onto the rest by reflection or rotation, for a fair layout. Pick a Symmetry (two-way mirrors, a 180° turn, four-way and eight-way, and approximate three-way and six-way modes for three or six players) and the Source slice, which is shaded on the map. Tick what to copy: Terrain, Elevation and Units.

With Units ticked, every other slice's units are rewritten from the source slice. Units with a real facing turn to match, a wall's run turns with the mirror, and a gate becomes the orientation it is mirrored into. Overlapping copies chooses what happens where a copy would land on an original: refuse the whole mirror, skip those copies, or place them anyway. Grass, plants, flowers and other objects units can walk through never count as overlapping: their copies land anywhere, and other copies may land on them. Rotate ownership by moves each slice's copies that many places further along the scenario's player list than the slice before; GAIA's units stay GAIA's.

Preview applies the mirror and leaves the dialog open, so you can look at the result. Changing an option or pressing Cancel takes the preview back, and Apply keeps it. Until you quit DEscape, the dialog opens with the options you last applied; Cancel leaves them as they were. Mirror Map needs a square map. In the in-game editor, the closest thing is the Terrain tab's Map Copy.

## Disabled Objects

**Map > Disabled Objects…** chooses which buildings, units and technologies each player cannot use. Pick a player, then move entries between the Available and Disabled lists on the Buildings, Units and Techs tabs. OK applies every player's changes as one undo step. The Players panel has the same dialog. In the in-game editor, this is the Options tab's Disable Objects.

## Swap Players

**Map > Swap Players…** makes two players exchange slots: their units, settings and disabled objects, and optionally their trigger references and diplomacy, as one undo step. The Players panel has the same dialog; [Players](players.md) covers what moves.

## Map Analysis

**Tools > Map Analysis…** checks the open map for common problems before you ship it, and changes nothing. It covers illegal elevation steps, off-map and off-grid objects, objects on terrain they can't stand on or inside each other, GAIA-only objects owned by a player, dangling trigger, unit and garrison references, trigger display order, players with no units, player units stranded on small land areas, missing scenario instructions and a map that is not square.

Its Train and research buttons check warns when a Change Train Location, Add Train Location or Change Research Location effect's button won't show: a unit the civ can't have, a tech the player's starting age already researched, one hidden by a lower-id unit, a covered tech, or a slot with no button.

Each finding is an Error, a Warning or Info, and a check that could not run says why. Findings are marked on the map while the dialog is open. Double-click one to jump to its tile or unit, and press Refresh to check again after a fix.

## Find and Replace

**Edit > Find and Replace…** searches every placed object, including ones hidden by the Filters menu, garrisoned or off the map. Its Objects tab searches by name (substring or regex, `#id` for an exact type, optionally captions too), owner, category, kind, specific types and area.

Tick the results to select them on the map, delete them, change their owner, replace them with another object type or export them as CSV. A replaced object keeps its id, so triggers and garrisons that name it still work, and Replace offers to retarget the trigger effects that select it by type, in the same undo step. Double-click a result to centre the map on it. Each operation is one undo step.

The Triggers tab searches and edits trigger text and types; see [Triggers](triggers.md).

## Scatter Units in Region

**Edit > Scatter Units in Region…** places random copies of the object picked in the Units panel, for the owner you choose. The area is a region drawn with the Select tool in Terrain mode, or the bounding box of the units selected in Units mode, grown by a few tiles. Restrict it to any tile, water only or one terrain, ask for a count or a density, and set the seed, jitter and minimum spacing. The live eligible-tile count is exactly what gets placed. GAIA-only objects always go to GAIA. The dialog opens with your last scatter's settings, except the area and the terrain, until you quit DEscape.

Its other mode, Move the selected units, moves each selected unit to a random tile of the area instead, keeping every other field.

## History and Clipboard History

**Edit > History…** lists every edit you can undo, from Opened file down. The current one is bold, undone ones are grey and a tick marks the state last saved. Double-click an entry, or press Jump Here, to undo or redo up to it.

**Edit > Clipboard History…** keeps the last regions or units you copied this session. Set Active (or a double-click) picks the one Paste uses. Paste lands on the tile under the mouse, so click the map, hover where it should go, then paste.
