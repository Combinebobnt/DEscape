# Units

Units mode places, selects, moves and edits objects: units, buildings, trees, resources and scenery. The left panel holds the object catalog and Owner, with the Inspector below. Each tool's options appear on the Tool Options row under the toolbar.

Where this is in the in-game editor: the Units tab. Convert matches its Converting, and the Eraser the Terrain tab's Erase.

## Picking an object

The Units, Buildings, Heroes and Other buttons show one category of the catalog. The class dropdown narrows it, the filter box finds an object by name or Unit ID, and Icons, Names and Grid change the view. Tick "Show editor-hidden objects" for objects the game's editor hides. "Villager (Random)" places a male or female villager at random. Double-click an object, or press Enter on it, to switch to Place Unit.

Owner sets who owns what you place. Objects the game's editor lists only for GAIA (wild animals, resources, trees, rocks, relics and more) are hidden while Owner is a player, and always go to GAIA; **Settings > Advanced** can keep them listed. While nothing is selected, the Inspector shows the picked object's portrait, Unit ID and base stats.

## Placing objects

**Place Unit** puts the picked object down one click at a time, for the chosen owner, snapped to the grid and centred on the cursor. A preview follows the cursor. Each click is one undo step, and the new unit becomes the selection.

- The game's placement rules apply: the object needs terrain it can stand on and room clear of other objects. A refused click places nothing and the status log under the map says why. Tick Free placement to place it anyway.
- The preview shows a refusal before you click: where the game would refuse the object, a sprite preview is tinted red, and a coloured mark (in Flat, or for an object with no sprite) is filled red and cross-hatched. It updates without a mouse move when you tick Free placement or place a unit under it.
- Tick Off-grid, or hold Alt for one click, to place exactly under the cursor.
- To turn the object before you click, hover the map and use the Rotate commands' keys or the mouse wheel's rotate binding ([Keyboard and Mouse](keyboard-and-mouse.md) shows both). Units turn, trees and scenery step through their variants, gates switch orientation, and later clicks keep that facing or variant.

**Unit Brush** paints the picked object across the brush in one drag, as one undo step. Density is the chance each tile gets one; every tile is rolled once per drag, so going over it again adds nothing, and a drag never stacks objects on itself. The same placement rules apply, and Free placement lifts them. With Random facing/variant ticked, each object gets a random facing, or a random variant for trees and scenery; unticked, each gets the one last set in Place Unit or by the Eyedropper. Off-grid, or Alt held as the drag starts, sets each object a little off its tile's centre. The Unit Brush refuses walls, gates and cliffs, which have their own tools.

### Walls and gates

With a wall picked, Place Unit drags a run that bends once; Shift makes it straight, and right-click or Esc cancels. **Wall Rectangle** and **Wall Circle** drag a box and lay walls around its edge or around the ellipse inside it; Shift makes a square or a circle. The Wall button on the Tool Options row picks the last wall you used, and its arrow lists every wall.

New walls join and reshape the walls they touch, always snap to tiles, and ignore Off-grid and Free placement. "Walls skip occupied tiles" leaves a gap at trees, resources, buildings, units, obstacles and invisible blockers (passable eye candy never blocks), and a wall or gate is never doubled. A gate placed on a wall line replaces the walls under it and reshapes those beside it. A moved wall takes the shape of its new place, and the walls beside where it left and where it landed reshape too; a moved wall left standing alone becomes a tower.

## Selecting and moving

With **Select**, click a unit to select it; click the same spot again to walk down a stack. Ctrl+click toggles a unit, Shift+click adds one, and a plain click on empty ground clears the selection. Drag to draw a box, even starting on a unit: Ctrl+drag toggles what it covers and Shift+drag adds it. **Edit > Select All** selects every unit the filters show, **Edit > Deselect** or Esc clears the selection, and **Edit > Select Whole Stack** adds every unit stacked with a selected one. The selection stays when you switch to Terrain mode.

With **Move**, drag a unit to move it, or a selected unit to move the whole selection. On a stack, the drag takes its selected unit, so pick a lower one with Select first. Placement rules, Free placement, Off-grid and Alt work as for Place Unit, and Esc cancels the drag. The dragged preview turns red where the drop would be refused; in a group drag only the units that can't go there turn red, though the whole group stays put on release. A drop onto a building that can garrison the units never shows red, and one that is full or can't hold them always does. Drop units on a building that can hold them to garrison them; Shift keeps it a plain move. Whatever a unit garrisons moves with it. With "Moving a gate updates walls" ticked, a moved gate clears the walls at its new place and fills the gap it leaves with wall.

Under any tool, with the map focused, the arrow keys nudge the selection by a tenth of a tile, or a whole tile with Shift, and Delete removes it without asking (**Edit > Undo** brings it back). Deleting a building also deletes what it garrisons. A nudge and a paste skip the placement rules.

## Turning, variants and copies

**Edit > Rotate Selection** and the two Rotate buttons on the toolbar turn each selected unit about its own centre. Gates cycle through their four orientations, trees and scenery step through their graphic variants, and walls, cliffs and single-frame objects are skipped. While Place Unit's preview is on the map, the Rotate keys turn the preview instead. **Edit > Cycle Variant** picks the previous, next or a random variant for trees, plants, rocks and scenery.

To flip or turn several units as one group, select them and use **Edit > Flip Selection > Flip Horizontal** or **Flip Vertical**, or **Edit > Rotate Selection > Rotate Group ↻ (Quarter Turn)** or **Rotate Group ↺ (Quarter Turn)**. The group mirrors or turns about the centre of the box around the selected units, and whatever they garrison goes along. Each unit stays the same unit with the same owner, so triggers that name it still find it. Facings turn with the group, a gate takes the orientation that matches its new direction, and a wall takes the shape of its new run; cliffs and objects with no turned shape keep theirs, and the status log under the map counts them. Flip and turn as often as you like: the whole run is one undo step. Press Enter to keep the result, or Esc to put everything back as it was before the first flip; a click elsewhere or any other edit keeps it too. If a unit would land on something in the way, nothing changes and the status log says why; tick Free placement to allow it. Nothing may land off the map.

With units selected, **Edit > Copy Units** and **Edit > Cut Units** take them and whatever they garrison. **Edit > Paste Units** centres them on the tile under the cursor as one undo step, and selects them. Copies are kept in **Edit > Clipboard History…**; for regions of terrain, see [Terrain](terrain.md).

## The Inspector

For one selected unit the Inspector shows its portrait, name, Unit ID and stats, then Owner, X, Y, Z, Rotation, Capture Flag, Unit State, Reference ID and Garrisoned in. Each change is one undo step.

Rotation is a facing number, a variant number for trees and scenery, or a gate's orientation, 0 to 3. Walls and cliffs are read-only, because the game works out their shape. Click, drag or scroll the dial beside it to turn the unit. Capture Flag needs a scenario saved by the September 2026 game update (version 1.59).

With several units selected, a field shows their shared value or "(mixed)". A typed value applies to all of them, and the X, Y and Z arrows move every one by a step, keeping the formation.

## Garrisons

A unit that can hold others shows a Garrison list. Its units are hidden on the map unless **Filters > Show Garrisoned Units** is on; then double-click a row to select one. Add... creates an object inside, and Pick from map puts the units you click inside until Esc or a right-click. Cut, Copy and Paste move or copy rows between hosts; Cut keeps their IDs, so triggers that name them still work. Unload puts rows on the tile beside the host, Delete removes them, and Owner... gives them a new owner.

## Other tools

The Tool Options row's settings here (Free placement, Off-grid, the wall the Wall button picks, the Unit Brush's Density and Random facing/variant, Convert's Sources and the Eraser's Targets, among others) keep what you set until you close DEscape. **Settings > Tools** sets what they start at.

- **Convert**: drag over the units of the players ticked under Sources to give them to the Owner. GAIA-only objects stay with GAIA.
- **Eraser** works as in Terrain mode; see [Terrain](terrain.md).
- **Eyedropper**: click a unit to make its type, owner and facing or variant the next placement. Shift+click takes the type only.
- **Pan** and **Ruler** work in every mode; [Keyboard and Mouse](keyboard-and-mouse.md) lists every gesture.
