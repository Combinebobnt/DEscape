# Terrain

Terrain mode paints the ground: terrain types, elevation, cliffs and the layered terrain drawn over a tile. Pick a tool on the toolbar, choose a terrain in the panel on the left, and set the tool's options on the Tool Options row under the toolbar. Each drag or click is one undo step.

Where this is in the in-game editor: the Terrain tab (painting, elevation, cliffs, map copy and erase). The map's water look and shore waves are set in [Map Options](map-options.md) here.

## Choosing a terrain

The panel on the left lists every terrain by category, each with a swatch of its ground texture. Type in the filter box to find one by name or number. The List and Grid buttons switch between a list and a grid of swatches; **Settings > Appearance** sets the grid's swatch size. In the grid, click a category heading to collapse it or open it again. Tick "Show unused / moddable terrains" to list the unused, moddable and obsolete terrain slots too. Forest and snow variants share a ground texture, so their swatches look alike. The line at the bottom of the panel describes the tile under the cursor.

## The tools

- **Draw** paints the picked terrain under the brush. Drag to paint a trail.
- **Paint Can** fills the connected area of matching terrain with one click. Clicking an area that already has the picked terrain clears its layered terrain.
- **Draw Line** paints a straight line as you drag. Hold Shift to snap it to one of 16 directions.
- **Draw Rectangle** paints a rectangle, Filled or Outline. Hold Shift to make it a square.
- **Elevate** raises the ground under the brush by one level. Right-drag, or Shift+drag, lowers it.
- **Set Elevation** sets the ground under the brush to the Level on the Tool Options row, 0 to 15.
- **Cliff**, **Eraser** and **Select** are described below.
- **Eyedropper**: click a tile to load its terrain into the panel and its height into Level. With Target on Layer it picks the tile's layered terrain.

Pan and Ruler work in every mode; see [Keyboard and Mouse](keyboard-and-mouse.md).

Draw Line and Draw Rectangle show the shape while you drag and paint it when you let go. Right-click or Esc cancels the drag.

When you change a tile's height, the tiles around it rise or fall with it, so no step between neighbours is more than one level. Raising with an outline brush fills holes up to two tiles wide. Elevate, Set Elevation and pasting elevation work on square maps only.

## Tool options

The controls on the Tool Options row keep what you set until you close DEscape, Trees and Eye candy included. **Settings > Tools** sets what they start at ([Settings](settings.md)).

- **Target** (Draw, Paint Can, Draw Line, Draw Rectangle) picks which terrain a tool writes. Terrain paints the base terrain and clears any layered terrain on the tile. Layer paints the layered terrain drawn over the base, like the in-game Layering Mode. Clear layer removes it. Target is hidden for a file that stores no layered terrain.
- **Auto beach** (Draw, Draw Line, Draw Rectangle) lays a beach shoreline between water and the terrain your stroke paints. Pick the beach terrain (Auto matches the land's climate) and its width, 1 to 3 tiles. Painting a beach, bridge or non-navigable shore lays none.
- **Trees** and **Eye candy** (with Target on Terrain) place the matching GAIA trees on forest terrain, and the grass tufts and underbrush a terrain carries, like the in-game Eye Candy option. Painting another terrain over the tile removes them.

### The brush

Draw, Elevate, Set Elevation, Draw Line, the Eraser and Draw Rectangle's Outline paint through a brush. Size runs from 1 to 9 tiles. Shape is Square, Circle, Rounded square, Square outline, Circle outline, Plus, or one of two lines along the tile axes (diagonal in the isometric views). An outline shape adds a Thickness.

The mouse wheel's brush size binding and the Increase Tool Value and Decrease Tool Value commands step the size. Where a tool has no brush, those two commands step Level instead. [Keyboard and Mouse](keyboard-and-mouse.md) shows their current keys.

## Cliffs

The Cliff tool's options pick a Cliff family, a Piece and a Frame, with a preview beside them. Hovering the map shows what a click would place.

- Click to place one piece, centred on the tile.
- Drag to lay a connected run in the picked family. The drag direction sets which way the cliff face points. Start the drag on an existing cliff to continue it as one wall; the piece it joins is reshaped to fit where the family allows.
- Right-click or right-drag erases cliffs. A run left with one neighbour gets its end cap back.

Cliffs belong to GAIA, so unticking **Filters > Show GAIA** hides them.

## Eraser

Drag to remove the objects under the brush that match a ticked entry in the Targets menu: Units, Buildings, Trees, Terrain Overlay (the layered terrain), Invisible Objects and Gaia Objects. Everything ticks those six. Cliffs is a separate tick, because erasing one piece breaks its chain. Objects that Filters hide are never erased, and erasing a building also erases what it garrisons. The Eraser works the same way in Units mode.

## Regions: copy, paste and turn

With Select, drag a rectangle to select a region. **Edit > Select All** selects the whole map, and **Edit > Deselect** or Esc clears the region.

**Edit > Copy Region** copies the region's terrain, elevation and units. **Edit > Paste Region** pastes them at the tile under the cursor, as one undo step. While Select is the tool, the Terrain, Elevation and Units ticks on the Tool Options row choose what a paste writes.

A region you have just pasted stays live until your next edit:

- Drag inside it with Select to move it.
- **Edit > Rotate Selection** turns it a quarter turn, and so does the mouse wheel's rotate binding.
- **Edit > Flip Selection** mirrors it left/right or top/bottom on screen.

Walls, gates and unit facings turn with the block, and the paste plus every move and turn of it stay one undo step.

A selected region that you did not just paste is flipped or turned in place: **Edit > Flip Selection** mirrors it, and **Edit > Rotate Selection** (any of its turn entries) turns it a quarter turn, as long as no units are selected in Units mode. Its terrain, its elevation and the units standing on it move together, and the units stay the same units, so triggers that name them still work. A region that is not square turns about its own centre: the tiles it no longer covers keep what they had, and the selection follows the turned rectangle. A run of flips and turns is one undo step; press Enter to keep it, or Esc to put the region back as it was. The turn is refused if part of the region would leave the map, on a map that is not square, or where the file's terrain can't be edited. The mouse wheel's rotate binding still turns only a just-pasted region.

Copied regions are kept in **Edit > Clipboard History…**, and **Edit > Scatter Units in Region…** fills a region with objects; both are in [Map Tools](map-tools.md).

## Not in DEscape yet

- A paste keeps every unit's owner. The in-game Map Copy can hand pasted units to another player; here, use Convert in [Units](units.md) mode afterwards.
- An existing cliff keeps its frame, since Rotate skips cliffs. Pick the Frame before you place the piece.
- Elevation edits need a square map.
