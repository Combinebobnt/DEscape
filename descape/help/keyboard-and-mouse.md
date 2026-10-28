# Keyboard and Mouse

In DEscape this page shows your current keys and mouse wheel settings next, read from **Settings > Keybinds** each time you open it, so it stays right after you rebind something. Below them are the gestures you cannot rebind, grouped by where they work. Everything else in the guide names a command rather than its key, and you find the key here.

## Anywhere on the map

- Middle-drag pans, whatever the mode or tool, and also while you pick something on the map.
- With the Pan tool, a left drag pans too.
- The mouse wheel's actions, alone or with Ctrl, Shift or Alt held, are set in **Settings > Keybinds** and listed under Your current keys.
- A double-click counts as two clicks, so on a stack of units it steps down twice.
- Keys reach the map only while it has keyboard focus. If one seems to do nothing, click the map first.
- Esc cancels whatever is in progress, nearest first: a pick on the map, an area drag, a line, rectangle or wall drag, a ruler measurement, a unit drag or selection box, a region move or a region drag. Next, a live flip or turn of a selection or region goes back to where it started; Enter keeps it instead. With nothing in progress, Esc clears the unit selection under Units mode's Select tool, and otherwise clears the selected region.
- Dropping a `.aoe2scenario` file onto the window opens it.

## Picking on the map

These apply while a trigger field's Set button or the Garrison list's Pick from map is waiting for the map.

- A location takes one click. An area takes a drag, and is set when you let go.
- Picking units: click a unit, and click again to walk down a stack.
- Right-click or Esc cancels the pick, or finishes Pick from map.
- With Edit Area ticked under Pan in Triggers mode, drag the middle of the shown area to move it, or an edge or corner to resize it. Right-click or Esc during the drag cancels it.
- With the Create Objects tool, a left click stamps; a right click does nothing.

## Ruler

- Drag, or click, move and click again, to measure between two tiles. The result stays pinned on the map.
- Right-click cancels a measurement in progress, or removes the pinned ruler whose end is under the cursor. Clear rulers removes them all.

## Terrain mode

- Elevate: right-drag or Shift+drag lowers instead of raising.
- Draw, Set Elevation and the Eraser: right-drag does the same as a left drag.
- Cliff: right-click or right-drag erases cliffs.
- Draw Line: hold Shift to snap to one of 16 directions. Draw Rectangle: hold Shift for a square. Both follow Shift as you press or release it during the drag.
- Right-click cancels a line or rectangle drag.
- Select: drag to select a region. Dragging inside a region you have just pasted moves it.
- Paint Can and the Eyedropper ignore the right button.

## Units mode

- Select: a click selects a unit, and a repeat click walks down a stack. Ctrl+click toggles a unit, Shift+click adds one, and a plain click on empty ground clears the selection. A drag draws a box, even from on top of a unit; Ctrl+drag toggles what it covers, Shift+drag adds it.
- Move: drag a unit, or a selected unit to move the whole selection. Dropping on a building that can hold them garrisons them; hold Shift to keep it a move.
- Hold Alt as you click with Place Unit, drop with Move, or start a Unit Brush drag, for one off-grid placement.
- Arrow keys nudge the selection by a tenth of a tile, Shift+arrow by a whole tile. With nothing selected, an arrow key does whatever the Pan commands bind it to.
- Delete deletes the selection, without asking.
- Place Unit with a wall picked: drag to lay a run, and hold Shift for one straight segment in one of 8 directions. Wall Rectangle and Wall Circle: hold Shift for a square or a circle. Right-click cancels any of these drags.
- Eyedropper: a click copies a unit's type, owner and facing or variant; Shift+click copies the type only; a repeat click walks down a stack.
- Convert and the Eraser: right-drag does the same as a left drag. The Unit Brush paints with the left button only.
- Pan: the first click on a unit shows a hint naming the tools that select and move units.

## Triggers panel

- The ▲ and ▼ buttons: a click moves the selection one place, Shift+click five places, and Ctrl+click to the top or bottom. Ctrl wins when both are held.
- Right-click the trigger list for Cut, Copy, Paste, the four Move commands, Select Section and Select Tag.
- Right-click the conditions and effects for Cut, Copy, Paste, the four Move commands and Select Same Type.
- Double-click a variable in the Trigger Variables window to rename it. Double-click in the type picker to accept a type.
- Neither list reorders by dragging, and Delete does nothing in them.

## Other panels and windows

- Units catalog: double-click an object, or press Enter, to switch to Place Unit with it.
- Garrison list: double-click a row to select that unit (with **Filters > Show Garrisoned Units** on).
- Facing dial: click or drag to set the facing, scroll to step it.
- X, Y and Z with several units selected: the arrows, the arrow keys or the wheel move every selected unit by one step.
- In the Units Inspector, the wheel over an Owner, Unit State or Capture Flag dropdown scrolls the Inspector until you click the dropdown. The mouse wheel dropdowns in **Settings > Keybinds** ignore the wheel.
- Text boxes in the trigger form and Messages mode: drag the grip on the bottom edge to change the height in whole lines; double-click the grip to reset it.
- Filters menu: a click or Space toggles an entry, and the menu stays open.
- Find and Replace results: Space ticks or unticks the highlighted rows, and right-click offers Tick selected and Untick selected. Double-click a row to centre the map on that object (Objects tab) or open the trigger (Triggers tab). Click a type chip to remove it.
- Map Analysis, History and Clipboard History: double-click a row to go to that finding, jump to that point in the history, or make that copy the one Paste uses.
- Disabled Objects: Enter or a double-click moves a row to the other list.
- View mode's stats table: Ctrl+C (Cmd+C on a Mac) copies the selected rows as tab-separated text.
