# Triggers

Triggers mode edits the scenario's triggers: their conditions, their effects and the order they run in. Pick **Triggers** in the toolbar's Mode list. The left pane shows the trigger list on top and the selected trigger's conditions and effects below it, and the map stays on the right.

The line above the list counts the triggers and says which order they run in, a setting in [Map Options](map-options.md). The first column is each trigger's place in display order. It is headed **Exec #** when that is also the order they run in, and **Display #** when the file runs them by trigger ID or doesn't store which order it uses.

## The trigger list

Type in **Filter triggers…** to show only the triggers whose names match. The sort box beside it switches between **Display order**, the order the in-game editor shows, and **File order (trigger ID)**. A disabled trigger shows "(disabled)" after its name.

Click a trigger to edit it. Ctrl+click and Shift+click select several; the lower pane then shows only a count, and the buttons act on all of them. **Edit > Select All** selects every trigger the filter shows.

**Sections.** A trigger whose name starts and ends with a run of dashes or similar characters, such as `--- Setup ---`, is a section header. In Display order the triggers below it, up to the next header, sit under it and collapse with it. The **Sections** button adds a section header at the end of the current trigger's section, renames the selected one, moves the selected triggers into another section, collapses or expands every section and jumps to any of them. A collapsed header stands for its whole section: Copy, Cut, Duplicate and Delete take the section with it.

**Tags.** A name that starts with a bracketed tag, such as `[P1] Spawn wave`, carries that tag, and `[P1][Wave] Spawn` carries both; `(tag)` and `<tag>` work too. Pick a tag in the tag box (it starts on **All tags**) to show only the triggers that carry it. The **Tags** button adds a tag to the selected triggers or removes one from them, and renames or removes the chosen tag on every trigger that carries it, hidden ones included. Each is one undo step.

## Adding, moving and copying triggers

The buttons under the list:

- **New** adds a trigger at the end of the list. **Delete** removes the selected triggers.
- **Copy** puts the selected triggers on the clipboard, and **Paste** puts them below the current trigger. Links between the copied triggers point at the pasted copies.
- **Duplicate** copies the selected triggers in place. Their trigger links still point at the originals; use Copy and Paste to re-link a block.
- ▲ and ▼ move the selected triggers one place in display order. Shift+click moves them five places, and Ctrl+click to the top or bottom of their section (of the list, if it has no sections). ⇈ and ⇊ move the selected trigger's whole section past the next one.

Moving needs Display order and an empty filter and tag; a greyed button's tooltip says why. You can't drag rows to reorder them.

In Triggers mode the Edit menu reads **Edit > Cut Triggers**, **Edit > Copy Triggers** and **Edit > Paste Triggers**, unless a text field or the condition and effect list has focus. In a text field its plain **Edit > Cut**, **Edit > Copy** and **Edit > Paste** act on the text. A cut trigger pasted back keeps its name and the links other triggers held to it. Copied triggers also paste into another scenario of the same version: copy, open the other file and paste. Links to triggers outside the block and placed-unit references are cleared then, since they name the first file's triggers and units.

Right-click a trigger for Cut, Copy, Paste below it, the four moves, and **Select Section** or **Select Tag**. The Move Trigger/Condition/Effect commands make the same moves from the keyboard; [Keyboard and Mouse](keyboard-and-mouse.md) lists their keys.

## Conditions and effects

The lower pane lists a **Trigger** row, which holds the trigger's own fields (name, descriptions, enabled, looping and the objective settings), then its Conditions and Effects in display order.

- **New** opens a list of every condition and effect type. Filter it, then double-click a type or press **Add**. With Conditions or Effects highlighted first, the other group starts collapsed.
- **Type…** changes the selected condition or effect to another type. Fields the new type lacks are dropped, and the status log under the map names them.
- **Duplicate**, **Delete**, ▲ and ▼ act on the selected ones, with the same Shift and Ctrl clicks as the trigger list.
- Right-click for Cut, Copy, Paste, the moves and **Select Same Type**. Conditions and effects have their own clipboard and paste into any trigger. With this list focused, the Edit menu reads **Edit > Cut Conditions/Effects**, **Edit > Copy Conditions/Effects** and **Edit > Paste Conditions/Effects**.

Select a row to edit its fields underneath. Select several and the form shows only the fields they share; one they disagree on reads "(differs)" and changes nothing until you set it. Each change is one undo step.

Unit, building and technology fields take a typed name, or their "…" button browses the list. An unset number reads "(unset)". Descriptions, messages and scripts get a full-width box that applies your text when you leave it; while it has focus, **Edit > Undo** and **Edit > Redo** act on its text. Drag the strip along a box's bottom edge to change its height in whole lines, or double-click the strip to reset it. Each field remembers its height.

Some types show extra help under their fields:

- **Display Instructions** shows a mock game screen with the box at its position, the portrait, the coloured text and the display time. Sounds are named, not played.
- **Create Object** and **Replace Object** show roughly which way the unit will face. That mapping isn't confirmed in the game yet.
- **Change Train Location**, **Add Train Location** and **Change Research Location** show the building's command buttons under the button slot field, as they will look after the trigger fires; click a cell to pick that slot. The grid shows only what the effect player's civilization can have (all civilizations when it is random). It frames the effect's own button in green, or in red when it won't show or covers a technology; hover a cell for the details. Tick **Show all triggers' changes** to apply that player's other such effects too. The [Map Analysis](map-tools.md) "Train and research buttons" check warns when such a button won't show: a unit the civilization can't have, a technology the player's starting age already researched, one hidden by a lower-id unit, a covered technology, or a slot with no button.

## Picking on the map

Location, area and object fields have **Set**, **Go to** and **Reset** buttons. For a location, press **Set** and click a tile; for an area, press **Set** and drag a rectangle. A single unit field has a **Pick** button: press it and click the unit. For a list of objects, press **Set** and click units to add or remove them. A repeat click on a stack picks the next unit down, and units hidden by Filters can't be picked. Right-click or Esc cancels, or finishes a list. **Go to** centres the map on the field, even on a read-only file, and **Reset** clears it.

To move or resize an area on the map, keep **View > Show Trigger Overlay** on (the default), pick the Pan tool and tick **Edit Area** on the Tool Options row. Select one condition or effect with an area, then drag the area's middle to move it, or an edge or corner to resize it. **Move** and **Resize** choose which parts you can grab, and **Tile borders** lets you resize from the whole outer ring of tiles. Each drag is one undo step; right-click or Esc cancels it.

With a Create Object effect selected, the **Create Objects** tool on the toolbar adds one copy of that effect per tile under the brush, skipping tiles that already have one. Each click is one undo step.

## Status colours

A trigger, condition or effect shows green when every field its type needs is set and every unit and trigger it names exists, and red when something is missing. Hover a red row to see what. Green means complete, not that the trigger does what you meant in the game: nothing here runs it. A trigger with no effects is red unless it is shown as an objective, an objective header or on screen. A condition or effect of a type DEscape doesn't know is left unmarked. A section row is judged by its own conditions and effects, not the triggers in its section, and isn't coloured unless you ask.

**Settings > Appearance > Trigger status** switches the colours to tick and cross icons, or both, or off. It can stop marking rows that pass so only problems stand out, colour section rows, and change the two colours. The icons sit in their own column between ID and Trigger, so the names line up; a section with no conditions or effects shows a dash there.

## The overlay and variables

**View > Show Trigger Overlay** draws the selected trigger's areas and locations on the map, with a line from an area to its destination, and keeps them there when you switch modes. The selected condition or effect is drawn strongest. **View > Range Rings > Trigger Changes** adds a dashed ring for each trigger that changes a selected building's range; see [View and Filters](view-and-filters.md).

**Variables…** under the panel lists the scenario's trigger variables. Add one by name, or rename (also by double-clicking) or remove the selected one.

## Script Call and XS

A Script Call effect or condition holds XS script in a monospace box with no line wrapping. DEscape sets no limit on its length, so a script can be longer than the in-game editor accepts. The [XS scripting notes](https://github.com/Combinebobnt/DEscape/blob/main/docs/XS_SCRIPTING.md) explain how XS reaches the game and the rules a script must follow. A scenario-wide script file is named in Map Options instead.

## Find and Replace

**Edit > Find and Replace…** has a Triggers tab. It searches trigger names, descriptions, messages, identifiers and XS (as plain text or a regex), the unit types conditions and effects name, and which triggers use given placed objects. It replaces text or types, deletes, enables or disables the ticked matches, or exports them as CSV; each operation is one undo step. Double-click a match to jump to it. On the Objects tab, **Show trigger usages** lists the triggers that name the ticked objects, and replacing an object's type offers to retarget the effects that select it by type, in the same undo step. [Map Tools](map-tools.md) covers the Objects tab.

## Further reading

The AoE2 Scenario Editor Guide describes each [effect](https://ugc.aoe2.rocks/scenarios/triggers/effects/effects/) and [condition](https://ugc.aoe2.rocks/scenarios/triggers/conditions/conditions/), and the [XS language](https://ugc.aoe2.rocks/general/xs/) with its [functions](https://ugc.aoe2.rocks/general/xs/functions/functions/).

Where this is in the in-game editor: the Triggers tab.

DEscape doesn't run triggers or test-play the scenario, so test a trigger in the game.
