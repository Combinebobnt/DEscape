# Triggers

Triggers mode edits the scenario's triggers: their conditions, their effects and the order they run in. Pick **Triggers** in the toolbar's Mode list. On a very large file the first visit shows a progress window while the triggers are read, and **Cancel** returns to the previous mode, or to View if you were opening a file. The left pane shows the trigger list on top and the selected trigger's conditions and effects below it. The selected row's fields fill the **Trigger Details** panel under it, and the map stays on the right.

Trigger Details shows only in Triggers mode. Like the other panels it can float, dock on any edge of the main window, or sit in a tab with another panel: drag it by its title bar onto the middle of the Status Log to read a trigger's fields under the map while you scroll the list. Switching to Triggers mode, or opening the list of condition and effect types, brings its tab to the front. **View > Panels > Reset Layout** puts it back under the left pane, and [View and Filters](view-and-filters.md) covers the panels.

Docked above or below the map, Trigger Details spreads its fields across the panel instead of one tall column. In **Game layout** each field sits where the game's own trigger editor puts it: the object list type and object list in the first column, the source player, object group and object type in the second, Set Objects, Set Area and Set Location in the third, and so on. Typed coordinates sit on one line under their Set, Go to and Reset buttons, each beside its caption. A field the game's editor does not place takes the first free spot, fields move up and left into empty spots so a type with few fields keeps them together, and each column stacks on its own, so a tall box never pushes the next column's fields down. **Compact columns** keeps the usual field order and fills columns down to the panel's height, captions beside the fields. Pick one under **Settings > Appearance > Trigger Details under the map** ([Settings](settings.md)). Docked on the left or right it is one column either way. Floating, it uses that layout while the window is wider than it is tall, and one column otherwise.

The line above the list counts the triggers and says which order they run in, a setting in [Map Options](map-options.md). The first column is each trigger's place in display order. It is headed **Exec #** when that is also the order they run in, and **Display #** when the file runs them by trigger ID or doesn't store which order it uses.

## The trigger list

Type in **Filter triggers…** to show only the triggers whose names match. The sort box beside it switches between **Display order**, the order the in-game editor shows, and **File order (trigger ID)**. A disabled trigger shows "(disabled)" after its name.

Click a trigger to edit it. Ctrl+click and Shift+click select several; Trigger Details then shows only a count, and the buttons act on all of them. **Edit > Select All** selects every trigger the filter shows.

Double-click a trigger, or right-click it and choose **Rename**, to rename it right in the list. Enter or clicking away keeps the new name, Esc cancels, and each rename is one undo step. On a section header you edit only its title, and its dashes stay as they are; change those in the trigger's Name field. A section header with triggers under it keeps collapsing on a double-click, so rename it from the right-click menu or the **Sections** button.

**Sections.** A trigger whose name starts and ends with a run of dashes or similar characters, such as `--- Setup ---`, is a section header. So is one that starts with such a run followed by a space, letter or digit (`----- Ambush`), and one wrapped in a single `~` on each side (`~Part I~`). A section header DEscape adds always has a run on both sides, in the file's most common style. In Display order the triggers below it, up to the next header, sit under it and collapse with it. The **Sections** button adds a section header at the end of the current trigger's section, renames the selected one, moves the selected triggers into another section, collapses or expands every section and jumps to any of them. A collapsed header stands for its whole section: Copy, Cut, Duplicate and Delete take the section with it.

**Tags.** A name that starts with a bracketed tag, such as `[P1] Spawn wave`, carries that tag, and `[P1][Wave] Spawn` carries both; `(tag)` and `<tag>` work too. Pick a tag in the tag box (it starts on **All tags**) to show only the triggers that carry it. The **Tags** button adds a tag to the selected triggers or removes one from them, and renames or removes the chosen tag on every trigger that carries it, hidden ones included. Each is one undo step.

## Adding, moving and copying triggers

The buttons under the list:

- **New** adds a trigger at the end of the list. **Delete** removes the selected triggers.
- **Copy** puts the selected triggers on the clipboard, and **Paste** puts them below the current trigger. Links between the copied triggers point at the pasted copies.
- **Duplicate** copies the selected triggers in place. Their trigger links still point at the originals; use Copy and Paste to re-link a block.
- ▲ and ▼ move the selected triggers one place in display order. Shift+click moves them five places, and Ctrl+click to the top or bottom of their section (of the list, if it has no sections). ⇈ and ⇊ move the selected trigger's whole section past the next one.

Moving needs an empty filter and tag; a greyed button's tooltip says why. You can't drag rows to reorder them. In a file whose Map Options trigger execution order is Legacy, moving in Display order changes the list but not which trigger runs first, and the status log under the map says so on the first move. To make such a file run in the listed order, arrange the list and then right-click and choose **Renumber IDs to Display Order**: every trigger gets the ID of its place in the list, so File order matches Display order, and trigger links (Activate Trigger, Deactivate Trigger and Trigger Active) follow their triggers. It is one undo step.

In File order (trigger ID) view the same moves change trigger IDs instead: the selected triggers take earlier or later IDs, the triggers they pass shift by one, and trigger links follow. Ctrl+click gives them the first or last IDs. Every trigger keeps its place in Display order. In a Legacy file this changes which trigger runs first; in a file that runs in display order it does not, and the status log says so on the first move. Each press is one undo step. The section buttons stay greyed in this view, since it has no sections.

In Triggers mode the Edit menu reads **Edit > Cut Triggers**, **Edit > Copy Triggers** and **Edit > Paste Triggers**, unless a text field or the condition and effect list has focus. In a text field its plain **Edit > Cut**, **Edit > Copy** and **Edit > Paste** act on the text. A cut trigger pasted back keeps its name and the links other triggers held to it. Copied triggers also paste into another scenario of the same version: copy, open the other file and paste. Links to triggers outside the block and placed-unit references are cleared then, since they name the first file's triggers and units.

Right-click a trigger for Cut, Copy, Paste below it, **Rename**, the four moves, **Renumber IDs to Display Order**, and **Select Section** or **Select Tag**. The Move Trigger/Condition/Effect commands make the same moves from the keyboard; [Keyboard and Mouse](keyboard-and-mouse.md) lists their keys.

## Conditions and effects

The lower pane lists a **Trigger** row, which holds the trigger's own fields (name, descriptions, enabled, looping and the objective settings), then its Conditions and Effects in display order.

- **New** opens a list of every condition and effect type. Filter it, then double-click a type or press **Add**. With Conditions or Effects highlighted first, the other group starts collapsed.
- **Type…** changes the selected condition or effect to another type. Fields the new type lacks are dropped, and the status log under the map names them.
- **Duplicate**, **Delete**, ▲ and ▼ act on the selected ones, with the same Shift and Ctrl clicks as the trigger list.
- Right-click for Cut, Copy, Paste, the moves and **Select Same Type**. Conditions and effects have their own clipboard and paste into any trigger. With this list focused, the Edit menu reads **Edit > Cut Conditions/Effects**, **Edit > Copy Conditions/Effects** and **Edit > Paste Conditions/Effects**.

Each condition and effect's Detail reads as a short sentence, such as `P2 to P1, visible` for Set Player Visibility or `"Wave 2"` for Activate Trigger. Underlined parts are links:

- Coordinates centre the map on the area or location.
- A placed object, or several selected objects, centres the map on them.
- An object type such as `P7 Castle` centres on the next matching object on the map at each click, only those inside the entry's area when it has one; the status log counts them.
- An object Create Object or Place Foundation is about to make goes to where it will appear, and Replace Object's replacement goes to what it replaces.
- A trigger name selects that trigger.

A field the sentence leaves out is added after a `+` when it is set. Hover a Detail for its set fields as stored. **Settings > Appearance > Condition and effect details** switches to Full sentences or the raw fields, and changes how players, flags, ids and long text read.

Select a row to edit its fields in Trigger Details. Select several and the form shows only the fields they share; one they disagree on reads "(differs)" and changes nothing until you set it. Each change is one undo step.

Conditions and effects keep the fields they share in the same order, so a field sits in the same place from one type to the next, and Modify Attribute and its siblings start with their quantity rows.

Unit, building and technology fields take a typed name, or their "…" button browses the list. An object field's list works like the Units catalog: category buttons, a class dropdown, a filter box, a view dropdown (Icons, Names or Grid), and the object's portrait and stats underneath. The technology list stays a plain list. It leaves out placeholder and helper technologies (such as Blank Technology and the "requirement" techs) until you tick **Show internal techs**; a field that already holds one still shows it. An unset number reads "(unset)". Descriptions, messages and scripts get a full-width box that applies your text when you leave it or save; while it has focus, **Edit > Undo** and **Edit > Redo** act on its text. Drag the strip along a box's bottom edge to change its height in whole lines, or double-click the strip to reset it. Each field remembers its height.

An object list field has an **object list type** row above it, as in the in-game editor: Units, Buildings, Heroes or Others. The field's suggestions list that kind first, then the others, and "…" opens on it, with buttons for the other kinds. It follows the field's value, but isn't saved in the scenario. Typing or picking a name of another kind applies it and turns the row to it, as picking another kind in "…" does. Object Group and Object Type sit right under the object list, above the Set rows.

Some types show extra help under their fields:

- **Display Instructions** draws its message on the map for as long as the effect is selected, sized and placed as the game draws it, with the source player's portrait and the coloured text. The box stays put while you pan or zoom; **View > Show Instruction on Map** turns it off. Under the fields, the display time and sound stay listed. Sounds are named, not played.
- **Create Object** and **Replace Object** show roughly which way the unit will face. That mapping isn't confirmed in the game yet.
- **Change Train Location**, **Add Train Location** and **Change Research Location** show the building's command buttons under the button slot field, as they will look after the trigger fires; click a cell to pick that slot. The grid shows only what the effect player's civilization can have (all civilizations when it is random), with unit portraits in that player's civilization and colour. It frames the effect's own button in green, or in red when it won't show or covers a technology; hover a cell for the details. Tick **Show all triggers' changes** to apply that player's other such effects too. The [Map Analysis](map-tools.md) "Train and research buttons" check warns when such a button won't show: a unit the civilization can't have, a technology the player's starting age already researched, one hidden by a lower-id unit, a covered technology, or a slot with no button.

## Picking on the map

Location, area and object fields have **Set**, **Go to** and **Reset** buttons. For a location, press **Set** and click a tile; for an area, press **Set** and drag a rectangle. A single unit field has a **Pick** button: press it and click the unit. For a list of objects, press **Set** and click units to add or remove them. An object type field, such as Modify Attribute's source player unit, has an eyedropper button after its **…**: press it and click a unit on the map to copy that unit's type, with the player field left as it was. A repeat click on a stack picks the next unit down, and units hidden by Filters can't be picked. Right-click or Esc cancels, or finishes a list. **Go to** centres the map on the field, even on a read-only file, and **Reset** clears it.

To move or resize an area on the map, keep **View > Show Trigger Overlay** on (the default), pick the Pan tool and tick **Edit Area** on the Tool Options row. Select one condition or effect with an area, then drag the area's middle to move it, or an edge or corner to resize it. **Move** and **Resize** choose which parts you can grab, and **Tile borders** lets you resize from the whole outer ring of tiles. Each drag is one undo step; right-click or Esc cancels it.

With **Move** ticked, the same setup moves a location: select one condition or effect with a location and drag its mark. The mark sits on top of the area, so it wins where they overlap. While you drag, the line from the area or units and its distance follow the mark. A dimmed mark can't be dragged: the effect's location object is its real destination and the game ignores that location, so press **Set** on the Location row if you need to change it anyway.

With a Create Object effect selected, the **Create Objects** tool on the toolbar adds one copy of that effect per tile under the brush, skipping tiles that already have one. Each click is one undo step.

## Status colours

A trigger, condition or effect shows green when every field its type needs is set and every unit and trigger it names exists, and red when something is missing. Hover a red row to see what. Green means complete, not that the trigger does what you meant in the game: nothing here runs it. A trigger with no effects is red unless it is shown as an objective, an objective header or on screen. A condition or effect of a type DEscape doesn't know is left unmarked. A section row is judged by its own conditions and effects, not the triggers in its section, and isn't coloured unless you ask.

**Settings > Appearance > Trigger status** switches the colours to tick and cross icons, or both, or off. It can stop marking rows that pass so only problems stand out, colour section rows, and change the two colours. The icons sit in their own column between ID and Trigger, so the names line up; a section with no conditions or effects shows a dash there.

## The overlay and variables

**View > Show Trigger Overlay** draws the selected trigger's areas and locations on the map, with a line from an area to its destination, and keeps them there when you switch modes. The selected condition or effect is drawn strongest. **View > Show Trigger-Created Units** draws each unit a Create Object or Place Foundation effect would create, see-through on its location, as if the effect fired. **View > Range Rings > Trigger Changes** adds a dashed ring for each trigger that changes a selected building's range; see [View and Filters](view-and-filters.md).

**Variables…** under the panel lists the scenario's trigger variables. Add one by name, or rename (also by double-clicking) or remove the selected one.

**Settings > Keybinds** has unbound commands for **Trigger Variables** in Triggers mode and **Add Variable**, **Rename Variable** and **Remove Variable** inside the dialog. The editing commands work only while the dialog has focus, even if you switch modes with it open. Add needs a name; Rename and Remove need a selected row. Read-only scenarios can browse the list but cannot edit it. Each edit is one undo step.

## Objectives panel

The **Objectives** panel lists every trigger set to Display as Objective, Header or Display on Screen, in the order the game lists them: higher **Description Order** first. Triggers with the same Description Order keep their trigger list order, which is how DEscape assumes the game breaks a tie. By default it sits in a tab behind the Status Log, in Triggers mode only; **View > Panels > Objectives** shows or hides it.

The columns are Order, Header, Objective (Display as Objective), On Screen (Display on Screen), Mute, an eye (shown in the preview), Short description and Trigger. Drag a column's header to move it; the order and widths are remembered. Untick a row's eye to leave it out of the preview; that works on a read-only file too, where the flags cannot change. Click a box to set a flag; with several rows selected it sets them all. Double-click the Order to type one, or the short description to edit it. A short description with a language string id shows the game's text in italics; its tooltip names the string. The game shows that string, not the stored text, so editing such a row asks whether to clear the id so your text shows, and right-click > **Use My Text (Clear String ID)** does that for the selected rows. Text typed in Trigger Details' short description shows in the list and the box as you type. A trigger that starts disabled is greyed, since the game shows it only once a trigger activates it. With the list focused, **Edit > Select All** and **Deselect** act on its rows, and Cut, Copy and Paste do nothing. Selecting a row selects its trigger in the trigger list, where Trigger Details edits the long Description, and the other way round.

Drag rows, or use the up and down buttons, to reorder them. DEscape gives a moved row a Description Order between its new neighbours when there is room. Otherwise it takes the value just under the row above, and the rows below move down one each until there is a gap, so a header further down keeps its value; only when that would go below 0 is the whole list renumbered. **Renumber** counts the whole list down by 1 from the top row's value, in the listed order. With several rows selected it renumbers only those, counting down from the first selected row's value; a row between or below them moves down only where it would collide, so the order never changes. Order never goes below 0.

**New Objective** and **New Header** add a trigger under the selected row, flags and text already set; a new objective's text starts with "+ ". **Add Trigger…** shows existing triggers as objectives. **Remove** clears the Objective, Header and On Screen flags and keeps the trigger. Each action is one undo step. A read-only scenario can still browse the list and preview it.

**Preview** draws the game's on-screen objectives box **In dock** (beside the list, or below it while the panel is taller than it is wide, as when docked on a side), **On map** (over the map's top-right corner, where the game draws it, also **View > Show Objectives on Map**) or **Off**. **All objectives** shows every line; **At game start** only triggers enabled when the game starts. As in the game, a short description starting with **+** draws a square checkbox and one starting with **-** a round one (an optional objective); neither sign shows as text, and a line with neither has no checkbox. **Variables** sets how a `<name>` tag naming a trigger variable reads in the box: **Name** shows the tag as typed, as the game draws it; **Value** shows its value at game start (0); **Both** shows `0 (name)`. Clicking a box's checkbox shows that objective as done. Right-click rows and choose **Hide in Preview** to leave them out of the box; their short description greys in the list, and **Show in Preview** or **Show All in Preview** brings them back. A long box scrolls in the panel. The box is never narrower than the game's, so a panel narrower than that scrolls it sideways while the list stays put. On the map it stops at the map's bottom edge, and the mouse wheel over it scrolls its lines; elsewhere the wheel zooms as usual. Ticks and hidden rows are preview only: never saved, and cleared when another scenario opens or a trigger is added or deleted.

## Script Call and XS

A Script Call effect or condition holds XS script in a monospace box with no line wrapping. DEscape sets no limit on its length, so a script can be longer than the in-game editor accepts. The [XS scripting notes](https://github.com/Combinebobnt/DEscape/blob/main/docs/XS_SCRIPTING.md) explain how XS reaches the game and the rules a script must follow. A scenario-wide script file is named in Map Options instead.

## Find and Replace

**Edit > Find and Replace…** has a Triggers tab. It searches trigger names, descriptions, messages, identifiers and XS (as plain text or a regex), the unit types conditions and effects name, and which triggers use given placed objects. It replaces text or types, deletes, enables or disables the ticked matches, or exports them as CSV; each operation is one undo step. Double-click a match to jump to it. On the Objects tab, **Show trigger usages** lists the triggers that name the ticked objects, and replacing an object's type offers to retarget the effects that select it by type, in the same undo step. [Map Tools](map-tools.md) covers the Objects tab.

## Further reading

The AoE2 Scenario Editor Guide describes each [effect](https://ugc.aoe2.rocks/scenarios/triggers/effects/effects/) and [condition](https://ugc.aoe2.rocks/scenarios/triggers/conditions/conditions/), and the [XS language](https://ugc.aoe2.rocks/general/xs/) with its [functions](https://ugc.aoe2.rocks/general/xs/functions/functions/).

Where this is in the in-game editor: the Triggers tab.

DEscape doesn't run triggers or test-play the scenario, so test a trigger in the game.
