# Messages

Messages mode edits the scenario's six text fields: **Instructions**, **Hints**, **Victory**, **Loss**, **History** and **Scouts**. Pick **Messages** in the toolbar's Mode list, and each field gets its own box in the left pane.

In the game, Instructions, Hints and Scouts fill the matching tabs of the objectives panel, and Victory and Loss are shown when the game is won or lost.

## Editing the text

Type in a box; the text applies when you leave the box, as one undo step. While a box has focus, **Edit > Undo** and **Edit > Redo** act on its own text. Line breaks are written back the way that field already stored them.

Drag the strip along a box's bottom edge to change its height in whole lines, or double-click the strip to reset it. Each box remembers its height between sessions.

A field holds at most 32,767 bytes of text, and DEscape refuses a longer one rather than save a file the game can't reopen. The status log under the map says why an edit was refused.

## String table IDs

A field can name a string from the game's language files instead of using its own text. When one does, its box shows a warning that the game displays that language-file string, not the text below, and the line at the top counts such fields. Press **Use my text (clear string ID)** to clear the ID so the game shows your text again. DEscape can clear a string ID but has no field to set a new one.

When the whole page is read-only for a file, the line at the top says so and its tooltip says why.

Where this is in the in-game editor: the Messages tab.
