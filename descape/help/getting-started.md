# Getting Started

DEscape opens, edits and saves Age of Empires II: Definitive Edition scenario files (`.aoe2scenario`) outside the game. It is not a copy of the in-game Scenario Editor. It is meant for the edits that are tedious there: painting large areas of terrain and elevation, changing many units at once, and working through long trigger lists.

**Help > DEscape Guide** opens the page for the mode you are in, and **Help > Keyboard and Mouse** lists your keys.

## Open a scenario

- **File > Open .aoe2scenario…** opens a file from disk.
- **File > Open Recent** lists the scenarios you opened before. A file that has gone missing drops off the list until it is back.
- Drop a `.aoe2scenario` file onto the window to open it. If you drop several, the first one opens.
- **File > New Map…** starts a blank scenario. Pick its size (one of the game's sizes, or a square Custom size), the game it is for (Age of Empires II or Return of Rome), the terrain every tile starts with and its elevation. Elevation is the value stored in the file: the game's Scenario Editor shows it one higher. The dialog remembers your last choice.
- **File > Close Map** closes the scenario.

Opening another file, closing or quitting asks first when there are unsaved changes. While there are, the window title starts with `*`.

## Save

**File > Save** writes your changes back to the open file, and **File > Save As…** writes them to a new file. A new map, or a scenario recovered from an autosave, has no file yet, so Save asks for a name.

Saving leaves a `.bak` and, on the first save of a given file, a one-time `.orig` snapshot beside it, so there is always something to fall back to. The `.bak` holds the file as it was just before the latest save. **Settings > Saving** can switch that pair off, except in the game's own profile folder, where Save always writes both.

You can save straight into the game's scenario folder. A subscribed-mods `mods/subscribed/` folder is refused, because the game's mod manager re-syncs it and would silently undo the edit.

## Autosave

Autosave (**Settings > Saving**, on by default) writes a recovery snapshot every few minutes to its own rotating slot. It never writes the file you have open, and never clears the unsaved-changes marker. It skips a turn while the document has no unsaved changes, is unchanged since its last autosave, or is in the middle of a stroke, so it only ever writes an idle map.

Snapshots go in an `autosave` folder beside DEscape's `config.yaml`, or a folder you pick in **Settings > Saving**, or beside the scenario itself if you choose that. A scenario in the game's profile folder keeps its snapshots in the autosave folder, out of the game's own folder. A `mods/subscribed/` folder is refused. Switching folders leaves earlier snapshots where they are, and Recover still lists them.

**File > Recover from Autosave…** lists the snapshots and opens one as a new untitled document, so recovering can never overwrite the file you are comparing it against. It works with no map open. Saving a document removes its snapshots, since the file on disk now holds that state.

## Return of Rome and data mods

A Return of Rome scenario opens with Return of Rome's own units, names, art, civilizations and terrain, and the window title says `[Return of Rome]`. **Settings > Advanced** can show a file with the other game's data instead. A file shown that way is read-only, and the status bar says why. Set Game data back to Auto to edit it.

**File > Data Mod** shows an Age of Empires II scenario with a data mod's own units, stats and names. DEscape remembers the choice for each saved scenario.

## Move around the map

- Drag with the Pan tool, which every mode has, or middle-drag in any tool.
- Zoom with the mouse wheel. **Settings > General** picks whether it zooms around the cursor or the middle of the view, and **Settings > Keybinds** sets what the wheel does with each modifier held.
- Hold a Pan key (the Pan Up, Down, Left and Right rows in **Settings > Keybinds**) to scroll. Keys on the map only work while it has focus, so click it first.
- The Elevation View box on the toolbar picks how the map shows height: Flat (none), Stepped (each tile raised to its height, as blocks) or Sloped (smooth slopes between heights).
- In Flat, **View > Isometric View (game-style)** switches between the game's isometric look and a plain top-down grid.

[View and Filters](view-and-filters.md) covers what else the map can show or hide.

## Modes

The Mode box at the left of the toolbar picks what the side panel shows and which tools appear. Each mode also has a command in **Settings > Keybinds**. Pan and Ruler work in every mode; the other tools show only where they apply.

| Mode | For | Page |
|---|---|---|
| View | Statistics about the open file | This page |
| Terrain | Terrain, elevation, cliffs, copying and pasting regions | [Terrain](terrain.md) |
| Units | Selecting, placing, moving and turning units | [Units](units.md) |
| Triggers | Triggers, their conditions and effects | [Triggers](triggers.md) |
| Map Options | Scenario-wide settings and victory | [Map Options](map-options.md) |
| Players | Each player's settings | [Players](players.md) |
| Diplomacy | Who is allied with whom | [Diplomacy](diplomacy.md) |
| Messages | Instructions, hints, victory and defeat text | [Messages](messages.md) |

View mode's side panel shows the file, one player's counts at a time (pick the player in its first row), the eight most common terrains, and technical details. The line above it describes the tile under the cursor. You can select rows and copy them as text ([Keyboard and Mouse](keyboard-and-mouse.md) has the key). The in-game editor has no counterpart to View mode.

The commands that work on the whole map, such as Resize Map, Mirror Map, Find and Replace and Map Analysis, are in [Map Tools](map-tools.md).

## What stays in the in-game editor

DEscape never runs a scenario: nothing here fires a trigger, so play-test it in the game. It makes blank maps, not random ones, and has nothing for the in-game editor's Cinematics tab. The elevation tools, **Map > Mirror Map…** and **Map > Resize Map…** need a square map.

## Further reading

- The community [AoE2 Scenario Editor guide](https://ugc.aoe2.rocks/scenarios/) (also **Help > AoE2 Scenario Editor Guide (ugc.aoe2.rocks)**), with its [trigger effects](https://ugc.aoe2.rocks/scenarios/triggers/effects/effects/), [trigger conditions](https://ugc.aoe2.rocks/scenarios/triggers/conditions/conditions/), [XS](https://ugc.aoe2.rocks/general/xs/) and [XS function](https://ugc.aoe2.rocks/general/xs/functions/functions/) references.
- [Scriptable batch edits](https://github.com/Combinebobnt/DEscape#scriptable-batch-edits): edit scenarios from a Python script.
- [XS scripting in DEscape](https://github.com/Combinebobnt/DEscape/blob/main/docs/XS_SCRIPTING.md), including scripts longer than the in-game editor accepts.
- **Help > Release Notes** and **Help > Report a Bug…**.
