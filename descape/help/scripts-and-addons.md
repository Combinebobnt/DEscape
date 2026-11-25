# Scripts

A script is a small Python file that edits the open map: paint a patch of terrain, move every unit of a kind, raise the ground under a player's buildings. It runs once, against the scenario open in the window, and what it changes is one undo step.

## Running a script

**Tools > Scripts** lists the `.py` files in your scripts folder, by file name. The list is read again each time the menu opens, so a file you just saved there shows up without a restart. Files whose names start with a dot or an underscore are skipped.

**Tools > Scripts > Open Scripts Folder** creates the folder if needed and opens it in your file manager. It is `scripts` in DEscape's per-user data folder: `%LOCALAPPDATA%\DEscape\scripts` on Windows, `~/.local/share/DEscape/scripts` on Linux and `~/Library/Application Support/DEscape/scripts` on macOS. The environment variable `DESCAPE_SCRIPT_PATH` points it somewhere else.

The first time you run a script, DEscape asks "Run script?" and shows its path. No is the default. Once you say Yes it runs that file without asking again, until the file changes: an edited script asks again.

A script needs an open scenario that can be edited. It is refused while another edit is still in progress, and on a document shown with another game's data. Starting DEscape with `--no-addons`, or with `DESCAPE_NO_ADDONS=1` set, turns scripts off for that session: the rows are greyed out and say why.

## What a run does

- **One undo step.** Everything a run changes, terrain and units together, is one entry in the history, labelled with the script's name. **Edit > Undo** puts all of it back. A run that changed nothing adds no entry, and the status line says so.
- **All or nothing.** If the script raises an error, or you press Cancel, everything it changed is put back and the history is untouched. An error opens a message box saying what went wrong, with the traceback behind Show Details; a cancel is only reported on the status line.
- **Progress.** A script that calls `editor.progress()` gets a progress window with Cancel. Until then none is shown, and the window does not respond while the script runs.
- **Output.** What the script prints goes to **Help > Debug Log**, the last 8,000 characters of it per run.
- **Saving is yours.** A script does not save. Look at the result, then use **File > Save** as for any other edit.

## Writing a script

A script gets one name, `editor`, and that is the whole supported API. It needs no imports for it:

```python
# Paint a 5x5 patch of terrain 2 in the top corner, then move one unit 3 tiles east.
for x in range(5):
    for y in range(5):
        editor.set_terrain(x, y, 2)

archer = editor.units(1)[0]
editor.move_unit(archer, archer.x + 3, archer.y)
print("done")
```

What a script reads:

- `editor.width`, `editor.height` and `editor.name`, the map's size and the scenario's file name.
- `editor.tile(x, y)`, the terrain tile at a tile position, with its `terrain_id`, `elevation` and `layer`.
- `editor.units(player)`, a player's units (0 is GAIA), or every unit with no player given.
- `editor.selected_units`, the units selected in Units mode, and `editor.selected_tiles`, the (x, y) tiles of the Select region.
- `editor.loaded`, the scenario itself, for the read helpers of the [batch editing API](https://github.com/Combinebobnt/DEscape#scriptable-batch-edits) such as `buildings_of`, `neighbors` and `is_water`.

What a script changes, each with the same rules the editor's own tools keep:

- Terrain: `set_terrain(x, y, terrain_id)` (which also clears a layered terrain), `set_layer(x, y, terrain_id)` (None removes it), `set_elevation(x, y, level)` and `raise_elevation(units_or_tiles, amount)`. Elevation needs a square map.
- Units: `move_unit(unit, x, y)` (a unit with others garrisoned inside carries them), `add_unit(const, x, y, player, facing=None, variant=None)`, `remove(units)`, `reassign(unit, player)`, `rotate(unit, facing)`, `set_variant(unit, index)`, `garrison(unit, host)` (`garrison(unit, None)` unloads it), `set_status(unit, value)` and `set_capture_flag(unit, value)`.
- `progress(done, total, note="")` updates the progress window, and is where Cancel takes effect.

A request `editor` cannot carry out, such as rotating a building or garrisoning into a full house, raises `editor.ScriptError` with the reason. The script may catch it and carry on; uncaught, it fails the run.

## The rules a run keeps

- **A unit changes only through `editor`.** Setting a unit's fields yourself, or adding to or removing from a player's unit list directly, fails the run and puts everything back, even when `editor` is used afterwards. A direct change would skip the checks that keep the file valid.
- **Terrain may be set directly**, by assigning a tile's `terrain_id`, `elevation` or `layer`, and is still undone with the rest. The run fails if a tile it changed holds something the file cannot store or the game cannot load: an id outside 0 to 255, a layer equal to its own terrain, a layer on a file with no layer field, or an elevation out of range or more than one level from a neighbouring tile.
- Triggers, map options, players and messages are not part of `editor`. A script that changes them through `editor.loaded` may have the change lost at save, or saved with no undo.

## Scripts are not sandboxed

A script is ordinary Python, running with your own privileges: it can read and write any file you can. The rules above keep `editor` inside what DEscape's own tools may do and catch mistakes; they do not stop harmful code. Run only scripts you trust, which is why DEscape asks before each new or changed one.

In a prebuilt build (the Windows, macOS and Linux downloads), a script can import only what the app ships: Python's standard library modules that DEscape itself uses, numpy, PyQt5, AoE2ScenarioParser and DEscape's own modules. Run from source, it can import anything installed in DEscape's Python environment.
