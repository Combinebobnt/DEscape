# Settings

**Edit > Settings…** opens the Settings dialog. A change applies as soon as you make it and is remembered between launches; there is no OK or Apply button. DEscape keeps its settings in a `config.yaml` file in your user settings folder: `~/.config/DEscape` on Linux, `~/Library/Application Support/DEscape` on macOS and `%APPDATA%\DEscape` on Windows.

Each tab:

## General

- Mouse wheel zoom: zoom around the mouse cursor, or around the middle of the view.
- AoE2DE install path: optional. Type the game's install folder or Browse… to it, then Load, which checks the path and saves it. With an install, terrain draws with its real textures and **View > Show sprites** can draw units as the game does. Without one, terrain falls back to flat colours.
- On Linux, X11 compatibility on Wayland runs DEscape under XWayland from the next launch. Try it if map drags lag or freeze for seconds at a time; it costs a blurry window under fractional display scaling.
- Settings file: Export settings… saves your preferences as a `.descape-settings.yaml` file, and Import settings… loads one, say on another computer. The file carries the preferences from every tab, the View menu and the panels: keybinds and mouse wheel actions, the look, tool defaults, saving options and the rest. It never carries folder paths (the install, the profile and the autosave folder), your recent files, the New Map dialog's last choices, **File > Data Mod**'s choice for each map, the window's size and layout, or X11 compatibility on Wayland; those stay as they are on this computer. Export refuses a `mods/subscribed/` folder, and will not replace `config.yaml` or the import files kept beside it.
- An import replaces your preferences rather than adding to them: one the file leaves at its default goes back to the default here too. Import settings… first lists what would change and what would reset. Nothing changes until the next time DEscape starts, so after Apply you can quit DEscape at once or carry on and restart later. Until then a line under the buttons names the file, and Discard cancels the import; if this version of DEscape cannot apply the staged import, the line says so instead. When it is applied, the old `config.yaml` is kept as `config.before-import.yaml` beside it.

## Mods

- AoE2DE profile folder: optional, and usually found on its own. It is where Players mode finds custom AIs from your profile and mods, and where your mods' art comes from. Browse… and Load set it; Use auto-detect forgets your choice and finds it from the install again. Below it is every folder DEscape reads AIs and other content from. A mod disabled in the game's mod list is left out there, though the Mod art list can still tick its art on, and **File > Data Mod** still offers a disabled data mod.
- Mod art list: every mod in the profile the game last used that ships art, in the game's mod order, the top one winning where two ship the same file. Untick a mod to leave it out of **View > Layers > Mod Art**; it stays off even if you reinstall it. A mod disabled in the game's own mod list is listed too, marked "(disabled in game)" and off until you tick it. Ticking it draws its art in DEscape only; the game's mod list is not changed. Mods are read once per launch, so after enabling or disabling one in the game, restart DEscape or load the profile folder again.

## Appearance

Theme, chrome colours, UI font and toolbar buttons are covered in [View and Filters](view-and-filters.md). The rest of the tab:

- Preload neighbouring zoom levels prepares the next zoom levels and the map just outside the view in idle time, so zooming and panning don't stutter. It costs some background work after a map opens, and memory.
- Graphics quality: Potatest, Potato, Default or Enhanced. Enhanced draws large maps at double resolution and, with the game's free Enhanced Graphics DLC installed, uses its high-resolution unit art when zoomed in. The line below the slider says whether the DLC was found. Return of Rome scenarios use Return of Rome's own high-resolution art, with no DLC needed.
- Isometric elevation height: how tall one elevation level looks in Stepped and in Flat with Isometric View.
- Pan speed: how fast holding a Pan key scrolls the map.
- Wall opacity: how solid walls and gates look while **View > Layers > Translucent Walls** is on.
- Tool overlay colors: the colours the tools and overlays draw on the map, grouped by what draws them, such as the terrain brush, the unit selection, the ruler, the Select tool's region and the trigger overlay. Each colour has a Default button, and some groups have an opacity slider. The Units group's Refused placement row is the red a refused Place Unit or Move ghost is tinted toward; the cross-hatch on a refused mark is black or white, whichever reads on that colour.
- Ruler label size, Distance ticks label size and Stacked-unit badge position set the text drawn on the map.
- Terrain grid swatch size: how big each swatch is when Terrain mode's terrain picker shows a grid.
- Grid line blend and Grid thickness: how **View > Grid** looks. Left of the middle, blend darkens the terrain under the lines towards black; right of it, it lightens it towards white; in the middle the lines are invisible.

### Trigger status

**Settings > Appearance > Trigger status** sets how the Triggers panel marks each trigger, condition and effect as complete or missing something: Colour, Tick and cross icons, Colour and icons, or Off. Unticking Mark triggers that pass leaves only the problems marked. Colour section rows colours a section's own row by its own conditions and effects. The two swatches change the Complete and Something missing colours.

The icons sit in their own column between ID and Trigger, so the names line up, and a section with no conditions or effects shows a dash there. What counts as complete is in [Triggers](triggers.md).

## Tools

**Settings > Tools** sets what each control on the Tool Options row starts at when you launch DEscape: the brush size, shape and outline thickness; Auto beach, the beach terrain and width, Trees and Eye candy; the Set Elevation level and whether Draw Rectangle is filled; the Paste Region ticks; Free placement, Off-grid, Walls skip occupied tiles and Moving a gate updates walls; the wall the Wall button picks, and the Cliff family, piece and frame; the Unit Brush's density and Random facing/variant; Convert's sources, the Eraser's targets, and the Edit Area options for trigger areas and locations.

A change on this tab shows on the toolbar at once and is kept for every later launch. A change you make on the toolbar itself lasts until you close DEscape. **Default** on a row puts that option back to DEscape's own default, **Load From Toolbar** makes everything the toolbar shows right now the default, and **Reset All to Defaults** puts every option, and the toolbar, back to DEscape's own defaults. Changing the Cliff family resets the piece and frame, and changing the piece resets the frame. A wall, beach or cliff the current game data doesn't have shows as Unavailable and is kept until you change it. Target always starts on Terrain.

## Saving

- Autosave on a timer, Autosave every and Keep per document: whether autosave runs, how often, and how many snapshots each document keeps.
- Autosaves go: in the autosave folder, or beside the scenario file. Beside the file falls back to the autosave folder for an untitled document, a `mods/subscribed/` path, a scenario in the game's own profile folder, or a shipped template.
- Autosave folder: Browse… picks one and Use Default goes back to the folder beside `config.yaml`. A `mods/subscribed/` folder is refused.
- Write .bak and .orig backups on save. Saves into the game's own profile folder write both even when this is off.

[Getting Started](getting-started.md) explains saving, autosave and recovering.

## Keybinds

**Settings > Keybinds** rebinds every shortcut and sets what the mouse wheel does over the map. The search box at the top filters the rows by command or key. [Keyboard and Mouse](keyboard-and-mouse.md) shows what you have now.

The Mouse Wheel section comes first, with a row for the wheel alone and one each for Ctrl + Wheel, Shift + Wheel and Alt + Wheel. Each row is one of:

- Zoom.
- Brush size: steps the size in tools with a Brush size box.
- Rotate preview: turns a unit you're dragging with Move, Place Unit's preview or a just-pasted region.
- Brush size or rotate preview: whichever of the two applies.
- Pan up / down, or Pan left / right.
- Nothing.

Each choice's tooltip says where it applies. Where Brush size or Rotate preview has nothing to act on, the wheel does nothing rather than zoom. A modifier combination with no row of its own zooms. Holding Shift or Alt for a tool's own modifier, such as a straight wall run or off-grid placement, also uses that row. Some desktops take Alt + Wheel for themselves.

Below it is a row for every command, grouped like the menus. Click a row's box and press the keys you want. A command that works only in some modes shows them after its name, and its key can also go on a command of another mode, each working in its own. Commands that work in every mode keep their key to themselves. Giving one command a key another has in a mode they share clears it from the other, and the tab says which. Default puts the shipped key back, unless another command has it now in a mode they share, and Clear removes the key.

## Advanced

- Game data: which game's units, names and art draw a scenario. Auto uses the game the file was made for; leave it there unless a file names the wrong game. Age of Empires II or Return of Rome shows every file in that game's data, and a file made for the other game opens read-only. It applies to the open scenario at once.
- GAIA-only objects: wild animals, resources and trees can only belong to GAIA, as in the game's editor. Choose whether the Units catalog hides them while the owner is a player, or keeps them listed. Placing one gives it to GAIA either way.
