# View and Filters

The View menu, the Filters button and the look settings change what you see. None of them change the scenario.

## The View menu

The View menu stays open while you tick its entries, so you can switch several in a row. It remembers its choices between launches, except Show sprites and Isometric View, which start on each time.

- **View > Isometric View (game-style)** draws Flat the way the game does: terrain as diamonds and units as upright sprites. Untick it for a plain top-down grid. It is greyed in Stepped and Sloped, which always draw that way.
- **View > Show sprites** draws units as their real game sprites instead of coloured marks. It needs the AoE2:DE install path in **Settings > General**. Turning it on decodes the sprites once, which can take a few seconds.
- **View > Show Stacked-Unit Badges** puts a count, in Units mode, over every spot where a unit is hidden under another. Grass and other passable eye candy never count. Click the spot repeatedly to cycle through the stack.
- **View > Colour Selection by Owner** outlines selected units in their owner's colour. GAIA and the box you drag keep the Selection colour from **Settings > Appearance**.
- **View > Show Player Cameras** puts a camera on each player's starting view tile, in that player's colour. In Players mode the selected player's camera stands out.
- **View > Show Trigger Overlay** draws the trigger selected in Triggers mode on the map, in every mode: its areas, its locations, and a line with the distance from an area to its destination. The selected condition or effect is drawn strongest.
- **View > Show Trigger-Created Units** draws the units the selected trigger's Create Object and Place Foundation effects would create, see-through on their tiles, with or without the overlay. Their facing is not shown yet.

### Range Rings

- **View > Range Rings > Selected Buildings** circles each selected building's attack range, in Units mode.
- **View > Range Rings > Trigger Changes** adds a dashed ring for each trigger that changes a selected building's range, as if that trigger fired once. Open Triggers mode once first so the triggers are read.
- **View > Range Rings > All Buildings** circles every building that has a range, in every mode.
- **View > Range Rings > Selected Map Revealers** and **View > Range Rings > All Map Revealers** circle the area a map revealer reveals.
- **View > Range Rings > Colour by Owner** draws each ring in its owner's colour.

No technology effects are applied, and siege units get no ring.

### Distance Ticks, Grid and Footprint Outlines

- **View > Distance Ticks > Show** draws ruler ticks just outside the map border, numbered, with each edge marked X or Y for the axis it measures. Pick **View > Distance Ticks > Every 4 tiles** or **View > Distance Ticks > Every 5 tiles** for the spacing.
- **View > Grid > Show** draws a line on every tile boundary, with a stronger line every 2, 4, 8 or 16 tiles (**View > Grid > Stronger line every 4 tiles** and its neighbours). **View > Grid > Follow Terrain Elevation** drapes the grid over raised ground in Stepped and Sloped. With it on, and always in Flat, the grid is part of the map: units stand over it and nearer high ground hides the lines behind. Line blend and thickness are in **Settings > Appearance**.
- **View > Footprint Outlines > Show** outlines the tiles each unit occupies, which helps where two buildings share an edge or sprite art hides the ground. Pick which units get one: Multi-tile objects, All buildings, Units only or All units. Merge tiles draws one outline per footprint, and Colour by Owner uses each owner's colour.

### Layers

**View > Layers** turns whole kinds of drawing on or off:

- Terrain Textures: off, each tile is its flat terrain colour.
- Terrain Blending: tiles blend into higher-priority neighbouring terrain, and a tile's layered terrain draws over it.
- Farm Terrain Overlay: a farm's footprint draws as farm terrain in its owner's colour.
- Small Trees: trees at 60% size, so the tiles around them stay readable.
- Hero Glow: a gold outline round hero units, as in the game.
- Fish Outlines: a ring round every fish, so they are easy to find on the water. A deep-water fish, dolphin or whale is ringed round its whole footprint. Fish the Filters hide get no ring, and the colour is set in **Settings > Appearance**.
- Translucent Walls: see-through walls and gates, at the opacity set in **Settings > Appearance**.
- Mod Art: the art your enabled cosmetic mods ship, plus any mod disabled in the game that you tick in **Settings > Mods**, which lists them all.

A greyed layer has nothing to do in the current view, and its tooltip says why.

## Filters

The Filters button on the toolbar shows or hides units by kind and owner, in every mode. Its menu stays open while you tick entries: Show GAIA, Show Trees, Show Walls, Show Buildings, Show Eye Candy, Show Obstacles, Show Invisible Objects and Show Garrisoned Units, one entry per player, then All Players, No Players, **Filters > Show All** and **Filters > Hide All**.

Hidden units can't be selected, and the Eraser never erases them, but **Edit > Find and Replace…** still finds them. Show Garrisoned Units starts off, as in the game: a garrisoned unit is stored at its host's position, so showing it stacks it on the host. Filters reset at every launch, so a map never opens with units hidden by a choice you have forgotten.

## Themes and fonts

**Settings > Appearance** holds the app's look. Theme picks a preset for the app's menus, dialogs and toolbars: Light, Dark, Dim, two high contrast, two Solarized, and Wood and Dark Wood in red, blue, green or purple. The map's own colours never change with it. Chrome colours overrides single colours on top of the preset, and picking another preset clears those overrides.

Export theme… saves the tab's look as a `.descape-theme.yaml` file, and Import theme… loads one: theme and chrome colours, UI font, tool overlay colours, label sizes, badge position and grid look. Graphics quality, preloading, elevation height, pan speed, toolbar buttons and trigger status are not part of a theme file, though a settings file from **Settings > General** carries them with the rest of your preferences. Export refuses a `mods/subscribed/` folder, and will not replace `config.yaml` or the import files kept beside it.

UI font sets the font and size of menus, dialogs, panels, toolbars and the status log; Reset to default goes back to the platform's. Text drawn on the map keeps its own sizes, set by Ruler label size and Distance ticks label size. Toolbar buttons shows the tools as icons, text, or both.
