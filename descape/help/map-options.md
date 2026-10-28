# Map Options

Map Options mode edits the scenario-wide settings: how the game is won, the map's rules, its water look, its XS script file and the order its triggers run in. Pick **Map Options** in the toolbar's Mode list, and the settings fill the left pane in groups.

The line at the top counts the settings, names the scenario version and says how many are read-only. A setting this file's scenario version doesn't store isn't listed at all. Hover a greyed row to see why it can't be edited. A stored value no editor could show truthfully, such as a number outside the row's range, is shown as stored and kept exactly; hover it for details. Each change is one undo step.

## Global Victory

**Victory condition** picks how the scenario is won: Standard, Conquest, Score, Time limit, Custom, or Secondary game mode (the mode chosen under Map). Its tooltip says what each one means.

- **Score to win** applies to Score victory. A file that never set a score shows 14000, the value the in-game editor fills in; type a number to store one.
- **Time limit, years** applies to Time limit victory: how many in-game years the game lasts, to a tenth of a year. The highest score at the end wins.
- **Conquest**, **Exploration, % of map**, **Relics** and **Conditions needed** set up a Custom victory. A 0 turns Exploration or Relics off, relics need no hold time, and Conditions needed says whether any one enabled condition wins or all of them are needed. These four rows stay greyed, still showing their stored values, until Victory condition is Custom.

## Map

- **Collide and correcting**: standing units step aside for moving units passing through.
- **Villager force drop**: villagers drop what they carry the moment their task changes, not only once they start the new task.
- **Lock co-op alliances**: human players can't change their team or diplomacy with each other in the game. Older game versions called it Block Humanity Team Change.
- **Secondary game mode**: Empire Wars, Sudden Death, Regicide or King of the Hill, played on top of the victory condition, or None.
- **No waves on shore**: hides the animated waves along shorelines.
- **Water definition**: the map-wide water look (sun, sky and water colours, waves and sea floor), from the same list as the in-game editor's Water Definitions. An older file's legacy name, or no name, is shown as stored and kept until you pick another entry. DEscape doesn't preview the look.
- **Script filename**: the XS script file the game runs with this scenario, typed without `.xs`. Leave it empty for none.

The game looks the script file up by name in your profile's `resources/_common/xs` folder and sends it to the other players in a lobby, but not to spectators. Script placed in a Script Call trigger reaches everyone, spectators included; see [Triggers](triggers.md). The name applies when you press Enter or leave the box. DEscape drops spaces around it and a trailing `.xs`, and refuses a folder path or a character a Windows file name can't hold. The row is read-only on a file that already embeds a script.

## Options

- **Full tech tree**: every player gets the full tech tree. Unlike the lobby option, civilization bonuses stay.
- **AI map type**: tells the AI what kind of map this is, for example so it builds a navy on a water map. It is the raw map type number.

## Triggers

**Trigger execution order** sets the order triggers run in. **Display order** follows the trigger list as Triggers mode shows it; **Legacy - trigger ID order** runs them by trigger ID, as older game versions did. Triggers mode's first column follows this setting.

## Elsewhere

The team settings (Lock teams, Players choose teams, Random start points and Max number of teams) are in [Diplomacy](diplomacy.md). Each player's starting camera is in [Players](players.md). The buildings, units and technologies each player can't use are under **Map > Disabled Objects…**, and the map's size under **Map > Resize Map…**; see [Map Tools](map-tools.md).

Where this is in the in-game editor: these settings are spread over the Map, Global Victory and Options tabs, with No waves on shore and Water Definitions on the Terrain tab.

DEscape doesn't edit the map's Colour Mood, Team Positions or Testing Difficulty, and has no Cinematics page. Set those in the in-game editor.
