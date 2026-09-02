# AoE2:DE in-game Scenario Editor — feature reference

A full traversal of the in-game editor's tabs and options, gathered to help
decide what this external tool should mirror next.

## Sources

- Primary: [AoE2DE UGC Guide](https://ugc.aoe2.rocks/scenarios/) — a
  community reference, one long page with numbered sections (1. Map through
  11. Useful Hotkeys), plus dedicated sub-pages for
  [Conditions](https://ugc.aoe2.rocks/scenarios/triggers/conditions/conditions/)
  and [Effects](https://ugc.aoe2.rocks/scenarios/triggers/effects/effects/).
- Official docs exist at `support.ageofempires.com` (e.g. the
  "Main-Menu-Scenario" and "Scenario-Editor-Triggers" articles) but returned
  HTTP 403 to automated fetching — likely bot-blocked, same category of
  problem as this project's Zandronum/ZDoom-wiki situation in the sibling
  orc_slayer project. Not pulled from here; if more detail is needed later,
  these would need to be saved manually (Ctrl+S / print-to-PDF from a real
  browser session) rather than fetched.
- [Age of Empires Series Wiki (Fandom)](https://ageofempires.fandom.com/wiki/Scenario_Editor_(Age_of_Empires_II))
  exists as a secondary source, not yet cross-checked against the above.

All content below is paraphrased/summarized from the UGC guide, not a verbatim copy.

## Tab: Functionality tree

Quick-scan index — full descriptions are in the numbered sections below.

```
Map
├─ Map Style: Blank Map / Random Map / Seed Map
├─ AI Map Type
├─ Colour Mood
├─ Team Positions
├─ Extend Map: New Map Size, Map Offset (N/S/E/W)
└─ Script Filename

Global Victory
├─ Standard
├─ Conquest
├─ Score
├─ Time Limit
└─ Custom Victory: Conquest, Exploration, Relics, Any-One/All switch

Terrain
├─ Map (painting): Brush Size, Layering Mode, Eye Candy, No Waves on Shore,
│  Water Definitions, Beach Type
├─ Elevation: Brush Size
├─ Cliffs: Granite, Sandstone
├─ Map Copy: rotate/flip, Change Player
└─ Erase: Buildings, Gaia Objects, Invisible Units, Layered Terrain, Trees, Units

Options
├─ Point of View: set view, go to view
├─ Testing Difficulty
├─ Full Tech Tree
├─ Disable Objects: Full List <-> Disabled List (Buildings/Units/Techs)
├─ Collide & Correcting
├─ Villager Force Drop
└─ Block Humanity Team Change

Players (per-player)
├─ Number of Players
├─ Starting Age
├─ Colour
├─ Starting Resources: Food, Wood, Stone, Gold
├─ Pop Limit
├─ Base Priority
├─ Tribe Name
├─ Name String ID
├─ Personality (AI script)
├─ Player Type
├─ Civilization
├─ Lock Civ
├─ Architecture
└─ Swap Players

Messages
├─ Scenario Instructions
├─ Objectives
├─ Hints
├─ Scout
├─ Victory
└─ Defeat

Units
├─ Placing: Units / Buildings / Heroes / Others menus
├─ Deleting: single (Delete) / bulk (Terrain -> Erase -> Units)
├─ Moving
├─ Rotating: click CW / right-click CCW
├─ Selecting: HP/Attack/Armour, garrisoned contents, garrison delete
└─ Converting: source player checkboxes, destination dropdown, brush size

Cinematics
└─ Pre/post-scenario movie clips (campaign feature; buggy standalone)

Diplomacy
├─ Diplomacy Stance: per-player-pair grid
├─ Lock Teams
├─ Players Choose Teams
└─ Random Start Points

Triggers
├─ Trigger fields: Enabled, Looping, Description, Display as objective,
│  Short Description, Display on Screen, Make Header, Mute Objectives,
│  Display Order
├─ Editor UI: trigger-list panel (New/Copy/Delete/Info),
│  condition/effect panel (New Effect/New Condition/Delete/Copy)
├─ Conditions (40) -- AI Signal, AI Signal Multiplayer, Accumulate Attribute,
│  And, Bring Object To Area, Bring Object To Object, Building Is Trading,
│  Capture Object, Chance, Compare Variables, Decision Triggered,
│  Destroy Object, Difficulty Level, Diplomacy State,
│  Display Timer Triggered, Hero Power Cast, Local Tech Researched,
│  Object Attacked, Object HP, Object Has Action, Object Has Target,
│  Object Not Visible, Object Selected, Object Selected Multiplayer,
│  Object Visible, Object Visible Multiplayer, Objects In Area, Or,
│  Own Fewer Objects, Own Objects, Player Defeated, Research Technology,
│  Researching Tech, Script Call, Technology State, Timer, Trigger Active,
│  Units Garrisoned, Variable Value, Victory Timer
└─ Effects (100) -- AI Script Goal, Acknowledge AI Signal,
   Acknowledge Multiplayer AI Signal, Activate Trigger, Add Train Location,
   Attack Move, Change Civilization Name, Change Color Mood,
   Change Diplomacy, Change Object Armor, Change Object Attack,
   Change Object Caption, Change Object Civilization Name,
   Change Object Cost, Change Object Description, Change Object HP,
   Change Object Icon, Change Object Name, Change Object Player Color,
   Change Object Player Name, Change Object Range, Change Object Speed,
   Change Object Stance, Change Object Visibility, Change Ownership,
   Change Player Color, Change Player Name, Change Research Location,
   Change Technology Cost, Change Technology Description,
   Change Technology Hotkey, Change Technology Icon, Change Technology Name,
   Change Technology Research Time, Change Train Location, Change Variable,
   Change View, Clear Instructions, Clear Timer, Counts Units Into Variable,
   Create Decision, Create Garrisoned Object, Create Object,
   Create Object Armor, Create Object Attack, Damage Object,
   Deactivate Trigger, Declare Victory, Delete Key, Disable Object Deletion,
   Disable Object Selection, Disable Technology Stacking,
   Disable Unit Attackable, Disable Unit Targeting, Display Instructions,
   Display Timer, Enable Object Deletion, Enable Object Selection,
   Enable Technology Stacking, Enable Unit Attackable,
   Enable Unit Targeting, Enable/Disable Object, Enable/Disable Technology,
   Freeze Object, Heal Object, Initiate Research, Kill Object,
   Load Key Value, Lock Gate, Modify Attribute, Modify Attribute By Variable,
   Modify Attribute For Class, Modify Object Attribute,
   Modify Object Attribute By Variable, Modify Resource,
   Modify Resource By Variable, Modify Variable By Attribute,
   Modify Variable By Resource, Modify Variable By Variable, Patrol,
   Place Foundation, Play Sound, Remove Object, Replace Object,
   Research Local Technology, Research Technology, Script Call, Send Chat,
   Set Building Gather Point, Set Object Cost (deprecated),
   Set Player Visibility, Stop Object, Store Key Value, Task Object,
   Teleport Object, Train Unit, Tribute, Unload, Unlock Gate,
   Use Advanced Buttons (deprecated)

Useful Hotkeys
├─ General: scroll cycles values, letters cycle placement menus, 1-8 select
│  player, Ctrl+Shift+F# switch control (testing), Ctrl+G cycle grid/stack
├─ File: Ctrl+Q/S/L/N (quit/save/load/new)
├─ View: Ctrl+A toggle hitboxes
├─ Tabs: F1-F10 direct select; Ctrl+W/R/T/U/O/P/D/C/V/M per tab
└─ Testing: Ctrl+Space test scenario
```

## 1. Map

- **Map Style**: Blank Map (single terrain throughout), Random Map (RMS-generated), Seed Map (RMS with a fixed seed for reproducible generation).
- **AI Map Type**: classifies the map for AI strategy selection (e.g. water maps trigger water-build strategies).
- **Colour Mood**: cosmetic map color/lighting theme; default "Empty" (no theme).
- **Team Positions**: for RMS maps — same-team players spawn in player-number order, or randomly distributed.
- **Extend Map**: checkbox + New Map Size + Map Offset (N/S/E/W) to expand an existing map.
- **Script Filename**: names an XS script file used by the map.

## 2. Global Victory

- **Standard**: win by defeating all enemies, holding 5 relics for 200 years, or holding a wonder for 200 years.
- **Conquest**: win only by defeating all enemy players.
- **Score**: win by reaching a target score.
- **Time Limit**: game runs until time expires; highest score wins.
- **Custom Victory**: independently toggleable conditions — Conquest, Exploration (reach X% map explored), Relics (capture X relics, no hold-time needed), plus an Any One / All switch for how many of the enabled conditions must be met.

## 3. Terrain

- **Map** (painting): Brush Size, Layering Mode (blend a second terrain over the base), Eye Candy (auto-spawn decorative units), No Waves on Shore, Water Definitions (preset visual styles: Preset_Main, Preset_FE1, Preset_FE2, Preset_WickedWitch), Beach Type.
- **Elevation**: raise/lower tile height, with Brush Size.
- **Cliffs**: place Granite or Sandstone cliff objects (visual variants of the same gaia object; rotatable).
- **Map Copy**: brush-based copy/paste of map regions with rotate/flip, plus a Change Player option to reassign copied units' owner on paste.
- **Erase**: selective bulk removal by category — Buildings, Gaia Objects, Invisible Units, Layered Terrain, Trees, Units.

## 4. Options

- **Point of View**: capture a player's starting camera view ("set view"), verify with "go to view".
- **Testing Difficulty**: difficulty used when test-playing the scenario.
- **Full Tech Tree**: enable full tech tree for all players (does *not* disable civ bonuses, unlike the normal-game version of this option).
- **Disable Objects**: move Buildings/Units/Techs between a Full List and Disabled List; the guide notes triggers are the better way to do this.
- **Collide & Correcting**: stationary units auto-step aside for moving units passing through.
- **Villager Force Drop**: villagers lose carried resources the instant their task changes, rather than only once they start the new task.
- **Block Humanity Team Change**: locks teams for human players only.

## 5. Players

Per-player settings (14 total): Number of Players, Starting Age, Colour, Starting Resources (Food/Wood/Stone/Gold), Pop Limit (max 500), Base Priority (guide notes it appears to have no observed effect), Tribe Name (AI-only display name), Name String ID (built-in name reference, can auto-set Tribe Name), Personality (AI script; `E3-p2.ai` simulates an AFK player), Player Type (no gameplay effect beyond whether "Either" auto-fills lobby slots with AI), Civilization, Lock Civ, Architecture (visual set borrowed from another civ), Swap Players (swap two players' settings).

## 6. Messages

Six text fields: Scenario Instructions, Objectives, Hints, Scout — each populates the matching tab of the in-game Objectives Panel; Victory / Defeat — text shown before the stats screen on win/loss.

## 7. Units

- **Placing**: four category menus (Units, Buildings, Heroes, Others); default random rotation, scroll to adjust before placing; selecting the Gaia player exposes hidden objects (trees, mines); some objects are only reachable via trigger `Create Object` or the Advanced Genie Editor.
- **Deleting**: single-unit Delete, or bulk via Terrain tab's Erase → Units.
- **Moving**: reposition placed units.
- **Rotating**: click = clockwise, right-click = counter-clockwise.
- **Selecting**: shows HP/Attack/Armour and garrisoned contents; "garrison delete" removes units garrisoned inside another unit.
- **Converting**: reassign units' owner — checkboxes pick source player(s), a dropdown picks the destination player, brush size sets the area.

## 8. Cinematics

Configures pre/post-scenario movie clips. Mainly a campaign feature; the guide notes this is buggy when used in standalone scenarios.

## 9. Diplomacy

- **Diplomacy Stance**: a directional 8x8 grid, one row per player's stance
  toward every other player (stance[i][j] need not equal stance[j][i], and
  the file measurably distinguishes both directions rather than storing one
  shared value per pair). Each cell is one of three states -- Ally, Neutral,
  Enemy -- not a checkbox: on-disk sampling across a large scenario corpus
  found Neutral in real use on hundreds of cells, so a two-state reading
  would misrepresent files that use it. There is no GAIA row or column; only
  players 1-8 are addressable.
- **Lock Teams**: prevents in-game team changes by players (triggers can still change them).
- **Players Choose Teams**: disabling removes team choice from the lobby (teams can still change in-game unless Lock Teams is also on).
- **Random Start Points**: guide notes it appears to have no observed effect.

## 10. Triggers

The most powerful part of the editor: each trigger has **Effects** (actions it performs) and **Conditions** (checks that gate whether it fires) — a trigger only runs its effects once every one of its conditions is true.

**Per-trigger fields**: Enabled (active at game start), Looping (effects re-run once/second vs. fire once), Description (shown in the Objectives panel), Display as objective, Short Description (on-screen objective text), Display on Screen, Make Header (larger header text, combined with the display flags), Mute Objectives (suppress completion notifications), Display Order.

**Editor UI**: two left-side panels — an upper trigger-list panel (New / Copy / Delete / Info) and a lower conditions-and-effects panel (New Effect / New Condition / Delete / Copy).

### Conditions (40)

`AI Signal`, `AI Signal Multiplayer`, `Accumulate Attribute`, `And`, `Bring Object To Area`, `Bring Object To Object`, `Building Is Trading`, `Capture Object`, `Chance`, `Compare Variables`, `Decision Triggered`, `Destroy Object`, `Difficulty Level`, `Diplomacy State`, `Display Timer Triggered`, `Hero Power Cast`, `Local Tech Researched`, `Object Attacked`, `Object HP`, `Object Has Action`, `Object Has Target`, `Object Not Visible`, `Object Selected`, `Object Selected Multiplayer`, `Object Visible`, `Object Visible Multiplayer`, `Objects In Area`, `Or`, `Own Fewer Objects`, `Own Objects`, `Player Defeated`, `Research Technology`, `Researching Tech`, `Script Call`, `Technology State`, `Timer`, `Trigger Active`, `Units Garrisoned`, `Variable Value`, `Victory Timer`.

(See [the guide's Conditions page](https://ugc.aoe2.rocks/scenarios/triggers/conditions/conditions/) for the one-line description of each.)

### Effects (100)

`AI Script Goal`, `Acknowledge AI Signal`, `Acknowledge Multiplayer AI Signal`, `Activate Trigger`, `Add Train Location`, `Attack Move`, `Change Civilization Name`, `Change Color Mood`, `Change Diplomacy`, `Change Object Armor`, `Change Object Attack`, `Change Object Caption`, `Change Object Civilization Name`, `Change Object Cost`, `Change Object Description`, `Change Object HP`, `Change Object Icon`, `Change Object Name`, `Change Object Player Color`, `Change Object Player Name`, `Change Object Range`, `Change Object Speed`, `Change Object Stance`, `Change Object Visibility`, `Change Ownership`, `Change Player Color`, `Change Player Name`, `Change Research Location`, `Change Technology Cost`, `Change Technology Description`, `Change Technology Hotkey`, `Change Technology Icon`, `Change Technology Name`, `Change Technology Research Time`, `Change Train Location`, `Change Variable`, `Change View`, `Clear Instructions`, `Clear Timer`, `Counts Units Into Variable`, `Create Decision`, `Create Garrisoned Object`, `Create Object`, `Create Object Armor`, `Create Object Attack`, `Damage Object`, `Deactivate Trigger`, `Declare Victory`, `Delete Key`, `Disable Object Deletion`, `Disable Object Selection`, `Disable Technology Stacking`, `Disable Unit Attackable`, `Disable Unit Targeting`, `Display Instructions`, `Display Timer`, `Enable Object Deletion`, `Enable Object Selection`, `Enable Technology Stacking`, `Enable Unit Attackable`, `Enable Unit Targeting`, `Enable/Disable Object`, `Enable/Disable Technology`, `Freeze Object`, `Heal Object`, `Initiate Research`, `Kill Object`, `Load Key Value`, `Lock Gate`, `Modify Attribute`, `Modify Attribute By Variable`, `Modify Attribute For Class`, `Modify Object Attribute`, `Modify Object Attribute By Variable`, `Modify Resource`, `Modify Resource By Variable`, `Modify Variable By Attribute`, `Modify Variable By Resource`, `Modify Variable By Variable`, `Patrol`, `Place Foundation`, `Play Sound`, `Remove Object`, `Replace Object`, `Research Local Technology`, `Research Technology`, `Script Call`, `Send Chat`, `Set Building Gather Point`, `Set Object Cost` *(deprecated)*, `Set Player Visibility`, `Stop Object`, `Store Key Value`, `Task Object`, `Teleport Object`, `Train Unit`, `Tribute`, `Unload`, `Unlock Gate`, `Use Advanced Buttons` *(deprecated)*.

(See [the guide's Effects page](https://ugc.aoe2.rocks/scenarios/triggers/effects/effects/) for the one-line description of each.)

## 11. Useful Hotkeys

General: scroll-wheel cycles dropdown values; letter keys cycle object-placement menus; 1-8 switches players; Ctrl+Shift+F# switches player control while testing; scrolling during placement rotates the object; Ctrl+G cycles 4 grid/stacking placement modes.
File: Ctrl+Q quit, Ctrl+S save, Ctrl+L load, Ctrl+N new scenario.
View: Ctrl+A toggle hitboxes.
Tabs: F1-F10 select tabs directly; Ctrl+W Messages, Ctrl+R Triggers, Ctrl+T Terrain, Ctrl+U Units, Ctrl+O Options, Ctrl+P Players, Ctrl+D Diplomacy, Ctrl+C Cinematics, Ctrl+V Global Victory, Ctrl+M Map.
Testing: Ctrl+Space test scenario.
