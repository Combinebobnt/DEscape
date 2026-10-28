# Players

Players mode edits each player's own settings: name, civilization, starting age and resources, colour, AI and starting camera. Pick **Players** in the toolbar's Mode list.

**Number of players** at the top sets how many players the scenario has, from 2 to 8. Lowering it deactivates the highest-numbered players and leaves their other settings as they were, and Diplomacy's stance list follows it. It can be read-only on a file whose other player settings are editable; its tooltip says why.

The selector below it picks the player to edit: GAIA first, then Players 1 to 8. The Select Player commands pick one from the keyboard; [Keyboard and Mouse](keyboard-and-mouse.md) lists their keys. Each change is one undo step. A setting this file's scenario version doesn't store isn't listed, and a greyed row's tooltip says why it is read-only.

## Identity

- **Tribe name**. Every player selector in DEscape shows it after the player's number, as in "P8 - Horde Army", so a tribe name set here labels that player everywhere.
- **Name string ID** names the player with a game string instead; -2 means none.
- **Civilization** and **Architecture**. Architecture is the building style, picked by the civilization it comes from.
- **Lock civilization** and **Lock personality** are the in-game editor's two lock checkboxes.

## Start

- **Starting age**. A Return of Rome scenario lists that game's ages.
- **Starting food**, **Starting wood**, **Starting stone** and **Starting gold**.
- **Population limit**, which can't go below 0.
- **Colour**.

For Players 1 to 8, the four resources and the population limit stop at 16,777,216, the most the file's second copy of each can hold exactly.

## AI

- **Player type** is Either or Computer. Either lets a human or the computer take the slot; the game's Human stores the same value, so a Human slot shows as Either. Computer is always the AI.
- **Personality** is the AI script a computer player runs: Standard, None, or a custom AI from the game's ai folder, your profile or your mods. DEscape finds your profile through the folder set in **Settings > Mods**. A custom AI's tooltip names the folder it came from, and one DEscape can't use is greyed with the reason. A stored personality that matches none of these stays listed as stored until you pick another. Picking a custom AI stores its script in the scenario the way the in-game editor does.
- **AI type** shows Custom, Standard or None and follows Personality; you set it by picking a personality.
- **Base priority**, from 0 to 255.

## Point of View

**Initial camera X** and **Initial camera Y** are the tile the player's camera starts on, or -1 for unset. The four buttons under them:

- **Set View** uses the tile at the centre of the map view.
- **Set to Centre** uses the middle of the map.
- **Go to View** scrolls the map to the player's camera, also on a read-only file.
- **Reset View** clears it back to unset.

Each sets both coordinates in one undo step. **View > Show Player Cameras** marks every player's starting camera on the map in their colour, and in Players mode the selected player's mark stands out.

## Disabled objects and GAIA

**Disabled Objects…** at the bottom of the panel opens the list of buildings, units and technologies each player can't use, the same dialog as **Map > Disabled Objects…** ([Map Tools](map-tools.md) covers it).

GAIA has only Civilization, Architecture, the two locks, Starting age, the four resources, Colour and Population limit, the rows the in-game editor shows for it.

Where this is in the in-game editor: the Players tab, with the starting camera under the Options tab's Point of View.

DEscape has no Swap Players. Swap two players' settings in the in-game editor.
