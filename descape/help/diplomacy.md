# Diplomacy

Diplomacy mode edits how the players stand toward each other and how teams work. Pick **Diplomacy** in the toolbar's Mode list.

The selector at the top picks a player. It lists only the players the scenario defines, so it follows **Number of players** in [Players](players.md), and GAIA has no stances to set. The Select Player commands pick a player from the keyboard here too; [Keyboard and Mouse](keyboard-and-mouse.md) lists their keys. Each change is one undo step.

## Stances

Each **Toward Player N** row sets the selected player's stance toward that player: **Ally**, **Neutral** or **Enemy**. A stance runs one way only. Player 2's stance back toward Player 1 is set on Player 2's page, and the two can differ.

**Allied victory** lets the selected player win together with its allies.

A stance the file stores as something other than Ally, Neutral or Enemy shows as "stored value" and its number, and is kept as it is rather than snapped to one of the three. A player's stance toward itself isn't a real setting and has no row, but if the file stores an unusual one, the line at the top says so and DEscape keeps it.

## Teams

These four settings are scenario-wide, so they read the same whichever player is selected.

- **Lock teams**: players can't change teams in the game. Triggers can still change them.
- **Players choose teams**: players can pick their team in the lobby. Off removes that choice; teams can still change in the game unless Lock teams is on.
- **Random start points**: no observed effect in the in-game editor.
- **Max number of teams**: the most teams the scenario allows, up to 8.

A greyed row's tooltip says why it can't be edited for this file.

Where this is in the in-game editor: the Diplomacy tab.
