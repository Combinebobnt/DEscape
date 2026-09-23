"""Which units a render (or, from phase 3 on, a pick) is allowed to see.

A leaf module by construction: it imports terrain_palette and unit_kind
(both leaves themselves, stdlib-only) and nothing else from this package, so
render.py can import it with no cycle risk, and unit_pick.py can import both.

Filtering exists because of scale, not taste. This project's example
scenarios reach ~10,871 units in a single file, ~9,988 of them GAIA clutter
-- overwhelmingly trees. At that density "find the unit you meant" is not a
hit-testing problem, it's a visibility problem, so the same filter object
drives both what gets rendered and what can be picked. A unit hidden by a
filter is not pickable either; there is deliberately no way for the two to
disagree.
"""

from __future__ import annotations

from dataclasses import dataclass

from descape import unit_kind
from descape.terrain_palette import TREE_UNIT_IDS

# scenario.unit_manager.units is indexed by player, and index 0 is GAIA (see
# scenario_io.LoadedScenario's number_of_unit_sections: 9 == GAIA + 8
# players, and PLAYER_COLORS' own index-0 entry).
GAIA_PLAYER_ID = 0

# GAIA plus 8 real players. The Filters menu offers a checkbox per real
# player; GAIA gets its own separate entry (see UnitFilter.matches).
MAX_PLAYER_ID = 8


@dataclass(frozen=True)
class UnitFilter:
    """An immutable "show these units" predicate, defaulting to showing
    everything.

    Frozen so it can be stored on a chunk cache and compared by value: the
    caches key correctness on "did the filter change", and a mutable filter
    edited in place would silently skip the invalidation that makes a filter
    change visible (units are baked into cached chunk pixels -- see
    render_cache._ChunkCacheBase.set_unit_filter).

    show_trees is deliberately separate from show_gaia rather than folded
    into it. Trees are the bulk of the GAIA object count, but cliffs, gold
    and stone are GAIA too and are usually the thing you're looking *for*,
    so collapsing the two into one toggle would make the useful case
    unreachable.

    show_walls and show_eye_candy (GH #65) are const gates rather than owner
    gates for the same reason show_trees is: a wall or a doodad belongs to
    whoever placed it, and both are commonly owned by a real player, so an
    owner gate would answer the wrong question. Their const sets are derived
    from the committed .dat tables by unit_kind, not listed here -- see that
    module for what each covers and why.

    show_invisible (GH #53) is the fourth const gate: objects the .dat gives
    no standing graphic (Invisible Object A-E, Map Revealers, Blockers), which
    the game never draws and DEscape otherwise shows as a coloured box.

    show_garrisoned (GH #42) is the one per-unit gate rather than a const or
    owner one: a unit whose garrisoned_in_id names another unit is inside that
    unit in game, and the file stores it at the host's own point, so with it
    on a tower's five occupants draw stacked on the tower. The dataclass
    default stays True like every other field (an unfiltered UnitFilter() has
    to keep hiding nothing, for the byte-identity gate below); it is the
    viewer's Filters menu that ships this one unchecked.
    """

    show_gaia: bool = True
    show_trees: bool = True
    show_walls: bool = True
    show_eye_candy: bool = True
    show_invisible: bool = True
    show_garrisoned: bool = True
    # None means every player, which is NOT the same as frozenset(range(9)):
    # a scenario may have fewer players, and None avoids having to know how
    # many before building a default filter.
    players: frozenset[int] | None = None

    def matches(self, player_id: int, unit) -> bool:
        """Whether this unit, owned by player_id, should be drawn/picked.

        Seven independent gates, ANDed. Order between them doesn't matter
        (they never disagree about a unit, only about why it's hidden), but
        which field governs which gate does:

        - Trees are gated by show_trees regardless of owner, matching
          _unit_color()'s own "dark green regardless of owner" rule. A tree
          assigned to a real player is still a tree.
        - Walls (and gates), eye candy and invisible objects are gated the same owner-blind way,
          which is why the four const gates all sit ahead of the owner ones.
        - A garrisoned unit is gated by show_garrisoned regardless of owner
          or kind, and -1 (no host) and a self-reference (legal on disk, and
          its own unit's deletion guard already excludes it) both read as
          "not garrisoned".
        - GAIA is gated by show_gaia alone.
        - players gates only the non-GAIA slots. Folding GAIA into players
          too would double-gate it, so a Filters menu offering "Show GAIA"
          and "Player 1..8" as separate controls would need the two kept in
          sync to avoid GAIA vanishing while its own checkbox stayed ticked.
        """
        if not self.show_trees and unit.unit_const in TREE_UNIT_IDS:
            return False
        if not self.show_walls and unit.unit_const in unit_kind.wall_consts():
            return False
        if not self.show_eye_candy and unit.unit_const in unit_kind.eye_candy_consts():
            return False
        if not self.show_invisible and unit.unit_const in unit_kind.invisible_consts():
            return False
        if not self.show_garrisoned:
            host_id = getattr(unit, "garrisoned_in_id", -1)
            if host_id != -1 and host_id != unit.reference_id:
                return False
        if player_id == GAIA_PLAYER_ID:
            return self.show_gaia
        return self.players is None or player_id in self.players

    @property
    def is_default(self) -> bool:
        """True when this filter hides nothing.

        The byte-identity gate leans on this: a default filter must leave
        every renderer's output bit-for-bit unchanged from the pre-filter
        code, so callers can assert against it rather than against a
        hand-listed set of field values.
        """
        return (
            self.show_gaia
            and self.show_trees
            and self.show_walls
            and self.show_eye_candy
            and self.show_invisible
            and self.show_garrisoned
            and self.players is None
        )
