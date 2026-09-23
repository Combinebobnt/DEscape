"""Qt-free specs for the Map Options panel's fields -- default-tier testable
without a QApplication, and it keeps the panel itself dumb. Mirrors
descape/trigger_fields.py's shape and reasoning.

Each spec's field scope and the reasoning behind what is excluded is
recorded beside the spec itself, so this module stands on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

from descape.scenario_io import LoadedScenario, retriever_length

CHECKBOX = "checkbox"
COMBO = "combo"
SPINBOX = "spinbox"


@dataclass(frozen=True)
class OptionFieldSpec:
    field_id: str  # stable id, independent of label text
    label: str
    group: str  # panel group box title, e.g. "Global Victory"
    section: str  # scenario section name, e.g. "GlobalVictory"
    retriever: str  # retriever name within that section, e.g. "mode"
    kind: str  # CHECKBOX / COMBO / SPINBOX
    choices: tuple[tuple[int, str], ...] = ()  # (value, label) pairs, COMBO only
    minimum: int | None = None  # SPINBOX bounds
    maximum: int | None = None
    scale: float = 1.0  # display_value = raw_value / scale, e.g. 10 for years
    tooltip: str = ""
    sentinel: int | None = None  # a raw value that means "unset"
    sentinel_display: int | None = None  # what to show/start editing from instead
    # Which left-page panel renders this row -- "map_options" (default) or
    # "diplomacy". Independent of `section`, which names the scenario
    # section this field's bytes live in (the four Diplomacy-group specs
    # below stay section="Diplomacy" -- that is where options_model.py
    # walks for their offsets -- even though panel="diplomacy" moves their
    # UI row to DiplomacyPanel, step 5). Each panel filters specs_for() to
    # its own value.
    panel: str = "map_options"


_SPECS: tuple[OptionFieldSpec, ...] = (
    # -- Global Victory -------------------------------------------------------
    # Choices are hardcoded literals, matching the secondary_game_modes
    # precedent below -- this module imports no library enums. They transcribe
    # AoE2ScenarioParser.datasets.trigger_lists.VictoryCondition verbatim,
    # including its gap: there is no member 5. An out-of-enum value is handled
    # faithfully by the combo branch's "unknown (N)" row.
    OptionFieldSpec(
        "victory_condition", "Victory condition", "Global Victory",
        "GlobalVictory", "mode", COMBO,
        tooltip=(
            "How the scenario is won. Standard: conquest, relics or a wonder. "
            "Conquest: defeat all enemies. Score: reach the score below. Time "
            "limit: highest score when time runs out. Custom: the custom "
            "conditions below. Secondary game mode: the mode chosen under Map."
        ),
        choices=(
            (0, "Standard"), (1, "Conquest"), (2, "Score"),
            (3, "Time limit"), (4, "Custom"), (6, "Secondary game mode"),
        ),
    ),
    OptionFieldSpec(
        "victory_score", "Score to win", "Global Victory",
        "GlobalVictory", "required_score_for_score_victory", SPINBOX,
        tooltip="Score victory only: the score a player must reach to win.",
        # 0xFFFFFFFF (five corpus files) means "no explicit score was ever
        # chosen" -- the real editor pre-fills 14000, an ordinary editable
        # number, the first time Score victory is selected. sentinel_display
        # mirrors that default rather than leaving the row as-stored.
        minimum=0, maximum=9_999_999, sentinel=0xFFFFFFFF, sentinel_display=14000,
    ),
    OptionFieldSpec(
        "victory_years", "Time limit, years", "Global Victory",
        "GlobalVictory", "time_for_timed_game_in_10ths_of_a_year", SPINBOX,
        tooltip=(
            "Time limit victory only: how many in-game years the game lasts. The "
            "highest score at the end wins."
        ),
        # Raw units (10ths), divided by scale for display. 100_000, not
        # 999_999: the scale branch's floor/round agreement below was only
        # measured over raw 0..100000, and going wider would leave the
        # reachable domain outside that verified window.
        minimum=0, maximum=100_000, scale=10.0,
    ),

    # -- Global Victory (custom-victory fields) ------------------------------
    # These four only apply when victory_condition is Custom. Always listed,
    # greyed off Custom by the panel: 2_Joan_coop_2_v0_15 stores
    # all_custom_conditions_required=1 under Conquest, so hiding them would
    # conceal a real stored value. Labels follow the in-game Custom Victory
    # menu (docs/INGAME_EDITOR_REFERENCE.md). The file stores only amounts,
    # no separate Exploration/Relics enable toggle: 0 means off.
    OptionFieldSpec(
        "custom_conquest", "Conquest", "Global Victory",
        "GlobalVictory", "conquest_required", CHECKBOX,
        tooltip="Custom victory: win by defeating all enemy players.",
    ),
    OptionFieldSpec(
        "custom_explored_percent", "Exploration, % of map", "Global Victory",
        "GlobalVictory", "explored_percent_of_map_required", SPINBOX,
        tooltip=(
            "Custom victory: the percentage of the map a player must explore. 0 "
            "turns this condition off."
        ),
        # A percentage, so >100 is meaningless and falls to as-stored.
        minimum=0, maximum=100,
    ),
    OptionFieldSpec(
        "custom_relics", "Relics", "Global Victory",
        "GlobalVictory", "artifacts_required", SPINBOX,
        tooltip=(
            "Custom victory: how many relics a player must capture, with no hold "
            "time. 0 turns this condition off."
        ),
        # A count, not a flag: C2_ElCid_coop_1/_3 store 20. A UI bound on a
        # u32; anything above renders as-stored.
        minimum=0, maximum=9999,
    ),
    OptionFieldSpec(
        "custom_all_conditions", "Conditions needed", "Global Victory",
        "GlobalVictory", "all_custom_conditions_required", COMBO,
        tooltip=(
            "Custom victory: whether meeting any one enabled condition wins, or all"
            " of them are needed."
        ),
        # The in-game Any One / All switch, not a bare flag.
        choices=((0, "Any one"), (1, "All")),
    ),

    # -- Teams (rendered on the Diplomacy panel, not Map Options -- see
    # OptionFieldSpec.panel's docstring; `section` stays "Diplomacy" since
    # that is the real scenario section these four scalars live in) --------
    OptionFieldSpec(
        "lock_teams", "Lock teams", "Teams",
        "Diplomacy", "lock_teams", CHECKBOX,
        tooltip="Players cannot change teams in-game. Triggers can still change them.",
        panel="diplomacy",
    ),
    OptionFieldSpec(
        "allow_players_choose_teams", "Players choose teams", "Teams",
        "Diplomacy", "allow_players_choose_teams", CHECKBOX,
        tooltip=(
            "Players can pick their team in the lobby. Off removes that choice; "
            "teams can still change in-game unless Lock teams is on."
        ),
        panel="diplomacy",
    ),
    OptionFieldSpec(
        "random_start_points", "Random start points", "Teams",
        "Diplomacy", "random_start_points", CHECKBOX,
        tooltip="No observed effect in the in-game editor.",
        panel="diplomacy",
    ),
    OptionFieldSpec(
        "max_number_of_teams", "Max number of teams", "Teams",
        "Diplomacy", "max_number_of_teams", SPINBOX, minimum=0, maximum=8,
        tooltip="The most teams the scenario allows.",
        panel="diplomacy",
    ),

    # -- Map --------------------------------------------------------------
    OptionFieldSpec(
        "collide_and_correct", "Collide and correcting", "Map",
        "Map", "collide_and_correct", CHECKBOX,
        tooltip="Stationary units step aside for moving units passing through.",
    ),
    OptionFieldSpec(
        "villager_force_drop", "Villager force drop", "Map",
        "Map", "villager_force_drop", CHECKBOX,
        tooltip=(
            "Villagers drop carried resources the moment their task changes, not "
            "only once they start the new task."
        ),
        # Stays as-stored on the six 1.41 C2_ElCid_coop_* files (90/119/167/
        # 255): the byte was never a flag there, so a checkbox would overwrite
        # different semantics. _is_representable() is what enforces that.
    ),
    OptionFieldSpec(
        "lock_coop_alliances", "Block humanity team change", "Map",
        "Map", "lock_coop_alliances", CHECKBOX,
        tooltip=(
            "Locks co-op alliances: human players cannot change their team or "
            "diplomacy with each other in-game."
        ),
    ),
    OptionFieldSpec(
        "secondary_game_modes", "Secondary game mode", "Map",
        "Map", "secondary_game_modes", COMBO,
        tooltip=(
            "A game mode played on top of the victory condition: Empire Wars, "
            "Sudden Death, Regicide or King of the Hill."
        ),
        # Modelled as bit flags in AoE2ScenarioParser, but every value across
        # all 20 example files is 0 -- no corpus evidence the game combines
        # them, so this is a conservative single-select rather than four
        # checkboxes.
        choices=(
            (0, "None"), (1, "Empire Wars"), (2, "Sudden Death"),
            (4, "Regicide"), (8, "King of the Hill"),
        ),
    ),
    OptionFieldSpec(
        "no_waves_on_shore", "No waves on shore", "Map",
        "Map", "no_waves_on_shore", CHECKBOX,
        tooltip="Hides the animated waves along shorelines.",
    ),

    # -- Options ------------------------------------------------------------
    OptionFieldSpec(
        "all_techs", "Full tech tree", "Options",
        "Options", "all_techs", CHECKBOX,
        tooltip=(
            "Every player gets the full tech tree. Unlike the lobby option, "
            "civilization bonuses stay."
        ),
    ),
    OptionFieldSpec(
        "ai_map_type", "AI map type", "Options",
        "Options", "ai_map_type", SPINBOX, minimum=-2_147_483_648, maximum=2_147_483_647,
        tooltip=(
            "Tells the AI what kind of map this is, e.g. a water map makes it build"
            " a navy. Stored as the raw map type number."
        ),
    ),

    # -- Triggers -------------------------------------------------------------
    # legacy_exec_order's read/write path is TriggerEditModel, not
    # options_model.py: it lives inside the Triggers region, so it is emitted
    # by that model's serialize() tail rather than byte-patched. It is listed
    # here anyway so the panel renders and gates it the same way as every
    # other row; options_model.field_offsets() explicitly skips this section.
    OptionFieldSpec(
        "legacy_exec_order", "Trigger execution order", "Triggers",
        "Triggers", "legacy_exec_order", COMBO,
        tooltip=(
            "The order triggers run in. Display order follows the trigger list; "
            "Legacy runs them by trigger ID, as older game versions did."
        ),
        choices=((0, "Display order"), (1, "Legacy - trigger ID order")),
    ),
)


def specs_for(loaded: LoadedScenario) -> tuple[OptionFieldSpec, ...]:
    """Every OptionFieldSpec whose retriever is actually present and
    non-empty on `loaded`.

    A retriever can be missing outright (its section not yet loaded --
    "Triggers" until parse_triggers() has run) or present but zero bytes
    long (e.g. legacy_exec_order on a file whose trigger_version is below
    4.5, or secondary_game_modes on a pre-1.42 file -- its SET_REPEAT eval
    depends on runtime values, not just the section being in scope).
    Presence is read from the already-loaded data, never inferred from
    scenario_version. F7_3_York is the file that settles this: scenario
    version 1.55, but its trigger version is below 4.5, so legacy_exec_order
    consumes zero bytes there despite the version implying otherwise.
    """
    out = []
    for spec in _SPECS:
        section = loaded._scenario.sections.get(spec.section)
        if section is None:
            continue
        retriever = section.retriever_map.get(spec.retriever)
        if retriever is None:
            continue
        if retriever_length(retriever) == 0:
            continue
        out.append(spec)
    return tuple(out)


def current_value(loaded: LoadedScenario, spec: OptionFieldSpec) -> int:
    """The field's current raw value, as parsed -- callers apply `scale`
    themselves (e.g. victory_years divides by 10 for display). Only valid
    for a spec that specs_for(loaded) actually returned.

    Every spec in _SPECS today resolves to a plain int, because none of them
    point at a repeat-driven retriever -- but two that would (
    per_player_starting_age, per_player_population_cap) already sit in the
    sections this module reads, and belong to a future Players mode. Raises
    rather than silently returning a list, so a spec added later without
    updating this function fails immediately instead of reaching a
    QSpinBox/QComboBox in the panel with a list. See
    test_option_fields.py's test_every_spec_resolves_to_a_scalar_int.
    """
    value = loaded._scenario.sections[spec.section].retriever_map[spec.retriever].data
    if not isinstance(value, int):
        raise TypeError(
            f"{spec.field_id!r} resolved to a {type(value).__name__}, not an int -- "
            f"current_value() only supports scalar retrievers"
        )
    return value
