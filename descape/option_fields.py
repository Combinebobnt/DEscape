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
        choices=(
            (0, "Standard"), (1, "Conquest"), (2, "Score"),
            (3, "Time limit"), (4, "Custom"), (6, "Secondary game mode"),
        ),
    ),
    OptionFieldSpec(
        "victory_score", "Score to win", "Global Victory",
        "GlobalVictory", "required_score_for_score_victory", SPINBOX,
        # 0xFFFFFFFF (five corpus files) means "no explicit score was ever
        # chosen" -- the real editor pre-fills 14000, an ordinary editable
        # number, the first time Score victory is selected. sentinel_display
        # mirrors that default rather than leaving the row as-stored.
        minimum=0, maximum=9_999_999, sentinel=0xFFFFFFFF, sentinel_display=14000,
    ),
    OptionFieldSpec(
        "victory_years", "Time limit, years", "Global Victory",
        "GlobalVictory", "time_for_timed_game_in_10ths_of_a_year", SPINBOX,
        # Raw units (10ths), divided by scale for display. 100_000, not
        # 999_999: the scale branch's floor/round agreement below was only
        # measured over raw 0..100000, and going wider would leave the
        # reachable domain outside that verified window.
        minimum=0, maximum=100_000, scale=10.0,
    ),

    # -- Global Victory (custom-victory fields): DEFERRED --------------------
    # conquest_required, artifacts_required, explored_percent_of_map_required
    # and all_custom_conditions_required only apply when victory_condition is
    # Custom, so unlike every other group in this panel they want show/hide
    # behaviour rather than a flat form of always-on rows -- a design question,
    # not a row-set addition.

    # -- Teams (rendered on the Diplomacy panel, not Map Options -- see
    # OptionFieldSpec.panel's docstring; `section` stays "Diplomacy" since
    # that is the real scenario section these four scalars live in) --------
    OptionFieldSpec(
        "lock_teams", "Lock teams", "Teams",
        "Diplomacy", "lock_teams", CHECKBOX,
        panel="diplomacy",
    ),
    OptionFieldSpec(
        "allow_players_choose_teams", "Players choose teams", "Teams",
        "Diplomacy", "allow_players_choose_teams", CHECKBOX,
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
        panel="diplomacy",
    ),

    # -- Map --------------------------------------------------------------
    OptionFieldSpec(
        "collide_and_correct", "Collide and correcting", "Map",
        "Map", "collide_and_correct", CHECKBOX,
    ),
    OptionFieldSpec(
        "villager_force_drop", "Villager force drop", "Map",
        "Map", "villager_force_drop", CHECKBOX,
    ),
    OptionFieldSpec(
        "lock_coop_alliances", "Block humanity team change", "Map",
        "Map", "lock_coop_alliances", CHECKBOX,
    ),
    OptionFieldSpec(
        "secondary_game_modes", "Secondary game mode", "Map",
        "Map", "secondary_game_modes", COMBO,
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
    ),

    # -- Options ------------------------------------------------------------
    OptionFieldSpec(
        "all_techs", "Full tech tree", "Options",
        "Options", "all_techs", CHECKBOX,
    ),
    OptionFieldSpec(
        "ai_map_type", "AI map type", "Options",
        "Options", "ai_map_type", SPINBOX, minimum=-2_147_483_648, maximum=2_147_483_647,
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
