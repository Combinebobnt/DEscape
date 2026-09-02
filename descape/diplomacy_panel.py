"""DiplomacyPanel: the Diplomacy mode's left page -- per-player stances and
allied-victory flags, writable through the same OptionsEditModel Map Options
and Players mode use.

Structurally closest to PlayersPanel: a player selector above a form, and
the selector is rebuilt (and the selection reset to the file's first defined
player) only when `show_scenario()` is handed a genuinely different document
-- see that method's own docstring for why, the same reasoning
PlayersPanel.show_scenario() gives. Unlike PlayersPanel, the selector here
only lists the players this file actually defines
(diplomacy_fields.defined_player_ids()), not a fixed P1..P8: a stance
toward or from an undefined player has no in-game meaning to edit, even
though the byte grid stores a slot for it. Driven off that list directly
rather than range(1, count + 1): the active set is a contiguous prefix in
every file measured so far, but nothing here should silently mislabel a
row if a sparse one ever turns up.

Each opponent row is a QButtonGroup of three QRadioButtons, one per
DiplomacyState member -- reusing trigger_fields.enum_choices("DiplomacyState")
for the labels rather than redefining "Ally"/"Neutral"/"Enemy" here, since
that mapping (ALLY=0, NEUTRAL=1, ENEMY=3, with no member 2) is already wired
into the trigger UI. The self cell (stance[i][i]) is never shown as a row --
a player's relationship to itself is not a real in-game setting -- but it is
real per-file data (some files store it as something other than the default),
so a self-stance other than the default (Enemy) is surfaced as a status note
instead of silently dropped. It is also never editable, for the same reason.

Cell ids (stance:row:col, allied_victory:player) are the field ids this
panel reports through its `on_diplomacy_field` callback and expects in
`editable_fields`/`read_only_reasons`/`pending_values` -- see
diplomacy_fields.stance_cell_id()/allied_victory_cell_id(). They already
encode both players, so unlike PlayersPanel's `pending_values` (keyed by
bare field id, nested per player) this panel's is a flat {cell_id: value}
dict. The same three dicts also carry the four Teams-group scalar rows
below (step 5), keyed by their own bare OptionFieldSpec.field_id -- there is
no collision with a cell id, since every cell id contains a ":".

The Teams group (lock_teams, allow_players_choose_teams,
random_start_points, max_number_of_teams) moved here from Map Options mode:
they are scenario-wide, not per-player, but still live inside this panel's
per-player rebuild, since
the whole host is replaced wholesale on every populate anyway (see
_rebuild_host()). They report through a second callback,
`on_option_field(spec, value)` -- the same shape MapOptionsPanel's own
callback has, wired to the same ViewerWindow.set_option_field() those rows
always used, so their proven write path and
test_every_writable_option_repacks_its_own_stored_bytes coverage stay
untouched. Representability (a stored value no editor could show
truthfully) reuses MapOptionsPanel's own `_is_representable`/
`_out_of_range_label` rather than a second copy of that int-range check --
unlike the CHECKBOX/SPINBOX construction itself, which PlayersPanel already
shows is idiomatic to keep local to each panel (different field kinds, own
edge cases).
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from descape import option_fields, trigger_fields
from descape.diplomacy_fields import (
    allied_victory_cell_id,
    allied_victory_value,
    defined_player_ids,
    stance_cell_id,
    stance_value,
)
from descape.map_options_panel import MapOptionsPanel
from descape.scenario_io import LoadedScenario
from descape.viewer_common import _make_spinbox

# (label, value) pairs for Ally/Neutral/Enemy, reused rather than redefined
# -- see module docstring. trigger_fields.enum_choices() returns the raw
# enum member names (ALLY/NEUTRAL/ENEMY), which reads fine next to a numeric
# value in the trigger UI's technical property combo but not as three radio
# labels in a row of their own; title-casing here only changes display text,
# not the {0,1,3} mapping the module docstring says must not be redefined.
_STANCE_CHOICES = tuple(
    (label.title(), value) for label, value in trigger_fields.enum_choices("DiplomacyState")
)
_STANCE_LABEL = {value: label for label, value in _STANCE_CHOICES}
_STANCE_VALUES = frozenset(value for _, value in _STANCE_CHOICES)

# The stored default for stance[i][i] across every corpus file -- not
# assumed anywhere else, only used here to decide whether the self-stance
# note is worth showing.
_DEFAULT_SELF_STANCE = 3


class DiplomacyPanel(QWidget):
    """Per-player stances and allied-victory flags (the Diplomacy mode's
    left page).

    Dumb by design, same as MapOptionsPanel/PlayersPanel: it reports "the
    user set cell X to raw value V" through the callback it is constructed
    with, and never touches an edit model itself.
    """

    # Measured against the widest row this panel actually builds -- a
    # "Toward Player N" label plus three radio buttons, wider than any row
    # MapOptionsPanel/PlayersPanel/TriggerPanel have needed so far. 420 is
    # the exact width an offscreen 8-player file's rows stop wrapping the
    # label above the field (WrapLongRows); 10 px of headroom over that
    # measured tipping point. See tests/test_diplomacy_panel.py's width
    # check.
    MIN_USEFUL_WIDTH = 430

    _NO_DOCUMENT = "No map open."
    # Fallback only -- the window always resolves a specific per-cell reason
    # (see viewer.py's _diplomacy_read_only_reasons()); this covers a panel
    # built directly, e.g. by a test, with no reasons supplied.
    _READ_ONLY_REASON = "Read-only for this file."

    def __init__(self, on_diplomacy_field=None, on_option_field=None):
        super().__init__()
        # No-op defaults so the panel stays constructible on its own, the
        # same contract MapOptionsPanel/PlayersPanel's callbacks have.
        self._on_diplomacy_field = on_diplomacy_field or (lambda *args: None)
        self._on_option_field = on_option_field or (lambda *args: None)

        self._loaded: LoadedScenario | None = None
        self._player_id = 1
        # The Teams group's specs (step 5) -- scenario-wide, not per-player,
        # but rebuilt on every populate along with everything else in the
        # host (see _rebuild_host()).
        self._option_specs: tuple[option_fields.OptionFieldSpec, ...] = ()
        # The file's active player numbers, not assumed contiguous from 1 --
        # see diplomacy_fields.defined_player_ids()'s own docstring. The
        # combo and every opponent row are driven off this list, never off
        # range(1, count + 1), so a sparse active set (were one ever found)
        # would still select and label real players rather than silently
        # resolving combo index N to the wrong player number.
        self._active_players: list[int] = []
        self._widgets: dict[str, QWidget] = {}
        # cell_id -> the raw value the currently-shown row is displaying.
        # Not derivable from the widget once editing lands, for the same
        # reason MapOptionsPanel._values isn't: a byte-patch write leaves
        # the parsed retriever holding the ORIGINAL value.
        self._values: dict[str, int] = {}
        self._editable_fields: frozenset[str] = frozenset()
        self._read_only_reasons: dict[str, str] = {}
        # {cell_id: value} across every player pair edited this session, not
        # just the one currently shown -- switching the player combo away
        # and back must not lose an edit the window was never asked to
        # repopulate for (a successful edit does not trigger a repopulate;
        # only a refused one, or an undo/redo, does).
        self._pending_values: dict[str, int] = {}
        self._populating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.status = QLabel(self._NO_DOCUMENT)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.player_combo = QComboBox()
        self.player_combo.setEnabled(False)
        self.player_combo.currentIndexChanged.connect(self._on_player_changed)
        layout.addWidget(self.player_combo)

        self.area = QScrollArea()
        self.area.setWidgetResizable(True)
        layout.addWidget(self.area, stretch=1)
        self._rebuild_host()

    def _rebuild_host(self) -> None:
        """Replace the scroll area's contents with a fresh, empty host --
        wholesale rather than clearing rows in place, since the row set
        depends on both the file (active player list) and the selected
        player (which row is skipped as "self") -- same reasoning as
        MapOptionsPanel._rebuild_host/PlayersPanel._rebuild_host."""
        self.host = QWidget()
        self.host_layout = QVBoxLayout(self.host)
        self.host_layout.setContentsMargins(6, 6, 6, 6)
        self.area.setWidget(self.host)
        self._widgets = {}

    # -- document state -------------------------------------------------

    def clear_document(self) -> None:
        self._populating = True
        try:
            self._loaded = None
            self._player_id = 1
            self._active_players = []
            self._option_specs = ()
            self._values = {}
            self._editable_fields = frozenset()
            self._read_only_reasons = {}
            self._pending_values = {}
            self.player_combo.blockSignals(True)
            self.player_combo.clear()
            self.player_combo.blockSignals(False)
            self.player_combo.setEnabled(False)
            self._rebuild_host()
            self.status.setText(self._NO_DOCUMENT)
            self.status.setToolTip("")
        finally:
            self._populating = False

    def show_scenario(
        self,
        loaded: LoadedScenario | None,
        option_specs: Sequence[option_fields.OptionFieldSpec] = (),
        editable_fields: Iterable[str] = (),
        read_only_reasons: Mapping[str, str] | None = None,
        pending_values: Mapping[str, int] | None = None,
    ) -> None:
        """Populate from `loaded`. The single repopulate path -- both
        ViewerWindow._show_diplomacy() (mode entry, a new document) and
        ._repopulate_diplomacy() (after an edit, an undo, or a refused
        edit) call only this.

        The player selector is only rebuilt -- and the selection reset to
        the file's first defined player -- when `loaded` is a genuinely
        different document (identity, not equality: the same
        LoadedScenario is passed on every repopulate of one open file), OR
        when the active player list itself changed. The second condition
        does not fire from anything in the app today (nothing writes
        DataHeader.player_data_1[].active yet -- see TODO.md's "Player
        options write path"), but it is what a test harness uses to reach
        a non-default player count by mutating the already-loaded
        scenario in place rather than depending on a corpus file (see
        tests/test_diplomacy_panel.py's module docstring) -- identity
        alone would treat that mutation as "nothing changed" and leave the
        selector showing the old count. Otherwise the selector and the
        player currently showing are left alone, so an edit on P5 does not
        silently jump the panel back to P1 -- the same rule
        PlayersPanel.show_scenario() follows and for the same reason.

        `option_specs` is the Teams group's specs, pre-filtered to
        `panel == "diplomacy"` by the window (option_fields.specs_for()
        returns every panel's specs together -- see OptionFieldSpec.panel's
        docstring). `editable_fields`/`read_only_reasons`/`pending_values`
        cover both the grid's cell ids and these specs' bare field ids in
        one set each (see module docstring for why that never collides),
        resolved by the window the same way Map Options' and Players
        mode's own gates are.
        """
        if loaded is None:
            self.clear_document()
            return

        new_active_players = defined_player_ids(loaded)
        same_document = loaded is self._loaded and new_active_players == self._active_players
        self._loaded = loaded
        self._active_players = new_active_players
        self._option_specs = tuple(option_specs)
        self._editable_fields = frozenset(editable_fields)
        self._read_only_reasons = dict(read_only_reasons or {})
        self._pending_values = dict(pending_values or {})

        if not same_document:
            self._player_id = self._active_players[0] if self._active_players else 1
            self.player_combo.blockSignals(True)
            self.player_combo.clear()
            for player_id in self._active_players:
                self.player_combo.addItem(
                    _swatch_icon(loaded.player_colors[player_id]), f"P{player_id}"
                )
            self.player_combo.setCurrentIndex(0)
            self.player_combo.blockSignals(False)
            self.player_combo.setEnabled(True)

        self._populate_current_player()

    def select_player(self, player_id: int) -> bool:
        """Backs the player_select_0..8 shortcut in Diplomacy mode. False
        for GAIA and for any player the scenario hasn't defined -- see
        _active_players's own docstring for why it's never assumed
        contiguous from 1."""
        if player_id not in self._active_players:
            return False
        self.player_combo.setCurrentIndex(self._active_players.index(player_id))
        return True

    def _on_player_changed(self, index: int) -> None:
        if index < 0:
            return
        self._player_id = self._active_players[index]
        self._populate_current_player()

    def _populate_current_player(self) -> None:
        self._populating = True
        try:
            self._rebuild_host()
            self._values = {}
            for opponent in self._active_players:
                if opponent == self._player_id:
                    continue
                cell_id = stance_cell_id(self._player_id, opponent)
                self._values[cell_id] = self._pending_values.get(
                    cell_id, stance_value(self._loaded, self._player_id, opponent)
                )
            allied_id = allied_victory_cell_id(self._player_id)
            self._values[allied_id] = self._pending_values.get(
                allied_id, allied_victory_value(self._loaded, self._player_id)
            )
            for spec in self._option_specs:
                self._values[spec.field_id] = self._pending_values.get(
                    spec.field_id, option_fields.current_value(self._loaded, spec)
                )
            self._build_stances_group()
            self._build_teams_group()
            self.host_layout.addStretch(1)
        finally:
            self._populating = False

        rows = max(len(self._active_players) - 1, 0)
        total = rows + 1 + len(self._option_specs)  # + allied-victory + Teams rows
        read_only = total - sum(1 for cid in self._values if cid in self._editable_fields)
        version = self._loaded.scenario_version
        summary = [f"{rows} stance{'' if rows == 1 else 's'} for Player {self._player_id} — scenario {version}"]
        if read_only == total and total:
            summary.append("all read-only")
        elif read_only:
            summary.append(f"{read_only} read-only")
        text = ", ".join(summary) + "."
        tooltip = (
            "A cell stored as a value no editor could show truthfully is shown "
            "as stored instead -- hover it for details."
        )

        # The self-cell note: per selected player, not per file (see module
        # docstring), and its own status term rather than routed through an
        # as-stored widget count -- there is no row for this cell to be a
        # widget of.
        self_value = stance_value(self._loaded, self._player_id, self._player_id)
        if self_value != _DEFAULT_SELF_STANCE:
            label = _STANCE_LABEL.get(self_value, f"stored value {self_value}")
            text += f" Self-stance stored as {label}."
            tooltip += (
                f" Player {self._player_id}'s own stance toward itself is stored "
                f"as {label} rather than the default (Enemy) -- kept exactly as "
                "recorded. There is no row for it here, since a player's "
                "relationship to itself is not a real in-game setting."
            )

        self.status.setText(text)
        self.status.setToolTip(tooltip)

    # -- building the form ------------------------------------------------

    def _build_stances_group(self) -> None:
        box = QGroupBox("Stances")
        form = QFormLayout(box)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        for opponent in self._active_players:
            if opponent == self._player_id:
                continue
            cell_id = stance_cell_id(self._player_id, opponent)
            value = self._values[cell_id]
            editable = cell_id in self._editable_fields
            widget = self._build_stance_row(cell_id, value, editable)
            if not widget.toolTip():
                widget.setToolTip(self._read_only_reasons.get(cell_id, self._READ_ONLY_REASON))
            self._widgets[cell_id] = widget
            form.addRow(f"Toward Player {opponent}", widget)

        allied_id = allied_victory_cell_id(self._player_id)
        allied_value = self._values[allied_id]
        allied_editable = allied_id in self._editable_fields
        allied_widget = QCheckBox()
        allied_widget.setChecked(bool(allied_value))
        allied_widget.setEnabled(allied_editable)
        if allied_editable:
            allied_widget.toggled.connect(
                lambda checked, cid=allied_id: self._changed(cid, int(checked))
            )
        else:
            allied_widget.setToolTip(self._read_only_reasons.get(allied_id, self._READ_ONLY_REASON))
        self._widgets[allied_id] = allied_widget
        form.addRow("Allied victory", allied_widget)

        self.host_layout.addWidget(box)

    def _build_teams_group(self) -> None:
        """The four scalar rows step 5 moved here from Map Options mode --
        see module docstring. Nothing here if the file does not carry the
        group (option_specs empty, e.g. an unwritable file's specs_for()
        still returns them -- specs presence is independent of the write
        gate, same as Map Options)."""
        if not self._option_specs:
            return
        box = QGroupBox(self._option_specs[0].group)
        form = QFormLayout(box)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        for spec in self._option_specs:
            value = self._values[spec.field_id]
            editable = spec.field_id in self._editable_fields
            widget = self._build_option_widget(spec, value, editable)
            tip = self._read_only_reasons.get(spec.field_id, "") or spec.tooltip
            if tip and not widget.toolTip():
                widget.setToolTip(tip)
            self._widgets[spec.field_id] = widget
            form.addRow(spec.label, widget)

        self.host_layout.addWidget(box)

    def _build_option_widget(self, spec, value: int, editable: bool) -> QWidget:
        """CHECKBOX/SPINBOX only -- the two kinds the four Teams specs
        actually use (no COMBO, no scale, no sentinel among them). Widget
        construction stays local to this panel, same as PlayersPanel's own
        copy of MapOptionsPanel's shape; only the representability check
        (`MapOptionsPanel._is_representable`/`_out_of_range_label`) is
        reused rather than duplicated, since that logic is pure int-range
        checking with no per-panel divergence.
        """
        if not MapOptionsPanel._is_representable(spec, value):
            return MapOptionsPanel._out_of_range_label(spec, value)

        if spec.kind == option_fields.CHECKBOX:
            widget = QCheckBox()
            widget.setChecked(bool(value))
            widget.setEnabled(editable)
            widget.toggled.connect(
                lambda checked, s=spec: self._option_changed(s, int(checked))
            )
            return widget

        # SPINBOX -- max_number_of_teams is the only one, plain-ranged (no
        # scale, no sentinel).
        widget = _make_spinbox(value, editable, minimum=spec.minimum, maximum=spec.maximum)
        widget.valueChanged.connect(lambda new, s=spec: self._option_changed(s, new))
        return widget

    def _build_stance_row(self, cell_id: str, value: int, editable: bool) -> QWidget:
        """Three QRadioButtons in one QButtonGroup, exclusive by default.
        The group's ids are the raw stance values themselves (0/1/3), so a
        test can read the shown value back with group.checkedId() rather
        than matching on label text.

        A cell whose stored value is not in {0,1,3} is rendered disabled
        regardless of `editable` -- see the module docstring's out-of-enum
        handling; a label emits no signal, and neither does a disabled
        radio group with nothing checked, so no write path can reach it.
        """
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)

        representable = value in _STANCE_VALUES
        row_editable = editable and representable

        group = QButtonGroup(container)
        for label, choice in _STANCE_CHOICES:
            radio = QRadioButton(label)
            radio.setEnabled(row_editable)
            group.addButton(radio, choice)
            if choice == value:
                radio.setChecked(True)
            if row_editable:
                # Per-radio toggled(), filtered on checked, rather than the
                # group's own idClicked/idToggled: an exclusive group emits
                # toggled(False) for the button losing the check and
                # toggled(True) for the one gaining it, and only the latter
                # is a real edit.
                radio.toggled.connect(
                    lambda checked, cid=cell_id, v=choice: self._changed(cid, v) if checked else None
                )
            row.addWidget(radio)
        container.button_group = group

        if not representable:
            note = QLabel(f"stored value {value}")
            note.setToolTip(
                f"Stored as {value}, outside {{Ally, Neutral, Enemy}}. Shown as "
                "stored rather than snapped to one of the three radio choices."
            )
            row.addWidget(note)

        return container

    # -- reporting an edit ---------------------------------------------------

    def _changed(self, cell_id: str, value: int) -> None:
        """The one place a widget signal becomes a reported edit -- same two
        guards as MapOptionsPanel._changed()/PlayersPanel._changed(), and
        for the same reasons their docstrings give: `_populating` covers a
        programmatic populate (radio.setChecked(True) during
        _build_stance_row would otherwise fire toggled(True) mid-populate,
        recording a phantom undo step per row), and the equality check
        stops re-selecting the already-checked radio from recording one.

        Also updates `_pending_values`, not just `_values`: the window's
        own callback does not repopulate this panel on a successful edit
        (only on a refused one, or an undo/redo), so without this,
        switching to another player and back would show the file's
        original value until some unrelated event happened to trigger a
        repopulate.
        """
        if self._populating or cell_id not in self._editable_fields:
            return
        if self._values.get(cell_id) == value:
            return
        self._values[cell_id] = value
        self._pending_values[cell_id] = value
        self._on_diplomacy_field(cell_id, value)

    def _option_changed(self, spec, value: int) -> None:
        """The Teams group's counterpart to _changed(), reporting through
        `on_option_field(spec, value)` instead -- the same shape and the
        same two guards as MapOptionsPanel._changed(); see that method's
        docstring for why both are needed."""
        if self._populating or spec.field_id not in self._editable_fields:
            return
        if self._values.get(spec.field_id) == value:
            return
        self._values[spec.field_id] = value
        self._pending_values[spec.field_id] = value
        self._on_option_field(spec, value)

    # -- read access, for tests -------------------------------------------

    def widget_for(self, field_id: str) -> QWidget | None:
        return self._widgets.get(field_id)

    def current_values(self) -> dict[str, int]:
        """Raw value per cell_id, as currently shown for the selected
        player."""
        return dict(self._values)


def _swatch_icon(rgb: tuple[int, int, int]) -> QIcon:
    pixmap = QPixmap(16, 16)
    pixmap.fill(QColor(*rgb))
    return QIcon(pixmap)
