"""PlayersPanel: the Players mode's left page -- per-player settings for
P1..P8.

Structurally a sibling of MapOptionsPanel (same group-box-per-spec-group
form, same _fit_combo_width/_make_spinbox reuse, same three-way tooltip
precedence, same edit-callback/`values`/`current_values()` shape), plus one
thing MapOptionsPanel has no equivalent of: a player selector, since every
spec here is an 8-wide array rather than a scalar.

Writable as of step 3c, for whichever fields the window's
`editable_fields` names -- never worked out here, the same split
MapOptionsPanel keeps (two independent gates: the per-player write path
verifying at all, and the Map Options carrier model it additively rides in
also having to verify -- see the maintainer plan's decision 1). Three
distinct read-only reasons, not two:

- Tier 1 fields (fixed byte length) that this file's own write path failed
  to verify for -- the window's `read_only_reasons`.
- Tier 2 fields (personality, civilization, architecture), stored as
  variable-length data no byte-patch could reach regardless of file --
  `_TIER2_REASON`, resolved here rather than by the window, since it is a
  fact about the field, not the file.
- `player_type`, whose semantics are unconfirmed (not "no write path yet"
  -- one exists, but writing a byte nobody has identified is exactly what
  AGENTS.md's pass-it-through-verbatim rules exist to prevent) --
  `_PLAYER_TYPE_REASON`, also resolved here.

`tribe_name` is a fourth, temporary case: player_fields.write_targets()
already resolves its offset, but its `"c256"` codec is a string and
OptionsEditModel stays int-only until step 3d, so it is never in
`editable_fields` yet and gets its own pending-step reason here too.

GAIA is deliberately never selectable: eight of PlayerFieldSpec's arrays
have no GAIA slot at all (PlayerArrayLayout.index_for raises for one), and
GAIA's own PlayerDataTwo color slot is known junk (terrain_palette.py).
"""

from __future__ import annotations

from typing import Iterable, Mapping

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from descape.player_fields import (
    CHECKBOX,
    COMBO,
    TEXT,
    current_value,
    resolve_civilization_name,
    specs_for,
)
from descape.scenario_io import LoadedScenario
from descape.viewer_common import _fit_combo_width, _make_spinbox

# resolve_civilization_name() is only correct for these two fields --
# everything else that renders as TEXT (tribe_name, personality) is a plain
# string already, and running it through the civilization dataset would risk
# an accidental match (a tribe name that happens to collide with a
# Civilization enum value) rather than a deliberate one.
_CIVILIZATION_STYLE_FIELDS = frozenset({"civilization", "architecture"})

# Personality/civilization/architecture are the Step 1 doc's Tier 2:
# variable-length data (ai_names is str16; player_data_1's civ fields are
# str16 from 1.56 on) that no byte-patch write path could reach even once
# one exists for the fixed-length Tier 1 fields. Distinct from Tier 1's
# reason on purpose, so a future write path can tell them apart.
_TIER2_FIELDS = frozenset({"personality", "civilization", "architecture"})

# The four resource-mirrored fields plus Pop Limit: each has an f32 mirror
# (player_data_4), so the largest integer either stored copy can hold
# exactly is 2**24, well below the s32 range the spec itself declares.
# Narrowing the spinbox (not spec.minimum/maximum, which _is_representable()
# still uses -- see that function) makes the editable range match what a
# save can actually preserve. Safe against every corpus file measured: the
# real maximum anywhere is 99999, so no existing row's appearance changes.
_F32_EXACT_RANGE_FIELDS = frozenset({"food", "wood", "stone", "gold", "pop_limit"})
_F32_EXACT_RANGE = 16_777_216
_F32_EXACT_RANGE_TOOLTIP = (
    f"Limited to ±{_F32_EXACT_RANGE:,}, the largest integer this field's "
    "f32 mirror can represent exactly -- a larger value would desync the two "
    "stored copies."
)


class PlayersPanel(QWidget):
    """Per-player settings: a P1..P8 selector plus one form (the Players
    mode's left page).

    Dumb by design, same as MapOptionsPanel/TriggerPanel: it reports "the
    user set field X for player N to raw value V" through the callback it
    is constructed with, and never touches an edit model itself.
    """

    # Measured against the widest row this panel actually builds (Tribe
    # name / Personality can run long) -- see tests/test_players_panel.py's
    # width check.
    MIN_USEFUL_WIDTH = 320

    _NO_DOCUMENT = "No map open."
    _TIER2_REASON = "Stored as a variable-length field, which this tool cannot patch in place."
    _PLAYER_TYPE_REASON = (
        "Unconfirmed -- a hypothesis for the in-game 'Player Type' dropdown, not "
        "documented anywhere in AoE2ScenarioParser. Writing a byte nobody has "
        "identified risks corrupting a value whose meaning isn't confirmed."
    )
    _TRIBE_NAME_PENDING_REASON = (
        "Its offset is resolved, but editing text needs a widget kind this "
        "panel hasn't built yet."
    )

    def __init__(self, on_player_field=None):
        super().__init__()
        # No-op default so the panel stays constructible on its own, the
        # same contract MapOptionsPanel/TriggerPanel's callbacks have.
        self._on_player_field = on_player_field or (lambda *args: None)

        self._loaded: LoadedScenario | None = None
        self._specs = ()
        self._player_id = 1
        self._widgets: dict[str, QWidget] = {}
        # field_id -> the raw value the CURRENT player's widget is showing.
        # Not derivable from the widget once editing lands, for the same
        # reason MapOptionsPanel._values isn't: a byte-patch write leaves
        # the parsed retriever holding the ORIGINAL value.
        self._values: dict[str, int] = {}
        self._editable_fields: frozenset[str] = frozenset()
        self._read_only_reasons: dict[str, str] = {}
        # {field_id: {player_id: value}} for every player at once, so
        # switching the player combo can show pending edits for the newly
        # selected player without a round trip back to the window.
        self._pending_values: dict[str, dict[int, int]] = {}
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
        wholesale rather than clearing rows in place, since which groups
        exist depends on which specs the file has (see
        MapOptionsPanel._rebuild_host, same reasoning)."""
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
            self._specs = ()
            self._player_id = 1
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
        editable_fields: Iterable[str] = (),
        read_only_reasons: Mapping[str, str] | None = None,
        pending_values: Mapping[str, Mapping[int, int]] | None = None,
    ) -> None:
        """Populate from `loaded`. The single repopulate path -- both
        ViewerWindow._show_players() (mode entry, a new document) and
        ._repopulate_players() (after an edit, an undo, or a refused edit)
        call only this.

        The player selector is only rebuilt -- and the selection reset to
        P1 -- when `loaded` is a genuinely different document (identity,
        not equality: the same LoadedScenario is passed on every
        repopulate of one open file). Otherwise the selector and the
        player currently showing are left alone, so an edit on P5 does not
        silently jump the panel back to P1.

        `editable_fields` names the rows the window will accept edits for
        (bare field ids -- the write path's gate is document-wide, not
        per-player, so this set is the same regardless of which player is
        selected). `read_only_reasons` is why, per field id, for ordinary
        Tier-1 rows a gate refused; Tier 2, `player_type` and `tribe_name`
        get their own reasons unconditionally, resolved here rather than by
        the window (see the module docstring).

        `pending_values` is `{field_id: {player_id: value}}` for every
        player at once, not just the one currently shown -- switching the
        player combo has to be able to show a *different* player's pending
        edit without a round trip back to the window.
        """
        if loaded is None:
            self.clear_document()
            return

        same_document = loaded is self._loaded
        self._loaded = loaded
        self._specs = specs_for(loaded)
        self._editable_fields = frozenset(editable_fields)
        self._read_only_reasons = dict(read_only_reasons or {})
        self._pending_values = {k: dict(v) for k, v in (pending_values or {}).items()}

        if not same_document:
            self._player_id = 1
            self.player_combo.blockSignals(True)
            self.player_combo.clear()
            for player_id in range(1, 9):
                self.player_combo.addItem(_swatch_icon(loaded.player_colors[player_id]), f"P{player_id}")
            self.player_combo.setCurrentIndex(0)
            self.player_combo.blockSignals(False)
            self.player_combo.setEnabled(True)

        self._populate_current_player()

    def select_player(self, player_id: int) -> bool:
        """Backs the player_select_1..8 shortcut in Players mode. GAIA
        (player_id 0) and "no scenario loaded" both return False -- see this
        panel's own module docstring for why GAIA has no slot here."""
        if not (1 <= player_id <= 8) or not self.player_combo.isEnabled():
            return False
        self.player_combo.setCurrentIndex(player_id - 1)
        return True

    def _on_player_changed(self, index: int) -> None:
        if index < 0:
            return
        self._player_id = index + 1
        self._populate_current_player()

    def _populate_current_player(self) -> None:
        self._populating = True
        try:
            self._rebuild_host()
            self._values = {
                spec.field_id: self._pending_values.get(spec.field_id, {}).get(
                    self._player_id, current_value(self._loaded, spec, self._player_id)
                )
                for spec in self._specs
            }
            self._build_groups()
        finally:
            self._populating = False

        total = len(self._specs)
        read_only = total - len(self._editable_fields)
        version = self._loaded.scenario_version
        summary = f"{total} setting{'' if total == 1 else 's'} for P{self._player_id} — scenario {version}"
        if read_only == total and total:
            summary += ", all read-only"
        elif read_only:
            summary += f", {read_only} read-only"
        self.status.setText(summary + ".")
        self.status.setToolTip(
            "Settings this scenario version does not store are not listed. "
            "A setting the file stores as a value no editor could show "
            "truthfully is shown as stored instead -- hover it for details."
        )

    # -- building the form ------------------------------------------------

    def _build_groups(self) -> None:
        """One QGroupBox per spec group, in first-appearance order -- order
        comes from player_fields._SPECS rather than a second list here, same
        reasoning as MapOptionsPanel._build_groups()."""
        forms: dict[str, QFormLayout] = {}
        for spec in self._specs:
            form = forms.get(spec.group)
            if form is None:
                box = QGroupBox(spec.group)
                form = QFormLayout(box)
                form.setRowWrapPolicy(QFormLayout.WrapLongRows)
                form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
                forms[spec.group] = form
                self.host_layout.addWidget(box)
            value = self._values[spec.field_id]
            widget = self._build_widget(spec, value)
            self._widgets[spec.field_id] = widget
            # Three-way precedence, same as MapOptionsPanel: an out-of-range
            # label has already set its own explanatory tooltip, so only a
            # widget that hasn't gets a reason -- Tier 2/player_type/
            # tribe_name resolve their own unconditionally (facts about the
            # field, not the file); everything else falls back to the
            # window's per-file gate reason, then the spec's own tooltip.
            if not widget.toolTip():
                if spec.field_id in _TIER2_FIELDS:
                    reason = self._TIER2_REASON
                elif spec.field_id == "player_type":
                    reason = self._PLAYER_TYPE_REASON
                elif spec.field_id == "tribe_name" and spec.field_id not in self._editable_fields:
                    reason = self._TRIBE_NAME_PENDING_REASON
                else:
                    reason = self._read_only_reasons.get(spec.field_id, "")
                widget.setToolTip(reason or spec.tooltip)
            form.addRow(spec.label, widget)
        self.host_layout.addStretch(1)

    def _build_widget(self, spec, value: int | str) -> QWidget:
        if not self._is_representable(spec, value):
            return self._out_of_range_label(spec, value)
        editable = spec.field_id in self._editable_fields

        if spec.kind == TEXT:
            text = resolve_civilization_name(value) if spec.field_id in _CIVILIZATION_STYLE_FIELDS else str(value)
            label = QLabel(text)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            return label

        if spec.kind == CHECKBOX:
            widget = QCheckBox()
            widget.setChecked(bool(value))
            widget.setEnabled(editable)
            widget.toggled.connect(lambda checked, s=spec: self._changed(s, int(checked)))
            return widget

        if spec.kind == COMBO:
            widget = QComboBox()
            _fit_combo_width(widget)
            for choice, choice_label in spec.choices:
                widget.addItem(choice_label, choice)
            if widget.findData(value) < 0:
                # Same rule as MapOptionsPanel/TriggerPanel: an out-of-enum
                # value is kept as its own row rather than snapped to the
                # nearest legal one.
                widget.addItem(f"unknown ({value})", value)
            widget.setCurrentIndex(max(widget.findData(value), 0))
            widget.setEnabled(editable)
            widget.currentIndexChanged.connect(
                lambda _, s=spec, w=widget: self._changed(s, w.currentData())
            )
            return widget

        # SPINBOX -- value is guaranteed int here, never str (current_value()
        # only returns a str for a TEXT-kind spec).
        minimum, maximum = spec.minimum, spec.maximum
        if spec.field_id in _F32_EXACT_RANGE_FIELDS:
            # Widened rather than clamped if `value` itself somehow falls
            # outside ±_F32_EXACT_RANGE (no corpus file's does): a spinbox
            # range narrower than the value it is about to show would
            # silently redraw it as something else, the exact thing
            # AGENTS.md's pass-it-through-verbatim rules exist to prevent.
            minimum = min(-_F32_EXACT_RANGE, value)
            maximum = max(_F32_EXACT_RANGE, value)
        widget = _make_spinbox(value, editable, minimum=minimum, maximum=maximum)
        if editable and spec.field_id in _F32_EXACT_RANGE_FIELDS:
            # Only when editable: a read-only row's tooltip is its gate
            # reason, which is the more useful explanation when the range
            # note would otherwise displace it (see _build_groups()).
            widget.setToolTip(_F32_EXACT_RANGE_TOOLTIP)
        widget.valueChanged.connect(lambda new, s=spec: self._changed(s, new))
        return widget

    @staticmethod
    def _is_representable(spec, value: int | str) -> bool:
        """Whether this row's normal editor can show `value` truthfully --
        widened for int | str (TEXT specs may hold either), same rule
        MapOptionsPanel's own version follows for its int-only specs.

        TEXT is always representable (a label shows any int or str
        verbatim), which is what makes the Tier-2 civ type switch -- the
        same field storing an int below scenario version 1.56 and a str
        from it on -- a non-issue here despite being exactly the kind of
        version-dependent type surprise that would need special-casing for
        a CHECKBOX or SPINBOX row.
        """
        if spec.kind == TEXT:
            return True
        if spec.kind == CHECKBOX:
            return isinstance(value, int) and value in (0, 1)
        if spec.kind == COMBO:
            return True
        return isinstance(value, int) and spec.minimum <= value <= spec.maximum

    @staticmethod
    def _out_of_range_label(spec, value: int | str) -> QLabel:
        """A stored value this row's editor could not show truthfully,
        shown as stored instead -- same rule AGENTS.md states for GAIA
        rotation and MapOptionsPanel follows for its own out-of-range rows:
        never redraw a value the user did not touch as something else."""
        expected = "0 or 1" if spec.kind == CHECKBOX else f"{spec.minimum} to {spec.maximum}"
        label = QLabel(str(value))
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setToolTip(
            f"Stored as {value}, outside this setting's editable range "
            f"({expected}). Shown as stored and kept verbatim rather than "
            f"forced into range."
        )
        return label

    # -- reporting an edit -------------------------------------------------

    def _changed(self, spec, value: int) -> None:
        """The one place a widget signal becomes a reported edit -- same two
        guards as MapOptionsPanel._changed(), and for the same reasons its
        docstring gives: `_populating` covers a programmatic populate, and
        the equality check stops tabbing through the form from recording a
        phantom undo step per field.

        Also updates `_pending_values` for the current player, not just
        `_values` for the currently-shown widget: the window's own callback
        does not repopulate this panel on a successful edit (only on a
        refused one, or an undo/redo), so without this, switching to
        another player and back would show the file's original value until
        some unrelated event happened to trigger a repopulate.
        """
        if self._populating or spec.field_id not in self._editable_fields:
            return
        if self._values.get(spec.field_id) == value:
            return
        self._values[spec.field_id] = value
        self._pending_values.setdefault(spec.field_id, {})[self._player_id] = value
        self._on_player_field(spec, self._player_id, value)

    # -- read access, for the window and for tests -------------------------

    def widget_for(self, field_id: str) -> QWidget | None:
        return self._widgets.get(field_id)

    def current_values(self) -> dict[str, int]:
        """Raw value per field_id, as currently shown for the selected
        player."""
        return dict(self._values)


def _swatch_icon(rgb: tuple[int, int, int]) -> QIcon:
    pixmap = QPixmap(16, 16)
    pixmap.fill(QColor(*rgb))
    return QIcon(pixmap)
