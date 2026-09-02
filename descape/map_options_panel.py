"""MapOptionsPanel: the Map Options mode's left page -- victory
conditions, diplomacy, map flags and the trigger execution-order mode."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import math

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


from descape import (
    option_fields,
)
from descape.scenario_io import (
    LoadedScenario,
)
from descape.viewer_common import _fit_combo_width, _make_spinbox


class MapOptionsPanel(QWidget):
    """Scenario-wide settings: victory conditions, diplomacy, map flags, and
    the trigger execution-order mode (the Map Options mode's left page).

    Dumb by design, exactly as TriggerPanel is. It never imports
    OptionsEditModel or TriggerEditModel and never touches EditHistory: it
    reports "the user set field X to raw value V" through the callback it is
    constructed with, and the window decides what that means. Which model a
    given row routes to is deliberately not visible from here -- exec-order
    goes to TriggerEditModel and everything else to OptionsEditModel, and
    hiding that split behind one callback is what keeps that decision in one
    place.

    Values are raw, as stored in the file. `spec.scale` is a display-only
    divisor, applied on the way into a widget and undone on the way back out,
    so the callback and the model both only ever see file units.
    """

    # Its own constant rather than TriggerPanel's 340: this is a single-column
    # form of short labels, not a tree plus a detail pane, and there is no
    # reason a width tuned for one suits the other. Measured against the
    # widest row this panel actually builds -- see
    # tests/test_map_options_panel.py's width check.
    MIN_USEFUL_WIDTH = 300

    _NO_DOCUMENT = "No map open."
    # Not one flag: which rows are editable is two independent gates, and the
    # window resolves both (options_model verifies the byte-patched scalars and
    # says nothing about exec-order; trigger_write_supported says nothing about
    # the scalars). A file can legitimately land on either side of one gate and
    # the other side of the other, so the panel is handed the resolved set of
    # field ids rather than a boolean it would have to interpret.
    _ALL_READ_ONLY_NOTE = "Read-only for this file -- nothing here can be written back."
    _SOME_READ_ONLY_NOTE = "{count} of {total} settings are read-only for this file."
    # Why a row can be absent rather than merely greyed: presence is read per
    # file from the parsed retriever, never inferred from the scenario
    # version. See option_fields.specs_for().
    _ABSENT_NOTE = "Settings this scenario version does not store are not listed."
    # A row can also be listed but not editable, which is a different
    # situation and needs saying here rather than only in the row's tooltip --
    # a user looking at "villager_force_drop  90" on a 1.41 file has no other
    # way to find out why it is not a checkbox. See _is_representable().
    _AS_STORED_NOTE = (
        "{count} setting{s} hold{verb} a value no editor could show truthfully "
        "and {is_are} listed as stored -- hover for details."
    )

    def __init__(self, on_option_field=None):
        super().__init__()
        # No-op default so the panel stays constructible on its own, the same
        # contract TriggerPanel's callbacks have (screenshot and width tests
        # build it with no window).
        self._on_option_field = on_option_field or (lambda *args: None)

        self._loaded: LoadedScenario | None = None
        self._specs: tuple[option_fields.OptionFieldSpec, ...] = ()
        # field_id -> the widget, and field_id -> the raw value that widget is
        # currently showing. The second is not derivable from the first once
        # editing lands: a byte-patch write leaves the parsed retriever holding
        # the ORIGINAL value, so re-reading the file would make every edit look
        # like a fresh change on the next populate.
        self._widgets: dict[str, QWidget] = {}
        self._values: dict[str, int] = {}
        self._editable_fields: frozenset[str] = frozenset()
        self._read_only_reasons: dict[str, str] = {}
        self._populating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.status = QLabel(self._NO_DOCUMENT)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.area = QScrollArea()
        # The one setting that transfers from TriggerPanel's own scroll area:
        # without it the inner widget keeps its sizeHint width and the form
        # never tracks the pane, so every row overflows instead of wrapping.
        # (_configure_scrolling's other two settings are QTreeWidget-only --
        # a QScrollArea over group boxes has neither a header nor an elide
        # mode.)
        self.area.setWidgetResizable(True)
        layout.addWidget(self.area, stretch=1)
        self._rebuild_host()

    def _rebuild_host(self) -> None:
        """Replace the scroll area's contents with a fresh, empty host.

        Wholesale rather than clearing rows in place: the group boxes present
        depend on which specs the file actually has, so a repopulate can
        change the form's structure and not just its values. QScrollArea takes
        ownership of the widget it is given, so the previous host is deleted
        by setWidget().
        """
        self.host = QWidget()
        self.host_layout = QVBoxLayout(self.host)
        self.host_layout.setContentsMargins(6, 6, 6, 6)
        self.area.setWidget(self.host)
        self._widgets = {}

    # -- document state ------------------------------------------------------

    def clear_document(self) -> None:
        self._populating = True
        try:
            self._loaded = None
            self._specs = ()
            self._values = {}
            self._editable_fields = frozenset()
            self._read_only_reasons = {}
            self._rebuild_host()
            self.status.setText(self._NO_DOCUMENT)
            self.status.setToolTip("")
        finally:
            self._populating = False

    def show_scenario(
        self,
        loaded: LoadedScenario | None,
        values: dict[str, int] | None = None,
        editable_fields: Iterable[str] = (),
        read_only_reasons: Mapping[str, str] | None = None,
        notes: Sequence[str] = (),
    ) -> None:
        """Populate from `loaded`, one group box per spec group.

        `values` overrides individual fields' raw values by field_id. The
        window passes it whenever there are pending edits, because those live
        in an edit model and not in the parsed retrievers this otherwise reads
        from -- a byte-patch write never mutates the retriever it patches, so
        a repopulate that skipped this would silently reset the form to the
        file's values while the model still held the edits.

        `editable_fields` names the rows the window will accept edits for.
        Everything else is built disabled. The panel deliberately does not work
        out which rows those are: that is two separate gates the window owns
        (see the class comment on _ALL_READ_ONLY_NOTE).

        `read_only_reasons` is why, per field id, for the rows that are not --
        shown as that row's tooltip, the same way the Show sprites action
        explains its own disabled state. A greyed row with no tooltip is the
        failure this prevents: the exec-order row carries no spec tooltip of
        its own, so without this a user hovering it learns nothing at all.
        `notes` is the status-line summary alongside it.

        The Triggers group's single row (exec-order) only appears if the
        caller has already parsed the Triggers section: option_fields
        resolves presence from loaded._scenario.sections, and "Triggers" is
        absent there until parse_triggers() has run. The window's mode-entry
        branch does that first, so this method does not have to know about it.
        """
        if loaded is None:
            self.clear_document()
            return

        self._populating = True
        try:
            self._loaded = loaded
            # Diplomacy-group specs render on DiplomacyPanel now (step 5) --
            # filtered here rather than a second call, so a spec added
            # later without a `panel` value defaults to "map_options" and
            # still shows up somewhere.
            self._specs = tuple(s for s in option_fields.specs_for(loaded) if s.panel == "map_options")
            self._values = {
                spec.field_id: option_fields.current_value(loaded, spec) for spec in self._specs
            }
            if values:
                self._values.update({k: v for k, v in values.items() if k in self._values})
            self._editable_fields = frozenset(editable_fields) & set(self._values)
            self._read_only_reasons = dict(read_only_reasons or {})
            self._rebuild_host()
            self._build_groups()
            text, tooltip = self._status_text(loaded, notes)
            self.status.setText(text)
            self.status.setToolTip(tooltip)
        finally:
            self._populating = False

    def _status_text(self, loaded: LoadedScenario, notes: Sequence[str]) -> tuple[str, str]:
        """(label text, tooltip text) for the summary above the form.

        Two strings, not one, and the split is load-bearing rather than
        cosmetic. This label word-wraps in a ~300 px column, so every sentence
        added to it costs two or three lines of the form below. Spelling out
        all four explanations inline grew it to 209 px of a 760 px panel and
        pushed the Triggers group -- the single row one of those explanations
        was about -- below the fold, with the note about it still on screen.
        Measured from an offscreen capture, which is the only thing that showed
        it: every width and height assertion passed throughout.

        So the label carries counts only, and the prose that accounts for them
        moves to its tooltip. Nothing is lost: each count's *per-row* detail is
        already on the row itself (an as-stored label explains its own raw
        number, a greyed row carries the gate that refused it), so this text is
        a summary of things individually explained elsewhere.
        """
        total = len(self._specs)
        read_only = total - len(self._editable_fields)
        as_stored = sum(1 for w in self._widgets.values() if isinstance(w, QLabel))

        summary = [f"{total} setting{'' if total == 1 else 's'} — scenario {loaded.scenario_version}"]
        detail = [self._ABSENT_NOTE]
        if read_only == total and total:
            summary.append("all read-only")
            detail.insert(0, self._ALL_READ_ONLY_NOTE)
        elif read_only:
            summary.append(f"{read_only} read-only")
            detail.insert(0, self._SOME_READ_ONLY_NOTE.format(count=read_only, total=total))
        if as_stored:
            summary.append(f"{as_stored} shown as stored")
            detail.append(
                self._AS_STORED_NOTE.format(
                    count=as_stored,
                    s="" if as_stored == 1 else "s",
                    verb="s" if as_stored == 1 else "",
                    is_are="is" if as_stored == 1 else "are",
                )
            )
        detail.extend(notes)
        return ", ".join(summary) + ".", " ".join(detail)

    # -- building the form ---------------------------------------------------

    def _build_groups(self) -> None:
        """One QGroupBox per spec group, in first-appearance order.

        Order comes from option_fields._SPECS rather than from a second list
        here, so adding a spec cannot silently land its group in a different
        place than the field list reads.
        """
        forms: dict[str, QFormLayout] = {}
        for spec in self._specs:
            form = forms.get(spec.group)
            if form is None:
                box = QGroupBox(spec.group)
                form = QFormLayout(box)
                # Both policies are load-bearing together in a narrow column,
                # and for the same reasons TriggerPanel's property form
                # documents: long-labelled rows drop their editor to the next
                # line instead of being squeezed by a label column sized to
                # the longest label, and the editors then actually use the
                # width that frees up.
                form.setRowWrapPolicy(QFormLayout.WrapLongRows)
                form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
                forms[spec.group] = form
                self.host_layout.addWidget(box)
            widget = self._build_widget(spec)
            self._widgets[spec.field_id] = widget
            # Only if the widget has not already explained itself: an
            # out-of-range row's tooltip is the only place its bare number is
            # accounted for, and neither note below displaces it.
            # random_start_points is the live case -- it carries a spec tooltip
            # and is a checkbox, so it can take any of these branches.
            # Between the other two, why the row cannot be edited at all wins
            # over what the setting does.
            tip = self._read_only_reasons.get(spec.field_id, "") or spec.tooltip
            if tip and not widget.toolTip():
                widget.setToolTip(tip)
            form.addRow(spec.label, widget)
        # Absorbs the leftover height so the groups stay stacked at the top
        # instead of spreading out over a tall pane.
        self.host_layout.addStretch(1)

    def _build_widget(self, spec) -> QWidget:
        # int() once, up front: a u8 retriever can parse as a bool, and both
        # QComboBox.findData and the range comparison below treat True and 1
        # as different keys.
        value = int(self._values[spec.field_id])
        # A sentinel substitutes its display value before the representability
        # check, so the sentinel case flows through the ordinary widget path
        # instead of the as-stored label. self._values itself is untouched --
        # only this local `value` feeds the widget constructor -- so a pure
        # browse that never touches the row stays byte-identical on save.
        substituted = spec.sentinel is not None and value == spec.sentinel
        if substituted:
            value = spec.sentinel_display
        if not self._is_representable(spec, value):
            return self._out_of_range_label(spec, value)
        editable = spec.field_id in self._editable_fields

        if spec.kind == option_fields.CHECKBOX:
            widget = QCheckBox()
            widget.setChecked(bool(value))
            widget.setEnabled(editable)
            widget.toggled.connect(
                lambda checked, s=spec: self._changed(s, int(checked))
            )
        elif spec.kind == option_fields.COMBO:
            widget = QComboBox()
            _fit_combo_width(widget)
            for choice, label in spec.choices:
                widget.addItem(label, choice)
            if widget.findData(value) < 0:
                # A value outside the shipped enum is kept as its own row
                # rather than snapped to the nearest legal one, which would
                # rewrite a field the user never touched. Same rule as
                # TriggerPanel's ENUM branch.
                widget.addItem(f"unknown ({value})", value)
            widget.setCurrentIndex(max(widget.findData(value), 0))
            widget.setEnabled(editable)
            widget.currentIndexChanged.connect(
                lambda _, s=spec, w=widget: self._changed(s, w.currentData())
            )
        elif spec.scale != 1.0:
            # A scaled field is fractional by construction, so an integer
            # spinbox would truncate whatever the file already held -- 12.5
            # reading back as 12.
            #
            # Its live consumer is victory_years (a count of 10ths of a
            # year) -- see option_fields._SPECS. The direction of the
            # division and the floor() below were both settled by
            # measurement against the corpus and the library before this
            # branch had a real spec to serve.
            widget = QDoubleSpinBox()
            widget.setDecimals(1)
            widget.setRange(spec.minimum / spec.scale, spec.maximum / spec.scale)
            widget.setValue(value / spec.scale)
            widget.setEnabled(editable)
            widget.setKeyboardTracking(False)
            widget.valueChanged.connect(
                # floor(shown * scale), mirroring OptionManager.victory_years'
                # own setter (`floor(value * 10)`), so a value round-trips
                # through the library's accessor unchanged. Checked, not
                # assumed: floor and round agree for every raw value in
                # 0..100000 once decimals is 1, because n/10*10 recovers n
                # exactly at these magnitudes.
                lambda shown, s=spec: self._changed(s, math.floor(shown * s.scale))
            )
        else:
            widget = self._plain_spinbox(spec, value, editable)

        if substituted:
            widget.setToolTip(
                f"Stored as unset. Shows {spec.sentinel_display}, the same "
                "default the in-game editor fills in the first time you "
                "switch to this victory condition -- type a value to set it "
                "explicitly."
            )
        return widget

    @staticmethod
    def _is_representable(spec, value: int) -> bool:
        """Whether this row's normal editor can show `value` truthfully.

        Both false cases are real corpus data, not hypotheticals, and both
        would otherwise misreport the file rather than fail loudly:

        - A checkbox off bool(value) shows 90, 119, 167 and 255 -- every value
          villager_force_drop carries on the 1.41 files, where the byte exists
          but was never a flag -- as an indistinguishable ticked box, which a
          save would then write back as 1 over the stored byte.
        - A spinbox raises OverflowError past int32 and *silently clamps*
          anything else outside its range. The instance that forced this rule
          was required_score_for_score_victory, 0xFFFFFFFF on five corpus
          files -- now handled upstream by spec.sentinel instead, which
          substitutes an in-range display value before this check ever runs,
          but the rule stays in place for any value a sentinel doesn't cover.

        A row that fails this is rendered as a plain label, which emits no
        signal, so _changed() can never fire for it and no write path can reach
        the stored value. That is the containment, not a convention.

        A combo needs no check: its own branch keeps an out-of-enum value as an
        "unknown (N)" row, which is already faithful.
        """
        if spec.kind == option_fields.CHECKBOX:
            return value in (0, 1)
        if spec.kind == option_fields.COMBO:
            return True
        return spec.minimum <= value <= spec.maximum

    @staticmethod
    def _out_of_range_label(spec, value: int) -> QLabel:
        """A stored value this row's editor could not show truthfully, shown
        as stored instead.

        The same rule the combo branch follows for an out-of-enum value, and
        the same rule AGENTS.md already states for GAIA rotation: never redraw
        a value the user did not touch as something else. It is also what
        keeps step 3 from being able to corrupt one -- a label emits no signal,
        so _changed() can never fire for such a row.

        Deliberately not disabled: greying reads as "temporarily unavailable",
        and this is permanent for as long as the file holds this value.
        Selectable, so the number can at least be copied out.
        """
        expected = (
            "0 or 1"
            if spec.kind == option_fields.CHECKBOX
            else f"{spec.minimum} to {spec.maximum}"
        )
        label = QLabel(str(value))
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setToolTip(
            f"Stored as {value}, outside this setting's editable range "
            f"({expected}). Shown as stored and kept verbatim rather than "
            f"forced into range."
        )
        return label

    def _plain_spinbox(self, spec, value: int, editable: bool) -> QSpinBox:
        spin = _make_spinbox(
            value, editable, minimum=spec.minimum, maximum=spec.maximum
        )
        spin.valueChanged.connect(lambda new, s=spec: self._changed(s, new))
        return spin

    # -- reporting an edit ---------------------------------------------------

    def _changed(self, spec, value: int) -> None:
        """The one place a widget signal becomes a reported edit.

        Both guards are needed, and TriggerPanel._changed()'s own docstring
        records why: Qt emits valueChanged/currentIndexChanged during a
        programmatic populate, and commit pushes unconditionally, so without
        the equality check tabbing through the form would record one phantom
        undo step per field. `_populating` covers the populate itself, which
        the equality check alone would not: a populate sets the widget before
        _values is consulted only because _build_widget reads _values first,
        and that ordering is not something a future edit should have to
        preserve silently.

        The membership test is belt-and-braces over the widget's own disabled
        state, and is the guard that survives a future row that stays enabled
        for some other reason -- a disabled widget emitting no signal is a Qt
        behaviour, not a contract this panel gets to rely on.
        """
        if self._populating or spec.field_id not in self._editable_fields:
            return
        if self._values.get(spec.field_id) == value:
            return
        self._values[spec.field_id] = value
        self._on_option_field(spec, value)

    # -- read access, for the window and for tests ---------------------------

    def current_values(self) -> dict[str, int]:
        """Raw value per field_id, as currently shown."""
        return dict(self._values)

    def widget_for(self, field_id: str) -> QWidget | None:
        return self._widgets.get(field_id)
