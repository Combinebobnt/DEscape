"""Settings > Appearance's condition and effect details: how the Triggers
panel's Detail column reads (GH #388).

One stored dict, settings.get_trigger_details(). Each edit saves at once and
re-renders the open panel (TriggerPanel.refresh_trigger_details()), like the
other tabs' live controls. The Preview tree renders three sample rows through
the same summariser and delegate as the panel.

The page fills its controls with their signals blocked and only then
connects them, so opening and closing the dialog writes nothing.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from descape import debug_log, library_compat, player_labels, settings, terrain_palette, trigger_summary
from descape.trigger_summary_delegate import SUMMARY_ROLE, TriggerSummaryDelegate
from descape.viewer_common import FocusWheelComboBox

if TYPE_CHECKING:
    from descape.viewer import ViewerWindow

_STYLES = (("short", "Short (P2 to P1, visible)"), ("full", "Full (P2 sees P1 as visible)"))
_PLAYER_TEXTS = (("number", "P2"), ("full", "P2 - Name"), ("name", "Name"))
_PLAYER_COLOURS = (("off", "Off"), ("box", "Box"), ("text", "Text"), ("both", "Box and text"))
_TEXT_TAGS = (("colour", "As colours"), ("raw", "Show the tags"))
# (key, label, tooltip)
_CHECKS = (
    ("show_off_flags", "Flags that are off", "Show a yes/no field that is off, as in \"not forced\""),
    ("show_ids", "Ids after names", "The object or technology id after its name, as in \"Castle #82\""),
    ("show_unit_ids", "Placed object ids", "A placed object's own id after it, as in \"P1 Archer #500\""),
    ("show_hp", "HP on heal and damage", "\"5 HP to P1 Archer\" rather than \"5 to P1 Archer\" on Heal Object and Damage Object"),
)
_SELECTED_WITH_AREA = (
    ("area", "The area"),
    ("objects", "Selected objects"),
    ("objects_first", "Both, objects first"),
    ("area_first", "Both, area first"),
)
_LINKS = (
    ("link_coords", "Coordinates", "A click centres the map on the area or location"),
    ("link_objects", "Objects", "A click centres the map on the object, or steps through the objects that match"),
    ("link_triggers", "Trigger names", "A click selects that trigger"),
)
# The Preview's rows: (kind, type name, field values over the type's defaults).
_SAMPLES = (
    (
        "condition",
        "objects_in_area",
        {"quantity": 1, "object_list": 82, "source_player": 7, "area_x1": 21, "area_y1": 215, "area_x2": 24, "area_y2": 218},
    ),
    ("effect", "set_player_visibility", {"source_player": 2, "target_player": 1, "visibility_state": 0}),
    ("effect", "research_technology", {"source_player": 4, "technology": 22, "force_research_technology": 0}),
    ("effect", "modify_attribute", {"quantity": 1, "object_list_unit_id": 82, "source_player": 7, "operation": 1, "object_attributes": 0}),
    ("effect", "activate_trigger", {"trigger_id": 4}),
    ("effect", "display_instructions", {"source_player": 1, "message": "<AQUA>Defend the castle until the reinforcements arrive"}),
    ("condition", "destroy_object", {"unit_object": 500}),
    (
        "effect",
        "kill_object",
        {"source_player": 1, "object_list_unit_id": 4, "selected_object_ids": [500, 501, 502, 503],
         "area_x1": 10, "area_y1": 20, "area_x2": 15, "area_y2": 25},
    ),
)
_SAMPLE_TRIGGERS = {4: "Wave 2"}
_SAMPLE_UNITS = {
    ref: SimpleNamespace(player_id=player, unit_const=const, bounds=(0, 0, 1, 1), own_tile=(0, 0))
    for ref, player, const in ((500, 7, 82), (501, 1, 4), (502, 1, 4), (503, 1, 4))
}
_SAMPLE_NAMES = ("", "Britons", "Franks", "Goths", "Mongols", "Celts", "Huns", "Byzantines", "Persians")


class TriggerDetailsPage(QWidget):
    def __init__(self, window: ViewerWindow):
        super().__init__()
        self._window = window
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        details = QGroupBox("Condition and effect details")
        form = QFormLayout(details)
        mode = QWidget()
        mode_row = QHBoxLayout(mode)
        mode_row.setContentsMargins(0, 0, 0, 0)
        self.mode_buttons = QButtonGroup(self)
        for value, text in (("readable", "Readable"), ("raw", "Raw fields")):
            button = QRadioButton(text)
            button.setProperty("value", value)
            self.mode_buttons.addButton(button)
            mode_row.addWidget(button)
        mode_row.addStretch(1)
        form.addRow("Detail view:", mode)
        self.style_combo = FocusWheelComboBox()
        for value, text in _STYLES:
            self.style_combo.addItem(text, value)
        form.addRow("Sentence style:", self.style_combo)
        self.player_combo = FocusWheelComboBox()
        for value, text in _PLAYER_TEXTS:
            self.player_combo.addItem(text, value)
        form.addRow("Player text:", self.player_combo)
        self.colour_combo = FocusWheelComboBox()
        for value, text in _PLAYER_COLOURS:
            self.colour_combo.addItem(text, value)
        self.colour_combo.setToolTip("Show a player's colour as a box before it, as its text colour, or both")
        form.addRow("Player colour:", self.colour_combo)
        self.checks: dict[str, QCheckBox] = {}
        for key, text, tip in _CHECKS:
            self.checks[key] = QCheckBox(text)
            self.checks[key].setToolTip(tip)
            form.addRow("", self.checks[key])
        self.cap_spin = QSpinBox()
        self.cap_spin.setRange(0, settings.TRIGGER_DETAILS_TEXT_CAP_MAX)
        self.cap_spin.setSuffix(" characters")
        self.cap_spin.setSpecialValueText("Never")
        self.cap_spin.setToolTip("Quoted text longer than this ends in an ellipsis; the row's tooltip has it whole")
        form.addRow("Shorten text to:", self.cap_spin)
        self.tags_combo = FocusWheelComboBox()
        for value, text in _TEXT_TAGS:
            self.tags_combo.addItem(text, value)
        self.tags_combo.setToolTip("Draw quoted text after a <COLOR> tag in that colour, or show the tags as written")
        form.addRow("Colour tags:", self.tags_combo)
        self.id_limit_spin = QSpinBox()
        self.id_limit_spin.setRange(0, settings.TRIGGER_DETAILS_UNIT_ID_LIMIT_MAX)
        self.id_limit_spin.setSuffix(" ids")
        self.id_limit_spin.setSpecialValueText("Count only")
        self.id_limit_spin.setToolTip("How many ids \"3 selected P1 objects\" lists before it ends in an ellipsis")
        form.addRow("List selected:", self.id_limit_spin)
        self.area_combo = FocusWheelComboBox()
        for value, text in _SELECTED_WITH_AREA:
            self.area_combo.addItem(text, value)
        self.area_combo.setToolTip("What an effect with both selected objects and an area shows")
        form.addRow("Objects and area:", self.area_combo)
        layout.addWidget(details)

        links = QGroupBox("Links in the Detail column")
        links_layout = QHBoxLayout(links)
        for key, text, tip in _LINKS:
            self.checks[key] = QCheckBox(text)
            self.checks[key].setToolTip(tip)
            links_layout.addWidget(self.checks[key])
        links_layout.addStretch(1)
        layout.addWidget(links)

        preview_box = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_box)
        self.preview = QTreeWidget()
        self.preview.setHeaderLabels(["Item", "Detail"])
        self.preview.setRootIsDecorated(False)
        self.preview.setUniformRowHeights(True)
        self._preview_delegate = TriggerSummaryDelegate(self.preview)
        self._preview_delegate.player_colors = tuple(terrain_palette.PLAYER_COLORS[pid] for pid in range(9))
        self.preview.setItemDelegateForColumn(1, self._preview_delegate)
        preview_layout.addWidget(self.preview)
        layout.addWidget(preview_box)

        reset_row = QHBoxLayout()
        self.reset_button = QPushButton("Reset to Defaults")
        reset_row.addWidget(self.reset_button)
        reset_row.addStretch(1)
        layout.addLayout(reset_row)

        self._show(settings.get_trigger_details())
        self.mode_buttons.buttonToggled.connect(lambda button, on: on and self._store("mode", button.property("value")))
        self.style_combo.currentIndexChanged.connect(lambda _i: self._store("sentence_style", self.style_combo.currentData()))
        self.player_combo.currentIndexChanged.connect(lambda _i: self._store("player_text", self.player_combo.currentData()))
        self.colour_combo.currentIndexChanged.connect(lambda _i: self._store("player_colour", self.colour_combo.currentData()))
        for key, box in self.checks.items():
            box.toggled.connect(lambda on, k=key: self._store(k, on))
        self.cap_spin.valueChanged.connect(lambda value: self._store("text_cap", value))
        self.tags_combo.currentIndexChanged.connect(lambda _i: self._store("text_tags", self.tags_combo.currentData()))
        self.id_limit_spin.valueChanged.connect(lambda value: self._store("unit_id_limit", value))
        self.area_combo.currentIndexChanged.connect(lambda _i: self._store("selected_with_area", self.area_combo.currentData()))
        self.reset_button.clicked.connect(self._reset)

    def _show(self, values: dict) -> None:
        """Fill every control from `values` with signals blocked, then re-render the Preview."""
        widgets = (
            self.mode_buttons, self.style_combo, self.player_combo, self.colour_combo, self.cap_spin, self.id_limit_spin,
            self.area_combo, self.tags_combo,
            *self.checks.values(),
        )
        for widget in widgets:
            widget.blockSignals(True)
        try:
            for button in self.mode_buttons.buttons():
                button.setChecked(button.property("value") == values["mode"])
            self.style_combo.setCurrentIndex(self.style_combo.findData(values["sentence_style"]))
            self.player_combo.setCurrentIndex(self.player_combo.findData(values["player_text"]))
            self.colour_combo.setCurrentIndex(self.colour_combo.findData(values["player_colour"]))
            for key, box in self.checks.items():
                box.setChecked(bool(values[key]))
            self.cap_spin.setValue(values["text_cap"])
            self.tags_combo.setCurrentIndex(self.tags_combo.findData(values["text_tags"]))
            self.id_limit_spin.setValue(values["unit_id_limit"])
            self.area_combo.setCurrentIndex(self.area_combo.findData(values["selected_with_area"]))
        finally:
            for widget in widgets:
                widget.blockSignals(False)
        self._render_preview(values)

    def _store(self, key: str, value) -> None:
        self._save({key: value})

    def _reset(self) -> None:
        self._save(dict(settings.TRIGGER_DETAILS_DEFAULTS))
        self._show(settings.get_trigger_details())

    def _save(self, values: dict) -> None:
        try:
            settings.set_trigger_details(values)
        except ValueError as e:
            self._window._log_status(f"Trigger detail option not saved: {e}", level=debug_log.WARNING)
            return
        self._render_preview(settings.get_trigger_details())
        self._window.trigger_panel.refresh_trigger_details()

    def _render_preview(self, values: dict) -> None:
        self.preview.clear()
        version = library_compat.vocabulary_versions()[-1]
        vocabulary = library_compat.load_vocabulary(version)
        options = trigger_summary.Options.from_mapping(values)
        context = trigger_summary.Context(
            labels=player_labels.labels_for(_SAMPLE_NAMES), trigger_name=_SAMPLE_TRIGGERS.get, unit=_SAMPLE_UNITS.get
        )
        for kind, name, fields in _SAMPLES:
            entries = vocabulary.conditions if kind == "condition" else vocabulary.effects
            presentation = vocabulary.condition_presentation if kind == "condition" else vocabulary.effect_presentation
            definition = next((d for d in entries.values() if d.name == name), None)
            if definition is None:
                continue
            entry = SimpleNamespace(**{**definition.default_attributes, **fields})
            summary = trigger_summary.summarize(kind, entry, definition, presentation, context, options)
            item = QTreeWidgetItem([name.replace("_", " "), summary.text])
            item.setData(1, SUMMARY_ROLE, summary if options.mode == trigger_summary.READABLE else None)
            self.preview.addTopLevelItem(item)
        self.preview.resizeColumnToContents(0)
        self.preview.resizeColumnToContents(1)
        rows = self.preview.topLevelItemCount()
        self.preview.setFixedHeight(self.preview.header().sizeHint().height() + rows * self.preview.sizeHintForRow(0) + 6)
