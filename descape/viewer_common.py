"""Widget helpers and tool-id lookups shared by the viewer's panels.

Split out of viewer.py so trigger_panel and map_options_panel can share
them without importing viewer.py, which imports both of them. The
settings.TOOLS-derived sets live here as one documented family even
though their consumers are split across map_view.py and viewer.py.

They are deliberately not in settings.py next to the TOOLS registry they
derive from: they are view-layer lookups rather than settings (three of the
seven are private by name), and settings.py's import surface is depended on
directly by a lot of tests."""

from __future__ import annotations

import math

from PyQt5.QtGui import QColor, QFontMetricsF, QIcon, QPixmap, QValidator
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
)

from descape import (
    settings,
)

# Derived from settings.TOOLS -- the single tool registry -- rather than
# hand-duplicated here, so a new tool added there can't silently miss an
# entry in any of these. Internal tool identifiers (e.g. "elevation",
# "set_level") are unchanged even when a display label is renamed later
# (Elevate / Set Elevation), so settings.get_keybind()'s persisted
# "tool_elevation"/"tool_set_level" keys and every other internal reference
# keep working regardless.
EDIT_TOOLS = frozenset(t.tool_id for t in settings.TOOLS if t.is_edit_tool)
_TOOL_LABELS = {t.tool_id: t.label for t in settings.TOOLS}
_STROKE_LABELS = {t.tool_id: t.stroke_label for t in settings.TOOLS if t.is_edit_tool}
# One-shot click tools (Paint Can) that must never enter the drag-stroke
# path -- see MapView.mousePressEvent's early CLICK_TOOLS branch and
# ViewerWindow.on_fill.
CLICK_TOOLS = frozenset(t.tool_id for t in settings.TOOLS if t.click_only)
# Which toolbar param widget ("terrain" | "level" | "") the active tool
# reads -- see ViewerWindow._update_tool_enabled's tool-param visibility
# block.
_TOOL_PARAM = {t.tool_id: t.param_widget for t in settings.TOOLS}
# Tools whose stroke applies across a brush footprint (size + shape) rather
# than always exactly one tile -- see ToolDef.supports_brush's own comment.
BRUSH_TOOLS = frozenset(t.tool_id for t in settings.TOOLS if t.supports_brush)
# Draw Line / Draw Rectangle: press-drag-preview, commit once at release.
# Members are edit tools but must never reach the per-tile stroke path --
# see MapView's shape branches, which sit above the EDIT_TOOLS ones.
SHAPE_TOOLS = frozenset(t.tool_id for t in settings.TOOLS if t.drag_shape)
_TOOL_SHAPE = {t.tool_id: t.drag_shape for t in settings.TOOLS}
# Tools that offer the free (non-snapped) placement checkbox -- see
# ToolDef.supports_free_place's own comment.
FREE_PLACE_TOOLS = frozenset(t.tool_id for t in settings.TOOLS if t.supports_free_place)
# The Ruler and Eyedropper are the only tools that are neither an edit tool
# nor Pan, so they can't be recognised by set membership the way the four
# sets above are.
TOOL_RULER = "ruler"
TOOL_EYEDROPPER = "eyedropper"
TOOL_SELECT = "select"

# Which mode(s) each tool's toolbar button shows in -- see
# settings.ToolDef.modes.
_TOOL_MODES = {t.tool_id: t.modes for t in settings.TOOLS}


def brush_applicable(tool_id: str, *, rect_filled: bool) -> bool:
    """True if `tool_id` paints across a brush footprint right now.

    BRUSH_TOOLS membership on its own is not the answer for Draw Rectangle:
    it supports a brush in Outline mode (a thicker border) but not in Filled
    mode, where a brush would dilate the rectangle past the bounds the
    preview just showed. That carve-out lives here, in one Qt-free place,
    rather than in the two viewer.py call sites plus the TOOLS-derived
    expectations in tests/test_toolbar_params.py -- which is what keeps
    those expectations generated rather than hand-listed.
    """
    if tool_id not in BRUSH_TOOLS:
        return False
    return not (_TOOL_SHAPE.get(tool_id) == "rect" and rect_filled)


def tool_applicable(tool_id: str, mode: str) -> bool:
    """True if `tool_id`'s toolbar button should be visible in `mode`.

    Empty ToolDef.modes means every mode. Used by
    ViewerWindow._update_tool_enabled() to hide mode-inapplicable tools
    instead of just greying them out.
    """
    modes = _TOOL_MODES.get(tool_id, ())
    return not modes or mode in modes

# How many characters a property-form combo box sizes itself for. Without it a
# combo's minimumSizeHint is its longest *item*, and ObjectAttribute's longest
# is 430 px in a 340 px panel -- so the form sat permanently below its own
# stated minimum, which is what made every attempt to narrow the label column
# tip the whole form into horizontal overflow.
_COMBO_CONTENTS_CHARS = 12


# QFontMetricsF.averageCharWidth() of the font every panel width was measured
# against: the test suite's pinned DejaVu Sans 12pt at 96 DPI.
_BASELINE_AVG_CHAR_PX = 8.109375


class FontScaledWidth:
    """A panel's MIN_USEFUL_WIDTH, read as a class attribute like the plain
    int it replaces. `base_px` is the width measured at the baseline font;
    a larger app font (Windows 150% scaling reports 144 DPI to Qt 5, or a
    bigger Settings > Appearance size) scales it up, since every label and
    field in the panel grows with the font while a pixel constant would not.
    Never scales below `base_px`, so a smaller font keeps today's width."""

    def __init__(self, base_px: int) -> None:
        self.base_px = base_px

    def __get__(self, obj, owner=None) -> int:
        if QApplication.instance() is None:
            return self.base_px
        ratio = QFontMetricsF(QApplication.font()).averageCharWidth() / _BASELINE_AVG_CHAR_PX
        return max(self.base_px, math.ceil(self.base_px * ratio))


def _fit_combo_width(combo: QComboBox) -> None:
    """Size a combo for a fixed character count rather than for its longest
    item -- see _COMBO_CONTENTS_CHARS. The popup still shows every name in
    full. Shared by TriggerPanel and MapOptionsPanel: VictoryCondition and
    SecondaryGameMode are the same class of long-label enum ObjectAttribute
    is."""
    combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(_COMBO_CONTENTS_CHARS)


def _swatch_icon(rgb: tuple[int, int, int]) -> QIcon:
    """A 16px player-colour square for a player combo's item icon."""
    pixmap = QPixmap(16, 16)
    pixmap.fill(QColor(*rgb))
    return QIcon(pixmap)


class _IndeterminateMixin:
    """A blank "the selection differs" state for a spinbox (GH #60).

    A spinbox has no blank value and -1 is a real one (trigger_fields.UNSET),
    so the box parks on its minimum and renders "" instead. specialValueText
    beats textFromValue() at the minimum, so it is parked too and put back on
    the first real change. The panel clears the state from valueChanged, or
    from editingFinished when the typed value equals the parked one.

    That second case cannot use lineEdit().isModified(): Qt re-renders the
    text before editingFinished, which clears the flag (measured). So the last
    user-typed text is latched from textEdited, which programmatic sets never emit.
    """

    _indeterminate = False
    _parked_special_text = ""
    _typed_text = ""

    def set_indeterminate(self) -> None:
        if self._indeterminate:
            return
        self._indeterminate = True
        self._typed_text = ""
        self._parked_special_text = self.specialValueText()
        self.setSpecialValueText("")
        self.setValue(self.minimum())
        # By hand: setValue() on a box already at its minimum re-renders nothing.
        self.lineEdit().setText("")
        self.lineEdit().textEdited.connect(self._note_typed)

    def _note_typed(self, text: str) -> None:
        if self._indeterminate:
            self._typed_text = text.strip()

    def typed_parked_value(self) -> bool:
        """Whether the user typed the very value the blank box is parked on,
        the one edit that emits no valueChanged."""
        if not self._indeterminate or not self._typed_text:
            return False
        state, _text, _pos = self.validate(self._typed_text, 0)
        return state == QValidator.Acceptable and self.valueFromText(self._typed_text) == self.value()

    def is_indeterminate(self) -> bool:
        return self._indeterminate

    def clear_indeterminate(self) -> None:
        """Leave the blank state and show the value it now holds."""
        if not self._indeterminate:
            return
        self._indeterminate = False
        self.setSpecialValueText(self._parked_special_text)
        # The value that ended the state was rendered blank; Qt will not repaint it.
        value = self.value()
        at_special = value == self.minimum() and bool(self._parked_special_text)
        self.lineEdit().setText(self._parked_special_text if at_special else self.textFromValue(value))

    def textFromValue(self, value):  # Qt override
        return "" if self._indeterminate else super().textFromValue(value)


class IndeterminateSpinBox(_IndeterminateMixin, QSpinBox):
    pass


class IndeterminateDoubleSpinBox(_IndeterminateMixin, QDoubleSpinBox):
    pass


def _make_spinbox(
    value,
    editable: bool,
    *,
    minimum: int,
    maximum: int,
    special_value_text: str = "",
    indeterminate: bool = False,
) -> QSpinBox:
    """A property-form spinbox with keyboard tracking off.

    `setKeyboardTracking(False)` is the load-bearing one: with it on, typing
    "12" over a 5 emits valueChanged(1) and then valueChanged(12), so a
    two-digit edit records two undo steps and briefly writes a value the user
    never asked for.

    Range is a parameter rather than the hardcoded trigger range this was
    promoted from, because Options.ai_map_type's own minimum is INT32_MIN and
    a -1 floor would silently clamp it. `special_value_text` renders
    `minimum` as text instead of a number whose meaning the user would have to
    know -- TriggerPanel passes it for trigger_fields.UNSET's -1 sentinel, and
    Map Options, which has no unset sentinel, passes nothing.
    """
    spin = IndeterminateSpinBox() if indeterminate else QSpinBox()
    spin.setRange(minimum, maximum)
    if special_value_text:
        spin.setSpecialValueText(special_value_text)
    # Falls back to `minimum` rather than to 0, so TriggerPanel keeps landing
    # on UNSET (and therefore on special_value_text) for a None-valued field.
    spin.setValue(value if isinstance(value, int) and not isinstance(value, bool) else minimum)
    spin.setEnabled(editable)
    spin.setKeyboardTracking(False)
    if indeterminate:
        # Last, so it parks the special text set above.
        spin.set_indeterminate()
    return spin
