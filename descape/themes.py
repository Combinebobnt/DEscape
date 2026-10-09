"""Qt-free app-chrome theme data: the built-in presets and the colour roles a
user may override, Settings > Appearance.

One active theme is a preset plus optional per-role overrides (settings.py's
`theme` and `theme_colors`). viewer_dialogs.apply_theme() turns that into a
QPalette; this module only holds the data and the colour arithmetic, so it
never imports PyQt5 or descape.settings (settings imports it).

Scope is app chrome only. The map view's tool overlay colours
(settings.OVERLAY_COLORS) are a separate setting no preset touches.
"""

from __future__ import annotations

import re

# (role_id, user-facing label, QPalette.ColorRole attribute name).
THEME_ROLES: list[tuple[str, str, str]] = [
    ("window", "Window background", "Window"),
    ("window_text", "Window text", "WindowText"),
    ("base", "Input background", "Base"),
    ("alternate_base", "Alternate row background", "AlternateBase"),
    ("text", "Input text", "Text"),
    ("button", "Button", "Button"),
    ("button_text", "Button text", "ButtonText"),
    ("highlight", "Selection", "Highlight"),
    ("highlighted_text", "Selected text", "HighlightedText"),
    ("tooltip_base", "Tooltip background", "ToolTipBase"),
    ("tooltip_text", "Tooltip text", "ToolTipText"),
    ("link", "Link", "Link"),
]
ROLE_IDS: tuple[str, ...] = tuple(role_id for role_id, _label, _qt in THEME_ROLES)
_ROLE_LABELS: dict[str, str] = {role_id: label for role_id, label, _qt in THEME_ROLES}
ROLE_QT_NAMES: dict[str, str] = {role_id: qt_name for role_id, _label, qt_name in THEME_ROLES}

# Keys only the legacy `dark` preset carries: its recipe's five explicit
# Disabled colours and BrightText, beyond the overridable roles.
LEGACY_EXTRA_KEYS: tuple[str, ...] = (
    "disabled_window_text",
    "disabled_text",
    "disabled_button_text",
    "disabled_highlight",
    "disabled_highlighted_text",
    "bright_text",
)

# The presets whose palette is exactly the pre-theme app's: `light` is Fusion's
# standardPalette() untouched, `dark` the old dark-mode recipe.
LEGACY_PRESETS: tuple[str, ...] = ("light", "dark")
THEME_DEFAULT = "light"

# Disabled-group derivation: (role, its background role, blend toward it).
DISABLED_BLENDS: tuple[tuple[str, str, float], ...] = (
    ("window_text", "window", 0.5),
    ("text", "base", 0.5),
    ("button_text", "button", 0.5),
    ("highlight", "window", 0.6),
)
# highlighted_text's Disabled colour is Disabled text.
DERIVED_DISABLED_ROLES: tuple[str, ...] = (*(role for role, _bg, _t in DISABLED_BLENDS), "highlighted_text")

# Solarized colours: Ethan Schoonover's palette (MIT), ethanschoonover.com/solarized.
_SOL_BASE03 = "#002b36"
_SOL_BASE02 = "#073642"
_SOL_BASE1 = "#93a1a1"
_SOL_BASE2 = "#eee8d5"
_SOL_BASE3 = "#fdf6e3"
_SOL_BLUE = "#268bd2"
_SOL_CYAN = "#2aa198"

PRESETS: dict[str, tuple[str, dict[str, str]]] = {
    "light": ("Light", {}),
    "dark": (
        "Dark",
        {
            "window": "#353535",
            "window_text": "#e6e6e6",
            "base": "#232323",
            "alternate_base": "#353535",
            "text": "#ffffff",
            "button": "#353535",
            "button_text": "#ffffff",
            "highlight": "#2a82da",
            "highlighted_text": "#000000",
            "tooltip_base": "#353535",
            "tooltip_text": "#ffffff",
            "link": "#2a82da",
            "disabled_window_text": "#7f7f7f",
            "disabled_text": "#7f7f7f",
            "disabled_button_text": "#7f7f7f",
            "disabled_highlight": "#505050",
            "disabled_highlighted_text": "#7f7f7f",
            "bright_text": "#ff0000",
        },
    ),
    "dim": (
        "Dim",
        {
            "window": "#45484d",
            "window_text": "#e8e8e8",
            "base": "#37393d",
            "alternate_base": "#3f4246",
            "text": "#f0f0f0",
            "button": "#53575c",
            "button_text": "#f0f0f0",
            "highlight": "#4f86c6",
            "highlighted_text": "#ffffff",
            "tooltip_base": "#2b2d30",
            "tooltip_text": "#f0f0f0",
            "link": "#8ab4f8",
        },
    ),
    "high_contrast_dark": (
        "High contrast dark",
        {
            "window": "#000000",
            "window_text": "#ffffff",
            # Not black: Fusion outlines a check box from the window colour, so
            # a black base made the indicator vanish on the black window.
            "base": "#1c1c1c",
            "alternate_base": "#2a2a2a",
            "text": "#ffffff",
            "button": "#262626",
            "button_text": "#ffffff",
            "highlight": "#1aebff",
            "highlighted_text": "#000000",
            "tooltip_base": "#000000",
            "tooltip_text": "#ffff00",
            "link": "#ffff00",
        },
    ),
    "high_contrast_light": (
        "High contrast light",
        {
            "window": "#ffffff",
            "window_text": "#000000",
            "base": "#ffffff",
            "alternate_base": "#ebebeb",
            "text": "#000000",
            "button": "#e6e6e6",
            "button_text": "#000000",
            "highlight": "#37006e",
            "highlighted_text": "#ffffff",
            "tooltip_base": "#ffffff",
            "tooltip_text": "#000000",
            "link": "#00009f",
        },
    ),
    "solarized_light": (
        "Solarized light",
        {
            "window": _SOL_BASE2,
            "window_text": _SOL_BASE02,
            "base": _SOL_BASE3,
            "alternate_base": _SOL_BASE2,
            "text": _SOL_BASE02,
            "button": _SOL_BASE2,
            "button_text": _SOL_BASE02,
            "highlight": _SOL_BLUE,
            "highlighted_text": _SOL_BASE3,
            "tooltip_base": _SOL_BASE02,
            "tooltip_text": _SOL_BASE2,
            "link": _SOL_BLUE,
        },
    ),
    "solarized_dark": (
        "Solarized dark",
        {
            "window": _SOL_BASE02,
            "window_text": _SOL_BASE1,
            "base": _SOL_BASE03,
            "alternate_base": _SOL_BASE02,
            "text": _SOL_BASE1,
            "button": _SOL_BASE02,
            # base2, not base1: Fusion paints a dark button lighter, which took base1 to 3.85:1.
            "button_text": _SOL_BASE2,
            "highlight": _SOL_BLUE,
            "highlighted_text": _SOL_BASE3,
            "tooltip_base": _SOL_BASE03,
            "tooltip_text": _SOL_BASE1,
            "link": _SOL_CYAN,
        },
    ),
}
HIGH_CONTRAST_PRESETS: tuple[str, ...] = ("high_contrast_dark", "high_contrast_light")


def preset_label(preset_id: str) -> str:
    return PRESETS[preset_id][0] if preset_id in PRESETS else preset_id


def role_label(role_id: str) -> str:
    return _ROLE_LABELS.get(role_id, role_id)


def normalize_hex(value: str) -> str:
    """Lowercased "#rrggbb", accepted case-insensitively. Raises ValueError on
    anything else. settings.py re-imports this as _normalize_hex."""
    if isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        return value.lower()
    raise ValueError(f"colour must be '#rrggbb', got {value!r}")


def _rgb(hex_str: str) -> tuple[int, int, int]:
    value = normalize_hex(hex_str)
    return int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16)


def blend(color: str, toward: str, amount: float) -> str:
    """`color` moved `amount` (0..1) of the way to `toward`, per channel."""
    a, b = _rgb(color), _rgb(toward)
    mixed = (round(ca + (cb - ca) * amount) for ca, cb in zip(a, b, strict=True))
    return "#{:02x}{:02x}{:02x}".format(*mixed)


def relative_luminance(hex_str: str) -> float:
    """WCAG 2.x relative luminance of an sRGB colour."""

    def channel(c: int) -> float:
        s = c / 255
        return s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(c) for c in _rgb(hex_str))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a: str, b: str) -> float:
    """WCAG contrast ratio, 1..21, order-independent."""
    la, lb = relative_luminance(a), relative_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)
