#!/usr/bin/env python3
"""Regenerates descape/app_icon.png, a placeholder application icon used by
setWindowIcon() and the Linux AppImage's desktop integration. Pillow-only
(already a runtime dependency), swappable for real artwork later without
any other change -- see tests/test_packaging_assets.py, which asserts
properties (size, mode) rather than byte-identity against this generator.

    python3 tools/gen_app_icon.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_PATH = ROOT / "descape" / "app_icon.png"
SIZE = 256

_BG = (90, 90, 90, 255)
_FG = (40, 160, 70, 255)


def render() -> Image.Image:
    img = Image.new("RGBA", (SIZE, SIZE), _BG)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=int(SIZE * 0.7))
    bbox = draw.textbbox((0, 0), "D", font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pos = (SIZE / 2 - text_w / 2 - bbox[0], SIZE / 2 - text_h / 2 - bbox[1])
    draw.text(pos, "D", fill=_FG, font=font)
    return img


def main() -> None:
    render().save(OUT_PATH)
    print(f"wrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
