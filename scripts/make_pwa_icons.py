#!/usr/bin/env python3
"""Render the PWA/Apple-touch icons from the favicon.svg design using PIL.

Mirrors nuclearcutter/static/favicon.svg geometry: dark rounded square,
green trefoil (outer r=42, blade inner r=13, hub r=8, blades 60 deg wide at
120 deg spacing starting at top), red ring r=45 stroke 7, red diagonal.
Drawn at 4x then downsampled (LANCZOS) for clean anti-aliased edges.
"""
from pathlib import Path
from PIL import Image, ImageDraw

OUT = Path("/mnt/ai-workspace/NuclearCutter/nuclearcutter/static/icons")
OUT.mkdir(exist_ok=True)
SS = 4  # supersample
BG = (13, 13, 13, 255)
GREEN = (0, 255, 65, 255)
RED = (255, 95, 95, 255)


def draw_mark(d, cx, cy, s, ring=True):
    """cx,cy = center px; s = px per ONE svg unit (viewBox is 100 wide)."""
    def px(v):  # svg-unit distance -> px offset
        return v * s
    # trefoil blades: pieslice angles are clockface-from-3-o'clock, y-down
    for start in (240, 360, 120):
        d.pieslice([cx - px(42), cy - px(42), cx + px(42), cy + px(42)],
                   start, start + 60, fill=GREEN)
    # hub: erase blade inner ends with bg, then green hub
    d.ellipse([cx - px(13), cy - px(13), cx + px(13), cy + px(13)], fill=BG)
    d.ellipse([cx - px(8), cy - px(8), cx + px(8), cy + px(8)], fill=GREEN)
    if ring:
        d.ellipse([cx - px(45), cy - px(45), cx + px(45), cy + px(45)],
                  outline=RED, width=max(1, round(px(7))))
        # diagonal 18.2..81.8 both axes (svg coords), round caps
        x1, y1 = cx - px(50 - 18.2), cy - px(50 - 18.2)
        x2, y2 = cx + px(50 - 18.2), cy + px(50 - 18.2)
        d.line([x1, y1, x2, y2], fill=RED, width=max(1, round(px(7))))
        r = px(7) / 2
        for x, y in ((x1, y1), (x2, y2)):
            d.ellipse([x - r, y - r, x + r, y + r], fill=RED)


def render(size, rounded=True, scale=1.0):
    """rounded=False gives full-bleed bg (iOS/apple/maskable requirement)."""
    im = Image.new("RGBA", (size * SS, size * SS),
                   (0, 0, 0, 0) if rounded else BG)
    d = ImageDraw.Draw(im)
    w = im.width
    if rounded:
        r = 0.18 * w
        d.rounded_rectangle([0, 0, w - 1, w - 1], radius=r, fill=BG)
    # content: svg viewBox is 100 units; scale=1 spans the full canvas
    s = w / 100.0 * scale
    draw_mark(d, w / 2, w / 2, s)
    return im.resize((size, size), Image.LANCZOS)


render(192).save(OUT / "icon-192.png")
render(512).save(OUT / "icon-512.png")
render(512, rounded=False, scale=0.62).save(OUT / "icon-maskable-512.png")
render(180, rounded=False, scale=0.80).save(OUT / "apple-touch-icon.png")
print("wrote:", sorted(p.name for p in OUT.iterdir()))
