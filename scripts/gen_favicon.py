"""Generate the NuclearCutter favicon: a green (UI --green) trefoil with a
red circle-and-slash "not" symbol over it. Outputs an SVG (crisp at any
size) plus 32x32 and 16x16 PNGs (rendered from the same geometry at 4x and
downscaled for antialiasing). Run once; commit the outputs."""
import math
from pathlib import Path

from PIL import Image, ImageDraw

STATIC = Path(__file__).resolve().parents[1] / "nuclearcutter" / "static"
GREEN = (0, 255, 65, 255)      # CSS --green: #00ff41 (nuclear green)
RED = (255, 68, 68, 255)       # red for the "not" symbol
BG = (0, 0, 0, 0)

# Geometry in a 64x64 design space (center 32,32), matching the classic
# trefoil proportions: center dot, three 60-degree annular blades. The
# blades sit just inside the red "not" ring so green stays visible around
# it.
CX = CY = 32.0
R_DOT = 6.0        # center dot radius
R_IN = 12.0        # blade inner radius (gap between dot and blade)
R_OUT = 24.0       # blade outer radius (just inside the red ring)
BLADE_ARC = 60.0   # degrees
BLADE_ANGLES = (-90.0, 30.0, 150.0)  # one blade pointing up
SLASH_W = 7.0      # "not" ring / slash thickness
RING_R = 27.5      # "not" ring radius


def trefoil_mask(size: int) -> Image.Image:
    """Render the green trefoil (blades + dot) on a transparent square."""
    s = size * 4  # supersample for antialiasing
    scale = s / 64.0
    img = Image.new("RGBA", (s, s), BG)
    d = ImageDraw.Draw(img)
    cx = cy = CX * scale
    for a in BLADE_ANGLES:
        d.pieslice([cx - R_OUT * scale, cy - R_OUT * scale,
                    cx + R_OUT * scale, cy + R_OUT * scale],
                   start=a - BLADE_ARC / 2, end=a + BLADE_ARC / 2,
                   fill=GREEN)
        # carve the inner hole of the blade back out
        d.ellipse([cx - R_IN * scale, cy - R_IN * scale,
                   cx + R_IN * scale, cy + R_IN * scale],
                  fill=BG)
    d.ellipse([cx - R_DOT * scale, cy - R_DOT * scale,
               cx + R_DOT * scale, cy + R_DOT * scale], fill=GREEN)
    return img.resize((size, size), Image.LANCZOS)


def not_symbol_mask(size: int) -> Image.Image:
    """Red circle + 45-degree slash (top-left to bottom-right), transparent bg."""
    s = size * 4
    scale = s / 64.0
    img = Image.new("RGBA", (s, s), BG)
    d = ImageDraw.Draw(img)
    cx = cy = CX * scale
    w = SLASH_W * scale
    d.ellipse([cx - RING_R * scale, cy - RING_R * scale,
               cx + RING_R * scale, cy + RING_R * scale],
              outline=RED, width=int(w))
    # slash across the circle at 45 degrees (screen convention: y down)
    a = math.radians(45)
    dx, dy = RING_R * scale * math.cos(a), RING_R * scale * math.sin(a)
    d.line([cx - dx, cy - dy, cx + dx, cy + dy], fill=RED, width=int(w))
    return img.resize((size, size), Image.LANCZOS)


def favicon_rgba(size: int) -> Image.Image:
    img = trefoil_mask(size)
    img.alpha_composite(not_symbol_mask(size))
    return img


def svg() -> str:
    # Same geometry, exact vector form (64x64 viewBox).
    def blade(a: float) -> str:
        r1, r2, arc = R_IN, R_OUT, BLADE_ARC / 2
        def pt(r, ang):
            t = math.radians(ang)
            return (CX + r * math.cos(t), CY + r * math.sin(t))
        o1, o2 = pt(r2, a - arc), pt(r2, a + arc)
        i1, i2 = pt(r1, a - arc), pt(r1, a + arc)
        return (f'<path d="M {o1[0]:.3f} {o1[1]:.3f} '
                f'A {r2} {r2} 0 0 1 {o2[0]:.3f} {o2[1]:.3f} '
                f'L {i2[0]:.3f} {i2[1]:.3f} '
                f'A {r1} {r1} 0 0 0 {i1[0]:.3f} {i1[1]:.3f} Z"/>')
    blades = "".join(blade(a) for a in BLADE_ANGLES)
    # slash endpoints at 45 degrees on the ring circle
    d = RING_R * math.cos(math.radians(45))
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <g fill="#00ff41">
    {blades}
    <circle cx="32" cy="32" r="{R_DOT}"/>
  </g>
  <g fill="none" stroke="#ff4444" stroke-width="{SLASH_W}">
    <circle cx="32" cy="32" r="{RING_R}"/>
    <line x1="{32 - d:.3f}" y1="{32 - d:.3f}" x2="{32 + d:.3f}" y2="{32 + d:.3f}"/>
  </g>
</svg>
'''


def main() -> None:
    STATIC.mkdir(parents=True, exist_ok=True)
    (STATIC / "favicon.svg").write_text(svg(), encoding="utf-8")
    for size in (32, 16):
        favicon_rgba(size).save(STATIC / f"favicon-{size}.png")
    # /favicon.ico convention: serve the 32px PNG bytes from the server
    # (browsers accept PNG at that path); also write it under the .png name.
    print("wrote:", ", ".join(p.name for p in sorted(STATIC.glob("favicon*"))))


if __name__ == "__main__":
    main()
