#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render the Mandelbrot habitat and overlay where the population actually lived.

Reads positions.jsonl (written by world.py, next to this file) and produces:
  output/mandelbrot_colony.png    poster, two panels
  output/mandelbrot_colony.gif    animation, colony growing along the boundary
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
POS = []

NAVY = (14, 40, 91)
GOLD = (184, 125, 55)
INK = (232, 239, 249)
MUTE = (147, 169, 198)

FONTS = {False: ["/System/Library/Fonts/Supplemental/Arial.ttf",            # macOS
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",          # Debian, Ubuntu
                 "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",        # Fedora
                 "C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf", "Arial.ttf"],
         True: ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
                "C:/Windows/Fonts/arialbd.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf"]}


def font(sz, bold=False):
    for path in FONTS[bold]:
        try:
            return ImageFont.truetype(path, sz)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=sz)       # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def mandel(cr0, cr1, ci0, ci1, w, h, maxiter=260):
    """Escape-time render, only iterating the points that are still alive."""
    x = np.linspace(cr0, cr1, w, dtype=np.float64)
    y = np.linspace(ci1, ci0, h, dtype=np.float64)
    C = x[None, :] + 1j * y[:, None]
    Z = np.zeros_like(C)
    out = np.zeros(C.shape, dtype=np.float32)
    alive = np.ones(C.shape, dtype=bool)
    idx = np.flatnonzero(alive.ravel())
    zz, cc = Z.ravel()[idx].copy(), C.ravel()[idx].copy()
    esc = np.zeros(idx.size, dtype=np.float32)
    still = np.arange(idx.size)
    for n in range(maxiter):
        zz[still] = zz[still] * zz[still] + cc[still]
        mag = zz[still].real ** 2 + zz[still].imag ** 2
        gone = mag > 4.0
        if gone.any():
            hit = still[gone]
            esc[hit] = n + 1 - np.log2(np.log(np.sqrt(mag[gone])) + 1e-12)
            still = still[~gone]
        if still.size == 0:
            break
    esc[still] = maxiter
    flat = out.ravel()
    flat[idx] = esc
    return out.reshape(C.shape), (cr0, cr1, ci0, ci1)


def palette(esc, maxiter):
    """Navy interior, gold filigree on the boundary, pale exterior."""
    t = np.clip(esc / maxiter, 0, 1)
    inside = esc >= maxiter
    img = np.zeros(esc.shape + (3,), dtype=np.uint8)
    # exterior: deep navy → mid blue → gold, cycling softly
    r = (18 + 210 * t ** 2.2)
    g = (34 + 150 * t ** 3.0)
    b = (72 + 60 * t ** 1.2)
    img[..., 0] = np.clip(r, 0, 255)
    img[..., 1] = np.clip(g, 0, 255)
    img[..., 2] = np.clip(b, 0, 255)
    img[inside] = (10, 24, 56)
    # glow right at the boundary
    edge = (~inside) & (esc > maxiter * 0.55)
    img[edge] = (240, 214, 160)
    return Image.fromarray(img, "RGB")


def gen_color(gen, gmax):
    """early = cool teal, late = gold."""
    t = gen / max(1, gmax)
    r = int(95 + (255 - 95) * t)
    g = int(211 + (198 - 211) * t)
    b = int(196 + (120 - 196) * t)
    return (r, g, b)


def panel(img, box, view, pts, title, sub, maxiter, gmax, dot=3):
    x0, y0, x1, y1 = box
    arr, _ = mandel(view[0], view[1], view[2], view[3], x1 - x0, y1 - y0, maxiter)
    img.paste(palette(arr, maxiter), (x0, y0))
    d = ImageDraw.Draw(img, "RGBA")
    cr0, cr1, ci0, ci1 = view
    for p in pts:
        px = x0 + (p["cr"] - cr0) / (cr1 - cr0) * (x1 - x0)
        py = y0 + (ci1 - p["ci"]) / (ci1 - ci0) * (y1 - y0)
        if x0 <= px <= x1 and y0 <= py <= y1:
            c = gen_color(p["gen"], gmax)
            a = 40 if not p["alive"] else 150
            d.ellipse([px - dot, py - dot, px + dot, py + dot], fill=c + (a,))
    d.rectangle([x0, y0, x1, y1], outline=(60, 90, 140))
    d.text((x0 + 16, y0 + 12), title, font=font(20, True), fill=INK)
    d.text((x0 + 16, y0 + 40), sub, font=font(13), fill=MUTE)


def poster():
    W, H = 1600, 980
    img = Image.new("RGB", (W, H), (8, 18, 42))
    pts = POS
    gmax = max(p["gen"] for p in pts)
    xs = [p["cr"] for p in pts]
    ys = [p["ci"] for p in pts]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    span = max(max(xs) - min(xs), max(ys) - min(ys)) * 1.35

    panel(img, (24, 96, 804, 900), (-2.05, 0.55, -1.15, 1.15), pts,
          "the whole set", "where everything is possible, and almost nothing is fertile", 220, gmax, 2)
    panel(img, (820, 96, 1576, 900),
          (cx - span / 2, cx + span / 2, cy - span / 2.55, cy + span / 2.55),
          pts, "the colony", "the sampled region, %d generations of drift" % (gmax + 1), 320, gmax, 3)

    d = ImageDraw.Draw(img)
    d.text((24, 28), "HCS world — %d generations on the Mandelbrot boundary" % (gmax + 1),
           font=font(30, True), fill=INK)
    d.text((24, 66), "each dot is an organism; colour runs from first generation (teal) to last (gold)",
           font=font(15), fill=MUTE)
    path = OUT / "mandelbrot_colony.png"
    img.save(path)
    print(path, img.size)


def animate():
    W, H = 1080, 760
    pts = POS
    gmax = max(p["gen"] for p in pts)
    xs = [p["cr"] for p in pts]
    ys = [p["ci"] for p in pts]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    span = max(max(xs) - min(xs), max(ys) - min(ys)) * 1.5
    view = (cx - span / 2, cx + span / 2, cy - span / 2.6, cy + span / 2.6)
    arr, _ = mandel(view[0], view[1], view[2], view[3], W, H, 340)
    base = palette(arr, 340)

    frames = []
    for g in range(0, gmax + 1):
        im = base.copy()
        d = ImageDraw.Draw(im, "RGBA")
        for p in pts:
            if p["gen"] > g:
                continue
            px = (p["cr"] - view[0]) / (view[1] - view[0]) * W
            py = (view[3] - p["ci"]) / (view[3] - view[2]) * H
            c = gen_color(p["gen"], gmax)
            r = 5 if p["alive"] else 3
            d.ellipse([px - r, py - r, px + r, py + r], fill=c + (215,))
            d.ellipse([px - r - 3, py - r - 3, px + r + 3, py + r + 3], outline=c + (70,))
        d.rectangle([0, 0, W, 74], fill=(10, 24, 56))
        d.text((24, 20), "life on the boundary", font=font(24, True), fill=INK)
        d.text((24, 50), "generation %d of %d" % (g, gmax), font=font(14), fill=MUTE)
        frames.append(im)

    small = [f.resize((W // 2, H // 2), Image.LANCZOS) for f in frames]
    pal = [s.convert("P", palette=Image.ADAPTIVE, colors=200) for s in small]
    path = OUT / "mandelbrot_colony.gif"
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=90, loop=0,
                optimize=False, disposal=1)
    print(path, len(pal), "frames")


if __name__ == "__main__":
    src = ROOT / "positions.jsonl"
    if not src.exists():
        sys.exit("no %s yet: run world.py first" % src.name)
    POS[:] = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    OUT.mkdir(exist_ok=True)
    poster()
    animate()
