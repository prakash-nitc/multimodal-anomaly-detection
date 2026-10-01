# -*- coding: utf-8 -*-
"""One place that decides how a figure is written to disk.

Every figure used to be raster only, at 400 dpi and 6.9in wide -- about 425 dpi
at print size, which is ample for paper and still reads as soft on screen. A PDF
viewer at 150-200% zoom resamples those pixels, and "the diagrams look a little
blurry" is exactly what raster text looks like under that zoom. Nobody reads a
report at 100%.

Vector output has no resolution at all: lines and glyphs are drawn by the
viewer's own rasteriser at whatever magnification is asked for. So each figure is
now written twice, from the same draw call:

    .pdf   vector -- what the LaTeX report includes
    .png   raster -- what python-pptx embeds, since PowerPoint cannot place a PDF

Photographs inside a figure stay raster in both, which is correct and
unavoidable; what turns crisp is every label, axis, box, arrow and number drawn
around them.
"""
from __future__ import annotations

import os

import matplotlib

# Type 42 embeds TrueType outlines, so text in the PDF stays real text --
# selectable, searchable, and rendered by the viewer at its own resolution.
# matplotlib's default (Type 3) is a PostScript construct that several viewers
# render badly at high zoom, and is the other common cause of a figure that
# "looks fuzzy" despite a high dpi.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def _warn_if_clipped(png_path: str, margin: int = 6) -> bool:
    """Report a figure whose ink runs off the canvas.

    tight_layout() repacks the axes but never shrinks or wraps a title, so
    raising the type scale can push a long title straight off the right edge --
    silently, since the figure still writes successfully and only looks wrong.
    Any ink in the outermost few pixels means something did not fit.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:                      # checking is a convenience, not a job
        return False
    a = np.asarray(Image.open(png_path).convert("L"))
    if a.size == 0:
        return False
    edges = {"left": a[:, :margin], "right": a[:, -margin:],
             "top": a[:margin, :], "bottom": a[-margin:, :]}
    hit = [name for name, band in edges.items() if (band < 245).any()]
    if hit:
        print("  WARNING: %s -- content touches the %s edge; a title or label "
              "is probably clipped. Shorten it or wrap it with an explicit \\n."
              % (os.path.basename(png_path), "/".join(hit)))
    return bool(hit)


def save(fig, png_path: str, dpi: int = 400, facecolor: str = "white"):
    """Write <name>.pdf and <name>.png next to each other. Returns both paths.

    Callers pass the .png path they always passed, so adding vector output did
    not require touching a single figure's layout code.
    """
    base = os.path.splitext(png_path)[0]
    pdf, png = base + ".pdf", base + ".png"
    fig.savefig(pdf, facecolor=facecolor)          # vector: no dpi applies
    fig.savefig(png, dpi=dpi, facecolor=facecolor)  # raster: for the deck
    _warn_if_clipped(png)
    return [pdf, png]
