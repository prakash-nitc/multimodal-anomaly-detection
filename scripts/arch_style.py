# -*- coding: utf-8 -*-
"""Shared print styling for the papers' architecture figures.

Both papers place their architecture figure on its own landscape page, where the
9.4 in text height is the available width, so each figure is drawn at its full
design size (8.4 in) and printed slightly larger than that. This module makes every
label bold and sets one font, so the figures read clearly in print.

    import arch_style
    with arch_style.bold_print_style():
        build(...)
"""
from __future__ import annotations

import contextlib

import matplotlib
import matplotlib.axes as maxes
import matplotlib.pyplot as plt

# Segoe UI lacks the snowflake glyph used for "frozen"; matplotlib falls back along
# font.family entries glyph by glyph (not along font.sans-serif entries).
FAMILY = ["Segoe UI", "Segoe UI Symbol", "DejaVu Sans"]


@contextlib.contextmanager
def bold_print_style():
    saved_rc = {k: matplotlib.rcParams[k] for k in ("font.family", "font.weight")}
    saved_text = maxes.Axes.text

    def bold_text(self, *args, **kwargs):
        kwargs["fontweight"] = "bold"        # overrides the base scripts' "medium"
        return saved_text(self, *args, **kwargs)

    matplotlib.rcParams["font.family"] = FAMILY
    matplotlib.rcParams["font.weight"] = "bold"
    maxes.Axes.text = bold_text
    try:
        yield
    finally:
        maxes.Axes.text = saved_text
        for k, v in saved_rc.items():
            matplotlib.rcParams[k] = v
        plt.close("all")
