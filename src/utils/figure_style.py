"""Paper-wide Matplotlib typography and panel-label validation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text


FIGURE_FONT = "Times New Roman"
BODY_FONT_SIZE = 11
PANEL_TITLE_SIZE = 12


def apply_paper_style() -> None:
    """Apply the typography required for every paper figure."""
    font_manager.findfont(
        font_manager.FontProperties(family=FIGURE_FONT),
        fallback_to_default=False,
    )
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [FIGURE_FONT],
            "font.size": BODY_FONT_SIZE,
            "axes.titlesize": PANEL_TITLE_SIZE,
            "axes.titleweight": "bold",
            "axes.labelsize": BODY_FONT_SIZE,
            "xtick.labelsize": BODY_FONT_SIZE,
            "ytick.labelsize": BODY_FONT_SIZE,
            "legend.fontsize": BODY_FONT_SIZE,
            "legend.title_fontsize": BODY_FONT_SIZE,
            "figure.titlesize": BODY_FONT_SIZE,
            "mathtext.fontset": "custom",
            "mathtext.rm": FIGURE_FONT,
            "mathtext.it": f"{FIGURE_FONT}:italic",
            "mathtext.bf": f"{FIGURE_FONT}:bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_title(axis: plt.Axes, letter: str, title: str) -> None:
    """Set a bold 12 pt panel title with a lowercase parenthesized label."""
    axis.set_title(
        f"({letter.lower()}) {title}",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
    )


def save_figure(fig: plt.Figure, output: Path) -> None:
    """Enforce and audit typography before writing PNG and vector PDF."""
    panel_titles = [axis.title for axis in fig.axes if axis.get_title()]
    for item in fig.findobj(Text):
        item.set_fontfamily(FIGURE_FONT)
        if item in panel_titles:
            item.set_fontsize(PANEL_TITLE_SIZE)
            item.set_fontweight("bold")
        else:
            item.set_fontsize(BODY_FONT_SIZE)
            item.set_fontweight("normal")

    issues = []
    for item in fig.findobj(Text):
        if FIGURE_FONT not in item.get_fontfamily():
            issues.append(f"font:{item.get_text()[:40]}")
        expected_size = PANEL_TITLE_SIZE if item in panel_titles else BODY_FONT_SIZE
        if not math.isclose(float(item.get_fontsize()), expected_size):
            issues.append(f"size:{item.get_text()[:40]}")
        expected_weight = "bold" if item in panel_titles else "normal"
        if item.get_fontweight() != expected_weight:
            issues.append(f"weight:{item.get_text()[:40]}")
    for title in panel_titles:
        if not title.get_text().startswith(
            tuple(f"({letter}) " for letter in "abcdefghijklmnopqrstuvwxyz")
        ):
            issues.append(f"panel_label:{title.get_text()[:40]}")
        if title.get_fontweight() != "bold":
            issues.append(f"panel_weight:{title.get_text()[:40]}")
    if issues:
        raise AssertionError("Figure style audit failed: " + ", ".join(issues))

    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.15)
    output.with_suffix(".style.json").write_text(
        json.dumps(
            {
                "font_family": FIGURE_FONT,
                "body_font_size_pt": BODY_FONT_SIZE,
                "subplot_title_size_pt": PANEL_TITLE_SIZE,
                "subplot_title_weight": "bold",
                "subplot_labels": [
                    title.get_text().split(" ", 1)[0] for title in panel_titles
                ],
                "audit_passed": True,
                "png": output.name,
                "pdf": output.with_suffix(".pdf").name,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


apply_paper_style()
