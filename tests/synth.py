"""
Synthetic chart generator: bar charts and dot plots with exactly known values,
in the styles seen in real papers (outlined/unoutlined, coloured, scatter
dots, SEM caps, negative axes, gridlines, frames, small/low-DPI images).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


@dataclass
class ChartSpec:
    means: List[float]
    errs: List[float]
    kind: str = "bar"                 # "bar" | "dot"
    colors: Optional[List[str]] = None
    edgecolor: Optional[str] = "black"
    linewidth: float = 1.5
    dots: bool = False
    error_color: Optional[str] = "black"
    ylim: Optional[tuple] = None
    yticks: Optional[List[float]] = None
    grid: bool = False
    frame: bool = False
    labels: Optional[List[str]] = None
    figsize: tuple = (3.2, 3.0)
    dpi: int = 200
    seed: int = 0
    extra: dict = field(default_factory=dict)


def render(spec: ChartSpec) -> np.ndarray:
    """Render to a BGR uint8 array (what the extractor receives from a PDF)."""
    rng = np.random.default_rng(spec.seed)
    n = len(spec.means)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=spec.figsize, dpi=spec.dpi)
    colors = spec.colors or ["white"] * n
    if spec.kind == "bar":
        ax.bar(x, spec.means, width=0.6, color=colors, edgecolor=spec.edgecolor,
               linewidth=spec.linewidth if spec.edgecolor else 0, zorder=2)
        # one-sided error bar away from zero with a single cap, as Prism draws
        # over a filled bar (matplotlib's errorbar would add a cap at the mean)
        ec = spec.error_color or "black"
        for xi, m, e in zip(x, spec.means, spec.errs):
            tip = m + abs(e) * (1 if m >= 0 else -1)
            ax.plot([xi, xi], [m, tip], color=ec, linewidth=spec.linewidth, zorder=3, solid_capstyle="butt")
            ax.plot([xi - 0.1, xi + 0.1], [tip, tip], color=ec, linewidth=spec.linewidth, zorder=3,
                    solid_capstyle="butt")
    else:
        for i, (m, e) in enumerate(zip(spec.means, spec.errs)):
            c = colors[i] if colors[i] != "white" else "tab:blue"
            ax.hlines(m, i - 0.25, i + 0.25, colors="black", linewidth=2.2, zorder=3)
            ax.errorbar([i], [m], yerr=[[e], [e]], fmt="none", ecolor=c, elinewidth=1.4,
                        capsize=4, capthick=1.4, zorder=3)
    if spec.dots:
        for i, (m, e) in enumerate(zip(spec.means, spec.errs)):
            pts = m + rng.normal(0, abs(e) * 2.0 + 1e-9, 8)
            ax.scatter(i + rng.uniform(-0.18, 0.18, 8), pts, s=14, facecolors="white",
                       edgecolors="black", linewidths=0.8, zorder=4)
    if spec.ylim:
        ax.set_ylim(*spec.ylim)
    if spec.yticks is not None:
        ax.set_yticks(spec.yticks)
    if spec.grid:
        ax.yaxis.grid(True, color="#cccccc", linewidth=0.6, zorder=0)
    ax.spines["top"].set_visible(spec.frame)
    ax.spines["right"].set_visible(spec.frame)
    ax.set_xticks(x)
    ax.set_xticklabels(spec.labels or [f"G{i + 1}" for i in range(n)])
    ax.set_ylabel("Value (units)")
    ax.set_xlim(-0.6, n - 0.4)
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    rgb = rgba[:, :, :3]
    return np.ascontiguousarray(rgb[:, :, ::-1])
