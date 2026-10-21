"""
Synthetic chart generator: bar charts, dot plots and box plots with exactly known values,
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


@dataclass
class BoxSpec:
    """Box plots drawn from exact statistics: (min, Q1, median, Q3, max) per box."""
    stats: List[tuple]
    colors: Optional[List[str]] = None
    linewidth: float = 1.2
    outliers: Optional[List[List[float]]] = None
    means: Optional[List[float]] = None      # drawn as a marker when given
    caps: bool = True
    dashed: bool = False                     # dashed whiskers (R base style)
    ylim: Optional[tuple] = None
    yticks: Optional[List[float]] = None
    log: bool = False
    grid: bool = False
    frame: bool = False
    figsize: tuple = (3.2, 3.0)
    dpi: int = 200


def render_box(spec: BoxSpec) -> np.ndarray:
    """Render box plots to a BGR uint8 array."""
    n = len(spec.stats)
    bxp = []
    for i, (lo, q1, med, q3, hi) in enumerate(spec.stats):
        d = dict(whislo=lo, q1=q1, med=med, q3=q3, whishi=hi,
                 fliers=(spec.outliers[i] if spec.outliers else []))
        if spec.means:
            d["mean"] = spec.means[i]
        bxp.append(d)
    fig, ax = plt.subplots(figsize=spec.figsize, dpi=spec.dpi)
    lw = spec.linewidth
    arts = ax.bxp(bxp, positions=range(n), widths=0.55, patch_artist=True, showmeans=bool(spec.means),
                  showcaps=spec.caps,
                  boxprops=dict(linewidth=lw, edgecolor="black"),
                  medianprops=dict(color="black", linewidth=lw * 1.6),
                  whiskerprops=dict(linewidth=lw, linestyle="--" if spec.dashed else "-"),
                  capprops=dict(linewidth=lw),
                  flierprops=dict(marker="o", markersize=4, markerfacecolor="none"),
                  meanprops=dict(marker="+", markeredgecolor="black", markersize=7))
    for patch, med, c in zip(arts["boxes"], arts["medians"], spec.colors or ["white"] * n):
        patch.set_facecolor(c)
        # a black median on a dark box is invisible; real charts contrast it
        if np.mean(matplotlib.colors.to_rgb(c)) < 0.3:
            med.set_color("white")
    if spec.log:
        ax.set_yscale("log")
    if spec.ylim:
        ax.set_ylim(*spec.ylim)
    if spec.yticks is not None:
        ax.set_yticks(spec.yticks)
        if spec.log:
            ax.set_yticklabels([f"{t:g}" for t in spec.yticks])
            ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    if spec.grid:
        ax.yaxis.grid(True, color="#cccccc", linewidth=0.6, zorder=0)
    ax.spines["top"].set_visible(spec.frame)
    ax.spines["right"].set_visible(spec.frame)
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"G{i + 1}" for i in range(n)])
    ax.set_ylabel("Value (units)")
    ax.set_xlim(-0.6, n - 0.4)
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return np.ascontiguousarray(rgba[:, :, :3][:, :, ::-1])


def _to_bgr(fig) -> np.ndarray:
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return np.ascontiguousarray(rgba[:, :, :3][:, :, ::-1])


@dataclass
class ViolinSpec:
    """
    Violin plots of real samples, trimmed to the data range (Prism's and
    ggplot's default). `inner` is what is drawn inside: "lines" (Prism:
    dashed median, dotted quartiles), "box" (ggplot: a narrow box plot) or
    "median" (median line only).
    """
    data: List[np.ndarray]
    inner: str = "lines"
    colors: Optional[List[str]] = None
    edgecolor: str = "black"
    linewidth: float = 1.2
    dots: bool = False
    ylim: Optional[tuple] = None
    figsize: tuple = (3.2, 3.0)
    dpi: int = 200
    seed: int = 0

    @property
    def stats(self) -> List[tuple]:
        """(min, Q1, median, Q3, max) of each group, as drawn."""
        return [tuple(float(v) for v in np.percentile(d, [0, 25, 50, 75, 100])) for d in self.data]


def render_violin(spec: ViolinSpec) -> np.ndarray:
    from scipy.stats import gaussian_kde
    rng = np.random.default_rng(spec.seed)
    n = len(spec.data)
    fig, ax = plt.subplots(figsize=spec.figsize, dpi=spec.dpi)
    colors = spec.colors or ["white"] * n
    for i, d in enumerate(spec.data):
        lo, hi = float(d.min()), float(d.max())
        ys = np.linspace(lo, hi, 200)
        dens = gaussian_kde(d)(ys)
        half = 0.38 * dens / dens.max()
        ax.fill_betweenx(ys, i - half, i + half, facecolor=colors[i], edgecolor=spec.edgecolor,
                         linewidth=spec.linewidth, zorder=2)

        def half_at(y):
            return float(np.interp(y, ys, half))
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        dark_fill = np.mean(matplotlib.colors.to_rgb(colors[i])) < 0.35
        ink = "white" if dark_fill else "black"
        if spec.inner == "box":
            w = 0.06
            ax.add_patch(plt.Rectangle((i - w, q1), 2 * w, q3 - q1, facecolor="white", edgecolor="black",
                                       linewidth=spec.linewidth, zorder=3))
            ax.plot([i - w, i + w], [med, med], color="black", linewidth=spec.linewidth * 1.6, zorder=4)
        else:
            ax.plot([i - half_at(med), i + half_at(med)], [med, med], color=ink, linestyle="--",
                    linewidth=spec.linewidth * 1.6, zorder=3)
            if spec.inner == "lines":
                for q in (q1, q3):
                    ax.plot([i - half_at(q), i + half_at(q)], [q, q], color=ink, linestyle=":",
                            linewidth=spec.linewidth * 1.3, zorder=3)
        if spec.dots:
            ax.scatter(i + rng.uniform(-0.12, 0.12, d.size), d, s=10, facecolors="white",
                       edgecolors="black", linewidths=0.6, zorder=4)
    if spec.ylim:
        ax.set_ylim(*spec.ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"G{i + 1}" for i in range(n)])
    ax.set_ylabel("Value (units)")
    ax.set_xlim(-0.6, n - 0.4)
    return _to_bgr(fig)


@dataclass
class LineSpec:
    """
    Line plots with mean ± error at each x: one list of y values (and errors)
    per series. Markers None draws a plain line.
    """
    x: List[float]
    ys: List[List[float]]
    errs: Optional[List[List[float]]] = None
    colors: Optional[List[str]] = None
    markers: Optional[List[Optional[str]]] = None
    hollow: bool = False
    linestyles: Optional[List[str]] = None
    linewidth: float = 1.5
    capsize: float = 3.0
    legend: bool = False
    grid: bool = False
    ylim: Optional[tuple] = None
    xlabels: Optional[List[str]] = None       # categorical x tick labels
    figsize: tuple = (3.6, 3.0)
    dpi: int = 200


def render_line(spec: LineSpec) -> np.ndarray:
    fig, ax = plt.subplots(figsize=spec.figsize, dpi=spec.dpi)
    k = len(spec.ys)
    colors = spec.colors or ["black", "tab:red", "tab:blue", "tab:green"][:k]
    markers = spec.markers or ["o", "s", "^", "D"][:k]
    styles = spec.linestyles or ["-"] * k
    for i, y in enumerate(spec.ys):
        m = markers[i]
        kw = dict(color=colors[i], linestyle=styles[i], linewidth=spec.linewidth, label=f"Series {i + 1}",
                  marker=m or "", markersize=6, zorder=3)
        if m and spec.hollow:
            kw["markerfacecolor"] = "white"
        if spec.errs:
            ax.errorbar(spec.x, y, yerr=spec.errs[i], capsize=spec.capsize, elinewidth=spec.linewidth * 0.8,
                        capthick=spec.linewidth * 0.8, **kw)
        else:
            ax.plot(spec.x, y, **kw)
    if spec.ylim:
        ax.set_ylim(*spec.ylim)
    if spec.grid:
        ax.yaxis.grid(True, color="#cccccc", linewidth=0.6, zorder=0)
    if spec.legend:
        ax.legend(frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(spec.x)
    if spec.xlabels:
        ax.set_xticklabels(spec.xlabels)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Value (units)")
    pad = 0.06 * (max(spec.x) - min(spec.x))
    ax.set_xlim(min(spec.x) - pad, max(spec.x) + pad)
    return _to_bgr(fig)
