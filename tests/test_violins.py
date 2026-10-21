"""
Violin plots: synthetic violins of real samples with exactly known
min/Q1/median/Q3/max, in the styles Prism, ggplot2 and matplotlib draw.

    python -m pytest tests/test_violins.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backend"))
sys.path.insert(0, HERE)

from estimators import estimate  # noqa: E402
from figure_extractor import ExtractOptions, FigureExtractor  # noqa: E402
from synth import BoxSpec, ChartSpec, ViolinSpec, render, render_box, render_violin  # noqa: E402

VIOLIN = ExtractOptions(bar_names=False, titles=False, chart_type="violin")


@pytest.fixture(scope="module")
def extractor():
    return FigureExtractor()


def _data(seed=1):
    rng = np.random.default_rng(seed)
    return [rng.normal(10, 2, 14), rng.normal(14, 3, 14), rng.gamma(4, 2, 14)]


CASES = {
    "prism_lines_white": ViolinSpec(_data(), inner="lines"),
    "prism_lines_colored": ViolinSpec(_data(), inner="lines", colors=["#888888", "#d62728", "#1f77b4"]),
    "lines_with_dots": ViolinSpec(_data(), inner="lines", colors=["#888888", "#d62728", "#1f77b4"], dots=True),
    "ggplot_inner_box": ViolinSpec(_data(), inner="box", colors=["#ffd27f"] * 3),
    "dark_fill": ViolinSpec(_data(2), inner="lines", colors=["#222244", "#333333", "#552222"]),
    "small_image": ViolinSpec(_data(3), inner="lines", dpi=120),
}


def _span(p):
    return abs(p.value_at(p.plot_top) - p.value_at(p.axis_y))


@pytest.mark.parametrize("name", sorted(CASES))
def test_violin_values(extractor, name):
    spec = CASES[name]
    panels = extractor.extract(render_violin(spec), VIOLIN)
    assert len(panels) == 1 and panels[0].kind == "violin"
    p = panels[0]
    assert len(p.boxes) == len(spec.data)
    span = _span(p)
    # ends move by a scatter dot's radius when a dot sits on the tip
    end_tol = 0.015 if spec.dots else 0.01
    for b, (lo, q1, med, q3, hi) in zip(p.boxes, spec.stats):
        assert abs(b.median - med) / span < 0.01, (b.median, med)
        assert abs(b.q1 - q1) / span < 0.01 and abs(b.q3 - q3) / span < 0.01, (b.q1, q1, b.q3, q3)
        assert abs(b.lo - lo) / span < end_tol and abs(b.hi - hi) / span < end_tol, (b.lo, lo, b.hi, hi)


def test_median_only_violin(extractor):
    spec = ViolinSpec(_data(), inner="median", colors=["#bbbbbb"] * 3)
    p = extractor.extract(render_violin(spec), VIOLIN)[0]
    for b, (lo, q1, med, q3, hi) in zip(p.boxes, spec.stats):
        assert abs(b.median - med) / _span(p) < 0.01
        assert b.q1 is None and b.q3 is None and not b.quartiles
    # S1 estimate from min, median, max
    e = estimate(None, p.boxes[0].median, None, 14, p.boxes[0].lo, p.boxes[0].hi)
    assert e.scenario == "S1" and e.mean is not None and e.sd > 0


def test_violin_page_ignores_other_charts(extractor):
    for img in (render(ChartSpec([12.0, 31.5, 22.0], [1.5, 3.0, 2.2])),
                render(ChartSpec([80, 40, 75, 55], [4, 3, 5, 4], dots=True, seed=3)),
                render_box(BoxSpec([(2, 5, 7, 9, 13), (4, 8, 11, 15, 19)], ylim=(0, 20)))):
        assert extractor.extract(img, VIOLIN) == []


def test_other_pages_ignore_violins(extractor):
    img = render_violin(CASES["prism_lines_colored"])
    for chart_type in ("bar", "box"):
        opts = ExtractOptions(bar_names=False, titles=False, chart_type=chart_type)
        assert extractor.extract(img, opts) == []


def test_s1_estimators_recover_normal_samples():
    rng = np.random.default_rng(0)
    for n in (10, 50):
        ms, sds = [], []
        for _ in range(400):
            x = rng.normal(50, 10, n)
            e = estimate(None, float(np.median(x)), None, n, x.min(), x.max())
            ms.append(e.mean)
            sds.append(e.sd)
        assert abs(np.mean(ms) - 50) < 0.6
        assert abs(np.mean(sds) / 10 - 1) < 0.08
