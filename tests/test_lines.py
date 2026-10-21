"""
Line plots: synthetic charts with exactly known points and error bars.

    python -m pytest tests/test_lines.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backend"))
sys.path.insert(0, HERE)

from figure_extractor import ExtractOptions, FigureExtractor  # noqa: E402
from synth import BoxSpec, ChartSpec, LineSpec, render, render_box, render_line  # noqa: E402

LINE = ExtractOptions(bar_names=False, titles=False, chart_type="line")


@pytest.fixture(scope="module")
def extractor():
    return FigureExtractor()


CASES = {
    "two_series_errors": LineSpec([0, 7, 14, 21, 28], [[20, 22, 25, 27, 30], [16, 17, 17.5, 18, 19]],
                                  errs=[[1, 1.2, 1.1, 1.5, 1.4], [0.8, 1, 0.9, 1.1, 1.2]], legend=True),
    "three_colors": LineSpec([1, 2, 3, 4, 5, 6], [[5, 8, 6, 9, 11, 10], [3, 4, 4.5, 5, 6, 7], [9, 7, 6, 4, 3, 2]],
                             colors=["tab:red", "tab:blue", "tab:green"], markers=["o", "s", "D"],
                             errs=[[0.5] * 6, [0.4] * 6, [0.6] * 6]),
    "single_series": LineSpec([0, 1, 2, 3, 4, 5, 6, 7], [[2, 3, 5, 4, 6, 8, 7, 9]], errs=[[0.4] * 8]),
    "gridlines": LineSpec([0, 10, 20, 30], [[50, 60, 75, 70], [40, 42, 47, 52]], grid=True,
                          colors=["tab:purple", "tab:orange"], errs=[[3] * 4, [2] * 4]),
    "small_image": LineSpec([0, 1, 2, 3, 4], [[1, 2, 2.5, 3, 4]], errs=[[0.2] * 5], dpi=120),
}


def _span(p):
    return abs(p.value_at(p.plot_top) - p.value_at(p.axis_y))


def _match(series, truth_ys):
    """Extracted series whose point means are closest to a truth series."""
    best = None
    for s in series:
        means = [pt.mean for pt in s.points]
        if len(means) != len(truth_ys):
            continue
        d = float(np.mean(np.abs(np.array(means) - np.array(truth_ys))))
        if best is None or d < best[0]:
            best = (d, s)
    return best[1] if best else None


@pytest.mark.parametrize("name", sorted(CASES))
def test_line_values(extractor, name):
    spec = CASES[name]
    panels = extractor.extract(render_line(spec), LINE)
    assert len(panels) == 1 and panels[0].kind == "line"
    p = panels[0]
    span = _span(p)
    assert p.x_slope is not None, [t[2] for t in p.x_ticks]
    for k, ys in enumerate(spec.ys):
        s = _match(p.series, ys)
        assert s is not None, [[round(pt.mean, 2) for pt in sr.points] for sr in p.series]
        for pt, x, y, e in zip(s.points, spec.x, ys, spec.errs[k]):
            assert abs(pt.mean - y) / span < 0.01, (pt.mean, y)
            assert abs(pt.x - x) < 0.02 * (max(spec.x) - min(spec.x)) + 1e-9, (pt.x, x)
            if pt.uncertain and pt.mean_plus_err is None:
                continue                 # a marker hidden under another one: flagged for a check
            assert pt.mean_plus_err is not None and abs((pt.mean_plus_err - pt.mean) - e) / span < 0.015


def test_line_without_markers_sampled_at_ticks(extractor):
    spec = LineSpec([0, 10, 20, 30, 40, 50], [[1, 3, 2, 5, 4, 6]], markers=[None], colors=["tab:red"])
    p = extractor.extract(render_line(spec), LINE)[0]
    s = p.series[0]
    assert not s.markers and len(s.points) == 6
    for pt, y in zip(s.points, spec.ys[0]):
        assert abs(pt.mean - y) / _span(p) < 0.01


def test_line_page_ignores_bars_and_boxes(extractor):
    for img in (render(ChartSpec([12.0, 31.5, 22.0], [1.5, 3.0, 2.2])),
                render_box(BoxSpec([(2, 5, 7, 9, 13), (4, 8, 11, 15, 19)], ylim=(0, 20)))):
        assert extractor.extract(img, LINE) == []
