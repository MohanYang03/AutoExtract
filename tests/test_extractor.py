"""
Accuracy tests on synthetic charts with known values.

    .venv/Scripts/python -m pytest tests -q

Tolerance is 1% of the axis range for means and 1.5% for error-bar caps
(the cap is a thin line, so a pixel matters more).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backend"))
sys.path.insert(0, HERE)

from figure_extractor import ExtractOptions, FigureExtractor, parse_number  # noqa: E402
from synth import ChartSpec, render  # noqa: E402

FAST = ExtractOptions(bar_names=False, titles=False)


@pytest.fixture(scope="module")
def extractor():
    return FigureExtractor()


CASES = {
    "outlined_white": ChartSpec([12.0, 31.5, 22.0, 8.4], [1.5, 3.0, 2.2, 0.9]),
    "filled_colors": ChartSpec([45, 72, 60], [5, 6, 4], colors=["#4c72b0", "#dd8452", "#55a868"]),
    "dark_fill": ChartSpec([2.4, 1.1, 3.3], [0.3, 0.2, 0.4], colors=["#222244", "#333333", "#552222"]),
    "no_outline": ChartSpec([150, 210, 95, 180], [12, 20, 9, 15], colors=["#e377c2", "#17becf", "#bcbd22", "#7f7f7f"], edgecolor=None),
    "colored_error": ChartSpec([30, 52, 41], [3, 5, 4], colors=["#ffd27f", "#b0c4de", "#d8bfd8"],
                               edgecolor=None, error_color="#7a5cb8"),
    "with_dots": ChartSpec([80, 40, 75, 55], [4, 3, 5, 4], dots=True, seed=3),
    "gridlines": ChartSpec([0.62, 0.81, 0.44], [0.05, 0.06, 0.04], grid=True, colors=["#999999"] * 3),
    "frame": ChartSpec([3200, 1800, 2600], [180, 150, 210], frame=True),
    "many_bars": ChartSpec([5, 9, 7, 12, 3, 8, 10, 6], [0.5, 0.8, 0.6, 1.0, 0.4, 0.7, 0.9, 0.5],
                           figsize=(5.0, 3.0)),
    "small_image": ChartSpec([18, 27, 22], [2, 3, 2], dpi=110),
    "decimal_ticks": ChartSpec([0.35, 1.25, 0.8], [0.1, 0.15, 0.12], ylim=(0, 1.5), yticks=[0, 0.5, 1.0, 1.5]),
    "negative": ChartSpec([2.5, -4.5, -5.5, 1.0], [0.8, 1.2, 1.5, 0.5], ylim=(-10, 5),
                          colors=["#666666", "#d62728", "#1f77b4", "#ff7f0e"]),
    "nonzero_floor": ChartSpec([62, 85, 74], [3, 4, 3], ylim=(50, 100), yticks=[50, 60, 70, 80, 90, 100]),
    "dot_plot": ChartSpec([110, 170, 126, 116], [44, 52, 35, 45], kind="dot", dots=True,
                          colors=["#1f77b4", "#d62728", "#2ca02c", "#9467bd"], ylim=(0, 300)),
}


def _axis_range(panel):
    return abs(panel.value_at(panel.plot_top) - panel.value_at(panel.axis_y))


@pytest.mark.parametrize("name", sorted(CASES))
def test_values(extractor, name):
    spec = CASES[name]
    panels = extractor.extract(render(spec), FAST)
    assert len(panels) == 1, f"expected one panel, found {len(panels)}"
    p = panels[0]
    assert len(p.bars) == len(spec.means), [round(b.mean, 3) for b in p.bars]
    rng = _axis_range(p)
    for bar, m, e in zip(p.bars, spec.means, spec.errs):
        assert abs(bar.mean - m) / rng < 0.01, f"mean {bar.mean:.4g} vs {m}"
        got_err = bar.mean_plus_err - bar.mean
        if not bar.uncertain:
            assert abs(got_err - abs(e)) / rng < 0.015, f"error {got_err:.4g} vs {e}"


def test_kind_dot(extractor):
    p = extractor.extract(render(CASES["dot_plot"]), FAST)[0]
    assert p.kind == "dot"


def test_bar_names(extractor):
    spec = ChartSpec([10, 20, 15], [1, 2, 1], labels=["Control", "CUMS", "CUMS+EA"])
    p = extractor.extract(render(spec), ExtractOptions(bar_names=True, titles=True))[0]
    assert [b.label for b in p.bars] == ["Control", "CUMS", "CUMS+EA"]
    assert "Value" in p.y_title


def test_blank_and_photo_images(extractor):
    assert extractor.extract(np.full((400, 600, 3), 255, np.uint8)) == []
    noise = np.random.default_rng(1).integers(0, 255, (500, 700, 3), dtype=np.uint8)
    assert extractor.extract(noise) == []
    assert extractor.extract(np.zeros((0, 0, 3), np.uint8)) == []


def test_grayscale_and_alpha_inputs(extractor):
    img = render(CASES["outlined_white"])
    gray = img[:, :, 1].copy()
    bgra = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
    assert len(extractor.extract(gray, FAST)) == 1
    assert len(extractor.extract(bgra, FAST)) == 1


@pytest.mark.parametrize("text,value", [
    ("100", 100), ("0.5", 0.5), ("-10", -10), ("−5", -5), ("2.5.", 2.5), ("50°", 50),
    ("1,000", 1000), ("。", 0), ("O", 0), ("Control", None), ("1 000", None), ("***", None),
])
def test_parse_number(text, value):
    assert parse_number(text) == value
