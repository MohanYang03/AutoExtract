"""
Box plots: detection and five-number summary on synthetic charts with exact
statistics, and the mean/SD estimators (Python and the Excel formulas).

    python -m pytest tests/test_boxes.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backend"))
sys.path.insert(0, HERE)

from estimators import MEAN_METHODS, SD_METHODS, estimate, skewness_warning  # noqa: E402
from exporter import box_formulas  # noqa: E402
from figure_extractor import ExtractOptions, FigureExtractor  # noqa: E402
from synth import BoxSpec, ChartSpec, render, render_box  # noqa: E402

FAST = ExtractOptions(bar_names=False, titles=False, chart_type="box")
STATS4 = [(2, 5, 7, 9, 13), (4, 8, 11, 15, 19), (1, 3, 4, 6, 8), (6, 9, 12, 13, 17)]


@pytest.fixture(scope="module")
def extractor():
    return FigureExtractor()


CASES = {
    "white": BoxSpec(STATS4, ylim=(0, 20)),
    "filled": BoxSpec(STATS4, colors=["#4c72b0", "#dd8452", "#55a868", "#c44e52"], ylim=(0, 20)),
    "light_fill": BoxSpec(STATS4, colors=["#dddddd"] * 4, ylim=(0, 20)),
    "outliers": BoxSpec(STATS4, outliers=[[16], [1, 2], [], [19.5]], ylim=(0, 20)),
    "mean_marker": BoxSpec(STATS4, means=[8.2, 12.8, 5.0, 10.6], ylim=(0, 20)),
    "no_caps": BoxSpec(STATS4, caps=False, ylim=(0, 20)),
    "dashed": BoxSpec(STATS4, dashed=True, ylim=(0, 20)),
    "gridlines": BoxSpec(STATS4, grid=True, ylim=(0, 20)),
    "frame": BoxSpec(STATS4, frame=True, ylim=(0, 20)),
    "small_image": BoxSpec(STATS4[:3], dpi=110, ylim=(0, 20)),
    "decimals": BoxSpec([(0.1, 0.32, 0.41, 0.55, 0.8), (0.2, 0.45, 0.6, 0.7, 0.95)], ylim=(0, 1),
                        yticks=[0, 0.25, 0.5, 0.75, 1.0]),
    "nonzero_floor": BoxSpec([(52, 60, 66, 71, 80), (58, 70, 74, 79, 92)], ylim=(50, 100)),
    "log_axis": BoxSpec([(2, 5, 9, 20, 60), (8, 15, 30, 70, 200)], log=True, ylim=(1, 1000),
                        yticks=[1, 10, 100, 1000]),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_box_values(extractor, name):
    spec = CASES[name]
    panels = extractor.extract(render_box(spec), FAST)
    assert len(panels) == 1, f"expected one panel, found {len(panels)}"
    p = panels[0]
    assert p.kind == "box"
    assert len(p.boxes) == len(spec.stats), [(b.q1, b.median, b.q3) for b in p.boxes]
    if spec.log:
        assert p.scale == "log"
        span = np.log10(spec.ylim[1]) - np.log10(spec.ylim[0])

        def err(got, want):
            return abs(np.log10(got) - np.log10(want)) / span
    else:
        span = spec.ylim[1] - spec.ylim[0]

        def err(got, want):
            return abs(got - want) / span
    for b, (lo, q1, med, q3, hi) in zip(p.boxes, spec.stats):
        for got, want, what in ((b.q1, q1, "Q1"), (b.median, med, "median"), (b.q3, q3, "Q3"),
                                (b.lo, lo, "min"), (b.hi, hi, "max")):
            assert got is not None and err(got, want) < 0.01, f"{what} {got} vs {want}"
    if spec.means:
        for b, m in zip(p.boxes, spec.means):
            assert b.mean_marker is not None and err(b.mean_marker, m) < 0.015
    if spec.outliers:
        for b, outs in zip(p.boxes, spec.outliers):
            assert len(b.outliers) == len(outs), (b.outliers, outs)


def test_box_page_ignores_bar_charts(extractor):
    """Papers mix chart types: bar charts and dot plots give no box panels."""
    for spec in (ChartSpec([12.0, 31.5, 22.0], [1.5, 3.0, 2.2]),
                 ChartSpec([80, 40, 75, 55], [4, 3, 5, 4], dots=True, seed=3),
                 ChartSpec([110, 170, 126], [44, 52, 35], kind="dot", dots=True, ylim=(0, 300))):
        assert extractor.extract(render(spec), FAST) == []


def test_bar_page_ignores_box_plots(extractor):
    bar_opts = ExtractOptions(bar_names=False, titles=False, chart_type="bar")
    panels = extractor.extract(render_box(CASES["white"]), bar_opts)
    assert all(p.kind != "box" for p in panels)


def test_chart_type_option():
    assert ExtractOptions.from_dict({"chart_type": "box", "titles": 0}).chart_type == "box"
    assert ExtractOptions.from_dict({}).chart_type == "bar"
    with pytest.raises(ValueError):
        ExtractOptions.from_dict({"chart_type": "pie"})


# ---- estimators -------------------------------------------------------------- #

def test_estimators_recover_normal_samples():
    """Averaged over many normal samples, mean and SD estimates are nearly unbiased."""
    rng = np.random.default_rng(0)
    for n in (10, 50, 200):
        for mm, sm, whisk in (("luo", "shi", "minmax"), ("wan", "wan", "minmax"), ("luo", "shi", "tukey")):
            ms, sds = [], []
            for _ in range(400):
                x = rng.normal(50, 10, n)
                q1, med, q3 = np.percentile(x, [25, 50, 75])
                e = estimate(q1, med, q3, n, x.min(), x.max(), whisk, mm, sm)
                ms.append(e.mean)
                sds.append(e.sd)
            assert abs(np.mean(ms) - 50) < 0.6, (n, mm, np.mean(ms))
            assert abs(np.mean(sds) / 10 - 1) < 0.06, (n, sm, whisk, np.mean(sds))


def test_estimate_needs_n():
    e = estimate(3, 4, 6, None, 1, 9)
    assert e.mean is None and e.sd is None and e.note


def test_scenario_follows_whiskers():
    assert estimate(3, 4, 6, 20, 1, 9, "minmax").scenario == "S2"
    for w in ("tukey", "percentile", "unknown"):
        assert estimate(3, 4, 6, 20, 1, 9, w).scenario == "S3"


def test_skewness_warning():
    assert not skewness_warning(3, 4.5, 6)
    assert skewness_warning(3, 3.2, 6)


@pytest.mark.parametrize("whisk", ["minmax", "tukey"])
@pytest.mark.parametrize("mm", MEAN_METHODS)
@pytest.mark.parametrize("sm", SD_METHODS)
def test_excel_formulas_match_python(whisk, mm, sm):
    """The workbook's live formulas compute the same estimate as the backend."""
    env = dict(A=1.0, B=3.0, C=4.2, D=6.0, E=11.0, NORMSINV=norm.ppf, SQRT=np.sqrt,
               IF=lambda c, a, b: a if c else b)
    for n in (5, 12, 40, 100):
        env["N"] = n
        mean_f, sd_f = box_formulas("A", "B", "C", "D", "E", "N", whisk, mm, sm)
        e = estimate(3.0, 4.2, 6.0, n, 1.0, 11.0, whisk, mm, sm)
        assert eval(mean_f.replace("^", "**"), env) == pytest.approx(e.mean, rel=1e-9)
        assert eval(sd_f.replace("^", "**"), env) == pytest.approx(e.sd, rel=1e-9)
