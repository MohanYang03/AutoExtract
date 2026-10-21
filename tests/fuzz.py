"""
Randomised stress test: many synthetic charts with random values and styles.

    .venv/Scripts/python tests/fuzz.py [N] [--seed S] [--show-failures DIR] [--box]

Reports detection rate and error distribution; failures can be saved as
images for inspection. Exit code 1 if fewer than 95% of charts are measured
with every mean within 1% of the axis range (--box: every box statistic).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backend"))
sys.path.insert(0, HERE)

from figure_extractor import ExtractOptions, FigureExtractor  # noqa: E402
from synth import BoxSpec, ChartSpec, render, render_box  # noqa: E402

PALETTES = [
    None,
    ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860", "#da8bc3", "#8c8c8c"],
    ["#222222", "#888888", "#cccccc", "#555555", "#aaaaaa", "#333333", "#999999", "#666666"],
    ["#ffd27f", "#b0c4de", "#d8bfd8", "#98fb98", "#f4a460", "#87ceeb", "#ffb6c1", "#e0e0e0"],
]


def random_spec(rng: np.random.Generator) -> ChartSpec:
    n = int(rng.integers(2, 9))
    scale = float(10.0 ** rng.integers(-1, 4))
    kind = "dot" if rng.random() < 0.15 else "bar"
    negative = kind == "bar" and rng.random() < 0.12
    means = rng.uniform(0.15, 1.0, n) * scale
    if negative:
        means *= rng.choice([-1, 1], n)
    errs = means * 0 + rng.uniform(0.02, 0.15, n) * scale
    pal = PALETTES[int(rng.integers(0, len(PALETTES)))]
    colors = None if pal is None else [pal[i % len(pal)] for i in range(n)]
    edge = None if (colors and rng.random() < 0.4) else "black"
    err_color = "black" if (edge or rng.random() < 0.5) else "#555599"
    ylim = None
    if negative:
        lo = float(min(0, (means - errs).min()) * 1.3)
        hi = float(max(0, (means + errs).max()) * 1.3)
        ylim = (lo, hi)
    return ChartSpec(
        means=[float(m) for m in means], errs=[float(e) for e in errs], kind=kind,
        colors=colors, edgecolor=edge, error_color=err_color,
        linewidth=float(rng.choice([1.0, 1.5, 2.0])),
        dots=bool(rng.random() < 0.35), grid=bool(rng.random() < 0.15), frame=bool(rng.random() < 0.15),
        ylim=ylim, figsize=(float(1.6 + 0.45 * n), float(rng.uniform(2.4, 3.6))),
        dpi=int(rng.choice([110, 150, 200, 300])), seed=int(rng.integers(0, 10 ** 6)),
    )


def random_box_spec(rng: np.random.Generator) -> BoxSpec:
    n = int(rng.integers(1, 7))
    scale = float(10.0 ** rng.integers(-1, 4))
    stats = []
    for _ in range(n):
        q = np.sort(rng.uniform(0.05, 0.95, 5)) * scale
        # keep the box and whiskers at least a few pixels apart
        if min(np.diff(q)) < 0.03 * scale:
            q = np.linspace(0.1, 0.9, 5) * scale * rng.uniform(0.6, 1.0)
        stats.append(tuple(float(v) for v in q))
    pal = PALETTES[int(rng.integers(0, len(PALETTES)))]
    colors = None if pal is None else [pal[i % len(pal)] for i in range(n)]
    outliers = None
    if rng.random() < 0.3:
        outliers = [[s[4] + 0.04 * scale] if rng.random() < 0.5 else [] for s in stats]
    return BoxSpec(
        stats=stats, colors=colors, outliers=outliers, ylim=(0, scale),
        linewidth=float(rng.choice([0.8, 1.2, 1.8])), caps=bool(rng.random() < 0.8),
        dashed=bool(rng.random() < 0.15), grid=bool(rng.random() < 0.15), frame=bool(rng.random() < 0.15),
        figsize=(float(1.8 + 0.5 * n), float(rng.uniform(2.4, 3.6))), dpi=int(rng.choice([110, 150, 200, 300])),
    )


def fuzz_boxes(n: int, rng, ex, opts, show) -> int:
    ok, errs, failures = 0, [], []
    t0 = time.time()
    for i in range(n):
        spec = random_box_spec(rng)
        img = render_box(spec)
        panels = ex.extract(img, opts)
        reason = None
        if len(panels) != 1 or panels[0].kind != "box":
            reason = f"{len(panels)} panels" + (f" ({panels[0].kind})" if panels else "")
        elif len(panels[0].boxes) != len(spec.stats):
            reason = f"{len(panels[0].boxes)}/{len(spec.stats)} boxes"
        else:
            span = spec.ylim[1] - spec.ylim[0]
            e = [abs(got - want) / span * 100 if got is not None else 100.0
                 for b, st in zip(panels[0].boxes, spec.stats)
                 for got, want in zip((b.lo, b.q1, b.median, b.q3, b.hi), st)]
            errs += e
            if max(e) >= 1:
                reason = f"stat err {max(e):.2f}%"
        if reason is None:
            ok += 1
        else:
            failures.append((i, spec, reason))
            if show:
                os.makedirs(show, exist_ok=True)
                cv2.imwrite(os.path.join(show, f"box_fail_{i:03d}.png"), img)
    dt = time.time() - t0
    print(f"box plots: {n}  fully correct: {ok} ({100 * ok / n:.1f}%)  time {dt:.1f}s ({1000 * dt / n:.0f} ms/chart)")
    if errs:
        e = np.array(errs)
        print(f"stat err: median {np.median(e):.3f}%  p95 {np.percentile(e, 95):.3f}%  max {e.max():.2f}%")
    for i, spec, reason in failures[:25]:
        print(f"  #{i}: {reason}  n={len(spec.stats)} colors={'yes' if spec.colors else 'no'} caps={spec.caps} "
              f"dashed={spec.dashed} grid={spec.grid} frame={spec.frame} lw={spec.linewidth} dpi={spec.dpi}")
    return 0 if ok >= 0.95 * n else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--show-failures", default=None)
    ap.add_argument("--box", action="store_true", help="random box plots instead of bar charts")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    ex = FigureExtractor()
    opts = ExtractOptions(bar_names=False, titles=False)
    if args.box:
        return fuzz_boxes(args.n, rng, ex, opts, args.show_failures)
    ok = 0
    mean_errs, cap_errs, flagged, failures = [], [], 0, []
    t0 = time.time()
    for i in range(args.n):
        spec = random_spec(rng)
        img = render(spec)
        panels = ex.extract(img, opts)
        reason = None
        if len(panels) != 1:
            reason = f"{len(panels)} panels"
        elif len(panels[0].bars) != len(spec.means):
            reason = f"{len(panels[0].bars)}/{len(spec.means)} bars"
        else:
            p = panels[0]
            rngv = abs(p.value_at(p.plot_top) - p.value_at(p.axis_y))
            em = [abs(b.mean - m) / rngv * 100 for b, m in zip(p.bars, spec.means)]
            mean_errs += em
            for b, e in zip(p.bars, spec.errs):
                if b.uncertain:
                    flagged += 1
                else:
                    cap_errs.append(abs((b.mean_plus_err - b.mean) - e) / rngv * 100)
            if max(em) >= 1:
                reason = f"mean err {max(em):.2f}%"
        if reason is None:
            ok += 1
        else:
            failures.append((i, spec, reason))
            if args.show_failures:
                os.makedirs(args.show_failures, exist_ok=True)
                cv2.imwrite(os.path.join(args.show_failures, f"fail_{i:03d}.png"), img)
    dt = time.time() - t0
    print(f"charts: {args.n}  fully correct: {ok} ({100 * ok / args.n:.1f}%)  time {dt:.1f}s ({1000 * dt / args.n:.0f} ms/chart)")
    if mean_errs:
        me, ce = np.array(mean_errs), np.array(cap_errs) if cap_errs else np.zeros(1)
        print(f"mean err: median {np.median(me):.3f}%  p95 {np.percentile(me, 95):.3f}%  max {me.max():.2f}%")
        print(f"cap err (unflagged): median {np.median(ce):.3f}%  p95 {np.percentile(ce, 95):.3f}%  max {ce.max():.2f}%  "
              f"| flagged {flagged} of {flagged + len(cap_errs)}")
    for i, spec, reason in failures[:25]:
        print(f"  #{i}: {reason}  kind={spec.kind} n={len(spec.means)} edge={spec.edgecolor} colors={'yes' if spec.colors else 'no'} "
              f"dots={spec.dots} grid={spec.grid} frame={spec.frame} dpi={spec.dpi} neg={spec.ylim is not None}")
    return 0 if ok >= 0.95 * args.n else 1


if __name__ == "__main__":
    sys.exit(main())
