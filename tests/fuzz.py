"""
Randomised stress test: many synthetic charts with random values and styles.

    .venv/Scripts/python tests/fuzz.py [N] [--seed S] [--show-failures DIR]

Reports detection rate and error distribution; failures can be saved as
images for inspection. Exit code 1 if fewer than 95% of charts are measured
with every mean within 1% of the axis range.
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
from synth import ChartSpec, render  # noqa: E402

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--show-failures", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    ex = FigureExtractor()
    opts = ExtractOptions(bar_names=False, titles=False)
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
