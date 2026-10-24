"""
Benchmark against the ICPR 2022 CHART-Info UB-Unitec PMC test set.

    python eval/chartinfo_eval.py --data C:/Users/gordo/chartinfo_data/test/ICPR2022_CHARTINFO_UB_UNITEC_PMC_TEST_v2.1
        [--type box|bar] [--limit N] [--show DIR] [--list]

Real PubMed Central charts with hand-annotated ground truth (task 6). Each
ground-truth element is matched to a detected box/bar by its pixel x
position; errors are reported as % of the y-axis range (from the ground
truth tick values) so charts of any scale are comparable.

Box plots: min, Q1, median, Q3, max per box.
Bars:      the value of each bar. Stacked bars (several elements in one
           column) are skipped: the tool reports grouped/single bars only.
The dataset is CC BY-NC-SA 4.0: use for evaluation only, never commit it.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backend"))

BOX_STATS = ("min", "first_quartile", "median", "third_quartile", "max")
OUR_STATS = ("lo", "q1", "median", "q3", "hi")


def load_cases(root, chart_type):
    out = []
    for f in sorted(glob.glob(os.path.join(root, "final_full_GT", "split_[45]", "annotations_JSON", "*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        if d["task1"]["output"]["chart_type"] != chart_type:
            continue
        t6 = d.get("task6") or {}
        if not t6.get("output"):
            continue
        name = os.path.basename(f)[:-5]
        imgs = glob.glob(os.path.join(root, "chart_images", "*", "images", name + ".*"))
        if imgs:
            out.append((name, imgs[0], d))
    return out


def gt_axis_range(d):
    """Value range of the y-axis from the annotated tick labels (task 4 + task 2/3 text)."""
    cal = gt_calibration(d)
    if cal is None:
        return None, None
    return cal[3], d["task4"]["output"]["_plot_bb"]


def gt_calibration(d):
    """
    The annotated y-axis: (to_value function, is_log, ticks, value range),
    fitted from the annotated tick positions and tick label text. Ground truth
    for each element is its annotated pixel row through this mapping, which
    needs no assumption about how the data-series list is ordered.
    """
    try:
        from figure_extractor import parse_number
        texts = {b["id"]: b["text"] for b in d["task2"]["output"]["text_blocks"]}
        ticks = d["task4"]["output"]["axes"]["y-axis"]
        pts = [(t["tick_pt"]["y"], parse_number(texts[t["id"]])) for t in ticks if t["id"] in texts]
        pts = [(y, v) for y, v in pts if v is not None]
    except Exception:
        return None
    if len(pts) < 2:
        return None
    ys = np.array([p[0] for p in pts], float)
    vs = np.array([p[1] for p in pts], float)
    a, b = np.polyfit(ys, vs, 1)
    lin_res = np.abs(a * ys + b - vs).max() / max(np.ptp(vs), 1e-9)
    if lin_res < 0.02:
        return (lambda y: a * y + b), False, pts, float(np.ptp(vs))
    if (vs > 0).all() and len(pts) >= 3:
        la, lb = np.polyfit(ys, np.log10(vs), 1)
        log_res = np.abs(la * ys + lb - np.log10(vs)).max() / max(np.ptp(np.log10(vs)), 1e-9)
        if log_res < 0.02:
            return (lambda y: 10 ** (la * y + lb)), True, pts, float(np.ptp(vs))
    return None


def eval_box(cases, ex, opts, show=None):
    rows, found_charts, n_gt, n_matched = [], 0, 0, 0
    for name, path, d in cases:
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        panels = ex.extract(img, opts)
        gt = d["task6"]["output"]["visual elements"]["boxplots"]
        n_gt += len(gt)
        cal = gt_calibration(d)
        rng = cal[3] if cal else None
        boxes = [b for p in panels if p.kind == "box" for b in p.boxes]
        if boxes:
            found_charts += 1
        if show is not None:
            from exporter import draw_overlay
            os.makedirs(show, exist_ok=True)
            cv2.imwrite(os.path.join(show, name + ".jpg"), draw_overlay(img, panels))
        if not boxes or not rng:
            continue
        to_value = cal[0]
        for g in gt:
            if "median" not in g:
                continue
            gx = g["median"]["x"]
            near = [b for b in boxes if b.x1 - 3 <= gx <= b.x2 + 3]
            if not near:
                continue
            b = min(near, key=lambda b: abs((b.x1 + b.x2) / 2 - gx))
            n_matched += 1
            errs = []
            for gs, os_ in zip(BOX_STATS, OUR_STATS):
                if gs in g and getattr(b, os_) is not None:
                    errs.append(100 * abs(getattr(b, os_) - to_value(g[gs]["y"])) / rng)
                else:
                    errs.append(np.nan)
            rows.append((name, errs, b.uncertain))
    return rows, found_charts, n_gt, n_matched


def eval_bar(cases, ex, opts, show=None):
    rows, found_charts, n_gt, n_matched, skipped = [], 0, 0, 0, 0
    for name, path, d in cases:
        vis = d["task6"]["output"]["visual elements"]["bars"]
        # stacked: several bar elements share a column
        xs = [round(b["x0"] + b["width"] / 2) for b in vis]
        if len(xs) != len(set(xs)):
            skipped += 1
            continue
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        panels = ex.extract(img, opts)
        cal = gt_calibration(d)
        bars = [b for p in panels if p.kind in ("bar", "dot") for b in p.bars]
        n_gt += len(vis)
        if bars:
            found_charts += 1
        if show is not None:
            from exporter import draw_overlay
            os.makedirs(show, exist_ok=True)
            cv2.imwrite(os.path.join(show, name + ".jpg"), draw_overlay(img, panels))
        if not bars or cal is None:
            continue
        to_value, _, _, rng = cal
        for g in vis:
            gx = g["x0"] + g["width"] / 2
            near = [b for b in bars if b.x1 - 3 <= gx <= b.x2 + 3]
            if not near:
                continue
            b = min(near, key=lambda b: abs((b.x1 + b.x2) / 2 - gx))
            # the bar's value end is its edge away from the baseline: the top
            # for upward bars, the bottom for bars hanging below zero
            top_v, bot_v = to_value(g["y0"]), to_value(g["y0"] + g["height"])
            truth = top_v if abs(top_v) >= abs(bot_v) else bot_v
            n_matched += 1
            rows.append((name, [100 * abs(b.mean - truth) / rng], b.uncertain))
    return rows, found_charts, n_gt, n_matched, skipped


def eval_line(cases, ex, opts, show=None):
    """
    Each annotated line is matched to the extracted series that follows it
    best; every annotated point inside that series' x range is scored by the
    series' height there (% of the y range). Points outside it are missed.
    """
    rows, found_charts, n_gt, n_matched = [], 0, 0, 0
    for name, path, d in cases:
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        panels = ex.extract(img, opts)
        gt_lines = d["task6"]["output"]["visual elements"]["lines"]
        n_gt += sum(len(gl) for gl in gt_lines)
        series = [s for p in panels if p.kind == "line" for s in p.series]
        if series:
            found_charts += 1
        if show is not None:
            from exporter import draw_overlay
            os.makedirs(show, exist_ok=True)
            cv2.imwrite(os.path.join(show, name + ".jpg"), draw_overlay(img, panels))
        cal = gt_calibration(d)
        if not series or cal is None:
            continue
        to_value, rng = cal[0], cal[3]
        used = set()
        for gl in sorted(gt_lines, key=len, reverse=True):
            best = None
            for k, s in enumerate(series):
                if k in used:
                    continue
                errs = [abs(s.y_at(pt["x"]) - pt["y"]) for pt in gl if s.y_at(pt["x"]) is not None]
                if len(errs) >= 0.5 * len(gl) and (best is None or np.median(errs) < best[0]):
                    best = (float(np.median(errs)), k)
            if best is None:
                continue
            used.add(best[1])
            s = series[best[1]]
            for pt in gl:
                y = s.y_at(pt["x"])
                if y is None:
                    continue
                n_matched += 1
                rows.append((name, [100 * abs(to_value(y) - to_value(pt["y"])) / rng], False))
    return rows, found_charts, n_gt, n_matched


def summarise(label, rows, found, n_charts, n_gt, n_matched, stat_names):
    print(f"\n=== {label}")
    print(f"charts: {n_charts}   with detections: {found} ({100 * found / max(n_charts, 1):.0f}%)")
    print(f"ground-truth elements: {n_gt}   matched: {n_matched} ({100 * n_matched / max(n_gt, 1):.0f}%)")
    if not rows:
        return
    E = np.array([r[1] for r in rows], float)
    for j, s in enumerate(stat_names):
        col = E[:, j][~np.isnan(E[:, j])]
        if col.size:
            print(f"  {s:>8s}: median {np.median(col):.2f}%  p90 {np.percentile(col, 90):.2f}%  "
                  f"<1%: {100 * (col < 1).mean():.0f}%  <2%: {100 * (col < 2).mean():.0f}%  <5%: {100 * (col < 5).mean():.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--type", choices=["box", "bar", "both", "line"], default="both")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show", default=None)
    ap.add_argument("--list", action="store_true", help="print per-chart results")
    ap.add_argument("--cross", action="store_true", help="also run each page on the other chart type")
    args = ap.parse_args()

    from figure_extractor import ExtractOptions, FigureExtractor
    ex = FigureExtractor()
    bar_opts = ExtractOptions(bar_names=False, titles=False, chart_type="bar")
    box_opts = ExtractOptions(bar_names=False, titles=False, chart_type="box")
    line_opts = ExtractOptions(bar_names=False, titles=False, chart_type="line")
    t0 = time.time()
    if args.type == "line":
        cases = load_cases(args.data, "line")[: args.limit or None]
        rows, found, n_gt, n_m = eval_line(cases, ex, line_opts, args.show and os.path.join(args.show, "line"))
        summarise("line plots", rows, found, len(cases), n_gt, n_m, ["y"])
        if args.list:
            per = {}
            for name, errs, _ in rows:
                per.setdefault(name, []).append(errs[0])
            for name, e in per.items():
                print(f"   {name[:50]:50s} n={len(e):3d} median {np.median(e):6.2f}")
        print(f"\n{time.time() - t0:.0f}s")
        return
    t0 = time.time()
    if args.type in ("box", "both"):
        cases = load_cases(args.data, "vertical box")[: args.limit or None]
        rows, found, n_gt, n_m = eval_box(cases, ex, box_opts, args.show and os.path.join(args.show, "box"))
        summarise("vertical box plots", rows, found, len(cases), n_gt, n_m, ["min", "Q1", "median", "Q3", "max"])
        if args.list:
            for name, errs, unc in rows:
                print(f"   {name[:50]:50s} " + " ".join(f"{e:6.2f}" for e in errs) + (" FLAG" if unc else ""))
    if args.type in ("bar", "both"):
        cases = load_cases(args.data, "vertical bar")[: args.limit or None]
        rows, found, n_gt, n_m, skipped = eval_bar(cases, ex, bar_opts, args.show and os.path.join(args.show, "bar"))
        print(f"\n(stacked bar charts skipped: {skipped})")
        summarise("vertical bars (non-stacked)", rows, found, len(cases) - skipped, n_gt, n_m, ["value"])
        if args.list:
            for name, errs, unc in rows:
                print(f"   {name[:50]:50s} {errs[0]:6.2f}" + (" FLAG" if unc else ""))
    if args.cross:
        # Each page meets the other kind of chart in real papers: count the
        # charts where it wrongly reports something.
        violin_opts = ExtractOptions(bar_names=False, titles=False, chart_type="violin")
        for label, chart_type, opts in (("box page on bar charts", "vertical bar", box_opts),
                                        ("bar page on box plots", "vertical box", bar_opts),
                                        ("bar page on line plots", "line", bar_opts),
                                        ("box page on line plots", "line", box_opts),
                                        ("line page on bar charts", "vertical bar", line_opts),
                                        ("line page on box plots", "vertical box", line_opts),
                                        ("violin page on bar charts", "vertical bar", violin_opts),
                                        ("violin page on box plots", "vertical box", violin_opts),
                                        ("violin page on line plots", "line", violin_opts)):
            cases = load_cases(args.data, chart_type)[: args.limit or None]
            wrong = 0
            for name, path, d in cases:
                img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
                wrong += bool(ex.extract(img, opts))
            print(f"\n{label}: {wrong} of {len(cases)} charts give panels ({100 * wrong / max(len(cases), 1):.0f}%)")
    print(f"\n{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
