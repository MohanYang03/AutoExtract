"""
Generalisation check across a whole meta-analysis folder.

    python eval/sweep.py --data "C:/path/to/mice studies" --cache eval/out/sweep.json

Every paper folder's PDF is extracted once (cached as JSON). Each
Long_format_*.xlsx is then compared with the extracted panel that matches it
best: same number of bars, from figure N when the file name ends in _N, and
the lowest mean error. Matching on error is mildly optimistic, so read the
result as "could the right panel have been found", then check the overlay.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backend"))

from run_eval import load_truth_both  # noqa: E402


def extract_folder(folder, extractor):
    from pdf_utils import figure_number, page_images
    out = []
    for pdf in glob.glob(os.path.join(folder, "*.pdf")):
        for page, img in page_images(pdf):
            fig = figure_number(pdf, page)
            for p in extractor.extract(img):
                lo, hi = p.value_at(p.axis_y), p.value_at(p.plot_top)
                out.append(dict(pdf=os.path.basename(pdf), page=page, figure=fig, letter=p.letter,
                                title=p.y_title, lo=lo, hi=hi,
                                bars=[(b.label, b.mean, b.mean_plus_err, b.uncertain) for b in p.bars]))
    return out


def subset_match(truth, truth_sem, truth_sd, panels, want_fig):
    """
    Sheets sometimes take a few bars from one or more panels (e.g. only the
    post-treatment groups). Align the sheet's rows, in order, to an ordered
    subsequence of all extracted bars (dynamic programming on % error).
    """
    bars = []
    for p in panels:
        if p["lo"] is None or (want_fig and p["figure"] and p["figure"] != want_fig):
            continue
        rng = abs(p["hi"] - p["lo"]) or 1
        for b in p["bars"]:
            if b[1] is not None:
                bars.append((b, rng, f"Fig{p['figure']}{p['letter']}"))
    T, B = len(truth), len(bars)
    if B < T or T == 0:
        return None
    INF = float("inf")
    cost = [[INF] * (B + 1) for _ in range(T + 1)]
    back = [[None] * (B + 1) for _ in range(T + 1)]
    for j in range(B + 1):
        cost[0][j] = 0.0
    for i in range(1, T + 1):
        for j in range(i, B + 1):
            skip = cost[i][j - 1]
            take = cost[i - 1][j - 1] + 100 * abs(bars[j - 1][0][1] - truth[i - 1][1]) / bars[j - 1][1]
            if take <= skip:
                cost[i][j], back[i][j] = take, ("take", j - 1)
            else:
                cost[i][j], back[i][j] = skip, ("skip", None)
    i, j, picked = T, B, []
    while i > 0:
        kind, idx = back[i][j]
        if kind == "take":
            picked.append(idx)
            i -= 1
        j -= 1
    picked.reverse()
    em = [100 * abs(bars[k][0][1] - t[1]) / bars[k][1] for k, t in zip(picked, truth)]
    if np.mean(em) > 5:
        return None
    et = min(([100 * abs(bars[k][0][2] - t[2]) / bars[k][1] for k, t in zip(picked, tr)]
              for tr in (truth_sem, truth_sd) if tr), key=np.mean)
    return em, et, ",".join(sorted({bars[k][2] for k in picked}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", default=os.path.join(HERE, "out", "sweep.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.cache), exist_ok=True)
    cache = json.load(open(args.cache)) if os.path.exists(args.cache) else {}

    folders = sorted(glob.glob(os.path.join(args.data, "(paper*")),
                     key=lambda f: int(re.search(r"paper(\d+)", f).group(1)))
    extractor = None
    for f in folders:
        key = os.path.basename(f)
        if key in cache:
            continue
        if extractor is None:
            from figure_extractor import FigureExtractor
            extractor = FigureExtractor()
        t = time.time()
        cache[key] = extract_folder(f, extractor)
        json.dump(cache, open(args.cache, "w"))
        print(f"extracted {key}: {len(cache[key])} panels in {time.time() - t:.0f}s", flush=True)

    rows = []
    for f in folders:
        key = os.path.basename(f)
        panels = cache.get(key, [])
        for x in sorted(glob.glob(os.path.join(f, "Long_format_*.xlsx"))):
            name = os.path.basename(x)
            try:
                truth_sem, truth_sd = load_truth_both(x)
                truth = truth_sem or truth_sd
                if not truth:
                    raise ValueError("no usable rows")
            except Exception as e:
                rows.append((key, name, f"unreadable ({e})", None, None, ""))
                continue
            m = re.search(r"_(\d+)\.xlsx$", name)
            want_fig = m.group(1) if m else None
            best = None
            for p in panels:
                if len(p["bars"]) != len(truth) or p["lo"] is None:
                    continue
                if want_fig and p["figure"] and p["figure"] != want_fig:
                    continue
                rng = abs(p["hi"] - p["lo"])
                em = [100 * abs(b[1] - t[1]) / rng for b, t in zip(p["bars"], truth)]
                # compare caps with whichever error type the figure shows
                et = min(([100 * abs(b[2] - t[2]) / rng for b, t in zip(p["bars"], tr)]
                          for tr in (truth_sem, truth_sd) if tr), key=np.mean)
                if best is None or np.mean(em) < np.mean(best[0]):
                    best = (em, et, p)
            if best is None:
                sub = subset_match(truth, truth_sem, truth_sd, panels, want_fig)
                if sub is None:
                    counts = sorted({len(p["bars"]) for p in panels})
                    rows.append((key, name, f"no match (panel bar counts {counts})", None, None, ""))
                else:
                    em, et, where = sub
                    rows.append((key, name, f"subset {where}", float(np.mean(em)), float(np.mean(et)), ""))
            else:
                em, et, p = best
                rows.append((key, name, f"Fig{p['figure']}{p['letter']} {p['title'][:22]}",
                             float(np.mean(em)), float(np.mean(et)),
                             f"{sum(b[3] for b in p['bars'])}/{len(p['bars'])} flagged"))

    print(f"\n{'paper':38s} {'sheet':42s} {'matched panel':34s} {'mean%':>6s} {'cap%':>6s}")
    good = []
    for key, name, what, em, et, flags in rows:
        if em is None:
            print(f"{key[:38]:38s} {name[:42]:42s} {what}")
        else:
            good.append((em, et))
            print(f"{key[:38]:38s} {name[:42]:42s} {what[:34]:34s} {em:6.2f} {et:6.2f}  {flags}")
    g = np.array(good) if good else np.zeros((0, 2))
    print(f"\nsheets: {len(rows)}  matched: {len(g)}")
    if len(g):
        print(f"matched sheets with mean err < 1%: {(g[:, 0] < 1).sum()}   < 2%: {(g[:, 0] < 2).sum()}")
        print(f"median mean err {np.median(g[:, 0]):.2f}%   median cap err {np.median(g[:, 1]):.2f}%")


if __name__ == "__main__":
    main()
