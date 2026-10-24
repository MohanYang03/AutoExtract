"""
Accuracy benchmark: compare extracted bar values against hand-digitized spreadsheets.

    python eval/run_eval.py --data "C:/path/to/mice studies" [--method new|yolo] [--debug out_dir]

Error is reported as % of the panel's y-axis range (the standard digitizer metric),
for both the bar mean and the mean+error (cap) value.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backend"))

from pdf_utils import page_images  # noqa: E402


def load_truth(path, error_type="auto"):
    """
    [(group, mean, cap)] from a hand-digitized long-format sheet.

    The cap is what the figure's error bar actually shows. Sheets record it as
    'mean+sem' or 'mean+sd'; some only record sd and n. With error_type='auto'
    returns both candidates via load_truth_both().
    """
    rows_sem, rows_sd = load_truth_both(path)
    if error_type == "sd":
        return rows_sd
    return rows_sem if rows_sem else rows_sd


def load_truth_both(path):
    """(rows with cap = mean+SEM, rows with cap = mean+SD); either may be empty."""
    ws = openpyxl.load_workbook(path, data_only=True).active
    rows = list(ws.iter_rows(values_only=True))
    hdr = [str(h).strip().lower() if h else "" for h in rows[0]]
    col = {h: i for i, h in enumerate(hdr) if h}
    i_mean, i_grp = col["mean"], col.get("treatment_group")
    sem_rows, sd_rows = [], []
    for r in rows[1:]:
        if not isinstance(r[i_mean], (int, float)):
            continue
        get = lambda name: r[col[name]] if name in col and isinstance(r[col[name]], (int, float)) else None
        mean, sd, sem, n = float(r[i_mean]), get("sd"), get("sem"), get("sample_size")
        grp = r[i_grp] if i_grp is not None else ""
        top_sem = get("mean+sem")
        top_sd = get("mean+sd")
        if top_sem is None and sem is not None:
            top_sem = mean + sem
        if top_sd is None and sd is not None:
            top_sd = mean + sd
        if top_sem is None and sd is not None and n:
            top_sem = mean + sd / n ** 0.5
        if top_sem is not None:
            sem_rows.append((grp, mean, float(top_sem)))
        if top_sd is not None:
            sd_rows.append((grp, mean, float(top_sd)))
    n_rows = sum(isinstance(r[i_mean], (int, float)) for r in rows[1:])
    return (sem_rows if len(sem_rows) == n_rows else [], sd_rows if len(sd_rows) == n_rows else [])


def run_new(img, point, extractor):
    panels = extractor.extract(img)
    px, py = point
    hit = [p for p in panels if p.axis_x <= px <= p.plot_right and p.plot_top <= py <= p.axis_y]
    if not hit:
        return None, None
    p = hit[0]
    lo = p.value_at(p.axis_y)
    hi = p.value_at(p.plot_top)
    return [(b.label, b.mean, b.mean_plus_err, b.uncertain) for b in p.bars], (lo, hi, p)


def run_yolo(img, point, analyzer, extractor):
    """Original pipeline, fed the same panel crop the new method found."""
    panels = extractor.extract(img)
    px, py = point
    hit = [p for p in panels if p.axis_x <= px <= p.plot_right and p.plot_top <= py <= p.axis_y]
    if not hit:
        return None, None
    p = hit[0]
    H, W = img.shape[:2]
    x1 = int(max(0, p.axis_x - 0.12 * W)); x2 = int(min(W, p.plot_right + 20))
    y1 = int(max(0, p.plot_top - 0.04 * H)); y2 = int(min(H, p.axis_y + 0.06 * H))
    crop = img[y1:y2, x1:x2]
    tmp = os.path.join(HERE, "_yolo_tmp.png")
    cv2.imwrite(tmp, crop)
    try:
        analyzer.detect_box(tmp)
        res = analyzer.calculate_heights()
    except Exception as e:
        return f"failed: {e}", (p.value_at(p.axis_y), p.value_at(p.plot_top), p)
    finally:
        os.remove(tmp)
    r = next(iter(res.values()))
    bars = r["bar_heights"]
    errs = r["uptail_heights"]
    out = [(r["bar_names"][i], m, m + (errs[i] if i < len(errs) else 0), False) for i, m in enumerate(bars)]
    return out, (p.value_at(p.axis_y), p.value_at(p.plot_top), p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--method", default="new", choices=["new", "yolo"])
    args = ap.parse_args()

    from figure_extractor import FigureExtractor
    extractor = FigureExtractor()
    analyzer = None
    if args.method == "yolo":
        import pathlib
        if os.name == "nt":   # best.pt was pickled on macOS/Linux
            pathlib.PosixPath = pathlib.WindowsPath
        from model_manager import model_manager
        analyzer = model_manager.get_analyzer()

    cases = json.load(open(os.path.join(HERE, "cases.json")))["cases"]
    cache = {}
    all_err_mean, all_err_top, n_missing = [], [], 0
    for c in cases:
        folder = os.path.join(args.data, c["paper"])
        key = (c["pdf"], c["figure_page"])
        if key not in cache:
            cache[key] = page_images(os.path.join(folder, c["pdf"]), pages=[c["figure_page"]])[0][1]
        img = cache[key]
        truth = load_truth(os.path.join(folder, c["truth"]))
        if args.method == "new":
            got, meta = run_new(img, c["point"], extractor)
        else:
            got, meta = run_yolo(img, c["point"], analyzer, extractor)

        print(f"\n== {c['truth']}  (page {c['figure_page']})")
        if not isinstance(got, list):
            print(f"   NO RESULT: {got}")
            n_missing += len(truth)
            continue
        lo, hi, panel = meta
        rng = abs(hi - lo) if lo is not None and hi is not None else None
        if rng is None:
            print("   no axis calibration")
            n_missing += len(truth)
            continue
        if len(got) != len(truth):
            print(f"   bar count mismatch: got {len(got)}, expected {len(truth)}")
        print(f"   {'group (truth)':28s} {'label (found)':22s} {'mean':>9s} {'truth':>9s} {'err%':>6s} | {'top':>9s} {'truth':>9s} {'err%':>6s}")
        for i, (grp, tm, tt) in enumerate(truth):
            if i >= len(got) or got[i][1] is None:
                print(f"   {str(grp):28s} MISSING")
                n_missing += 1
                continue
            lab, gm, gt, unsure = got[i]
            em = 100 * abs(gm - tm) / rng
            et = 100 * abs(gt - tt) / rng
            all_err_mean.append(em)
            all_err_top.append(et)
            print(f"   {str(grp):28s} {str(lab)[:22]:22s} {gm:9.2f} {tm:9.2f} {em:6.2f} | {gt:9.2f} {tt:9.2f} {et:6.2f} {'FLAGGED' if unsure else ''}")

    print("\n================ SUMMARY ({}) ================".format(args.method))
    if all_err_mean:
        em, et = np.array(all_err_mean), np.array(all_err_top)
        print(f"bars measured: {len(em)}   missing: {n_missing}")
        print(f"mean      : avg err {em.mean():.2f}% of axis range, max {em.max():.2f}%")
        print(f"mean+sem  : avg err {et.mean():.2f}% of axis range, max {et.max():.2f}%")
        print(f"within 1% : {(em < 1).mean() * 100:.0f}% of means, {(et < 1).mean() * 100:.0f}% of caps")
    else:
        print(f"nothing measured, missing {n_missing}")


if __name__ == "__main__":
    main()
