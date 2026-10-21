"""
Box-plot detection: box (Q1–Q3), median, whiskers, outliers, mean marker.

Works in pixel space on one calibrated plot region; FigureExtractor turns
the pixel rows into values with the panel's axis calibration.

How a box is found
    1. Foreground = everything but the open background (white box interiors
       are enclosed, so outlined boxes become solid rectangles).
    2. A morphological opening removes thin structures (whiskers, caps,
       outlines-only artefacts, gridlines) and leaves the boxes.
    3. Rectangular components of consistent width are boxes; boxes that touch
       their neighbours are split where the top or bottom edge steps.
    4. Box edges give Q3 (top) and Q1 (bottom); the median is the full-width
       line inside the box that contrasts with the fill.
    5. Whiskers are followed up and down from the box centre, bridging the
       gaps of dashed whiskers; a horizontal cap ends them when present.
    6. Small blobs beyond the whisker ends on the box's column are outliers;
       a small blob inside the box that isn't the median is a mean marker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Box:
    x1: int
    x2: int
    q1_px: float
    median_px: float
    q3_px: float
    lo_px: float                 # lower whisker end (equals q1_px if none)
    hi_px: float                 # upper whisker end (equals q3_px if none)
    outliers_px: List[float] = field(default_factory=list)
    mean_px: Optional[float] = None      # a drawn mean marker, if any
    # values (filled in from the axis calibration)
    q1: Optional[float] = None
    median: Optional[float] = None
    q3: Optional[float] = None
    lo: Optional[float] = None
    hi: Optional[float] = None
    outliers: List[float] = field(default_factory=list)
    mean_marker: Optional[float] = None
    label: str = ""
    n: Optional[int] = None
    uncertain: bool = False          # median or a whisker could not be confirmed
    note: str = ""


def _runs(mask_1d: np.ndarray) -> List[Tuple[int, int]]:
    m = np.concatenate([[False], np.asarray(mask_1d, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.where(d == 1)[0].tolist(), (np.where(d == -1)[0] - 1).tolist()))


def find_boxes(image: np.ndarray, gray: np.ndarray, dark: np.ndarray, ink: np.ndarray,
               fg: np.ndarray, x_lo: int, y_lo: int, axis_y: float, line_w: int) -> List[Box]:
    """
    Boxes in the plot region whose foreground mask `fg` starts at (y_lo, x_lo).
    Returns [] when the region doesn't look like a box plot.
    """
    h, w = fg.shape
    if h < 20 or w < 20:
        return []
    k = max(3, 2 * line_w + 1)
    solid = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=4)
    axis_local = axis_y - y_lo

    cands = []
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if cw < max(6, 0.008 * w) or ch < 3 or cw > 0.6 * w:
            continue
        pieces = _split_touching(labels[y:y + ch, x:x + cw] == i, x, y)
        # Boxes drawn side by side share one width; a curved shape (violin,
        # density) falls apart into slices of very different widths.
        pw = [p[1] - p[0] + 1 for p in pieces]
        if len(pw) > 2 and min(pw) < 0.6 * max(pw):
            continue
        for bx1, bx2, by1, by2 in pieces:
            bw = bx2 - bx1 + 1
            sub = labels[by1:by2 + 1, bx1:bx2 + 1] == i
            if bw < max(6, 0.008 * w) or not sub.size or sub.mean() < 0.8:
                continue
            if by2 - by1 + 1 <= 2 * line_w + 2:
                continue                 # a whisker cap or a line, not a box

            cands.append((bx1, bx2, by1, by2))
    if not cands:
        return []

    # boxes in one chart share a width; outlier dots, stars and legend
    # swatches are narrower
    # (thick whisker dashes can outnumber the boxes, so look at the wide ones)
    widths = np.array([c[1] - c[0] + 1 for c in cands], float)
    typical = float(np.median(widths[widths >= 0.5 * widths.max()]))
    cands = [c for c in cands if c[1] - c[0] + 1 >= 0.6 * typical]
    # a box floats: shapes standing on the x-axis are bars
    floating = [c for c in cands if axis_local - c[3] > 2 * line_w + 3]
    if len(floating) < max(1, 0.5 * len(cands)):
        return []
    # One box per column (grouped boxes sit side by side). Shapes stacked in
    # a column are scatter dots, which become solid once their hollow faces
    # are filled; a chart full of them is not a box plot.
    def stacked(c):
        cw = c[1] - c[0] + 1
        return sum(1 for o in floating if o is not c
                   and abs((o[1] - o[0] + 1) - cw) <= 0.3 * cw      # a similar shape, not an outlier
                   and min(c[1], o[1]) - max(c[0], o[0]) > 0.5 * min(c[1] - c[0], o[1] - o[0]))
    single = [c for c in floating if stacked(c) == 0]
    if len(single) < 0.7 * len(floating):
        return []
    floating = single
    # Bars standing on a zero line above the x-axis also "float", sharing
    # one bottom edge; scatter dots inside them can pass for mean markers,
    # so such shapes must show real median lines to count as boxes.
    shared_base = (len(floating) >= 2
                   and max(c[3] for c in floating) - min(c[3] for c in floating) <= 2
                   and max(c[2] for c in floating) - min(c[2] for c in floating) > 2 * line_w + 3)

    boxes = []
    for bx1, bx2, by1, by2 in sorted(floating):
        box = _measure_box(image, gray, dark, ink, fg, bx1, bx2, by1, by2, x_lo, y_lo, line_w, axis_local)
        if box is not None:
            boxes.append(box)
    if not boxes:
        return []
    # Dotted gridlines show up as an "outlier" at the same row in many boxes
    if len(boxes) >= 3:
        rows = [y for b in boxes for y in b.outliers_px]
        for b in boxes:
            b.outliers_px = [y for y in b.outliers_px
                             if sum(abs(y - o) <= 1.5 for o in rows) < max(3, 0.4 * len(boxes))]
    # a single "box" with no whiskers is more likely a legend or a label frame
    has_whisker = [b.hi_px < b.q3_px - line_w - 1 or b.lo_px > b.q1_px + line_w + 1 for b in boxes]
    if len(boxes) == 1 and not has_whisker[0]:
        return []
    if sum(has_whisker) < 0.5 * len(boxes) and len(boxes) < 3:
        return []
    # most boxes show a median line (or a mean marker) inside
    inner_mark = sum((not b.uncertain) or (b.mean_px is not None and not shared_base) for b in boxes)
    if inner_mark < 0.5 * len(boxes):
        return []
    return boxes


def _split_touching(mask: np.ndarray, ox: int, oy: int) -> List[Tuple[int, int, int, int]]:
    """
    Split a component made of boxes drawn edge to edge: columns are grouped
    while the top and bottom edges stay level; a step starts a new box.
    """
    h, w = mask.shape
    cols = mask.any(axis=0)
    tops = np.where(cols, mask.argmax(axis=0), -1)
    bots = np.where(cols, h - 1 - mask[::-1].argmax(axis=0), -1)
    out = []
    start = None
    for c in range(w + 1):
        if c < w and cols[c] and start is not None and (
                abs(tops[c] - tops[c - 1]) > 2 or abs(bots[c] - bots[c - 1]) > 2):
            out.append((start, c - 1))
            start = c
        elif c < w and cols[c] and start is None:
            start = c
        elif (c == w or not cols[c]) and start is not None:
            out.append((start, c - 1))
            start = None
    # A whisker with a wide cap can survive the opening as a bump on the
    # box's top or bottom edge: a step out and back to the same level is
    # part of the box, not a neighbour.
    def level(seg):
        s, e = seg
        return np.median(tops[s:e + 1]), np.median(bots[s:e + 1])

    # The bump may itself be stepped (whisker, then cap), so look for the
    # farthest piece on the same level with only outward steps in between.
    merged, i = [], 0
    while i < len(out):
        t0, b0 = level(out[i])
        k_best = i
        for k in range(len(out) - 1, i + 1, -1):
            if any(out[j + 1][0] != out[j][1] + 1 for j in range(i, k)):
                continue                     # not one contiguous shape
            tk, bk = level(out[k])
            if abs(t0 - tk) > 2 or abs(b0 - bk) > 2:
                continue
            inner = [level(out[j]) for j in range(i + 1, k)]
            if all(t <= max(t0, tk) + 2 and b >= min(b0, bk) - 2 for t, b in inner):
                k_best = k
                break
        flanks = [out[i], out[k_best]]
        merged.append((out[i][0], out[k_best][1], flanks))
        i = k_best + 1
    boxes = []
    for s, e, flanks in merged:
        if not cols[s:e + 1].any():
            continue
        # bumps only stick out, so the box edges are the flanks' edge rows
        fc = np.concatenate([np.arange(a, b + 1) for a, b in flanks])
        top = int(np.percentile(tops[fc], 80))
        bot = int(np.ceil(np.percentile(bots[fc], 20)))
        boxes.append((ox + s, ox + e, oy + top, oy + bot))
    return boxes


def _edge_row(dark_region: np.ndarray, rows: range, line_w: int) -> Optional[float]:
    """Centre of the first full-width dark band met when scanning `rows`."""
    band = []
    for r in rows:
        if 0 <= r < dark_region.shape[0] and dark_region[r].mean() >= 0.8:
            band.append(r)
        elif band:
            break
        if not band and abs(r - rows.start) > 2 * line_w + 3:
            break
    return float(np.mean(band)) if band else None


def _measure_box(image, gray, dark, ink, fg, bx1, bx2, by1, by2, x_lo, y_lo, line_w, axis_local) -> Optional[Box]:
    bw = bx2 - bx1 + 1
    inner = slice(x_lo + bx1 + max(1, bw // 8), x_lo + bx2 - max(1, bw // 8) + 1)
    X1, X2 = x_lo + bx1, x_lo + bx2
    Y1, Y2 = y_lo + by1, y_lo + by2
    cx = (X1 + X2) // 2

    # ---- Q3 / Q1: box edges (outline centre, or the fill edge) ----
    d_in = dark[:, inner]
    top = _edge_row(d_in, range(Y1, Y1 + 2 * line_w + 4), line_w)
    bot = _edge_row(d_in, range(Y2, Y2 - 2 * line_w - 4, -1), line_w)
    q3_px = top if top is not None else Y1 - 0.5
    q1_px = bot if bot is not None else Y2 + 0.5

    # ---- median: full-width contrasting line inside the box ----
    body = image[Y1:Y2 + 1, inner].astype(float)
    uncertain = False
    median_px = None
    if body.shape[0] >= 3:
        lab = cv2.cvtColor(image[Y1:Y2 + 1, inner], cv2.COLOR_BGR2LAB).astype(float)
        fill = np.median(lab.reshape(-1, 3), axis=0)
        contrast = (np.linalg.norm(lab - fill, axis=2) > 40).mean(axis=1)
        margin = max(1, line_w) + 1
        cand = []
        for s, e in _runs(contrast >= 0.7):
            if e - s + 1 > 3 * line_w + 4:
                continue
            centre = (s + e) / 2.0
            # skip the box's own top and bottom outlines
            if centre <= (q3_px - Y1) + margin or centre >= (q1_px - Y1) - margin:
                continue
            cand.append((float(contrast[s:e + 1].mean()), Y1 + centre))
        if cand:
            median_px = max(cand)[1]
    if median_px is None:
        # median drawn on (or merged with) an edge, or not drawn: thickest edge band
        uncertain = True
        top_band = d_in[Y1:Y1 + 3 * line_w + 6].mean(axis=1) >= 0.8
        bot_band = d_in[max(0, Y2 - 3 * line_w - 5):Y2 + 1].mean(axis=1) >= 0.8
        median_px = q3_px if top_band.sum() > bot_band.sum() else q1_px

    # ---- whiskers ----
    hi_px, hi_ok = _whisker(dark, ink, cx, int(round(q3_px)) - line_w - 1, -1, bw, line_w, y_lo)
    lo_px, lo_ok = _whisker(dark, ink, cx, int(round(q1_px)) + line_w + 1, +1, bw, line_w,
                            int(y_lo + axis_local) - 1)
    if hi_px is None:
        hi_px = q3_px
    if lo_px is None:
        lo_px = q1_px
    uncertain = uncertain or not (hi_ok and lo_ok)

    # ---- outliers: blobs on the box's column beyond the whiskers ----
    outliers = _outliers(ink, X1, X2, hi_px, lo_px, y_lo, int(y_lo + axis_local) - line_w - 1, line_w)

    # ---- mean marker: a small blob inside the box that isn't the median ----
    mean_px = _mean_marker(image, X1, X2, q3_px, q1_px, median_px, line_w)

    return Box(x1=X1, x2=X2, q1_px=float(q1_px), median_px=float(median_px), q3_px=float(q3_px),
               lo_px=float(lo_px), hi_px=float(hi_px), outliers_px=outliers, mean_px=mean_px,
               uncertain=uncertain)


def _whisker(dark, ink, cx, start, step, box_w, line_w, limit) -> Tuple[Optional[float], bool]:
    """
    Follow a whisker from `start` in direction `step` (-1 up, +1 down).
    Dashed whiskers are bridged; a horizontal cap ends the whisker.
    Returns (end row or None, confident).
    """
    H, W = dark.shape
    max_gap = max(6, 3 * line_w)              # dashed whiskers (R's default lty=2)
    reach = 2 * max_gap + 6                   # a dash may start a little past the box
    for mask in (dark, ink):
        # step past the box's own edge line (thicker than the axis, or the
        # quartile row sits inside a dark fill) so it isn't taken for a cap
        x_a, x_b = max(0, cx - box_w // 3), min(W, cx + box_w // 3 + 1)
        skip = 0
        while (skip <= 2 * line_w + 3 and 0 <= start + step * skip < H
               and mask[start + step * skip, x_a:x_b].mean() > 0.9):
            skip += 1
        start_m = start + step * skip
        # whisker column: the one with most ink just past the box edge
        best_col, best = None, 0
        for c in range(cx - 3, cx + 4):
            if not 0 <= c < W:
                continue
            ys = [start_m + step * k for k in range(0, reach)]
            s = sum(int(mask[y, c]) for y in ys if 0 <= y < H)
            if s > best:
                best_col, best = c, s
        if best_col is None or best < 2:
            continue
        col = best_col
        y, end, gap = start_m, None, 0
        wt = None                         # the whisker's own thickness
        narrow_end = start_m              # last row only as wide as the whisker
        while (y >= limit if step < 0 else y <= limit) and 0 <= y < H:
            if mask[y, col] or mask[y, max(0, col - 1)] or mask[y, min(W - 1, col + 1)]:
                # an outlier just past a cap-less whisker (after a gap), or
                # touching it (the ink widens into a round blob)
                if end is not None and gap >= 2 and _round_blob(mask, y, col, step, wt or line_w, box_w, after_gap=True):
                    break
                wide = wt is not None and _wider_than(mask, y, col, 2 * wt + 2)
                if end is not None and wide and _round_blob(mask, y, col, step, wt, box_w):
                    end = narrow_end
                    break
                if not wide:
                    narrow_end = y
                end, gap = y, 0
                if wt is None:
                    l = r = col
                    while l > 0 and mask[y, l - 1] and col - l < box_w:
                        l -= 1
                    while r < W - 1 and mask[y, r + 1] and r - col < box_w:
                        r += 1
                    wt = r - l + 1
                    # dash gaps scale with the line width (thick dashed whiskers)
                    max_gap = max(max_gap, 3 * wt)
                # a cap: horizontal run well beyond the whisker's own width
                l = r = col
                while l > 0 and mask[y, l - 1] and col - l < box_w:
                    l -= 1
                while r < W - 1 and mask[y, r + 1] and r - col < box_w:
                    r += 1
                if r - l + 1 >= max(0.2 * box_w, 3 * line_w + 3) and abs((l + r) / 2 - col) <= 0.2 * box_w + 2:
                    band = [y]
                    while 0 <= band[-1] + step < H and mask[band[-1] + step, l:r + 1].mean() > 0.7:
                        band.append(band[-1] + step)
                    return float(np.mean(band)), True
            else:
                gap += 1
                # before the first dash, allow the dashed whisker's offset
                if gap > (max_gap if end is not None else reach):
                    break
            y += step
        if end is not None and abs(end - start_m) > line_w:
            return float(end), True       # no cap (ggplot style): the whisker's end
    return None, True                       # no whisker: the box edge is the extreme


def _wider_than(mask, y, col, width) -> bool:
    W = mask.shape[1]
    l = r = col
    while l > 0 and mask[y, l - 1] and col - l <= width:
        l -= 1
    while r < W - 1 and mask[y, r + 1] and r - col <= width:
        r += 1
    return r - l + 1 > width


def _round_blob(mask, y, col, step, wt, box_w, after_gap=False) -> bool:
    """
    Is the ink met at (y, col) a dot (about as tall as wide, wider than the
    whisker)? Past a gap, dashes are as wide as the whisker, so any blob a
    little wider is a dot; touching the whisker it must be clearly wider.
    """
    H, W = mask.shape
    span = max(6, int(0.6 * box_w))
    ya, yb = (y, min(H, y + span)) if step > 0 else (max(0, y - span + 1), y + 1)
    xa, xb = max(0, col - span), min(W, col + span + 1)
    win = mask[ya:yb, xa:xb].astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(win, connectivity=8)
    lab = labels[y - ya, col - xa]
    if lab == 0:
        nz = np.nonzero(labels[y - ya, max(0, col - xa - 1):col - xa + 2])[0]
        if not nz.size:
            return False
        lab = labels[y - ya, max(0, col - xa - 1) + nz[0]]
    _, _, bw, bh, _ = stats[lab]
    min_w = wt + 3 if after_gap else 2 * wt + 2
    return bw >= min_w and bh > 0.6 * bw and bh < 2.5 * bw


def _outliers(ink, X1, X2, hi_px, lo_px, plot_top, plot_bottom, line_w) -> List[float]:
    w = X2 - X1 + 1
    cx = (X1 + X2) / 2
    xa, xb = int(cx - 0.6 * w), int(cx + 0.6 * w) + 1
    out = []
    for ya, yb in ((plot_top, int(hi_px) - line_w - 2), (int(lo_px) + line_w + 2, plot_bottom)):
        if yb - ya < 3:
            continue
        region = ink[ya:yb, max(0, xa):xb].astype(np.uint8)
        n, _, stats, cents = cv2.connectedComponentsWithStats(region, connectivity=8)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if 3 <= bw <= 0.8 * w and 3 <= bh <= 0.8 * w and area >= 6:
                out.append(float(ya + cents[i][1]))
    return sorted(out)


def _mean_marker(image, X1, X2, q3_px, q1_px, median_px, line_w) -> Optional[float]:
    """A drawn mean (circle, square, diamond, +) inside the box, away from the median line."""
    w = X2 - X1 + 1
    y1, y2 = int(q3_px) + line_w + 1, int(q1_px) - line_w
    if y2 - y1 < 5:
        return None
    region = image[y1:y2, X1 + max(1, w // 8): X2 - max(1, w // 8) + 1]
    if region.size == 0:
        return None
    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB).astype(float)
    fill = np.median(lab.reshape(-1, 3), axis=0)
    mask = (np.linalg.norm(lab - fill, axis=2) > 40).astype(np.uint8)
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best, blobs = None, 0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        cy = y1 + cents[i][1]
        if bw > 0.7 * region.shape[1]:
            continue                     # spans the box: the median line
        if not (3 <= bw <= 0.6 * w and 3 <= bh <= 0.6 * w and 0.5 <= bw / max(bh, 1) <= 2.0):
            continue
        blobs += 1
        if abs(cy - median_px) <= line_w + 1:
            continue
        # centred in the box
        if abs((x + bw / 2) - region.shape[1] / 2) > 0.25 * w:
            continue
        if best is None or area > best[0]:
            best = (area, float(cy))
    # several marker-sized blobs are scatter dots (e.g. over a bar), not a mean
    return best[1] if best and blobs == 1 else None
