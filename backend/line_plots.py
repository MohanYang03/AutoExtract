"""
Line-plot extraction: each series' trace, its data points and error bars.

Published approaches range from colour masks traced column by column
(WebPlotDigitizer's automatic modes) to trained detectors: segmentation plus
min-cost-flow line tracing (Kato et al., WACV 2022), keypoint transformers
(LineEX, WACV 2023) and instance segmentation (LineFormer). This module uses
the trace-as-optimal-path idea without a trained model:

    1. Series are separated by colour: ink pixels are grouped by hue
       (chromatic series) or as one dark group (black / grey series).
    2. Each colour group is traced by dynamic programming across columns:
       the path through the group's pixels that covers the most columns
       with the smallest vertical jumps (gaps of dashed lines are bridged).
       Several series of one colour are taken one after another, removing
       each traced path before looking for the next.
    3. Markers (circles, squares, triangles, filled or hollow) are blobs
       thicker than the line lying on the trace: they are the data points.
       Without markers, the trace itself is the data (sampled at the x ticks
       for export).
    4. Error bars are vertical strokes from a marker ending in a cap.

All positions are in pixels; FigureExtractor converts them with the axis
calibrations (y from the tick labels, x from the x tick labels).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class LinePoint:
    x_px: float
    y_px: float
    cap_hi_px: Optional[float] = None    # upper error-bar end
    cap_lo_px: Optional[float] = None    # lower error-bar end
    x: Optional[float] = None            # calibrated x value (numeric x axis)
    x_label: str = ""                    # x tick label (categorical x axis)
    mean: Optional[float] = None
    mean_plus_err: Optional[float] = None
    n: Optional[int] = None
    uncertain: bool = False


@dataclass
class Series:
    color: Tuple[int, int, int]          # BGR
    trace: List[Tuple[float, float]] = field(default_factory=list)   # (x_px, y_px) every few px
    points: List[LinePoint] = field(default_factory=list)
    markers: bool = False                # points are drawn markers (else sampled from the trace)
    label: str = ""

    def line_y_at(self, x_px: float, half: float, marker_xs=()) -> Optional[float]:
        """Height at x from the line segments either side (see line_y_at)."""
        raw = getattr(self, "_path", None)
        if raw is None:
            return None
        x0, y0, ys = raw
        y = line_y_at(ys, x_px - x0, half, [m - x0 for m in marker_xs])
        return None if y is None else y0 + y

    def y_at(self, x_px: float) -> Optional[float]:
        """Trace height at x (linear between samples), None outside it."""
        if not self.trace:
            return None
        xs = np.array([p[0] for p in self.trace])
        if x_px < xs[0] - 2 or x_px > xs[-1] + 2:
            return None
        return float(np.interp(x_px, xs, [p[1] for p in self.trace]))


# --------------------------------------------------------------------------- #

def _colour_groups(region: np.ndarray, ink: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int]]]:
    """Masks of ink pixels by colour: one per hue cluster, plus dark/grey ink."""
    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB).astype(float)
    L, a, b = lab[:, :, 0], lab[:, :, 1] - 128, lab[:, :, 2] - 128
    chroma = np.hypot(a, b)
    hue = (np.degrees(np.arctan2(b, a)) + 360) % 360
    groups = []
    colour = ink & (chroma > 20)
    if colour.sum() > 30:
        hist, _ = np.histogram(hue[colour], bins=36, range=(0, 360))
        # peaks of the hue histogram, merged with their neighbours
        thresh = max(15, 0.01 * colour.sum())
        used = np.zeros(36, bool)
        for _ in range(8):
            cand = np.where(~used & (hist >= thresh))[0]
            if cand.size == 0:
                break
            p = cand[np.argmax(hist[cand])]
            lo = hi = p
            while hist[(lo - 1) % 36] >= 0.15 * hist[p] and not used[(lo - 1) % 36] and (p - lo) < 4:
                lo -= 1
            while hist[(hi + 1) % 36] >= 0.15 * hist[p] and not used[(hi + 1) % 36] and (hi - p) < 4:
                hi += 1
            bins = [(k % 36) for k in range(lo, hi + 1)]
            used[bins] = True
            hb = (hue // 10).astype(int) % 36
            m = colour & np.isin(hb, bins)
            if m.sum() > 30:
                groups.append((m, _mean_bgr(region, m)))
    dark = ink & (chroma <= 20) & (L < 150)
    if dark.sum() > 30:
        # black and dark grey series: split the lightness histogram at its
        # deepest valley when it has two clear peaks
        hist, edges = np.histogram(L[dark], bins=15, range=(0, 150))
        split = None
        peaks = [i for i in range(15) if hist[i] >= 0.15 * hist.max()
                 and hist[i] == hist[max(0, i - 1):i + 2].max()]
        if len(peaks) >= 2 and peaks[-1] - peaks[0] >= 4:
            lo, hi = peaks[0], peaks[-1]
            v = lo + int(np.argmin(hist[lo:hi + 1]))
            if hist[v] < 0.35 * min(hist[lo], hist[hi]):
                split = edges[v] + 5
        if split is None:
            groups.append((dark, _mean_bgr(region, dark)))
        else:
            for m in (dark & (L < split), dark & (L >= split)):
                if m.sum() > 30:
                    groups.append((m, _mean_bgr(region, m)))
    return groups


def _mean_bgr(region, mask) -> Tuple[int, int, int]:
    return tuple(int(v) for v in np.median(region[mask], axis=0))


def _trace_paths(mask: np.ndarray, line_w: int, min_cover: float) -> List[np.ndarray]:
    """
    Paths through `mask` by dynamic programming over columns, best first.
    Each path is an array of y per column (nan where it has no pixel).
    """
    h, w = mask.shape
    work = mask.copy()
    paths = []
    max_gap = max(12, int(0.06 * w))          # dashed lines, markers drawn over the line
    for _ in range(4):
        cands = []
        for c in range(w):
            col = work[:, c]
            if not col.any():
                cands.append([])
                continue
            d = np.diff(np.concatenate([[0], col.astype(np.int8), [0]]))
            starts, ends = np.where(d == 1)[0], np.where(d == -1)[0] - 1
            cands.append([((s + e) / 2.0, s, e) for s, e in zip(starts, ends)])
        # Score: columns covered. A line is connected: in the next column
        # its pixels touch the previous ones (however steep it is); across a
        # gap (dashes, a marker drawn over it) it may continue at a moderate
        # slope. Jumping to another line is not allowed.
        best_score = [None] * w
        back = [None] * w
        best_end = (0.0, None)
        lam = 1.0 / max(4.0, 3.0 * line_w)
        tol = line_w + 1
        for c in range(w):
            if not cands[c]:
                continue
            best_score[c] = []
            back[c] = []
            for y, s0, e0 in cands[c]:
                bs, bk = 1.0, None
                for pc in range(max(0, c - max_gap), c):
                    if not cands[pc]:
                        continue
                    dc = c - pc
                    for j, (py, ps, pe) in enumerate(cands[pc]):
                        if dc == 1:
                            if s0 > pe + tol or e0 < ps - tol:
                                continue
                            cost = 0.25 * lam * min(abs(y - py), 4 * tol)
                        else:
                            if abs(y - py) > 2 * line_w + 2 + 2.0 * dc:
                                continue
                            cost = lam * abs(y - py) / dc + 0.05 * (dc - 1)
                        sc = best_score[pc][j] + 1.0 - cost
                        if sc > bs:
                            bs, bk = sc, (pc, j)
                best_score[c].append(bs)
                back[c].append(bk)
                if bs > best_end[0]:
                    best_end = (bs, (c, len(best_score[c]) - 1))
        if best_end[1] is None:
            break
        ys = np.full(w, np.nan)
        node = best_end[1]
        while node is not None:
            c, j = node
            ys[c] = cands[c][j][0]
            node = back[c][j]
        covered = np.where(~np.isnan(ys))[0]
        if covered.size == 0 or (covered[-1] - covered[0]) < min_cover * w:
            break
        paths.append(ys)
        # remove this path (and its markers / error bars along it) before the next
        band = int(max(3, 2.5 * line_w))
        for c in covered:
            y = int(ys[c])
            work[max(0, y - band): y + band + 1, c] = False
        # a marker drawn over the path is wider than the band: clear its blob
        n, labels = cv2.connectedComponents(work.astype(np.uint8), connectivity=8)
        for c in covered[:: max(1, band)]:
            y = int(ys[c])
            for yy in (y - band - 1, y + band + 1):
                if 0 <= yy < h and labels[yy, c]:
                    blob = labels == labels[yy, c]
                    if blob.sum() < 40 * band * band:
                        work[blob] = False
    return paths


def _fill_path(ys: np.ndarray) -> np.ndarray:
    """Interpolate a path across its gaps (between its first and last column)."""
    idx = np.where(~np.isnan(ys))[0]
    out = ys.copy()
    if idx.size >= 2:
        span = np.arange(idx[0], idx[-1] + 1)
        out[span] = np.interp(span, idx, ys[idx])
    return out


def _stroke(mask: np.ndarray, ys: np.ndarray) -> int:
    """
    The series' own line thickness: the typical vertical run through its
    trace (markers and error bars are rarer than plain line columns).
    Steep stretches look thicker, so take a low percentile.
    """
    runs = []
    for c in np.where(~np.isnan(ys))[0]:
        y = int(round(ys[c]))
        if not (0 <= y < mask.shape[0]) or not mask[y, c]:
            continue
        a = b = y
        while a > 0 and mask[a - 1, c]:
            a -= 1
        while b < mask.shape[0] - 1 and mask[b + 1, c]:
            b += 1
        runs.append(b - a + 1)
    return int(np.percentile(runs, 30)) if runs else 1


def _markers(mask: np.ndarray, ys: np.ndarray, line_w: int) -> List[Tuple[float, float, int]]:
    """Marker blobs lying on a traced path: (x, y, width)."""
    filled = mask.astype(np.uint8).copy()
    # hollow markers: fill enclosed holes
    flood = filled.copy()
    ff = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 1)
    filled = filled | (flood == 0).astype(np.uint8)
    # just thicker than the line: a line is removed, small triangles and
    # diamonds (whose inscribed circle is about half their width) survive
    k = max(3, line_w + 2)
    blobs = cv2.morphologyEx(filled, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, _, stats, cents = cv2.connectedComponentsWithStats(blobs, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        cx, cy = cents[i]
        # plotting software centres a marker on its point by its bounding
        # box; for a triangle (about half its box) the centre of mass sits a
        # third of the way down, so use the box there
        if area < 0.62 * bw * bh:
            cy = y + (bh - 1) / 2.0
        if not (k <= bw <= 12 * k and k <= bh <= 12 * k and 0.4 <= bw / max(bh, 1) <= 2.5):
            continue
        c = int(round(cx))
        if c < 0 or c >= ys.size or np.isnan(ys[c]) or abs(ys[c] - cy) > max(bh, 2 * k):
            continue
        out.append((float(cx), float(cy), int(bw)))
    out.sort()
    # one marker per x (an error-bar cap can survive the opening next to it)
    merged = []
    for m in out:
        if merged and m[0] - merged[-1][0] < 0.6 * max(m[2], merged[-1][2]):
            if m[2] > merged[-1][2]:
                merged[-1] = m
        else:
            merged.append(m)
    return merged


def _error_bar(mask: np.ndarray, cx: float, cy: float, half: float, step: int, line_w: int) -> Optional[float]:
    """
    Follow a vertical stroke from the marker edge in direction `step` and
    return the row of its cap, or None when there is no capped error bar.
    """
    h, w = mask.shape
    c = int(round(cx))
    if not 0 <= c < w:
        return None
    y = int(round(cy + step * (half + 1)))
    end, gap = None, 0
    while 0 <= y < h:
        if mask[y, max(0, c - 1):c + 2].any():
            end, gap = y, 0
        else:
            gap += 1
            if gap > max(2, line_w):
                break
        y += step
    if end is None or abs(end - cy) <= half + line_w + 2:
        return None
    # a cap: a horizontal run at the end, wider than the stroke, centred
    best = None
    for yy in range(end - step * (2 * line_w + 2), end + step, step):
        if not 0 <= yy < h:
            continue
        row = mask[yy]
        lft = rgt = c
        while lft > 0 and row[lft - 1]:
            lft -= 1
        while rgt < w - 1 and row[rgt + 1]:
            rgt += 1
        if rgt - lft + 1 >= max(5, 3 * line_w) and abs((lft + rgt) / 2 - c) <= 2 + line_w:
            best = yy if best is None or abs(yy - end) < abs(best - end) else best
    return float(best) if best is not None else None


def _frames_and_references(mask: np.ndarray, line_w: int) -> np.ndarray:
    """
    Pixels of straight lines that are not data: the plot frame, reference
    lines (solid or dashed, e.g. p = 0.05) and legend boxes.
    """
    h, w = mask.shape
    out = np.zeros_like(mask)
    m8 = mask.astype(np.uint8)
    thick = 2 * line_w + 2
    # long solid horizontal / vertical lines
    hor = cv2.morphologyEx(m8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, int(0.5 * w)), 1)))
    ver = cv2.morphologyEx(m8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, int(0.5 * h)))))
    out |= hor.astype(bool) | ver.astype(bool)
    # dashed reference lines: a thin band of many short pieces across the plot
    for axis in (0, 1):
        m = mask if axis == 0 else mask.T
        frac = m.mean(axis=1)
        for s, e in _runs1d(frac >= 0.3):
            if e - s + 1 > thick:
                continue
            band = m[s:e + 1].any(axis=0)
            runs = _runs1d(band)
            span = np.where(band)[0]
            # dashes repeat: similar lengths at similar spacing (error bars and
            # markers stacked in one column do not)
            lens = np.array([b - a + 1 for a, b in runs], float)
            gaps = np.array([runs[i + 1][0] - runs[i][1] - 1 for i in range(len(runs) - 1)], float)
            # (many pieces across the plot are dashes even when noisy)
            regular = len(runs) >= 8 or (len(runs) >= 5 and lens.std() <= 0.5 * lens.mean()
                                         and gaps.size and gaps.std() <= 0.5 * gaps.mean())
            if regular and span[-1] - span[0] >= 0.6 * band.size:
                if axis == 0:
                    out[s:e + 1] |= mask[s:e + 1]
                else:
                    out[:, s:e + 1] |= mask[:, s:e + 1]
    # legend boxes: closed rectangles inside the plot
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 0.08 * w or bh < 0.05 * h or bw > 0.7 * w or bh > 0.7 * h:
            continue
        approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and cv2.contourArea(cnt) > 0.85 * bw * bh:
            out[y:y + bh, x:x + bw] = True
    return out


def _runs1d(m: np.ndarray) -> List[Tuple[int, int]]:
    d = np.diff(np.concatenate([[0], np.asarray(m, np.int8), [0]]))
    return list(zip(np.where(d == 1)[0].tolist(), (np.where(d == -1)[0] - 1).tolist()))


def find_series(image: np.ndarray, ink: np.ndarray, dark: np.ndarray, x_lo: int, x_hi: int,
                y_lo: int, axis_y: float, line_w: int) -> List[Series]:
    """Line series in the plot region [y_lo, axis) x [x_lo, x_hi]."""
    y_hi = int(round(axis_y)) - line_w - 1
    x0 = x_lo + line_w + 1
    if y_hi - y_lo < 20 or x_hi - x0 < 30:
        return []
    region = image[y_lo:y_hi, x0:x_hi + 1]
    rink = ink[y_lo:y_hi, x0:x_hi + 1].astype(bool)
    h, w = rink.shape
    # light gridlines are not series: drop light grey ink, but keep light
    # coloured lines (a pale yellow series is still a series)
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB).astype(float)
    chroma = np.hypot(lab[:, :, 1] - 128, lab[:, :, 2] - 128)
    rink = (rink & (gray < 200)) | ((chroma > 25) & (gray < 250))
    # A line plot is lines on a plain background. A picture, heatmap or 3D
    # surface inside axes is not (and tracing its texture would be slow).
    if (gray > 200).mean() < 0.5 and np.abs(np.diff(gray.astype(float), axis=1)).mean() > 12:
        return []
    rink &= ~_frames_and_references(rink, line_w)
    runs_per_col = (np.diff(rink.astype(np.int8), axis=0) == 1).sum(axis=0)
    if np.median(runs_per_col) > 8:
        return []                        # texture, not a few lines

    series = []
    taken = np.zeros_like(rink)
    for mask, colour in _colour_groups(region, rink):
        mask = mask & ~taken
        for ys in _trace_paths(mask, line_w, min_cover=0.3):
            covered_cols = np.where(~np.isnan(ys))[0]
            # a line has ink in most columns it spans (dashes too); scatter
            # points chained across gaps do not
            if covered_cols.size < 0.45 * (covered_cols[-1] - covered_cols[0] + 1):
                continue
            # a significance bracket is flat and spans part of the plot; a
            # flat data series runs across most of it
            if (np.nanmax(ys) - np.nanmin(ys) <= 2 * line_w + 2
                    and covered_cols[-1] - covered_cols[0] < 0.6 * w):
                continue
            full = _fill_path(ys)
            covered = np.where(~np.isnan(full))[0]
            s = Series(color=colour)
            # the raw trace (gaps not bridged), for reading hidden points later
            s._path = (x0, y_lo, ys)
            step = max(1, w // 300)
            s.trace = [(x0 + float(c), y_lo + float(full[c])) for c in covered[::step]]
            # mask around this path only (other series of the colour excluded)
            near = np.zeros_like(mask)
            band = int(max(6, 6 * line_w))
            for c in covered:
                y = int(full[c])
                near[max(0, y - band): y + band + 1, c] = True
            # the trace starts and ends inside the end markers: widen sideways
            near = cv2.dilate(near.astype(np.uint8),
                              cv2.getStructuringElement(cv2.MORPH_RECT, (2 * band + 1, 1))).astype(bool)
            own = mask & near
            marks = _markers(own, full, max(line_w, _stroke(own, full)))
            # Many unevenly spaced "markers" are noise on a dense trace (a
            # recording, a mean with a shaded band): read the trace instead.
            if len(marks) > 15:
                gaps = np.diff([m[0] for m in marks])
                if gaps.std() > 0.5 * gaps.mean():
                    marks = []
            if len(marks) >= 2:
                s.markers = True
                xs_m = [m[0] for m in marks]
                for cx, cy, mw in marks:
                    # the line runs straight into the marker centre from both
                    # sides: that is robust to marker shape and to markers
                    # partly covered by another series
                    yl = line_y_at(full, cx, mw / 2.0, xs_m)
                    if yl is not None and abs(yl - cy) <= 0.6 * mw:
                        cy = yl
                    p = LinePoint(x_px=x0 + cx, y_px=y_lo + cy)
                    # error bars in the series colour, else in black
                    for m in (mask, dark[y_lo:y_hi, x0:x_hi + 1].astype(bool)):
                        hi = _error_bar(m, cx, cy, mw / 2.0, -1, line_w)
                        lo = _error_bar(m, cx, cy, mw / 2.0, +1, line_w)
                        if hi is not None or lo is not None:
                            p.cap_hi_px = y_lo + hi if hi is not None else None
                            p.cap_lo_px = y_lo + lo if lo is not None else None
                            # bars are drawn both ways; one side alone (or two
                            # very different sides) may belong to a neighbour
                            if hi is None or lo is None or abs((cy - hi) - (lo - cy)) > 0.35 * max(cy - hi, lo - cy):
                                p.uncertain = True
                            break
                    s.points.append(p)
            # pixels of this series are not available to the dark group
            taken |= near & mask
            series.append(s)
    # the same trace found twice (a colour split in two hue bins)
    unique = []
    for s in sorted(series, key=lambda s: -len(s.trace)):
        if any(_same_trace(s, u, line_w) for u in unique):
            continue
        unique.append(s)
    return sorted(unique, key=lambda s: np.mean([p[1] for p in s.trace]))


def line_y_at(path: np.ndarray, cx: float, half: float, marker_xs=()) -> Optional[float]:
    """
    Height of a data point at column cx from the line segments either side
    of its marker: straight-line fits just outside the marker, met at cx.
    `path` is the traced height per column (nan where absent).
    """
    w = path.size
    others = sorted(abs(m - cx) for m in marker_xs if abs(m - cx) > half)
    reach = max(4.0, min(3 * 2 * half, 0.45 * others[0] if others else 6 * half))
    preds = []
    for side in (-1, 1):
        a = cx + side * (half + 2)
        b = a + side * reach
        lo, hi = int(min(a, b)), int(max(a, b))
        cols = np.arange(max(0, lo), min(w, hi + 1))
        ys = path[cols] if cols.size else np.array([])
        ok = ~np.isnan(ys)
        if ok.sum() >= 3:
            k, c0 = np.polyfit(cols[ok], ys[ok], 1)
            preds.append(k * cx + c0)
    return float(np.mean(preds)) if preds else None


def _same_trace(a: Series, b: Series, line_w: int) -> bool:
    xs = [p[0] for p in a.trace]
    d = [abs(a.y_at(x) - b.y_at(x)) for x in xs if b.y_at(x) is not None]
    return len(d) > 0.6 * len(xs) and float(np.median(d)) <= 2 * line_w + 1
