"""
Violin-plot detection: the five-number summary drawn on each violin.

A violin is a box plot plus a density trace (Hintze & Nelson 1998). What a
figure shows of the box plot depends on the software:

    Prism         dashed median line, dotted quartile lines across the violin
    ggplot2       a narrow box plot drawn inside (geom_boxplot(width = 0.1)),
                  or quantile lines (draw_quantiles = c(.25, .5, .75))
    matplotlib    median / quantile lines (showmedians, quantiles)

Both Prism ("truncated violin") and ggplot (trim = TRUE) end the violin at
the smallest and largest value by default, so the tips are min and max.

How a violin is found
    1. Foreground = everything but the open background; an opening removes
       lines and dots and leaves the filled bodies.
    2. A body is a violin when it is mirror-symmetric about a vertical line
       and its width changes along its height (a box keeps one width).
    3. Inner lines are thin bands that contrast with the fill and run across
       both halves of the violin (scatter dots cover a small part of a row).
       Three lines are Q3, median, Q1 from the top; one line is the median.
    4. An inner narrow box gives Q1/Q3 from its edges and the median from the
       line inside it.

The result reuses the Box structure (lo/hi = violin ends), so the box-plot
estimators, export and editor work unchanged.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from box_plots import Box, _runs


def find_violins(image: np.ndarray, gray: np.ndarray, dark: np.ndarray, ink: np.ndarray,
                 fg: np.ndarray, x_lo: int, y_lo: int, axis_y: float, line_w: int) -> List[Box]:
    """Violins in the plot region whose foreground mask `fg` starts at (y_lo, x_lo)."""
    h, w = fg.shape
    if h < 30 or w < 20:
        return []
    k = max(3, 2 * line_w + 1)
    solid = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    axis_local = axis_y - y_lo

    found = []
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if cw < max(10, 0.02 * w) or ch < max(20, 0.12 * h) or cw > 0.85 * w:
            continue
        if y + ch >= axis_local - line_w:            # stands on the axis: a bar or a filled area
            continue
        shape = _violin_shape(labels[y:y + ch, x:x + cw] == i)
        if shape is None:
            continue
        lefts, rights = shape
        found.append((x, y, cw, ch, lefts, rights))
    if not found:
        return []

    violins = []
    for x, y, cw, ch, lefts, rights in sorted(found):
        v = _measure_violin(image, dark, x_lo + x, y_lo + y, cw, ch, lefts, rights, line_w)
        if v is not None:
            violins.append(v)
    if not violins:
        return []
    return violins


def _violin_shape(mask: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Per-row left/right edges when the body is a violin, else None."""
    rows = np.where(mask.any(axis=1))[0]
    if rows.size < 20:
        return None
    lefts = np.full(mask.shape[0], -1)
    rights = np.full(mask.shape[0], -1)
    for r in rows:
        cols = np.where(mask[r])[0]
        lefts[r], rights[r] = cols[0], cols[-1]
    valid = lefts >= 0
    widths = (rights - lefts + 1)[valid]
    centres = ((lefts + rights) / 2.0)[valid]
    wmax = widths.max()
    # mirror-symmetric about one vertical line
    if np.percentile(np.abs(centres - np.median(centres)), 90) > max(2.0, 0.08 * wmax):
        return None
    # solid inside its outline (not a ring or a cluster of dots)
    if mask[valid].sum() < 0.85 * widths.sum():
        return None
    # the width changes along the height: a box keeps one width
    # (even a bimodal "hourglass" narrows by a fifth at its waist)
    mid = widths[int(0.1 * widths.size): max(int(0.1 * widths.size) + 1, int(0.9 * widths.size))]
    if (wmax - mid.min()) < 0.12 * wmax:
        return None
    # smooth outline: neighbouring rows differ little. Only a truncated end
    # may jump; a box with its whisker and caps steps abruptly in between.
    steps = np.abs(np.diff(widths))
    if np.percentile(steps, 95) > max(4, 0.15 * wmax):
        return None
    # (a scatter dot on a tip makes a step near the end, so allow those)
    end = max(3, int(0.1 * steps.size))
    if steps.size > 2 * end + 2 and steps[end:-end].max() > max(6, 0.25 * wmax):
        return None
    return lefts, rights


def _measure_violin(image, dark, X, Y, cw, ch, lefts, rights, line_w) -> Optional[Box]:
    valid = lefts >= 0
    rows = np.where(valid)[0]
    top, bot = int(rows[0]), int(rows[-1])
    widths = np.where(valid, rights - lefts + 1, 0)

    region = image[Y:Y + ch, X:X + cw]
    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB).astype(float)
    body = np.zeros((ch, cw), np.uint8)
    for r in rows:
        body[r, lefts[r]:rights[r] + 1] = 1

    def interior(lw):
        # inside the outline and clear of it in every direction (where the
        # violin bulges, its outline runs almost horizontally across rows)
        m = 2 * lw + 3
        ins = cv2.erode(body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (m, m))).astype(bool)
        ins[:top + lw + 3] = False
        ins[max(0, bot - lw - 2):] = False
        return ins

    inside = interior(line_w)
    if inside.sum() < 50:
        return None
    fill = np.median(lab[inside], axis=0)
    # The violin's own lines can be much thicker than the axis: measure its
    # outline where it is widest and size every line test by that.
    wide = [r for r in rows if widths[r] >= 0.8 * widths.max()]
    runs = []
    for r in wide[:: max(1, len(wide) // 12)]:
        diff = np.linalg.norm(lab[r, lefts[r]:rights[r] + 1] - fill, axis=1) > 45
        run = int(np.argmin(diff)) if not diff.all() else diff.size
        runs.append(run)
    lw = max(line_w, int(np.median(runs)) if runs else line_w)
    if lw > line_w:
        inside = interior(lw)
        if inside.sum() < 50:
            return None
        fill = np.median(lab[inside], axis=0)
    line_w = lw
    hi_px = Y + top + (line_w - 1) / 2.0       # outline centre of the top end
    lo_px = Y + bot - (line_w - 1) / 2.0
    contrast = (np.linalg.norm(lab - fill, axis=2) > 45) & inside

    box = _inner_box(lab, contrast, inside, lefts, rights, widths, line_w)
    # Lines are a line-width tall (each dash or dot of a dotted line too);
    # scatter dots drawn over the violin are taller. Keep pixels whose
    # vertical run in their column is line-thin: where a dot crosses a line
    # only that short stretch of the line is lost.
    # Lines are drawn in ink: darker than a light or mid fill, lighter than
    # a dark fill. White dot centres on a grey violin contrast too, but are
    # not ink.
    L = lab[:, :, 0]
    # Mid-grey fills get either: try both inks and keep the one whose lines
    # are clearer.
    best = None
    for ink_px in (L < fill[0] - 40, L > fill[0] + 40):
        thin = _thin_rows(contrast & ink_px, 2 * line_w + 3)
        found = _inner_lines(thin, inside, lefts, rights, rows, line_w)
        score = sorted((_band_strength(thin, inside, b, line_w) for b in found), reverse=True)[:3]
        if best is None or sum(score) > best[0]:
            best = (sum(score), thin, found)
    _, lines_only, line_bands = best
    if box is not None:
        q3_r, med_r, q1_r = box
        q3, med, q1 = Y + q3_r, (Y + med_r if med_r is not None else None), Y + q1_r
        lines_found = 3 if med is not None else 2
    else:
        bands = line_bands
        q3 = med = q1 = None
        lines_found = len(bands)
        clear_median = False
        if len(bands) == 3:
            q3, med, q1 = (Y + b for b in bands)
        elif len(bands) == 1:
            med = Y + bands[0]
        elif len(bands) > 3:
            # extra rows from dots: the median is the strongest band (drawn
            # thicker), the quartiles its nearest neighbours above and below
            strength = {b: _band_strength(lines_only, inside, b, line_w) for b in bands}
            strongest = max(bands, key=strength.get)
            runner_up = max(v for b, v in strength.items() if b != strongest)
            clear_median = strength[strongest] >= 1.25 * runner_up
            above = [b for b in bands if b < strongest]
            below = [b for b in bands if b > strongest]
            med = Y + strongest
            q3 = Y + above[-1] if above else None
            q1 = Y + below[0] if below else None
        elif len(bands) == 2:
            med = Y + max(bands, key=lambda b: _band_strength(lines_only, inside, b, line_w))
    if med is None:
        return None
    # one line (median only) or three (quartiles and median) are the styles
    # software draws; anything else is a guess
    uncertain = lines_found not in (1, 3) and not (box is None and clear_median and q1 is not None
                                                  and q3 is not None)
    note = ""
    if q1 is None or q3 is None:
        note = "only the median is drawn: quartiles not shown"
    elif uncertain:
        note = "check the quartile lines"
    return Box(x1=int(X), x2=int(X + cw - 1), q1_px=float(q1 if q1 is not None else med),
               median_px=float(med), q3_px=float(q3 if q3 is not None else med),
               lo_px=float(lo_px), hi_px=float(hi_px), uncertain=uncertain, note=note,
               quartiles=q1 is not None and q3 is not None)


def _thin_rows(mask: np.ndarray, max_run: int) -> np.ndarray:
    """Pixels of `mask` in vertical runs no longer than max_run."""
    out = np.zeros_like(mask)
    for c in range(mask.shape[1]):
        for s, e in _runs(mask[:, c]):
            if e - s + 1 <= max_run:
                out[s:e + 1, c] = True
    return out


def _flat_pieces(thin: np.ndarray, ink: np.ndarray, max_h: int) -> np.ndarray:
    """
    Keep the thin pixels whose ink piece (8-connected) is no taller than a
    line: dashes and the dots of a dotted line. Markers drawn over the
    violin are closed shapes taller than that, even where only their flat
    top or bottom edge survived the thin-row filter.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    flat = np.zeros(n, bool)
    flat[1:] = stats[1:, cv2.CC_STAT_HEIGHT] <= max_h
    # a dash crossed by a marker merges with it; keep long horizontal pieces too
    flat[1:] |= stats[1:, cv2.CC_STAT_WIDTH] >= 4 * stats[1:, cv2.CC_STAT_HEIGHT]
    return thin & flat[labels]


def _band_strength(contrast, inside, centre, line_w) -> float:
    r0, r1 = int(centre - line_w), int(centre + line_w) + 1
    sub_c, sub_i = contrast[max(0, r0):r1], inside[max(0, r0):r1]
    return float(sub_c.sum() / max(1, sub_i.sum()))


def _inner_lines(contrast, inside, lefts, rights, rows, line_w) -> List[float]:
    """
    Centres of thin bands that cross the violin: contrasting pixels cover a
    good share of the row and both halves of it (a dashed or dotted line
    does; a scatter dot covers one spot).
    """
    ch = contrast.shape[0]
    wmax = float((rights - lefts + 1)[lefts >= 0].max())
    score = np.zeros(ch)
    for r in rows:
        span = inside[r]
        n_in = span.sum()
        if n_in < 6:
            continue
        cols = np.where(span)[0]
        mid = (cols[0] + cols[-1]) / 2.0
        c = contrast[r]
        left = c[cols[cols < mid]].mean() if (cols < mid).any() else 0
        right = c[cols[cols >= mid]].mean() if (cols >= mid).any() else 0
        if min(left, right) >= 0.18:
            score[r] = c[span].mean()
    bands = []
    for s, e in _runs(score >= 0.3):
        if e - s + 1 > 3 * line_w + 3:
            continue                       # a thick blob, not a line
        # a line runs from one side of the violin to the other; a dot near a
        # narrow tip covers a big share of its row but not the whole width
        cols = np.where(contrast[s:e + 1].any(axis=0))[0]
        span_in = np.where(inside[s:e + 1].any(axis=0))[0]
        if cols.size == 0 or cols[-1] - cols[0] < 0.75 * (span_in[-1] - span_in[0]):
            continue
        # quartiles and median sit where most of the data is, so the violin
        # is wide there; its narrow tips hold dots at the extremes
        if (rights[s] - lefts[s] + 1) < 0.3 * wmax:
            continue
        weights = score[s:e + 1]
        bands.append(float(np.average(np.arange(s, e + 1), weights=weights)))
    # merge bands closer than a line width (a line split by anti-aliasing)
    merged = []
    for b in bands:
        if merged and b - merged[-1] <= line_w + 1:
            merged[-1] = (merged[-1] + b) / 2.0
        else:
            merged.append(b)
    return merged


def _inner_box(lab, contrast, inside, lefts, rights, widths, line_w) -> Optional[Tuple[float, Optional[float], float]]:
    """
    ggplot style: a narrow box plot inside the violin. Returns the rows of
    (Q3, median, Q1) inside the region, or None when there is no inner box.
    """
    wmax = widths.max()
    k = max(3, 2 * line_w + 1)
    # the box differs from the violin fill over a long, narrow, solid stretch
    solid = cv2.morphologyEx(contrast.astype(np.uint8), cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    # a box drawn in the violin's own fill colour is just an outline: fill it
    holes = solid.copy()
    flood = np.zeros((holes.shape[0] + 2, holes.shape[1] + 2), np.uint8)
    cv2.floodFill(holes, flood, (0, 0), 1)
    solid = solid | (holes == 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=4)
    centre = float(np.median(((lefts + rights) / 2.0)[lefts >= 0]))
    best = None
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if not (0.08 * wmax <= bw <= 0.45 * wmax and bh >= 4 * line_w + 6):
            continue
        if abs(x + bw / 2.0 - centre) > 0.15 * wmax + 2:
            continue
        if area < 0.6 * bw * bh:
            continue
        # straight vertical sides (a pile of scatter dots has ragged ones)
        comp = labels[y:y + bh, x:x + bw] == i
        rows_l = comp.argmax(axis=1)
        rows_r = bw - 1 - comp[:, ::-1].argmax(axis=1)
        if (np.abs(rows_l - np.median(rows_l)) <= line_w + 1).mean() < 0.8                 or (np.abs(rows_r - np.median(rows_r)) <= line_w + 1).mean() < 0.8:
            continue
        # a line across the violin is wide and thin; a box is narrow and tall
        if bh < 1.2 * bw and bh < 6 * line_w:
            continue
        if best is None or bh > best[3]:
            best = (x, y, bw, bh)
    if best is None:
        return None
    x, y, bw, bh = best
    q3, q1 = y + (line_w - 1) / 2.0, y + bh - 1 - (line_w - 1) / 2.0
    # median: a row inside the box (clear of its outline) that differs from
    # the box's own fill across most of its width
    margin = line_w + 2
    cols = slice(x + max(1, bw // 5), x + bw - max(1, bw // 5))
    body = lab[y + margin:y + bh - margin, cols]
    med = None
    if body.shape[0] >= 3 and body.shape[1] >= 2:
        box_fill = np.median(body.reshape(-1, 3), axis=0)
        frac = (np.linalg.norm(body - box_fill, axis=2) > 45).mean(axis=1)
        cand = [(frac[s:e + 1].mean(), (s + e) / 2.0) for s, e in _runs(frac >= 0.7)
                if e - s + 1 <= 3 * line_w + 4]
        if cand:
            med = y + margin + max(cand)[1]
    return q3, med, q1
