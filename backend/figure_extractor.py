"""
Geometry-based bar chart extraction.

Finds every bar-chart panel in a figure image (multi-panel figures are fine),
calibrates each y-axis from *all* of its tick labels, and measures each bar's
top and upper error-bar cap directly from the pixels.

Pipeline per figure:
    1. Find L-shaped axis pairs (long vertical line meeting a long horizontal
       line at its left end) -> one panel each.
    2. Find y-axis tick marks, OCR the label beside each, and fit
       value = slope * pixel_y + intercept robustly (misread labels are
       rejected as outliers).
    3. Segment bars as solid regions standing on the x-axis (white bars count,
       because their interior is enclosed by the outline), then measure each
       bar's top edge and its error-bar cap.
    4. Name bars from the x-axis labels, or from a matching legend.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

import cv2
import numpy as np


DARK_THRESHOLD = 128      # gray < this counts as "ink" (lines, text, outlines)
WHITE_THRESHOLD = 235     # gray >= this (and unsaturated) counts as background
TEXT_THRESHOLD = 200      # gray < this counts as text when cropping labels


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass
class Bar:
    x1: int
    x2: int
    top_px: float
    cap_px: Optional[float]
    mean: Optional[float] = None
    mean_plus_err: Optional[float] = None
    label: str = ""
    color: Tuple[int, int, int] = (255, 255, 255)
    uncertain: bool = False   # error bar could not be confirmed; worth a manual check
    n: Optional[int] = None   # sample size, entered by the user for the export

    @property
    def err(self) -> Optional[float]:
        if self.mean is None or self.mean_plus_err is None:
            return None
        return self.mean_plus_err - self.mean


@dataclass
class Panel:
    x1: int                  # bounding box of the whole panel incl. labels
    y1: int
    x2: int
    y2: int
    axis_x: float            # y-axis line center (px)
    axis_y: float            # x-axis line center (px)
    plot_top: int
    plot_right: int
    ticks: List[Tuple[float, Optional[float], str]] = field(default_factory=list)
    slope: Optional[float] = None
    intercept: Optional[float] = None
    calibration_inliers: int = 0
    label_left: Optional[int] = None   # left edge of the tick labels (px)
    bars: List[Bar] = field(default_factory=list)
    y_title: str = ""
    letter: str = ""          # panel letter ("A", "B", ...) if one was found
    warnings: List[str] = field(default_factory=list)

    def value_at(self, py: float) -> Optional[float]:
        if self.slope is None:
            return None
        return self.slope * py + self.intercept

    def to_dict(self) -> dict:
        d = asdict(self)
        for b, bd in zip(self.bars, d["bars"]):
            bd["err"] = b.err
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Panel":
        names = {f for f in cls.__dataclass_fields__}
        bar_names = {f for f in Bar.__dataclass_fields__}
        p = cls(**{k: v for k, v in d.items() if k in names and k != "bars"})
        p.bars = [Bar(**{k: v for k, v in b.items() if k in bar_names}) for b in d.get("bars", [])]
        return p


# --------------------------------------------------------------------------- #
# OCR wrapper (RapidOCR: pip-installable, no system binary needed)
# --------------------------------------------------------------------------- #

class OCR:
    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR
        self.engine = RapidOCR()

    def boxes(self, image: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], str, float]]:
        """Detect + recognise all text. Returns [((x1, y1, x2, y2), text, score)]."""
        result, _ = self.engine(image)
        out = []
        for quad, text, score in result or []:
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            out.append(((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))), text, float(score)))
        return out

    def line(self, image: np.ndarray) -> Tuple[str, float]:
        """Recognise a single pre-cropped line of text (no detection step)."""
        h = image.shape[0]
        if h < 32:
            f = 40 / max(h, 1)
            image = cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        image = cv2.copyMakeBorder(image, 8, 8, 12, 12, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        result, _ = self.engine(image, use_det=False, use_cls=False, use_rec=True)
        if not result:
            return "", 0.0
        text, score = result[0]
        return text, float(score)


_NUM_FIX = str.maketrans({"O": "0", "o": "0", "D": "0", "l": "1", "I": "1", "|": "1",
                          "S": "5", "−": "-", "–": "-", "—": "-", ",": ""})


def parse_number(text: str) -> Optional[float]:
    t = text.strip()
    if t in ("。", "o", "O", "()", "D"):     # how a lone "0" tends to come back
        return 0.0
    if re.search(r"[A-Za-z]{2,}", t) or not re.search(r"\d", t):   # words, not a tick label
        return None
    if re.search(r"\d\s+\d", t) or len(re.sub(r"\D", "", t)) > 7:      # digit soup, not a label
        return None
    t = t.translate(_NUM_FIX).replace(" ", "")
    # tolerate stray punctuation the recogniser glues on ("60.", "20÷", "50°")
    t = re.sub(r"^[^\d\-]+|[^\d]+$", "", t)
    m = re.fullmatch(r"-?\d+(\.\d+)?", t)
    return float(t) if m else None


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #

def _runs(mask_1d: np.ndarray) -> List[Tuple[int, int]]:
    """Inclusive (start, end) index pairs of True runs."""
    m = np.concatenate([[False], mask_1d.astype(bool), [False]])
    d = np.diff(m.astype(np.int8))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _line_components(mask: np.ndarray, kernel: Tuple[int, int], max_thickness: int) -> List[Tuple[int, int, int, int]]:
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, kernel))
    n, _, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    comps = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if min(w, h) <= max_thickness:
            comps.append((int(x), int(y), int(w), int(h)))
    return comps


def robust_linear_fit(points: List[Tuple[float, float]]) -> Tuple[Optional[float], Optional[float], int]:
    """
    Fit value = a * px + b to (px, value) pairs, rejecting misread labels.
    Every pair of points proposes a line; the one agreeing with the most
    labels wins and is refined by least squares on its inliers.
    """
    if len(points) < 2:
        return None, None, 0
    pts = np.array(points, dtype=float)
    vrange = max(np.ptp(pts[:, 1]), 1e-9)
    best = (None, None, 0, np.inf)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            (p1, v1), (p2, v2) = pts[i], pts[j]
            if abs(p2 - p1) < 1e-6 or v1 == v2:
                continue
            a = (v2 - v1) / (p2 - p1)
            if a >= 0:          # y grows downward, values grow upward
                continue
            b = v1 - a * p1
            resid = np.abs(a * pts[:, 0] + b - pts[:, 1])
            inl = resid < 0.02 * vrange
            score = (int(inl.sum()), -float(resid[inl].sum()))
            if score > (best[2], -best[3]):
                best = (a, b, int(inl.sum()), float(resid[inl].sum()))
    a, b, n_in, _ = best
    if a is None:
        return None, None, 0
    resid = np.abs(a * pts[:, 0] + b - pts[:, 1])
    inl = pts[resid < 0.02 * vrange]
    if len(inl) >= 2:
        a, b = np.polyfit(inl[:, 0], inl[:, 1], 1)
    return float(a), float(b), n_in


def _is_nice_step(step: float) -> bool:
    """Axis steps are 1, 2, 2.5 or 5 times a power of ten."""
    if step <= 0 or not np.isfinite(step):
        return False
    exp = np.floor(np.log10(step))
    mant = step / 10 ** exp
    return any(abs(mant - m) / m < 0.03 for m in (1, 2, 2.5, 5, 10))


def _plausible_axis(slope: float, tick_ys: List[float], inliers: int) -> bool:
    """
    Reject calibrations built from noise (e.g. a micrograph mistaken for a
    chart): the value step between neighbouring ticks must be a round number.
    """
    ys = sorted(tick_ys)
    if len(ys) < 2 or slope >= 0:       # values must grow upward (pixel y shrinks)
        return False
    gaps = np.diff(ys)
    spacing = float(np.median(gaps))
    return _is_nice_step(abs(slope) * spacing) or (inliers >= 4 and _is_nice_step(abs(slope) * spacing / 2))


# --------------------------------------------------------------------------- #
# Extractor
# --------------------------------------------------------------------------- #

@dataclass
class ExtractOptions:
    """
    What to read besides the numbers. Numbers (axis calibration, bar means,
    error bars) are always extracted; everything here is optional.
    """
    bar_names: bool = True        # category labels under each bar (recognition only, cheap)
    titles: bool = True           # panel letter + y-axis title, used to name sheets (cheap)
    legend_colors: bool = False   # name bars from a colour legend (needs full-figure OCR, slow)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ExtractOptions":
        d = d or {}
        return cls(**{k: bool(v) for k, v in d.items() if k in cls.__dataclass_fields__})


class FigureExtractor:
    def __init__(self, ocr: Optional[OCR] = None) -> None:
        self.ocr = ocr or OCR()

    # ---- public ---------------------------------------------------------- #

    def extract(self, image: np.ndarray, options: Optional[ExtractOptions] = None) -> List[Panel]:
        opts = options or ExtractOptions()
        if image is None or image.size == 0:
            return []
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dark = (gray < DARK_THRESHOLD).astype(np.uint8)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ink = ((gray < WHITE_THRESHOLD) | (hsv[:, :, 1] > 40)).astype(np.uint8)

        panels = []
        for axes in self._find_axes(dark):
            try:
                panel = self._build_panel(image, gray, dark, ink, axes, opts)
            except Exception as e:  # one bad panel must not sink the figure
                print(f"panel at {axes} failed: {type(e).__name__}: {e}")
                continue
            if panel is not None and panel.bars and panel.slope is not None:
                panels.append(panel)
        panels = self._drop_nested(panels)

        if opts.legend_colors and any(not any(b.label for b in p.bars) for p in panels):
            text_boxes = self.ocr.boxes(image)       # slow: only when asked for
            self._assign_legend_labels(image, dark, panels, text_boxes)
        if opts.titles:
            for p in panels:
                p.letter = self._find_letter_glyph(image, dark, p)
        panels.sort(key=lambda p: (round(p.plot_top / max(image.shape[0] * 0.1, 1)), p.axis_x))
        return panels

    @staticmethod
    def _drop_nested(panels: List[Panel]) -> List[Panel]:
        """A long bar edge can pose as a y-axis inside a real panel; keep the outer panel."""
        keep = []
        for p in panels:
            inside = any(q is not p and q.axis_x < p.axis_x - 3 and p.plot_right <= q.plot_right + 3
                         and abs(q.axis_y - p.axis_y) < 6 for q in panels)
            if not inside:
                keep.append(p)
        return keep

    def _find_letter_glyph(self, image, dark, p: Panel) -> str:
        """Text detection often skips a lone letter; look for a big bold glyph up-left of the plot."""
        H, W = dark.shape
        x1, x2 = int(max(0, p.axis_x - 0.25 * W)), int(p.axis_x)
        y1, y2 = int(max(0, p.plot_top - 0.12 * H)), int(p.plot_top + 0.02 * H)
        region = dark[y1:y2, x1:x2]
        if region.size == 0:
            return ""
        n, _, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
        cands = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if 0.012 * H <= h <= 0.05 * H and 0.4 * h <= w <= 1.3 * h and area >= 0.25 * w * h:
                cands.append((x + y, x, y, w, h))
        for _, x, y, w, h in sorted(cands):
            pad = max(3, h // 4)
            crop = image[y1 + max(0, y - pad): y1 + y + h + pad, x1 + max(0, x - pad): x1 + x + w + pad]
            text, score = self.ocr.line(crop)
            if re.fullmatch(r"[A-Z]", text.strip()) and score > 0.6:
                return text.strip()
        return ""

    # ---- step 1: axes ---------------------------------------------------- #

    def _find_axes(self, dark: np.ndarray) -> List[dict]:
        H, W = dark.shape
        kv = max(40, int(0.06 * H))
        kh = max(40, int(0.04 * W))
        verts = _line_components(dark, (1, kv), max_thickness=15)

        # The x-axis is found by scanning rows at the bottom of each vertical
        # line, not by morphology: dark filled bars standing on the axis would
        # merge with it into one thick blob.
        found = []
        for vx, vy, vw, vh in verts:
            v_bottom = vy + vh - 1
            # The x-axis meets the vertical line at or a little above its bottom
            # (a group-separator tick can hang below the axis), so scan the
            # lowest ~20% of the line.
            reach = max(16, int(0.2 * vh))
            rows = []
            for y in range(max(vy, v_bottom - reach), min(H, v_bottom + 16)):
                right_run = dark[y, vx:]
                r = int(np.argmin(right_run)) if not right_run.all() else right_run.size
                left_run = dark[y, :vx][::-1]
                l = int(np.argmin(left_run)) if left_run.size and not left_run.all() else left_run.size
                rows.append((y, r, l))
            if not rows:
                continue
            longest = max(r for _, r, _ in rows)
            if longest < max(kh, 3 * vw):
                continue
            # A bar edge also meets the x-axis, but there the axis continues
            # to the left; at the real y-axis only a short tick does.
            cand = [y for y, r, l in rows if r >= 0.6 * longest and l < 60]
            if not cand:
                continue
            # keep the lowest contiguous band (the axis, not a bar top above it)
            axis_rows = [cand[-1]]
            for y in reversed(cand[:-1]):
                if y != axis_rows[0] - 1:
                    break
                axis_rows.insert(0, y)
            if len(axis_rows) > 15:
                continue
            right = vx + max(r for y, r, _ in rows if y in axis_rows) - 1
            found.append(dict(axis_left=vx, axis_w=vw, top=vy,
                              axis_y=(axis_rows[0] + axis_rows[-1]) / 2.0,
                              axis_h=axis_rows[-1] - axis_rows[0] + 1, right=right))
        # de-duplicate (a thick axis can yield two vertical components)
        uniq = []
        for f in sorted(found, key=lambda f: (f["axis_left"], f["top"])):
            if not any(abs(f["axis_left"] - u["axis_left"]) < 12 and abs(f["axis_y"] - u["axis_y"]) < 12 for u in uniq):
                uniq.append(f)
        return uniq

    # ---- step 2-4: one panel -------------------------------------------- #

    def _build_panel(self, image, gray, dark, ink, ax, opts: ExtractOptions) -> Optional[Panel]:
        H, W = gray.shape
        axis_left, axis_w = ax["axis_left"], ax["axis_w"]
        axis_x = axis_left + (axis_w - 1) / 2.0
        axis_y, axis_h = ax["axis_y"], ax["axis_h"]
        top, right = ax["top"], ax["right"]

        panel = Panel(x1=axis_left, y1=top, x2=right, y2=int(axis_y),
                      axis_x=axis_x, axis_y=axis_y, plot_top=top, plot_right=right)

        self._calibrate(panel, image, dark, axis_left, top, int(round(axis_y)), axis_h)
        if panel.slope is None:
            return None                 # no readable axis: not a chart we can measure

        line_w = max(1, axis_h)
        panel.bars = self._find_bars(image, gray, dark, ink, axis_left + axis_w, right, top, axis_y, line_w)
        for bar in panel.bars:
            bar.mean = panel.value_at(bar.top_px)
            bar.mean_plus_err = panel.value_at(bar.cap_px) if bar.cap_px is not None else bar.mean

        if opts.bar_names and panel.bars:
            self._read_bar_names(image, gray, panel, line_w)
        if opts.titles:
            panel.y_title = self._read_y_title(image, panel)
        panel.y2 = int(min(H - 1, axis_y + 0.12 * H))
        panel.x1 = int(max(0, axis_left - 0.12 * W))
        return panel

    def _calibrate(self, panel: Panel, image, dark, axis_left, top, axis_row, line_w) -> None:
        H, W = dark.shape
        reach = 40
        lo = max(0, axis_left - reach)
        rows = range(max(0, top - 6), min(H, axis_row + line_w + 6))
        run_len = np.zeros(H, dtype=int)
        for y in rows:
            seg = dark[y, lo:axis_left][::-1]
            if seg.size and seg[0]:
                stop = np.argmin(seg) if not seg.all() else seg.size
                run_len[y] = stop
        tick_rows = (run_len >= 3) & (run_len <= reach - 2)
        ticks = []
        for s, e in _runs(tick_rows):
            if e - s + 1 > 3 * line_w + 4:   # too thick to be a tick
                continue
            length = int(np.median(run_len[s:e + 1]))
            ticks.append(((s + e) / 2.0, length))
        if not ticks:
            return
        tick_len = int(np.median([t[1] for t in ticks]))
        ticks = [t for t in ticks if abs(t[1] - tick_len) <= max(3, 0.4 * tick_len)]

        spacing = np.median(np.diff([t[0] for t in ticks])) if len(ticks) > 1 else 40
        half = int(max(8, min(0.45 * spacing, 30)))
        pts = []
        for ty, _ in ticks:
            label_right = axis_left - tick_len - 2
            crop, box = self._crop_label(image, dark, int(ty), half, label_right)
            val, text = None, ""
            if crop is not None:
                text, _ = self.ocr.line(crop)
                val = parse_number(text)
                panel.label_left = min(panel.label_left or box[0], box[0])
            panel.ticks.append((ty, val, text))
            if val is not None:
                pts.append((ty, val))
        a, b, n = robust_linear_fit(pts)
        if a is not None and not _plausible_axis(a, [t[0] for t in ticks], n):
            a, b, n = None, None, 0
        panel.slope, panel.intercept, panel.calibration_inliers = a, b, n
        if n and n < len(ticks):
            panel.warnings.append(f"{len(ticks) - n} of {len(ticks)} tick labels ignored as misreads")

    def _crop_label(self, image, dark, ty, half, label_right):
        """
        Crop the tick label left of a tick: take the character blob nearest
        the tick on its row, then absorb neighbours to the left while the gap
        stays character-sized (so the rotated axis title is left out).
        """
        y1, y2 = max(0, ty - half), min(dark.shape[0], ty + half + 1)
        x0 = max(0, label_right - 250)
        if label_right - x0 < 2 or y2 - y1 < 2:
            return None, ""
        # Small labels are often grey anti-aliased text that the strict "dark"
        # threshold breaks into fragments; text on white is safely < 200.
        band_gray = cv2.cvtColor(image[y1:y2, x0:label_right], cv2.COLOR_BGR2GRAY)
        band = (band_gray < TEXT_THRESHOLD).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(band, connectivity=8)
        blobs = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            cy = y + h / 2
            if area < 4 or abs(cy - (ty - y1)) > 0.6 * half or h > 2.2 * half:
                continue
            blobs.append([x, y, x + w - 1, y + h - 1])
        if not blobs:
            return None, ""
        blobs.sort(key=lambda b: -b[2])
        if band.shape[1] - 1 - blobs[0][2] > 40:
            return None, ""
        text_h = max(b[3] - b[1] + 1 for b in blobs[:1])
        group = [blobs[0]]
        for b in blobs[1:]:
            left = min(g[0] for g in group)
            overlaps_row = b[3] >= min(g[1] for g in group) and b[1] <= max(g[3] for g in group)
            if overlaps_row and left - b[2] <= max(3, 0.6 * text_h):
                group.append(b)
        gx1 = min(g[0] for g in group) + x0
        gx2 = max(g[2] for g in group) + x0
        gy1 = min(g[1] for g in group) + y1
        gy2 = max(g[3] for g in group) + y1
        pad = max(3, int(0.35 * (gy2 - gy1 + 1)))
        crop = image[max(0, gy1 - pad): gy2 + pad + 1, max(0, gx1 - pad): gx2 + pad + 1]
        return crop, (gx1, gy1, gx2, gy2)

    def _find_bars(self, image, gray, dark, ink, x_lo, x_hi, top, axis_y, line_w) -> List[Bar]:
        axis_row = int(round(axis_y))
        y_lo = max(0, top)
        y_hi = axis_row + 1
        region_ink = ink[y_lo:y_hi, x_lo:x_hi + 1]
        if region_ink.size == 0:
            return []
        # Background = white area connected to the top of the plot. Anything
        # enclosed (white bar interiors, hollow dots) counts as foreground.
        bg = (1 - region_ink).astype(np.uint8)
        n, labels = cv2.connectedComponents(bg, connectivity=4)
        outside_labels = set(np.unique(labels[0, :])) - {0}
        outside = np.isin(labels, list(outside_labels)) & (bg == 1)
        fg = ~outside
        h = fg.shape[0]

        # Column profile: unbroken foreground height standing on the x-axis.
        start = h - 1 - (line_w // 2) - 1
        run = np.zeros(fg.shape[1], dtype=int)
        for c in range(fg.shape[1]):
            col = fg[: start + 1, c][::-1]
            run[c] = np.argmin(col) if not col.all() else col.size
        min_h = max(3, line_w + 1)
        segs = [(s, e) for s, e in _runs(run >= min_h) if e - s + 1 >= 6]
        if not segs:
            return []
        widths = np.array([e - s + 1 for s, e in segs])
        typical = np.median(widths[widths >= np.percentile(widths, 50)])
        segs = [(s, e) for s, e in segs if (e - s + 1) >= 0.45 * typical]

        bars = []
        for s, e in segs:
            w = e - s + 1
            inner = slice(s + max(1, int(0.15 * w)), e - max(1, int(0.15 * w)) + 1)
            frac = fg[: start + 1, inner].mean(axis=1)
            # topmost row such that every row beneath it is (almost) solid
            solid = frac >= 0.9
            y = start
            while y >= 0 and solid[y]:
                y -= 1
            bar_top_row = y + 1
            if start - bar_top_row < 1:
                continue
            # Dots and the cap resting on the bar can enclose white pockets that
            # look solid. The real top edge is an ink line spanning (almost) the
            # whole bar, so step down to the first such row.
            # Bars without a dark outline have no such line: stop above the
            # x-axis (itself a full-width dark line) and keep the fill edge.
            span = slice(s + 2, e - 1)
            search_end = start - line_w - 1          # rows above the axis line
            has_outline = False
            if search_end > bar_top_row:
                dark_frac = dark[y_lo + bar_top_row: y_lo + search_end + 1,
                                 x_lo + span.start: x_lo + span.stop].mean(axis=1)
                full = np.where(dark_frac >= 0.85)[0]
                if full.size:
                    first = int(full[0])
                    band_end = first
                    while band_end + 1 < len(dark_frac) and dark_frac[band_end + 1] >= 0.85:
                        band_end += 1
                    thin = band_end - first + 1 <= 2 * line_w + 3
                    reaches_axis = band_end == len(dark_frac) - 1      # dark-filled bar
                    if thin or reaches_axis:
                        bar_top_row += first
                        has_outline = True
            # An outline is centred on the value; a plain fill starts at it.
            top_px = y_lo + bar_top_row + ((line_w - 1) / 2.0 if has_outline else -0.5)
            # Trace the error bar in black ink first. Some journals draw it in
            # the group's colour instead; only then fall back to any ink (the
            # looser mask also merges anti-aliased dots into the whisker).
            cap_candidates = self._cap_candidates(dark, y_lo + bar_top_row, x_lo + s, x_lo + e, line_w, y_lo)
            if cap_candidates["bar_top_y"] - cap_candidates["whisker_end"] <= line_w + 1:
                colored = self._cap_candidates(ink, y_lo + bar_top_row, x_lo + s, x_lo + e, line_w, y_lo)
                if colored["bar_top_y"] - colored["whisker_end"] > line_w + 1:
                    cap_candidates = colored

            fill = image[y_lo + bar_top_row + line_w + 2: axis_row - line_w - 1, x_lo + inner.start: x_lo + inner.stop]
            color = (255, 255, 255)
            if fill.size:
                px = fill.reshape(-1, 3)
                g = cv2.cvtColor(fill, cv2.COLOR_BGR2GRAY).reshape(-1)
                keep = px[(g > 40)] if (g > 40).sum() > 10 else px
                color = tuple(int(v) for v in np.median(keep, axis=0))
            bars.append(Bar(x1=x_lo + s, x2=x_lo + e, top_px=top_px, cap_px=None, color=color))
            bars[-1]._caps = cap_candidates
        self._choose_caps(bars, line_w)
        return bars

    def _cap_candidates(self, dark, bar_top_y, x1, x2, line_w, plot_top) -> dict:
        """
        Follow the whisker up from the bar top and list every horizontal ink
        band crossing it that could be the cap: [(y_center, length)].
        """
        w = x2 - x1 + 1
        cx = (x1 + x2) / 2.0
        win = max(2, int(0.08 * w))
        best_col, best_end = None, bar_top_y
        for c in range(int(cx - win), int(cx + win) + 1):
            y, gap = bar_top_y - 1, 0
            end = bar_top_y
            while y >= plot_top:
                if dark[y, c]:
                    end, gap = y, 0
                else:
                    gap += 1
                    if gap > 1:
                        break
                y -= 1
            if end < best_end:
                best_col, best_end = c, end
        info = dict(bar_top_y=bar_top_y, whisker_end=best_end, bands=[], width=w,
                    col=best_col, dark=dark)
        if best_col is None:
            return info
        rows = []
        for y in range(best_end, bar_top_y + 1):
            row = dark[y]
            l = r = best_col
            while l > x1 - 2 and row[l - 1]:
                l -= 1
            while r < x2 + 2 and row[r + 1]:
                r += 1
            length = r - l + 1
            symmetric = abs((l + r) / 2 - best_col) <= max(2, 0.06 * w)
            if 0.12 * w <= length <= 0.9 * w and symmetric:
                rows.append((y, length, l, r))
        # merge consecutive rows into bands
        bands = []
        for y, length, l, r in rows:
            if bands and y == bands[-1]["y1"] + 1:
                bands[-1]["y1"] = y
                bands[-1]["lens"].append(length)
                bands[-1]["l"] = min(bands[-1]["l"], l)
                bands[-1]["r"] = max(bands[-1]["r"], r)
            else:
                bands.append(dict(y0=y, y1=y, lens=[length], l=l, r=r))
        for b in bands:
            if b["y1"] - b["y0"] + 1 > 2 * line_w + 2:
                continue
            # A cap's ends float in white space; the top/bottom chord of a
            # scatter dot's ring has the ring continuing just past its ends.
            ends_clean = True
            for ex in (slice(b["l"], b["l"] + 2), slice(b["r"] - 1, b["r"] + 1)):
                above = dark[max(0, b["y0"] - 3): b["y0"] - 1, ex]
                below = dark[b["y1"] + 2: min(b["y1"] + 4, bar_top_y - 1), ex]   # stop at the bar outline
                if (above.size and above.mean() > 0.25) or (below.size and below.mean() > 0.25):
                    ends_clean = False
            if ends_clean:
                info["bands"].append(((b["y0"] + b["y1"]) / 2.0, float(np.median(b["lens"])), b["y1"] - b["y0"] + 1))
        return info

    @staticmethod
    def _choose_caps(bars: List["Bar"], line_w: int) -> None:
        """
        Plotting software draws every cap in a panel at the same width, so the
        width most bars agree on identifies the real caps; bands of any other
        width (scatter dots, brackets) are ignored.
        """
        lengths = []
        for b in bars:
            seen = set()
            for _, length, _ in b._caps["bands"]:
                key = int(round(length / 2.0))
                if key not in seen:
                    lengths.append(length)
                    seen.add(key)
        cap_len = None
        if lengths:
            votes = [sum(abs(o - l) <= max(2, 0.08 * l) for o in lengths) for l in lengths]
            cap_len = lengths[int(np.argmax(votes))] if max(votes) >= 2 or len(bars) == 1 else None

        for b in bars:
            info = b._caps
            top = info["bar_top_y"] + (line_w - 1) / 2.0
            matches = [band for band in info["bands"]
                       if cap_len is not None and abs(band[1] - cap_len) <= max(3, 0.12 * cap_len)
                       and band[0] < info["bar_top_y"] - 1]
            if matches:
                # closest to the panel's cap width wins; near-ties go to the topmost
                best_diff = min(abs(m[1] - cap_len) for m in matches)
                close = [m for m in matches if abs(m[1] - cap_len) <= best_diff + 1]
                b.cap_px = float(min(close, key=lambda m: m[0])[0])
            elif info["bar_top_y"] - info["whisker_end"] <= line_w + 1:
                b.cap_px = top            # error bar too small to show past the outline
            else:
                # Dots touching the cap hide its ends. Fall back to the topmost
                # row where ink spans a full cap width centred on the whisker.
                b.cap_px = float(info["whisker_end"] + (line_w - 1) / 2.0)
                if cap_len is not None and info["col"] is not None:
                    half = int(cap_len // 2) - 1
                    c = info["col"]
                    covered = [y for y in range(info["whisker_end"], info["bar_top_y"] - 1)
                               if info["dark"][y, c - half: c + half + 1].mean() >= 0.95]
                    if covered:
                        band = [covered[0]]
                        for y in covered[1:]:
                            if y != band[-1] + 1:
                                break
                            band.append(y)
                        b.cap_px = float(np.mean(band))
                b.uncertain = True
            del b._caps

    # ---- labels ---------------------------------------------------------- #

    @staticmethod
    def _is_label_text(text: str) -> bool:
        t = text.strip()
        if not re.search(r"[A-Za-z]", t):          # "***", "##", stray marks
            return False
        return t.lower().replace(" ", "") not in ("n.s.", "ns", "n.s")

    def _read_bar_names(self, image, gray, panel: Panel, line_w: int) -> None:
        """
        Category labels under the x-axis, without running text detection:
        find the text rows from the ink profile, split each row into words by
        column gaps, and recognise each word group. A second row spanning a
        bracket ("Res", "Sus") becomes a prefix for the bars under it.
        """
        H, W = gray.shape
        bars = panel.bars
        centers = np.array([(b.x1 + b.x2) / 2 for b in bars])
        pitch = float(np.median(np.diff(centers))) if len(bars) > 1 else 2.0 * (bars[0].x2 - bars[0].x1)
        x0 = int(max(0, centers[0] - pitch / 2))
        x1 = int(min(W, centers[-1] + pitch / 2))
        y0 = int(panel.axis_y + line_w + 2)
        y1 = int(min(H, panel.axis_y + 0.35 * (panel.axis_y - panel.plot_top) + 40))
        if y1 - y0 < 6 or x1 - x0 < 6:
            return
        text = (gray[y0:y1, x0:x1] < TEXT_THRESHOLD).astype(np.uint8)
        # drop x-axis tick marks: blobs touching the top edge of the strip
        n, lab, stats, _ = cv2.connectedComponentsWithStats(text, connectivity=8)
        for i in range(1, n):
            if stats[i, 1] <= 2 and stats[i, 3] < 0.12 * (y1 - y0) + 12 and stats[i, 2] <= 2 * line_w + 4:
                text[lab == i] = 0

        rows = _runs(text.any(axis=1))
        bands = []           # text rows; thin wide rows are brackets
        for s, e in rows:
            h = e - s + 1
            # a bracket has an unbroken horizontal run; text never does
            longest = max((rb - ra + 1 for y in range(s, e + 1) for ra, rb in _runs(text[y])), default=0)
            if longest > 0.6 * pitch:
                bands.append(("bracket", s, e))
            elif h >= 5:
                bands.append(("text", s, e))
        text_rows = [b for b in bands if b[0] == "text"]
        if not text_rows:
            return
        # Labels must start right under the axis; rotated labels make one tall
        # row that recognition can't read, so skip those rather than guess.
        first = text_rows[0]
        char_h = first[2] - first[1] + 1
        if first[1] > 3 * char_h + 10 or char_h > 0.18 * (panel.axis_y - panel.plot_top) + 20:
            return

        def words(s, e):
            cols = text[s:e + 1].any(axis=0)
            h = e - s + 1
            out, cur = [], None
            for a, b in _runs(cols):
                if cur and a - cur[1] <= 0.8 * h:
                    cur[1] = b
                else:
                    if cur:
                        out.append(cur)
                    cur = [a, b]
            if cur:
                out.append(cur)
            return out

        def read(s, e, a, b):
            pad = max(2, (e - s + 1) // 4)
            crop = image[max(0, y0 + s - pad): y0 + e + pad + 1, max(0, x0 + a - pad): x0 + b + pad + 1]
            if crop.size == 0:
                return ""
            txt, score = self.ocr.line(crop)
            txt = txt.strip()
            return txt if score > 0.5 and self._is_label_text(txt) else ""

        for a, b in words(first[1], first[2]):
            cx = x0 + (a + b) / 2
            i = int(np.argmin(np.abs(centers - cx)))
            if abs(centers[i] - cx) <= 0.6 * pitch:
                name = read(first[1], first[2], a, b)
                if name:
                    bars[i].label = f"{bars[i].label} {name}".strip()

        # second-level group labels
        later = [b for b in bands if b[1] > first[2]]
        for k, (kind, s, e) in enumerate(later[:3]):
            if kind != "text" or s - first[2] > 4 * char_h:
                continue
            brackets = [bb for bb in later[:k] if bb[0] == "bracket"]
            for a, b in words(s, e):
                name = read(s, e, a, b)
                if not name:
                    continue
                cx = (a + b) / 2
                span = None
                for _, bs, be in brackets:
                    for y in range(bs, be + 1):
                        for ra, rb in _runs(text[y]):
                            if ra - 3 <= cx <= rb + 3 and rb - ra > 0.6 * pitch:
                                span = (x0 + ra, x0 + rb)
                if span is None:
                    continue
                for bar, c in zip(bars, centers):
                    if span[0] - 3 <= c <= span[1] + 3:
                        bar.label = f"{name} {bar.label}".strip()

    @staticmethod
    def _bracket_span(dark, cx, label_top, axis_y) -> Optional[Tuple[int, int]]:
        for y in range(label_top - 1, max(axis_y, label_top - 60), -1):
            if not dark[y, cx]:
                continue
            l = r = cx
            while l > 0 and dark[y, l - 1]:
                l -= 1
            while r < dark.shape[1] - 1 and dark[y, r + 1]:
                r += 1
            if r - l > 40:
                return l, r
        return None

    def _read_y_title(self, image, panel: Panel) -> str:
        """
        The rotated y-axis title: the column of ink just left of the tick
        labels. Cropped tightly and read with recognition only (running text
        detection on every panel was the main cost of a batch run).
        """
        H = image.shape[0]
        label_left = int(panel.label_left if panel.label_left is not None else panel.axis_x - 60)
        x2 = max(0, label_left - 3)
        x1 = max(0, x2 - 110)
        span = panel.axis_y - panel.plot_top
        y1, y2 = int(max(0, panel.plot_top - 0.35 * span)), int(min(H, panel.axis_y + 0.2 * span))
        gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        if gray.size == 0:
            return ""
        ink = gray < DARK_THRESHOLD
        cols = ink.sum(axis=0) > 2
        runs = [r for r in _runs(cols) if r[1] - r[0] >= 8]
        if not runs:
            return ""
        c1, c2 = runs[-1]                          # nearest band to the tick labels
        rows = np.where(ink[:, c1:c2 + 1].any(axis=1))[0]
        # keep the longest vertical stretch without big gaps (drops stray marks)
        segs, start = [], rows[0]
        for a, b in zip(rows[:-1], rows[1:]):
            if b - a > 2 * (c2 - c1 + 1):
                segs.append((start, a))
                start = b
        segs.append((start, rows[-1]))
        r1, r2 = max(segs, key=lambda s: s[1] - s[0])
        pad = 4
        crop = image[y1 + max(0, r1 - pad): y1 + r2 + pad + 1, x1 + max(0, c1 - pad): x1 + c2 + pad + 1]
        text, score = self.ocr.line(cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE))
        return text.strip() if score > 0.5 else ""

    def _assign_legend_labels(self, image, dark, panels: List[Panel], text_boxes) -> None:
        """For panels without x-axis labels, match bar fill colours to legend swatches."""
        need = [p for p in panels if not any(b.label for b in p.bars)]
        if not need:
            return
        swatches = self._find_swatches(image, dark, text_boxes)
        if not swatches:
            return
        # group swatches into legends by vertical stacking at the same x
        legends: List[List[dict]] = []
        for s in sorted(swatches, key=lambda s: (s["x"], s["y"])):
            for lg in legends:
                if abs(lg[-1]["x"] - s["x"]) < 10 and 0 < s["y"] - lg[-1]["y"] < 3 * s["h"]:
                    lg.append(s)
                    break
            else:
                legends.append([s])

        def cdist(c1, c2):
            return float(np.linalg.norm(np.array(c1, float) - np.array(c2, float)))

        for p in need:
            best = None
            pcx = (p.x1 + p.x2) / 2
            pcy = (p.y1 + p.y2) / 2
            for lg in legends:
                if len(lg) < 2:
                    continue
                cost = sum(min(cdist(b.color, s["color"]) for s in lg) for b in p.bars) / len(p.bars)
                dist = np.hypot(lg[0]["x"] - pcx, lg[0]["y"] - pcy) / max(image.shape)
                score = cost + 20 * dist + (15 if len(lg) < len(p.bars) else 0)
                if len(lg) == len(p.bars):
                    score -= 10        # a legend with exactly one entry per bar is the likely match
                if best is None or score < best[0]:
                    best = (score, lg, cost)
            if best is None or best[2] > 40:
                continue
            lg = best[1]
            cost = np.array([[cdist(b.color, s["color"]) for s in lg] for b in p.bars])
            if len(lg) >= len(p.bars):
                # one legend entry per bar
                from scipy.optimize import linear_sum_assignment
                rows, cols = linear_sum_assignment(cost)
                for r, c in zip(rows, cols):
                    p.bars[r].label = lg[c]["text"]
            else:
                for i, b in enumerate(p.bars):
                    b.label = lg[int(np.argmin(cost[i]))]["text"]

    @staticmethod
    def _find_swatches(image, dark, text_boxes) -> List[dict]:
        out = []
        for (x1, y1, x2, y2), text, _ in text_boxes:
            if not FigureExtractor._is_label_text(text) or len(text.strip()) < 2:
                continue
            h = y2 - y1
            cy = (y1 + y2) // 2
            # look for a filled/outlined square just left of the text
            lo = max(0, x1 - 4 * h)
            band = dark[max(0, cy - h // 2): cy + h // 2 + 1, lo:max(lo, x1 - 2)]
            if band.size == 0:
                continue
            cols = np.where(band.any(axis=0))[0]
            if cols.size < 5:
                continue
            sx1, sx2 = lo + cols[0], lo + cols[-1]
            if not (0.5 * h <= sx2 - sx1 <= 2.5 * h):
                continue
            inner = image[cy - max(1, h // 6): cy + max(1, h // 6) + 1,
                          sx1 + (sx2 - sx1) // 3: sx2 - (sx2 - sx1) // 3 + 1]
            if inner.size == 0:
                continue
            color = tuple(int(v) for v in np.median(inner.reshape(-1, 3), axis=0))
            out.append(dict(x=int(sx1), y=int(cy), h=h, color=color, text=text.strip()))
        return out
