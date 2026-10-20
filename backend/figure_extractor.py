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

def _plain(obj):
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj

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
    kind: str = "bar"         # "bar", or "dot" for mean-line-and-whisker scatter plots
    error_type: str = "sem"   # what the error bars show: "sem" or "sd" (from the caption, editable)
    warnings: List[str] = field(default_factory=list)

    def value_at(self, py: float) -> Optional[float]:
        if self.slope is None:
            return None
        return self.slope * py + self.intercept

    def to_dict(self) -> dict:
        """JSON-ready dict (numpy scalars from OpenCV converted to plain Python)."""
        d = asdict(self)
        for b, bd in zip(self.bars, d["bars"]):
            bd["err"] = b.err
        return _plain(d)

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


def _merge_collinear(comps: List[Tuple[int, int, int, int]], min_len: int,
                     max_gap: int = 6) -> List[Tuple[int, int, int, int]]:
    """
    Join vertical line pieces at the same x separated by a few pixels (a thin
    white line across a scanned figure can cut an axis in two), then keep
    lines at least min_len long.
    """
    comps = sorted(comps, key=lambda c: (c[0], c[1]))
    merged: List[List[int]] = []
    for x, y, w, h in comps:
        for m in merged:
            if abs(m[0] - x) <= 2 and abs(m[2] - w) <= 3 and 0 <= y - (m[1] + m[3]) <= max_gap:
                m[3] = y + h - m[1]
                m[2] = max(m[2], w)
                break
        else:
            merged.append([x, y, w, h])
    return [tuple(m) for m in merged if m[3] >= min_len]


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
        for axes in self._find_axes(dark, ink):
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

    def _find_axes(self, dark: np.ndarray, ink: np.ndarray) -> List[dict]:
        H, W = dark.shape
        kv = max(40, int(0.06 * H))
        kh = max(40, int(0.04 * W))
        verts = _merge_collinear(_line_components(dark, (1, max(20, kv // 3)), max_thickness=15), kv)

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
            # Coloured bars are often drawn over the axis line, and a hollow
            # scatter dot at zero blanks a few pixels of it; carry on along any
            # ink, bridging dot-sized gaps (neighbouring panels sit further away).
            band = ink[axis_rows[0]:axis_rows[-1] + 1].any(axis=0)
            x, gap = right, 0
            while x + 1 < W:
                x += 1
                if band[x]:
                    right, gap = x, 0
                else:
                    gap += 1
                    if gap > 12:
                        break
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
        x_lo = axis_left + axis_w
        # Bars stand on zero. When the axis runs below zero, zero is a row
        # inside the plot: bars above it point up, bars below it point down.
        base_y = axis_y
        v_bottom = panel.value_at(axis_y)
        if v_bottom < 0:
            # zero row, clamped to the plot (an all-negative axis has it at the top)
            base_y = float(min(axis_y, max(top, -panel.intercept / panel.slope)))
        tick_rows = [t[0] for t in panel.ticks]
        panel.bars = []
        if base_y > top + 3 * line_w:
            panel.bars = self._find_bars(image, gray, dark, ink, x_lo, right, top, base_y, line_w, tick_rows)
        if base_y < axis_y - 3 * line_w:
            below = self._find_bars_below(image, gray, dark, ink, x_lo, right, base_y, axis_y, line_w, tick_rows)
            both = panel.bars + below
            if both:
                # bars in one chart share a width; narrower blobs are dots
                typical = float(np.median([b.x2 - b.x1 for b in both]))
                both = [b for b in both if b.x2 - b.x1 >= 0.6 * typical]
            panel.bars = sorted(both, key=lambda b: b.x1)
        if len(panel.bars) < 2:
            # scatter/dot plot: mean line with error whiskers, no bars
            dots = self._find_dot_groups(ink, axis_left + axis_w, right, top, axis_y, line_w)
            if dots:
                panel.bars, panel.kind = dots, "dot"
        if len(panel.bars) < 2:
            # Real one-bar charts are vanishingly rare; a lone "bar" is far more
            # often a box plot's box or a filled shape, so don't report it.
            return None
        for bar in panel.bars:
            # the cap lies beyond the mean, away from zero (up for positive bars)
            down = bar.top_px > base_y
            if bar.cap_px is not None and ((bar.cap_px > bar.top_px + 0.5) != down) and abs(bar.cap_px - bar.top_px) > 0.5:
                bar.cap_px, bar.uncertain = bar.top_px, True
            bar.mean = panel.value_at(bar.top_px)
            err = abs(panel.value_at(bar.cap_px) - bar.mean) if bar.cap_px is not None else 0.0
            bar.mean_plus_err = bar.mean + err      # error stored as a positive magnitude

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
        if ticks:
            tick_len = int(np.median([t[1] for t in ticks]))
            ticks = [t for t in ticks if abs(t[1] - tick_len) <= max(3, 0.4 * tick_len)]
        if len(ticks) < 2:
            # Some styles draw no tick marks: labels are centred on their value,
            # so the label rows beside the axis stand in for ticks.
            ticks = self._label_rows_as_ticks(image, axis_left, top, axis_row, line_w)
            tick_len = 0
        if not ticks:
            return

        spacing = np.median(np.diff([t[0] for t in ticks])) if len(ticks) > 1 else 40
        half = int(max(8, min(0.45 * spacing, 30)))
        pts = []
        for ty, _ in ticks:
            label_right = axis_left - tick_len - (2 if tick_len else 4)   # keep the axis line out
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
        if a is not None and n < max(2, int(np.ceil(0.4 * len(ticks)))):
            a, b, n = None, None, 0      # too few labels agree: likely not an axis
        panel.slope, panel.intercept, panel.calibration_inliers = a, b, n
        if n and n < len(ticks):
            panel.warnings.append(f"{len(ticks) - n} of {len(ticks)} tick labels ignored as misreads")

    @staticmethod
    def _label_rows_as_ticks(image, axis_left, top, axis_row, line_w) -> List[Tuple[float, int]]:
        """Vertical centres of the text rows right-aligned against the y-axis."""
        H = image.shape[0]
        x0, x1 = max(0, axis_left - 120), max(0, axis_left - 1)
        y0, y1 = max(0, top - 15), min(H, axis_row + line_w + 15)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return []
        text = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY) < TEXT_THRESHOLD
        # only text ending close to the axis (tick labels, not the axis title)
        near = text[:, max(0, text.shape[1] - 30):]
        rows = []
        for s, e in _runs(near.any(axis=1)):
            h = e - s + 1
            if 5 <= h <= 60:
                rows.append((y0 + (s + e) / 2.0, 0))
        if len(rows) < 2:
            return []
        gaps = np.diff([r[0] for r in rows])
        if gaps.min() < 0.5 * np.median(gaps):      # stray marks, not a label column
            return []
        return rows

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
        # never pad rightwards into the tick or axis line (it reads as "l" or "1")
        crop = image[max(0, gy1 - pad): gy2 + pad + 1, max(0, gx1 - pad): min(gx2 + pad, label_right - 1) + 1]
        return crop, (gx1, gy1, gx2, gy2)

    def _find_bars_below(self, image, gray, dark, ink, x_lo, x_hi, base_y, axis_y, line_w,
                         grid_rows: Optional[List[float]] = None) -> List[Bar]:
        """
        Negative bars hang down from the zero row. Mirror the strip between
        zero and the x-axis vertically so they stand "up" on zero, reuse the
        upward detector, then map the rows back.
        """
        y_a = max(0, int(base_y) - line_w - 2)
        y_b = min(gray.shape[0] - 1, int(round(axis_y)))      # start at the axis line itself
        if y_b - y_a < 8:
            return []

        def flip(a):
            return np.ascontiguousarray(a[y_a:y_b + 1][::-1])

        base_f = y_b - base_y
        mirrored = [y_b - ty for ty in (grid_rows or []) if y_a <= ty <= y_b]
        bars = self._find_bars(flip(image), flip(gray), flip(dark), flip(ink), x_lo, x_hi, 0, base_f, line_w, mirrored)
        for b in bars:
            b.top_px = y_b - b.top_px
            if b.cap_px is not None:
                b.cap_px = y_b - b.cap_px
        # a "bar" filling the whole strip is the x-axis region, not data
        return [b for b in bars if abs(b.top_px - base_y) > line_w + 1]

    def _find_bars(self, image, gray, dark, ink, x_lo, x_hi, top, axis_y, line_w,
                   grid_rows: Optional[List[float]] = None) -> List[Bar]:
        axis_row = int(round(axis_y))
        y_lo = max(0, top)
        y_hi = axis_row + 1
        region_ink = ink[y_lo:y_hi, x_lo:x_hi + 1]
        if region_ink.size == 0:
            return []
        # Light gridlines would cut the background into enclosed strips that
        # look like bar interior. A gridline is a thin row band where light
        # ink covers a large share of the width (it shows between the bars);
        # light-filled bars make thick bands instead, so they are kept.
        region_gray = gray[y_lo:y_hi, x_lo:x_hi + 1]
        # only vertically thin light ink can be a gridline; bar fill is tall
        tall = cv2.morphologyEx(region_ink, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))).astype(bool)
        light = (region_gray > 140) & (region_ink == 1) & ~tall
        frac = light.mean(axis=1)
        keep_bottom = region_ink.shape[0] - (2 * line_w + 3)     # x-axis region stays
        region_ink = region_ink.copy()
        for s, e in _runs(frac >= 0.25):
            if e - s + 1 <= 3 and e < keep_bottom:
                region_ink[s:e + 1][light[s:e + 1]] = 0
        # Gridlines sit on the tick rows and leave the y-axis: with only a
        # couple of wide bars little of them shows, so check those rows directly.
        for ty in grid_rows or []:
            r = int(round(ty)) - y_lo
            if 2 <= r < keep_bottom:
                band = slice(r - 2, r + 3)
                if light[band, :8].any(axis=0).mean() > 0.6:
                    region_ink[band][light[band]] = 0
        # Background = white area connected to the top of the plot. Anything
        # enclosed (white bar interiors, hollow dots) counts as foreground.
        bg = (1 - region_ink).astype(np.uint8)
        n, labels = cv2.connectedComponents(bg, connectivity=4)
        # seed from the top few rows (a boxed plot's frame line covers row 0)
        # plus the largest white region, which is the plot's open background
        seed_rows = labels[: min(labels.shape[0], 2 * line_w + 4), :]
        outside_labels = set(np.unique(seed_rows)) - {0}
        if n > 1:
            areas = np.bincount(labels[bg == 1].ravel(), minlength=n)
            areas[0] = 0
            outside_labels.add(int(np.argmax(areas)))
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
            gap_max = max(2, line_w)
            while y >= 0:
                if solid[y]:
                    y -= 1
                    continue
                # A thin light line across the bars (gridline, scan artefact)
                # is not the top if solid fill resumes right above it.
                g = y
                while g >= 0 and not solid[g] and y - g < gap_max:
                    g -= 1
                if g - 2 >= 0 and solid[g - 2:g + 1].all() and y - g <= gap_max:
                    y = g
                    continue
                break
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

    def _find_dot_groups(self, ink, x_lo, x_hi, top, axis_y, line_w) -> List[Bar]:
        """
        Dot plots draw each group's mean as a short horizontal line, with a
        narrower cap above it joined by a vertical whisker. Find horizontal
        segments, group them by centre, and read mean and upper cap.
        """
        y_hi = int(axis_y) - line_w - 1
        if y_hi - top < 20 or x_hi - x_lo < 20:
            return []
        region = ink[top:y_hi, x_lo:x_hi + 1]
        k = max(15, int(0.06 * (x_hi - x_lo)))
        opened = cv2.morphologyEx(region, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
        n, _, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
        segs = []
        for i in range(1, n):
            x, y, w, h, _ = stats[i]
            # mean lines can be drawn thicker than the axis; require a line shape
            if h > max(2 * line_w + 6, 16) or w < 3 * h:
                continue
            x1, x2, yc = x_lo + x, x_lo + x + w - 1, top + y + (h - 1) / 2.0
            if yc - top < 2 * line_w + 3:
                continue          # plot frame / top tick level, not a mean line
            # significance brackets turn down at both ends
            below = top + y + h + 1                     # first row under the segment
            ends_down = all(ink[below + 1: below + 7, ex].mean() > 0.5
                            for ex in (x1 + 1, x2 - 1)) if below + 7 < ink.shape[0] else False
            # gridlines and frame lines run across the whole plot
            if not ends_down and w < 0.6 * (x_hi - x_lo):
                segs.append(dict(x1=x1, x2=x2, yc=yc, w=w, y0=top + y))
        if not segs:
            return []
        # A bracket spans several groups: drop segments whose extent covers
        # the centres of two or more other column positions.
        centers = sorted((s["x1"] + s["x2"]) / 2 for s in segs)
        cols: List[float] = []
        for c in centers:
            if not cols or c - cols[-1] > 8:
                cols.append(c)
        segs = [s for s in segs
                if sum(s["x1"] - 2 <= c <= s["x2"] + 2 for c in cols) <= 1]
        if not segs:
            return []
        widest = max(s["w"] for s in segs)
        groups: List[List[dict]] = []
        for s in sorted(segs, key=lambda s: (s["x1"] + s["x2"]) / 2):
            c = (s["x1"] + s["x2"]) / 2
            for g in groups:
                gc = np.mean([(t["x1"] + t["x2"]) / 2 for t in g])
                if abs(gc - c) <= 0.25 * widest + 3:
                    g.append(s)
                    break
            else:
                groups.append([s])

        bars, infos = [], []
        for g in groups:
            mean_line = max(g, key=lambda s: s["w"])
            if mean_line["w"] < 0.5 * widest:
                continue
            # a box plot has box edges and median at the same width: not mean ± error
            if sum(s["w"] >= 0.8 * mean_line["w"] for s in g) >= 2:
                return []
            cx = int(round((mean_line["x1"] + mean_line["x2"]) / 2))
            infos.append(self._dot_cap(ink, cx, mean_line, line_w, top))
            bars.append(Bar(x1=int(mean_line["x1"]), x2=int(mean_line["x2"]),
                            top_px=float(mean_line["yc"]), cap_px=None))
        # a lone segment is more likely a stray line than a plot; mean lines
        # in one panel are drawn at one width
        if len(bars) < 2:
            return []
        widths = np.array([b.x2 - b.x1 for b in bars], dtype=float)
        if widths.max() > 2.0 * widths.min():      # a dot can hide part of a mean line
            return []
        self._choose_dot_caps(bars, infos)
        # A dashed reference line ("100% of control") breaks into equal
        # segments on one row with no whiskers: a dot plot has error bars.
        with_whisker = sum(bool(info and info["bands"]) for info in infos)
        if with_whisker < 0.5 * len(bars):
            return []
        if len(bars) >= 3 and np.ptp([b.top_px for b in bars]) < 1.0:
            return []
        return bars

    @staticmethod
    def _dot_cap(ink, cx, mean_line, line_w, plot_top) -> Optional[dict]:
        """
        Walk up the whisker from a mean line; the cap is a centred horizontal
        run where it ends. A gap is bridged only when a thin whisker clearly
        continues above it (the whisker passing behind a hollow dot).
        """
        w = mean_line["x2"] - mean_line["x1"] + 1
        y = int(mean_line["y0"]) - 1                 # just above the (possibly thick) mean line
        # pick the whisker column (it may sit a pixel or two off centre)
        col = max(range(cx - 2, cx + 3), key=lambda c: ink[max(plot_top, y - 6): y, c].sum())
        def run_width(yy):
            l = r = col
            while l > 0 and ink[yy, l - 1] and col - l <= 20:
                l -= 1
            while r < ink.shape[1] - 1 and ink[yy, r + 1] and r - col <= 20:
                r += 1
            return r - l + 1

        # this whisker's own width: a low percentile over a stretch above the
        # mean line (dots overlapping it only ever make rows wider)
        start_rows = [yy for yy in range(y, max(plot_top, y - 15) - 1, -1) if ink[yy, col]]
        if not start_rows:
            return None
        max_w = int(np.percentile([run_width(yy) for yy in start_rows], 20)) + 2

        def thin(yy):
            """Ink at the whisker column with a whisker-narrow run around it."""
            if not ink[yy, col]:
                return False
            l = r = col
            while l > 0 and ink[yy, l - 1] and col - l <= max_w:
                l -= 1
            while r < ink.shape[1] - 1 and ink[yy, r + 1] and r - col <= max_w:
                r += 1
            return r - l + 1 <= max_w

        end, gap = None, 0
        while y >= plot_top:
            if ink[y, col]:
                end, gap = y, 0
            else:
                gap += 1
                if gap > 1:
                    # Bridge a hollow dot lying on the whisker only if a thin
                    # whisker clearly resumes above it (5 rows in a row); a
                    # dot sitting above the cap never looks like that.
                    resume = None
                    for yy in range(y - 1, max(plot_top + 4, y - 24) - 1, -1):
                        if all(thin(yy - k) for k in range(5)):
                            resume = yy
                            break
                    if resume is None:
                        break
                    y, gap = resume, 0
                    continue
            y -= 1
        if end is None or mean_line["yc"] - end < line_w + 2:
            return None
        # Every cap-like band along the traced whisker: a centred horizontal
        # run whose ends float in white (a dot's ring continues past them).
        bands = []
        yy = end
        stop = int(mean_line["y0"]) - 1
        while yy < stop:
            row = ink[yy]
            l = r = col
            while l > col - w and row[l - 1]:
                l -= 1
            while r < col + w and row[r + 1]:
                r += 1
            length = r - l + 1
            if 0.15 * w <= length <= 1.2 * w and abs((l + r) / 2 - col) <= max(2, 0.15 * w):
                y1 = yy
                while y1 + 1 < stop and ink[y1 + 1, l:r + 1].mean() > 0.8:
                    y1 += 1
                clean = True
                for ex in (l, r):
                    above = ink[max(0, yy - 3): yy - 1, max(0, ex - 1): ex + 2]
                    below = ink[y1 + 2: min(y1 + 4, stop), max(0, ex - 1): ex + 2]
                    if (above.size and above.mean() > 0.34) or (below.size and below.mean() > 0.34):
                        clean = False
                bands.append(((yy + y1) / 2.0, length, clean))
                yy = y1 + 1
            else:
                yy += 1
        return {"end": end, "bands": bands, "width": w, "window": 3 * line_w + 4}

    @staticmethod
    def _choose_dot_caps(bars: List["Bar"], infos: List[Optional[dict]]) -> None:
        """Pick each group's cap: clean-ended, at the width most groups agree on."""
        # each group votes once per width (dots contribute many small chords
        # per group, the real cap only one); ties go to the wider width
        per_group = [sorted({b[1] for b in info["bands"] if b[2]}) for info in infos if info]
        widths = sorted({x for g in per_group for x in g})
        cap_len = None
        if widths:
            votes = [(sum(any(abs(o - x) <= max(2, 0.12 * x) for o in g) for g in per_group), x) for x in widths]
            best_votes, cap_len = max(votes)
            if best_votes < 2:
                cap_len = None
        for bar, info in zip(bars, infos):
            if not info or not info["bands"]:
                bar.cap_px, bar.uncertain = bar.top_px, True
                continue
            # The cap is where the whisker ends. Clean ends at the agreed
            # width confirm it; anything else is reported but flagged.
            near = [b for b in info["bands"] if b[0] - info["end"] <= info["window"]] or info["bands"][:1]
            good = [b for b in near if b[2] and (cap_len is None or abs(b[1] - cap_len) <= max(3, 0.2 * cap_len))]
            pick = min(near)                      # the band at the whisker's end
            bar.cap_px, bar.uncertain = float(pick[0]), pick not in good

    def _cap_candidates(self, dark, bar_top_y, x1, x2, line_w, plot_top, max_gap: int = 1) -> dict:
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
                    if gap > max_gap:
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
            # A cap-width line sitting on the bar edge is only the cap when
            # nothing further up qualifies (some styles cap the mean itself).
            away = [m for m in matches if m[0] < info["bar_top_y"] - line_w - 1]
            if away and len(away) < len(matches):
                matches = away
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
        if x2 - x1 < 4 or y2 - y1 < 4:
            return ""
        gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
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
