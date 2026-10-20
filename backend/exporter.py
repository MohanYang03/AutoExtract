"""
Spreadsheet export in the meta-analysis "long format".

Each panel gets its own sheet whose first 12 columns match the hand-made
Long_format_*.xlsx files exactly, so rows can be pasted straight across.
`sd` is a live formula (sem * sqrt(sample_size)) that fills in once the
sample size is typed in. Bars whose error bar could not be confirmed are
highlighted for a quick manual check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

from figure_extractor import Panel

LONG_COLUMNS = ["mean", "sd", "age(week)", "sex(male)", "acupoint", "treatment_group",
                "sample_size", "treatment_duration", "depression_model_duration",
                "paper_id", "sem", "mean+sem"]

FLAG_FILL = PatternFill("solid", start_color="FFF2CC")
HEADER_FONT = Font(bold=True)


@dataclass
class PanelResult:
    source: str            # file name the figure came from
    page: int              # 0-based page index (or -1 for an image file)
    figure: str            # figure number from the caption, if found
    panel: Panel
    paper_id: Optional[int] = None

    @property
    def name(self) -> str:
        fig = f"Fig{self.figure}" if self.figure else f"p{self.page + 1}"
        return f"{fig}{self.panel.letter}" if self.panel.letter else fig


def paper_id_from_path(path: str) -> Optional[int]:
    m = re.search(r"\(paper(\d+)\)", path)
    return int(m.group(1)) if m else None


def _sheet_title(base: str, used: set) -> str:
    title = re.sub(r"[\[\]\*\?/\\:]", "", base)[:31] or "Panel"
    candidate, i = title, 2
    while candidate in used:
        suffix = f" ({i})"
        candidate = title[: 31 - len(suffix)] + suffix
        i += 1
    used.add(candidate)
    return candidate


def _write_long_rows(ws, start_row: int, result: PanelResult, first_col: int = 1) -> int:
    """Write one row per bar in LONG_COLUMNS order. Returns the next free row."""
    r = start_row
    col = {name: first_col + i for i, name in enumerate(LONG_COLUMNS)}
    for bar in result.panel.bars:
        mean = bar.mean
        top = bar.mean_plus_err if bar.mean_plus_err is not None else mean
        sem = None if mean is None or top is None else top - mean
        ws.cell(r, col["mean"], mean)
        sem_ref = f"{get_column_letter(col['sem'])}{r}"
        n_ref = f"{get_column_letter(col['sample_size'])}{r}"
        ws.cell(r, col["sd"], f'=IF({n_ref}="","",{sem_ref}*SQRT({n_ref}))')
        ws.cell(r, col["treatment_group"], bar.label)
        if bar.n:
            ws.cell(r, col["sample_size"], bar.n)
        if result.paper_id is not None:
            ws.cell(r, col["paper_id"], result.paper_id)
        ws.cell(r, col["sem"], sem)
        ws.cell(r, col["mean+sem"], top)
        if bar.uncertain:
            for c in range(first_col, first_col + len(LONG_COLUMNS)):
                ws.cell(r, c).fill = FLAG_FILL
        r += 1
    return r


def build_workbook(results: List[PanelResult]) -> Workbook:
    wb = Workbook()
    summary = wb.active
    summary.title = "Summary"
    summary.append(["source", "figure", "panel", "y-axis title", "bars", "ticks used",
                    "flagged bars", "warnings", "sheet"])
    for c in summary[1]:
        c.font = HEADER_FONT

    # One combined sheet: context columns, then the 12 long-format columns.
    combined = wb.create_sheet("All bars")
    combined.append(["source", "figure", "panel", "y-axis title", "check"] + LONG_COLUMNS)
    for c in combined[1]:
        c.font = HEADER_FONT

    used = {"Summary", "All bars"}
    next_combined = 2
    for res in results:
        p = res.panel
        title = _sheet_title(f"{res.name} {p.y_title}".strip(), used)
        ws = wb.create_sheet(title)
        ws.append(LONG_COLUMNS + ["", "check"])
        for c in ws[1]:
            c.font = HEADER_FONT
        _write_long_rows(ws, 2, res)
        for i, bar in enumerate(p.bars):
            if bar.uncertain:
                ws.cell(2 + i, len(LONG_COLUMNS) + 2, "check error bar")
        ws.column_dimensions["F"].width = 28

        for i, bar in enumerate(p.bars):
            row = next_combined + i
            combined.cell(row, 1, res.source)
            combined.cell(row, 2, res.figure)
            combined.cell(row, 3, p.letter)
            combined.cell(row, 4, p.y_title)
            combined.cell(row, 5, "check error bar" if bar.uncertain else "")
        next_combined = _write_long_rows(combined, next_combined, res, first_col=6)

        summary.append([res.source, res.figure, p.letter, p.y_title, len(p.bars),
                        f"{p.calibration_inliers}/{len(p.ticks)}",
                        sum(b.uncertain for b in p.bars), "; ".join(p.warnings), title])

    for ws in (summary, combined):
        for col in range(1, ws.max_column + 1):
            ws.column_dimensions[get_column_letter(col)].width = 16
    return wb


def draw_overlay(image: np.ndarray, panels: List[Panel]) -> np.ndarray:
    """Annotated copy of the figure: axes, ticks, bars, caps, values. For eyeballing results."""
    out = image.copy()
    for p in panels:
        magenta, orange, blue = (200, 0, 200), (0, 150, 255), (220, 60, 0)
        cv2.line(out, (int(p.axis_x), p.plot_top), (int(p.axis_x), int(p.axis_y)), magenta, 2)
        cv2.line(out, (int(p.axis_x), int(p.axis_y)), (p.plot_right, int(p.axis_y)), magenta, 2)
        for ty, val, _ in p.ticks:
            color = (0, 170, 0) if val is not None else (0, 0, 255)
            cv2.circle(out, (int(p.axis_x) + 8, int(ty)), 4, color, -1)
        for b in p.bars:
            cv2.rectangle(out, (b.x1, int(round(b.top_px))), (b.x2, int(p.axis_y)), orange, 2)
            cx = (b.x1 + b.x2) // 2
            if b.cap_px is not None:
                cap_color = (0, 0, 255) if b.uncertain else blue
                cv2.line(out, (cx - 14, int(round(b.cap_px))), (cx + 14, int(round(b.cap_px))), cap_color, 3)
            if b.mean is not None:
                cv2.putText(out, f"{b.mean:.3g}", (b.x1, int(b.top_px) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, blue, 2)
        if p.letter:
            cv2.putText(out, p.letter, (int(p.axis_x) + 10, p.plot_top + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, magenta, 2)
    return out
