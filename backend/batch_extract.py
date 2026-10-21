"""
Batch extraction: papers in, one spreadsheet out.

    python backend/batch_extract.py PATH [PATH ...] --out results.xlsx [--review review_dir]
                                    [--no-names] [--no-titles] [--legend] [--error-type sem|sd]

PATH can be a PDF, an image, or a folder (searched recursively for both).
Every bar-chart (or dot-plot) panel found becomes a sheet in long format;
with --review, an annotated PNG of each figure is written so results can be
eyeballed. Error bars default to what each figure caption says (SEM/SD).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Iterable, List, Optional

import cv2
import numpy as np

from exporter import PanelResult, build_workbook, draw_overlay, paper_id_from_path
from figure_extractor import ExtractOptions, FigureExtractor
from pdf_utils import figures

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def iter_inputs(paths: Iterable[str]) -> List[str]:
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                for n in sorted(names):
                    if os.path.splitext(n)[1].lower() in IMAGE_EXTS | {".pdf"}:
                        files.append(os.path.join(root, n))
        elif os.path.isfile(p):
            files.append(p)
    return files


def figures_in(path: str) -> List[dict]:
    """[{page, figure, caption, error_type, image}] for a PDF or image file."""
    if path.lower().endswith(".pdf"):
        return figures(path)
    # imdecode handles non-ASCII Windows paths that imread cannot
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return []
    return [dict(page=-1, figure="", caption="", error_type="", image=img)]


def extract_file(path: str, extractor: FigureExtractor, options: ExtractOptions,
                 error_type: Optional[str] = None) -> List[tuple]:
    """[(figure dict, [Panel])] for one file; panels inherit the figure's error type."""
    out = []
    for fig in figures_in(path):
        panels = extractor.extract(fig["image"], options)
        et = error_type or fig["error_type"] or "sem"
        for p in panels:
            p.error_type = et
        if panels:
            out.append((fig, panels))
    return out


def run(paths: Iterable[str], out_path: str, review_dir: Optional[str] = None,
        options: Optional[ExtractOptions] = None, error_type: Optional[str] = None) -> List[PanelResult]:
    options = options or ExtractOptions()
    extractor = FigureExtractor()
    results: List[PanelResult] = []
    files = iter_inputs(paths)
    t_all = time.time()
    for fi, path in enumerate(files, 1):
        t0 = time.time()
        pid = paper_id_from_path(path)
        found = 0
        for fig, panels in extract_file(path, extractor, options, error_type):
            for p in panels:
                results.append(PanelResult(source=os.path.basename(path), page=fig["page"],
                                           figure=fig["figure"], panel=p, paper_id=pid))
            found += len(panels)
            if review_dir:
                os.makedirs(review_dir, exist_ok=True)
                # paper titles make long file names; Windows paths stop at 260 chars
                stem = os.path.splitext(os.path.basename(path))[0][:60].rstrip(" .")
                tag = f"fig{fig['figure']}" if fig["figure"] else f"page{fig['page'] + 1}"
                ok, buf = cv2.imencode(".png", draw_overlay(fig["image"], panels))
                if ok:
                    try:
                        buf.tofile(os.path.join(review_dir, f"{stem}_{tag}_{fig['page']}.png"))
                    except OSError as e:   # a review image must not stop the run
                        print(f"  could not save review image: {e}")
        print(f"[{fi}/{len(files)}] {os.path.basename(path)}: {found} panels ({time.time() - t0:.1f}s)")

    if not results:
        print("No bar charts or box plots found.")
        return results
    build_workbook(results).save(out_path)
    print(f"\nWrote {len(results)} panels, {sum(len(r.panel.bars) for r in results)} bars "
          f"in {time.time() - t_all:.0f}s -> {out_path}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default="extracted.xlsx")
    ap.add_argument("--review", default=None, help="folder for annotated figure PNGs")
    ap.add_argument("--no-names", action="store_true", help="skip reading bar names")
    ap.add_argument("--no-titles", action="store_true", help="skip panel letters and axis titles")
    ap.add_argument("--legend", action="store_true", help="name bars from colour legends (slower)")
    ap.add_argument("--error-type", choices=["sem", "sd"], default=None,
                    help="override what the error bars show (default: read from each caption, else SEM)")
    args = ap.parse_args()
    opts = ExtractOptions(bar_names=not args.no_names, titles=not args.no_titles, legend_colors=args.legend)
    run(args.paths, args.out, args.review, opts, args.error_type)


if __name__ == "__main__":
    sys.exit(main())
