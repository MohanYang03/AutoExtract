"""
Batch extraction: papers in, one spreadsheet out.

    python backend/batch_extract.py PATH [PATH ...] --out results.xlsx [--review review_dir]

PATH can be a PDF, an image, or a folder (searched recursively for both).
Every bar-chart panel found becomes a sheet in long format; with --review,
an annotated PNG of each figure is written so results can be eyeballed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Iterable, List, Tuple

import cv2
import numpy as np

from exporter import PanelResult, build_workbook, draw_overlay, paper_id_from_path
from figure_extractor import FigureExtractor
from pdf_utils import figure_number, page_images

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


def figures_in(path: str) -> List[Tuple[int, str, np.ndarray]]:
    """[(page_index, figure_number, image)] for a PDF or image file."""
    if path.lower().endswith(".pdf"):
        return [(page, figure_number(path, page), img) for page, img in page_images(path)]
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return [(-1, "", img)] if img is not None else []


def run(paths: Iterable[str], out_path: str, review_dir: str | None = None) -> List[PanelResult]:
    extractor = FigureExtractor()
    results: List[PanelResult] = []
    files = iter_inputs(paths)
    for fi, path in enumerate(files, 1):
        t0 = time.time()
        pid = paper_id_from_path(path)
        found = 0
        for page, fig, img in figures_in(path):
            panels = extractor.extract(img)
            for p in panels:
                results.append(PanelResult(source=os.path.basename(path), page=page,
                                           figure=fig, panel=p, paper_id=pid))
            found += len(panels)
            if review_dir and panels:
                os.makedirs(review_dir, exist_ok=True)
                stem = os.path.splitext(os.path.basename(path))[0]
                tag = f"fig{fig}" if fig else f"page{page + 1}"
                dst = os.path.join(review_dir, f"{stem}_{tag}.png")
                ok, buf = cv2.imencode(".png", draw_overlay(img, panels))
                if ok:
                    buf.tofile(dst)
        print(f"[{fi}/{len(files)}] {os.path.basename(path)}: {found} bar-chart panels ({time.time() - t0:.1f}s)")

    build_workbook(results).save(out_path)
    print(f"\nWrote {len(results)} panels, {sum(len(r.panel.bars) for r in results)} bars -> {out_path}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default="extracted.xlsx")
    ap.add_argument("--review", default=None, help="folder for annotated figure PNGs")
    args = ap.parse_args()
    run(args.paths, args.out, args.review)


if __name__ == "__main__":
    sys.exit(main())
