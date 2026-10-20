"""Overlay one figure of a paper: python eval/show.py <pdf> <figure number> <out.png> [x0 y0 x1 y1 crop]"""

import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))
from exporter import draw_overlay  # noqa: E402
from figure_extractor import FigureExtractor  # noqa: E402
from pdf_utils import figure_number, page_images  # noqa: E402

pdf, want, dst = sys.argv[1], sys.argv[2], sys.argv[3]
crop = [int(v) for v in sys.argv[4:8]] if len(sys.argv) >= 8 else None
ex = FigureExtractor()
for page, img in page_images(pdf):
    if figure_number(pdf, page) != want:
        continue
    panels = ex.extract(img)
    print(f"page {page} size {img.shape[1]}x{img.shape[0]}")
    for p in panels:
        print(f"  {p.letter or '?'} axis_x={p.axis_x:.0f} axis_y={p.axis_y:.0f} top={p.plot_top} right={p.plot_right} "
              f"ticks={[(round(t[0]), t[2]) for t in p.ticks]} lw-bars={[(b.x1, b.x2, round(b.top_px, 1), b.cap_px and round(b.cap_px, 1), b.mean and round(b.mean, 2), b.mean_plus_err and round(b.mean_plus_err, 2), b.uncertain) for b in p.bars]}")
    out = draw_overlay(img, panels)
    if crop:
        out = out[crop[1]:crop[3], crop[0]:crop[2]]
    cv2.imwrite(dst, out)
    break
