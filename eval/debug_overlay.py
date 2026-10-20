"""Draw what the extractor found on top of a figure: python eval/debug_overlay.py file.pdf page out.png"""

import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))
from figure_extractor import FigureExtractor  # noqa: E402
from pdf_utils import page_images  # noqa: E402


def draw(img, panels):
    out = img.copy()
    for p in panels:
        cv2.line(out, (int(p.axis_x), p.plot_top), (int(p.axis_x), int(p.axis_y)), (255, 0, 255), 2)
        cv2.line(out, (int(p.axis_x), int(p.axis_y)), (p.plot_right, int(p.axis_y)), (255, 0, 255), 2)
        for ty, val, text in p.ticks:
            cv2.circle(out, (int(p.axis_x) - 25, int(ty)), 4, (0, 160, 0) if val is not None else (0, 0, 255), -1)
            cv2.putText(out, str(text), (int(p.axis_x) + 6, int(ty) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 0), 1)
        for b in p.bars:
            cv2.rectangle(out, (b.x1, int(b.top_px)), (b.x2, int(p.axis_y)), (0, 200, 255), 2)
            if b.cap_px is not None:
                cx = (b.x1 + b.x2) // 2
                cv2.line(out, (cx - 15, int(b.cap_px)), (cx + 15, int(b.cap_px)), (255, 0, 0), 2)
            cv2.putText(out, f"{b.mean:.1f}" if b.mean is not None else "?", (b.x1, int(b.top_px) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 1)
        for w in p.warnings:
            print("WARN", w)
    return out


if __name__ == "__main__":
    pdf, page, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    img = page_images(pdf, pages=[page])[0][1]
    panels = FigureExtractor().extract(img)
    for p in panels:
        print(f"panel axis_x={p.axis_x:.0f} axis_y={p.axis_y:.0f} top={p.plot_top} right={p.plot_right} "
              f"slope={p.slope} inl={p.calibration_inliers} ticks={[(round(t[0]), t[2]) for t in p.ticks]} "
              f"bars={len(p.bars)} title={p.y_title!r}")
    cv2.imwrite(dst, draw(img, panels))
