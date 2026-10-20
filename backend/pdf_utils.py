"""
PDF helpers: pull figure images out of papers.

Journal PDFs usually embed each figure as one raster image, so we extract the
embedded image at its native resolution (sharper than rendering the page).
Pages whose figures are vector graphics are rendered at high DPI instead.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np


MIN_FIGURE_PIXELS = 600 * 400


def _to_bgr(pix) -> np.ndarray:
    import pymupdf
    if pix.alpha or pix.n - pix.alpha != 3:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        if pix.alpha:
            pix = pymupdf.Pixmap(pix, 0)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def page_images(pdf_path: str, pages: Optional[Iterable[int]] = None,
                render_dpi: int = 300) -> List[Tuple[int, np.ndarray]]:
    """
    Return [(page_index, BGR image)] for every figure-sized image in the PDF.

    Args:
        pdf_path: path to the PDF.
        pages: optional 0-based page indices to restrict to.
        render_dpi: DPI used when a page has drawings but no large raster image.
    """
    import pymupdf
    doc = pymupdf.open(pdf_path)
    out = []
    for i in (pages if pages is not None else range(len(doc))):
        page = doc[i]
        found = False
        for info in page.get_images(full=True):
            xref = info[0]
            w, h = info[2], info[3]
            if w * h < MIN_FIGURE_PIXELS:
                continue
            out.append((i, _to_bgr(pymupdf.Pixmap(doc, xref))))
            found = True
        if not found and len(page.get_drawings()) > 200:   # vector figure
            out.append((i, _to_bgr(page.get_pixmap(dpi=render_dpi))))
    return out
