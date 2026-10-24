"""
PDF helpers: pull figure images out of papers.

Journal PDFs usually embed each figure as one raster image, so we extract the
embedded image at its native resolution (sharper than rendering the page).
Pages whose figures are vector graphics are rendered at high DPI instead.
Each image is paired with its own caption, so a page holding two figures
reports two figure numbers.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np


MIN_FIGURE_PIXELS = 600 * 400
CAPTION_RE = re.compile(r"\s*(?:Fig\.?|Figure)\s*(\d+)", re.IGNORECASE)


def _to_bgr(pix) -> np.ndarray:
    import pymupdf
    if pix.alpha or pix.n - pix.alpha != 3:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        if pix.alpha:
            pix = pymupdf.Pixmap(pix, 0)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _captions(page) -> List[Tuple[float, str, str]]:
    """[(top y, figure number, caption text)] for caption blocks on the page."""
    out = []
    for block in page.get_text("blocks"):
        m = CAPTION_RE.match(block[4])
        if m:
            out.append((block[1], m.group(1), block[4]))
    return sorted(out)


def _caption_for(rect, captions) -> Tuple[str, str]:
    """Captions sit below their figure: take the first one starting under the image."""
    if not captions:
        return "", ""
    if rect is not None:
        below = [c for c in captions if c[0] >= rect.y1 - 5]
        if below:
            return below[0][1], below[0][2]
        return captions[-1][1], captions[-1][2]
    return captions[0][1], captions[0][2]


def figure_number(pdf_path: str, page_index: int) -> str:
    """The first figure number captioned on the page ("Fig. 7." -> "7"), or ''."""
    import pymupdf
    caps = _captions(pymupdf.open(pdf_path)[page_index])
    return caps[0][1] if caps else ""


def error_type_from_caption(caption: str) -> str:
    """'sd' or 'sem' if the caption says which the error bars show, else ''."""
    t = caption.replace("\n", " ")
    sem = re.search(r"\bS\.?\s?E\.?\s?M\.?\b|standard error", t, re.IGNORECASE)
    sd = re.search(r"(?:±|\+/-|\bmean\s*\(?)\s*S\.?D\.?\b|standard deviation", t, re.IGNORECASE)
    if sem and not sd:
        return "sem"
    if sd and not sem:
        return "sd"
    return ""


def figures(pdf_path: str, pages: Optional[Iterable[int]] = None,
            render_dpi: int = 300) -> List[dict]:
    """
    Every figure-sized image in the PDF:
    [{page, figure, caption, error_type, image}] with image as BGR.
    """
    import pymupdf
    doc = pymupdf.open(pdf_path)
    out = []
    for i in (pages if pages is not None else range(len(doc))):
        page = doc[i]
        caps = _captions(page)
        found = False
        for info in page.get_images(full=True):
            xref, w, h = info[0], info[2], info[3]
            if w * h < MIN_FIGURE_PIXELS:
                continue
            try:
                rects = page.get_image_rects(xref)
            except Exception:
                rects = []
            fig, caption = _caption_for(rects[0] if rects else None, caps)
            try:
                img = _to_bgr(pymupdf.Pixmap(doc, xref))
            except Exception:
                continue
            out.append(dict(page=i, figure=fig, caption=caption,
                            error_type=error_type_from_caption(caption), image=img))
            found = True
        if not found and len(page.get_drawings()) > 200:   # vector figure
            fig, caption = _caption_for(None, caps)
            out.append(dict(page=i, figure=fig, caption=caption,
                            error_type=error_type_from_caption(caption),
                            image=_to_bgr(page.get_pixmap(dpi=render_dpi))))
    return out


def page_images(pdf_path: str, pages: Optional[Iterable[int]] = None,
                render_dpi: int = 300) -> List[Tuple[int, np.ndarray]]:
    """[(page_index, BGR image)] for every figure-sized image in the PDF."""
    return [(f["page"], f["image"]) for f in figures(pdf_path, pages, render_dpi)]
