"""
Batch extraction API.

POST /api/extract   multipart `file` (PDF or image) -> every bar-chart panel found
POST /api/export    JSON {figures: [...]} (possibly edited)  -> .xlsx download

Unlike the single-image bar analyzer, these endpoints keep no per-user state on
the server: results go back to the browser, and edits come back with the export
request. Figure images are written under static/jobs/<job id>/.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import time
import uuid
from typing import Any, Dict, Tuple

import cv2
from flask import request, send_file, send_from_directory
from werkzeug.utils import secure_filename

from config import BASE_DIR
from exporter import PanelResult, build_workbook, paper_id_from_path
from figure_extractor import ExtractOptions, FigureExtractor, Panel

JOBS_DIR = os.path.join(BASE_DIR, "static", "jobs")
EXTRACT_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp"}

_extractor: FigureExtractor | None = None
_lock = threading.Lock()


JOB_MAX_AGE_S = 24 * 3600


def _clean_old_jobs() -> None:
    """Figure images are only needed while a result is being reviewed; drop day-old jobs."""
    if not os.path.isdir(JOBS_DIR):
        return
    cutoff = time.time() - JOB_MAX_AGE_S
    for name in os.listdir(JOBS_DIR):
        path = os.path.join(JOBS_DIR, name)
        try:
            if os.path.isdir(path) and os.path.getmtime(path) < cutoff:
                for f in os.listdir(path):
                    os.remove(os.path.join(path, f))
                os.rmdir(path)
        except OSError:
            pass


def _get_extractor() -> FigureExtractor:
    global _extractor
    with _lock:
        if _extractor is None:
            _extractor = FigureExtractor()
    return _extractor


def handle_extract() -> Tuple[Dict[str, Any], int]:
    from batch_extract import extract_file

    file = request.files.get("file")
    if not file or not file.filename:
        return {"error": "No file uploaded"}, 400
    name = secure_filename(file.filename)
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in EXTRACT_EXTENSIONS:
        return {"error": f"Unsupported file type: .{ext}"}, 400

    job = request.form.get("job") or uuid.uuid4().hex
    if not job.isalnum():
        return {"error": "Invalid job id"}, 400
    if not request.form.get("job"):
        _clean_old_jobs()
    job_dir = os.path.join(JOBS_DIR, job)
    os.makedirs(job_dir, exist_ok=True)

    try:
        options = ExtractOptions.from_dict(json.loads(request.form.get("options") or "{}"))
    except (ValueError, TypeError):
        return {"error": "Invalid options"}, 400
    error_type = request.form.get("error_type") or None
    if error_type not in (None, "sem", "sd"):
        return {"error": "error_type must be 'sem' or 'sd'"}, 400

    fd, tmp = tempfile.mkstemp(suffix="." + ext)
    os.close(fd)
    try:
        file.save(tmp)
        extractor = _get_extractor()
        t0 = time.time()
        with _lock:   # OCR sessions are shared; keep one extraction at a time
            found = extract_file(tmp, extractor, options, error_type)
        figures = []
        for fig, panels in found:
            img = fig["image"]
            fig_id = uuid.uuid4().hex[:10]
            # JPEG keeps big scans light to send; quality 92 is visually lossless for review
            cv2.imwrite(os.path.join(job_dir, f"{fig_id}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            figures.append({
                "id": fig_id,
                "source": file.filename,
                "page": fig["page"],
                "figure": fig["figure"],
                "error_type_from_caption": fig["error_type"],
                "paper_id": paper_id_from_path(file.filename),
                "image_url": f"/static/jobs/{job}/{fig_id}.jpg",
                "width": int(img.shape[1]),
                "height": int(img.shape[0]),
                "panels": [p.to_dict() for p in panels],
            })
        return {"job": job, "source": file.filename, "figures": figures,
                "seconds": round(time.time() - t0, 2)}, 200
    except Exception as e:
        return {"error": f"Extraction failed: {e}"}, 500
    finally:
        os.remove(tmp)


def handle_export():
    data = request.get_json(silent=True) or {}
    results = []
    try:
        for fig in data.get("figures", []):
            for pd in fig.get("panels", []):
                if pd.get("excluded"):
                    continue
                panel = Panel.from_dict(pd)
                for b in panel.bars:   # values must be numbers (they came from us, but were editable)
                    for v in (b.mean, b.mean_plus_err):
                        if v is not None and not isinstance(v, (int, float)):
                            raise ValueError("bar values must be numbers")
                    if b.n is not None:
                        b.n = int(b.n)
                if panel.error_type not in ("sem", "sd"):
                    panel.error_type = "sem"
                results.append(PanelResult(source=str(fig.get("source", "")), page=int(fig.get("page", -1)),
                                           figure=str(fig.get("figure", "")), panel=panel,
                                           paper_id=fig.get("paper_id")))
    except (TypeError, ValueError, AttributeError) as e:
        return {"error": f"Invalid export data: {e}"}, 400
    if not results:
        return {"error": "Nothing to export"}, 400
    buf = io.BytesIO()
    build_workbook(results).save(buf)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=data.get("filename") or "extracted.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def handle_job_file(job: str, filename: str):
    return send_from_directory(os.path.join(JOBS_DIR, secure_filename(job)), secure_filename(filename))
