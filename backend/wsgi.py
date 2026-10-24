"""WSGI entrypoint for production servers (e.g. gunicorn)."""

import logging
import sys

from app import create_app

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
_log = logging.getLogger("autoextract.wsgi")

application = create_app()

# The classic Bar Analyzer's YOLO model loads lazily on its first request
# (see model_manager); Batch Extract does not need it.
_log.info("AutoExtract ready")
