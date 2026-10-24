"""Gunicorn settings for Docker and production.

Bind stays 0.0.0.0 so port mapping works. Default Gunicorn INFO logs print
http://0.0.0.0:<port>, which Docker Desktop turns into a link browsers refuse.
Keep loglevel at warning and print the URL people should actually open.
"""

import os
import sys

_port = os.environ.get("PORT", "2800")
bind = f"0.0.0.0:{_port}"
workers = 1
threads = 4
timeout = 300
loglevel = "warning"


def on_starting(server):
    print(
        f"Open http://127.0.0.1:{_port} in your browser. "
        f"In Docker Desktop set Host port to {_port}. "
        f"Do not open http://0.0.0.0:{_port}.",
        file=sys.stderr,
        flush=True,
    )
