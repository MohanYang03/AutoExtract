# Build: docker build -t yanghenry/autoextract:latest .
# Run:  docker run --rm -p 2800:2800 yanghenry/autoextract:latest
# Open http://127.0.0.1:2800  (not http://0.0.0.0:2800)
# EXPOSE does not publish the port; -p 2800:2800 is required on Mac and Windows.

FROM node:22-bookworm-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt "gunicorn>=22.0.0"

# Pre-cache torch.hub YOLOv5 checkout
RUN python -c "import torch; torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=False, trust_repo=True)"

COPY backend/ ./backend/
# Fail fast if default weights are missing from build context (see backend/models/best.pt in repo)
RUN test -f /app/backend/models/best.pt
COPY --from=frontend-build /app/frontend/dist ./frontend_dist

ENV PYTHONPATH=/app/backend \
    FRONTEND_DIST=/app/frontend_dist \
    DEBUG=false \
    PORT=2800

WORKDIR /app/backend
EXPOSE 2800

# gunicorn.conf.py binds 0.0.0.0:$PORT and prints http://127.0.0.1 (not 0.0.0.0).
CMD ["gunicorn", "--config", "gunicorn.conf.py", "wsgi:application"]
