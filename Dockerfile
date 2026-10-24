# Build: docker build -t yourdockerhub/autoextract:latest .
# Run:  docker run --rm -p 2800:2800 yourdockerhub/autoextract:latest
# Optional custom weights: -v /path/to/best.pt:/app/backend/models/best.pt:ro
# This image listens on 2800 (not 9000). Map host 2800 -> container 2800.

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

# Use PORT from the environment (default 2800). Long timeout: YOLO load and PDF extract can be slow on CPU.
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-2800} --workers 1 --threads 4 --timeout 300 wsgi:application"]
