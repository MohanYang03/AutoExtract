# AutoExtract

A web app that extracts data from bar charts in scientific papers. Drop in whole PDFs: every bar chart and dot plot is found, its y-axis calibrated from all tick labels, and each group's mean and error bar measured from the pixels. Results go to one Excel workbook laid out like a meta-analysis long-format sheet, ready to paste.

The backend runs on Flask; the frontend is a Vite + React app. The original YOLO-based single-image Bar Analyzer is still included.

## Quick Start

**Pull and run** (default YOLO weights are inside the image)

```bash
docker pull yanghenry/autoextract:latest
docker run --rm -p 9000:9000 yanghenry/autoextract:latest
```

Open **http://localhost:9000** and choose **Batch Extract**.

**Use your own weights** (optional, classic Bar Analyzer only): bind-mount over the default file path.

```bash
docker run --rm -p 9000:9000 \
  -v /absolute/path/on/host/best.pt:/app/backend/models/best.pt:ro \
  yanghenry/autoextract:latest
```

The image bundles the production-built frontend, the Flask API, and **`backend/models/best.pt`** from this repository so a volume is not required.

**Run without Docker**

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # Windows: .venv\Scripts\pip
cd frontend && npm install && npm run build && cd ..
FRONTEND_DIST=frontend/dist .venv/bin/python backend/app.py
```

## Batch Extract

1. Drop PDFs or figure images onto the page (several at once is fine).
2. Each panel appears next to its numbers. Orange lines mark means, blue lines mark error-bar caps; **red caps are flagged** for a quick check.
3. To correct a value, drag its line (or click it and use ↑/↓, Shift for larger steps). Values update instantly.
4. Set the error-bar type per panel (**SEM/SD**, defaulted from the figure caption) and the sample size **n**; the other of SD/SEM is computed.
5. **Copy rows** pastes straight into a long-format sheet; **Download Excel** gives one workbook for everything.

Options (numbers are always extracted):

| Option | Default | What it does |
|---|---|---|
| Bar names | on | Reads the category label under each bar into `treatment_group` |
| Panel letters & axis titles | on | Names each sheet, e.g. `Fig7B Struggling time(s)` |
| Match legend colours | off | Names bars from a colour legend when there are no labels (slower) |
| Error bars show | Auto | SEM or SD; Auto reads the caption ("mean ± SEM"), else SEM |

### Spreadsheet layout

Each panel gets a sheet whose first 12 columns are:

`mean, sd, age(week), sex(male), acupoint, treatment_group, sample_size, treatment_duration, depression_model_duration, paper_id, sem, mean+sem`

The measured error goes in `sem` (or `sd`, with the last column named `mean+sd`); the other is a live formula on `sample_size`. `paper_id` is filled from folder names like `(paper11) ...`. An **All bars** sheet stacks every panel with source/figure/panel columns, and **Summary** lists each panel with how many tick labels were used and how many bars need a check.

### Command line

```bash
python backend/batch_extract.py "path/to/papers" --out results.xlsx --review review_images/
# options: --no-names  --no-titles  --legend  --error-type sem|sd
```

`--review` writes an annotated copy of each figure so results can be eyeballed quickly.

### What it handles

Bar charts with or without outlines, white/coloured/dark fills, black or coloured error bars, scatter dots over bars, gridlines, boxed plots, axes that go below zero (bars hang down from zero), axes without tick marks, small low-resolution figures, multi-panel figures, and dot plots (mean line ± whiskers). Box plots, stacked bars, line graphs and micrographs are skipped.

### Accuracy

Measured against hand-digitized meta-analysis spreadsheets (21 papers, 98 sheets) and against synthetic charts with known values:

- Paper 11 (30 bars): mean error **0.23 %** of the axis range, all means within 1 %; caps 0.6 %.
- 21 papers: 95 of 98 sheets matched; median error **0.40 %** (means) and **0.32 %** (caps).
- 200 random synthetic charts: 97 % measured exactly right (every mean within 1 %), and every *unflagged* error bar within 1 %.
- A whole paper (27 panels) extracts in about 5 s on a laptop CPU.

The original two-label YOLO pipeline scored 17 % mean error on the same 30 bars.

## Tests and benchmarks

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q                 # synthetic accuracy + API tests
python tests/fuzz.py 200                  # randomised stress test
python tests/ui_check.py                  # headless browser check (app must be running)
python eval/run_eval.py --data "path/to/mice studies"   # vs hand-digitized sheets
python eval/sweep.py --data "path/to/mice studies"      # every paper in the folder
```

## API Overview

- POST `/api/extract`: multipart `file` (PDF or image), optional `options` (JSON: `bar_names`, `titles`, `legend_colors`), `error_type` (`sem`/`sd`) and `job`. Returns every panel found with its calibration, bars (`mean`, `mean_plus_err`, pixel positions, `uncertain` flag) and a figure image URL.
- POST `/api/export`: JSON `{figures: [...]}` (as returned, possibly edited). Returns the `.xlsx` workbook.
- POST `/api/bar_analyzer`: classic single-image YOLO analysis (multipart `file`).
- POST `/api/update_values`: JSON `{ "origin_value": number|string, "ymax_value": number|string }`

## Architecture

Backend (`backend/`)
- `figure_extractor.py`: geometry-based extraction (axes, tick calibration, bars, error bars, dot plots, labels)
- `pdf_utils.py`: figure images and captions from PDFs (SEM/SD from the caption)
- `exporter.py`: long-format workbook and review overlays
- `batch_extract.py`: command-line batch runner
- `extract_routes.py`: `/api/extract` and `/api/export`
- `app.py`: Flask app factory, routes registration, error handlers, entry point
- `routes.py`, `bar_graph_analyzer.py`, `model_manager.py`, `vision_utils.py`: classic YOLO Bar Analyzer (model loads on first use)
- `config.py`: Centralized settings (paths, ports, secrets, model, classes)

Frontend (`frontend/`)
- `src/components/BatchExtractor.jsx`, `PanelEditor.jsx`: Batch Extract page
- `src/components/BarAnalyzer.jsx`: classic interactive viewer (Fabric.js)
- Proxy config in `vite.config.js` to reach Flask during development

## Acknowledgements

- RapidOCR (text recognition) and PyMuPDF (PDF figures)
- Ultralytics YOLOv5 (object detection, classic analyzer)
- Flask (backend) and React + Vite (frontend)
- Fabric.js (interactive canvas)
