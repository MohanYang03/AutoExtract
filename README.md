# AutoDigitizer

A web app that extracts data from bar charts in scientific papers. Drop in whole PDFs: every bar chart and dot plot is found, its y-axis calibrated from all tick labels, and each group's mean and error bar measured from the pixels. Results go to one Excel workbook laid out like a meta-analysis long-format sheet, ready to paste.

Built on [AutoExtract](https://github.com/MohanYang03/AutoExtract) by Mohan Yang; its YOLO-based single-image Bar Analyzer is still included. The backend runs on Flask; the frontend is a Vite + React app.

## Setup (once)

You need:
- **Python 3.10–3.12**: [python.org](https://www.python.org/downloads/). Python 3.13+ won't install the pinned PyTorch. On Windows, tick "Add python.exe to PATH" in the installer.
- **Node.js 20 or newer**: [nodejs.org](https://nodejs.org/) (the LTS version is fine)
- **Git**: [git-scm.com](https://git-scm.com/)

Then, in a terminal:

**Windows (PowerShell)**
```powershell
git clone https://github.com/gordonzyxu/AutoDigitizer.git
cd AutoDigitizer
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
cd frontend; npm install; npm run build; cd ..
```

**macOS / Linux**
```bash
git clone https://github.com/gordonzyxu/AutoDigitizer.git
cd AutoDigitizer
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cd frontend && npm install && npm run build && cd ..
```

The Python install downloads about 2 GB the first time (PyTorch, used only by the classic Bar Analyzer).

## Run

From the `AutoDigitizer` folder:

```powershell
.venv\Scripts\python backend\app.py      # Windows
.venv/bin/python backend/app.py          # macOS / Linux
```

Open **http://localhost:2800/extract** in your browser. Keep the terminal open while you use it; press **Ctrl+C** to stop.

## Update to the latest version

```powershell
git pull
.venv\Scripts\python -m pip install -r requirements.txt   # macOS/Linux: .venv/bin/python
cd frontend; npm install; npm run build; cd ..
```

Then run it as above.

## Troubleshooting

- **Page says "Resource not found" / 404**: the web page isn't built yet. Run `cd frontend; npm install; npm run build; cd ..` and restart.
- **`python` not found (Windows)**: reinstall Python with "Add python.exe to PATH" ticked, or use `py -3.12 -m venv .venv`.
- **pip can't find `torch==2.3.0`**: your Python is too new; install Python 3.12 and recreate `.venv`.
- **pip error mentioning "Long Path support" (Windows)**: the folder path is too long for PyTorch's files. Clone somewhere shorter (e.g. `C:\Code\AutoDigitizer`), delete `.venv`, and run the setup again.
- **Port 2800 already in use**: start on another port, e.g. PowerShell `$env:PORT=2801; .venv\Scripts\python backend\app.py`, then open http://localhost:2801/extract.
- **`git clone` asks for access / "repository not found"**: the repository is private; ask the owner to add your GitHub account as a collaborator.

## Docker (optional)

Instead of the Python/Node setup, with Docker installed:

```bash
git clone https://github.com/gordonzyxu/AutoDigitizer.git
cd AutoDigitizer
docker build -t autodigitizer .
docker run --rm -p 2800:2800 autodigitizer
```

Open **http://localhost:2800/extract**. The image includes the built page and the Bar Analyzer weights (`backend/models/best.pt`); to use your own weights add `-v /absolute/path/best.pt:/app/backend/models/best.pt:ro` to `docker run`.

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
.venv/bin/python -m pip install -r requirements-dev.txt   # Windows: .venv\Scripts\python
# run the commands below with the same .venv python
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
