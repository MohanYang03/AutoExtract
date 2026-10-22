# AutoExtract

A web app that extracts data from charts in scientific papers: bar charts, dot plots, box plots, violin plots and line plots. Drop in whole PDFs: every chart of the page's type is found, its axes calibrated from the tick labels, and each group's mean and error bar (or min, quartiles, median and max; or each line point and its error bar) measured from the pixels. Box and violin plots also get an estimated mean and SD for meta-analysis. Results go to one Excel workbook laid out like a meta-analysis long-format sheet, ready to paste.

Built on [AutoExtract](https://github.com/MohanYang03/AutoExtract) by Mohan Yang; its Bar, Box and Line Analyzer pages are now batch extractors (plus a new Violin Analyzer), and its YOLO-based single-image analysis remains available as an API endpoint. The backend runs on Flask; the frontend is a Vite + React app.

## Setup (once)

You need:
- **Python 3.10–3.12**: [python.org](https://www.python.org/downloads/). Python 3.13+ won't install the pinned PyTorch. On Windows, tick "Add python.exe to PATH" in the installer.
- **Node.js 20 or newer**: [nodejs.org](https://nodejs.org/) (the LTS version is fine)
- **Git**: [git-scm.com](https://git-scm.com/)

Then, in a terminal:

**Windows (PowerShell)**
```powershell
git clone https://github.com/gordonzyxu/AutoExtract.git
cd AutoExtract
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
cd frontend; npm install; npm run build; cd ..
```

**macOS / Linux**
```bash
git clone https://github.com/gordonzyxu/AutoExtract.git
cd AutoExtract
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cd frontend && npm install && npm run build && cd ..
```

The Python install downloads about 2 GB the first time (PyTorch, used only by the classic YOLO endpoint).

## Run

From the `AutoExtract` folder:

```powershell
.venv\Scripts\python backend\app.py      # Windows
.venv/bin/python backend/app.py          # macOS / Linux
```

Open **http://localhost:2800/bar_analyzer** in your browser. Keep the terminal open while you use it; press **Ctrl+C** to stop.

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
- **pip error mentioning "Long Path support" (Windows)**: the folder path is too long for PyTorch's files. Clone somewhere shorter (e.g. `C:\Code\AutoExtract`), delete `.venv`, and run the setup again.
- **Port 2800 already in use**: start on another port, e.g. PowerShell `$env:PORT=2801; .venv\Scripts\python backend\app.py`, then open http://localhost:2801/bar_analyzer.
- **`git clone` asks for access / "repository not found"**: the repository is private; ask the owner to add your GitHub account as a collaborator.

## Docker (optional)

Instead of the Python/Node setup, with Docker installed:

```bash
git clone https://github.com/gordonzyxu/AutoExtract.git
cd AutoExtract
docker build -t autoextract .
docker run --rm -p 2800:2800 autoextract
```

Open **http://localhost:2800/bar_analyzer**. The image includes the built page and the Bar Analyzer weights (`backend/models/best.pt`); to use your own weights add `-v /absolute/path/best.pt:/app/backend/models/best.pt:ro` to `docker run`.

## Using it

There is one page per chart type, and each takes whole papers, several at once:

- **Bar Analyzer** (`/bar_analyzer`): bar charts and dot plots (mean ± error bar).
- **Box Analyzer** (`/box_analyzer`): box plots.
- **Violin Analyzer** (`/violin_analyzer`): violin plots.
- **Line Analyzer** (`/line_analyzer`): line plots (one row per point of each series).

Papers mix chart types, so each page leaves the other kinds out: drop the same PDFs on each page whose charts a paper has. Results stay on each page while you switch between them.

### Bar Analyzer

1. Drop PDFs or figure images onto the page (several at once is fine).
2. Each panel appears next to its numbers. Orange lines mark means, blue lines mark error-bar caps; **red caps are flagged** for a quick check.
3. To correct a value, drag its line (or click it and use ↑/↓, Shift for larger steps). Values update instantly.
4. Set the error-bar type per panel (**SEM/SD**, defaulted from the figure caption) and the sample size **n**; the other of SD/SEM is computed.
5. **Copy rows** pastes straight into a long-format sheet; **Download Excel** gives one workbook for everything.

### Box Analyzer

Each box shows min, Q1, median, Q3 and max; drag any of those lines to correct it. Per panel, choose:

- **Whiskers**: *Min–max* (whiskers are the range), *1.5×IQR* (Tukey), *Percentile* or *Not stated*. Only min–max whiskers are used as the range; otherwise only the quartiles are used.
- **Mean**: Luo 2018 (recommended), Wan 2014 or Hozo 2005. **SD**: Shi 2020 (recommended), Wan 2014 or Hozo 2005.

Type **n** per box (or the same n for every box); the estimated mean and SD appear once n is known. A ⚠ marks skewed boxes, where normal-theory estimates may be biased.

### Violin Analyzer

Works like the Box Analyzer. Each violin gives its ends (min and max: Prism's "truncated" and ggplot's `trim = TRUE` violins end at the data range, the default in both) and the median and quartiles from the lines drawn inside it (Prism: dashed median, dotted quartiles) or from a narrow box plot drawn inside it (ggplot). When only the median is drawn, the estimate uses min, median and max (Luo 2018 / Wan 2014, scenario C1). Set **Violin ends** to *Not the range* when the tails run past the data.

### Line Analyzer

Each series is traced by its colour (several series of one colour are traced one after another). Its markers are the data points, read where the line segments on either side meet (so a marker partly covered by another series still reads true), each with its error bar; the x value comes from the x-axis labels. A marker hidden under another series' marker is read from its line and flagged. Name each series once (it becomes `treatment_group` for all its points); set SEM/SD and n as on the Bar Analyzer. Lines without markers are read at the x ticks.

Bar Analyzer options (numbers are always extracted; the Box Analyzer has the first two):

| Option | Default | What it does |
|---|---|---|
| Bar names | on | Reads the category label under each bar into `treatment_group` |
| Panel letters & axis titles | on | Names each sheet, e.g. `Fig7B Struggling time(s)` |
| Match legend colours | off | Names bars from a colour legend when there are no labels (slower) |
| Error bars show | Auto | SEM or SD; Auto reads the caption ("mean ± SEM"), else SEM |

### Spreadsheet layout

Each panel gets a sheet whose first 12 columns are:

`mean, sd, age(week), sex(male), acupoint, treatment_group, sample_size, treatment_duration, depression_model_duration, paper_id, sem, mean+sem`

The measured error goes in `sem` (or `sd`, with the last column named `mean+sd`); the other is a live formula on `sample_size`. `paper_id` is filled from folder names like `(paper11) ...`. An **All data** sheet stacks every panel with source/figure/panel columns, and **Summary** lists each panel with how many tick labels were used and how many values need a check.

Box- and violin-plot sheets fill `mean` and `sd` with live formulas that estimate them from the min/Q1/median/Q3/max columns (written to the right) and `sample_size`, using the panel's chosen methods; `sem` is `sd/√n`. They fill in as soon as n is typed. Line-plot sheets have one row per point, with `series` and `x` columns after the 12.

### Command line

```bash
python backend/batch_extract.py "path/to/papers" --out results.xlsx --review review_images/
python backend/batch_extract.py "path/to/papers" --type box --out box_plots.xlsx   # or --type violin / line
# options: --no-names  --no-titles  --legend  --error-type sem|sd
```

`--review` writes an annotated copy of each figure so results can be eyeballed quickly.

### What it handles

- Bar charts with or without outlines, white/coloured/dark fills, black or coloured error bars, scatter dots over bars, gridlines, boxed plots, axes that go below zero (bars hang down from zero), axes without tick marks, small low-resolution figures, multi-panel figures, and dot plots (mean line ± whiskers).
- Box plots: white, filled or dark boxes, with or without caps, dashed whiskers, outliers, mean markers, log axes.
- Violin plots: Prism lines, ggplot inner boxes, median-only, white/coloured/dark fills, scatter dots over the violin.
- Line plots: several series (by colour, or one colour with different markers), filled or hollow circles, squares, triangles and diamonds, error bars, legends inside the plot, frames and reference lines, numeric or text x labels.
- Not handled: stacked bars, horizontal charts, scatter plots, micrographs.

### Accuracy

Measured against hand-digitized meta-analysis spreadsheets (21 papers, 98 sheets) and against synthetic charts with known values:

- Paper 11 (30 bars): mean error **0.23 %** of the axis range, all means within 1 %; caps 0.6 %.
- 21 papers: 95 of 98 sheets matched; median error **0.40 %** (means) and **0.32 %** (caps).
- 200 random synthetic charts: 97 % measured exactly right (every mean within 1 %), and every *unflagged* error bar within 1 %.
- A whole paper (27 panels) extracts in about 5 s on a laptop CPU.

- Random synthetic charts with exact values (share with every value within 1 % of the axis): bar charts 97–99 %, box plots 92–95 %, violin plots 71–75 %, line plots 60–66 % (typical point error 0.09 %).
- Real charts from PubMed Central ([CHART-Info 2022](https://chartinfo.github.io/) test set, hand-annotated), share within 1 % of the axis: bar values 78 %; box Q1/median/Q3 88–91 % (whisker ends 77 %); line points 59 %. Many misses are charts whose axis labels cannot be read at all.
- Each page on the other chart types (CHART-Info): the box and violin pages report nothing on 280 bar charts; at most 7 % of other charts give a stray panel on any page.

The original two-label YOLO pipeline scored 17 % mean error on the same 30 bars.

## Tests and benchmarks

```bash
.venv/bin/python -m pip install -r requirements-dev.txt   # Windows: .venv\Scripts\python
# run the commands below with the same .venv python
python -m pytest tests -q                 # synthetic accuracy + API tests
python tests/fuzz.py 200                  # randomised stress test (--box / --violin / --line for those)
python tests/ui_check.py                  # headless browser check (app must be running)
python eval/run_eval.py --data "path/to/mice studies"   # vs hand-digitized sheets
python eval/sweep.py --data "path/to/mice studies"      # every paper in the folder
python eval/chartinfo_eval.py --data path/to/ICPR2022_CHARTINFO_UB_UNITEC_PMC_TEST_v2.1 [--type bar|box|line] [--cross]
                                          # real-chart benchmark (download separately; CC BY-NC-SA, don't commit)
```

## API Overview

- POST `/api/extract`: multipart `file` (PDF or image), optional `options` (JSON: `chart_type` `bar`|`box`|`violin`|`line` (default `bar`), `bar_names`, `titles`, `legend_colors`), `error_type` (`sem`/`sd`) and `job`. Returns every panel found with its calibration, `kind` (`bar`, `dot`, `box`, `violin` or `line`), bars (`mean`, `mean_plus_err`, pixel positions, `uncertain` flag), boxes (`lo`, `q1`, `median`, `q3`, `hi`, outliers, pixel positions; violins too) or series (`color`, `label`, `trace`, `points` with `x`, `mean`, `mean_plus_err`) and a figure image URL.
- POST `/api/export`: JSON `{figures: [...]}` (as returned, possibly edited). Returns the `.xlsx` workbook.
- POST `/api/bar_analyzer`: classic single-image YOLO analysis (multipart `file`).
- POST `/api/update_values`: JSON `{ "origin_value": number|string, "ymax_value": number|string }`

## Architecture

Backend (`backend/`)
- `figure_extractor.py`: geometry-based extraction (axes, tick calibration, bars, error bars, dot plots, labels)
- `box_plots.py`: box-plot detection (quartiles, median, whiskers, outliers, mean markers)
- `violin_plots.py`: violin-plot detection (ends, quartile/median lines or inner box)
- `line_plots.py`: line-plot tracing (series by colour, markers, error bars)
- `estimators.py`: mean/SD from box/violin statistics (Luo, Wan, Shi, Hozo; scenarios C1–C3)
- `pdf_utils.py`: figure images and captions from PDFs (SEM/SD from the caption)
- `exporter.py`: long-format workbook and review overlays
- `batch_extract.py`: command-line batch runner
- `extract_routes.py`: `/api/extract` and `/api/export`
- `app.py`: Flask app factory, routes registration, error handlers, entry point
- `routes.py`, `bar_graph_analyzer.py`, `model_manager.py`, `vision_utils.py`: classic YOLO Bar Analyzer (model loads on first use)
- `config.py`: Centralized settings (paths, ports, secrets, model, classes)

Frontend (`frontend/`)
- `src/components/BatchExtractor.jsx`: the Bar, Box, Violin and Line Analyzer pages (one component, a mode each)
- `PanelEditor.jsx`, `BoxPanelEditor.jsx`, `LinePanelEditor.jsx`: one bar/dot, box/violin or line panel with its table
- Proxy config in `vite.config.js` to reach Flask during development

## Acknowledgements

- RapidOCR (text recognition) and PyMuPDF (PDF figures)
- Ultralytics YOLOv5 (object detection, classic analyzer)
- Flask (backend) and React + Vite (frontend)
- Fabric.js (interactive canvas)
