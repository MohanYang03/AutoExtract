/**
 * BatchExtractor Component
 *
 * Drop in papers (PDF) or figure images, several at once; every chart of the
 * page's kind is measured automatically: bar charts and dot plots on the Bar
 * Analyzer page, box plots on the Box Analyzer page. Review and adjust each
 * panel, then download one workbook in the meta-analysis long format or copy
 * rows to the clipboard.
 */

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { downloadWorkbook, extractFile } from '../services/extractService';
import { boxWithValues, panelItems, panelToTSV, pointWithValues, withValues } from '../utils/panelMath';
import BoxPanelEditor from './BoxPanelEditor';
import LinePanelEditor from './LinePanelEditor';
import PanelEditor from './PanelEditor';

const ACCEPT = '.pdf,.png,.jpg,.jpeg,.tif,.tiff,.bmp';

// What each page looks for and offers. Numbers are always extracted; the
// options are the optional extras.
const MODES = {
    bar: {
        title: 'Bar Analyzer',
        intro: 'Drop papers (PDF) or figure images, as many as you like. Every bar chart and dot plot is measured '
            + 'automatically: the axis is calibrated from all tick labels, and bars and error bars are read from the pixels.',
        noun: 'bar charts or dot plots',
        options: [
            { key: 'bar_names', label: 'Bar names', hint: 'Read category labels under each bar (fast)' },
            { key: 'titles', label: 'Panel letters & axis titles', hint: 'Name each sheet, e.g. "Fig 7B Struggling time" (fast)' },
            { key: 'legend_colors', label: 'Match legend colours', hint: 'Name bars from a colour legend when there are no labels (slower, ~2 s per figure)' },
        ],
        defaults: { bar_names: true, titles: true, legend_colors: false },
        errorType: true,
        file: 'bar_charts.xlsx',
    },
    box: {
        title: 'Box Analyzer',
        intro: 'Drop papers (PDF) or figure images, as many as you like. Every box plot is measured automatically '
            + '(min, Q1, median, Q3, max), and the mean and SD are estimated for meta-analysis once you enter n.',
        noun: 'box plots',
        options: [
            { key: 'bar_names', label: 'Group names', hint: 'Read the category label under each box (fast)' },
            { key: 'titles', label: 'Panel letters & axis titles', hint: 'Name each sheet, e.g. "Fig 2B Chao value" (fast)' },
        ],
        defaults: { bar_names: true, titles: true },
        errorType: false,
        file: 'box_plots.xlsx',
    },
    violin: {
        title: 'Violin Analyzer',
        intro: 'Drop papers (PDF) or figure images, as many as you like. Every violin plot is measured automatically '
            + '(ends, quartile and median lines, or an inner box plot), and the mean and SD are estimated once you enter n.',
        noun: 'violin plots',
        options: [
            { key: 'bar_names', label: 'Group names', hint: 'Read the category label under each violin (fast)' },
            { key: 'titles', label: 'Panel letters & axis titles', hint: 'Name each sheet (fast)' },
        ],
        defaults: { bar_names: true, titles: true },
        errorType: false,
        file: 'violin_plots.xlsx',
    },
    line: {
        title: 'Line Analyzer',
        intro: 'Drop papers (PDF) or figure images, as many as you like. Every line plot is traced series by series: '
            + 'each point (marker) with its error bar, and the x value read from the x axis.',
        noun: 'line plots',
        options: [
            { key: 'titles', label: 'Panel letters & axis titles', hint: 'Name each sheet (fast)' },
        ],
        defaults: { titles: true },
        errorType: true,
        file: 'line_plots.xlsx',
    },
};

const ITEM_NOUN = { bar: 'bars', box: 'boxes', violin: 'violins', line: 'points' };
const EDITORS = { box: BoxPanelEditor, violin: BoxPanelEditor, line: LinePanelEditor };

const settingsKey = (mode) => `analyzer.${mode}.settings.v1`;

const loadSettings = (mode) => {
    const defaults = { options: MODES[mode].defaults, errorType: '' };   // '' = from each caption
    try {
        const saved = JSON.parse(localStorage.getItem(settingsKey(mode)) || 'null');
        if (saved && saved.options) {
            return { ...defaults, ...saved, options: { ...defaults.options, ...saved.options } };
        }
    } catch {
        // storage unavailable or corrupt: fall back to defaults
    }
    return defaults;
};

const BatchExtractor = ({ mode = 'bar' }) => {
    const cfg = MODES[mode];
    const [figures, setFigures] = useState([]);
    const [queue, setQueue] = useState({ done: 0, total: 0, current: '' });
    const [errors, setErrors] = useState([]);
    const [dragActive, setDragActive] = useState(false);
    const [job, setJob] = useState(null);
    const [exporting, setExporting] = useState(false);
    const [settings, setSettings] = useState(() => loadSettings(mode));
    const [flaggedOnly, setFlaggedOnly] = useState(false);
    const [copiedAll, setCopiedAll] = useState(false);
    const [seconds, setSeconds] = useState(0);

    useEffect(() => {
        try {
            localStorage.setItem(settingsKey(mode), JSON.stringify(settings));
        } catch {
            // private mode etc.: settings just won't persist
        }
    }, [mode, settings]);

    const busy = queue.done < queue.total;

    const processFiles = useCallback(async (fileList) => {
        const files = Array.from(fileList || []);
        if (!files.length) return;
        setQueue((q) => ({ done: q.done, total: q.total + files.length, current: files[0].name }));
        let jobId = job;
        // One file per request keeps each request short and shows progress as we go
        for (const file of files) {
            setQueue((q) => ({ ...q, current: file.name }));
            try {
                const data = await extractFile(file, {
                    job: jobId,
                    options: { ...settings.options, chart_type: mode },
                    errorType: cfg.errorType ? settings.errorType : '',
                });
                jobId = data.job;
                setJob(data.job);
                setSeconds((s) => s + (data.seconds || 0));
                if (!data.figures.length) {
                    setErrors((e) => [...e, `${file.name}: no ${cfg.noun} found`]);
                }
                setFigures((prev) => [...prev, ...data.figures]);
            } catch (err) {
                setErrors((e) => [...e, `${file.name}: ${err.message}`]);
            }
            setQueue((q) => ({ ...q, done: q.done + 1 }));
        }
    }, [cfg, job, mode, settings]);

    const updatePanel = useCallback((figId, panelIdx, change) => {
        setFigures((prev) => prev.map((fig) => {
            if (fig.id !== figId) return fig;
            const panels = fig.panels.map((p, i) => {
                if (i !== panelIdx) return p;
                if (change.type === 'panel') return { ...p, ...change.patch };
                if (change.type === 'series' || change.type === 'point') {
                    const series = p.series.map((sr, k) => {
                        if (k !== change.series) return sr;
                        if (change.type === 'series') return { ...sr, ...change.patch };
                        const points = sr.points.map((pt, j) => {
                            if (j !== change.index) return pt;
                            const next = { ...pt, ...change.patch };
                            const moved = Object.keys(change.patch).some((key) => key.endsWith('_px'));
                            if (moved) next.uncertain = false;
                            return moved ? pointWithValues(p, next) : next;
                        });
                        return { ...sr, points };
                    });
                    return { ...p, series };
                }
                if (change.type === 'box') {
                    const boxes = p.boxes.map((b, j) => {
                        if (j !== change.index) return b;
                        const next = { ...b, ...change.patch };
                        const moved = Object.keys(change.patch).some((k) => k.endsWith('_px'));
                        if (moved) next.uncertain = false;
                        return moved ? boxWithValues(p, next) : next;
                    });
                    return { ...p, boxes };
                }
                const bars = p.bars.map((b, j) => {
                    if (j !== change.index) return b;
                    const next = { ...b, ...change.patch };
                    // a manually placed cap is, by definition, checked
                    if ('cap_px' in change.patch) next.uncertain = false;
                    return 'top_px' in change.patch || 'cap_px' in change.patch ? withValues(p, next) : next;
                });
                return { ...p, bars };
            });
            return { ...fig, panels };
        }));
    }, []);

    const onDrop = (e) => {
        e.preventDefault();
        setDragActive(false);
        processFiles(e.dataTransfer.files);
    };

    const exportAll = async () => {
        setExporting(true);
        try {
            await downloadWorkbook(figures, cfg.file);
        } catch (err) {
            setErrors((e) => [...e, err.message]);
        } finally {
            setExporting(false);
        }
    };

    const included = useMemo(
        () => figures.flatMap((f) => f.panels.filter((p) => !p.excluded).map((p) => ({ f, p }))),
        [figures],
    );
    const nItems = included.reduce((n, { p }) => n + panelItems(p).length, 0);
    const nFlagged = included.reduce((n, { p }) => n + panelItems(p).filter((b) => b.uncertain).length, 0);

    const copyAll = async () => {
        const text = included.map(({ f, p }) => panelToTSV(p, f.paper_id)).join('\n');
        try {
            await navigator.clipboard.writeText(text);
            setCopiedAll(true);
            setTimeout(() => setCopiedAll(false), 1500);
        } catch (err) {
            setErrors((e) => [...e, `Copy failed: ${err.message}`]);
        }
    };

    const clearAll = () => {
        setFigures([]);
        setErrors([]);
        setSeconds(0);
        setQueue({ done: 0, total: 0, current: '' });
    };

    // Group figures by source file for display
    const bySource = figures.reduce((acc, fig) => {
        (acc[fig.source] = acc[fig.source] || []).push(fig);
        return acc;
    }, {});

    const setOption = (key, value) => setSettings((s) => ({ ...s, options: { ...s.options, [key]: value } }));

    return (
        <div className="container-fluid pb-5 batch-extract" id={`page-${mode}`}>
            <div className="mb-3">
                <h1 className="mb-1">{cfg.title}</h1>
                <p className="text-muted mb-0">{cfg.intro}</p>
            </div>

            <div className="row g-3 mb-3">
                <div className="col-lg-7">
                    <div
                        className={`drop-zone rounded p-4 text-center h-100 ${dragActive ? 'active' : ''}`}
                        onDragEnter={(e) => { e.preventDefault(); setDragActive(true); }}
                        onDragOver={(e) => e.preventDefault()}
                        onDragLeave={() => setDragActive(false)}
                        onDrop={onDrop}
                    >
                        <i className="bi bi-files text-muted fs-1"></i>
                        <p className="mb-1">
                            Drag and drop PDFs or images here, or{' '}
                            <label htmlFor={`batch-input-${mode}`} className="text-primary fw-bold" style={{ cursor: 'pointer' }}>browse files</label>
                        </p>
                        <input id={`batch-input-${mode}`} type="file" multiple accept={ACCEPT} className="d-none"
                            onChange={(e) => { processFiles(e.target.files); e.target.value = ''; }} />
                        <small className="text-muted">PDF, PNG, JPG, TIFF. Several files at once is fine.</small>
                        {busy && (
                            <div className="mt-3">
                                <div className="progress mb-1" style={{ height: 6 }}>
                                    <div className="progress-bar" style={{ width: `${(queue.done / queue.total) * 100}%` }}></div>
                                </div>
                                <small className="text-muted">
                                    <span className="spinner-border spinner-border-sm me-2"></span>
                                    Extracting {queue.current} ({queue.done + 1} of {queue.total})
                                </small>
                            </div>
                        )}
                    </div>
                </div>
                <div className="col-lg-5">
                    <div className="card border-0 shadow-sm h-100">
                        <div className="card-body">
                            <h6 className="mb-2">Options</h6>
                            <div className="small text-muted mb-2">
                                {mode === 'box' || mode === 'violin' ? 'Min, quartiles, median and max' : 'Means and error bars'} are always extracted. Extras:
                            </div>
                            {cfg.options.map(({ key, label, hint }) => (
                                <div className="form-check mb-1" key={key}>
                                    <input className="form-check-input" type="checkbox" id={`opt-${mode}-${key}`}
                                        checked={settings.options[key]} onChange={(e) => setOption(key, e.target.checked)} />
                                    <label className="form-check-label" htmlFor={`opt-${mode}-${key}`}>
                                        {label} <span className="text-muted small">— {hint}</span>
                                    </label>
                                </div>
                            ))}
                            {cfg.errorType && (
                            <div className="d-flex align-items-center gap-2 mt-2">
                                <label htmlFor="opt-error" className="small mb-0">Error bars show</label>
                                <select id="opt-error" className="form-select form-select-sm" style={{ width: 'auto' }}
                                    value={settings.errorType}
                                    onChange={(e) => setSettings((s) => ({ ...s, errorType: e.target.value }))}>
                                    <option value="">Auto (from caption, else SEM)</option>
                                    <option value="sem">SEM</option>
                                    <option value="sd">SD</option>
                                </select>
                            </div>
                            )}
                            <div className="small text-muted mt-2">
                                {mode === 'box' || mode === 'violin'
                                    ? 'Options apply to files added next. Whisker type, estimation methods and n are set per panel.'
                                    : 'Options apply to files added next; SEM/SD can also be switched per panel.'}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {figures.length > 0 && (
                <div className="results-toolbar d-flex flex-wrap align-items-center gap-2 mb-3 p-2 rounded shadow-sm">
                    <span className="small">
                        <strong>{Object.keys(bySource).length}</strong> files · <strong>{included.length}</strong> panels ·{' '}
                        <strong>{nItems}</strong> {ITEM_NOUN[mode]}
                        {nFlagged > 0 && <> · <span className="text-warning-emphasis fw-bold">{nFlagged} to check</span></>}
                        {seconds > 0 && <span className="text-muted"> · {seconds.toFixed(1)} s</span>}
                    </span>
                    <div className="form-check form-switch ms-2 mb-0">
                        <input className="form-check-input" type="checkbox" id={`flagged-only-${mode}`}
                            checked={flaggedOnly} onChange={(e) => setFlaggedOnly(e.target.checked)} />
                        <label className="form-check-label small" htmlFor={`flagged-only-${mode}`}>Only panels to check</label>
                    </div>
                    <div className="ms-auto d-flex gap-2">
                        <button className="btn btn-sm btn-outline-secondary" onClick={clearAll} disabled={busy}>Clear</button>
                        <button className="btn btn-sm btn-outline-primary" onClick={copyAll} disabled={!included.length}>
                            <i className="bi bi-clipboard me-1"></i>{copiedAll ? 'Copied' : 'Copy all rows'}
                        </button>
                        <button className="btn btn-sm btn-primary" onClick={exportAll} disabled={exporting || busy || !included.length}>
                            <i className="bi bi-file-earmark-excel me-1"></i>{exporting ? 'Exporting...' : 'Download Excel'}
                        </button>
                    </div>
                </div>
            )}

            {errors.length > 0 && (
                <div className="alert alert-warning small d-flex">
                    <div className="flex-grow-1">{errors.map((e, i) => <div key={i}>{e}</div>)}</div>
                    <button type="button" className="btn-close ms-2" aria-label="Dismiss" onClick={() => setErrors([])}></button>
                </div>
            )}

            {Object.entries(bySource).map(([source, figs]) => (
                <div key={source} className="mb-4">
                    <h5 className="mb-2"><i className="bi bi-file-earmark-pdf me-2"></i>{source}</h5>
                    {figs.map((fig) => fig.panels.map((panel, pi) => (
                        (!flaggedOnly || panelItems(panel).some((b) => b.uncertain)) && React.createElement(
                            EDITORS[panel.kind] || PanelEditor, {
                                key: `${fig.id}-${pi}`,
                                figure: fig,
                                panel,
                                onChange: (change) => updatePanel(fig.id, pi, change),
                            })
                    )))}
                </div>
            ))}
        </div>
    );
};

export default BatchExtractor;
