/**
 * BatchExtractor Component
 *
 * Drop in papers (PDF) or figure images; every bar chart and dot plot is
 * measured automatically. Review and adjust each panel, then download one
 * workbook in the meta-analysis long format or copy rows to the clipboard.
 */

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { downloadWorkbook, extractFile } from '../services/extractService';
import { panelToTSV, withValues } from '../utils/panelMath';
import PanelEditor from './PanelEditor';

const ACCEPT = '.pdf,.png,.jpg,.jpeg,.tif,.tiff,.bmp';
const SETTINGS_KEY = 'batchExtract.settings.v1';

// Numbers are always extracted; these are the optional extras.
const OPTION_INFO = [
    { key: 'bar_names', label: 'Bar names', hint: 'Read category labels under each bar (fast)' },
    { key: 'titles', label: 'Panel letters & axis titles', hint: 'Name each sheet, e.g. "Fig 7B Struggling time" (fast)' },
    { key: 'legend_colors', label: 'Match legend colours', hint: 'Name bars from a colour legend when there are no labels (slower, ~2 s per figure)' },
];
const DEFAULT_SETTINGS = {
    options: { bar_names: true, titles: true, legend_colors: false },
    errorType: '',   // '' = read from each figure caption (SEM if not stated)
};

const loadSettings = () => {
    try {
        const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) || 'null');
        if (saved && saved.options) {
            return { ...DEFAULT_SETTINGS, ...saved, options: { ...DEFAULT_SETTINGS.options, ...saved.options } };
        }
    } catch {
        // storage unavailable or corrupt: fall back to defaults
    }
    return DEFAULT_SETTINGS;
};

const BatchExtractor = () => {
    const [figures, setFigures] = useState([]);
    const [queue, setQueue] = useState({ done: 0, total: 0, current: '' });
    const [errors, setErrors] = useState([]);
    const [dragActive, setDragActive] = useState(false);
    const [job, setJob] = useState(null);
    const [exporting, setExporting] = useState(false);
    const [settings, setSettings] = useState(loadSettings);
    const [flaggedOnly, setFlaggedOnly] = useState(false);
    const [copiedAll, setCopiedAll] = useState(false);
    const [seconds, setSeconds] = useState(0);

    useEffect(() => {
        try {
            localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
        } catch {
            // private mode etc.: settings just won't persist
        }
    }, [settings]);

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
                const data = await extractFile(file, { job: jobId, options: settings.options, errorType: settings.errorType });
                jobId = data.job;
                setJob(data.job);
                setSeconds((s) => s + (data.seconds || 0));
                if (!data.figures.length) {
                    setErrors((e) => [...e, `${file.name}: no bar charts or dot plots found`]);
                }
                setFigures((prev) => [...prev, ...data.figures]);
            } catch (err) {
                setErrors((e) => [...e, `${file.name}: ${err.message}`]);
            }
            setQueue((q) => ({ ...q, done: q.done + 1 }));
        }
    }, [job, settings]);

    const updatePanel = useCallback((figId, panelIdx, change) => {
        setFigures((prev) => prev.map((fig) => {
            if (fig.id !== figId) return fig;
            const panels = fig.panels.map((p, i) => {
                if (i !== panelIdx) return p;
                if (change.type === 'panel') return { ...p, ...change.patch };
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
            await downloadWorkbook(figures, 'extracted_figures.xlsx');
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
    const nBars = included.reduce((n, { p }) => n + p.bars.length, 0);
    const nFlagged = included.reduce((n, { p }) => n + p.bars.filter((b) => b.uncertain).length, 0);

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
        <div className="container-fluid pb-5 batch-extract">
            <div className="mb-3">
                <h1 className="mb-1">Batch Extract</h1>
                <p className="text-muted mb-0">
                    Drop papers (PDF) or figure images. Every bar chart and dot plot is measured automatically:
                    the axis is calibrated from all tick labels, and bars and error bars are read from the pixels.
                </p>
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
                            <label htmlFor="batch-input" className="text-primary fw-bold" style={{ cursor: 'pointer' }}>browse files</label>
                        </p>
                        <input id="batch-input" type="file" multiple accept={ACCEPT} className="d-none"
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
                            <div className="small text-muted mb-2">Means and error bars are always extracted. Extras:</div>
                            {OPTION_INFO.map(({ key, label, hint }) => (
                                <div className="form-check mb-1" key={key}>
                                    <input className="form-check-input" type="checkbox" id={`opt-${key}`}
                                        checked={settings.options[key]} onChange={(e) => setOption(key, e.target.checked)} />
                                    <label className="form-check-label" htmlFor={`opt-${key}`}>
                                        {label} <span className="text-muted small">— {hint}</span>
                                    </label>
                                </div>
                            ))}
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
                            <div className="small text-muted mt-2">Options apply to files added next; SEM/SD can also be switched per panel.</div>
                        </div>
                    </div>
                </div>
            </div>

            {figures.length > 0 && (
                <div className="results-toolbar d-flex flex-wrap align-items-center gap-2 mb-3 p-2 rounded shadow-sm">
                    <span className="small">
                        <strong>{Object.keys(bySource).length}</strong> files · <strong>{included.length}</strong> panels ·{' '}
                        <strong>{nBars}</strong> bars
                        {nFlagged > 0 && <> · <span className="text-warning-emphasis fw-bold">{nFlagged} to check</span></>}
                        {seconds > 0 && <span className="text-muted"> · {seconds.toFixed(1)} s</span>}
                    </span>
                    <div className="form-check form-switch ms-2 mb-0">
                        <input className="form-check-input" type="checkbox" id="flagged-only"
                            checked={flaggedOnly} onChange={(e) => setFlaggedOnly(e.target.checked)} />
                        <label className="form-check-label small" htmlFor="flagged-only">Only panels to check</label>
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
                        (!flaggedOnly || panel.bars.some((b) => b.uncertain)) && (
                            <PanelEditor
                                key={`${fig.id}-${pi}`}
                                figure={fig}
                                panel={panel}
                                onChange={(change) => updatePanel(fig.id, pi, change)}
                            />
                        )
                    )))}
                </div>
            ))}
        </div>
    );
};

export default BatchExtractor;
