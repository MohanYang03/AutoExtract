/**
 * BatchExtractor Component
 *
 * Drop in papers (PDF) or figure images; every bar-chart panel is extracted
 * automatically. Review and adjust each panel, then download one workbook in
 * the meta-analysis long format or copy a panel's rows to the clipboard.
 */

import React, { useCallback, useState } from 'react';
import { downloadWorkbook, extractFile } from '../services/extractService';
import { withValues } from '../utils/panelMath';
import PanelEditor from './PanelEditor';

const ACCEPT = '.pdf,.png,.jpg,.jpeg,.tif,.tiff,.bmp';

const BatchExtractor = () => {
    const [figures, setFigures] = useState([]);
    const [queue, setQueue] = useState({ done: 0, total: 0, current: '' });
    const [errors, setErrors] = useState([]);
    const [dragActive, setDragActive] = useState(false);
    const [job, setJob] = useState(null);
    const [exporting, setExporting] = useState(false);

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
                const data = await extractFile(file, jobId);
                jobId = data.job;
                setJob(data.job);
                if (!data.figures.length) {
                    setErrors((e) => [...e, `${file.name}: no bar charts found`]);
                }
                setFigures((prev) => [...prev, ...data.figures]);
            } catch (err) {
                setErrors((e) => [...e, `${file.name}: ${err.message}`]);
            }
            setQueue((q) => ({ ...q, done: q.done + 1 }));
        }
    }, [job]);

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

    const nPanels = figures.reduce((n, f) => n + f.panels.filter((p) => !p.excluded).length, 0);
    const nFlagged = figures.reduce((n, f) => n + f.panels.filter((p) => !p.excluded)
        .reduce((m, p) => m + p.bars.filter((b) => b.uncertain).length, 0), 0);

    // Group figures by source file for display
    const bySource = figures.reduce((acc, fig) => {
        (acc[fig.source] = acc[fig.source] || []).push(fig);
        return acc;
    }, {});

    return (
        <div className="container-fluid pb-5">
            <div className="d-flex align-items-end flex-wrap gap-3 mb-3">
                <div>
                    <h1 className="mb-1">Batch Extract</h1>
                    <p className="text-muted mb-0">
                        Drop papers (PDF) or figure images. Every bar chart is measured automatically:
                        axis calibrated from all tick labels, bars and error bars read from the pixels.
                    </p>
                </div>
                {figures.length > 0 && (
                    <div className="ms-auto d-flex gap-2 align-items-center">
                        <span className="text-muted small">
                            {nPanels} panels{nFlagged ? `, ${nFlagged} error bars to check` : ''}
                        </span>
                        <button className="btn btn-primary" onClick={exportAll} disabled={exporting || busy || !nPanels}>
                            <i className="bi bi-file-earmark-excel me-2"></i>
                            {exporting ? 'Exporting...' : 'Download Excel'}
                        </button>
                    </div>
                )}
            </div>

            <div
                className={`border border-2 rounded p-4 text-center mb-3 ${dragActive ? 'border-primary bg-light' : 'border-secondary'}`}
                style={{ borderStyle: 'dashed' }}
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

            {errors.length > 0 && (
                <div className="alert alert-warning small">
                    {errors.map((e, i) => <div key={i}>{e}</div>)}
                </div>
            )}

            {Object.entries(bySource).map(([source, figs]) => (
                <div key={source} className="mb-4">
                    <h5 className="mb-3"><i className="bi bi-file-earmark-pdf me-2"></i>{source}</h5>
                    {figs.map((fig) => fig.panels.map((panel, pi) => (
                        <PanelEditor
                            key={`${fig.id}-${pi}`}
                            figure={fig}
                            panel={panel}
                            onChange={(change) => updatePanel(fig.id, pi, change)}
                        />
                    )))}
                </div>
            ))}
        </div>
    );
};

export default BatchExtractor;
