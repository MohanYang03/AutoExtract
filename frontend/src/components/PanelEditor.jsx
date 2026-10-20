/**
 * PanelEditor Component
 *
 * One extracted chart panel: the figure crop with each bar's mean (orange)
 * and error-bar cap (blue, red when unconfirmed) drawn on top, plus the
 * long-format table. Drag a line, or click it and use the arrow keys
 * (Shift = faster), to correct it; values update live.
 */

import React, { useCallback, useRef, useState } from 'react';
import { barRow, fmt, panelName, panelToTSV } from '../utils/panelMath';

const PanelEditor = ({ figure, panel, onChange }) => {
    const svgRef = useRef(null);
    const [drag, setDrag] = useState(null);          // {bar, key} while dragging
    const [selected, setSelected] = useState(null);  // {bar, key} for keyboard nudging
    const [copied, setCopied] = useState(false);
    const [nAll, setNAll] = useState('');

    const plotH = panel.axis_y - panel.plot_top;
    const x0 = Math.max(0, Math.min(panel.label_left ?? panel.axis_x - 80, panel.axis_x - 80) - 10);
    const y0 = Math.max(0, panel.plot_top - 0.08 * plotH);
    const x1 = Math.min(figure.width, panel.plot_right + 15);
    const y1 = Math.min(figure.height, panel.axis_y + 0.22 * plotH);
    const lineW = Math.max(1.5, (x1 - x0) / 300);
    const isDot = panel.kind === 'dot';
    const errLabel = panel.error_type === 'sd' ? 'SD' : 'SEM';

    const toImage = useCallback((evt) => {
        const svg = svgRef.current;
        const pt = svg.createSVGPoint();
        pt.x = evt.clientX;
        pt.y = evt.clientY;
        return pt.matrixTransform(svg.getScreenCTM().inverse());
    }, []);

    const clampY = (y) => Math.max(panel.plot_top - 0.05 * plotH, Math.min(panel.axis_y + 0.25 * plotH, y));
    const moveLine = (target, y) => onChange({ type: 'bar', index: target.bar, patch: { [target.key]: clampY(y) } });

    const onPointerMove = (evt) => {
        if (drag) moveLine(drag, toImage(evt).y);
    };

    // Grab the mean or cap line nearest the pointer within the bar under it,
    // so lines a few pixels thin don't have to be hit exactly.
    const onPointerDown = (evt) => {
        const { x, y } = toImage(evt);
        const tol = 8 * lineW;
        let best = null;
        panel.bars.forEach((b, i) => {
            if (x < b.x1 - tol || x > b.x2 + tol) return;
            for (const key of ['top_px', 'cap_px']) {
                if (b[key] == null) continue;
                const d = Math.abs(b[key] - y);
                if (d <= tol && (!best || d < best.d)) best = { bar: i, key, d };
            }
        });
        if (!best) return;
        evt.preventDefault();
        svgRef.current?.focus();
        svgRef.current?.setPointerCapture?.(evt.pointerId);
        setDrag({ bar: best.bar, key: best.key });
        setSelected({ bar: best.bar, key: best.key });
    };

    const onKeyDown = (evt) => {
        if (!selected || !['ArrowUp', 'ArrowDown'].includes(evt.key)) return;
        evt.preventDefault();
        const step = (evt.shiftKey ? 2 : 0.5) * (evt.key === 'ArrowUp' ? -1 : 1);
        moveLine(selected, panel.bars[selected.bar][selected.key] + step);
    };

    const copyRows = async () => {
        try {
            await navigator.clipboard.writeText(panelToTSV(panel, figure.paper_id));
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
        } catch {
            setCopied(false);
        }
    };

    const applyN = () => {
        const n = Number(nAll);
        if (!n || n < 1) return;
        panel.bars.forEach((_, i) => onChange({ type: 'bar', index: i, patch: { n } }));
    };

    const flagged = panel.bars.filter((b) => b.uncertain).length;
    const isSel = (i, key) => selected && selected.bar === i && selected.key === key;

    return (
        <div className={`card border-0 shadow-sm mb-3 panel-card ${panel.excluded ? 'opacity-50' : ''}`}>
            <div className="panel-head d-flex align-items-center gap-2 flex-wrap">
                <strong>{panelName(figure, panel)}</strong>
                <span className="text-muted text-truncate" style={{ maxWidth: 320 }}>{panel.y_title}</span>
                {isDot && <span className="badge bg-info text-dark">dot plot</span>}
                {flagged > 0 && (
                    <span className="badge bg-warning text-dark">
                        {flagged} error bar{flagged > 1 ? 's' : ''} to check
                    </span>
                )}
                <div className="ms-auto d-flex gap-2 align-items-center">
                    <div className="btn-group btn-group-sm" role="group" aria-label="Error bar type">
                        {['sem', 'sd'].map((t) => (
                            <button key={t} type="button"
                                className={`btn ${panel.error_type === t ? 'btn-secondary' : 'btn-outline-secondary'}`}
                                onClick={() => onChange({ type: 'panel', patch: { error_type: t } })}
                                title={t === 'sem' ? 'Error bars show SEM' : 'Error bars show SD'}>
                                {t.toUpperCase()}
                            </button>
                        ))}
                    </div>
                    <button className="btn btn-sm btn-outline-primary" onClick={copyRows} disabled={panel.excluded}
                        title="Copy rows in long format (paste into your sheet)">
                        <i className="bi bi-clipboard me-1"></i>{copied ? 'Copied' : 'Copy rows'}
                    </button>
                    <button className="btn btn-sm btn-outline-secondary"
                        onClick={() => onChange({ type: 'panel', patch: { excluded: !panel.excluded } })}>
                        {panel.excluded ? 'Include' : 'Exclude'}
                    </button>
                </div>
            </div>
            <div className="card-body py-3">
                <div className="row g-3">
                    <div className="col-xl-5 col-lg-6">
                        <svg
                            ref={svgRef}
                            viewBox={`${x0} ${y0} ${x1 - x0} ${y1 - y0}`}
                            className="w-100 border rounded panel-svg"
                            tabIndex={0}
                            onKeyDown={onKeyDown}
                            onPointerDown={onPointerDown}
                            onPointerMove={onPointerMove}
                            onPointerUp={() => setDrag(null)}
                            onPointerCancel={() => setDrag(null)}
                            style={{ cursor: drag ? 'ns-resize' : 'crosshair' }}
                            aria-label={`${panelName(figure, panel)} chart with editable bar positions`}
                        >
                            <image href={figure.image_url} x="0" y="0" width={figure.width} height={figure.height} />
                            {panel.ticks.map(([ty, val], i) => (
                                <circle key={i} cx={panel.axis_x + 3 * lineW} cy={ty} r={1.5 * lineW}
                                    fill={val == null ? '#dc3545' : '#198754'} />
                            ))}
                            {panel.bars.map((b, i) => {
                                const cx = (b.x1 + b.x2) / 2;
                                const capHalf = Math.max(6, (b.x2 - b.x1) * (isDot ? 0.3 : 0.25));
                                return (
                                    <g key={i}>
                                        {!isDot && (
                                            <rect x={b.x1} y={b.top_px} width={b.x2 - b.x1}
                                                height={Math.max(0, panel.axis_y - b.top_px)}
                                                fill="rgba(253,126,20,0.10)" stroke="none" pointerEvents="none" />
                                        )}
                                        <text x={cx} y={panel.axis_y - 4 * lineW} fontSize={8 * lineW}
                                            textAnchor="middle" fill="#fd7e14" fontWeight="700">{i + 1}</text>
                                        <line x1={b.x1} x2={b.x2} y1={b.top_px} y2={b.top_px}
                                            stroke="#fd7e14" strokeWidth={isSel(i, 'top_px') ? 2 * lineW : lineW} />
                                        {b.cap_px != null && (
                                            <>
                                                <line x1={cx - capHalf} x2={cx + capHalf} y1={b.cap_px} y2={b.cap_px}
                                                    stroke={b.uncertain ? '#dc3545' : '#0d6efd'}
                                                    strokeWidth={isSel(i, 'cap_px') ? 2 * lineW : lineW} />
                                            </>
                                        )}
                                    </g>
                                );
                            })}
                        </svg>
                        <div className="small text-muted mt-1">
                            Drag a line, or click it and use ↑/↓ (Shift = faster). Axis: {panel.calibration_inliers}/{panel.ticks.length} tick labels used.
                        </div>
                    </div>
                    <div className="col-xl-7 col-lg-6">
                        <div className="table-responsive">
                            <table className="table table-sm align-middle mb-2">
                                <thead>
                                    <tr>
                                        <th style={{ width: 28 }}>#</th>
                                        <th>treatment_group</th>
                                        <th className="text-end">mean</th>
                                        <th className="text-end">{errLabel.toLowerCase()}</th>
                                        <th className="text-end">mean+{errLabel.toLowerCase()}</th>
                                        <th style={{ width: 76 }}>n</th>
                                        <th className="text-end">{panel.error_type === 'sd' ? 'sem' : 'sd'}</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {panel.bars.map((b, i) => {
                                        const row = barRow(b, figure.paper_id, panel.error_type);
                                        const err = panel.error_type === 'sd' ? row.sd : row.sem;
                                        const other = panel.error_type === 'sd' ? row.sem : row.sd;
                                        return (
                                            <tr key={i} className={b.uncertain ? 'table-warning' : ''}>
                                                <td className="text-muted">{i + 1}</td>
                                                <td>
                                                    <input className="form-control form-control-sm" value={b.label || ''}
                                                        placeholder="group name"
                                                        onChange={(e) => onChange({ type: 'bar', index: i, patch: { label: e.target.value } })} />
                                                </td>
                                                <td className="text-end font-monospace">{fmt(row.mean)}</td>
                                                <td className="text-end font-monospace">{fmt(err)}</td>
                                                <td className="text-end font-monospace">{fmt(row['mean+sem'])}</td>
                                                <td>
                                                    <input type="number" min="1" className="form-control form-control-sm"
                                                        value={b.n ?? ''}
                                                        onChange={(e) => onChange({ type: 'bar', index: i, patch: { n: e.target.value === '' ? null : Number(e.target.value) } })} />
                                                </td>
                                                <td className="text-end font-monospace text-muted">{fmt(other)}</td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                        <div className="d-flex align-items-center gap-2 small">
                            <span className="text-muted">Same n for every bar:</span>
                            <input type="number" min="1" className="form-control form-control-sm" style={{ width: 80 }}
                                value={nAll} onChange={(e) => setNAll(e.target.value)}
                                onKeyDown={(e) => { if (e.key === 'Enter') applyN(); }} />
                            <button className="btn btn-sm btn-outline-secondary" onClick={applyN} disabled={!nAll}>Apply</button>
                        </div>
                        {panel.warnings?.length > 0 && (
                            <div className="small text-muted mt-2">{panel.warnings.join('; ')}</div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default PanelEditor;
