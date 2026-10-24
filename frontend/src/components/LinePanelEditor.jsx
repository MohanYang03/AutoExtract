/**
 * LinePanelEditor Component
 *
 * One extracted line plot: each series' traced line, its data points
 * (orange) and error-bar caps (blue) drawn on the figure, and a long-format
 * table with one row per point. Drag a point or a cap, or click it and use
 * the arrow keys (Shift = faster), to correct it; values update live. Series
 * names are editable and become treatment_group.
 */

import React, { useCallback, useRef, useState } from 'react';
import { fmt, lineRow, panelName, panelToTSV, seriesName } from '../utils/panelMath';

const KEYS = ['y_px', 'cap_hi_px', 'cap_lo_px'];

const rgb = ([b, g, r]) => `rgb(${r},${g},${b})`;

const LinePanelEditor = ({ figure, panel, onChange }) => {
    const svgRef = useRef(null);
    const [drag, setDrag] = useState(null);          // {series, index, key} while dragging
    const [selected, setSelected] = useState(null);  // same, for keyboard nudging
    const [copied, setCopied] = useState(false);
    const [nAll, setNAll] = useState('');

    const plotH = panel.axis_y - panel.plot_top;
    const x0 = Math.max(0, Math.min(panel.label_left ?? panel.axis_x - 80, panel.axis_x - 80) - 10);
    const y0 = Math.max(0, panel.plot_top - 0.08 * plotH);
    const x1 = Math.min(figure.width, panel.plot_right + 15);
    const y1 = Math.min(figure.height, panel.axis_y + 0.22 * plotH);
    const lineW = Math.max(1.5, (x1 - x0) / 300);
    const errLabel = panel.error_type === 'sd' ? 'SD' : 'SEM';

    const toImage = useCallback((evt) => {
        const svg = svgRef.current;
        const pt = svg.createSVGPoint();
        pt.x = evt.clientX;
        pt.y = evt.clientY;
        return pt.matrixTransform(svg.getScreenCTM().inverse());
    }, []);

    const clampY = (y) => Math.max(panel.plot_top - 0.05 * plotH, Math.min(panel.axis_y + 0.25 * plotH, y));
    const move = (t, y) => onChange({ type: 'point', series: t.series, index: t.index, patch: { [t.key]: clampY(y) } });

    const onPointerDown = (evt) => {
        const { x, y } = toImage(evt);
        const tol = 8 * lineW;
        let best = null;
        panel.series.forEach((s, si) => s.points.forEach((p, i) => {
            if (Math.abs(p.x_px - x) > tol) return;
            for (const key of KEYS) {
                if (p[key] == null) continue;
                const d = Math.hypot(p[key] - y, (p.x_px - x) / 2);
                if (d <= tol && (!best || d < best.d)) best = { series: si, index: i, key, d };
            }
        }));
        if (!best) return;
        evt.preventDefault();
        svgRef.current?.focus();
        svgRef.current?.setPointerCapture?.(evt.pointerId);
        setDrag(best);
        setSelected(best);
    };

    const onKeyDown = (evt) => {
        if (!selected || !['ArrowUp', 'ArrowDown'].includes(evt.key)) return;
        evt.preventDefault();
        const step = (evt.shiftKey ? 2 : 0.5) * (evt.key === 'ArrowUp' ? -1 : 1);
        move(selected, panel.series[selected.series].points[selected.index][selected.key] + step);
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
        panel.series.forEach((s, si) => s.points.forEach((_, i) => onChange({ type: 'point', series: si, index: i, patch: { n } })));
    };

    const isSel = (si, i, key) => selected && selected.series === si && selected.index === i && selected.key === key;
    const nPoints = panel.series.reduce((n, s) => n + s.points.length, 0);

    return (
        <div className={`card border-0 shadow-sm mb-3 panel-card ${panel.excluded ? 'opacity-50' : ''}`}>
            <div className="panel-head d-flex align-items-center gap-2 flex-wrap">
                <strong>{panelName(figure, panel)}</strong>
                <span className="text-muted text-truncate" style={{ maxWidth: 320 }}>{panel.y_title}</span>
                <span className="badge bg-info text-dark">line plot</span>
                <span className="small text-muted">{panel.series.length} series · {nPoints} points</span>
                <div className="ms-auto d-flex gap-2 align-items-center">
                    <div className="btn-group btn-group-sm" role="group" aria-label="Error bar type">
                        {['sem', 'sd'].map((t) => (
                            <button key={t} type="button"
                                className={`btn ${panel.error_type === t ? 'btn-secondary' : 'btn-outline-secondary'}`}
                                onClick={() => onChange({ type: 'panel', patch: { error_type: t } })}>
                                {t.toUpperCase()}
                            </button>
                        ))}
                    </div>
                    <button className="btn btn-sm btn-outline-primary" onClick={copyRows} disabled={panel.excluded}
                        title="Copy rows in long format (one per point)">
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
                            onPointerMove={(evt) => { if (drag) move(drag, toImage(evt).y); }}
                            onPointerUp={() => setDrag(null)}
                            onPointerCancel={() => setDrag(null)}
                            style={{ cursor: drag ? 'ns-resize' : 'crosshair' }}
                            aria-label={`${panelName(figure, panel)} line plot with editable points`}
                        >
                            <image href={figure.image_url} x="0" y="0" width={figure.width} height={figure.height} />
                            {panel.ticks.map(([ty, val], i) => (
                                <circle key={i} cx={panel.axis_x + 3 * lineW} cy={ty} r={1.5 * lineW}
                                    fill={val == null ? '#dc3545' : '#198754'} />
                            ))}
                            {panel.series.map((s, si) => (
                                <g key={si}>
                                    <polyline points={s.trace.map(([x, y]) => `${x},${y}`).join(' ')} fill="none"
                                        stroke="#d63384" strokeWidth={lineW / 1.5} strokeOpacity="0.6" pointerEvents="none" />
                                    {s.points.map((p, i) => (
                                        <g key={i}>
                                            {['cap_hi_px', 'cap_lo_px'].map((k) => p[k] != null && (
                                                <line key={k} x1={p.x_px - 4 * lineW} x2={p.x_px + 4 * lineW} y1={p[k]} y2={p[k]}
                                                    stroke={p.uncertain ? '#dc3545' : '#0d6efd'}
                                                    strokeWidth={isSel(si, i, k) ? 2 * lineW : lineW} />
                                            ))}
                                            <circle cx={p.x_px} cy={p.y_px} r={3 * lineW} fill="none" stroke="#fd7e14"
                                                strokeWidth={isSel(si, i, 'y_px') ? 2 * lineW : lineW} />
                                        </g>
                                    ))}
                                </g>
                            ))}
                        </svg>
                        <div className="small text-muted mt-1">
                            Drag a point or an error-bar cap, or click it and use ↑/↓ (Shift = faster).
                            Axis: {panel.calibration_inliers}/{panel.ticks.length} tick labels used
                            {panel.x_slope == null ? '; x labels kept as text.' : '; x axis calibrated.'}
                        </div>
                    </div>
                    <div className="col-xl-7 col-lg-6">
                        <div className="table-responsive" style={{ maxHeight: 420 }}>
                            <table className="table table-sm align-middle mb-2">
                                <thead>
                                    <tr>
                                        <th>series (treatment_group)</th>
                                        <th className="text-end">x</th>
                                        <th className="text-end">mean</th>
                                        <th className="text-end">{errLabel.toLowerCase()}</th>
                                        <th style={{ width: 76 }}>n</th>
                                        <th className="text-end">{panel.error_type === 'sd' ? 'sem' : 'sd'}</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {panel.series.map((s, si) => s.points.map((p, i) => {
                                        const row = lineRow(p, s, si, figure.paper_id, panel.error_type);
                                        const err = panel.error_type === 'sd' ? row.sd : row.sem;
                                        const other = panel.error_type === 'sd' ? row.sem : row.sd;
                                        return (
                                            <tr key={`${si}-${i}`} className={p.uncertain ? 'table-warning' : ''}>
                                                <td>
                                                    {i === 0 ? (
                                                        <div className="d-flex align-items-center gap-1">
                                                            <span style={{ display: 'inline-block', width: 10, height: 10, background: rgb(s.color), borderRadius: 2 }}></span>
                                                            <input className="form-control form-control-sm" value={s.label || ''}
                                                                placeholder={seriesName(s, si)}
                                                                onChange={(e) => onChange({ type: 'series', series: si, patch: { label: e.target.value } })} />
                                                        </div>
                                                    ) : <span className="text-muted small ps-3">{seriesName(s, si)}</span>}
                                                </td>
                                                <td className="text-end font-monospace">{p.x != null ? fmt(p.x) : p.x_label}</td>
                                                <td className="text-end font-monospace">{fmt(row.mean)}</td>
                                                <td className="text-end font-monospace">{fmt(err)}</td>
                                                <td>
                                                    <input type="number" min="1" className="form-control form-control-sm"
                                                        value={p.n ?? ''}
                                                        onChange={(e) => onChange({ type: 'point', series: si, index: i, patch: { n: e.target.value === '' ? null : Number(e.target.value) } })} />
                                                </td>
                                                <td className="text-end font-monospace text-muted">{fmt(other)}</td>
                                            </tr>
                                        );
                                    }))}
                                </tbody>
                            </table>
                        </div>
                        <div className="d-flex align-items-center gap-2 small">
                            <span className="text-muted">Same n for every point:</span>
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

export default LinePanelEditor;
