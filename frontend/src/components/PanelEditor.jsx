/**
 * PanelEditor Component
 *
 * One extracted chart panel: the figure crop with the detected bar tops
 * (orange) and error-bar caps (blue, red when unconfirmed) drawn on top,
 * plus an editable long-format table. Dragging a line updates its value live.
 */

import React, { useCallback, useRef, useState } from 'react';
import { barRow, panelName, panelToTSV } from '../utils/panelMath';

const fmt = (v) => (v == null || v === '' ? '' : Number(v).toPrecision(5).replace(/\.?0+$/, ''));

const PanelEditor = ({ figure, panel, onChange }) => {
    const svgRef = useRef(null);
    const [drag, setDrag] = useState(null); // {bar, key}
    const [copied, setCopied] = useState(false);

    // Crop: plot area plus room for tick labels and category labels
    const plotH = panel.axis_y - panel.plot_top;
    const x0 = Math.max(0, Math.min(panel.label_left ?? panel.axis_x - 80, panel.axis_x - 80) - 10);
    const y0 = Math.max(0, panel.plot_top - 0.08 * plotH);
    const x1 = Math.min(figure.width, panel.plot_right + 15);
    const y1 = Math.min(figure.height, panel.axis_y + 0.22 * plotH);
    const lineW = Math.max(2, (x1 - x0) / 250);

    const toImageY = useCallback((evt) => {
        const svg = svgRef.current;
        const pt = svg.createSVGPoint();
        pt.x = evt.clientX;
        pt.y = evt.clientY;
        return pt.matrixTransform(svg.getScreenCTM().inverse()).y;
    }, []);

    const onPointerMove = (evt) => {
        if (!drag) return;
        const y = Math.max(panel.plot_top - 0.05 * plotH, Math.min(panel.axis_y, toImageY(evt)));
        onChange({ type: 'bar', index: drag.bar, patch: { [drag.key]: y } });
    };

    const startDrag = (bar, key) => (evt) => {
        evt.currentTarget.setPointerCapture?.(evt.pointerId);
        setDrag({ bar, key });
    };

    const copyRows = async () => {
        await navigator.clipboard.writeText(panelToTSV(panel, figure.paper_id));
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
    };

    const flagged = panel.bars.filter((b) => b.uncertain).length;

    return (
        <div className={`card border-0 shadow-sm mb-4 ${panel.excluded ? 'opacity-50' : ''}`}>
            <div className="card-header bg-light d-flex align-items-center gap-2 flex-wrap">
                <strong>{panelName(figure, panel)}</strong>
                <span className="text-muted">{panel.y_title}</span>
                {flagged > 0 && (
                    <span className="badge bg-warning text-dark">
                        {flagged} error bar{flagged > 1 ? 's' : ''} to check
                    </span>
                )}
                <div className="ms-auto d-flex gap-2">
                    <button className="btn btn-sm btn-outline-primary" onClick={copyRows} disabled={panel.excluded}>
                        <i className="bi bi-clipboard me-1"></i>{copied ? 'Copied' : 'Copy rows'}
                    </button>
                    <button className="btn btn-sm btn-outline-secondary"
                        onClick={() => onChange({ type: 'panel', patch: { excluded: !panel.excluded } })}>
                        {panel.excluded ? 'Include' : 'Exclude'}
                    </button>
                </div>
            </div>
            <div className="card-body">
                <div className="row g-3">
                    <div className="col-lg-5">
                        <svg
                            ref={svgRef}
                            viewBox={`${x0} ${y0} ${x1 - x0} ${y1 - y0}`}
                            className="w-100 border rounded panel-svg"
                            onPointerMove={onPointerMove}
                            onPointerUp={() => setDrag(null)}
                            onPointerLeave={() => setDrag(null)}
                        >
                            <image href={figure.image_url} x="0" y="0" width={figure.width} height={figure.height} />
                            {panel.ticks.map(([ty, val], i) => (
                                <circle key={i} cx={panel.axis_x + 3 * lineW} cy={ty} r={1.5 * lineW}
                                    fill={val == null ? '#dc3545' : '#198754'} />
                            ))}
                            {panel.bars.map((b, i) => {
                                const cx = (b.x1 + b.x2) / 2;
                                const capHalf = Math.max(8, (b.x2 - b.x1) * 0.25);
                                return (
                                    <g key={i}>
                                        <rect x={b.x1} y={b.top_px} width={b.x2 - b.x1}
                                            height={Math.max(0, panel.axis_y - b.top_px)}
                                            fill="rgba(255,140,0,0.12)" stroke="none" />
                                        <line x1={b.x1} x2={b.x2} y1={b.top_px} y2={b.top_px}
                                            stroke="#fd7e14" strokeWidth={lineW} />
                                        <line x1={b.x1} x2={b.x2} y1={b.top_px} y2={b.top_px}
                                            stroke="transparent" strokeWidth={6 * lineW}
                                            style={{ cursor: 'ns-resize' }} onPointerDown={startDrag(i, 'top_px')} />
                                        {b.cap_px != null && (
                                            <>
                                                <line x1={cx - capHalf} x2={cx + capHalf} y1={b.cap_px} y2={b.cap_px}
                                                    stroke={b.uncertain ? '#dc3545' : '#0d6efd'} strokeWidth={lineW} />
                                                <line x1={cx - capHalf} x2={cx + capHalf} y1={b.cap_px} y2={b.cap_px}
                                                    stroke="transparent" strokeWidth={6 * lineW}
                                                    style={{ cursor: 'ns-resize' }} onPointerDown={startDrag(i, 'cap_px')} />
                                            </>
                                        )}
                                    </g>
                                );
                            })}
                        </svg>
                        <small className="text-muted">
                            Drag an orange bar top or blue/red cap to adjust. Axis: {panel.calibration_inliers}/{panel.ticks.length} tick labels used.
                        </small>
                    </div>
                    <div className="col-lg-7">
                        <div className="table-responsive">
                            <table className="table table-sm align-middle mb-0">
                                <thead>
                                    <tr>
                                        <th>treatment_group</th>
                                        <th className="text-end">mean</th>
                                        <th className="text-end">sem</th>
                                        <th className="text-end">mean+sem</th>
                                        <th style={{ width: 80 }}>n</th>
                                        <th className="text-end">sd</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {panel.bars.map((b, i) => {
                                        const row = barRow(b, figure.paper_id);
                                        return (
                                            <tr key={i} className={b.uncertain ? 'table-warning' : ''}>
                                                <td>
                                                    <input className="form-control form-control-sm" value={b.label || ''}
                                                        onChange={(e) => onChange({ type: 'bar', index: i, patch: { label: e.target.value } })} />
                                                </td>
                                                <td className="text-end font-monospace">{fmt(row.mean)}</td>
                                                <td className="text-end font-monospace">{fmt(row.sem)}</td>
                                                <td className="text-end font-monospace">{fmt(row['mean+sem'])}</td>
                                                <td>
                                                    <input type="number" min="1" className="form-control form-control-sm"
                                                        value={b.n ?? ''}
                                                        onChange={(e) => onChange({ type: 'bar', index: i, patch: { n: e.target.value === '' ? null : Number(e.target.value) } })} />
                                                </td>
                                                <td className="text-end font-monospace">{fmt(row.sd)}</td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                        {panel.warnings?.length > 0 && (
                            <div className="small text-muted mt-2">{panel.warnings.join('; ')}</div>
                        )}
                        {panel.slope == null && (
                            <div className="alert alert-danger small mt-2 mb-0">
                                Axis could not be calibrated (tick labels unreadable), so values are unavailable.
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default PanelEditor;
