/**
 * BoxPanelEditor Component
 *
 * One extracted box-plot panel: the figure crop with each box's quartiles
 * (orange), median (blue) and whisker ends (orange ticks) drawn on top, the
 * five-number summary, and the mean/SD estimated from it for meta-analysis.
 * Drag a line, or click it and use the arrow keys (Shift = faster), to
 * correct it. What the whiskers show and the estimation methods are set per
 * panel; estimates need n.
 */

import React, { useCallback, useRef, useState } from 'react';
import {
    MEAN_METHODS, SD_METHODS, VIOLIN_ENDS, WHISKER_TYPES, estimateBox, fmt, isSkewed, panelName, panelToTSV,
} from '../utils/panelMath';

const LINES = ['hi_px', 'q3_px', 'median_px', 'q1_px', 'lo_px'];

const BoxPanelEditor = ({ figure, panel, onChange }) => {
    const svgRef = useRef(null);
    const [drag, setDrag] = useState(null);          // {box, key} while dragging
    const [selected, setSelected] = useState(null);  // {box, key} for keyboard nudging
    const [copied, setCopied] = useState(false);
    const [nAll, setNAll] = useState('');

    const plotH = panel.axis_y - panel.plot_top;
    const x0 = Math.max(0, Math.min(panel.label_left ?? panel.axis_x - 80, panel.axis_x - 80) - 10);
    const y0 = Math.max(0, panel.plot_top - 0.08 * plotH);
    const x1 = Math.min(figure.width, panel.plot_right + 15);
    const y1 = Math.min(figure.height, panel.axis_y + 0.22 * plotH);
    const lineW = Math.max(1.5, (x1 - x0) / 300);
    const usesRange = panel.whisker_type === 'minmax';
    const isViolin = panel.kind === 'violin';
    const noun = isViolin ? 'violin' : 'box';
    const medianOnly = panel.boxes.some((b) => b.q1 == null || b.q3 == null);

    const toImage = useCallback((evt) => {
        const svg = svgRef.current;
        const pt = svg.createSVGPoint();
        pt.x = evt.clientX;
        pt.y = evt.clientY;
        return pt.matrixTransform(svg.getScreenCTM().inverse());
    }, []);

    const clampY = (y) => Math.max(panel.plot_top - 0.05 * plotH, Math.min(panel.axis_y + 0.25 * plotH, y));
    const moveLine = (target, y) => onChange({ type: 'box', index: target.box, patch: { [target.key]: clampY(y) } });

    const onPointerMove = (evt) => {
        if (drag) moveLine(drag, toImage(evt).y);
    };

    const onPointerDown = (evt) => {
        const { x, y } = toImage(evt);
        const tol = 8 * lineW;
        let best = null;
        panel.boxes.forEach((b, i) => {
            if (x < b.x1 - tol || x > b.x2 + tol) return;
            for (const key of LINES) {
                if ((key === 'q1_px' || key === 'q3_px') && (b.q1 == null || b.q3 == null)) continue;
                const d = Math.abs(b[key] - y);
                if (d <= tol && (!best || d < best.d)) best = { box: i, key, d };
            }
        });
        if (!best) return;
        evt.preventDefault();
        svgRef.current?.focus();
        svgRef.current?.setPointerCapture?.(evt.pointerId);
        setDrag({ box: best.box, key: best.key });
        setSelected({ box: best.box, key: best.key });
    };

    const onKeyDown = (evt) => {
        if (!selected || !['ArrowUp', 'ArrowDown'].includes(evt.key)) return;
        evt.preventDefault();
        const step = (evt.shiftKey ? 2 : 0.5) * (evt.key === 'ArrowUp' ? -1 : 1);
        moveLine(selected, panel.boxes[selected.box][selected.key] + step);
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
        panel.boxes.forEach((_, i) => onChange({ type: 'box', index: i, patch: { n } }));
    };

    const setPanel = (patch) => onChange({ type: 'panel', patch });
    const flagged = panel.boxes.filter((b) => b.uncertain).length;
    const skewed = panel.boxes.filter(isSkewed).length;
    const isSel = (i, key) => selected && selected.box === i && selected.key === key;
    const width = (i, key) => (isSel(i, key) ? 2 * lineW : lineW);

    const choice = (id, label, value, options, key) => (
        <div className="d-flex align-items-center gap-1">
            <label htmlFor={id} className="small text-muted mb-0">{label}</label>
            <select id={id} className="form-select form-select-sm" style={{ width: 'auto' }} value={value}
                onChange={(e) => setPanel({ [key]: e.target.value })}
                title={options.find((o) => o.key === value)?.hint}>
                {options.map((o) => <option key={o.key} value={o.key}>{o.label}</option>)}
            </select>
        </div>
    );

    return (
        <div className={`card border-0 shadow-sm mb-3 panel-card ${panel.excluded ? 'opacity-50' : ''}`}>
            <div className="panel-head d-flex align-items-center gap-2 flex-wrap">
                <strong>{panelName(figure, panel)}</strong>
                <span className="text-muted text-truncate" style={{ maxWidth: 320 }}>{panel.y_title}</span>
                <span className="badge bg-info text-dark">{noun} plot</span>
                {panel.scale === 'log' && <span className="badge bg-secondary">log axis</span>}
                {flagged > 0 && (
                    <span className="badge bg-warning text-dark">{flagged} box{flagged > 1 ? 'es' : ''} to check</span>
                )}
                <div className="ms-auto d-flex gap-2 align-items-center">
                    <button className="btn btn-sm btn-outline-primary" onClick={copyRows} disabled={panel.excluded}
                        title="Copy rows in long format with the estimated mean and SD">
                        <i className="bi bi-clipboard me-1"></i>{copied ? 'Copied' : 'Copy rows'}
                    </button>
                    <button className="btn btn-sm btn-outline-secondary" onClick={() => setPanel({ excluded: !panel.excluded })}>
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
                            aria-label={`${panelName(figure, panel)} ${noun} plot with editable quartiles`}
                        >
                            <image href={figure.image_url} x="0" y="0" width={figure.width} height={figure.height} />
                            {panel.ticks.map(([ty, val], i) => (
                                <circle key={i} cx={panel.axis_x + 3 * lineW} cy={ty} r={1.5 * lineW}
                                    fill={val == null ? '#dc3545' : '#198754'} />
                            ))}
                            {panel.boxes.map((b, i) => {
                                const cx = (b.x1 + b.x2) / 2;
                                const capHalf = Math.max(5, (b.x2 - b.x1) * 0.3);
                                const edge = b.uncertain ? '#dc3545' : '#fd7e14';
                                return (
                                    <g key={i}>
                                        <rect x={b.x1} y={Math.min(b.q3_px, b.q1_px)} width={b.x2 - b.x1}
                                            height={Math.abs(b.q1_px - b.q3_px)}
                                            fill="rgba(253,126,20,0.10)" stroke="none" pointerEvents="none" />
                                        <line x1={cx} x2={cx} y1={b.hi_px} y2={b.q3_px} stroke={edge}
                                            strokeWidth={lineW / 2} strokeDasharray={`${2 * lineW} ${lineW}`} pointerEvents="none" />
                                        <line x1={cx} x2={cx} y1={b.q1_px} y2={b.lo_px} stroke={edge}
                                            strokeWidth={lineW / 2} strokeDasharray={`${2 * lineW} ${lineW}`} pointerEvents="none" />
                                        {['hi_px', 'lo_px'].map((k) => (
                                            <line key={k} x1={cx - capHalf} x2={cx + capHalf} y1={b[k]} y2={b[k]}
                                                stroke={edge} strokeWidth={width(i, k)} />
                                        ))}
                                        {b.q1 != null && b.q3 != null && ['q3_px', 'q1_px'].map((k) => (
                                            <line key={k} x1={b.x1} x2={b.x2} y1={b[k]} y2={b[k]} stroke={edge} strokeWidth={width(i, k)} />
                                        ))}
                                        <line x1={b.x1} x2={b.x2} y1={b.median_px} y2={b.median_px}
                                            stroke="#0d6efd" strokeWidth={width(i, 'median_px')} />
                                        {b.mean_px != null && (
                                            <path d={`M${cx - 2 * lineW},${b.mean_px} h${4 * lineW} M${cx},${b.mean_px - 2 * lineW} v${4 * lineW}`}
                                                stroke="#198754" strokeWidth={lineW / 1.5} pointerEvents="none" />
                                        )}
                                        <text x={cx} y={panel.axis_y - 4 * lineW} fontSize={8 * lineW}
                                            textAnchor="middle" fill="#fd7e14" fontWeight="700">{i + 1}</text>
                                    </g>
                                );
                            })}
                        </svg>
                        <div className="small text-muted mt-1">
                            Drag a line (whisker end, Q3, median, Q1), or click it and use ↑/↓ (Shift = faster).
                            Axis: {panel.calibration_inliers}/{panel.ticks.length} tick labels used.
                        </div>
                    </div>
                    <div className="col-xl-7 col-lg-6">
                        <div className="d-flex flex-wrap gap-3 mb-2">
                            {choice(`wh-${figure.id}-${panel.axis_x}`, isViolin ? 'Violin ends' : 'Whiskers',
                                panel.whisker_type || 'minmax', isViolin ? VIOLIN_ENDS : WHISKER_TYPES, 'whisker_type')}
                            {choice(`mm-${figure.id}-${panel.axis_x}`, 'Mean', panel.mean_method || 'luo', MEAN_METHODS, 'mean_method')}
                            {choice(`sm-${figure.id}-${panel.axis_x}`, 'SD', panel.sd_method || 'shi', SD_METHODS, 'sd_method')}
                        </div>
                        <div className="table-responsive">
                            <table className="table table-sm align-middle mb-2">
                                <thead>
                                    <tr>
                                        <th style={{ width: 28 }}>#</th>
                                        <th>treatment_group</th>
                                        <th className={`text-end ${usesRange ? '' : 'text-muted'}`}>min</th>
                                        <th className="text-end">Q1</th>
                                        <th className="text-end">median</th>
                                        <th className="text-end">Q3</th>
                                        <th className={`text-end ${usesRange ? '' : 'text-muted'}`}>max</th>
                                        <th style={{ width: 76 }}>n</th>
                                        <th className="text-end">mean*</th>
                                        <th className="text-end">sd*</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {panel.boxes.map((b, i) => {
                                        const est = estimateBox(b, panel);
                                        const skew = isSkewed(b);
                                        return (
                                            <tr key={i} className={b.uncertain ? 'table-warning' : ''} title={b.note || undefined}>
                                                <td className="text-muted">{i + 1}</td>
                                                <td>
                                                    <input className="form-control form-control-sm" value={b.label || ''}
                                                        placeholder="group name"
                                                        onChange={(e) => onChange({ type: 'box', index: i, patch: { label: e.target.value } })} />
                                                </td>
                                                <td className={`text-end font-monospace ${usesRange ? '' : 'text-muted'}`}>{fmt(b.lo)}</td>
                                                <td className="text-end font-monospace">{fmt(b.q1)}</td>
                                                <td className="text-end font-monospace">{fmt(b.median)}</td>
                                                <td className="text-end font-monospace">{fmt(b.q3)}</td>
                                                <td className={`text-end font-monospace ${usesRange ? '' : 'text-muted'}`}>{fmt(b.hi)}</td>
                                                <td>
                                                    <input type="number" min="2" className="form-control form-control-sm"
                                                        value={b.n ?? ''} placeholder="n"
                                                        onChange={(e) => onChange({ type: 'box', index: i, patch: { n: e.target.value === '' ? null : Number(e.target.value) } })} />
                                                </td>
                                                <td className="text-end font-monospace">
                                                    {fmt(est.mean)}
                                                    {skew && est.mean != null && (
                                                        <i className="bi bi-exclamation-triangle text-warning ms-1"
                                                            title="Skewed box: normal-theory estimates may be biased"></i>
                                                    )}
                                                </td>
                                                <td className="text-end font-monospace">{fmt(est.sd)}</td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                        <div className="d-flex align-items-center gap-2 small">
                            <span className="text-muted">Same n for every box:</span>
                            <input type="number" min="2" className="form-control form-control-sm" style={{ width: 80 }}
                                value={nAll} onChange={(e) => setNAll(e.target.value)}
                                onKeyDown={(e) => { if (e.key === 'Enter') applyN(); }} />
                            <button className="btn btn-sm btn-outline-secondary" onClick={applyN} disabled={!nAll}>Apply</button>
                        </div>
                        <div className="small text-muted mt-2">
                            * Estimated from {medianOnly ? 'min, median, max (only the median is drawn)'
                                : usesRange ? 'min, Q1, median, Q3, max'
                                    : `Q1, median, Q3 (${isViolin ? 'the ends' : 'whiskers'} are not the range)`} and n.
                            {skewed > 0 && ` ${skewed} skewed box${skewed > 1 ? 'es' : ''}: estimates may be biased.`}
                        </div>
                        {panel.warnings?.length > 0 && (
                            <div className="small text-muted mt-1">{panel.warnings.join('; ')}</div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default BoxPanelEditor;
