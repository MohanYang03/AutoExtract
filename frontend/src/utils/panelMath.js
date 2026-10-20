/**
 * Pure helpers for extracted panels: pixel <-> value conversion and the
 * long-format rows used for the table, the clipboard and the export.
 */

// Same column order as the meta-analysis Long_format_*.xlsx sheets
export const LONG_COLUMNS = [
    'mean', 'sd', 'age(week)', 'sex(male)', 'acupoint', 'treatment_group',
    'sample_size', 'treatment_duration', 'depression_model_duration',
    'paper_id', 'sem', 'mean+sem',
];

export const valueAt = (panel, px) =>
    panel.slope == null ? null : panel.slope * px + panel.intercept;

/**
 * Recompute a bar's values from its (possibly dragged) pixel positions.
 * The error is a magnitude, as on the server, so dragging the mean past the
 * cap can't produce a negative SEM/SD.
 */
export const withValues = (panel, bar) => {
    const mean = valueAt(panel, bar.top_px);
    const err = bar.cap_px == null || mean == null ? 0 : Math.abs(valueAt(panel, bar.cap_px) - mean);
    return { ...bar, mean, mean_plus_err: mean == null ? null : mean + err };
};

/**
 * One long-format row. The measured error is SEM or SD depending on the
 * panel; the other is derived from n when n is known.
 */
export const barRow = (bar, paperId, errorType = 'sem') => {
    const err = bar.mean == null || bar.mean_plus_err == null ? null : bar.mean_plus_err - bar.mean;
    const n = Number(bar.n) || null;
    let sem = null;
    let sd = null;
    if (err != null) {
        if (errorType === 'sd') {
            sd = err;
            sem = n ? err / Math.sqrt(n) : null;
        } else {
            sem = err;
            sd = n ? err * Math.sqrt(n) : null;
        }
    }
    return {
        'mean': bar.mean,
        'sd': sd ?? '',
        'age(week)': '',
        'sex(male)': '',
        'acupoint': '',
        'treatment_group': bar.label || '',
        'sample_size': n || '',
        'treatment_duration': '',
        'depression_model_duration': '',
        'paper_id': paperId ?? '',
        'sem': sem ?? '',
        'mean+sem': bar.mean_plus_err,
    };
};

const tsvCell = (v) => (v == null ? '' : String(v));

/** Tab-separated rows that paste straight into a spreadsheet (no header) */
export const panelToTSV = (panel, paperId) =>
    panel.bars
        .map((bar) => {
            const row = barRow(bar, paperId, panel.error_type);
            return LONG_COLUMNS.map((c) => tsvCell(row[c])).join('\t');
        })
        .join('\n');

export const panelName = (figure, panel) => {
    let fig = 'Image';
    if (figure.figure) fig = `Fig ${figure.figure}`;
    else if (figure.page >= 0) fig = `Page ${figure.page + 1}`;
    return `${fig}${panel.letter ? ` ${panel.letter}` : ''}`;
};

/** Format a number for display: 4 significant digits, no trailing zeros */
export const fmt = (v) => {
    if (v == null || v === '' || Number.isNaN(Number(v))) return '';
    const n = Number(v);
    if (n === 0) return '0';
    const digits = Math.abs(n) >= 1000 ? 0 : Math.max(0, 3 - Math.floor(Math.log10(Math.abs(n))));
    return n.toFixed(Math.min(digits, 4));
};
