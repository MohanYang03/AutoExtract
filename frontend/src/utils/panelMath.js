/**
 * Pure helpers for extracted panels: pixel <-> value conversion and the
 * long-format rows used for both the table and clipboard copy.
 */

// Same column order as the meta-analysis Long_format_*.xlsx sheets
export const LONG_COLUMNS = [
    'mean', 'sd', 'age(week)', 'sex(male)', 'acupoint', 'treatment_group',
    'sample_size', 'treatment_duration', 'depression_model_duration',
    'paper_id', 'sem', 'mean+sem',
];

export const valueAt = (panel, px) =>
    panel.slope == null ? null : panel.slope * px + panel.intercept;

/** Recompute a bar's values from its (possibly dragged) pixel positions */
export const withValues = (panel, bar) => {
    const mean = valueAt(panel, bar.top_px);
    const top = bar.cap_px == null ? mean : valueAt(panel, bar.cap_px);
    return { ...bar, mean, mean_plus_err: top };
};

export const barRow = (bar, paperId) => {
    const sem = bar.mean == null || bar.mean_plus_err == null ? null : bar.mean_plus_err - bar.mean;
    const n = Number(bar.n) || null;
    return {
        'mean': bar.mean,
        'sd': sem != null && n ? sem * Math.sqrt(n) : '',
        'age(week)': '',
        'sex(male)': '',
        'acupoint': '',
        'treatment_group': bar.label || '',
        'sample_size': n || '',
        'treatment_duration': '',
        'depression_model_duration': '',
        'paper_id': paperId ?? '',
        'sem': sem,
        'mean+sem': bar.mean_plus_err,
    };
};

/** Tab-separated rows that paste straight into a spreadsheet (no header) */
export const panelToTSV = (panel, paperId) =>
    panel.bars
        .map((bar) => LONG_COLUMNS.map((c) => {
            const v = barRow(bar, paperId)[c];
            return v == null ? '' : v;
        }).join('\t'))
        .join('\n');

export const panelName = (figure, panel) => {
    const fig = figure.figure ? `Fig ${figure.figure}` : `Page ${figure.page + 1}`;
    return `${fig}${panel.letter || ''}`;
};
