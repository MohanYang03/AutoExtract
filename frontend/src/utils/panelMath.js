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

export const valueAt = (panel, px) => {
    if (panel.slope == null || px == null) return null;
    const v = panel.slope * px + panel.intercept;
    return panel.scale === 'log' ? 10 ** v : v;
};

/** The measured groups of a panel: boxes for a box plot, else bars. */
export const panelItems = (panel) => (panel.kind === 'box' ? panel.boxes : panel.bars) || [];

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

// ---- box plots: mean/SD estimated from the five-number summary ---------- //
// Same estimators as backend/estimators.py (see there for references).

export const WHISKER_TYPES = [
    { key: 'minmax', label: 'Min–max', hint: 'Whiskers are the smallest and largest values' },
    { key: 'tukey', label: '1.5×IQR', hint: 'Tukey whiskers (outliers drawn as points): only the quartiles are used' },
    { key: 'percentile', label: 'Percentile', hint: 'e.g. 5th–95th or 10th–90th: only the quartiles are used' },
    { key: 'unknown', label: 'Not stated', hint: 'Only the quartiles are used (safest)' },
];
export const MEAN_METHODS = [
    { key: 'luo', label: 'Luo 2018', hint: 'Recommended' },
    { key: 'wan', label: 'Wan 2014', hint: '' },
    { key: 'hozo', label: 'Hozo 2005', hint: 'Older; uses min/median/max' },
];
export const SD_METHODS = [
    { key: 'shi', label: 'Shi 2020', hint: 'Recommended' },
    { key: 'wan', label: 'Wan 2014', hint: '' },
    { key: 'hozo', label: 'Hozo 2005', hint: 'Older; needs min/max' },
];

// Numerical Recipes erfc, relative error < 1.2e-7
const erfc = (x) => {
    const z = Math.abs(x);
    const t = 1 / (1 + 0.5 * z);
    const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418
        + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587
        + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? r : 2 - r;
};

/** Inverse standard normal CDF (Acklam's approximation plus one refinement step). */
export const normInv = (p) => {
    if (!(p > 0 && p < 1)) return NaN;
    const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
        1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
        6.680131188771972e+01, -1.328068155288572e+01];
    const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
        -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
    const tail = (q) => (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    let x;
    if (p < 0.02425) x = tail(Math.sqrt(-2 * Math.log(p)));
    else if (p > 1 - 0.02425) x = -tail(Math.sqrt(-2 * Math.log(1 - p)));
    else {
        const q = p - 0.5;
        const r = q * q;
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
    }
    const e = 0.5 * erfc(-x / Math.SQRT2) - p;
    const u = e * Math.sqrt(2 * Math.PI) * Math.exp((x * x) / 2);
    return x - u / (1 + (x * u) / 2);
};

const xi = (n) => 2 * normInv((n - 0.375) / (n + 0.25));
const eta = (n) => 2 * normInv((0.75 * n - 0.125) / (n + 0.25));

/**
 * Estimated mean and SD of one box. Needs n; the whisker ends count as
 * min/max only when the panel says the whiskers show the range.
 */
export const estimateBox = (box, panel) => {
    const { lo: a, q1, median: m, q3, hi: b } = box;
    const n = Number(box.n) || 0;
    if (q1 == null || m == null || q3 == null || n < 2) return { mean: null, sd: null };
    const meanM = panel.mean_method || 'luo';
    const sdM = panel.sd_method || 'shi';
    const range = (panel.whisker_type || 'minmax') === 'minmax' && a != null && b != null;
    let mean;
    let sd;
    if (range) {
        if (meanM === 'wan') mean = (a + 2 * q1 + 2 * m + 2 * q3 + b) / 8;
        else if (meanM === 'hozo') mean = (a + 2 * m + b) / 4;
        else {
            const w1 = 2.2 / (2.2 + n ** 0.75);
            const w2 = 0.7 - 0.72 / n ** 0.55;
            mean = (w1 * (a + b)) / 2 + (w2 * (q1 + q3)) / 2 + (1 - w1 - w2) * m;
        }
        if (sdM === 'wan') sd = 0.5 * ((b - a) / xi(n) + (q3 - q1) / eta(n));
        else if (sdM === 'hozo') {
            if (n <= 15) sd = Math.sqrt(((a - 2 * m + b) ** 2 / 4 + (b - a) ** 2) / 12);
            else sd = n <= 70 ? (b - a) / 4 : (b - a) / 6;
        } else {
            const w = 1 / (1 + 0.07 * n ** 0.6);
            sd = (w * (b - a)) / xi(n) + ((1 - w) * (q3 - q1)) / eta(n);
        }
    } else {
        if (meanM === 'luo') {
            const w = 0.7 + 0.39 / n;
            mean = (w * (q1 + q3)) / 2 + (1 - w) * m;
        } else mean = (q1 + m + q3) / 3;
        sd = (q3 - q1) / eta(n);
    }
    return { mean, sd };
};

/** Bowley skewness beyond ±0.3: normal-theory estimates may be biased. */
export const isSkewed = (box) => {
    if (box.q1 == null || box.q3 == null || box.median == null) return false;
    const iqr = box.q3 - box.q1;
    return iqr > 0 && Math.abs((box.q3 + box.q1 - 2 * box.median) / iqr) > 0.3;
};

/** Recompute a box's values from its (possibly dragged) pixel positions. */
export const boxWithValues = (panel, box) => ({
    ...box,
    q1: valueAt(panel, box.q1_px),
    median: valueAt(panel, box.median_px),
    q3: valueAt(panel, box.q3_px),
    lo: valueAt(panel, box.lo_px),
    hi: valueAt(panel, box.hi_px),
});

/** Long-format row for a box: estimated mean and SD; SEM = SD/√n. */
export const boxRow = (box, panel, paperId) => {
    const { mean, sd } = estimateBox(box, panel);
    const n = Number(box.n) || null;
    return {
        'mean': mean ?? '',
        'sd': sd ?? '',
        'age(week)': '',
        'sex(male)': '',
        'acupoint': '',
        'treatment_group': box.label || '',
        'sample_size': n || '',
        'treatment_duration': '',
        'depression_model_duration': '',
        'paper_id': paperId ?? '',
        'sem': sd != null && n ? sd / Math.sqrt(n) : '',
        'mean+sem': mean != null && sd != null ? mean + sd : '',
    };
};

const tsvCell = (v) => (v == null ? '' : String(v));

/** Tab-separated rows that paste straight into a spreadsheet (no header) */
export const panelToTSV = (panel, paperId) =>
    panelItems(panel)
        .map((item) => {
            const row = panel.kind === 'box' ? boxRow(item, panel, paperId) : barRow(item, paperId, panel.error_type);
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
