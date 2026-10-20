/**
 * API calls for batch figure extraction
 */

const API_BASE = '/api';

/**
 * Upload one PDF or image and get back every bar-chart panel found in it
 * @param {File} file - PDF or image
 * @param {string} [job] - job id from a previous call, so images share a folder
 * @returns {Promise<Object>} {job, source, figures: [...]}
 */
export const extractFile = async (file, job) => {
    const formData = new FormData();
    formData.append('file', file);
    if (job) formData.append('job', job);

    const response = await fetch(`${API_BASE}/extract`, { method: 'POST', body: formData });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(data.error || `HTTP ${response.status}: ${response.statusText}`);
    }
    return data;
};

/**
 * Build the long-format workbook on the server and download it
 * @param {Array} figures - figures with (possibly edited) panels
 * @param {string} filename - download name
 */
export const downloadWorkbook = async (figures, filename = 'extracted.xlsx') => {
    const response = await fetch(`${API_BASE}/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ figures, filename }),
    });
    if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || 'Export failed');
    }
    const blob = await response.blob();
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
    URL.revokeObjectURL(link.href);
};
