"""
Estimate a sample mean and SD from box-plot statistics, for meta-analysis.

Box plots report medians and quartiles; pooling them with mean ± SD studies
needs an estimate. The standard normal-theory estimators are implemented:

    Mean:  Luo et al. 2018 (recommended), Wan et al. 2014, Hozo et al. 2005
    SD:    Shi et al. 2020 (recommended), Wan et al. 2014, Hozo et al. 2005

Which statistics are usable depends on what the whiskers show:

    S1  min, median, max                  (whiskers = range, no quartiles)
    S2  min, Q1, median, Q3, max          (whiskers = range)
    S3  Q1, median, Q3                    (whiskers are NOT the range, e.g. Tukey 1.5×IQR)

A box plot always gives quartiles, so it is S2 when the whiskers are the
minimum and maximum, and S3 otherwise (Tukey or percentile whiskers are not
sample extremes and must not be used as such).

All estimators assume roughly normal data; `skewness_warning` flags boxes
where that looks doubtful (consider McGrath et al. 2020 / R `estmeansd`).

References
    Hozo SP, Djulbegovic B, Hozo I. BMC Med Res Methodol 2005;5:13.
    Wan X, Wang W, Liu J, Tong T. BMC Med Res Methodol 2014;14:135.
    Luo D, Wan X, Liu J, Tong T. Stat Methods Med Res 2018;27:1785-805.
    Shi J, Luo D, Weng H, et al. Res Synth Methods 2020;11:641-54.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from scipy.stats import norm

MEAN_METHODS = ("luo", "wan", "hozo")
SD_METHODS = ("shi", "wan", "hozo")
WHISKER_TYPES = ("minmax", "tukey", "percentile", "unknown")


@dataclass
class Estimate:
    mean: Optional[float]
    sd: Optional[float]
    scenario: str           # "S2" or "S3"
    note: str = ""


def _xi(n: float) -> float:
    """Expected range of n standard normals: 2·Φ⁻¹((n − 0.375)/(n + 0.25)) (Wan 2014)."""
    return 2.0 * norm.ppf((n - 0.375) / (n + 0.25))


def _eta(n: float) -> float:
    """Expected IQR of n standard normals: 2·Φ⁻¹((0.75n − 0.125)/(n + 0.25)) (Wan 2014)."""
    return 2.0 * norm.ppf((0.75 * n - 0.125) / (n + 0.25))


def scenario_for(whisker_type: str) -> str:
    return "S2" if whisker_type == "minmax" else "S3"


def estimate(q1: float, median: float, q3: float, n: Optional[float],
             minimum: Optional[float] = None, maximum: Optional[float] = None,
             whisker_type: str = "minmax", mean_method: str = "luo",
             sd_method: str = "shi") -> Estimate:
    """
    Mean and SD from box statistics. Needs n (all modern estimators depend on
    it); returns None values without it.
    """
    scen = scenario_for(whisker_type) if minimum is not None and maximum is not None else "S3"
    if not n or n < 2:
        return Estimate(None, None, scen, "enter n")
    n = float(n)
    a, b = minimum, maximum

    # ---- mean ----
    if scen == "S2":
        if mean_method == "wan":
            mean = (a + 2 * q1 + 2 * median + 2 * q3 + b) / 8.0
        elif mean_method == "hozo":
            # Hozo uses min/median/max only
            mean = (a + 2 * median + b) / 4.0
        else:  # Luo 2018, scenario C2 (five-number summary)
            w1 = 2.2 / (2.2 + n ** 0.75)
            w2 = 0.7 - 0.72 / n ** 0.55
            mean = w1 * (a + b) / 2 + w2 * (q1 + q3) / 2 + (1 - w1 - w2) * median
    else:
        if mean_method == "wan" or mean_method == "hozo":
            mean = (q1 + median + q3) / 3.0
        else:  # Luo 2018, scenario C3 (quartiles)
            w = 0.7 + 0.39 / n
            mean = w * (q1 + q3) / 2 + (1 - w) * median

    # ---- SD ----
    note = ""
    if scen == "S2":
        if sd_method == "wan":
            # Wan 2014 eq. for C2: average of the range- and IQR-based estimates
            sd = 0.5 * ((b - a) / _xi(n) + (q3 - q1) / _eta(n))
        elif sd_method == "hozo":
            if n <= 15:
                sd = (((a - 2 * median + b) ** 2) / 4 + (b - a) ** 2) ** 0.5 / 12 ** 0.5
            elif n <= 70:
                sd = (b - a) / 4.0
            else:
                sd = (b - a) / 6.0
        else:  # Shi 2020: weighted range and IQR estimators
            w = 1.0 / (1.0 + 0.07 * n ** 0.6)
            sd = w * (b - a) / _xi(n) + (1 - w) * (q3 - q1) / _eta(n)
    else:
        # quartiles only: Wan's IQR estimator (Shi reduces to it; Hozo needs range)
        sd = (q3 - q1) / _eta(n)
        if sd_method == "hozo":
            note = "Hozo needs min/max; used Wan (IQR)"
    return Estimate(float(mean), float(sd), scen, note)


def skewness_warning(q1: float, median: float, q3: float) -> bool:
    """Bowley skewness beyond ±0.3: normal-theory estimates may be biased."""
    iqr = q3 - q1
    if iqr <= 0:
        return False
    return abs((q3 + q1 - 2 * median) / iqr) > 0.3
