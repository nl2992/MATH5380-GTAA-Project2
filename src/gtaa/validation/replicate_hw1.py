"""HW1 replication: load regime data from Excel, rebuild every column, compare."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gtaa.regime.distances import euclidean_distances_to_reference, normalized_distances
from gtaa.regime.kernels import polynomial_kernel
from gtaa.regime.lowess import zero_order_lowess_forecast
from gtaa.regime.standardization import expanding_zscore
from gtaa.validation.compare_excel import compare_series, tolerance_report


# ── Constants that match the HW1 Excel workbook ──────────────────────────────
# The Excel formula anchors AVERAGE/STDEV to row 4 = 1992-05-29 (second data row).
# The reference date for distances is December 2010.
HW1_ZSCORE_START = "1992-05-29"
HW1_REFERENCE_DATE = pd.Timestamp("2010-12-31")


def _parse_regime_sheet(raw: pd.DataFrame) -> pd.DataFrame:
    """Parse the 'economic regimes' sheet from load_hw1_data() into a clean DataFrame.

    Returns a DataFrame with columns:
        lei, lei_mom, cpi_yoy          — raw macro variables
        z_lei, z_lei_mom, z_cpi_yoy   — Excel z-scores (reference)
        raw_dist, norm_dist, kernel    — Excel distances / weights (reference)
    """
    df = raw.copy()
    df.columns = [
        "date", "lei", "lei_mom", "cpi_yoy", "_b1",
        "date2", "z_lei", "z_lei_mom", "z_cpi_yoy", "_b2",
        "date3", "raw_dist", "norm_dist", "kernel",
    ]
    df = df.iloc[1:].reset_index(drop=True)  # drop the column-label row
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")
    for col in ["lei", "lei_mom", "cpi_yoy", "z_lei", "z_lei_mom",
                "z_cpi_yoy", "raw_dist", "norm_dist", "kernel"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _parse_forecast_sheet(raw: pd.DataFrame) -> pd.DataFrame:
    """Parse 'regime-based asset forecasts' sheet.

    Returns a DataFrame with columns:
        w              — kernel weight (same as regime sheet)
        sp500_ret      — S&P 500 monthly return at date t
        sp500_x_w      — w_t × sp500_ret_{t+1}  (Excel F column)
        barcap_ret     — BarCap Agg monthly return at date t
        barcap_x_w     — w_t × barcap_ret_{t+1}  (Excel J column)
    """
    df = raw.copy()
    df.columns = [
        "date", "w", "_c", "date2", "sp500_ret", "sp500_x_w",
        "_g", "date3", "barcap_ret", "barcap_x_w",
    ]
    df = df.iloc[1:].reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")
    for col in ["w", "sp500_ret", "sp500_x_w", "barcap_ret", "barcap_x_w"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_regime_vectors(regime: pd.DataFrame) -> pd.DataFrame:
    """Compute expanding z-scores matching the HW1 Excel formula exactly.

    Excel: =(x_t - AVERAGE(B$4:B_{t-1})) / STDEV(B$4:B_{t-1})
    - Anchor: 1992-05-29 (second row of data; first row has no LEI mom)
    - ddof=1 (Excel STDEV is sample std)
    - Prior history only (shift(1) so current obs not in its own mean/std)
    """
    lei = regime.loc[HW1_ZSCORE_START:, "lei"]
    mom = regime.loc[HW1_ZSCORE_START:, "lei_mom"].dropna()
    cpi = regime.loc[HW1_ZSCORE_START:, "cpi_yoy"].dropna()

    z_lei = expanding_zscore(lei, min_periods=2)
    z_mom = expanding_zscore(mom, min_periods=2)
    z_cpi = expanding_zscore(cpi, min_periods=2)

    return pd.DataFrame({"z_lei": z_lei, "z_mom": z_mom, "z_cpi": z_cpi}).dropna()


def build_distances_and_kernel(z_df: pd.DataFrame) -> pd.DataFrame:
    """Compute raw distances, normalized distances, and tricube kernel weights.

    Reference date = December 2010 (HW1 asks for the Jan 2011 forecast).
    Only dates strictly before the reference date receive a distance/weight.
    Normalization denominator = max raw distance over all pre-reference dates.
    Kernel: (1 - u³)³  where u = normalized distance.
    """
    ref_vec = z_df.loc[HW1_REFERENCE_DATE]
    z_pre = z_df.loc[z_df.index < HW1_REFERENCE_DATE]

    raw_d = np.sqrt(((z_pre - ref_vec) ** 2).sum(axis=1))
    norm_d = raw_d / raw_d.max()
    kernel = (1 - norm_d ** 3) ** 3

    return pd.DataFrame({
        "raw_dist": raw_d,
        "norm_dist": norm_d,
        "kernel": kernel,
    })


def compute_lowess_forecast(
    kernel: pd.Series,
    returns: pd.Series,
) -> float:
    """Zero-order LOWESS: Σ(w_t × r_{t+1}) / Σ(w_t).

    kernel : kernel weights indexed by date t
    returns: asset return series indexed by date (return AT date t)
    The return used is r_{t+1}, i.e. returns.shift(-1) aligned to kernel dates.
    """
    next_ret = returns.shift(-1)
    common = kernel.index.intersection(next_ret.dropna().index)
    w = kernel.loc[common]
    r = next_ret.loc[common]
    return float((w * r).sum() / w.sum())


def run_hw1_replication(hw1_sheets: dict[str, pd.DataFrame]) -> dict:
    """Run the full HW1 replication and return computed series + comparison reports.

    Args:
        hw1_sheets: Dict from load_hw1_data(). Must contain 'economic regimes'
                    and 'regime-based asset forecasts'.

    Returns:
        {
          'regime':    clean regime DataFrame (raw + Excel reference columns),
          'z_df':      our computed z-score DataFrame,
          'dist_df':   our computed distances + kernel weights,
          'forecasts': {'sp500': float, 'barcap': float},
          'reports':   tolerance_report DataFrame,
        }
    """
    raw_regime = hw1_sheets["economic regimes"]
    raw_fc = hw1_sheets["regime-based asset forecasts"]

    regime = _parse_regime_sheet(raw_regime)
    fc = _parse_forecast_sheet(raw_fc)

    # ── Rebuild everything ─────────────────────────────────────────────────
    z_df = build_regime_vectors(regime)
    dist_df = build_distances_and_kernel(z_df)

    # Forecasts using Excel's own cached return series
    sp500_forecast = compute_lowess_forecast(dist_df["kernel"], fc["sp500_ret"])
    barcap_forecast = compute_lowess_forecast(dist_df["kernel"], fc["barcap_ret"])

    # ── Compare against Excel reference columns ────────────────────────────
    excel_has_z = regime["z_lei"].notna()
    reports_raw = {
        "Z-score LEI": compare_series(
            z_df["z_lei"].reindex(regime.loc[excel_has_z].index),
            regime.loc[excel_has_z, "z_lei"],
            name="Z-score LEI", tol=1e-10,
        ),
        "Z-score LEI mom": compare_series(
            z_df["z_mom"].reindex(regime.loc[excel_has_z].index),
            regime.loc[excel_has_z, "z_lei_mom"],
            name="Z-score LEI mom", tol=1e-10,
        ),
        "Z-score CPI YoY": compare_series(
            z_df["z_cpi"].reindex(regime.loc[excel_has_z].index),
            regime.loc[excel_has_z, "z_cpi_yoy"],
            name="Z-score CPI YoY", tol=1e-10,
        ),
        "Raw distances": compare_series(
            dist_df["raw_dist"].reindex(regime["raw_dist"].dropna().index),
            regime["raw_dist"].dropna(),
            name="Raw distances", tol=1e-10,
        ),
        "Normalized distances": compare_series(
            # Row 1992-04-30 holds =MAX(L16:L227) in the Excel M column — a helper
            # value, not a real normalized distance. Filter to values in [0, 1].
            dist_df["norm_dist"].reindex(regime["norm_dist"].dropna().index),
            regime["norm_dist"].dropna().loc[lambda s: s <= 1.0],
            name="Normalized distances", tol=1e-10,
        ),
        "Kernel weights": compare_series(
            dist_df["kernel"].reindex(regime["kernel"].dropna().index),
            regime["kernel"].dropna(),
            name="Kernel weights", tol=1e-10,
        ),
    }

    return {
        "regime": regime,
        "z_df": z_df,
        "dist_df": dist_df,
        "fc": fc,
        "forecasts": {"sp500": sp500_forecast, "barcap": barcap_forecast},
        "reports": tolerance_report(reports_raw),
        "reports_raw": reports_raw,
    }
