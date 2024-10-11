"""No-look-ahead and consistency QA checks for the GTAA Mom-Carry strategy."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gtaa.risk.covariance import rolling_covariance_partial
from gtaa.risk.scaling import portfolio_vol


# ── Individual checks ─────────────────────────────────────────────────────────

def assert_weights_before_returns(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
) -> None:
    """Raise if weights extend past the return universe (would imply look-ahead)."""
    if weights.index.max() > returns.index.max():
        raise AssertionError(
            f"Weights extend to {weights.index.max().date()} but returns end at "
            f"{returns.index.max().date()}. Possible look-ahead."
        )
    if not weights.index.is_monotonic_increasing:
        raise AssertionError("Weights index is not sorted.")
    if not returns.index.is_monotonic_increasing:
        raise AssertionError("Returns index is not sorted.")


def check_raw_weights_sum_to_zero(
    raw_weights: pd.DataFrame,
    tolerance: float = 1e-8,
) -> bool:
    """Return True if all active rows of raw_weights sum to zero within tolerance."""
    row_sums = raw_weights.sum(axis=1)
    active = raw_weights.abs().sum(axis=1) > 0
    if active.sum() == 0:
        return True
    return bool((row_sums[active].abs() < tolerance).all())


def check_exante_vols(
    scaled_weights: pd.DataFrame,
    returns: pd.DataFrame,
    target_vol: float = 0.01,
    lookback: int = 36,
) -> pd.Series:
    """Compute ex-ante vol for every date in scaled_weights.

    Uses rolling_covariance_partial so partial universes are handled gracefully.

    Returns:
        Series of ex-ante annualized volatilities indexed by date.
    """
    vols: dict = {}

    for dt in scaled_weights.index.intersection(returns.index):
        try:
            cov = rolling_covariance_partial(returns, dt, lookback=lookback)
        except ValueError:
            continue

        if cov.empty:
            continue

        w = scaled_weights.loc[dt].reindex(cov.index).fillna(0.0)
        if w.abs().sum() == 0:
            continue

        vols[dt] = portfolio_vol(w, cov)

    return pd.Series(vols, name="ex_ante_vol")


# ── Full QA report ────────────────────────────────────────────────────────────

def run_gtaa_qa_checks(result: dict, data, config: dict) -> pd.DataFrame:
    """Run all QA checks and return a summary DataFrame.

    Args:
        result: Dict returned by run_gtaa_mom_carry_backtest.
        data:   GTAAMomCarryData object.
        config: Strategy config dict.

    Returns:
        DataFrame with columns [Check, Pass, Value].
    """
    lookback = config["risk"]["covariance_lookback_months"]
    fmp_tgt  = config["risk"]["fmp_target_vol"]
    fin_tgt  = config["risk"]["final_target_vol"]
    tol      = 1e-6

    eq_vols  = check_exante_vols(result["equity_scaled_weights"],    data.equity_returns,     fmp_tgt, lookback)
    com_vols = check_exante_vols(result["commodity_scaled_weights"],  data.commodity_returns,  fmp_tgt, lookback)
    fi_vols  = check_exante_vols(result["fi_scaled_weights"],        data.fi_returns,         fmp_tgt, lookback)
    all_ret  = data.all_returns.reindex(columns=result["final_weights"].columns)
    fin_vols = check_exante_vols(result["final_weights"],            all_ret,                  fin_tgt, lookback)

    def _vol_check(vols: pd.Series, target: float) -> bool:
        if vols.empty:
            return False
        return bool(np.nanmax(np.abs(vols - target)) < tol)

    rows = [
        {
            "Check": "Covariance lookback ≥ 36 months",
            "Pass":  lookback >= 36,
            "Value": lookback,
        },
        {
            "Check": "Equity raw weights sum to zero",
            "Pass":  check_raw_weights_sum_to_zero(result["equity_raw_weights"]),
            "Value": float(result["equity_raw_weights"].sum(axis=1).abs().max()),
        },
        {
            "Check": "Commodity raw weights sum to zero",
            "Pass":  check_raw_weights_sum_to_zero(result["commodity_raw_weights"]),
            "Value": float(result["commodity_raw_weights"].sum(axis=1).abs().max()),
        },
        {
            "Check": "Fixed-income raw weights sum to zero",
            "Pass":  check_raw_weights_sum_to_zero(result["fi_raw_weights"]),
            "Value": float(result["fi_raw_weights"].sum(axis=1).abs().max()),
        },
        {
            "Check": "Equity FMP ex-ante vol = 1%",
            "Pass":  _vol_check(eq_vols, fmp_tgt),
            "Value": float(eq_vols.mean()) if not eq_vols.empty else float("nan"),
        },
        {
            "Check": "Commodity FMP ex-ante vol = 1%",
            "Pass":  _vol_check(com_vols, fmp_tgt),
            "Value": float(com_vols.mean()) if not com_vols.empty else float("nan"),
        },
        {
            "Check": "Fixed-income FMP ex-ante vol = 1%",
            "Pass":  _vol_check(fi_vols, fmp_tgt),
            "Value": float(fi_vols.mean()) if not fi_vols.empty else float("nan"),
        },
        {
            "Check": "Final portfolio ex-ante vol = 1%",
            "Pass":  _vol_check(fin_vols, fin_tgt),
            "Value": float(fin_vols.mean()) if not fin_vols.empty else float("nan"),
        },
        {
            "Check": "Backtest ≥ 10 years (120 months)",
            "Pass":  len(result["final_returns"]) >= 120,
            "Value": len(result["final_returns"]),
        },
        {
            "Check": "Final returns contain no missing values",
            "Pass":  not result["final_returns"].isna().any(),
            "Value": int(result["final_returns"].isna().sum()),
        },
        {
            "Check": "Weights do not extend past return data (no look-ahead)",
            "Pass":  result["final_weights"].index.max() <= data.all_returns.index.max(),
            "Value": str(result["final_weights"].index.max().date()),
        },
    ]

    return pd.DataFrame(rows)
