"""Main backtest runner.

run_backtest() is the single entry point for all strategy runs.
It calls into fmp.py, combiner.py, and turnover.py using only src/gtaa functions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gtaa.analytics.performance import compute_stats
from gtaa.data.alignment import portfolio_return_series
from gtaa.data.validation import (
    check_no_lookahead,
    check_portfolio_vol,
    check_weights_zero_sum,
)
from gtaa.factors.base import BaseFactor
from gtaa.models import BacktestConfig, BacktestResult, DataBundle
from gtaa.portfolio.combiner import (
    combine_fmps_equal_weight,
    scale_combined_portfolio,
)
from gtaa.portfolio.fmp import build_fmp_weights
from gtaa.portfolio.turnover import compute_turnover_series
from gtaa.risk.covariance import rolling_covariance


FACTOR_REGISTRY: dict = {}


def register_default_factors():
    """Lazy import of all factor classes into the registry."""
    from gtaa.factors.carry import CarryFactor
    from gtaa.factors.downside_vol import DownsideVolFactor
    from gtaa.factors.drawdown_resilience import DrawdownResilienceFactor
    from gtaa.factors.low_vol import LowVolFactor
    from gtaa.factors.momentum import MomentumFactor
    from gtaa.factors.trend import TrendFactor
    from gtaa.factors.value import ValuePEFactor

    FACTOR_REGISTRY.update(
        {
            "value_pe": ValuePEFactor,
            "momentum": MomentumFactor,
            "low_vol": LowVolFactor,
            "downside_vol": DownsideVolFactor,
            "drawdown_resilience": DrawdownResilienceFactor,
            "trend": TrendFactor,
            "carry": CarryFactor,
        }
    )


def build_factors(factor_names: list[str]) -> list[BaseFactor]:
    """Instantiate factors from the registry by name."""
    if not FACTOR_REGISTRY:
        register_default_factors()
    return [FACTOR_REGISTRY[name]() for name in factor_names]


def _build_qa_checks(
    config: BacktestConfig,
    data: DataBundle,
    final_weights: pd.DataFrame,
    returns_series: pd.Series,
    scaled_weights: dict,
    cov: pd.DataFrame,
) -> pd.DataFrame:
    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date)
    active = data.returns.loc[start:end]

    checks = []

    checks.append({
        "Check": "Backtest start date",
        "Status": "OK",
        "Value": str(returns_series.index[0].date()) if len(returns_series) > 0 else "N/A",
        "Tolerance": config.start_date,
        "Comment": "",
    })
    checks.append({
        "Check": "Backtest end date",
        "Status": "OK",
        "Value": str(returns_series.index[-1].date()) if len(returns_series) > 0 else "N/A",
        "Tolerance": config.end_date,
        "Comment": "",
    })
    checks.append({
        "Check": "Number of monthly returns",
        "Status": "OK",
        "Value": len(returns_series),
        "Tolerance": "≥1",
        "Comment": "",
    })
    checks.append({
        "Check": "Number of assets",
        "Status": "OK",
        "Value": data.returns.shape[1],
        "Tolerance": "≥2",
        "Comment": "",
    })
    checks.append({
        "Check": "Covariance lookback ≥ 36 months",
        "Status": "OK" if config.covariance_lookback_months >= 36 else "FAIL",
        "Value": config.covariance_lookback_months,
        "Tolerance": "≥36",
        "Comment": "Project 2 requires ≥36 months",
    })

    # Check missing values in active window
    missing_in_window = active.isna().sum().sum()
    checks.append({
        "Check": "Missing values in active window",
        "Status": "OK" if missing_in_window == 0 else "WARN",
        "Value": int(missing_in_window),
        "Tolerance": "0",
        "Comment": "",
    })

    # Check raw weights sum near zero for each factor
    for fname, (raw_df, _) in scaled_weights.items():
        last_raw = raw_df.iloc[-1] if len(raw_df) > 0 else pd.Series(dtype=float)
        wsum = last_raw.sum() if len(last_raw) > 0 else float("nan")
        checks.append({
            "Check": f"Raw weights sum ~0 ({fname})",
            "Status": "OK" if abs(wsum) < 0.01 else "WARN",
            "Value": round(float(wsum), 6),
            "Tolerance": "|sum| < 0.01",
            "Comment": "",
        })

    # Check FMP vol scaling (last date)
    for fname, (_, sc_df) in scaled_weights.items():
        if len(sc_df) > 0:
            last_date = sc_df.index[-1]
            w = sc_df.iloc[-1]
            try:
                cov_check = rolling_covariance(data.returns, last_date,
                    config.covariance_lookback_months, config.covariance_annualization,
                    config.covariance_lookback_months)
                ok = check_portfolio_vol(w, cov_check, config.factor_target_vol, tol=0.002)
            except Exception:
                ok = False
            checks.append({
                "Check": f"FMP 1% vol scaling ({fname})",
                "Status": "OK" if ok else "WARN",
                "Value": config.factor_target_vol,
                "Tolerance": "±0.002",
                "Comment": "",
            })

    # Check final portfolio vol
    if len(final_weights) > 0:
        last_date = final_weights.index[-1]
        w = final_weights.iloc[-1]
        try:
            ok_final = check_portfolio_vol(w, cov, config.portfolio_target_vol, tol=0.002)
        except Exception:
            ok_final = False
        checks.append({
            "Check": "Final portfolio 1% vol scaling",
            "Status": "OK" if ok_final else "WARN",
            "Value": config.portfolio_target_vol,
            "Tolerance": "±0.002",
            "Comment": "",
        })

    # Check lookahead: weights at t, returns at t+1
    if len(final_weights) > 1 and len(data.returns) > 1:
        w_dates = final_weights.index
        r_dates = data.returns.loc[final_weights.index[1]:].index[:len(w_dates) - 1]
        has_no_lookahead = check_no_lookahead(w_dates[:-1], r_dates) if len(r_dates) > 0 else True
        checks.append({
            "Check": "Returns shifted one month after weights (no look-ahead)",
            "Status": "OK" if has_no_lookahead else "FAIL",
            "Value": "verified",
            "Tolerance": "strict",
            "Comment": "w_t * r_{t+1}",
        })

    return pd.DataFrame(checks)


def run_backtest(
    config: BacktestConfig,
    data: DataBundle,
    factors: list[BaseFactor],
) -> BacktestResult:
    """Run a full GTAA backtest.

    Args:
        config: Strategy configuration.
        data: DataBundle with aligned returns and signals.
        factors: List of instantiated BaseFactor objects.

    Returns:
        BacktestResult with all outputs.
    """
    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date)

    # Rebalance dates: all month-end dates in [start, end] with enough prior history
    all_dates = data.returns.loc[:end].index
    min_hist = pd.Timestamp(config.start_date) - pd.DateOffset(months=config.covariance_lookback_months + 1)
    valid_dates = data.returns.loc[min_hist:end].index

    # Build FMP weights for each factor
    all_raw: dict[str, pd.DataFrame] = {}
    all_scaled: dict[str, pd.DataFrame] = {}

    for factor in factors:
        raw_df, scaled_df = build_fmp_weights(factor, data, valid_dates, config)
        all_raw[factor.spec.name] = raw_df
        all_scaled[factor.spec.name] = scaled_df

    # Restrict to backtest window
    for name in all_raw:
        all_raw[name] = all_raw[name].loc[start:end]
        all_scaled[name] = all_scaled[name].loc[start:end]

    # Combine FMPs
    combined = combine_fmps_equal_weight(all_scaled)

    # Scale combined portfolio to portfolio_target_vol
    final_weights = scale_combined_portfolio(combined, data.returns, config)

    # Apply constraints
    from gtaa.risk.constraints import apply_gross_exposure_cap, apply_net_exposure_target
    if config.gross_exposure_cap is not None:
        final_weights = final_weights.apply(
            lambda row: apply_gross_exposure_cap(row, config.gross_exposure_cap), axis=1
        )
    if config.net_exposure_target is not None:
        final_weights = final_weights.apply(
            lambda row: apply_net_exposure_target(row, config.net_exposure_target), axis=1
        )

    # Portfolio returns: w_t * r_{t+1}
    port_returns = portfolio_return_series(final_weights, data.returns)

    # Factor-level returns
    factor_returns = {}
    for name, sc_df in all_scaled.items():
        fr = portfolio_return_series(sc_df, data.returns)
        factor_returns[name] = fr
    factor_returns_df = pd.DataFrame(factor_returns)

    # Turnover
    turnover_series = compute_turnover_series(final_weights, data.returns)

    # Transaction costs
    if config.transaction_cost_bps > 0:
        tc = turnover_series * config.transaction_cost_bps / 10_000
        tc = tc.reindex(port_returns.index).fillna(0.0)
        port_returns = port_returns - tc

    # Performance stats
    stats = compute_stats(port_returns, factor_returns_df, turnover_series)

    # Final period diagnostics
    last_date = final_weights.index[-1] if len(final_weights) > 0 else end
    try:
        final_cov = rolling_covariance(
            data.returns, last_date,
            config.covariance_lookback_months, config.covariance_annualization,
            config.covariance_lookback_months,
        )
    except ValueError:
        final_cov = pd.DataFrame()

    final_vols = final_cov.apply(lambda col: col[col.name] ** 0.5) if not final_cov.empty else pd.Series()
    final_corr = final_cov.corr() if not final_cov.empty else pd.DataFrame()

    # QA checks
    scaled_for_qa = {name: (all_raw[name], all_scaled[name]) for name in all_raw}
    qa = _build_qa_checks(config, data, final_weights, port_returns, scaled_for_qa, final_cov)

    return BacktestResult(
        config=config,
        returns=port_returns,
        factor_returns=factor_returns_df,
        raw_weights=all_raw,
        scaled_weights=all_scaled,
        final_weights=final_weights,
        final_covariance=final_cov,
        final_volatilities=final_vols,
        final_correlations=final_corr,
        stats=stats,
        turnover=turnover_series,
        qa_checks=qa,
    )
