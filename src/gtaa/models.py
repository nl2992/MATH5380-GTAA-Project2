"""Shared data model contracts used across the GTAA package."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class DataBundle:
    """All aligned data for a GTAA backtest.

    All DataFrames are indexed by month-end timestamps.
    Columns are asset identifiers.
    """

    returns: pd.DataFrame
    prices: Optional[pd.DataFrame] = None
    valuations: Optional[pd.DataFrame] = None
    macro: Optional[pd.DataFrame] = None
    asset_meta: Optional[pd.DataFrame] = None


@dataclass
class FactorSpec:
    """Metadata contract for every factor implementation."""

    name: str
    signal_type: str
    ascending_rank: bool
    target_vol: float = 0.01


@dataclass
class BacktestConfig:
    """All knobs available to a backtest run.

    Modify this object; never touch function internals.
    """

    name: str = "base"
    start_date: str = "2006-01-31"
    end_date: str = "2025-12-31"
    rebalance_frequency: str = "M"
    covariance_lookback_months: int = 36
    covariance_annualization: int = 12
    covariance_estimator: str = "population"
    factor_target_vol: float = 0.01
    portfolio_target_vol: float = 0.01
    factor_combination: str = "equal_weight"
    transaction_cost_bps: float = 0.0
    allow_short: bool = True
    gross_exposure_cap: Optional[float] = None
    net_exposure_target: Optional[float] = 0.0


@dataclass
class BacktestResult:
    """Container for all backtest outputs."""

    config: BacktestConfig
    returns: pd.Series
    factor_returns: pd.DataFrame
    raw_weights: dict
    scaled_weights: dict
    final_weights: pd.DataFrame
    final_covariance: pd.DataFrame
    final_volatilities: pd.Series
    final_correlations: pd.DataFrame
    stats: pd.DataFrame
    turnover: pd.Series
    qa_checks: pd.DataFrame
