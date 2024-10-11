"""Performance statistics.

Required by Project 2:
- Cumulative gross returns (growth of $1)
- Arithmetic annualized simple return
- Annualized volatility of simple returns
- Annualized information ratio (vs zero active-return benchmark)
- Average drawdown
- Maximum drawdown
- Turnover
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def growth_of_one(returns: pd.Series) -> pd.Series:
    """Cumulative compounded return starting from $1."""
    return (1 + returns).cumprod()


def annualized_arithmetic_return(returns: pd.Series) -> float:
    """12 × arithmetic mean of monthly simple returns."""
    return float(12 * returns.mean())


def annualized_volatility(returns: pd.Series) -> float:
    """Annualized standard deviation of monthly simple returns (ddof=0)."""
    return float(returns.std(ddof=0) * np.sqrt(12))


def information_ratio(returns: pd.Series) -> float:
    """Annualized return divided by annualized volatility (vs zero benchmark)."""
    vol = annualized_volatility(returns)
    return annualized_arithmetic_return(returns) / vol if vol > 0 else float("nan")


def drawdowns(returns: pd.Series) -> pd.Series:
    """Drawdown series: gross / running-max - 1."""
    gross = growth_of_one(returns)
    hwm = gross.cummax()
    return gross / hwm - 1


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown (most negative value in the drawdown series)."""
    return float(drawdowns(returns).min())


def avg_drawdown(returns: pd.Series) -> float:
    """Average of all negative drawdown values."""
    dd = drawdowns(returns)
    neg = dd[dd < 0]
    return float(neg.mean()) if len(neg) > 0 else 0.0


def compute_stats(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame | None = None,
    turnover_series: pd.Series | None = None,
) -> pd.DataFrame:
    """Build the Project 2 required statistics table.

    Rows: portfolio + each factor.
    Columns: Ann Return, Ann Vol, IR, Avg DD, Max DD, Avg Turnover.
    """
    records = []

    def _row(name, rets, to=None):
        r = {
            "Name": name,
            "Ann Return": annualized_arithmetic_return(rets),
            "Ann Vol": annualized_volatility(rets),
            "IR": information_ratio(rets),
            "Avg Drawdown": avg_drawdown(rets),
            "Max Drawdown": max_drawdown(rets),
            "Avg Turnover": float(to.mean()) if to is not None and len(to) > 0 else float("nan"),
        }
        return r

    records.append(_row("Portfolio", portfolio_returns, turnover_series))

    if factor_returns is not None:
        for col in factor_returns.columns:
            records.append(_row(col, factor_returns[col].dropna()))

    stats_df = pd.DataFrame(records).set_index("Name")
    return stats_df
