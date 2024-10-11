"""Return attribution helpers."""
from __future__ import annotations

import pandas as pd


def factor_attribution(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Compute correlation and beta of portfolio to each factor."""
    result = {}
    for col in factor_returns.columns:
        aligned = factor_returns[col].dropna().align(portfolio_returns, join="inner")
        f, p = aligned
        corr = float(f.corr(p))
        cov = float(f.cov(p))
        var = float(f.var(ddof=0))
        beta = cov / var if var > 0 else float("nan")
        result[col] = {"correlation": corr, "beta": beta}
    return pd.DataFrame(result).T
