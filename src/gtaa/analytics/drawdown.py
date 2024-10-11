"""Drawdown utilities (thin wrappers, main logic in performance.py)."""
from __future__ import annotations

import pandas as pd

from gtaa.analytics.performance import drawdowns, max_drawdown, avg_drawdown


def drawdown_periods(returns: pd.Series) -> pd.DataFrame:
    """Return a DataFrame of drawdown start, trough, end dates and depths."""
    dd = drawdowns(returns)
    in_dd = dd < 0
    periods = []
    start = None

    for date, is_dd in in_dd.items():
        if is_dd and start is None:
            start = date
        elif not is_dd and start is not None:
            window = dd.loc[start:date]
            trough_date = window.idxmin()
            periods.append({
                "start": start,
                "trough": trough_date,
                "end": date,
                "depth": float(window.min()),
                "duration_months": len(window),
            })
            start = None

    return pd.DataFrame(periods)
