"""Formatted report tables for notebook display."""
from __future__ import annotations

import pandas as pd

from gtaa.models import BacktestResult


def format_stats_table(result: BacktestResult) -> pd.DataFrame:
    """Return a nicely formatted stats table."""
    stats = result.stats.copy()

    def _pct(x):
        return f"{x*100:.2f}%" if pd.notna(x) else "—"

    def _num(x, dec=2):
        return f"{x:.{dec}f}" if pd.notna(x) else "—"

    formatted = pd.DataFrame(index=stats.index)
    formatted["Ann Return"] = stats["Ann Return"].map(_pct)
    formatted["Ann Vol"] = stats["Ann Vol"].map(_pct)
    formatted["IR"] = stats["IR"].map(lambda x: _num(x, 3))
    formatted["Avg Drawdown"] = stats["Avg Drawdown"].map(_pct)
    formatted["Max Drawdown"] = stats["Max Drawdown"].map(_pct)
    formatted["Avg Turnover"] = stats["Avg Turnover"].map(_pct)
    return formatted


def compare_results(results: list[BacktestResult]) -> pd.DataFrame:
    """Side-by-side comparison of multiple BacktestResult objects."""
    rows = []
    for res in results:
        s = res.stats.loc["Portfolio"]
        rows.append({
            "Strategy": res.config.name,
            "Start": res.config.start_date,
            "End": res.config.end_date,
            "Ann Return": s.get("Ann Return"),
            "Ann Vol": s.get("Ann Vol"),
            "IR": s.get("IR"),
            "Avg Drawdown": s.get("Avg Drawdown"),
            "Max Drawdown": s.get("Max Drawdown"),
            "Avg Turnover": s.get("Avg Turnover"),
            "Best Month": float(res.returns.max()),
            "Worst Month": float(res.returns.min()),
        })
    return pd.DataFrame(rows).set_index("Strategy")
