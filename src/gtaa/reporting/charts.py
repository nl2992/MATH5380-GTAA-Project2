"""Charting utilities for notebooks and Excel export."""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from gtaa.analytics.performance import drawdowns, growth_of_one


def plot_growth(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame | None = None,
    title: str = "Growth of $1",
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Plot growth-of-$1 for portfolio and each FMP."""
    fig, ax = plt.subplots(figsize=figsize)

    if factor_returns is not None:
        for col in factor_returns.columns:
            fret = factor_returns[col].dropna()
            ax.plot(growth_of_one(fret), alpha=0.6, linestyle="--", label=col)

    ax.plot(growth_of_one(portfolio_returns), linewidth=2, color="navy", label="Portfolio")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("Cumulative Value ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_drawdown(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame | None = None,
    title: str = "Drawdown",
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """Plot drawdown series for portfolio and each FMP."""
    fig, ax = plt.subplots(figsize=figsize)

    if factor_returns is not None:
        for col in factor_returns.columns:
            fret = factor_returns[col].dropna()
            ax.fill_between(fret.index, drawdowns(fret) * 100, 0, alpha=0.25, label=col)

    dd = drawdowns(portfolio_returns) * 100
    ax.fill_between(dd.index, dd, 0, alpha=0.7, color="darkred", label="Portfolio")
    ax.plot(dd, color="darkred", linewidth=1)
    ax.set_title(title)
    ax.set_ylabel("Drawdown (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_weights(
    weights: pd.DataFrame,
    title: str = "Portfolio Weights",
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """Stacked area chart of portfolio weights over time."""
    fig, ax = plt.subplots(figsize=figsize)
    pos = weights.clip(lower=0)
    neg = weights.clip(upper=0)
    ax.stackplot(weights.index, pos.T, labels=pos.columns, alpha=0.7)
    ax.stackplot(weights.index, neg.T, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("Weight")
    ax.legend(loc="upper left", fontsize=7, ncol=5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_turnover(turnover: pd.Series, figsize: tuple = (12, 3)) -> plt.Figure:
    """Bar chart of monthly one-way turnover."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(turnover.index, turnover * 100, width=20, color="steelblue", alpha=0.7)
    ax.axhline(turnover.mean() * 100, color="red", linestyle="--", linewidth=1, label=f"Avg {turnover.mean()*100:.1f}%")
    ax.set_title("Monthly One-Way Turnover")
    ax.set_ylabel("Turnover (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
