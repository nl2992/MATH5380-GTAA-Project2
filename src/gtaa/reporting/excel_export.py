"""Excel export – generates the Project 2 required workbook."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gtaa.analytics.performance import growth_of_one
from gtaa.models import BacktestResult


def _fmt_pct(x):
    return f"{x*100:.2f}%" if pd.notna(x) else ""


def export_project2_excel(result: BacktestResult, output_path: str | Path) -> None:
    """Write all Project 2 required sheets to an Excel workbook.

    Required sheets:
        Gross Returns, Portfolio Statistics, Raw Factor Weights,
        Factor1 1% Volatility Weights, Factor2 1% Volatility Weights,
        Final 1% Volatility Portfolio Weights,
        Covariances, Volatilities, Correlations

    Additional sheets:
        ReadMe, Data Audit, Turnover, QA Checks
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:

        # ── ReadMe ──────────────────────────────────────────────────────────
        readme_data = {
            "Sheet": [
                "Gross Returns", "Portfolio Statistics", "Raw Factor Weights",
                "Factor1 1% Volatility Weights", "Factor2 1% Volatility Weights",
                "Final 1% Volatility Portfolio Weights",
                "Covariances", "Volatilities", "Correlations",
                "Turnover", "QA Checks",
            ],
            "Description": [
                "Growth of $1 for each FMP and combined portfolio",
                "Ann return, vol, IR, avg/max drawdown, turnover",
                "Raw cross-sectional rank weights before vol scaling",
                "1%-scaled weights for Factor 1 (value_pe)",
                "1%-scaled weights for Factor 2 (low_vol)",
                "Final combined portfolio weights scaled to 1% vol",
                "Annualized covariance matrix (final period)",
                "Asset annualized volatilities (final period)",
                "Asset correlation matrix (final period)",
                "Monthly one-way turnover series",
                "QA validation checks",
            ],
        }
        pd.DataFrame(readme_data).to_excel(writer, sheet_name="ReadMe", index=False)

        # ── Gross Returns ────────────────────────────────────────────────────
        gross = pd.DataFrame({"Portfolio": growth_of_one(result.returns)})
        for col in result.factor_returns.columns:
            fret = result.factor_returns[col].dropna()
            gross[col] = growth_of_one(fret)
        gross.to_excel(writer, sheet_name="Gross Returns")

        # ── Portfolio Statistics ─────────────────────────────────────────────
        stats = result.stats.copy()
        pct_cols = ["Ann Return", "Ann Vol", "Avg Drawdown", "Max Drawdown", "Avg Turnover"]
        for col in pct_cols:
            if col in stats.columns:
                stats[col] = stats[col].map(lambda x: round(float(x), 6) if pd.notna(x) else None)
        stats.to_excel(writer, sheet_name="Portfolio Statistics")

        # ── Raw Factor Weights ───────────────────────────────────────────────
        factor_names = list(result.raw_weights.keys())
        for i, fname in enumerate(factor_names):
            raw_df = result.raw_weights[fname]
            raw_df.to_excel(writer, sheet_name=f"Raw {fname[:20]}")

        # ── Factor 1% Volatility Weights ─────────────────────────────────────
        for i, fname in enumerate(factor_names):
            sc_df = result.scaled_weights[fname]
            label = f"Factor{i+1} 1% Volatility Weights"
            sc_df.to_excel(writer, sheet_name=label[:31])

        # ── Final Portfolio Weights ───────────────────────────────────────────
        result.final_weights.to_excel(writer, sheet_name="Final 1% Volatility Portfolio Weights"[:31])

        # ── Covariances ──────────────────────────────────────────────────────
        if not result.final_covariance.empty:
            result.final_covariance.to_excel(writer, sheet_name="Covariances")

        # ── Volatilities ─────────────────────────────────────────────────────
        if not result.final_volatilities.empty:
            result.final_volatilities.to_frame("Ann Vol").to_excel(writer, sheet_name="Volatilities")

        # ── Correlations ─────────────────────────────────────────────────────
        if not result.final_correlations.empty:
            result.final_correlations.to_excel(writer, sheet_name="Correlations")

        # ── Turnover ─────────────────────────────────────────────────────────
        if len(result.turnover) > 0:
            result.turnover.to_frame("Turnover").to_excel(writer, sheet_name="Turnover")

        # ── QA Checks ────────────────────────────────────────────────────────
        result.qa_checks.to_excel(writer, sheet_name="QA Checks", index=False)

    print(f"Exported: {output_path}")
