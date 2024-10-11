"""Utilities for comparing computed series against Excel reference values."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compare_series(
    computed: pd.Series,
    reference: pd.Series,
    name: str = "",
    tol: float = 1e-6,
) -> pd.DataFrame:
    """Compare two series element-wise and return a tolerance report.

    Args:
        computed: Our computed values.
        reference: Excel / reference values.
        name: Label for the report.
        tol: Absolute tolerance for 'matches' flag.

    Returns:
        DataFrame with columns: computed, reference, abs_diff, matches.
    """
    c, r = computed.align(reference, join="inner")
    abs_diff = (c - r).abs()
    matches = abs_diff <= tol

    report = pd.DataFrame(
        {
            "computed": c,
            "reference": r,
            "abs_diff": abs_diff,
            "matches": matches,
        }
    )
    n_ok = matches.sum()
    n_total = len(matches)
    pct = 100 * n_ok / n_total if n_total > 0 else 0.0
    max_err = float(abs_diff.max()) if n_total > 0 else 0.0

    report.attrs["summary"] = (
        f"{name}: {n_ok}/{n_total} rows match within tol={tol:.2e}. "
        f"Max abs error = {max_err:.2e} ({pct:.1f}% pass)"
    )
    return report


def tolerance_report(reports: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Aggregate summary of multiple compare_series reports."""
    rows = []
    for name, rep in reports.items():
        matches = rep["matches"]
        abs_diff = rep["abs_diff"]
        rows.append({
            "Check": name,
            "N": len(matches),
            "Pass": int(matches.sum()),
            "Fail": int((~matches).sum()),
            "MaxAbsErr": float(abs_diff.max()),
            "MeanAbsErr": float(abs_diff.mean()),
            "Status": "OK" if matches.all() else "FAIL",
        })
    return pd.DataFrame(rows).set_index("Check")
