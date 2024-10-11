"""Euclidean distance between regime vectors.

HW_GTAA_1: For each date t, compute the Euclidean distance between the
standardized macro vector at t and the standardized macro vector at a
reference date (the last observation in the expanding window, i.e. the
current date being evaluated).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def euclidean_distance_to_point(
    df: pd.DataFrame,
    reference: pd.Series,
) -> pd.Series:
    """Euclidean distance from every row of df to a reference vector.

    Args:
        df: DataFrame of standardized regime vectors (dates × variables).
        reference: The current standardized vector to compare against.

    Returns:
        pd.Series of distances indexed like df.
    """
    diff = df.sub(reference, axis=1).fillna(0.0)
    return np.sqrt((diff ** 2).sum(axis=1))


def euclidean_distances_to_reference(
    standardized: pd.DataFrame,
    asof_date,
) -> pd.Series:
    """Distances from every historical row to the row at asof_date.

    Only rows up to and including asof_date are considered (no lookahead).
    """
    history = standardized.loc[:asof_date].dropna(how="all")
    if asof_date not in history.index:
        raise ValueError(f"{asof_date} not found in standardized index")
    ref = history.loc[asof_date]
    return euclidean_distance_to_point(history, ref)


def normalized_distances(distances: pd.Series) -> pd.Series:
    """Normalize distances to [0, 1] by dividing by the maximum distance."""
    dmax = distances.max()
    if dmax == 0:
        return pd.Series(0.0, index=distances.index)
    return distances / dmax
