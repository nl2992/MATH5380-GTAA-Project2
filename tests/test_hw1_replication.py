"""Priority 2: HW1 regime engine unit tests (no dependency on Excel file)."""
import numpy as np
import pandas as pd
import pytest

from gtaa.regime.distances import euclidean_distances_to_reference, normalized_distances
from gtaa.regime.kernels import normalize_weights, polynomial_kernel
from gtaa.regime.standardization import cpi_yoy, expanding_zscore


def _dummy_macro(n=50, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1993-01-31", periods=n, freq="ME")
    return pd.DataFrame(
        {"lei": rng.normal(0, 0.2, n), "lei_mom": rng.normal(0, 0.05, n), "cpi_yoy": rng.normal(0.03, 0.01, n)},
        index=dates,
    )


def test_expanding_zscore_mean_zero():
    """Expanding z-scores should converge to zero mean as series grows."""
    rng = np.random.default_rng(1)
    s = pd.Series(rng.normal(5, 2, 100))
    z = expanding_zscore(s)
    # After sufficient history, the long-run average z-score should be near zero
    assert abs(z.iloc[50:].mean()) < 0.15


def test_expanding_zscore_early_nan():
    """With min_periods=2, the first observation should be NaN."""
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    z = expanding_zscore(s, min_periods=2)
    assert np.isnan(z.iloc[0])


def test_cpi_yoy_matches_formula():
    """CPI YoY = CPI_t / CPI_{t-12} - 1."""
    cpi = pd.Series([100.0] * 24)
    yoy = cpi_yoy(cpi)
    assert (yoy.iloc[12:].abs() < 1e-12).all()


def test_euclidean_distances_self_is_zero():
    """Distance from a point to itself must be zero."""
    macro = _dummy_macro()
    std = macro.apply(expanding_zscore)
    asof = std.index[-1]
    distances = euclidean_distances_to_reference(std, asof)
    assert abs(distances[asof]) < 1e-10


def test_normalized_distances_in_zero_one():
    """Normalized distances must all be in [0, 1]."""
    macro = _dummy_macro()
    std = macro.apply(expanding_zscore).dropna()
    asof = std.index[-1]
    d = euclidean_distances_to_reference(std, asof)
    nd = normalized_distances(d)
    assert (nd >= 0).all() and (nd <= 1 + 1e-10).all()


def test_polynomial_kernel_range():
    """Kernel weights should be in [0, 1] for normalized distances."""
    nd = pd.Series(np.linspace(0, 1, 20))
    kw = polynomial_kernel(nd, degree=3)
    assert (kw >= 0).all() and (kw <= 1 + 1e-10).all()


def test_kernel_weight_at_zero_distance_is_one():
    """A point at zero distance should receive the maximum kernel weight."""
    nd = pd.Series([0.0, 0.5, 1.0])
    kw = polynomial_kernel(nd, degree=3)
    assert abs(kw.iloc[0] - 1.0) < 1e-10


def test_normalize_weights_sums_to_one():
    """After normalization, kernel weights should sum to 1."""
    nd = pd.Series(np.linspace(0, 0.8, 10))
    kw = polynomial_kernel(nd, degree=3)
    nw = normalize_weights(kw)
    assert abs(nw.sum() - 1.0) < 1e-10
