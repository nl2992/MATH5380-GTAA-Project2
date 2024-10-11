"""Priority 1: verify weights at t are multiplied by returns at t+1 (no look-ahead)."""
import numpy as np
import pandas as pd
import pytest

from gtaa.data.alignment import align_weights_to_next_returns, portfolio_return_series


def _make_monthly_data(n=24, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-31", periods=n, freq="ME")
    assets = ["A", "B", "C"]
    ret = pd.DataFrame(rng.normal(0, 0.05, (n, len(assets))), index=dates, columns=assets)
    wts = pd.DataFrame(rng.normal(0, 0.1, (n, len(assets))), index=dates, columns=assets)
    wts = wts.sub(wts.mean(axis=1), axis=0)  # demean rows
    return wts, ret


def test_aligned_weights_lead_returns_by_one_period():
    """Weight values formed at t must pair with returns at t+1.

    After alignment, both DataFrames share the same date index (the return
    dates).  The weight at index date d holds the value formed at date d-1,
    i.e. the shift-by-one has already been applied.  We verify this by
    checking that w_aligned.iloc[k] equals wts.iloc[k] (original row k
    now lives at the next date).
    """
    wts, ret = _make_monthly_data(n=24)
    w_aligned, r_aligned = align_weights_to_next_returns(wts, ret)

    assert len(w_aligned) == len(r_aligned), "Aligned series must have same length"
    # n-1 pairs: first weight date has no preceding weight, last return date has no subsequent weight
    assert len(w_aligned) == len(wts) - 1

    # w_aligned.iloc[0] should contain the values from wts.iloc[0] (first weight, now at dates[1])
    pd.testing.assert_series_equal(
        w_aligned.iloc[0].reset_index(drop=True),
        wts.iloc[0].reset_index(drop=True),
        check_names=False,
    )


def test_portfolio_return_first_date_uses_previous_weights():
    """The portfolio return at date d uses weights from date d-1.

    After alignment, the first return in port_ret is at dates[1],
    and it uses w at dates[0] paired with r at dates[1].
    """
    wts, ret = _make_monthly_data(n=6)
    port_ret = portfolio_return_series(wts, ret)

    # First portfolio return is at ret.index[1], using wts.iloc[0]
    expected_first = float((wts.iloc[0] * ret.iloc[1]).sum())
    actual_first = float(port_ret.iloc[0])

    assert abs(actual_first - expected_first) < 1e-10, (
        f"Expected {expected_first:.8f}, got {actual_first:.8f}"
    )


def test_no_lookahead_flag():
    """check_no_lookahead returns True when weight dates < return dates."""
    from gtaa.data.validation import check_no_lookahead

    w_dates = pd.date_range("2010-01-31", periods=10, freq="ME")
    r_dates = pd.date_range("2010-02-28", periods=10, freq="ME")
    assert check_no_lookahead(w_dates, r_dates)


def test_lookahead_detected():
    """check_no_lookahead returns False when weight and return dates are equal."""
    from gtaa.data.validation import check_no_lookahead

    dates = pd.date_range("2010-01-31", periods=10, freq="ME")
    assert not check_no_lookahead(dates, dates)
