"""Build the expanded no-look-ahead audit notebook.

Produces notebooks/09_no_lookahead_verification.ipynb with an expanded-strategy
audit covering:
  - return-based momentum timing
  - price-based momentum timing
  - 36-month covariance window integrity
  - carry pass-through verification
  - final portfolio weight-to-return alignment
  - future-shock contamination test on family/final weights
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf


nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.lstrip("\n")))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src.lstrip("\n")))


md(
    """
    # 09 — No-Look-Ahead Verification

    This notebook audits the **expanded six-sleeve GTAA submission** and shows
    that the final reported strategy is free of look-ahead bias.

    | # | Test | What it proves |
    |---|------|----------------|
    | 1 | Return-based momentum timing | Equity / FI momentum at date *t* uses returns *t-12 … t-2* only |
    | 2 | Price-based momentum timing | Commodity / FX momentum at date *t* uses prices *t-13* and *t-2* only |
    | 3 | Covariance window integrity | The 36-month covariance at *t* ends strictly at *t* and excludes *t+1* |
    | 4 | Carry pass-through | Expanded carry signals use contemporaneous signal sheets only |
    | 5 | Final portfolio alignment | Final portfolio weight at *t* earns return at *t+1* |
    | 6 | Future-shock contamination test | Shocking source data at *t+1* leaves family and final weights at *t* unchanged |
    """
)

code(
    """
    import sys
    import warnings
    from copy import deepcopy
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import yaml

    warnings.filterwarnings("ignore")

    ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
    sys.path.insert(0, str(ROOT / "src"))

    from gtaa.io.excel_loader import load_gtaa_expanded_data
    from gtaa.factors.signals import (
        momentum_12_1_from_returns,
        momentum_12_1_from_prices,
        fixed_income_carry_signal,
        fx_carry_signal,
    )
    from gtaa.factors.ranking import rank_standardized_weights
    from gtaa.risk.covariance import rolling_covariance_partial
    from gtaa.risk.scaling import scale_weight_panel
    from gtaa.portfolio.fmp import combine_weight_panels
    from gtaa.portfolio.returns import portfolio_returns_from_weights

    CFG_PATH = ROOT / "config" / "project2_gtaa_expanded.yaml"
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    DATA_PATH = ROOT / cfg["data"]["workbook"]
    data = load_gtaa_expanded_data(DATA_PATH)

    LOOKBACK = cfg["risk"]["covariance_lookback_months"]
    SLEEVE_VOL = cfg["risk"]["sleeve_target_vol"]
    FAMILY_VOL = cfg["risk"]["factor_family_target_vol"]
    FINAL_VOL = cfg["risk"]["final_target_vol"]

    SLEEVE_TICKERS = {
        "equity_momentum": list(cfg["sleeves"]["equity_momentum"]["assets"].keys()),
        "commodity_momentum": list(cfg["sleeves"]["commodity_momentum"]["assets"].keys()),
        "fixed_income_momentum": list(cfg["sleeves"]["fixed_income_momentum"]["assets"].keys()),
        "fx_momentum": list(cfg["sleeves"]["fx_momentum"]["assets"].keys()),
        "fixed_income_carry": list(cfg["sleeves"]["fixed_income_carry"]["assets"].keys()),
        "fx_carry": list(cfg["sleeves"]["fx_carry"]["assets"].keys()),
    }

    print("Expanded strategy data loaded.")
    print(f"  Equity returns:    {data.equity_returns.shape}  {data.equity_returns.index[0].date()} → {data.equity_returns.index[-1].date()}")
    print(f"  Commodity prices:  {data.commodity_prices.shape}  {data.commodity_prices.index[0].date()} → {data.commodity_prices.index[-1].date()}")
    print(f"  FI returns:        {data.fi_returns.shape}  {data.fi_returns.index[0].date()} → {data.fi_returns.index[-1].date()}")
    print(f"  FX prices:         {data.fx_prices.shape}  {data.fx_prices.index[0].date()} → {data.fx_prices.index[-1].date()}")
    print(f"  FI carry:          {data.fi_carry.shape}  {data.fi_carry.index[0].date()} → {data.fi_carry.index[-1].date()}")
    print(f"  FX carry:          {data.fx_carry.shape}  {data.fx_carry.index[0].date()} → {data.fx_carry.index[-1].date()}")
    """
)

code(
    """
    def combined_returns_from_bundle(bundle):
        all_assets = sorted(
            set(bundle.equity_returns.columns)
            | set(bundle.commodity_returns.columns)
            | set(bundle.fi_returns.columns)
            | set(bundle.fx_returns.columns)
        )
        return (
            pd.concat(
                [bundle.equity_returns, bundle.commodity_returns, bundle.fi_returns, bundle.fx_returns],
                axis=1,
            )
            .loc[:, lambda df: ~df.columns.duplicated()]
            .reindex(columns=all_assets)
            .sort_index()
        )


    def build_expanded_pipeline(bundle):
        signals = {
            "equity_momentum": momentum_12_1_from_returns(
                bundle.equity_returns[SLEEVE_TICKERS["equity_momentum"]]
            ),
            "commodity_momentum": momentum_12_1_from_prices(
                bundle.commodity_prices[SLEEVE_TICKERS["commodity_momentum"]]
            ),
            "fixed_income_momentum": momentum_12_1_from_returns(
                bundle.fi_returns[SLEEVE_TICKERS["fixed_income_momentum"]]
            ),
            "fx_momentum": momentum_12_1_from_prices(
                bundle.fx_prices[SLEEVE_TICKERS["fx_momentum"]]
            ),
            "fixed_income_carry": fixed_income_carry_signal(
                bundle.fi_carry[SLEEVE_TICKERS["fixed_income_carry"]]
            ),
            "fx_carry": fx_carry_signal(
                bundle.fx_carry[SLEEVE_TICKERS["fx_carry"]]
            ),
        }

        raw_weights = {name: rank_standardized_weights(sig) for name, sig in signals.items()}

        sleeve_returns_for_cov = {
            "equity_momentum": bundle.equity_returns[SLEEVE_TICKERS["equity_momentum"]],
            "commodity_momentum": bundle.commodity_returns[SLEEVE_TICKERS["commodity_momentum"]],
            "fixed_income_momentum": bundle.fi_returns[SLEEVE_TICKERS["fixed_income_momentum"]],
            "fx_momentum": bundle.fx_returns[SLEEVE_TICKERS["fx_momentum"]],
            "fixed_income_carry": bundle.fi_returns[SLEEVE_TICKERS["fixed_income_carry"]],
            "fx_carry": bundle.fx_returns[SLEEVE_TICKERS["fx_carry"]],
        }

        sleeve_scaled = {
            name: scale_weight_panel(raw_weights[name], sleeve_returns_for_cov[name], target_vol=SLEEVE_VOL, lookback=LOOKBACK)
            for name in raw_weights
        }

        combined_returns = combined_returns_from_bundle(bundle)

        mom_pre = combine_weight_panels(
            {k: sleeve_scaled[k] for k in cfg["factor_families"]["cross_asset_momentum"]},
            cfg["factor_families"]["cross_asset_momentum"],
        )
        carry_pre = combine_weight_panels(
            {k: sleeve_scaled[k] for k in cfg["factor_families"]["cross_asset_carry"]},
            cfg["factor_families"]["cross_asset_carry"],
        )

        momentum_family = scale_weight_panel(
            mom_pre, combined_returns.reindex(columns=mom_pre.columns), target_vol=FAMILY_VOL, lookback=LOOKBACK
        )
        carry_family = scale_weight_panel(
            carry_pre, combined_returns.reindex(columns=carry_pre.columns), target_vol=FAMILY_VOL, lookback=LOOKBACK
        )

        final_pre = combine_weight_panels(
            {
                "cross_asset_momentum": momentum_family,
                "cross_asset_carry": carry_family,
            },
            cfg["allocation"]["base_case"],
        )
        final_weights = scale_weight_panel(
            final_pre, combined_returns.reindex(columns=final_pre.columns), target_vol=FINAL_VOL, lookback=LOOKBACK
        )
        final_returns = portfolio_returns_from_weights(final_weights, combined_returns.reindex(columns=final_weights.columns))

        return {
            "signals": signals,
            "raw_weights": raw_weights,
            "sleeve_scaled": sleeve_scaled,
            "momentum_family": momentum_family,
            "carry_family": carry_family,
            "final_weights": final_weights,
            "combined_returns": combined_returns,
            "final_returns": final_returns,
        }


    pipeline = build_expanded_pipeline(data)
    live_weights = pipeline["final_weights"].dropna(how="all")
    print(f"Expanded final live weights: {live_weights.index[0].date()} → {live_weights.index[-1].date()}  ({len(live_weights)} rows)")
    print(f"Expanded final realized returns: {pipeline['final_returns'].index[0].date()} → {pipeline['final_returns'].index[-1].date()}  ({len(pipeline['final_returns'])} months)")
    """
)

md(
    """
    ## Test 1 — Return-Based Momentum Uses *t-12 … t-2* Only

    The return-based momentum signal for equity and fixed-income sleeves is:

    ```python
    gross.shift(2).rolling(11).apply(lambda x: x.prod() - 1, raw=True)
    ```

    So at date *t*, the signal uses the 11 returns from *t-12* through *t-2*,
    and explicitly excludes both *t-1* and *t*.
    """
)

code(
    """
    PROBE = pd.Timestamp("2018-01-31")

    eq_ret = data.equity_returns[SLEEVE_TICKERS["equity_momentum"]]
    eq_signal = momentum_12_1_from_returns(eq_ret)

    probe_loc = eq_ret.index.get_loc(PROBE)
    first_pos = probe_loc - 2 - 10
    last_pos = probe_loc - 2
    first_date = eq_ret.index[first_pos]
    last_date = eq_ret.index[last_pos]
    t_minus_1 = eq_ret.index[probe_loc - 1]

    asset = eq_ret.columns[0]
    manual_window = eq_ret.loc[first_date:last_date, asset]
    manual_signal = float((1 + manual_window).prod() - 1)
    api_signal = float(eq_signal.loc[PROBE, asset])

    print(f"Probe date (t):        {PROBE.date()}")
    print(f"Return window used:    {first_date.date()} → {last_date.date()}")
    print(f"Excluded t-1 date:     {t_minus_1.date()}")
    print(f"Asset:                 {asset}")
    print(f"Manual cumulative ret: {manual_signal:.8f}")
    print(f"API signal at t:       {api_signal:.8f}")
    print(f"Difference:            {abs(manual_signal - api_signal):.2e}")

    assert last_date == pd.Timestamp("2017-11-30")
    assert t_minus_1 == pd.Timestamp("2017-12-31")
    assert abs(manual_signal - api_signal) < 1e-12
    print("✓ Return-based momentum at t uses only t-12 … t-2.")
    """
)

md(
    """
    ## Test 2 — Price-Based Momentum Uses *P<sub>t-2</sub>* and *P<sub>t-13</sub>* Only

    The commodity and FX momentum sleeves use:

    ```python
    prices.shift(2) / prices.shift(13) - 1
    ```

    At date *t*, the signal is therefore built from *P<sub>t-2</sub>* and
    *P<sub>t-13</sub>* only.
    """
)

code(
    """
    PROBE_COM = pd.Timestamp("2018-01-31")

    com_prices = data.commodity_prices[SLEEVE_TICKERS["commodity_momentum"]]
    com_signal = momentum_12_1_from_prices(com_prices)
    asset = com_prices.columns[0]

    idx = com_prices.index.get_loc(PROBE_COM)
    t2_date = com_prices.index[idx - 2]
    t13_date = com_prices.index[idx - 13]
    p_t2 = float(com_prices.loc[t2_date, asset])
    p_t13 = float(com_prices.loc[t13_date, asset])
    manual_signal = p_t2 / p_t13 - 1.0
    api_signal = float(com_signal.loc[PROBE_COM, asset])

    print(f"Probe date (t):        {PROBE_COM.date()}")
    print(f"P_(t-2) date/value:    {t2_date.date()}  {p_t2:.4f}")
    print(f"P_(t-13) date/value:   {t13_date.date()}  {p_t13:.4f}")
    print(f"Manual price momentum: {manual_signal:.8f}")
    print(f"API signal at t:       {api_signal:.8f}")
    print(f"Difference:            {abs(manual_signal - api_signal):.2e}")

    assert abs(manual_signal - api_signal) < 1e-12
    print("✓ Price-based momentum at t uses only P_(t-2) and P_(t-13).")
    """
)

md(
    """
    ## Test 3 — Expanded Covariance Window Ends at *t*

    The expanded strategy uses a 36-month population covariance matrix built
    from the combined cross-asset return panel:

    ```python
    returns.loc[:t].tail(36).cov(ddof=0) * 12
    ```

    So month *t+1* must not affect the covariance used at *t*.
    """
)

code(
    """
    PROBE_COV = pd.Timestamp("2020-01-31")
    combined_returns = pipeline["combined_returns"]

    window = combined_returns.loc[:PROBE_COV].tail(LOOKBACK)
    t_plus_1 = combined_returns.index[combined_returns.index > PROBE_COV][0]

    print(f"Probe date (t):      {PROBE_COV.date()}")
    print(f"Window length:       {len(window)}")
    print(f"Earliest row:        {window.index[0].date()}")
    print(f"Latest row:          {window.index[-1].date()}")
    print(f"First excluded date: {t_plus_1.date()}")

    cov_original = rolling_covariance_partial(combined_returns, PROBE_COV, lookback=LOOKBACK)
    combined_shock = combined_returns.copy()
    combined_shock.loc[t_plus_1] *= 100.0
    cov_shocked = rolling_covariance_partial(combined_shock, PROBE_COV, lookback=LOOKBACK)

    max_diff = (cov_original - cov_shocked).abs().max().max()
    print(f"Max covariance element change after shocking t+1 returns: {max_diff:.2e}")

    assert len(window) == LOOKBACK
    assert window.index[-1] == PROBE_COV
    assert max_diff == 0.0
    print("✓ Expanded covariance at t excludes all information from t+1.")
    """
)

md(
    """
    ## Test 4 — Expanded Carry Signal Is a Pass-Through at *t*

    The expanded submission includes FX carry as a second carry sleeve. The
    signal function is a pass-through of the `fx_carry_signals` sheet:

    ```python
    fx_carry_signal(fx_carry) = fx_carry.copy()
    ```
    """
)

code(
    """
    TRACE_FX = pd.Timestamp("2020-01-31")

    fx_carry_raw = data.fx_carry[SLEEVE_TICKERS["fx_carry"]]
    fx_signal = fx_carry_signal(fx_carry_raw)

    diff = (fx_signal.loc[TRACE_FX] - fx_carry_raw.loc[TRACE_FX]).abs().max()
    print(f"FX carry probe date: {TRACE_FX.date()}")
    print("FX carry values at t:")
    print(fx_signal.loc[TRACE_FX].round(4).to_string())
    print(f"Max |signal - raw| at t: {diff:.2e}")

    assert diff < 1e-12
    print("✓ FX carry at t is a contemporaneous pass-through signal.")
    """
)

md(
    """
    ## Test 5 — Final Portfolio Weight at *t* Earns Return at *t+1*

    The final reported strategy uses the same no-look-ahead return alignment as
    the sleeve-level portfolios:

    ```python
    portfolio_returns_from_weights(weights, returns)
    ```

    which internally applies `returns.shift(-1)` before multiplying by weights.
    """
)

code(
    """
    final_weights = pipeline["final_weights"].dropna(how="all")
    final_returns = pipeline["final_returns"].dropna()
    combined_returns = pipeline["combined_returns"].reindex(columns=final_weights.columns)

    probe_weight_date = final_returns.index[60]
    next_date = combined_returns.index[combined_returns.index > probe_weight_date][0]

    manual_return = float(
        (
            final_weights.loc[probe_weight_date].fillna(0.0)
            * combined_returns.loc[next_date].reindex(final_weights.columns).fillna(0.0)
        ).sum()
    )
    api_return = float(final_returns.loc[probe_weight_date])

    print(f"Weight date (t):      {probe_weight_date.date()}")
    print(f"Return date (t+1):    {next_date.date()}")
    print(f"Manual Σ w_i(t)r_i(t+1): {manual_return:.8f}")
    print(f"API portfolio return:    {api_return:.8f}")
    print(f"Difference:              {abs(manual_return - api_return):.2e}")

    last_weight_date = final_weights.index[-1]
    last_return_date = final_returns.index[-1]
    print(f"Last weight date:     {last_weight_date.date()}")
    print(f"Last realized return: {last_return_date.date()}")

    assert abs(manual_return - api_return) < 1e-12
    assert last_weight_date > last_return_date
    print("✓ Final portfolio weight at t earns the return realized at t+1.")
    """
)

md(
    """
    ## Test 6 — Future-Shock Contamination Test on Expanded Family and Final Weights

    We now shock every relevant source panel at *t+1* and rebuild the expanded
    pipeline. If there is no look-ahead bias, then the family weights and final
    portfolio weights formed at *t* must be unchanged.
    """
)

code(
    """
    live_dates = final_weights.index
    PROBE_T = live_dates[len(live_dates) // 2]
    FUTURE_DATE = live_dates[live_dates.get_loc(PROBE_T) + 1]

    shocked = deepcopy(data)

    if FUTURE_DATE in shocked.equity_returns.index:
        shocked.equity_returns.loc[FUTURE_DATE] *= 100.0
    if FUTURE_DATE in shocked.commodity_returns.index:
        shocked.commodity_returns.loc[FUTURE_DATE] *= 100.0
    if FUTURE_DATE in shocked.commodity_prices.index:
        shocked.commodity_prices.loc[FUTURE_DATE] *= 5.0
    if FUTURE_DATE in shocked.fi_returns.index:
        shocked.fi_returns.loc[FUTURE_DATE] *= 100.0
    if FUTURE_DATE in shocked.fx_returns.index:
        shocked.fx_returns.loc[FUTURE_DATE] *= 100.0
    if FUTURE_DATE in shocked.fx_prices.index:
        shocked.fx_prices.loc[FUTURE_DATE] *= 5.0
    if FUTURE_DATE in shocked.fi_carry.index:
        shocked.fi_carry.loc[FUTURE_DATE] += 999.0
    if FUTURE_DATE in shocked.fx_carry.index:
        shocked.fx_carry.loc[FUTURE_DATE] += 999.0

    shocked_pipeline = build_expanded_pipeline(shocked)

    mom_diff = (
        pipeline["momentum_family"].loc[PROBE_T]
        - shocked_pipeline["momentum_family"].loc[PROBE_T]
    ).abs().max()
    carry_diff = (
        pipeline["carry_family"].loc[PROBE_T]
        - shocked_pipeline["carry_family"].loc[PROBE_T]
    ).abs().max()
    final_diff = (
        pipeline["final_weights"].loc[PROBE_T]
        - shocked_pipeline["final_weights"].loc[PROBE_T]
    ).abs().max()

    print(f"Probe weight date (t):   {PROBE_T.date()}")
    print(f"Shocked future date:     {FUTURE_DATE.date()}")
    print(f"Max momentum-family diff: {mom_diff:.2e}")
    print(f"Max carry-family diff:    {carry_diff:.2e}")
    print(f"Max final-weight diff:    {final_diff:.2e}")

    assert mom_diff == 0.0
    assert carry_diff == 0.0
    assert final_diff == 0.0
    print("✓ Expanded family and final weights at t are completely invariant to t+1 shocks.")
    """
)

md(
    """
    ## Summary

    All six tests pass for the **expanded final submission**:

    - return-based momentum only sees *t-12 … t-2*
    - price-based momentum only sees *P<sub>t-13</sub>* and *P<sub>t-2</sub>*
    - the 36-month covariance window ends exactly at *t*
    - the carry sleeve uses contemporaneous carry data only
    - final portfolio returns use weight-at-*t* times return-at-*t+1*
    - future shocks at *t+1* do not contaminate family or final weights at *t*

    This is the no-look-ahead audit relevant to the actual expanded strategy
    delivered in `FINAL_SUBMISSION.ipynb`.
    """
)

nb["cells"] = cells
out = Path(__file__).resolve().parents[1] / "notebooks" / "09_no_lookahead_verification.ipynb"
nbf.write(nb, out)
print(f"Wrote {out}")
print(
    f"Total cells: {len(cells)}  (markdown: {sum(1 for c in cells if c.cell_type == 'markdown')}, "
    f"code: {sum(1 for c in cells if c.cell_type == 'code')})"
)
