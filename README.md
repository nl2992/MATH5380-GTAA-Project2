# Long/Short Factor-Based GTAA Portfolio

**Columbia University · MAFN · MATH 5380 · Project 2**

This repository contains the final submission materials for our long/short
global tactical asset allocation project. The submitted strategy is the
**expanded six-sleeve specification** built from two factor families:
**Cross-Asset Momentum** and **Cross-Asset Carry**. The earlier three-sleeve
version is retained as a benchmark.

## Final Submission Files

| File | Location | Purpose |
|---|---|---|
| Main notebook | [notebooks/FINAL_SUBMISSION.ipynb](notebooks/FINAL_SUBMISSION.ipynb) | Full write-up of the model construction, results, and required charts |
| Audit notebook | [notebooks/09_no_lookahead_verification.ipynb](notebooks/09_no_lookahead_verification.ipynb) | Timing and no-look-ahead verification for the expanded strategy |
| Report | [outputs/MATH5380 Report.docx](outputs/MATH5380%20Report.docx) | Final written report |
| Supplementary analysis | [SUPPLEMENTARY_ANALYSIS.md](SUPPLEMENTARY_ANALYSIS.md) | Additional figures, interpretation, and extended discussion |

## Results Summary

| Strategy | Live sample | Months | Ann. return | Ann. vol | IR | Max DD | Ann. turnover |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline GTAA | Nov 2004 – Nov 2025 | 253 | **0.46%** | **1.06%** | **0.431** | **-2.83%** | 167.6% |
| Expanded GTAA | Dec 2005 – Nov 2025 | 240 | 0.42% | 1.15% | 0.361 | -2.86% | 236.3% |

The baseline is the stronger in-sample benchmark, but the expanded portfolio is
the broader GTAA design because it extends the same momentum-and-carry logic
across equities, commodities, fixed income, and FX while keeping drawdown close
to the baseline.

### Expanded Strategy: Family-Level Performance

| Sleeve or Family | Ann. return | Ann. vol | IR | Max DD | Avg DD | Ann. Turnover |
|---|---:|---:|---:|---:|---:|---:|
| Cross-Asset Momentum Family | 0.13% | 1.09% | 0.124 | -4.72% | -2.29% | 271.2% |
| Cross-Asset Carry Family | 0.35% | 1.06% | 0.326 | -3.36% | -0.92% | 90.2% |
| Expanded Final GTAA | 0.42% | 1.15% | 0.361 | -2.86% | -1.01% | 236.3% |

### Expanded Strategy: Sleeve-Level Performance

| Sleeve or Family | Ann. return | Ann. vol | IR | Max DD | Avg DD | Ann. Turnover |
|---|---:|---:|---:|---:|---:|---:|
| Equity Momentum FMP | 0.40% | 1.08% | 0.374 | -5.00% | -1.25% | 111.7% |
| Commodity Momentum FMP | 0.14% | 0.97% | 0.144 | -4.39% | -2.34% | 35.9% |
| Fixed-Income Momentum FMP | -0.07% | 0.99% | -0.075 | -4.96% | -2.71% | 187.8% |
| FX Momentum FMP | 0.02% | 0.87% | 0.027 | -3.43% | -1.08% | 166.2% |
| Fixed-Income Carry FMP | 0.13% | 0.58% | 0.217 | -4.21% | -1.32% | 27.2% |
| FX Carry FMP | -0.01% | 0.87% | -0.012 | -4.13% | -1.96% | 24.7% |

Image versions:
[family performance table](outputs/figures/expanded_family_metrics_table.png) and
[sleeve performance table](outputs/figures/expanded_sleeve_metrics_table.png).

## How to Reproduce the Submission

1. Create the environment once with `conda env create -f env.yml`.
2. Activate it with `conda activate MATH5320`.
3. Install the package locally with `pip install -e .`.
4. Regenerate the figure set with `python scripts/generate_visualizations.py`.
5. Rebuild the notebooks with `python scripts/build_final_submission_notebook.py` and `python scripts/build_no_lookahead_notebook.py`.
6. Execute the notebooks with `jupyter nbconvert --to notebook --execute --inplace notebooks/FINAL_SUBMISSION.ipynb` and `jupyter nbconvert --to notebook --execute --inplace notebooks/09_no_lookahead_verification.ipynb`.
7. Rebuild the report source export with `python scripts/build_report_docx.py`.
8. Run `pytest tests/ -v` if you want the full automated check suite.

For interactive review, open [notebooks/FINAL_SUBMISSION.ipynb](notebooks/FINAL_SUBMISSION.ipynb) in Jupyter Lab after Step 6.

## Repository Layout

```text
.
├── README.md
├── SUPPLEMENTARY_ANALYSIS.md
├── REPORT.md
├── env.yml
├── pyproject.toml
├── requirements.txt
├── config/
│   ├── project2_gtaa_expanded.yaml
│   └── project2_gtaa_mom_carry.yaml
├── data/
│   └── raw/
│       ├── Data for final project 2 .xlsx
│       ├── multi_asset_coverage.csv
│       └── multi_asset_universe.xlsx
├── docs/
│   └── Project 2 .pdf
├── notebooks/
│   ├── 09_no_lookahead_verification.ipynb
│   └── FINAL_SUBMISSION.ipynb
├── outputs/
│   ├── MATH5380 Report.docx
│   └── figures/
│       ├── cu_cross_asset_corr.png
│       ├── cu_expanded_all_portfolios.png
│       ├── cu_expanded_family_detail.png
│       ├── cu_expanded_sleeve_detail.png
│       ├── cu_hero_cumulative.png
│       ├── cu_turnover.png
│       ├── expanded_family_metrics_table.png
│       ├── expanded_metrics_table.png
│       ├── expanded_sleeve_growth.png
│       ├── expanded_sleeve_metrics_table.png
│       ├── expanded_vs_baseline_drawdown.png
│       └── expanded_vs_baseline_growth.png
├── scripts/
│   ├── add_fx_data.py
│   ├── build_final_submission_notebook.py
│   ├── build_no_lookahead_notebook.py
│   ├── build_report_docx.py
│   ├── generate_visualizations.py
│   └── pull_universe_data.py
├── src/
│   └── gtaa/
│       ├── __init__.py
│       ├── analytics/
│       ├── config.py
│       ├── data/
│       ├── factors/
│       ├── io/
│       ├── models.py
│       ├── portfolio/
│       ├── regime/
│       ├── reporting/
│       ├── risk/
│       └── validation/
└── tests/
    ├── __init__.py
    ├── test_covariance.py
    ├── test_date_alignment.py
    ├── test_expanded_backtest_integration.py
    ├── test_expanded_factor_family_construction.py
    ├── test_expanded_factor_family_scaling.py
    ├── test_expanded_no_lookahead.py
    ├── test_factor_weights.py
    ├── test_fi_momentum_signal.py
    ├── test_fx_carry_signal.py
    ├── test_fx_momentum_signal.py
    ├── test_hw1_replication.py
    ├── test_hw2_replication.py
    ├── test_no_lookahead_covariance.py
    ├── test_no_lookahead_portfolio_returns.py
    ├── test_no_lookahead_signals.py
    ├── test_performance_stats.py
    ├── test_rank_direction.py
    ├── test_turnover.py
    └── test_vol_scaling.py
```

## Project 2 Required Visuals

The assignment asks for embedded charts covering gross returns and the required
weight panels. Those figures are collected here in one place.

### Gross Returns: Final Portfolio

This figure shows the growth of $1 for the submitted expanded portfolio, with
the baseline and family series included for context.

![Final portfolio growth of $1](outputs/figures/cu_hero_cumulative.png)

### Gross Returns: Each FMP / Sleeve

![Growth of $1 for each sleeve](outputs/figures/expanded_sleeve_growth.png)

### Raw Factor Weights

![Raw factor weights](outputs/figures/q6_q7_from_docx/q6_raw_factor_weights.png)

### Factor 1 1% Volatility Weights

![Factor 1 momentum family weights](outputs/figures/q6_q7_from_docx/q6_factor1_momentum_weights.png)

### Factor 2 1% Volatility Weights

![Factor 2 carry family weights](outputs/figures/q6_q7_from_docx/q6_factor2_carry_weights.png)

### Final 1% Volatility Portfolio Weights

![Final 1% volatility portfolio weights](outputs/figures/q6_q7_from_docx/q6_final_portfolio_weights.png)

Additional tables, turnover plots, and final-period risk visuals are in
[SUPPLEMENTARY_ANALYSIS.md](SUPPLEMENTARY_ANALYSIS.md).

## Data Pull and Preparation Code

- [scripts/pull_universe_data.py](scripts/pull_universe_data.py): rebuilds the
  core multi-asset workbook from the provided equity file, Yahoo Finance ETF
  histories, and FRED rate and yield series.
- [scripts/add_fx_data.py](scripts/add_fx_data.py): appends the FX ETF price,
  return, and carry sheets used in the expanded specification.
- [src/gtaa/io/excel_loader.py](src/gtaa/io/excel_loader.py): loading and
  cleaning logic used by the notebooks.
- [data/raw/Data for final project 2 .xlsx](data/raw/Data%20for%20final%20project%202%20.xlsx):
  provided course file used in the initial universe build.
- [data/raw/multi_asset_universe.xlsx](data/raw/multi_asset_universe.xlsx):
  final workbook consumed by the notebooks and backtests.

## Notes

- The teaching team confirmed that a Python notebook is acceptable in place of
  a separate Excel model as long as the data handling, calculations, summary
  statistics, and required charts are embedded directly in the notebook.
- [REPORT.md](REPORT.md) and [scripts/build_report_docx.py](scripts/build_report_docx.py)
  are kept so the exported report can be regenerated from the repository text
  source.
- [SUPPLEMENTARY_ANALYSIS.md](SUPPLEMENTARY_ANALYSIS.md) holds the extended
  analysis, additional risk figures, and brief caveats that would have been too
  long for the five-page report.
