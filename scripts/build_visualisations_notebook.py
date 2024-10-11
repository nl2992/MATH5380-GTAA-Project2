"""Build notebooks/visualisations.ipynb — actually generates every chart from
the data, inline. Pulls the generation code straight out of
scripts/generate_visualizations.py rather than displaying pre-built PNGs.
"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
nb = nbf.v4.new_notebook()
cells = []

def md(t):   cells.append(nbf.v4.new_markdown_cell(t.lstrip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.lstrip("\n")))

# ── Title ─────────────────────────────────────────────────────────────────────
md("""
# Visualisations — GTAA Project 2

Every chart in `outputs/figures/` regenerated from the raw backtest output.
This is the same code that lives in `scripts/generate_visualizations.py`,
broken out cell-by-cell so each figure is reproducible and inspectable on
its own.

If you only want the numbers, read [`FINAL_SUBMISSION.ipynb`](FINAL_SUBMISSION.ipynb).
For the no-look-ahead audit, [`09_no_lookahead_verification.ipynb`](09_no_lookahead_verification.ipynb).
""")

# ── 0. Setup ──────────────────────────────────────────────────────────────────
md("## 0. Setup")

code("""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

# Columbia palette (kept consistent with scripts/generate_visualizations.py)
CU_NAVY, CU_BLUE, CU_LIGHT = "#003087", "#75AADB", "#B9D9EB"
CU_GOLD, CU_GREY, CU_RED, CU_GREEN = "#F2A900", "#6C6C6C", "#C4242B", "#2D7D46"

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10,
    "axes.titleweight": "bold",
    "axes.facecolor": "#FAFAFA",
    "grid.color": "#E5E5E5", "grid.linewidth": 0.5, "axes.grid": True,
})

RECESSIONS = [("2007-12-01","2009-06-30"), ("2020-02-01","2020-04-30")]

def shade_recessions(ax, alpha=0.13):
    for s, e in RECESSIONS:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=CU_GREY, alpha=alpha, lw=0)

def cumulative(rets):
    return (1 + rets.dropna()).cumprod()

def rolling_drawdown(rets):
    cum = cumulative(rets)
    return cum / cum.cummax() - 1
""")

# ── 1. Load data ──────────────────────────────────────────────────────────────
md("""
## 1. Run both backtests

Everything downstream is built off these two dictionaries (`br` for the
baseline three-sleeve, `er` for the expanded six-sleeve). They're the same
objects the production scripts use — same configs, same loaders.
""")

code("""
with open(ROOT / "config/project2_gtaa_mom_carry.yaml") as f:
    base_cfg = yaml.safe_load(f)
base_cfg["data"]["workbook"] = str(ROOT / base_cfg["data"]["workbook"])

with open(ROOT / "config/project2_gtaa_expanded.yaml") as f:
    exp_cfg = yaml.safe_load(f)
exp_cfg["data"]["workbook"] = str(ROOT / exp_cfg["data"]["workbook"])

from gtaa.portfolio.backtester_gtaa_mom_carry import run_gtaa_mom_carry_backtest
from gtaa.portfolio.backtester_gtaa_expanded import run_gtaa_expanded_backtest

br = run_gtaa_mom_carry_backtest(base_cfg)
er = run_gtaa_expanded_backtest(exp_cfg)

print(f"Baseline final returns:  {br['final_returns'].dropna().shape[0]} months")
print(f"Expanded final returns:  {er['final_returns'].dropna().shape[0]} months")
""")

# ── 2. Hero cumulative ────────────────────────────────────────────────────────
md("""
## 2. Hero — growth of $1 + drawdown

Top panel is wealth, bottom is drawdown. The expanded portfolio tracks the
baseline closely and bottoms out at the same place — the wider sleeve set
doesn't make tail risk worse.
""")

code("""
base_ret = br["final_returns"].dropna()
exp_ret  = er["final_returns"].dropna()
common_start = max(base_ret.index[0], exp_ret.index[0])

fig, axes = plt.subplots(2, 1, figsize=(13, 7),
                         gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
ax = axes[0]
ax.plot(cumulative(base_ret), color=CU_NAVY, lw=2.0, label="Baseline GTAA (3 sleeves)")
ax.plot(cumulative(exp_ret.loc[common_start:]), color=CU_BLUE, lw=2.0, label="Expanded GTAA (6 sleeves)")
ax.set_title("Growth of $1 — Baseline vs Expanded GTAA")
ax.set_ylabel("Wealth (start = 1)")
ax.legend(loc="upper left", frameon=False)
shade_recessions(ax)

ax2 = axes[1]
ax2.fill_between(rolling_drawdown(base_ret).index, rolling_drawdown(base_ret) * 100,
                 0, color=CU_NAVY, alpha=0.45, label="Baseline DD")
ax2.fill_between(rolling_drawdown(exp_ret.loc[common_start:]).index,
                 rolling_drawdown(exp_ret.loc[common_start:]) * 100,
                 0, color=CU_BLUE, alpha=0.45, label="Expanded DD")
ax2.set_ylabel("Drawdown (%)")
ax2.legend(loc="lower left", frameon=False)
shade_recessions(ax2)
plt.show()
""")

# ── 3. Baseline vs expanded — common-sample growth ────────────────────────────
md("""
## 3. Baseline vs expanded — common-sample growth

Same window for both, so the comparison is honest. Growth-of-$1 only.
""")

code("""
fig, ax = plt.subplots(figsize=(13, 5))
b = base_ret.loc[common_start:]
e = exp_ret.loc[common_start:]
ax.plot(cumulative(b), color=CU_NAVY, lw=2.0, label="Baseline GTAA")
ax.plot(cumulative(e), color=CU_BLUE, lw=2.0, label="Expanded GTAA")
ax.set_title(f"Growth of $1 — Common Sample ({common_start.date()} → {b.index[-1].date()})")
ax.set_ylabel("Wealth (start = 1)")
ax.legend(loc="upper left", frameon=False)
shade_recessions(ax)
plt.show()
""")

md("""
## 4. Baseline vs expanded — common-sample drawdown
""")

code("""
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.fill_between(rolling_drawdown(b).index, rolling_drawdown(b) * 100, 0,
                color=CU_NAVY, alpha=0.5, label=f"Baseline (max DD {rolling_drawdown(b).min()*100:.2f}%)")
ax.fill_between(rolling_drawdown(e).index, rolling_drawdown(e) * 100, 0,
                color=CU_BLUE, alpha=0.5, label=f"Expanded (max DD {rolling_drawdown(e).min()*100:.2f}%)")
ax.set_title("Drawdown — Common Sample")
ax.set_ylabel("Drawdown (%)")
ax.legend(loc="lower left", frameon=False)
shade_recessions(ax)
plt.show()
""")

# ── 5. Sleeve growth ──────────────────────────────────────────────────────────
md("""
## 5. Growth of $1 — each FMP sleeve

Six panels, one per sleeve. Equity Momentum and FI Carry are the two real
contributors; FX Carry and FI Momentum drift sideways or slightly down.
""")

code("""
SLEEVE_LABELS = {
    "equity_momentum":        "Equity Momentum FMP",
    "commodity_momentum":     "Commodity Momentum FMP",
    "fixed_income_momentum":  "Fixed-Income Momentum FMP",
    "fx_momentum":            "FX Momentum FMP",
    "fixed_income_carry":     "Fixed-Income Carry FMP",
    "fx_carry":               "FX Carry FMP",
}
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (sk, label) in zip(axes.flat, SLEEVE_LABELS.items()):
    r = er["sleeve_returns"][sk].dropna()
    ax.plot(cumulative(r), color=CU_NAVY, lw=1.5)
    ax.axhline(1, color=CU_GREY, lw=0.6)
    ax.set_title(label, fontsize=10)
    ax.set_ylabel("Wealth")
    shade_recessions(ax)
plt.suptitle("Growth of $1 — Each FMP Sleeve", fontweight="bold", y=1.0)
plt.tight_layout()
plt.show()
""")

# ── 6. Family lines ───────────────────────────────────────────────────────────
md("""
## 6. Family lines + final portfolio

The two factor families overlaid on the final portfolio. Carry is the
quieter, more consistent line.
""")

code("""
fig, ax = plt.subplots(figsize=(13, 5))
mom = er["factor_family_returns"]["cross_asset_momentum"].dropna()
car = er["factor_family_returns"]["cross_asset_carry"].dropna()
ax.plot(cumulative(mom), color=CU_NAVY, lw=1.6, label="Cross-Asset Momentum")
ax.plot(cumulative(car), color=CU_GOLD, lw=1.6, label="Cross-Asset Carry")
ax.plot(cumulative(exp_ret), color=CU_BLUE, lw=2.0, label="Expanded Final GTAA")
ax.set_title("Growth of $1 — Two Factor Families and the Final Portfolio")
ax.set_ylabel("Wealth")
ax.legend(loc="upper left", frameon=False)
shade_recessions(ax)
plt.show()
""")

# ── 7. Q6.1 raw factor weights ────────────────────────────────────────────────
md("""
## 7. Q6.1 — raw factor weights, all six sleeves

Cross-sectional rank standardisation, pre-vol-scaling. Bands are the live
universe at each date — they widen as more tickers come online.
""")

code("""
RAW_KEYS = list(SLEEVE_LABELS.keys())
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, sk in zip(axes.flat, RAW_KEYS):
    w = er["raw_weights"][sk].dropna(how="all")
    ax.plot(w.index, w.values, lw=0.7, alpha=0.75)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(SLEEVE_LABELS[sk], fontsize=10)
    shade_recessions(ax)
plt.suptitle("Raw Factor Weights — Pre-Vol-Scaling (Q6.1)", fontweight="bold", y=1.0)
plt.tight_layout()
plt.show()
""")

# ── 8. Q6.2a — Factor 1 (Momentum) 1% vol weights ─────────────────────────────
md("""
## 8. Q6.2a — Factor 1 (Cross-Asset Momentum) at 1% vol

Equal-weighted across the four momentum sleeves, rescaled to 1% annualised
volatility on the full cross-asset covariance.
""")

code("""
mom_w = er["factor_family_weights"]["cross_asset_momentum"].dropna(how="all")
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(mom_w.index, mom_w.values, lw=0.6, alpha=0.7)
ax.axhline(0, color="black", lw=0.5)
ax.set_title("Factor 1 — Cross-Asset Momentum Family at 1% Vol (Q6.2)")
ax.set_ylabel("Weight")
shade_recessions(ax)
plt.show()
""")

# ── 9. Q6.2b — Factor 2 (Carry) 1% vol weights ────────────────────────────────
md("""
## 9. Q6.2b — Factor 2 (Cross-Asset Carry) at 1% vol

Equal weights on FI Carry and FX Carry, then rescaled.
""")

code("""
car_w = er["factor_family_weights"]["cross_asset_carry"].dropna(how="all")
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(car_w.index, car_w.values, lw=0.6, alpha=0.7)
ax.axhline(0, color="black", lw=0.5)
ax.set_title("Factor 2 — Cross-Asset Carry Family at 1% Vol (Q6.2)")
ax.set_ylabel("Weight")
shade_recessions(ax)
plt.show()
""")

# ── 10. Q6.3 — Final 1% vol portfolio weights ─────────────────────────────────
md("""
## 10. Q6.3 — Final 1% vol portfolio weights

Both families equal-weighted, full portfolio rescaled one more time.
""")

code("""
fw = er["final_weights"].dropna(how="all")
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(fw.index, fw.values, lw=0.5, alpha=0.65)
ax.axhline(0, color="black", lw=0.5)
ax.set_title("Final 1% Vol Portfolio Weights (Q6.3)")
ax.set_ylabel("Weight")
shade_recessions(ax)
plt.show()
""")

# ── 11. Sleeve-level weight detail (six panels) ───────────────────────────────
md("""
## 11. Sleeve-level scaled weights (six-panel detail)

Same weights as Q6.1 but post-vol-scaling, one panel per sleeve. Easier to
read individual asset behaviour this way.
""")

code("""
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, sk in zip(axes.flat, RAW_KEYS):
    w = er["sleeve_scaled_weights"][sk].dropna(how="all")
    ax.plot(w.index, w.values, lw=0.7, alpha=0.75)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(SLEEVE_LABELS[sk], fontsize=10)
    shade_recessions(ax)
plt.suptitle("Sleeve-Level 1% Vol Weights — Detail", fontweight="bold", y=1.0)
plt.tight_layout()
plt.show()
""")

# ── 12. Family weight detail ──────────────────────────────────────────────────
md("""
## 12. Family-level weight detail

Two panels, one per family. Carry visibly steadier than Momentum.
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, (key, title) in zip(axes,
        [("cross_asset_momentum", "Cross-Asset Momentum"),
         ("cross_asset_carry",    "Cross-Asset Carry")]):
    w = er["factor_family_weights"][key].dropna(how="all")
    ax.plot(w.index, w.values, lw=0.6, alpha=0.7)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(title, fontsize=11)
    shade_recessions(ax)
plt.suptitle("Family-Level 1% Vol Weights", fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()
""")

# ── 13. All portfolios on one canvas ──────────────────────────────────────────
md("""
## 13. Six sleeves + two families + final portfolio — one canvas

Sanity-check: the family lines really do sit inside the sleeve cloud, and the
final portfolio is meaningfully more contained than any single sleeve.
""")

code("""
fig, ax = plt.subplots(figsize=(13, 6))
for sk in RAW_KEYS:
    r = er["sleeve_returns"][sk].dropna()
    ax.plot(cumulative(r), lw=1.0, alpha=0.7, label=SLEEVE_LABELS[sk])
ax.plot(cumulative(mom), color=CU_NAVY, lw=2.2, label="Momentum Family", linestyle="--")
ax.plot(cumulative(car), color=CU_GOLD, lw=2.2, label="Carry Family", linestyle="--")
ax.plot(cumulative(exp_ret), color="black", lw=2.5, label="Expanded Final GTAA")
ax.set_title("All Six Sleeves, Both Families, and the Final Portfolio")
ax.set_ylabel("Wealth (start = 1)")
ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=8)
shade_recessions(ax)
plt.show()
""")

# ── 14. Q7 final-period covariance ────────────────────────────────────────────
md("""
## 14. Q7.1 — final-period annualised covariance

The 36-month rolling estimate at the final weight date, annualised by ×12.
""")

code("""
cov_f = er["final_cov"]
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(cov_f.values * 1e4, cmap="RdBu_r", aspect="auto")
ax.set_xticks(range(len(cov_f))); ax.set_yticks(range(len(cov_f)))
ax.set_xticklabels(cov_f.columns, rotation=90, fontsize=7)
ax.set_yticklabels(cov_f.index, fontsize=7)
ax.set_title("Annualised Covariance Matrix — Final Period (Q7.1)")
plt.colorbar(im, ax=ax, label="Cov (×10⁻⁴)")
plt.tight_layout()
plt.show()
""")

# ── 15. Q7.2 — final-period correlation ───────────────────────────────────────
md("""
## 15. Q7.2 — final-period correlation

Equity ETFs cluster (0.5–0.9), Treasuries cluster by duration (0.7–0.95),
commodities are mostly uncorrelated with everything else. Those near-zero
cross-class correlations are the structural reason the combined max DD lands
at −2.83% — well inside the worst sleeve drawdown of −5.0%.
""")

code("""
corr_f = er["final_corr"]
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr_f.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(corr_f))); ax.set_yticks(range(len(corr_f)))
ax.set_xticklabels(corr_f.columns, rotation=90, fontsize=7)
ax.set_yticklabels(corr_f.index, fontsize=7)
ax.set_title("Correlation Matrix — Final Period (Q7.2)")
plt.colorbar(im, ax=ax, label="Correlation")
plt.tight_layout()
plt.show()
""")

# ── 16. Q7.3 — annualised vols ────────────────────────────────────────────────
md("""
## 16. Q7.3 — final-period annualised volatilities

UNG/PALL/SLV stand head and shoulders above everything else. Without
sleeve-level vol targeting, the commodity block would dominate portfolio
risk on its own.
""")

code("""
vols_f = er["final_vols"].sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(15, 5))
ax.bar(range(len(vols_f)), vols_f.values * 100, color=CU_NAVY, alpha=0.85,
       edgecolor="black", lw=0.4)
ax.set_xticks(range(len(vols_f)))
ax.set_xticklabels(vols_f.index, rotation=90, fontsize=8)
ax.set_ylabel("Annualised Volatility (%)")
ax.set_title("Asset Annualised Volatilities — Final Period (Q7.3)")
plt.tight_layout()
plt.show()
""")

# ── 17. Cross-asset corr overview ─────────────────────────────────────────────
md("""
## 17. Cross-asset correlation — broader overview

Same data as Section 15 but with class blocks shaded so block structure pops
out at a glance.
""")

code("""
# Order assets by class so the blocks line up
ASSET_ORDER = (
    ["EWA","EWC","EWQ","EWG","EWI","EWJ","EWN","EWP","EWU","SPY"]
  + ["GLD","SLV","CPER","PPLT","PALL","USO","BNO","UNG","DBA","CORN","WEAT","SOYB"]
  + ["SHV","SHY","IEI","IEF","TLH","TLT","VTIP","TIP","LQD","HYG"]
  + ["FXE","FXY","FXB","FXA","FXC","FXF"]
)
present = [a for a in ASSET_ORDER if a in corr_f.columns]
corr_o = corr_f.loc[present, present]

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr_o.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(corr_o))); ax.set_yticks(range(len(corr_o)))
ax.set_xticklabels(corr_o.columns, rotation=90, fontsize=7)
ax.set_yticklabels(corr_o.index, fontsize=7)

# Block dividers (equity / commodity / FI / FX)
for boundary in (10, 22, 32):
    if boundary < len(corr_o):
        ax.axhline(boundary - 0.5, color="black", lw=0.8)
        ax.axvline(boundary - 0.5, color="black", lw=0.8)
ax.set_title("Cross-Asset Correlation — Class-Ordered")
plt.colorbar(im, ax=ax, label="Correlation")
plt.tight_layout()
plt.show()
""")

# ── 18. Turnover ──────────────────────────────────────────────────────────────
md("""
## 18. Turnover — final portfolio

Funded-portfolio convention: at each rebalance, the previous weights drift
with one month of returns, then the difference between drifted and target
weights is what costs turnover. Annualised this comes out near 236%, mostly
driven by FI and FX momentum cycling rather than the carry sleeves.
""")

code("""
from gtaa.portfolio.turnover import compute_turnover

# combined returns panel for the final-portfolio universe
combined = pd.concat([er["data"].equity_returns, er["data"].commodity_returns,
                      er["data"].fi_returns,    er["data"].fx_returns], axis=1)
combined = combined.loc[:, ~combined.columns.duplicated()]
combined = combined.reindex(columns=er["final_weights"].columns)

turn = compute_turnover(er["final_weights"], combined).dropna()

fig, ax = plt.subplots(figsize=(13, 4.5))
ax.plot(turn.index, turn.values * 100, color=CU_NAVY, lw=1.0)
ax.axhline(turn.mean() * 100, color=CU_GOLD, lw=1.2, linestyle="--",
           label=f"mean ≈ {turn.mean()*100:.1f}%/mo  (ann. ≈ {turn.mean()*12*100:.1f}%)")
ax.set_title("Monthly One-Way Turnover — Final Portfolio")
ax.set_ylabel("Turnover (%)")
ax.legend(loc="upper left", frameon=False)
shade_recessions(ax)
plt.show()
""")

# ── 19. Summary stats table (rendered) ────────────────────────────────────────
md("""
## 19. Summary statistics — rendered table

Same numbers that go into the metrics-table images. Computed from the
backtest output dictionaries directly.
""")

code("""
def ann_return(r):  return r.dropna().mean() * 12             # arithmetic
def ann_vol(r):     return r.dropna().std(ddof=0) * np.sqrt(12)
def info_ratio(r):  rd = r.dropna(); return (rd.mean()*12) / (rd.std(ddof=0)*np.sqrt(12))
def max_dd(r):      cum = cumulative(r); return (cum/cum.cummax() - 1).min()
def avg_dd(r):
    cum = cumulative(r); dd = cum/cum.cummax() - 1
    return dd[dd < 0].mean()

def turnover_ann(weights, rets):
    w = weights.dropna(how="all")
    if len(w) < 2: return np.nan
    rets_aligned = rets.reindex(columns=w.columns).fillna(0.0)
    t = compute_turnover(w, rets_aligned).dropna()
    return t.mean() * 12 if len(t) else np.nan

rows = []
for sk, label in SLEEVE_LABELS.items():
    r = er["sleeve_returns"][sk]; w = er["sleeve_scaled_weights"][sk]
    rows.append([label, ann_return(r), ann_vol(r), info_ratio(r),
                 max_dd(r), avg_dd(r), turnover_ann(w, combined)])
for fk, label in [("cross_asset_momentum","Cross-Asset Momentum Family"),
                  ("cross_asset_carry",   "Cross-Asset Carry Family")]:
    r = er["factor_family_returns"][fk]; w = er["factor_family_weights"][fk]
    rows.append([label, ann_return(r), ann_vol(r), info_ratio(r),
                 max_dd(r), avg_dd(r), turnover_ann(w, combined)])
rows.append(["Expanded Final GTAA",
             ann_return(exp_ret), ann_vol(exp_ret), info_ratio(exp_ret),
             max_dd(exp_ret), avg_dd(exp_ret),
             turnover_ann(er["final_weights"], combined)])

stats = pd.DataFrame(rows, columns=[
    "Sleeve / Family", "Ann. return", "Ann. vol", "IR", "Max DD", "Avg DD", "Ann. Turnover"
])
disp = stats.copy()
for c in ["Ann. return","Ann. vol","Max DD","Avg DD","Ann. Turnover"]:
    disp[c] = disp[c].apply(lambda x: f"{x*100:.2f}%")
disp["IR"] = disp["IR"].apply(lambda x: f"{x:.3f}")
disp
""")

# ── Closing ───────────────────────────────────────────────────────────────────
md("""
## Where to go from here

- Numbers and full pipeline: [`FINAL_SUBMISSION.ipynb`](FINAL_SUBMISSION.ipynb)
- No-look-ahead audit: [`09_no_lookahead_verification.ipynb`](09_no_lookahead_verification.ipynb)
- Written commentary: [`SUPPLEMENTARY_ANALYSIS.md`](../SUPPLEMENTARY_ANALYSIS.md) and
  [`outputs/MATH5380 Report.docx`](../outputs/MATH5380%20Report.docx)
- Pre-rendered chart files: `outputs/figures/`
""")

# ── Save ──────────────────────────────────────────────────────────────────────
nb["cells"] = cells
out = ROOT / "notebooks" / "visualisations.ipynb"
nbf.write(nb, out)
print(f"Wrote {out}")
print(f"Cells: {len(cells)} ({sum(1 for c in cells if c.cell_type=='markdown')} md / {sum(1 for c in cells if c.cell_type=='code')} code)")
