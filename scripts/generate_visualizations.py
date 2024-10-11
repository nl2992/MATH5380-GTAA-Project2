"""Generate all GTAA visualizations with Columbia University theme.

Run from the repository root:
    python scripts/generate_visualizations.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
from scipy import stats

# ── Columbia palette ──────────────────────────────────────────────────────────
CU_NAVY  = "#003087"
CU_BLUE  = "#75AADB"
CU_LIGHT = "#B9D9EB"
CU_GOLD  = "#F2A900"
CU_GREY  = "#6C6C6C"
CU_RED   = "#C4242B"
CU_GREEN = "#2D7D46"

plt.rcParams.update({
    "figure.dpi": 140,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "grid.color": "#E5E5E5",
    "grid.linewidth": 0.5,
    "axes.grid": True,
})

FIGS = PROJECT_ROOT / "outputs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
REPORT_START = pd.Timestamp("2004-11-01")

RECESSIONS = [
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),
]


def shade_recessions(ax, alpha: float = 0.13) -> None:
    for s, e in RECESSIONS:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                   color=CU_GREY, alpha=alpha, zorder=0, label="_nolegend_")


def watermark(fig, text: str = "Columbia MAFN · MATH 5380") -> None:
    fig.text(0.995, 0.005, text, ha="right", va="bottom",
             fontsize=7, color=CU_GREY, alpha=0.55, style="italic")


def set_report_start(ax) -> None:
    """Keep report charts anchored to the project backtest window."""
    ax.set_xlim(left=REPORT_START)


def savefig(fig, name: str) -> None:
    path = FIGS / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved → {path.name}")
    plt.close(fig)


def rolling_drawdown(rets: pd.Series) -> pd.Series:
    cum = (1 + rets).cumprod()
    return (cum / cum.cummax()) - 1


def cumulative(rets: pd.Series) -> pd.Series:
    return (1 + rets).cumprod()


# ── Load strategies ────────────────────────────────────────────────────────────

def load_strategies():
    base_cfg_path = PROJECT_ROOT / "config" / "project2_gtaa_mom_carry.yaml"
    with open(base_cfg_path) as f:
        base_cfg = yaml.safe_load(f)
    base_cfg["data"]["workbook"] = str(
        PROJECT_ROOT / base_cfg["data"]["workbook"]
    )

    exp_cfg_path = PROJECT_ROOT / "config" / "project2_gtaa_expanded.yaml"
    with open(exp_cfg_path) as f:
        exp_cfg = yaml.safe_load(f)
    exp_cfg["data"]["workbook"] = str(
        PROJECT_ROOT / exp_cfg["data"]["workbook"]
    )

    from gtaa.portfolio.backtester_gtaa_mom_carry import run_gtaa_mom_carry_backtest
    from gtaa.portfolio.backtester_gtaa_expanded import run_gtaa_expanded_backtest

    print("Running baseline backtest…")
    br = run_gtaa_mom_carry_backtest(base_cfg)
    print("Running expanded backtest…")
    er = run_gtaa_expanded_backtest(exp_cfg)
    return br, er


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Hero: Growth of $1, both strategies, recession shading
# ══════════════════════════════════════════════════════════════════════════════

def fig_hero_cumulative(br, er):
    base_ret = br["final_returns"].dropna()
    exp_ret  = er["final_returns"].dropna()
    mom_fam  = er["factor_family_returns"]["cross_asset_momentum"].dropna()
    carry_fam = er["factor_family_returns"]["cross_asset_carry"].dropna()

    common_start = max(base_ret.index[0], exp_ret.index[0])

    fig, axes = plt.subplots(2, 1, figsize=(13, 8),
                             gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06})

    ax = axes[0]
    ax.plot(cumulative(base_ret), color=CU_NAVY, lw=2.0, label="Baseline GTAA (3 sleeves)")
    ax.plot(cumulative(exp_ret.loc[common_start:]),  color=CU_BLUE,  lw=2.0, label="Expanded GTAA (6 sleeves)")
    ax.plot(cumulative(carry_fam.loc[common_start:]), color=CU_GOLD, lw=1.4, ls="--", label="Carry Family")
    ax.plot(cumulative(mom_fam.loc[common_start:]),   color=CU_GREY, lw=1.4, ls=":", label="Momentum Family")
    ax.axhline(1, color="k", lw=0.6, ls="--", alpha=0.4)
    shade_recessions(ax)
    ax.set_ylabel("Growth of $1")
    ax.set_title("GTAA Factor Portfolios — Cumulative Performance", pad=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
    ax.legend(loc="upper left", framealpha=0.9)
    set_report_start(ax)
    ax.tick_params(axis="x", labelbottom=False)

    # Recession label on first shading
    ax.text(pd.Timestamp("2008-06-01"), ax.get_ylim()[0] * 1.01,
            "GFC", fontsize=7, color=CU_GREY, va="bottom")
    ax.text(pd.Timestamp("2020-02-15"), ax.get_ylim()[0] * 1.01,
            "COVID", fontsize=7, color=CU_GREY, va="bottom")

    # Bottom panel: drawdown
    ax2 = axes[1]
    ax2.fill_between(base_ret.index, rolling_drawdown(base_ret) * 100,
                     alpha=0.55, color=CU_NAVY, label="Baseline DD")
    ax2.fill_between(exp_ret.loc[common_start:].index,
                     rolling_drawdown(exp_ret.loc[common_start:]) * 100,
                     alpha=0.45, color=CU_BLUE, label="Expanded DD")
    shade_recessions(ax2)
    ax2.set_ylabel("Drawdown (%)")
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    set_report_start(ax2)

    watermark(fig)
    savefig(fig, "cu_hero_cumulative.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Annual returns bar chart (baseline vs expanded)
# ══════════════════════════════════════════════════════════════════════════════

def fig_annual_returns(br, er):
    base_ret = br["final_returns"].dropna()
    exp_ret  = er["final_returns"].dropna()

    def annual(r):
        return r.resample("YE").apply(lambda x: (1 + x).prod() - 1)

    base_ann = annual(base_ret) * 100
    exp_ann  = annual(exp_ret) * 100

    years = base_ann.index.year
    x = np.arange(len(years))
    w = 0.38

    fig, ax = plt.subplots(figsize=(14, 5))

    for i, (yr, bv) in enumerate(zip(years, base_ann)):
        ax.bar(x[i] - w/2, bv, w, color=CU_NAVY if bv >= 0 else CU_RED,
               alpha=0.85, label="_")

    exp_years = exp_ann.index.year
    for i, yr in enumerate(years):
        if yr in exp_years.values:
            ev = exp_ann[exp_ann.index.year == yr].values[0]
            ax.bar(x[i] + w/2, ev, w, color=CU_BLUE if ev >= 0 else "#E8A0A0",
                   alpha=0.85, label="_")

    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, ha="right")
    ax.set_ylabel("Annual Return (%)")
    ax.set_title("Annual Returns — Baseline vs Expanded GTAA")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

    legend_patches = [
        mpatches.Patch(color=CU_NAVY, alpha=0.85, label="Baseline GTAA (+)"),
        mpatches.Patch(color=CU_RED,  alpha=0.85, label="Baseline GTAA (−)"),
        mpatches.Patch(color=CU_BLUE, alpha=0.85, label="Expanded GTAA (+)"),
        mpatches.Patch(color="#E8A0A0", alpha=0.85, label="Expanded GTAA (−)"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", framealpha=0.9, ncol=2)

    watermark(fig)
    savefig(fig, "cu_annual_returns.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Monthly return calendar heatmap (baseline)
# ══════════════════════════════════════════════════════════════════════════════

def fig_calendar_heatmap(br, er):
    for label, ret, fname in [
        ("Baseline GTAA", br["final_returns"].dropna(), "cu_calendar_baseline.png"),
        ("Expanded GTAA", er["final_returns"].dropna(), "cu_calendar_expanded.png"),
    ]:
        ret_pct = ret * 100
        df = ret_pct.to_frame("r")
        df["year"]  = df.index.year
        df["month"] = df.index.month

        pivot = df.pivot(index="year", columns="month", values="r")
        month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"]
        pivot.columns = [month_labels[m-1] for m in pivot.columns]

        # Annual total
        annual = (1 + ret).resample("YE").apply(lambda x: (1+x).prod()-1) * 100
        pivot["Ann"] = annual.values[:len(pivot)]

        abs_max = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 1.0)
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

        cu_cmap = sns.diverging_palette(220, 10, s=80, l=50, as_cmap=True)

        fig, ax = plt.subplots(figsize=(16, max(5, len(pivot) * 0.45)))
        sns.heatmap(
            pivot, annot=True, fmt=".1f", cmap=cu_cmap, norm=norm,
            linewidths=0.4, linecolor="#D8D8D8",
            cbar_kws={"label": "Monthly Return (%)", "shrink": 0.6},
            ax=ax, annot_kws={"size": 7.5},
        )
        ax.set_title(f"Monthly Returns (%) — {label}", pad=12)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)

        # Thicker line before Ann column
        ax.axvline(12, color=CU_NAVY, lw=1.5)

        watermark(fig)
        savefig(fig, fname)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Rolling 24-month Information Ratio
# ══════════════════════════════════════════════════════════════════════════════

def fig_rolling_ir(br, er):
    base_ret = br["final_returns"].dropna()
    exp_ret  = er["final_returns"].dropna()

    def rolling_ir(r, window=24):
        mu  = r.rolling(window).mean() * 12
        sig = r.rolling(window).std(ddof=0) * np.sqrt(12)
        return mu / sig

    common_start = max(base_ret.index[0], exp_ret.index[0])

    fig, ax = plt.subplots(figsize=(13, 4.5))
    roll_base = rolling_ir(base_ret)
    roll_exp  = rolling_ir(exp_ret.loc[common_start:])

    ax.plot(roll_base.index, roll_base, color=CU_NAVY, lw=1.6, label="Baseline GTAA")
    ax.plot(roll_exp.index,  roll_exp,  color=CU_BLUE, lw=1.6, label="Expanded GTAA")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.axhline(roll_base.mean(), color=CU_NAVY, lw=0.8, ls=":", alpha=0.6,
               label=f"Baseline mean = {roll_base.mean():.2f}")
    ax.axhline(roll_exp.mean(),  color=CU_BLUE, lw=0.8, ls=":", alpha=0.6,
               label=f"Expanded mean = {roll_exp.mean():.2f}")
    shade_recessions(ax)
    ax.set_title("Rolling 24-Month Information Ratio")
    ax.set_ylabel("Information Ratio")
    ax.legend(loc="upper right", framealpha=0.9)
    set_report_start(ax)
    ax.fill_between(roll_base.index, roll_base, 0,
                    where=roll_base > 0, alpha=0.08, color=CU_NAVY)
    ax.fill_between(roll_base.index, roll_base, 0,
                    where=roll_base < 0, alpha=0.08, color=CU_RED)

    watermark(fig)
    savefig(fig, "cu_rolling_ir.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Rolling 12m realized vol vs 1% target
# ══════════════════════════════════════════════════════════════════════════════

def fig_rolling_vol(br, er):
    base_ret = br["final_returns"].dropna()
    exp_ret  = er["final_returns"].dropna()

    def roll_vol(r, w=12):
        return r.rolling(w).std(ddof=0) * np.sqrt(12) * 100

    common_start = max(base_ret.index[0], exp_ret.index[0])

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(roll_vol(base_ret), color=CU_NAVY, lw=1.5, label="Baseline realised vol")
    ax.plot(roll_vol(exp_ret.loc[common_start:]),  color=CU_BLUE, lw=1.5, label="Expanded realised vol")
    ax.axhline(1.0, color=CU_GOLD, lw=1.8, ls="--", label="1% vol target")
    ax.fill_between(roll_vol(base_ret).index, 0.8, 1.2,
                    alpha=0.07, color=CU_GOLD, label="±0.2% band")
    shade_recessions(ax)
    ax.set_title("Rolling 12-Month Realised Volatility vs 1% Target")
    ax.set_ylabel("Annualised Vol (%)")
    ax.legend(loc="upper right", framealpha=0.9)
    set_report_start(ax)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

    watermark(fig)
    savefig(fig, "cu_rolling_vol.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Return distribution: histogram + KDE + normal overlay
# ══════════════════════════════════════════════════════════════════════════════

def fig_return_distribution(br, er):
    base_ret = br["final_returns"].dropna() * 100
    exp_ret  = er["final_returns"].dropna() * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for ax, (ret, label, color) in zip(axes, [
        (base_ret, "Baseline GTAA (3 sleeves)", CU_NAVY),
        (exp_ret,  "Expanded GTAA (6 sleeves)", CU_BLUE),
    ]):
        mu, sd = ret.mean(), ret.std()
        x = np.linspace(ret.min() - 0.2, ret.max() + 0.2, 300)
        normal_pdf = stats.norm.pdf(x, mu, sd)

        ax.hist(ret, bins=35, color=color, alpha=0.55, density=True,
                edgecolor="white", linewidth=0.4)

        kde = stats.gaussian_kde(ret)
        ax.plot(x, kde(x), color=color, lw=2.0, label="Empirical KDE")
        ax.plot(x, normal_pdf, color=CU_GOLD, lw=1.6, ls="--", label="Normal fit")

        # VaR 5%
        var5 = np.percentile(ret, 5)
        cvar5 = ret[ret <= var5].mean()
        ax.axvline(var5,  color=CU_RED, lw=1.4, ls="--",
                   label=f"VaR 5% = {var5:.2f}%")
        ax.axvline(cvar5, color=CU_RED, lw=1.0, ls=":",
                   label=f"CVaR 5% = {cvar5:.2f}%")
        ax.axvline(0, color="k", lw=0.7, alpha=0.5)

        sk = stats.skew(ret)
        ku = stats.kurtosis(ret)
        ax.set_title(f"{label}\nskew={sk:+.2f}, ex.kurt={ku:.2f}")
        ax.set_xlabel("Monthly Return (%)")
        ax.set_ylabel("Density")
        ax.legend(framealpha=0.9, fontsize=7.5)

    watermark(fig)
    savefig(fig, "cu_return_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — QQ plot vs Normal
# ══════════════════════════════════════════════════════════════════════════════

def fig_qq_plot(br, er):
    base_ret = br["final_returns"].dropna() * 100
    exp_ret  = er["final_returns"].dropna() * 100

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, (ret, label, color) in zip(axes, [
        (base_ret, "Baseline GTAA", CU_NAVY),
        (exp_ret,  "Expanded GTAA", CU_BLUE),
    ]):
        qq = stats.probplot(ret, dist="norm", fit=True)
        theoretical_q, ordered_vals = qq[0]
        slope, intercept, _ = qq[1]

        ax.scatter(theoretical_q, ordered_vals, s=12, alpha=0.65, color=color, zorder=3)
        fit_line = slope * np.array([theoretical_q.min(), theoretical_q.max()]) + intercept
        ax.plot([theoretical_q.min(), theoretical_q.max()], fit_line,
                color=CU_GOLD, lw=1.8, ls="--", label="Normal reference")

        ax.set_title(f"Q–Q Plot vs Normal\n{label}")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles (%)")
        ax.legend(framealpha=0.9)

    watermark(fig)
    savefig(fig, "cu_qq_plot.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Sleeve annual return contribution (stacked bar)
# ══════════════════════════════════════════════════════════════════════════════

def fig_sleeve_contribution(br, er):
    eq_ret  = br["equity_fmp_returns"].loc["2004-11-30":].dropna()
    com_ret = br["commodity_fmp_returns"].dropna()
    fi_ret  = br["fi_fmp_returns"].loc["2004-11-30":].dropna()

    sleeve_rets = pd.DataFrame({
        "Equity Mom":    eq_ret,
        "Commodity Mom": com_ret,
        "FI Carry":      fi_ret,
    }).dropna()

    # Each sleeve contributes 1/3 of its return to the equally-weighted combo
    contrib = sleeve_rets / 3
    contrib_annual = contrib.resample("YE").apply(lambda x: (1+x).prod() - 1) * 100

    colors = [CU_NAVY, CU_BLUE, CU_GOLD]
    years  = contrib_annual.index.year
    x      = np.arange(len(years))

    fig, ax = plt.subplots(figsize=(13, 5))

    bottom_pos = np.zeros(len(years))
    bottom_neg = np.zeros(len(years))

    for col, c in zip(contrib_annual.columns, colors):
        vals = contrib_annual[col].values
        pos_vals = np.where(vals > 0, vals, 0)
        neg_vals = np.where(vals < 0, vals, 0)
        ax.bar(x, pos_vals, 0.65, bottom=bottom_pos, color=c, alpha=0.85, label=col)
        ax.bar(x, neg_vals, 0.65, bottom=bottom_neg, color=c, alpha=0.85)
        bottom_pos += pos_vals
        bottom_neg += neg_vals

    # Total line
    total_annual = (br["final_returns"].dropna().resample("YE")
                    .apply(lambda x: (1+x).prod()-1) * 100)
    ax.plot(x, total_annual.values, "o-", color=CU_RED, lw=1.6,
            ms=5, label="Final GTAA total", zorder=5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, ha="right")
    ax.set_ylabel("Return Contribution (%)")
    ax.set_title("Annual Return Attribution — Baseline GTAA (Sleeve Contributions)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.legend(loc="upper right", framealpha=0.9)

    watermark(fig)
    savefig(fig, "cu_baseline_attribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — Expanded sleeve attribution (all 6 sleeves × 2 families)
# ══════════════════════════════════════════════════════════════════════════════

def fig_expanded_attribution(er):
    sl = er["sleeve_returns"]
    sleeve_map = {
        "Equity Mom":    ("cross_asset_momentum", 0.25, sl["equity_momentum"]),
        "Commodity Mom": ("cross_asset_momentum", 0.25, sl["commodity_momentum"]),
        "FI Mom":        ("cross_asset_momentum", 0.25, sl["fixed_income_momentum"]),
        "FX Mom":        ("cross_asset_momentum", 0.25, sl["fx_momentum"]),
        "FI Carry":      ("cross_asset_carry",    0.50, sl["fixed_income_carry"]),
        "FX Carry":      ("cross_asset_carry",    0.50, sl["fx_carry"]),
    }
    family_weight = 0.50

    contrib = {}
    for name, (fam, wt, r) in sleeve_map.items():
        contrib[name] = r.dropna() * wt * family_weight

    contrib_df = pd.DataFrame(contrib).dropna()
    contrib_ann = contrib_df.resample("YE").apply(lambda x: (1+x).prod()-1) * 100

    mom_colors   = [CU_NAVY, CU_BLUE, CU_LIGHT, "#4A90D9"]
    carry_colors = [CU_GOLD, "#E8A030"]
    all_colors   = mom_colors + carry_colors

    years = contrib_ann.index.year
    x     = np.arange(len(years))

    fig, ax = plt.subplots(figsize=(14, 5))

    bottom_pos = np.zeros(len(years))
    bottom_neg = np.zeros(len(years))

    for col, c in zip(contrib_ann.columns, all_colors):
        vals = contrib_ann[col].values
        pos_vals = np.where(vals > 0, vals, 0)
        neg_vals = np.where(vals < 0, vals, 0)
        ax.bar(x, pos_vals, 0.65, bottom=bottom_pos, color=c, alpha=0.85, label=col)
        ax.bar(x, neg_vals, 0.65, bottom=bottom_neg, color=c, alpha=0.85)
        bottom_pos += pos_vals
        bottom_neg += neg_vals

    total_ann = (er["final_returns"].dropna().resample("YE")
                 .apply(lambda x: (1+x).prod()-1) * 100)
    valid_years = [i for i, yr in enumerate(years) if yr in total_ann.index.year]
    total_vals  = [total_ann[total_ann.index.year == yr].values[0] for yr in years if yr in total_ann.index.year]
    ax.plot([x[i] for i in valid_years], total_vals, "o-", color=CU_RED,
            lw=1.6, ms=5, label="Final GTAA total", zorder=5)

    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, ha="right")
    ax.set_ylabel("Return Contribution (%)")
    ax.set_title("Annual Return Attribution — Expanded GTAA (6 Sleeve Contributions)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.legend(loc="upper right", framealpha=0.9, ncol=3)

    watermark(fig)
    savefig(fig, "cu_expanded_attribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 10 — Rolling 24m correlation between Momentum and Carry families
# ══════════════════════════════════════════════════════════════════════════════

def fig_family_rolling_corr(er):
    mom_fam   = er["factor_family_returns"]["cross_asset_momentum"].dropna()
    carry_fam = er["factor_family_returns"]["cross_asset_carry"].dropna()

    common = mom_fam.index.intersection(carry_fam.index)
    roll_corr = (
        pd.DataFrame({"mom": mom_fam.loc[common], "carry": carry_fam.loc[common]})
        .rolling(24)
        .corr()
        .unstack()["mom"]["carry"]
    )

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(roll_corr.index, roll_corr, color=CU_NAVY, lw=1.6)
    ax.fill_between(roll_corr.index, roll_corr, 0,
                    where=roll_corr > 0, alpha=0.15, color=CU_RED)
    ax.fill_between(roll_corr.index, roll_corr, 0,
                    where=roll_corr < 0, alpha=0.15, color=CU_GREEN)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.axhline(roll_corr.mean(), color=CU_GOLD, lw=1.2, ls="--",
               label=f"Mean = {roll_corr.mean():.2f}")
    shade_recessions(ax)
    set_report_start(ax)
    ax.set_title("Rolling 24-Month Correlation: Momentum Family vs Carry Family")
    ax.set_ylabel("Correlation")
    ax.legend(framealpha=0.9)
    ax.set_ylim(-1, 1)

    watermark(fig)
    savefig(fig, "cu_family_rolling_corr.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 11 — Momentum signal cross-section snapshot (equity + commodity)
# ══════════════════════════════════════════════════════════════════════════════

def fig_signal_snapshot(br, er):
    eq_sig = br["equity_signal"].dropna(how="all").iloc[-1].sort_values(ascending=True)
    com_sig_raw = er["signals"]["commodity_momentum"].dropna(how="all").iloc[-1]
    com_sig = com_sig_raw.sort_values(ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (sig, label) in zip(axes, [
        (eq_sig,  "Equity 12-1 Momentum Signal (latest cross-section)"),
        (com_sig, "Commodity 12-1 Momentum Signal (latest cross-section)"),
    ]):
        colors = [CU_NAVY if v >= 0 else CU_RED for v in sig.values]
        ax.barh(range(len(sig)), sig.values, color=colors, alpha=0.85)
        ax.set_yticks(range(len(sig)))
        ax.set_yticklabels(sig.index, fontsize=8)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Signal (cumulative return t−12 to t−2)")

        long_label  = mpatches.Patch(color=CU_NAVY, alpha=0.85, label="Long (positive signal)")
        short_label = mpatches.Patch(color=CU_RED,  alpha=0.85, label="Short (negative signal)")
        ax.legend(handles=[long_label, short_label], fontsize=8, framealpha=0.9)

    watermark(fig)
    savefig(fig, "cu_signal_snapshot.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 12 — Turnover comparison (baseline vs expanded)
# ══════════════════════════════════════════════════════════════════════════════

def fig_turnover(br, er):
    base_to = br["final_turnover"].dropna() * 100
    exp_to  = er["final_turnover"].dropna() * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, (to, label, color) in zip(axes, [
        (base_to, f"Baseline GTAA\nMean = {base_to.mean():.1f}%/month ({base_to.mean()*12:.0f}%/yr)", CU_NAVY),
        (exp_to,  f"Expanded GTAA\nMean = {exp_to.mean():.1f}%/month ({exp_to.mean()*12:.0f}%/yr)",  CU_BLUE),
    ]):
        ax.bar(to.index, to, width=20, color=color, alpha=0.7, zorder=3)
        ax.axhline(to.mean(), color=CU_GOLD, lw=1.8, ls="--",
                   label=f"Mean = {to.mean():.1f}%")
        shade_recessions(ax)
        ax.set_title(label)
        ax.set_ylabel("Monthly Two-Way Turnover (%)")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.legend(framealpha=0.9)

    fig.suptitle("Portfolio Turnover", fontsize=12, fontweight="bold", y=1.01)
    watermark(fig)
    savefig(fig, "cu_turnover.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 13 — Underwater chart with rolling max DD
# ══════════════════════════════════════════════════════════════════════════════

def fig_underwater(br, er):
    base_ret = br["final_returns"].dropna()
    exp_ret  = er["final_returns"].dropna()
    common_start = max(base_ret.index[0], exp_ret.index[0])

    def rolling_max_dd(r, window=24):
        result = []
        for i in range(len(r)):
            start = max(0, i - window + 1)
            sub = r.iloc[start:i+1]
            cum = (1 + sub).cumprod()
            dd = ((cum / cum.cummax()) - 1).min()
            result.append(dd)
        return pd.Series(result, index=r.index) * 100

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                             gridspec_kw={"hspace": 0.06})

    ax = axes[0]
    dd_base = rolling_drawdown(base_ret) * 100
    dd_exp  = rolling_drawdown(exp_ret.loc[common_start:]) * 100
    ax.fill_between(dd_base.index, dd_base, 0, alpha=0.55, color=CU_NAVY, label="Baseline GTAA")
    ax.fill_between(dd_exp.index,  dd_exp,  0, alpha=0.45, color=CU_BLUE, label="Expanded GTAA")
    shade_recessions(ax)
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Underwater (Drawdown) Chart")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.legend(loc="lower right", framealpha=0.9)
    set_report_start(ax)

    ax2 = axes[1]
    print("  Computing rolling 24m max drawdown (may take a moment)…")
    rmdd_base = rolling_max_dd(base_ret, 24)
    rmdd_exp  = rolling_max_dd(exp_ret.loc[common_start:], 24)
    ax2.plot(rmdd_base.index, rmdd_base, color=CU_NAVY, lw=1.4, label="Baseline 24m rolling max DD")
    ax2.plot(rmdd_exp.index,  rmdd_exp,  color=CU_BLUE, lw=1.4, label="Expanded 24m rolling max DD")
    ax2.fill_between(rmdd_base.index, rmdd_base, 0, alpha=0.12, color=CU_NAVY)
    shade_recessions(ax2)
    ax2.set_ylabel("24m Max Drawdown (%)")
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax2.legend(loc="lower right", framealpha=0.9)

    watermark(fig)
    savefig(fig, "cu_underwater.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 14 — IR bar comparison across all sleeves + families
# ══════════════════════════════════════════════════════════════════════════════

def fig_ir_comparison(br, er):
    from gtaa.analytics.performance import information_ratio, annualized_arithmetic_return

    entries = [
        ("Eq Mom\n(baseline)",  br["equity_fmp_returns"].dropna()),
        ("Com Mom\n(baseline)", br["commodity_fmp_returns"].dropna()),
        ("FI Carry\n(baseline)", br["fi_fmp_returns"].dropna()),
        ("Final\n(baseline)",   br["final_returns"].dropna()),
        ("Eq Mom\n(expanded)",  er["sleeve_returns"]["equity_momentum"].dropna()),
        ("Com Mom\n(expanded)", er["sleeve_returns"]["commodity_momentum"].dropna()),
        ("FI Mom",              er["sleeve_returns"]["fixed_income_momentum"].dropna()),
        ("FX Mom",              er["sleeve_returns"]["fx_momentum"].dropna()),
        ("FI Carry\n(expanded)", er["sleeve_returns"]["fixed_income_carry"].dropna()),
        ("FX Carry",            er["sleeve_returns"]["fx_carry"].dropna()),
        ("Mom Family",          er["factor_family_returns"]["cross_asset_momentum"].dropna()),
        ("Carry Family",        er["factor_family_returns"]["cross_asset_carry"].dropna()),
        ("Final\n(expanded)",   er["final_returns"].dropna()),
    ]

    labels = [e[0] for e in entries]
    irs    = [information_ratio(e[1]) for e in entries]

    palette = (
        [CU_NAVY] * 3 + [CU_RED] +   # baseline sleeves + final
        [CU_BLUE] * 6 +               # expanded sleeves
        [CU_LIGHT, CU_GOLD] +         # families
        ["#C4242B"]                    # expanded final
    )

    fig, ax = plt.subplots(figsize=(15, 5))
    bars = ax.bar(range(len(labels)), irs,
                  color=[CU_NAVY if v >= 0 else CU_RED for v in irs],
                  alpha=0.85, zorder=3)

    # Override colors to match palette
    for bar, c in zip(bars, palette):
        bar.set_facecolor(c)

    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(3.5, color=CU_GREY, lw=1.0, ls="--", alpha=0.5)
    ax.axvline(9.5, color=CU_GREY, lw=1.0, ls="--", alpha=0.5)
    ax.text(1.5, ax.get_ylim()[0] * 0.7, "Baseline", color=CU_GREY,
            ha="center", fontsize=8)
    ax.text(7.0, ax.get_ylim()[0] * 0.7, "Expanded Sleeves", color=CU_GREY,
            ha="center", fontsize=8)

    for i, v in enumerate(irs):
        ax.text(i, v + (0.01 if v >= 0 else -0.03), f"{v:.2f}",
                ha="center", fontsize=7.5, color="black")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Information Ratio (full sample each)")
    ax.set_title("Information Ratio — All Sleeves, Families, and Final Portfolios")

    watermark(fig)
    savefig(fig, "cu_ir_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 15 — 6-panel expanded sleeve growth (report-compatible start date)
# ══════════════════════════════════════════════════════════════════════════════

def fig_expanded_sleeve_growth(er):
    sl = er["sleeve_returns"]
    entries = [
        ("equity_momentum",       "Equity Momentum",       CU_NAVY),
        ("commodity_momentum",    "Commodity Momentum",    "#FF8C00"),
        ("fixed_income_momentum", "Fixed Income Momentum", CU_GREEN),
        ("fx_momentum",           "FX Momentum",           "#E61B4B"),
        ("fixed_income_carry",    "FI Carry",              "#7A0E96"),
        ("fx_carry",              "FX Carry",              "#8B4513"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True,
                             gridspec_kw={"hspace": 0.35, "wspace": 0.25})

    for ax, (key, name, color) in zip(axes.flat, entries):
        r = sl[key].dropna()
        cum = cumulative(r)
        ax.plot(cum.index, cum, color=color, lw=1.4)
        ax.axhline(1, color="k", lw=0.6, ls="--")
        shade_recessions(ax)
        set_report_start(ax)
        ax.set_title(name, fontsize=10)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))

    fig.suptitle("Growth of $1 — Individual Sleeves (1% vol target each)",
                 fontsize=12, fontweight="bold", y=1.01)
    watermark(fig)
    savefig(fig, "expanded_sleeve_growth.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 16 — 6-panel expanded sleeve growth (Columbia palette)
# ══════════════════════════════════════════════════════════════════════════════

def fig_expanded_sleeves(er):
    sl = er["sleeve_returns"]
    entries = [
        ("equity_momentum",       "Equity Momentum",       CU_NAVY),
        ("commodity_momentum",    "Commodity Momentum",    CU_BLUE),
        ("fixed_income_momentum", "FI Momentum",           CU_LIGHT),
        ("fx_momentum",           "FX Momentum",           CU_GOLD),
        ("fixed_income_carry",    "FI Carry",              CU_GREEN),
        ("fx_carry",              "FX Carry",              CU_GREY),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True,
                             gridspec_kw={"hspace": 0.35, "wspace": 0.25})

    for ax, (key, name, color) in zip(axes.flat, entries):
        r  = sl[key].dropna()
        cum = cumulative(r)
        ax.plot(cum.index, cum, color=color, lw=1.4)
        ax.fill_between(cum.index, cum, 1,
                        where=cum >= 1, alpha=0.12, color=color)
        ax.fill_between(cum.index, cum, 1,
                        where=cum < 1,  alpha=0.12, color=CU_RED)
        ax.axhline(1, color="k", lw=0.6, ls="--")
        shade_recessions(ax)
        set_report_start(ax)

        from gtaa.analytics.performance import information_ratio, max_drawdown
        ir  = information_ratio(r)
        mdd = max_drawdown(r) * 100
        ax.set_title(f"{name}\nIR = {ir:+.2f}  |  Max DD = {mdd:.1f}%", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))

    fig.suptitle("Growth of $1 — Expanded GTAA Sleeves (1% vol target each)",
                 fontsize=12, fontweight="bold", y=1.01)
    watermark(fig)
    savefig(fig, "cu_expanded_sleeves.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 17 — Cross-asset correlation heatmap (Columbia diverging palette)
# ══════════════════════════════════════════════════════════════════════════════

def fig_cross_asset_corr(er):
    corr = er["final_corr"]
    if corr is None or corr.empty:
        return

    cu_div = sns.diverging_palette(220, 10, s=80, l=50, sep=10, as_cmap=True)

    fig, ax = plt.subplots(figsize=(18, 15))
    mask = corr.isnull()
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap=cu_div,
        vmin=-1, vmax=1, center=0,
        linewidths=0.3, linecolor="#D0D0D0",
        mask=mask,
        annot_kws={"size": 6.5},
        cbar_kws={"label": "Pearson Correlation", "shrink": 0.6},
        ax=ax,
    )
    ax.set_title("Cross-Asset Correlation Matrix (36-month rolling, as of final date)",
                 pad=12)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    watermark(fig)
    savefig(fig, "cu_cross_asset_corr.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 18 — Final weight allocation heatmap over time
# ══════════════════════════════════════════════════════════════════════════════

def fig_weight_heatmap(br):
    fw = br["final_weights"].loc["2004-11-01":].dropna(how="all")

    # Subsample to annual for readability
    fw_annual = fw.resample("YE").last()
    fw_annual.index = fw_annual.index.year

    cu_div = sns.diverging_palette(10, 220, s=80, l=50, sep=10, as_cmap=True)

    fig, ax = plt.subplots(figsize=(18, 6))
    sns.heatmap(
        fw_annual.T * 100,
        cmap=cu_div, center=0,
        linewidths=0.3, linecolor="#EEEEEE",
        cbar_kws={"label": "Weight (%)", "shrink": 0.6},
        annot=True, fmt=".1f", annot_kws={"size": 6.5},
        ax=ax,
    )
    ax.set_title("Baseline GTAA — Year-End Asset Weights (%)", pad=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Asset")
    ax.tick_params(axis="y", rotation=0)
    ax.tick_params(axis="x", rotation=0)

    watermark(fig)
    savefig(fig, "cu_weight_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 19 — Summary 2×3 performance dashboard
# ══════════════════════════════════════════════════════════════════════════════

def fig_dashboard(br, er):
    base_ret = br["final_returns"].dropna()
    exp_ret  = er["final_returns"].dropna()
    common_start = max(base_ret.index[0], exp_ret.index[0])
    base_cs = base_ret.loc[common_start:]
    exp_cs  = exp_ret.loc[common_start:]

    mom_fam   = er["factor_family_returns"]["cross_asset_momentum"].dropna()
    carry_fam = er["factor_family_returns"]["cross_asset_carry"].dropna()

    fig = plt.figure(figsize=(16, 11))
    gs  = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    # ── (0,0:2) Cumulative returns ─────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :2])
    ax0.plot(cumulative(base_ret), color=CU_NAVY, lw=1.8, label="Baseline GTAA")
    ax0.plot(cumulative(exp_cs),   color=CU_BLUE,  lw=1.8, label="Expanded GTAA")
    ax0.plot(cumulative(carry_fam.loc[common_start:]), color=CU_GOLD,
             lw=1.1, ls="--", label="Carry Family")
    ax0.axhline(1, color="k", lw=0.5, ls="--", alpha=0.4)
    shade_recessions(ax0)
    ax0.set_title("Growth of $1 (aligned sample)")
    ax0.legend(fontsize=7.5, framealpha=0.9)
    ax0.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
    set_report_start(ax0)

    # ── (0,2) IR bar chart ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 2])
    from gtaa.analytics.performance import information_ratio
    labels_ = ["Base\n3-sl.", "Exp\n6-sl.", "Mom\nFam.", "Carry\nFam."]
    irs_    = [information_ratio(base_cs), information_ratio(exp_cs),
               information_ratio(mom_fam.loc[common_start:]),
               information_ratio(carry_fam.loc[common_start:])]
    colors_ = [CU_NAVY, CU_BLUE, CU_GREY, CU_GOLD]
    bars_   = ax1.bar(labels_, irs_, color=colors_, alpha=0.85)
    for bar, v in zip(bars_, irs_):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                 f"{v:.2f}", ha="center", fontsize=8)
    ax1.axhline(0, color="k", lw=0.6)
    ax1.set_title("Information Ratio\n(common sample)")
    ax1.set_ylabel("IR")

    # ── (1,0:2) Drawdown ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.fill_between(base_ret.index, rolling_drawdown(base_ret) * 100,
                     alpha=0.5, color=CU_NAVY, label="Baseline")
    ax2.fill_between(exp_cs.index, rolling_drawdown(exp_cs) * 100,
                     alpha=0.45, color=CU_BLUE, label="Expanded")
    shade_recessions(ax2)
    ax2.set_title("Drawdown")
    ax2.set_ylabel("DD (%)")
    ax2.legend(fontsize=7.5)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    set_report_start(ax2)

    # ── (1,2) Return distribution overlay ─────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 2])
    for r, lbl, c in [(base_cs*100, "Baseline", CU_NAVY),
                      (exp_cs*100, "Expanded", CU_BLUE)]:
        ax3.hist(r, bins=28, alpha=0.45, density=True, color=c, label=lbl)
        kde = stats.gaussian_kde(r)
        x_  = np.linspace(r.min()-0.1, r.max()+0.1, 200)
        ax3.plot(x_, kde(x_), color=c, lw=1.5)
    ax3.set_title("Return Distribution")
    ax3.set_xlabel("Monthly Return (%)")
    ax3.legend(fontsize=7.5)

    # ── (2,0:2) Rolling 24m IR ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :2])
    def roll_ir(r, w=24):
        return r.rolling(w).mean() * 12 / (r.rolling(w).std(ddof=0) * np.sqrt(12))
    ax4.plot(roll_ir(base_ret), color=CU_NAVY, lw=1.4, label="Baseline")
    ax4.plot(roll_ir(exp_ret.loc[common_start:]), color=CU_BLUE, lw=1.4, label="Expanded")
    ax4.axhline(0, color="k", lw=0.6, ls="--")
    shade_recessions(ax4)
    ax4.set_title("Rolling 24-Month IR")
    ax4.set_ylabel("IR")
    ax4.legend(fontsize=7.5)
    set_report_start(ax4)

    # ── (2,2) Turnover histogram ────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.hist(br["final_turnover"].dropna()*100, bins=25,
             alpha=0.6, color=CU_NAVY, label="Baseline")
    ax5.hist(er["final_turnover"].dropna()*100, bins=25,
             alpha=0.5, color=CU_BLUE, label="Expanded")
    ax5.set_title("Turnover Distribution")
    ax5.set_xlabel("Monthly TO (%)")
    ax5.legend(fontsize=7.5)

    fig.suptitle("GTAA Factor Portfolio — Performance Dashboard",
                 fontsize=14, fontweight="bold", y=1.01)
    watermark(fig)
    savefig(fig, "cu_dashboard.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 20 — Expanded strategy raw weight panels (6 sleeves, 2×3 grid)
# ══════════════════════════════════════════════════════════════════════════════

_SLEEVE_PALETTES = {
    "equity_momentum":       ["#003087","#E63946","#F2A900","#2D7D46","#9B59B6",
                               "#00B4D8","#FF6B35","#1A936F","#C77DFF","#75AADB"],
    "commodity_momentum":    ["#003087","#E63946","#F2A900","#2D7D46","#9B59B6",
                               "#00B4D8","#FF6B35","#1A936F","#C77DFF","#75AADB",
                               "#6C6C6C","#C4242B"],
    "fixed_income_momentum": ["#003087","#E63946","#F2A900","#2D7D46","#9B59B6",
                               "#00B4D8","#FF6B35","#1A936F","#C77DFF","#75AADB"],
    "fx_momentum":           ["#003087","#E63946","#F2A900","#2D7D46","#9B59B6","#00B4D8"],
    "fixed_income_carry":    ["#003087","#E63946","#F2A900","#2D7D46","#9B59B6",
                               "#00B4D8","#FF6B35","#1A936F","#C77DFF","#75AADB"],
    "fx_carry":              ["#003087","#E63946","#F2A900","#2D7D46","#9B59B6","#00B4D8"],
}
_SLEEVE_LABELS = {
    "equity_momentum": "Equity Momentum",
    "commodity_momentum": "Commodity Momentum",
    "fixed_income_momentum": "FI Momentum",
    "fx_momentum": "FX Momentum",
    "fixed_income_carry": "FI Carry",
    "fx_carry": "FX Carry",
}
_LS_CYCLE = ["-", "--", "-.", ":", (0,(3,1,1,1)), (0,(5,1))]


def _plot_sleeve_weights(ax, weights, sleeve_key, title, ylabel="Weight"):
    w = weights.dropna(how="all")
    cols = list(w.columns)
    palette = _SLEEVE_PALETTES.get(sleeve_key, [CU_NAVY] * len(cols))
    for i, col in enumerate(cols):
        c  = palette[i % len(palette)]
        ls = _LS_CYCLE[i % len(_LS_CYCLE)]
        lw = 1.2 if ls == "-" else 1.0
        ax.plot(w.index, w[col], color=c, lw=lw, ls=ls, alpha=0.88, label=col)
    ax.axhline(0, color="#999", lw=0.6, ls="--")
    shade_recessions(ax)
    ax.set_title(title, fontsize=9.5, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=8)
    ax.legend(fontsize=6.5, ncol=2, loc="upper right",
              framealpha=0.7, borderpad=0.4, labelspacing=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def fig_expanded_raw_weights(er):
    import matplotlib.dates as mdates  # noqa: F401 — needed inside helper
    global mdates
    import matplotlib.dates as mdates

    sleeves = list(er["raw_weights"].keys())
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=False)
    fig.suptitle("Expanded Strategy — Raw Factor Weights (Pre-Scaling)", fontsize=13,
                 fontweight="bold", y=1.01)

    for ax, sk in zip(axes.flat, sleeves):
        _plot_sleeve_weights(ax, er["raw_weights"][sk], sk,
                             _SLEEVE_LABELS[sk], ylabel="Raw Weight")

    plt.tight_layout(h_pad=2.0, w_pad=1.5)
    watermark(fig)
    savefig(fig, "cu_expanded_raw_weights.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 21 — Expanded strategy 1%-vol-scaled weight panels (6 sleeves, 2×3)
# ══════════════════════════════════════════════════════════════════════════════

def fig_expanded_scaled_weights(er):
    import matplotlib.dates as mdates

    sleeves = list(er["sleeve_scaled_weights"].keys())
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=False)
    fig.suptitle("Expanded Strategy — 1% Vol-Scaled Weights (Per Sleeve)", fontsize=13,
                 fontweight="bold", y=1.01)

    for ax, sk in zip(axes.flat, sleeves):
        _plot_sleeve_weights(ax, er["sleeve_scaled_weights"][sk], sk,
                             _SLEEVE_LABELS[sk], ylabel="Scaled Weight")

    plt.tight_layout(h_pad=2.0, w_pad=1.5)
    watermark(fig)
    savefig(fig, "cu_expanded_scaled_weights.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 22 — Expanded final portfolio weights (all 38 assets) + gross exposure
# ══════════════════════════════════════════════════════════════════════════════

def fig_expanded_final_weights(er):
    import matplotlib.dates as mdates

    fw = er["final_weights"].dropna(how="all")

    _ALL_COLORS = [
        "#003087","#E63946","#F2A900","#2D7D46","#9B59B6","#00B4D8","#FF6B35",
        "#1A936F","#C77DFF","#75AADB","#6C6C6C","#C4242B","#0077B6","#FFC300",
        "#DAA520","#8B008B","#20B2AA","#DC143C","#4682B4","#FF8C00","#006400",
        "#8B4513","#708090","#2E8B57","#B8860B","#483D8B","#CD853F","#5F9EA0",
        "#BC8F8F","#4169E1","#DB7093","#BDB76B","#008080","#D2691E","#9400D3",
        "#556B2F","#FF1493","#1C1C1C",
    ]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(16, 9),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06}
    )
    fig.suptitle("Final Portfolio Weights — 1% Volatility Target (All 38 Assets, Expanded)",
                 fontsize=12, fontweight="bold")

    cols = list(fw.columns)
    for i, col in enumerate(cols):
        c  = _ALL_COLORS[i % len(_ALL_COLORS)]
        ls = _LS_CYCLE[i % len(_LS_CYCLE)]
        lw = 1.1 if ls == "-" else 0.9
        ax_top.plot(fw.index, fw[col], color=c, lw=lw, ls=ls, alpha=0.75, label=col)

    ax_top.axhline(0, color="#888", lw=0.7, ls="--")
    shade_recessions(ax_top)
    ax_top.set_ylabel("Portfolio Weight")
    ax_top.legend(fontsize=5.5, ncol=6, loc="upper right",
                  framealpha=0.7, borderpad=0.3, labelspacing=0.2)
    ax_top.xaxis.set_major_locator(mdates.YearLocator(4))
    ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    gross = fw.abs().sum(axis=1)
    ax_bot.fill_between(gross.index, gross.values, color=CU_BLUE, alpha=0.55)
    ax_bot.set_ylabel("|Gross Exp.|")
    ax_bot.set_ylim(bottom=0)
    shade_recessions(ax_bot)
    ax_bot.xaxis.set_major_locator(mdates.YearLocator(4))
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    watermark(fig)
    savefig(fig, "cu_expanded_final_weights.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 23 — 6-sleeve weight panels (weights + gross exposure per sleeve)
# ══════════════════════════════════════════════════════════════════════════════

def fig_expanded_sleeve_detail(er):
    import matplotlib.dates as mdates

    sleeves = list(er["sleeve_scaled_weights"].keys())
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        "Expanded Strategy — 1% Vol-Scaled Weights by Sleeve",
        fontsize=13, fontweight="bold", y=1.005
    )

    outer = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28)

    EXP_START = pd.Timestamp("2005-12-01")

    for idx, sk in enumerate(sleeves):
        row, col = divmod(idx, 3)
        inner = outer[row, col].subgridspec(
            2, 1, height_ratios=[3, 1], hspace=0.06
        )
        ax_top = fig.add_subplot(inner[0])
        ax_bot = fig.add_subplot(inner[1], sharex=ax_top)

        w = er["sleeve_scaled_weights"][sk].dropna(how="all").loc[EXP_START:]
        palette = _SLEEVE_PALETTES.get(sk, [CU_NAVY] * len(w.columns))
        for i, col_name in enumerate(w.columns):
            c  = palette[i % len(palette)]
            ls = _LS_CYCLE[i % len(_LS_CYCLE)]
            lw = 1.2 if ls == "-" else 1.0
            ax_top.plot(w.index, w[col_name], color=c, lw=lw, ls=ls,
                        alpha=0.88, label=col_name)

        ax_top.axhline(0, color="#999", lw=0.6, ls="--")
        shade_recessions(ax_top)
        ax_top.set_title(_SLEEVE_LABELS[sk], fontsize=9.5, fontweight="bold", pad=4)
        ax_top.set_ylabel("Weight", fontsize=7.5)
        ax_top.set_xlim(left=EXP_START)
        ax_top.legend(fontsize=5.5, ncol=2, loc="upper right",
                      framealpha=0.65, borderpad=0.3, labelspacing=0.25)
        ax_top.xaxis.set_major_locator(mdates.YearLocator(4))
        ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax_top.get_xticklabels(), visible=False)

        gross = w.abs().sum(axis=1)
        ax_bot.fill_between(gross.index, gross.values, color=CU_BLUE, alpha=0.55)
        ax_bot.set_ylabel("|Gross|", fontsize=7)
        ax_bot.set_ylim(bottom=0)
        ax_bot.set_xlim(left=EXP_START)
        shade_recessions(ax_bot)
        ax_bot.xaxis.set_major_locator(mdates.YearLocator(4))
        ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_bot.tick_params(axis="x", labelsize=7)

    watermark(fig)
    savefig(fig, "cu_expanded_sleeve_detail.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 24 — 2 factor family weight panels (weights + gross exposure)
# ══════════════════════════════════════════════════════════════════════════════

def fig_expanded_family_detail(er):
    import matplotlib.dates as mdates

    families = {
        "cross_asset_momentum": ("Cross-Asset Momentum Family", CU_NAVY),
        "cross_asset_carry":    ("Cross-Asset Carry Family",    CU_GOLD),
    }

    fig = plt.figure(figsize=(18, 9))
    fig.suptitle(
        "Expanded Strategy — Factor Family Weights (1% Vol-Scaled)",
        fontsize=13, fontweight="bold", y=1.005
    )

    outer = fig.add_gridspec(1, 2, hspace=0.35, wspace=0.22)

    family_keys = list(families.keys())
    for idx, fk in enumerate(family_keys):
        label, base_color = families[fk]
        inner = outer[idx].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.07)
        ax_top = fig.add_subplot(inner[0])
        ax_bot = fig.add_subplot(inner[1], sharex=ax_top)

        w = er["factor_family_weights"][fk].dropna(how="all")
        cols = list(w.columns)
        # Give each asset in the family a distinct colour
        palette = (
            _SLEEVE_PALETTES["equity_momentum"][:len(cols)]
            if idx == 0
            else _SLEEVE_PALETTES["fixed_income_carry"][:len(cols)]
        )
        for i, col_name in enumerate(cols):
            c  = palette[i % len(palette)]
            ls = _LS_CYCLE[i % len(_LS_CYCLE)]
            lw = 1.2 if ls == "-" else 1.0
            ax_top.plot(w.index, w[col_name], color=c, lw=lw, ls=ls,
                        alpha=0.88, label=col_name)

        ax_top.axhline(0, color="#999", lw=0.6, ls="--")
        shade_recessions(ax_top)
        ax_top.set_title(label, fontsize=10.5, fontweight="bold", pad=5)
        ax_top.set_ylabel("Weight", fontsize=8.5)
        ax_top.legend(fontsize=6, ncol=3, loc="upper right",
                      framealpha=0.7, borderpad=0.35, labelspacing=0.3)
        ax_top.xaxis.set_major_locator(mdates.YearLocator(4))
        ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax_top.get_xticklabels(), visible=False)

        gross = w.abs().sum(axis=1)
        ax_bot.fill_between(gross.index, gross.values,
                            color=base_color, alpha=0.45)
        ax_bot.set_ylabel("|Gross|", fontsize=8)
        ax_bot.set_ylim(bottom=0)
        shade_recessions(ax_bot)
        ax_bot.xaxis.set_major_locator(mdates.YearLocator(4))
        ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_bot.tick_params(axis="x", labelsize=8)

    watermark(fig)
    savefig(fig, "cu_expanded_family_detail.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 25 — All 6 sleeves + 2 families on one cumulative return chart
# ══════════════════════════════════════════════════════════════════════════════

def fig_expanded_all_portfolios(er):
    import matplotlib.dates as mdates

    EXP_START = pd.Timestamp("2005-12-01")

    sleeve_cfg = [
        ("equity_momentum",       "Equity Mom",     CU_NAVY,  "-",          1.8),
        ("commodity_momentum",    "Commodity Mom",  CU_GOLD,  "-",          1.8),
        ("fixed_income_momentum", "FI Mom",         CU_GREEN, "-",          1.8),
        ("fx_momentum",           "FX Mom",         CU_RED,   "-",          1.8),
        ("fixed_income_carry",    "FI Carry",       CU_BLUE,  "-",          1.8),
        ("fx_carry",              "FX Carry",       CU_GREY,  "-",          1.8),
    ]
    family_cfg = [
        ("cross_asset_momentum", "Momentum Family", CU_NAVY,  "--", 2.6),
        ("cross_asset_carry",    "Carry Family",    CU_GOLD,  "--", 2.6),
    ]

    fig, (ax, ax_gross) = plt.subplots(
        2, 1, figsize=(14, 9),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.07}
    )
    fig.suptitle(
        "Expanded Strategy — Portfolio Weights Over Time: 6 Sleeves & 2 Factor Families",
        fontsize=12, fontweight="bold"
    )

    # Top panel: net weight (sum across assets) — shows long/short tilt of each portfolio
    for sk, label, color, ls, lw in sleeve_cfg:
        w = er["sleeve_scaled_weights"][sk].dropna(how="all").loc[EXP_START:]
        net = w.sum(axis=1)          # net exposure — zero by construction for pure L/S
        gross = w.abs().sum(axis=1)  # gross exposure — the meaningful size signal
        ax.plot(gross.index, gross.values, color=color, lw=lw, ls=ls,
                alpha=0.80, label=label)

    for fk, label, color, ls, lw in family_cfg:
        w = er["factor_family_weights"][fk].dropna(how="all").loc[EXP_START:]
        gross = w.abs().sum(axis=1)
        ax.plot(gross.index, gross.values, color=color, lw=lw, ls=ls,
                alpha=1.0, label=label)

    ax.axhline(0, color="#888", lw=0.6, ls="--", alpha=0.4)
    shade_recessions(ax)
    ax.set_ylabel("Gross Exposure (Σ |w|)")
    ax.set_xlim(left=EXP_START)
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.legend(fontsize=8, ncol=4, loc="upper right",
              framealpha=0.75, borderpad=0.5, labelspacing=0.35)

    # Bottom panel: final portfolio gross exposure for scale reference
    fw = er["final_weights"].dropna(how="all").loc[EXP_START:]
    final_gross = fw.abs().sum(axis=1)
    ax_gross.fill_between(final_gross.index, final_gross.values,
                          color=CU_NAVY, alpha=0.35)
    ax_gross.plot(final_gross.index, final_gross.values,
                  color=CU_NAVY, lw=1.2)
    ax_gross.set_ylabel("Final |Gross|", fontsize=8)
    ax_gross.set_ylim(bottom=0)
    ax_gross.set_xlim(left=EXP_START)
    shade_recessions(ax_gross)
    ax_gross.xaxis.set_major_locator(mdates.YearLocator(4))
    ax_gross.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    watermark(fig)
    savefig(fig, "cu_expanded_all_portfolios.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 26 — Raw weights gross exposure: 6 sleeves + 2 families on one chart
# ══════════════════════════════════════════════════════════════════════════════

def fig_expanded_raw_all_portfolios(er):
    """Single chart: individual asset raw weights over time, colour-coded by family."""
    import matplotlib.dates as mdates

    EXP_START = pd.Timestamp("2005-12-01")

    # Momentum family assets + colours
    mom_keys   = ["equity_momentum", "commodity_momentum",
                  "fixed_income_momentum", "fx_momentum"]
    carry_keys = ["fixed_income_carry", "fx_carry"]

    # Distinct palettes per sleeve within each family
    _MOM_PALETTES = {
        "equity_momentum":       ["#003087","#1a5ea8","#2d7fc9","#75AADB",
                                   "#B9D9EB","#0d3d6e","#4a90d9","#0077b6",
                                   "#48cae4","#90e0ef"],
        "commodity_momentum":    ["#F2A900","#e08000","#c86000","#a04000",
                                   "#ff6b35","#ff9f1c","#ffca3a","#d4900a",
                                   "#b36800","#8b4500"],
        "fixed_income_momentum": ["#2D7D46","#3a9e5a","#1f5c33","#4cbb6e",
                                   "#27ae60","#1e8449","#a9dfb2","#52be80",
                                   "#82e0aa","#145a32"],
        "fx_momentum":           ["#C4242B","#e05560","#9b1c22","#ff6b6b"],
    }
    _CARRY_PALETTES = {
        "fixed_income_carry":    ["#6C6C6C","#999999","#444444","#bbbbbb",
                                   "#555555","#888888","#333333","#aaaaaa",
                                   "#777777","#222222"],
        "fx_carry":              ["#9B59B6","#7d3c98","#c39bd3","#6c3483"],
    }

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(16, 10),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06}
    )
    fig.suptitle(
        "Raw Factor Weights — All Assets, Pre-Scaling  |  "
        "Blues/Greens = Momentum Family · Greys/Purples = Carry Family",
        fontsize=11, fontweight="bold"
    )

    def plot_sleeve(ax, sk, palette_map, alpha=0.75):
        w = er["raw_weights"][sk].dropna(how="all").loc[EXP_START:]
        pal = palette_map[sk]
        for i, col in enumerate(w.columns):
            c  = pal[i % len(pal)]
            ls = _LS_CYCLE[i % len(_LS_CYCLE)]
            lw = 1.1 if ls == "-" else 0.9
            ax_top.plot(w.index, w[col].values, color=c, lw=lw, ls=ls,
                        alpha=alpha, label=col)

    for sk in mom_keys:
        plot_sleeve(ax_top, sk, _MOM_PALETTES)
    for sk in carry_keys:
        plot_sleeve(ax_top, sk, _CARRY_PALETTES)

    ax_top.axhline(0, color="#888", lw=0.7, ls="--")
    shade_recessions(ax_top)
    ax_top.set_ylabel("Raw Weight")
    ax_top.set_xlim(left=EXP_START)
    ax_top.xaxis.set_major_locator(mdates.YearLocator(4))
    ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.legend(fontsize=5.5, ncol=8, loc="upper right",
                  framealpha=0.7, borderpad=0.3, labelspacing=0.2)

    # Gross exposure split by family
    mom_gross = pd.concat(
        [er["raw_weights"][k].dropna(how="all") for k in mom_keys], axis=1
    ).loc[EXP_START:].abs().sum(axis=1)
    carry_gross = pd.concat(
        [er["raw_weights"][k].dropna(how="all") for k in carry_keys], axis=1
    ).loc[EXP_START:].abs().sum(axis=1)

    ax_bot.fill_between(mom_gross.index, mom_gross.values,
                        color=CU_NAVY, alpha=0.30, label="Momentum |Gross|")
    ax_bot.fill_between(carry_gross.index, carry_gross.values,
                        color=CU_GREY, alpha=0.35, label="Carry |Gross|")
    ax_bot.set_ylabel("|Gross Exp.|", fontsize=8)
    ax_bot.set_ylim(bottom=0)
    ax_bot.set_xlim(left=EXP_START)
    shade_recessions(ax_bot)
    ax_bot.legend(fontsize=7.5, loc="upper right")
    ax_bot.xaxis.set_major_locator(mdates.YearLocator(4))
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    watermark(fig)
    savefig(fig, "cu_expanded_raw_all_portfolios.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 28 — Corrected performance table cards (simple-return convention)
# ══════════════════════════════════════════════════════════════════════════════

def _metrics_row(name: str, rets: pd.Series, turnover: pd.Series | None = None) -> dict:
    from gtaa.analytics.performance import (
        annualized_arithmetic_return,
        annualized_volatility,
        information_ratio,
        max_drawdown,
        avg_drawdown,
    )

    ann_turnover = float(turnover.mean()) * 12 if turnover is not None and len(turnover) > 0 else float("nan")
    return {
        "Sleeve or Family": name,
        "Ann. return": annualized_arithmetic_return(rets),
        "Ann. vol": annualized_volatility(rets),
        "IR": information_ratio(rets),
        "Max DD": max_drawdown(rets),
        "Avg DD": avg_drawdown(rets),
        "Ann. Turnover": ann_turnover,
    }


def _render_metrics_table(
    title: str,
    rows: list[dict],
    out_name: str,
    *,
    figsize: tuple[float, float],
    highlight_last_row: bool = False,
) -> None:
    cols = ["Sleeve or Family", "Ann. return", "Ann. vol", "IR", "Max DD", "Avg DD", "Ann. Turnover"]

    def _fmt(col: str, val: float) -> str:
        if pd.isna(val):
            return "—"
        if col == "IR":
            return f"{val:.3f}"
        if col == "Ann. Turnover":
            return f"{100 * val:.1f}%"
        return f"{100 * val:.2f}%"

    cell_text = [[_fmt(col, row[col]) if col != "Sleeve or Family" else row[col] for col in cols] for row in rows]
    col_widths = [0.24, 0.12, 0.12, 0.08, 0.12, 0.12, 0.16]

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=14)

    tbl = ax.table(
        cellText=cell_text,
        colLabels=cols,
        colLoc="center",
        cellLoc="center",
        bbox=[0.015, 0.06, 0.97, 0.74],
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.25)

    alt_fill = "#D7E2EE"
    edge = "#7C7C7C"

    n_data_rows = len(rows)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(edge)
        cell.set_linewidth(1.0)
        if r == 0:
            cell.set_facecolor(CU_NAVY)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_fontsize(10.5)
        else:
            is_last = r == n_data_rows and highlight_last_row
            if is_last:
                cell.set_facecolor(CU_NAVY)
                cell.get_text().set_color("white")
                cell.get_text().set_fontweight("bold")
            else:
                cell.set_facecolor("white" if r % 2 == 1 else alt_fill)
                cell.get_text().set_color("#111111")

    fig.savefig(FIGS / out_name, dpi=150, bbox_inches="tight")
    print(f"  saved → {out_name}")
    plt.close(fig)


def fig_expanded_metrics_tables(er):
    from gtaa.portfolio.turnover import compute_turnover

    all_returns = er["data"].all_returns
    sleeves = [
        ("Equity Momentum FMP", "equity_momentum"),
        ("Commodity Momentum FMP", "commodity_momentum"),
        ("Fixed-Income Momentum FMP", "fixed_income_momentum"),
        ("FX Momentum FMP", "fx_momentum"),
        ("Fixed-Income Carry FMP", "fixed_income_carry"),
        ("FX Carry FMP", "fx_carry"),
    ]
    families = [
        ("Cross-Asset Momentum Family", "cross_asset_momentum"),
        ("Cross-Asset Carry Family", "cross_asset_carry"),
    ]

    sleeve_rows = []
    for label, key in sleeves:
        w = er["sleeve_scaled_weights"][key].dropna(how="all")
        r = er["sleeve_returns"][key].dropna()
        to = compute_turnover(w, all_returns.reindex(columns=w.columns))
        sleeve_rows.append(_metrics_row(label, r, to))

    family_rows = []
    for label, key in families:
        w = er["factor_family_weights"][key].dropna(how="all")
        r = er["factor_family_returns"][key].dropna()
        to = compute_turnover(w, all_returns.reindex(columns=w.columns))
        family_rows.append(_metrics_row(label, r, to))

    final_row = _metrics_row("Expanded Final GTAA", er["final_returns"].dropna(), er["final_turnover"].dropna())
    family_rows_with_final = family_rows + [final_row]
    combined_rows = sleeve_rows + family_rows_with_final

    sample_start = er["final_returns"].dropna().index[0].strftime("%b %Y")
    sample_end = er["final_returns"].dropna().index[-1].strftime("%b %Y")

    _render_metrics_table(
        "Expanded Strategy: Family-Level Performance",
        family_rows_with_final,
        "expanded_family_metrics_table.png",
        figsize=(11.2, 3.2),
    )
    _render_metrics_table(
        "Expanded Strategy: Sleeve-Level Performance",
        sleeve_rows,
        "expanded_sleeve_metrics_table.png",
        figsize=(13.5, 4.2),
    )
    _render_metrics_table(
        f"Expanded Strategy: Sleeve, Family, and Portfolio Metrics ({sample_start} – {sample_end})",
        combined_rows,
        "expanded_metrics_table.png",
        figsize=(15.0, 8.6),
        highlight_last_row=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  GTAA Columbia Visualizations")
    print("=" * 60)

    br, er = load_strategies()

    print("\nGenerating figures…")
    fig_hero_cumulative(br, er)
    fig_annual_returns(br, er)
    fig_calendar_heatmap(br, er)
    fig_rolling_ir(br, er)
    fig_rolling_vol(br, er)
    fig_return_distribution(br, er)
    fig_qq_plot(br, er)
    fig_sleeve_contribution(br, er)
    fig_expanded_attribution(er)
    fig_family_rolling_corr(er)
    fig_signal_snapshot(br, er)
    fig_turnover(br, er)
    fig_underwater(br, er)
    fig_ir_comparison(br, er)
    fig_expanded_sleeve_growth(er)
    fig_expanded_sleeves(er)
    fig_cross_asset_corr(er)
    fig_weight_heatmap(br)
    fig_dashboard(br, er)
    fig_expanded_raw_weights(er)
    fig_expanded_scaled_weights(er)
    fig_expanded_final_weights(er)
    fig_expanded_sleeve_detail(er)
    fig_expanded_family_detail(er)
    fig_expanded_all_portfolios(er)
    fig_expanded_raw_all_portfolios(er)
    fig_expanded_metrics_tables(er)

    print(f"\nAll figures saved to: {FIGS}")


if __name__ == "__main__":
    main()
