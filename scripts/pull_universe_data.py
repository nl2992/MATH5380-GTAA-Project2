"""
pull_universe_data.py
─────────────────────────────────────────────────────────────────────────────
Downloads and packages all data needed for the three-sleeve multi-asset FMP:
  1. Equity sleeve   – EWA/EWC/…/SPY prices + returns; MSCI P/E from project file
  2. Fixed-income    – SHV/SHY/…/HYG prices + returns; Treasury/credit yields from FRED
  3. Commodity       – GLD/SLV/…/CORN prices + returns (signal computed later from prices)

Output:
  data/raw/multi_asset_universe.xlsx  — one sheet per data type
  data/raw/multi_asset_universe_meta.csv — ticker → asset-class metadata

Run from the repo root:
  python scripts/pull_universe_data.py

Or pass --out to override the output path:
  python scripts/pull_universe_data.py --out /tmp/universe.xlsx
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW  = REPO_ROOT / "data" / "raw"
P2_FILE   = DATA_RAW / "Data for final project 2 .xlsx"
OUT_FILE  = DATA_RAW / "multi_asset_universe.xlsx"

# ── Universe definitions ───────────────────────────────────────────────────────
EQUITY_ETF_MAP = {
    "EWA": "Australia",
    "EWC": "Canada",
    "EWQ": "France",
    "EWG": "Germany",
    "EWI": "Italy",
    "EWJ": "Japan",
    "EWN": "Netherlands",
    "EWP": "Spain",
    "EWU": "United Kingdom",
    "SPY": "United States",
}

MSCI_TO_ETF = {
    "AUS": "EWA",
    "CAN": "EWC",
    "FRA": "EWQ",
    "GER": "EWG",
    "ITA": "EWI",
    "JPN": "EWJ",
    "NLD": "EWN",
    "ESP": "EWP",
    "GBR": "EWU",
    "USA": "SPY",
}

FI_ETF_MAP = {
    "SHV":  "Treasury 0-1Y (T-bills)",
    "SHY":  "Treasury 1-3Y",
    "IEI":  "Treasury 3-7Y",
    "IEF":  "Treasury 7-10Y",
    "TLH":  "Treasury 10-20Y",
    "TLT":  "Treasury 20Y+",
    "VTIP": "Short TIPS (real rate)",
    "TIP":  "Broad TIPS (real rate)",
    "LQD":  "Investment-Grade Corp Bonds",
    "HYG":  "High-Yield Corp Bonds",
}

FI_YIELD_MAP: dict[str, str] = {
    # FRED series ID → human label (maps to FI ETF carry signals)
    # Treasuries: constant-maturity yields from Fed H.15 (full history ~1962+)
    "DGS3MO": "Treasury 3-Month Yield (%)",
    "DGS1":   "Treasury 1-Year Yield (%)",
    "DGS2":   "Treasury 2-Year Yield (%)",
    "DGS5":   "Treasury 5-Year Yield (%)",
    "DGS7":   "Treasury 7-Year Yield (%)",
    "DGS10":  "Treasury 10-Year Yield (%)",
    "DGS20":  "Treasury 20-Year Yield (%)",
    "DGS30":  "Treasury 30-Year Yield (%)",
    # TIPS real yields (from 2003)
    "DFII5":  "TIPS 5-Year Real Yield (%)",
    "DFII10": "TIPS 10-Year Real Yield (%)",
    # Credit: Moody's corporate bond yields — free, full history (1983/1986+)
    # Note: ICE BofA series moved behind FRED paywall in 2023; Moody's is the
    # best freely available proxy.
    "DAAA":   "Moody's Aaa Corp Yield (%) [IG reference]",
    "DBAA":   "Moody's Baa Corp Yield (%) [IG carry proxy → LQD]",
    "BAA10YM":"Baa-10Y Credit Spread (%) [monthly, HY proxy → HYG]",
    # ICE BofA series — FRED only provides from 2023-05; included for completeness
    "BAMLC0A0CMEY":  "ICE BofA IG Corp Eff Yield (%) [2023+]",
    "BAMLH0A0HYM2EY":"ICE BofA HY Corp Eff Yield (%) [2023+]",
}

# Yield → ETF signal proxy map (for the metadata sheet)
YIELD_TO_ETF_SIGNAL: dict[str, str] = {
    "DGS3MO": "SHV",
    "DGS2":   "SHY",
    "DGS5":   "IEI",
    "DGS7":   "IEF",
    "DGS10":  "IEF,TLH",
    "DGS20":  "TLH,TLT",
    "DGS30":  "TLT",
    "DFII5":  "VTIP",
    "DFII10": "TIP",
    "DBAA":   "LQD",
    "BAA10YM": "HYG",
    "BAMLC0A0CMEY":   "LQD (2023+ supplement)",
    "BAMLH0A0HYM2EY": "HYG (2023+ supplement)",
}

COMMODITY_ETF_MAP = {
    "GLD":  "Gold",
    "SLV":  "Silver",
    "CPER": "Copper",
    "PPLT": "Platinum",
    "PALL": "Palladium",
    "USO":  "WTI Crude Oil",
    "BNO":  "Brent Crude Oil",
    "UNG":  "Natural Gas",
    "DBA":  "Agriculture Basket",
    "CORN": "Corn",
    # Extensions
    "WEAT": "Wheat",
    "SOYB": "Soybeans",
}

PULL_START = "2000-01-01"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _month_end(df: pd.DataFrame) -> pd.DataFrame:
    """Resample to month-end, taking the last observation of the month."""
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df.resample("ME").last()


def _monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple monthly return from end-of-month price series."""
    return prices.pct_change().iloc[1:]


def pull_etf_prices(tickers: list[str], start: str = PULL_START) -> pd.DataFrame:
    """Download monthly adjusted-close prices for a list of ETF tickers."""
    print(f"  yfinance: downloading {len(tickers)} tickers … ", end="", flush=True)
    raw = yf.download(
        tickers,
        start=start,
        auto_adjust=True,
        progress=False,
    )
    print("done")

    # yfinance returns multi-level columns when multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = tickers

    prices.index = pd.to_datetime(prices.index)
    prices = prices.resample("ME").last()
    prices.index.name = "date"
    return prices.sort_index()


def pull_fred_series(
    series_ids: list[str],
    start: str = PULL_START,
    retries: int = 3,
) -> pd.DataFrame:
    """Download one or more FRED daily series and resample to month-end."""
    dfs = []
    base = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    for sid in series_ids:
        for attempt in range(retries):
            try:
                url = f"{base}?id={sid}&observation_start={start}"
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                df = pd.read_csv(
                    io.StringIO(resp.text),
                    parse_dates=["observation_date"],
                    index_col="observation_date",
                )
                df.columns = [sid]
                df = df.replace(".", np.nan).astype(float)
                dfs.append(df)
                print(f"  FRED {sid}: {len(df)} daily obs from {df.index[0].date()}")
                break
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2)
                else:
                    print(f"  FRED {sid}: FAILED – {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, axis=1).sort_index()
    combined.index = pd.to_datetime(combined.index)
    monthly = combined.resample("ME").last()
    monthly.index.name = "date"
    return monthly


def load_msci_project_data(p2_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load original project data: MSCI returns and P/E ratios."""
    ret_raw = pd.read_excel(p2_path, sheet_name="Country equity returns",
                            index_col=0, header=0)
    pe_raw  = pd.read_excel(p2_path, sheet_name="Country equity PE ratios",
                            index_col=0, header=0)

    def _clean(df: pd.DataFrame, country_map: dict[str, str]) -> pd.DataFrame:
        df = df.copy()
        df.index = pd.to_datetime(df.index) + pd.offsets.MonthEnd(0)
        df.index.name = "date"
        df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all").sort_index()
        # Rename columns: match on partial string
        def _rn(col):
            for long, short in country_map.items():
                if long in col or short in col:
                    return short
            return col
        df.columns = [_rn(c) for c in df.columns]
        return df

    cmap = {
        "MSCI Australia NR LCL": "AUS",
        "MSCI Canada NR LCL":    "CAN",
        "MSCI France NR LCL":    "FRA",
        "MSCI Germany NR LCL":   "GER",
        "MSCI Italy NR LCL":     "ITA",
        "MSCI Japan NR JPY":     "JPN",
        "MSCI Netherlands NR LCL": "NLD",
        "MSCI Spain NR LCL":     "ESP",
        "MSCI United Kingdom NR LCL": "GBR",
        "MSCI United States NR USD":  "USA",
    }

    returns = _clean(ret_raw, cmap)
    pe      = _clean(pe_raw,  cmap)

    # Rename columns from country codes to ETF tickers
    def _to_etf(df):
        return df.rename(columns=MSCI_TO_ETF)

    return _to_etf(returns), _to_etf(pe)


def build_metadata() -> pd.DataFrame:
    """Build a ticker-level metadata table."""
    rows = []

    for tk, desc in EQUITY_ETF_MAP.items():
        rows.append({
            "ticker":      tk,
            "asset_class": "Equity",
            "sleeve":      "equity",
            "description": f"iShares MSCI {desc} ETF",
            "signal_type": "value_pe or momentum_12_1",
            "signal_source": "MSCI P/E (project data) or yfinance price",
        })

    fi_signal_src = {
        "SHV":  "DGS3MO (3M T-bill yield)",
        "SHY":  "DGS2 (2Y Treasury yield)",
        "IEI":  "DGS5 (5Y Treasury yield)",
        "IEF":  "DGS7 / DGS10 (7-10Y Treasury yield)",
        "TLH":  "DGS10 / DGS20 (10-20Y Treasury yield)",
        "TLT":  "DGS20 / DGS30 (20-30Y Treasury yield)",
        "VTIP": "DFII5 (5Y real yield)",
        "TIP":  "DFII10 (10Y real yield)",
        "LQD":  "DBAA (Moody's Baa yield — IG proxy, 1986+)",
        "HYG":  "BAA10YM (Baa-10Y credit spread — HY proxy, 1953+)",
    }
    for tk, desc in FI_ETF_MAP.items():
        rows.append({
            "ticker":      tk,
            "asset_class": "Fixed Income",
            "sleeve":      "fixed_income",
            "description": desc,
            "signal_type": "yield_carry",
            "signal_source": fi_signal_src.get(tk, "FRED yield"),
        })

    for tk, desc in COMMODITY_ETF_MAP.items():
        rows.append({
            "ticker":      tk,
            "asset_class": "Commodity",
            "sleeve":      "commodity",
            "description": desc,
            "signal_type": "momentum_12_1",
            "signal_source": "computed from price (yfinance)",
        })

    return pd.DataFrame(rows).set_index("ticker")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(out_path: Optional[Path] = None) -> None:
    out = Path(out_path) if out_path else OUT_FILE
    out.parent.mkdir(parents=True, exist_ok=True)

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("MULTI-ASSET UNIVERSE DATA PULL")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # ── 1. Original MSCI project data ─────────────────────────────────────────
    print("\n[1/5] Loading original MSCI project data …")
    msci_returns, msci_pe = load_msci_project_data(P2_FILE)
    print(f"  MSCI returns: {msci_returns.shape}  "
          f"({msci_returns.index[0].date()} – {msci_returns.index[-1].date()})")
    print(f"  MSCI P/E:     {msci_pe.shape}  "
          f"({msci_pe.dropna(how='all').index[0].date()} – {msci_pe.index[-1].date()})")

    # ── 2. Equity ETF prices ───────────────────────────────────────────────────
    print("\n[2/5] Equity ETF prices from yfinance …")
    eq_prices = pull_etf_prices(list(EQUITY_ETF_MAP.keys()))
    eq_returns = _monthly_returns(eq_prices)
    print(f"  Equity prices:  {eq_prices.shape}  "
          f"({eq_prices.index[0].date()} – {eq_prices.index[-1].date()})")
    print(f"  Equity returns: {eq_returns.shape}")
    # Coverage report
    for tk in EQUITY_ETF_MAP:
        first = eq_prices[tk].dropna().index[0].date() if tk in eq_prices else "N/A"
        print(f"    {tk:6s} first obs: {first}")

    # ── 3. Fixed-income ETF prices + FRED yields ───────────────────────────────
    print("\n[3/5] Fixed-income ETF prices from yfinance …")
    fi_prices = pull_etf_prices(list(FI_ETF_MAP.keys()))
    fi_returns = _monthly_returns(fi_prices)
    print(f"  FI prices:  {fi_prices.shape}  "
          f"({fi_prices.index[0].date()} – {fi_prices.index[-1].date()})")
    for tk in FI_ETF_MAP:
        first = fi_prices[tk].dropna().index[0].date() if tk in fi_prices else "N/A"
        print(f"    {tk:6s} first obs: {first}")

    print("\n  FRED yield series …")
    fred_yields = pull_fred_series(list(FI_YIELD_MAP.keys()))
    # Add human-readable labels as a header (stored in column names)
    fred_yields_labeled = fred_yields.rename(columns=FI_YIELD_MAP)
    print(f"  Yields: {fred_yields_labeled.shape}  "
          f"({fred_yields_labeled.index[0].date()} – {fred_yields_labeled.index[-1].date()})")

    # ── 4. Commodity ETF prices ────────────────────────────────────────────────
    print("\n[4/5] Commodity ETF prices from yfinance …")
    com_prices = pull_etf_prices(list(COMMODITY_ETF_MAP.keys()))
    com_returns = _monthly_returns(com_prices)
    print(f"  Commodity prices:  {com_prices.shape}  "
          f"({com_prices.index[0].date()} – {com_prices.index[-1].date()})")
    for tk in COMMODITY_ETF_MAP:
        first = com_prices[tk].dropna().index[0].date() if tk in com_prices else "N/A"
        print(f"    {tk:6s} first obs: {first}")

    # ── 5. Write Excel ─────────────────────────────────────────────────────────
    print(f"\n[5/5] Writing Excel workbook → {out} …")

    meta = build_metadata()

    # Carry signal proxy: one column per FI ETF, filled from the most relevant yield
    # Maps ETF → primary FRED series ID
    fi_carry_proxy_map = {
        "SHV":  "DGS3MO",
        "SHY":  "DGS2",
        "IEI":  "DGS5",
        "IEF":  "DGS7",
        "TLH":  "DGS20",
        "TLT":  "DGS30",
        "VTIP": "DFII5",
        "TIP":  "DFII10",
        "LQD":  "DBAA",            # Moody's Baa — best free IG proxy (1986+)
        "HYG":  "BAA10YM",         # Baa-10Y credit spread — HY proxy (1953+)
    }
    fi_carry = pd.DataFrame({
        etf: fred_yields[fred_id].rename(etf)
        for etf, fred_id in fi_carry_proxy_map.items()
        if fred_id in fred_yields.columns
    })
    fi_carry.index.name = "date"

    with pd.ExcelWriter(out, engine="openpyxl") as writer:

        def _write(df, sheet, freeze="B2"):
            df.to_excel(writer, sheet_name=sheet)
            ws = writer.sheets[sheet]
            if freeze:
                ws.freeze_panes = freeze
            # Auto-width (approximate)
            for col_cells in ws.columns:
                max_len = max(
                    len(str(c.value)) if c.value is not None else 0
                    for c in col_cells
                )
                ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 30)

        # Equity
        _write(msci_returns,  "equity_msci_returns")
        _write(msci_pe,       "equity_msci_pe")
        _write(eq_prices,     "equity_etf_prices")
        _write(eq_returns,    "equity_etf_returns")

        # Fixed income
        _write(fi_prices,           "fi_etf_prices")
        _write(fi_returns,          "fi_etf_returns")
        _write(fred_yields_labeled, "fi_yields_fred")
        _write(fi_carry,            "fi_carry_signals")

        # Commodities
        _write(com_prices,   "commodity_prices")
        _write(com_returns,  "commodity_returns")

        # Metadata
        _write(meta.reset_index(), "ticker_metadata")

    print(f"\n  Sheets written:")
    sheets_written = [
        "equity_msci_returns", "equity_msci_pe",
        "equity_etf_prices", "equity_etf_returns",
        "fi_etf_prices", "fi_etf_returns",
        "fi_yields_fred", "fi_carry_signals",
        "commodity_prices", "commodity_returns",
        "ticker_metadata",
    ]
    for s in sheets_written:
        print(f"    ✓ {s}")

    print(f"\n  Output: {out}  ({out.stat().st_size / 1024:.0f} kB)")

    # Also write a coverage summary CSV for quick inspection
    summary_rows = []
    for tk, desc in {**EQUITY_ETF_MAP, **FI_ETF_MAP, **COMMODITY_ETF_MAP}.items():
        src = "eq" if tk in EQUITY_ETF_MAP else ("fi" if tk in FI_ETF_MAP else "com")
        prices_df = eq_prices if src == "eq" else (fi_prices if src == "fi" else com_prices)
        col = prices_df[tk] if tk in prices_df.columns else pd.Series(dtype=float)
        col = col.dropna()
        summary_rows.append({
            "ticker":      tk,
            "sleeve":      src,
            "description": desc,
            "first_date":  col.index[0].date() if len(col) else "N/A",
            "last_date":   col.index[-1].date() if len(col) else "N/A",
            "n_months":    len(col),
        })

    summary = pd.DataFrame(summary_rows)
    csv_out = out.parent / "multi_asset_coverage.csv"
    summary.to_csv(csv_out, index=False)
    print(f"  Coverage CSV: {csv_out}")

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("DONE")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull multi-asset universe data")
    parser.add_argument("--out", default=None, help="Override output Excel path")
    args = parser.parse_args()
    main(args.out)
