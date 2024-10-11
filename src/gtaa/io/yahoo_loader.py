"""Yahoo Finance loader for ETF expansion module."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False


COUNTRY_ETF_MAP = {
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

MULTI_ASSET_UNIVERSE = {
    "equities": ["SPY", "EFA", "EEM", "EWJ", "VGK"],
    "bonds": ["SHY", "IEF", "TLT", "LQD", "HYG"],
    "commodities": ["GLD", "DBC", "SLV", "CPER"],
    "real_estate": ["VNQ"],
    "currencies": ["UUP", "FXE", "FXY", "FXB", "FXA"],
}


def download_monthly_returns(
    tickers: list[str],
    start: str = "2000-01-01",
    end: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Download adjusted-close prices and compute monthly simple returns."""
    if not _YF_AVAILABLE:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = "_".join(sorted(tickers))
        cache_path = cache_dir / f"yahoo_{key[:40]}_{start[:7]}.parquet"

    if cache_path is not None and cache_path.exists():
        prices = pd.read_parquet(cache_path)
    else:
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
        prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        prices = prices.resample("ME").last()
        if cache_path is not None:
            prices.to_parquet(cache_path)

    returns = prices.pct_change().iloc[1:]
    returns.index = returns.index + pd.offsets.MonthEnd(0)
    return returns


def load_country_etf_returns(
    start: str = "2000-01-01",
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Download country ETF monthly returns keyed by country code."""
    etf_returns = download_monthly_returns(
        list(COUNTRY_ETF_MAP.values()), start=start, cache_dir=cache_dir
    )
    # Rename ETF tickers back to country codes
    inv_map = {v: k for k, v in COUNTRY_ETF_MAP.items()}
    etf_returns.columns = [inv_map.get(c, c) for c in etf_returns.columns]
    return etf_returns
