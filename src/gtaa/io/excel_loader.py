"""Excel loaders for all project data files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from gtaa.models import DataBundle


# ── GTAA Mom-Carry data bundle ────────────────────────────────────────────────

@dataclass
class GTAAExpandedData:
    """Data bundle for the six-sleeve expanded GTAA strategy."""
    equity_returns:    pd.DataFrame
    commodity_prices:  pd.DataFrame
    commodity_returns: pd.DataFrame
    fi_returns:        pd.DataFrame
    fi_carry:          pd.DataFrame
    fx_prices:         pd.DataFrame
    fx_returns:        pd.DataFrame
    fx_carry:          pd.DataFrame
    all_returns:       pd.DataFrame


@dataclass
class GTAAMomCarryData:
    """All data required for the three-sleeve GTAA Mom-Carry strategy.

    Attributes:
        equity_returns:    Monthly MSCI country equity returns (date × 10 countries).
        commodity_prices:  Monthly commodity ETF adjusted-close prices (date × 12).
        commodity_returns: Monthly commodity ETF returns (date × 12).
        fi_returns:        Monthly FI ETF returns (date × 10).
        fi_carry:          Monthly carry signal per FI ETF (date × 10).
        all_returns:       Concatenated equity + commodity + FI returns (date × 32).
    """
    equity_returns:    pd.DataFrame
    commodity_prices:  pd.DataFrame
    commodity_returns: pd.DataFrame
    fi_returns:        pd.DataFrame
    fi_carry:          pd.DataFrame
    all_returns:       pd.DataFrame


def load_monthly_sheet(path: str | Path, sheet_name: str) -> pd.DataFrame:
    """Load a monthly-frequency sheet from the multi-asset universe workbook.

    Assumes the first column is the date column. Converts the index to
    month-end timestamps (period → timestamp).

    Args:
        path:       Path to the Excel workbook.
        sheet_name: Sheet to load.

    Returns:
        DataFrame with a DatetimeIndex (month-end) and numeric columns.
    """
    df = pd.read_excel(path, sheet_name=sheet_name)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df.index = df.index.to_period("M").to_timestamp("M")
    df.index.name = "date"
    return df.apply(pd.to_numeric, errors="coerce")


def load_gtaa_mom_carry_data(path: str | Path) -> GTAAMomCarryData:
    """Load all data for the GTAA Momentum + Carry strategy.

    Reads six sheets from multi_asset_universe.xlsx and assembles a
    GTAAMomCarryData bundle. The all_returns field is the horizontal
    concatenation of equity, commodity, and FI returns — used for the
    full-universe covariance in the final portfolio scaling step.

    Args:
        path: Path to multi_asset_universe.xlsx.

    Returns:
        Populated GTAAMomCarryData instance.
    """
    path = Path(path)

    equity_returns    = load_monthly_sheet(path, "equity_msci_returns")
    commodity_prices  = load_monthly_sheet(path, "commodity_prices")
    commodity_returns = load_monthly_sheet(path, "commodity_returns")
    fi_returns        = load_monthly_sheet(path, "fi_etf_returns")
    fi_carry          = load_monthly_sheet(path, "fi_carry_signals")

    all_returns = pd.concat(
        [equity_returns, commodity_returns, fi_returns],
        axis=1,
    ).sort_index()

    # Drop duplicate columns if any overlap exists between sleeves
    all_returns = all_returns.loc[:, ~all_returns.columns.duplicated()]

    return GTAAMomCarryData(
        equity_returns=equity_returns,
        commodity_prices=commodity_prices,
        commodity_returns=commodity_returns,
        fi_returns=fi_returns,
        fi_carry=fi_carry,
        all_returns=all_returns,
    )


def load_gtaa_expanded_data(path: str | Path) -> GTAAExpandedData:
    """Load data for the six-sleeve expanded GTAA strategy from multi_asset_universe.xlsx."""
    path = Path(path)

    equity_returns    = load_monthly_sheet(path, "equity_msci_returns")
    commodity_prices  = load_monthly_sheet(path, "commodity_prices")
    commodity_returns = load_monthly_sheet(path, "commodity_returns")
    fi_returns        = load_monthly_sheet(path, "fi_etf_returns")
    fi_carry          = load_monthly_sheet(path, "fi_carry_signals")
    fx_prices         = load_monthly_sheet(path, "fx_prices")
    fx_returns        = load_monthly_sheet(path, "fx_returns")
    fx_carry          = load_monthly_sheet(path, "fx_carry_signals")

    all_returns = pd.concat(
        [equity_returns, commodity_returns, fi_returns, fx_returns],
        axis=1,
    ).sort_index()
    all_returns = all_returns.loc[:, ~all_returns.columns.duplicated()]

    return GTAAExpandedData(
        equity_returns=equity_returns,
        commodity_prices=commodity_prices,
        commodity_returns=commodity_returns,
        fi_returns=fi_returns,
        fi_carry=fi_carry,
        fx_prices=fx_prices,
        fx_returns=fx_returns,
        fx_carry=fx_carry,
        all_returns=all_returns,
    )


def _read_monthly_sheet(path: Path, sheet: str, date_col: int = 0) -> pd.DataFrame:
    """Read a monthly-frequency sheet; coerce index to month-end timestamps."""
    df = pd.read_excel(path, sheet_name=sheet, index_col=date_col, header=0)
    df.index = pd.to_datetime(df.index) + pd.offsets.MonthEnd(0)
    df.index.name = "date"
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.sort_index()


def load_project2_data(path: str | Path) -> DataBundle:
    """Load 'Data for final project 2 .xlsx' into a DataBundle.

    Sheet layout (verified against source file):
      - 'Country equity returns' : monthly total returns, 10 MSCI countries
      - 'Country equity PE ratios': P/E ratios, 10 MSCI countries (starts ~2005)

    Returns:
        DataBundle with .returns and .valuations populated.
        Column names are cleaned to short country codes.
    """
    path = Path(path)

    returns = _read_monthly_sheet(path, "Country equity returns")
    pe = _read_monthly_sheet(path, "Country equity PE ratios")

    # Strip verbose MSCI prefix to short labels
    country_map = {
        "MSCI Australia NR LCL": "AUS",
        "MSCI Canada NR LCL": "CAN",
        "MSCI France NR LCL": "FRA",
        "MSCI Germany NR LCL": "GER",
        "MSCI Italy NR LCL": "ITA",
        "MSCI Japan NR JPY": "JPN",
        "MSCI Netherlands NR LCL": "NLD",
        "MSCI Spain NR LCL": "ESP",
        "MSCI United Kingdom NR LCL": "GBR",
        "MSCI United States NR USD": "USA",
    }
    # Flexible rename: match on partial string
    def _rename(col: str) -> str:
        for long, short in country_map.items():
            if long in col or short in col:
                return short
        return col

    returns.columns = [_rename(c) for c in returns.columns]
    pe.columns = [_rename(c) for c in pe.columns]

    # Drop the first (description) row if it is NaN data
    returns = returns.dropna(how="all")
    pe = pe.dropna(how="all")

    return DataBundle(returns=returns, valuations=pe)


def load_hw1_data(path: str | Path) -> dict[str, pd.DataFrame]:
    """Load HW_GTAA1 solution workbook into named DataFrames."""
    path = Path(path)
    sheets = {}
    for name in ["economic regimes", "regime-based asset forecasts", "CPI", "indices"]:
        try:
            df = pd.read_excel(path, sheet_name=name, index_col=None)
            sheets[name] = df
        except Exception:
            pass
    return sheets


def load_hw2_data(path: str | Path) -> dict[str, pd.DataFrame]:
    """Load HW_GTAA2 template workbook into named DataFrames."""
    path = Path(path)
    xl = pd.ExcelFile(path)
    return {name: pd.read_excel(path, sheet_name=name) for name in xl.sheet_names}
