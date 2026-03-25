"""
data/sectors.py
---------------
Sector momentum features.

This is the fix for the narrow leadership problem — tracks which sectors
are actually driving market gains so the ML model can find stocks in
leading sectors even when broad market breadth is weak.

Sector ETFs tracked:
  XLK  - Technology
  XLF  - Financials
  XLE  - Energy
  XLV  - Healthcare
  XLI  - Industrials
  XLY  - Consumer Discretionary
  XLP  - Consumer Staples
  XLB  - Materials
  XLU  - Utilities
  XLRE - Real Estate
  XLC  - Communication Services

Features added per stock-day:
  - sector_momentum_21d: 21-day return of the stock's sector ETF
  - sector_momentum_63d: 63-day return of the stock's sector ETF
  - sector_rs: sector's return rank vs all sectors (0-1)
  - sector_above_sma50: 1 if sector ETF is above its 50-day MA
  - in_leading_sector: 1 if stock's sector is in top 3 by momentum
  - spy_above_sma50: 1 if SPY is above 50-day MA (fixes narrow leadership problem)
  - spy_momentum_21d: SPY 21-day momentum
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from loguru import logger


SECTOR_ETFS = {
    "XLK":  "Technology",
    "XLF":  "Financials",
    "XLE":  "Energy",
    "XLV":  "Healthcare",
    "XLI":  "Industrials",
    "XLY":  "Consumer Discretionary",
    "XLP":  "Consumer Staples",
    "XLB":  "Materials",
    "XLU":  "Utilities",
    "XLRE": "Real Estate",
    "XLC":  "Communication Services",
    "SPY":  "S&P 500",          # Market benchmark
    "QQQ":  "Nasdaq",           # Tech-heavy benchmark
}

# GICS sector mapping for S&P 500 stocks
# Maps ticker -> sector ETF
# This is a representative mapping for large caps
TICKER_TO_SECTOR = {
    # Technology -> XLK
    "AAPL":"XLK","MSFT":"XLK","NVDA":"XLK","AVGO":"XLK","ORCL":"XLK",
    "CRM":"XLK","AMD":"XLK","QCOM":"XLK","TXN":"XLK","INTC":"XLK",
    "AMAT":"XLK","MU":"XLK","LRCX":"XLK","KLAC":"XLK","ADI":"XLK",
    "CDNS":"XLK","SNPS":"XLK","MRVL":"XLK","NXPI":"XLK","FTNT":"XLK",
    "PANW":"XLK","CRWD":"XLK","SNOW":"XLK","NOW":"XLK","TEAM":"XLK",
    "ZS":"XLK","OKTA":"XLK","MDB":"XLK","DDOG":"XLK","NET":"XLK",
    # Communication -> XLC
    "GOOGL":"XLC","GOOG":"XLC","META":"XLC","NFLX":"XLC","DIS":"XLC",
    "CMCSA":"XLC","VZ":"XLC","T":"XLC","TMUS":"XLC","CHTR":"XLC",
    "TTWO":"XLC","EA":"XLC","WBD":"XLC","PARA":"XLC","OMC":"XLC",
    # Consumer Discretionary -> XLY
    "AMZN":"XLY","TSLA":"XLY","HD":"XLY","MCD":"XLY","NKE":"XLY",
    "SBUX":"XLY","LOW":"XLY","TJX":"XLY","BKNG":"XLY","CMG":"XLY",
    "ORLY":"XLY","AZO":"XLY","MAR":"XLY","HLT":"XLY","RCL":"XLY",
    "CCL":"XLY","ABNB":"XLY","EXPE":"XLY","ROST":"XLY","YUM":"XLY",
    # Consumer Staples -> XLP
    "WMT":"XLP","PG":"XLP","COST":"XLP","KO":"XLP","PEP":"XLP",
    "PM":"XLP","MO":"XLP","MDLZ":"XLP","CL":"XLP","KMB":"XLP",
    "SYY":"XLP","KR":"XLP","GIS":"XLP","HSY":"XLP","CHD":"XLP",
    # Healthcare -> XLV
    "LLY":"XLV","UNH":"XLV","JNJ":"XLV","ABBV":"XLV","MRK":"XLV",
    "TMO":"XLV","ABT":"XLV","DHR":"XLV","BMY":"XLV","AMGN":"XLV",
    "PFE":"XLV","GILD":"XLV","ISRG":"XLV","SYK":"XLV","BSX":"XLV",
    "MDT":"XLV","ZTS":"XLV","REGN":"XLV","VRTX":"XLV","HCA":"XLV",
    # Financials -> XLF
    "BRK-B":"XLF","JPM":"XLF","V":"XLF","MA":"XLF","BAC":"XLF",
    "WFC":"XLF","GS":"XLF","MS":"XLF","BLK":"XLF","AXP":"XLF",
    "SPGI":"XLF","MCO":"XLF","COF":"XLF","USB":"XLF","PNC":"XLF",
    "TFC":"XLF","CME":"XLF","ICE":"XLF","CB":"XLF","PGR":"XLF",
    # Industrials -> XLI
    "GE":"XLI","CAT":"XLI","RTX":"XLI","HON":"XLI","UPS":"XLI",
    "BA":"XLI","LMT":"XLI","DE":"XLI","MMM":"XLI","GD":"XLI",
    "NOC":"XLI","FDX":"XLI","EMR":"XLI","ETN":"XLI","PH":"XLI",
    "ROK":"XLI","ITW":"XLI","CSX":"XLI","NSC":"XLI","UNP":"XLI",
    # Energy -> XLE
    "XOM":"XLE","CVX":"XLE","COP":"XLE","EOG":"XLE","SLB":"XLE",
    "MPC":"XLE","PSX":"XLE","VLO":"XLE","PXD":"XLE","OXY":"XLE",
    "HAL":"XLE","DVN":"XLE","HES":"XLE","BKR":"XLE","FANG":"XLE",
    # Materials -> XLB
    "LIN":"XLB","APD":"XLB","SHW":"XLB","ECL":"XLB","NEM":"XLB",
    "FCX":"XLB","NUE":"XLB","VMC":"XLB","MLM":"XLB","CF":"XLB",
    # Utilities -> XLU
    "NEE":"XLU","DUK":"XLU","SO":"XLU","D":"XLU","AEP":"XLU",
    "EXC":"XLU","SRE":"XLU","PCG":"XLU","XEL":"XLU","ED":"XLU",
    # Real Estate -> XLRE
    "PLD":"XLRE","AMT":"XLRE","EQIX":"XLRE","PSA":"XLRE","O":"XLRE",
    "WELL":"XLRE","DLR":"XLRE","SPG":"XLRE","AVB":"XLRE","EQR":"XLRE",
}


def fetch_sector_data(start_date: str, end_date: str = None) -> pd.DataFrame:
    """Downloads sector ETF price data."""
    logger.info("Downloading sector ETF data...")
    etfs = list(SECTOR_ETFS.keys())
    raw = yf.download(etfs, start=start_date, end=end_date,
                      auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw["Close"]
    else:
        raw = raw[["Close"]]

    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.index.name = "date"
    raw = raw.reset_index()
    raw = raw.melt(id_vars="date", var_name="etf", value_name="close")
    raw = raw.dropna()
    return raw


def compute_sector_features(sector_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes momentum and trend features for each sector ETF.
    Returns a date-indexed dataframe with features per sector.
    """
    sector_df = sector_df.copy().sort_values(["etf", "date"])

    sector_df["mom_21d"] = sector_df.groupby("etf")["close"].transform(
        lambda x: x / x.shift(21) - 1
    )
    sector_df["mom_63d"] = sector_df.groupby("etf")["close"].transform(
        lambda x: x / x.shift(63) - 1
    )
    sector_df["sma50"] = sector_df.groupby("etf")["close"].transform(
        lambda x: x.rolling(50).mean()
    )
    sector_df["above_sma50"] = (sector_df["close"] > sector_df["sma50"]).astype(int)

    # Cross-sectional rank of sector momentum each day (excluding SPY/QQQ)
    pure_sectors = sector_df[~sector_df["etf"].isin(["SPY","QQQ"])].copy()
    pure_sectors["sector_rs"] = pure_sectors.groupby("date")["mom_21d"].rank(pct=True)

    # Top 3 sectors each day
    top3 = pure_sectors.groupby("date").apply(
        lambda x: set(x.nlargest(3, "mom_21d")["etf"].tolist())
    ).reset_index()
    top3.columns = ["date", "top3_sectors"]

    return sector_df, pure_sectors, top3


def add_sector_features(df: pd.DataFrame, start_date: str,
                        end_date: str = None) -> pd.DataFrame:
    """
    Adds sector momentum features to the main dataframe.
    """
    logger.info("Adding sector momentum features...")

    sector_raw = fetch_sector_data(start_date, end_date)
    sector_feats, pure_sectors, top3 = compute_sector_features(sector_raw)

    # Get SPY features separately
    spy = sector_feats[sector_feats["etf"] == "SPY"][
        ["date", "mom_21d", "mom_63d", "above_sma50"]
    ].rename(columns={
        "mom_21d": "spy_momentum_21d",
        "mom_63d": "spy_momentum_63d",
        "above_sma50": "spy_above_sma50",
    })

    # Get QQQ features
    qqq = sector_feats[sector_feats["etf"] == "QQQ"][
        ["date", "mom_21d", "above_sma50"]
    ].rename(columns={
        "mom_21d": "qqq_momentum_21d",
        "above_sma50": "qqq_above_sma50",
    })

    # Map each ticker to its sector ETF
    df = df.copy()
    df["sector_etf"] = df["ticker"].map(TICKER_TO_SECTOR).fillna("SPY")

    # Merge sector features onto each stock
    sector_lookup = pure_sectors[["date","etf","mom_21d","mom_63d","above_sma50","sector_rs"]].rename(
        columns={
            "etf": "sector_etf",
            "mom_21d": "sector_momentum_21d",
            "mom_63d": "sector_momentum_63d",
            "above_sma50": "sector_above_sma50",
        }
    )

    df["date"] = pd.to_datetime(df["date"])
    sector_lookup["date"] = pd.to_datetime(sector_lookup["date"])
    spy["date"] = pd.to_datetime(spy["date"])
    qqq["date"] = pd.to_datetime(qqq["date"])
    top3["date"] = pd.to_datetime(top3["date"])

    df = df.merge(sector_lookup, on=["date","sector_etf"], how="left")
    df = df.merge(spy, on="date", how="left")
    df = df.merge(qqq, on="date", how="left")
    df = df.merge(top3, on="date", how="left")

    # Is this stock in a leading sector?
    df["in_leading_sector"] = df.apply(
        lambda r: 1 if isinstance(r.get("top3_sectors"), set)
                       and r["sector_etf"] in r["top3_sectors"] else 0,
        axis=1
    )
    df = df.drop(columns=["top3_sectors"], errors="ignore")

    # Fill missing sector data with neutral values
    sector_cols = ["sector_momentum_21d","sector_momentum_63d","sector_above_sma50",
                   "sector_rs","spy_momentum_21d","spy_momentum_63d","spy_above_sma50",
                   "qqq_momentum_21d","qqq_above_sma50","in_leading_sector"]
    for col in sector_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    logger.info(f"Sector features added. Leading sector coverage: "
                f"{df['in_leading_sector'].mean()*100:.1f}% of stock-days")
    logger.info(f"SPY above SMA50: {df['spy_above_sma50'].mean()*100:.1f}% of days")

    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.pipeline import load_processed

    df = load_processed()
    start = df["date"].min().strftime("%Y-%m-%d")
    end   = df["date"].max().strftime("%Y-%m-%d")

    df = add_sector_features(df, start, end)

    print("\nSector feature sample:")
    cols = ["ticker","date","sector_etf","sector_momentum_21d",
            "sector_rs","in_leading_sector","spy_above_sma50"]
    print(df[cols].dropna().tail(20).to_string(index=False))

    print(f"\nTop sectors today:")
    today = df[df["date"] == df["date"].max()]
    sector_perf = today.groupby("sector_etf")["sector_momentum_21d"].first().sort_values(ascending=False)
    print(sector_perf.to_string())
