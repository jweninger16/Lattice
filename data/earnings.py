"""
data/earnings.py
----------------
Earnings calendar features.

The goal is simple: don't hold stocks through earnings announcements.
Earnings create binary outcomes (big gap up or down) that are essentially
random from a technical analysis perspective.

Fixes over original:
  - Cache checks age — rebuilds if older than 7 days
  - Cache stores metadata (date built, tickers covered)
  - Falls back to fresh fetch if cached tickers don't overlap with request
  - Handles yfinance API failures gracefully per-ticker
  - Logs coverage stats so you can see what's working

Features added:
  - days_to_earnings: how many trading days until next earnings
  - days_since_earnings: how many trading days since last earnings
  - earnings_soon: 1 if earnings within 5 days (avoid entering)
  - earnings_just_passed: 1 if earnings within last 3 days (volatile)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import pickle
from datetime import datetime, timedelta


CACHE_DIR = Path("data/raw/earnings")
CACHE_MAX_AGE_DAYS = 7


def _cache_path():
    return CACHE_DIR / "earnings_dates.pkl"


def _cache_is_fresh(required_tickers: list = None) -> bool:
    """Checks if cache exists, is recent enough, and covers the needed tickers."""
    cache_file = _cache_path()
    if not cache_file.exists():
        return False

    # Check age
    mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
    age_days = (datetime.now() - mtime).days
    if age_days > CACHE_MAX_AGE_DAYS:
        logger.info(f"Earnings cache is {age_days} days old (max {CACHE_MAX_AGE_DAYS}), rebuilding...")
        return False

    # Check ticker coverage
    if required_tickers:
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            # Handle both old format (just dict) and new format (dict with metadata)
            if isinstance(data, dict) and "_metadata" in data:
                cached_tickers = set(data["_metadata"].get("tickers", []))
            elif isinstance(data, dict):
                cached_tickers = set(data.keys())
            else:
                return False

            required = set(required_tickers)
            overlap = len(required & cached_tickers) / max(len(required), 1)
            if overlap < 0.5:
                logger.info(f"Earnings cache covers {overlap*100:.0f}% of requested tickers, rebuilding...")
                return False
        except Exception:
            return False

    return True


def fetch_earnings_dates(tickers: list, use_cache: bool = True) -> dict:
    """
    Fetches earnings dates for all tickers via yfinance.
    Returns dict: {ticker: [date1, date2, ...]}

    Smart caching:
      - Uses cache if fresh (< 7 days) and covers most requested tickers
      - Otherwise fetches fresh data
      - Stores metadata for cache validation
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path()

    if use_cache and _cache_is_fresh(tickers):
        logger.info("Loading earnings dates from cache...")
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            # Handle both formats
            if isinstance(data, dict) and "_metadata" in data:
                metadata = data.pop("_metadata")
                logger.info(f"Cache built {metadata.get('date', '?')}, "
                            f"covers {len(data)} tickers")
                return data
            elif isinstance(data, dict):
                logger.info(f"Cache covers {len(data)} tickers (legacy format)")
                return data
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")

    # Fetch fresh data
    logger.info(f"Fetching earnings dates for {len(tickers)} tickers...")
    earnings_map = {}
    failed = 0

    for ticker in tqdm(tickers, desc="Earnings"):
        try:
            stock = yf.Ticker(ticker)
            cal = stock.earnings_dates
            if cal is not None and len(cal) > 0:
                dates = pd.to_datetime(cal.index).tz_localize(None)
                earnings_map[ticker] = sorted(dates.tolist())
        except Exception:
            failed += 1
            continue

    logger.info(f"Got earnings dates for {len(earnings_map)} tickers ({failed} failed)")

    # Save with metadata
    save_data = dict(earnings_map)
    save_data["_metadata"] = {
        "date": str(datetime.now().date()),
        "tickers": list(earnings_map.keys()),
        "count": len(earnings_map),
    }

    try:
        with open(cache_file, "wb") as f:
            pickle.dump(save_data, f)
        logger.info(f"Earnings cache saved ({len(earnings_map)} tickers)")
    except Exception as e:
        logger.warning(f"Failed to save earnings cache: {e}")

    return earnings_map


def add_earnings_features(df: pd.DataFrame, earnings_map: dict) -> pd.DataFrame:
    """
    Adds earnings proximity features to the dataframe.
    """
    logger.info("Adding earnings calendar features...")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Initialize columns
    df["days_to_earnings"]      = 999
    df["days_since_earnings"]   = 999
    df["earnings_soon"]         = 0
    df["earnings_just_passed"]  = 0

    tickers_with_data = set(earnings_map.keys())
    df_tickers = set(df["ticker"].unique())
    overlap = len(tickers_with_data & df_tickers)
    logger.info(f"Earnings data covers {overlap}/{len(df_tickers)} tickers in dataset")

    if overlap == 0:
        logger.warning("No earnings data overlaps with current tickers — "
                        "features will be defaults (999/0)")
        return df

    processed = 0

    for ticker, ticker_df in df.groupby("ticker"):
        if ticker not in tickers_with_data:
            continue

        earn_dates = pd.Series(sorted(earnings_map[ticker]))
        if len(earn_dates) == 0:
            continue

        idx = ticker_df.index
        dates = ticker_df["date"]

        days_to   = []
        days_since = []

        for d in dates:
            future = earn_dates[earn_dates > d]
            past   = earn_dates[earn_dates <= d]

            dte = (future.iloc[0] - d).days if len(future) > 0 else 999
            dse = (d - past.iloc[-1]).days  if len(past)   > 0 else 999

            # Convert calendar days to approximate trading days
            days_to.append(int(dte * 5/7))
            days_since.append(int(dse * 5/7))

        df.loc[idx, "days_to_earnings"]    = days_to
        df.loc[idx, "days_since_earnings"] = days_since
        processed += 1

    # Binary flags
    df["earnings_soon"]        = (df["days_to_earnings"]    <= 5).astype(int)
    df["earnings_just_passed"] = (df["days_since_earnings"] <= 3).astype(int)

    logger.info(f"Earnings features added for {processed} tickers")
    logger.info(f"  Earnings soon (<=5d):        {df['earnings_soon'].mean()*100:.1f}% of rows")
    logger.info(f"  Earnings just passed (<=3d): {df['earnings_just_passed'].mean()*100:.1f}% of rows")

    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.pipeline import load_processed
    from data.universe import build_universe
    import yaml

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    df = load_processed()
    tickers = df["ticker"].unique().tolist()

    earnings_map = fetch_earnings_dates(tickers, use_cache=False)
    df = add_earnings_features(df, earnings_map)

    print(f"\nEarnings filter impact:")
    print(f"  Would block {df['earnings_soon'].mean()*100:.1f}% of all stock-days")
    print(f"  Sample of earnings_soon=1 rows:")
    print(df[df["earnings_soon"] == 1][["ticker","date","days_to_earnings"]].head(10).to_string(index=False))
