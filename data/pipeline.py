"""
pipeline.py
-----------
Downloads, cleans, and caches OHLCV price data for the universe.
Uses yfinance (free). Data is stored as parquet for fast reloading.

Improvements:
  - Added gap detection (overnight gaps) as a feature
  - Added Bollinger Band width for volatility squeeze detection
  - Added on-balance volume trend
  - Better handling of corporate actions / price anomalies
  - Added intraday range feature (high-low / close)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import yaml


PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")


def download_price_data(
    tickers: list[str],
    start_date: str,
    end_date: str | None = None,
    chunk_size: int = 100,
) -> pd.DataFrame:
    """
    Downloads OHLCV data for all tickers in chunks (avoids yfinance rate limits).
    Returns a long-format DataFrame.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_data = []

    chunks = [tickers[i:i+chunk_size] for i in range(0, len(tickers), chunk_size)]
    logger.info(f"Downloading data for {len(tickers)} tickers in {len(chunks)} chunks...")

    for i, chunk in enumerate(tqdm(chunks, desc="Downloading")):
        try:
            raw = yf.download(
                chunk,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                threads=True,
            )

            if raw.empty:
                continue

            # Stack to long format
            raw.columns.names = ["Field", "Ticker"]
            stacked = raw.stack(level="Ticker", future_stack=True).reset_index()
            stacked.columns = [c.lower() for c in stacked.columns]
            all_data.append(stacked)

        except Exception as e:
            logger.warning(f"Chunk {i} failed: {e}")
            continue

    if not all_data:
        raise RuntimeError("No data downloaded. Check tickers and internet connection.")

    df = pd.concat(all_data, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    logger.info(f"Downloaded {len(df):,} rows for {df['ticker'].nunique()} tickers")
    return df


def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw price data:
    - Removes rows with missing OHLCV
    - Removes zero/negative prices
    - Detects and removes obvious bad data (>50% daily moves unless legit)
    """
    logger.info("Cleaning price data...")
    n_before = len(df)

    # Drop nulls in critical columns
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])

    # Drop bad prices
    df = df[(df["close"] > 0) & (df["volume"] > 0)]

    # Remove obvious data errors: >50% single-day moves that revert
    df = df.sort_values(["ticker", "date"])
    df["_ret"] = df.groupby("ticker")["close"].pct_change()
    df["_ret_next"] = df.groupby("ticker")["_ret"].shift(-1)
    bad_mask = (df["_ret"].abs() > 0.5) & (df["_ret_next"].abs() > 0.3)
    n_bad = bad_mask.sum()
    if n_bad > 0:
        logger.info(f"Removed {n_bad} suspected bad data points (>50% spike + revert)")
    df = df[~bad_mask].drop(columns=["_ret", "_ret_next"])

    logger.info(f"Cleaned: {n_before:,} → {len(df):,} rows ({n_before - len(df):,} removed)")
    return df.reset_index(drop=True)


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds core technical features used by our signal model.
    All calculated per-ticker to avoid lookahead.

    New features over original:
      - gap_pct: overnight gap (open vs previous close)
      - bb_width: Bollinger Band width (volatility squeeze)
      - intraday_range: (high - low) / close
      - obv_trend: on-balance volume 20-day slope
    """
    logger.info("Adding technical features...")

    results = []

    for ticker, grp in tqdm(df.groupby("ticker"), desc="Features"):
        grp = grp.sort_values("date").copy()
        c = grp["close"]
        o = grp["open"]
        v = grp["volume"]
        h = grp["high"]
        l = grp["low"]

        # --- Returns ---
        grp["ret_1d"] = c.pct_change(1)
        grp["ret_5d"] = c.pct_change(5)
        grp["ret_21d"] = c.pct_change(21)
        grp["ret_63d"] = c.pct_change(63)

        # --- Momentum ---
        grp["momentum_63d"] = c.pct_change(63)
        grp["momentum_21d"] = c.pct_change(21)

        # --- Volume features ---
        grp["avg_volume_20d"] = v.rolling(20).mean()
        grp["volume_ratio"] = v / grp["avg_volume_20d"].replace(0, np.nan)

        # --- Volatility (ATR) ---
        prev_close = c.shift(1)
        tr = pd.concat([
            h - l,
            (h - prev_close).abs(),
            (l - prev_close).abs()
        ], axis=1).max(axis=1)
        grp["atr_14"] = tr.rolling(14).mean()
        grp["atr_pct"] = grp["atr_14"] / c

        # --- Moving averages ---
        grp["sma_20"] = c.rolling(20).mean()
        grp["sma_50"] = c.rolling(50).mean()
        grp["above_sma20"] = (c > grp["sma_20"]).astype(int)
        grp["above_sma50"] = (c > grp["sma_50"]).astype(int)

        # --- Distance from moving averages ---
        grp["dist_sma20"] = (c - grp["sma_20"]) / grp["sma_20"].replace(0, np.nan)
        grp["dist_sma50"] = (c - grp["sma_50"]) / grp["sma_50"].replace(0, np.nan)

        # --- Realized volatility ---
        grp["realized_vol_21d"] = grp["ret_1d"].rolling(21).std() * np.sqrt(252)

        # --- NEW: Overnight gap ---
        grp["gap_pct"] = (o / prev_close - 1).fillna(0)

        # --- NEW: Bollinger Band width (volatility squeeze) ---
        bb_std = c.rolling(20).std()
        grp["bb_width"] = (4 * bb_std / grp["sma_20"]).replace([np.inf, -np.inf], np.nan)

        # --- NEW: Intraday range ---
        grp["intraday_range"] = (h - l) / c.replace(0, np.nan)

        # --- NEW: On-balance volume trend ---
        obv = (v * np.sign(grp["ret_1d"].fillna(0))).cumsum()
        grp["obv_slope_20d"] = obv.rolling(20).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else 0,
            raw=True
        )

        # --- Target: forward 5-day return ---
        grp["target_5d"] = c.shift(-5) / c - 1

        results.append(grp)

    out = pd.concat(results, ignore_index=True)
    logger.info(f"Features added. Shape: {out.shape}")
    return out


def add_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds features calculated cross-sectionally (across all stocks on same date).
    This is where relative strength rankings are computed.
    """
    logger.info("Adding cross-sectional features...")

    df = df.sort_values(["date", "ticker"])

    # Rank momentum cross-sectionally each day (0=worst, 1=best)
    df["rs_rank_63d"] = df.groupby("date")["momentum_63d"].rank(pct=True)
    df["rs_rank_21d"] = df.groupby("date")["momentum_21d"].rank(pct=True)

    # Composite relative strength score
    df["rs_score"] = 0.6 * df["rs_rank_63d"] + 0.4 * df["rs_rank_21d"]

    # Volatility rank (low vol = higher rank = preferred)
    df["vol_rank"] = df.groupby("date")["realized_vol_21d"].rank(pct=True, ascending=False)

    # NEW: Volume ratio rank (unusual volume)
    df["volume_rank"] = df.groupby("date")["volume_ratio"].rank(pct=True)

    logger.info("Cross-sectional features added.")
    return df


def save_processed(df: pd.DataFrame, filename: str = "price_features.parquet"):
    """Saves processed DataFrame to parquet."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / filename
    df.to_parquet(path, index=False)
    logger.info(f"Saved processed data to {path} ({path.stat().st_size / 1e6:.1f} MB)")


def load_processed(filename: str = "price_features.parquet") -> pd.DataFrame:
    """Loads processed data from parquet cache."""
    path = PROCESSED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"No processed data at {path}. Run pipeline first.")
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df):,} rows from {path}")
    return df


def run_pipeline(config: dict, tickers: list[str]) -> pd.DataFrame:
    """Full pipeline: download → clean → features → save."""
    logger.info("=" * 50)
    logger.info("Starting data pipeline")
    logger.info("=" * 50)

    df = download_price_data(
        tickers,
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )
    df = clean_price_data(df)
    df = add_technical_features(df)
    df = add_cross_sectional_features(df)
    save_processed(df)

    logger.info("Pipeline complete.")
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.universe import build_universe

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    universe = build_universe(config)
    df = run_pipeline(config, universe)
    print(df.tail())
    print(df.dtypes)
