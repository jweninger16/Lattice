"""
data/premarket_scanner.py
--------------------------
Pre-market gap and volume scanner using yfinance prepost=True.
Scans the ORB universe at ~9:00 AM to rank tickers by gap magnitude
and pre-market volume before the opening range forms.

Usage:
    from data.premarket_scanner import scan_premarket_gaps, rank_candidates

    # Standalone test:
    python data/premarket_scanner.py
"""

import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, date, time as dtime
from loguru import logger


SI_DATA_PATH = Path("data/gap_scanner_cache/float_short_data.json")


def scan_premarket_gaps(tickers: list[str], timeout: int = 60) -> pd.DataFrame:
    """
    Fetch pre-market data for all tickers and compute gap + volume metrics.

    Returns DataFrame with columns:
        ticker, prev_close, pm_price, gap_pct, gap_direction,
        pm_volume, avg_volume, pm_vol_ratio
    Sorted by abs(gap_pct) descending.
    Returns empty DataFrame on failure.
    """
    if not tickers:
        return pd.DataFrame()

    logger.info(f"Scanning pre-market data for {len(tickers)} tickers...")

    try:
        # Batch download with pre/post market data — 5 days for avg volume
        raw = yf.download(
            tickers,
            period="5d",
            interval="1m",
            prepost=True,
            auto_adjust=True,
            progress=False,
            threads=True,
            timeout=timeout,
        )
    except Exception as e:
        logger.warning(f"Pre-market download failed: {e}")
        return pd.DataFrame()

    if raw.empty:
        logger.warning("Pre-market download returned empty data")
        return pd.DataFrame()

    # Handle single-ticker case (no MultiIndex)
    if len(tickers) == 1:
        raw.columns = pd.MultiIndex.from_product([raw.columns, tickers])

    today = date.today()
    results = []

    for ticker in tickers:
        try:
            # Extract ticker's data
            if isinstance(raw.columns, pd.MultiIndex):
                tk = raw.xs(ticker, axis=1, level=1) if ticker in raw.columns.get_level_values(1) else None
            else:
                tk = raw

            if tk is None or tk.empty:
                continue

            tk = tk.dropna(subset=["Close"])
            if tk.empty:
                continue

            # Ensure timezone-aware index is converted to ET
            idx = tk.index
            if idx.tz is not None:
                idx = idx.tz_convert("America/New_York")

            tk = tk.copy()
            tk.index = idx
            tk["_date"] = idx.date
            tk["_time"] = idx.time

            # Previous trading day's close (last regular-hours bar before today)
            prev_days = tk[tk["_date"] < today]
            if prev_days.empty:
                continue
            # Regular hours only for prev close
            prev_rth = prev_days[(prev_days["_time"] >= dtime(9, 30)) &
                                 (prev_days["_time"] <= dtime(16, 0))]
            if prev_rth.empty:
                prev_close = prev_days["Close"].iloc[-1]
            else:
                prev_close = prev_rth["Close"].iloc[-1]

            # Today's pre-market data (4:00 AM - 9:29 AM)
            today_data = tk[tk["_date"] == today]
            if today_data.empty:
                continue
            pm_data = today_data[today_data["_time"] < dtime(9, 30)]

            if pm_data.empty:
                # No pre-market data yet, use today's first available price
                pm_price = today_data["Close"].iloc[-1]
                pm_volume = 0
            else:
                pm_price = pm_data["Close"].iloc[-1]
                pm_volume = pm_data["Volume"].sum()

            # Average daily volume (regular hours, prior days)
            avg_volume = 0
            for d in sorted(prev_days["_date"].unique()):
                day_data = prev_days[(prev_days["_date"] == d) &
                                     (prev_days["_time"] >= dtime(9, 30)) &
                                     (prev_days["_time"] <= dtime(16, 0))]
                avg_volume += day_data["Volume"].sum()
            n_days = len(prev_days["_date"].unique())
            avg_volume = avg_volume / max(n_days, 1)

            gap_pct = (pm_price / prev_close - 1) * 100 if prev_close > 0 else 0
            pm_vol_ratio = pm_volume / avg_volume if avg_volume > 0 else 0

            results.append({
                "ticker": ticker,
                "prev_close": round(prev_close, 2),
                "pm_price": round(pm_price, 2),
                "gap_pct": round(gap_pct, 4),
                "gap_direction": "up" if gap_pct > 0 else "down" if gap_pct < 0 else "flat",
                "pm_volume": int(pm_volume),
                "avg_volume": int(avg_volume),
                "pm_vol_ratio": round(pm_vol_ratio, 3),
            })

        except Exception as e:
            logger.debug(f"  {ticker}: pre-market parse error: {e}")
            continue

    if not results:
        logger.warning("No pre-market data parsed for any ticker")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values("gap_pct", key=abs, ascending=False).reset_index(drop=True)
    logger.info(f"Pre-market scan complete: {len(df)} tickers with data")
    return df


def load_short_interest() -> dict:
    """
    Load existing short interest data from gap_scanner_cache.
    Returns {ticker: {short_pct_float: float, ...}} or empty dict.
    """
    if not SI_DATA_PATH.exists():
        return {}
    try:
        with open(SI_DATA_PATH) as f:
            data = json.load(f)
        # Normalize: ensure short_pct_float exists
        result = {}
        for ticker, info in data.items():
            si_pct = info.get("short_pct_float") or info.get("shortPercentOfFloat") or 0
            if isinstance(si_pct, str):
                try:
                    si_pct = float(si_pct)
                except ValueError:
                    si_pct = 0
            result[ticker] = {"short_pct_float": si_pct, **info}
        return result
    except Exception as e:
        logger.warning(f"Could not load short interest data: {e}")
        return {}


def compute_intel_score(row: dict, si_data: dict, has_news_catalyst: bool = False) -> float:
    """
    Score a single ticker based on pre-market intel.
    Returns 0.0 - 1.0.
    """
    score = 0.0

    # 1. Gap magnitude (0-0.30)
    gap = abs(row.get("gap_pct", 0))
    if 0.2 <= gap <= 0.5:
        score += 0.30       # Sweet spot — within ORB gap filter, has momentum
    elif 0.5 < gap <= 2.0:
        score += 0.15       # Moderate gap — tradeable but may overshoot
    elif gap < 0.2:
        score += 0.20       # Flat open — fine for clean ORB
    # gap > 2% gets 0 (will be filtered by MAX_GAP_PCT anyway)

    # 2. Pre-market volume ratio (0-0.30)
    pm_vr = row.get("pm_vol_ratio", 0)
    if pm_vr >= 3.0:
        score += 0.30       # Heavy pre-market interest
    elif pm_vr >= 1.5:
        score += 0.20
    elif pm_vr >= 0.5:
        score += 0.10

    # 3. Short interest bonus (0-0.15)
    ticker = row.get("ticker", "")
    si_info = si_data.get(ticker, {})
    si_pct = si_info.get("short_pct_float", 0) or 0
    if si_pct > 0.15:
        score += 0.15       # High SI = squeeze candidate on breakout
    elif si_pct > 0.08:
        score += 0.10
    elif si_pct > 0.05:
        score += 0.05

    # 4. News catalyst (0-0.15)
    if has_news_catalyst:
        score += 0.15       # Gap with catalyst = more follow-through

    return min(score, 1.0)


def rank_candidates(
    gaps_df: pd.DataFrame,
    si_data: dict | None = None,
    earnings_today: set | None = None,
    news_tickers: set | None = None,
    max_watchlist: int = 30,
) -> pd.DataFrame:
    """
    Score and rank pre-market candidates.
    Returns DataFrame with intel_score, intel_rank, and skip_reason columns.
    """
    if gaps_df.empty:
        return gaps_df

    si_data = si_data or {}
    earnings_today = earnings_today or set()
    news_tickers = news_tickers or set()

    rows = []
    for _, row in gaps_df.iterrows():
        r = row.to_dict()
        ticker = r["ticker"]

        # Skip earnings-day stocks
        if ticker in earnings_today:
            r["intel_score"] = 0.0
            r["skip_reason"] = "earnings_today"
            rows.append(r)
            continue

        has_news = ticker in news_tickers
        r["intel_score"] = compute_intel_score(r, si_data, has_news)
        r["skip_reason"] = ""
        rows.append(r)

    df = pd.DataFrame(rows)
    df = df.sort_values("intel_score", ascending=False).reset_index(drop=True)
    df["intel_rank"] = range(1, len(df) + 1)

    # Log top candidates
    top = df[df["skip_reason"] == ""].head(10)
    if not top.empty:
        logger.info("Top pre-market candidates:")
        for _, r in top.iterrows():
            logger.info(f"  {r['ticker']:<6} gap={r['gap_pct']:+.2f}%  "
                        f"pm_vol={r['pm_vol_ratio']:.1f}x  "
                        f"score={r['intel_score']:.2f}")

    return df


if __name__ == "__main__":
    # Standalone test
    import sys
    sys.path.insert(0, ".")

    universe_path = Path("data/orb_universe.csv")
    if not universe_path.exists():
        print("No ORB universe. Run: python research/orb_volume_stock_backtest.py --universe-only")
        sys.exit(1)

    tickers = pd.read_csv(universe_path)["ticker"].tolist()
    print(f"Scanning {len(tickers)} tickers for pre-market activity...\n")

    gaps = scan_premarket_gaps(tickers)
    if gaps.empty:
        print("No pre-market data available (market may not be open yet)")
        sys.exit(0)

    si = load_short_interest()
    ranked = rank_candidates(gaps, si_data=si)

    print(f"\n{'='*80}")
    print(f"  PRE-MARKET SCAN RESULTS ({datetime.now().strftime('%H:%M:%S')})")
    print(f"{'='*80}")
    print(f"  {'Ticker':<8} {'Gap%':>7} {'Dir':>5} {'PM Vol':>10} {'PM/Avg':>7} "
          f"{'SI%':>6} {'Score':>6}")
    print(f"  {'-'*60}")
    for _, r in ranked.head(20).iterrows():
        si_pct = si.get(r["ticker"], {}).get("short_pct_float", 0) or 0
        skip = f" [{r['skip_reason']}]" if r.get("skip_reason") else ""
        print(f"  {r['ticker']:<8} {r['gap_pct']:>+6.2f}% {r['gap_direction']:>5} "
              f"{r['pm_volume']:>10,} {r['pm_vol_ratio']:>6.1f}x "
              f"{si_pct*100:>5.1f}% {r['intel_score']:>5.2f}{skip}")
