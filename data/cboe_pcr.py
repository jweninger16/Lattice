"""
data/cboe_pcr.py
-----------------
CBOE put/call ratio fetcher.

Downloads the equity put/call ratio from CBOE's public data.
No API key required.

Interpretation:
    PCR > 1.0  = heavy put buying (bearish sentiment, contrarian bullish)
    PCR < 0.7  = heavy call buying (bullish sentiment, contrarian bearish)
    PCR 0.7-1.0 = neutral

Usage:
    from data.cboe_pcr import fetch_put_call_ratio
    pcr = fetch_put_call_ratio()

    # Standalone test:
    python data/cboe_pcr.py
"""

import pickle
import requests
import pandas as pd
from io import StringIO
from pathlib import Path
from datetime import datetime
from loguru import logger


CACHE_DIR = Path("data/market_intel_cache")

# CBOE publishes historical P/C ratio data as CSV
CBOE_EQUITY_PCR_URL = "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypc.csv"
CBOE_TOTAL_PCR_URL = "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpc.csv"


def _download_pcr(url: str) -> pd.DataFrame | None:
    """Download PCR CSV from CBOE."""
    try:
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SwingTrader/1.0",
        })
        if resp.status_code != 200:
            logger.warning(f"CBOE PCR download: HTTP {resp.status_code}")
            return None

        # Parse CSV — CBOE format has a header row
        df = pd.read_csv(StringIO(resp.text))
        df.columns = [c.strip().lower() for c in df.columns]
        return df
    except Exception as e:
        logger.warning(f"CBOE PCR download failed: {e}")
        return None


def fetch_put_call_ratio() -> dict | None:
    """
    Fetch the latest put/call ratio from CBOE.

    Returns:
        {
            pcr_equity: float,   # Equity-only P/C ratio
            pcr_total: float,    # Total exchange P/C ratio
            signal: str,         # "bullish" / "bearish" / "neutral"
            date: str,           # Date of the data
        }
    or None on failure.
    """
    # Check cache (6-hour TTL — data updates once daily)
    cached = _load_cache("cboe_pcr", ttl_hours=6)
    if cached is not None:
        return cached

    # Try equity PCR first
    pcr_equity = None
    pcr_total = None
    pcr_date = None

    df = _download_pcr(CBOE_EQUITY_PCR_URL)
    if df is not None and not df.empty:
        # Find the P/C ratio column
        pcr_col = None
        for col in df.columns:
            if "p/c" in col or "put" in col:
                pcr_col = col
                break
        date_col = None
        for col in df.columns:
            if "date" in col:
                date_col = col
                break

        if pcr_col and date_col:
            last = df.iloc[-1]
            try:
                pcr_equity = float(last[pcr_col])
                pcr_date = str(last[date_col])
            except (ValueError, TypeError):
                pass

    # Try total PCR
    df = _download_pcr(CBOE_TOTAL_PCR_URL)
    if df is not None and not df.empty:
        pcr_col = None
        for col in df.columns:
            if "p/c" in col or "put" in col:
                pcr_col = col
                break
        if pcr_col:
            try:
                pcr_total = float(df.iloc[-1][pcr_col])
            except (ValueError, TypeError):
                pass

    if pcr_equity is None and pcr_total is None:
        logger.warning("CBOE: Could not fetch any put/call ratio data")
        return None

    # Use equity PCR for signal (more relevant for stock trading)
    pcr = pcr_equity or pcr_total
    if pcr > 1.0:
        signal = "bullish"    # Contrarian: heavy puts = fear = bottoming
    elif pcr < 0.7:
        signal = "bearish"    # Contrarian: heavy calls = complacency = topping
    else:
        signal = "neutral"

    result = {
        "pcr_equity": pcr_equity,
        "pcr_total": pcr_total,
        "signal": signal,
        "date": pcr_date,
    }

    _save_cache("cboe_pcr", result)
    logger.info(f"CBOE P/C ratio: equity={pcr_equity}, total={pcr_total}, "
                f"signal={signal}")
    return result


# ── Cache Helpers ─────────────────────────────────────────────────

def _load_cache(name: str, ttl_hours: int = 6):
    cache_file = CACHE_DIR / f"{name}.pkl"
    if not cache_file.exists():
        return None
    try:
        age = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime))
        if age.total_seconds() / 3600 > ttl_hours:
            return None
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _save_cache(name: str, data):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(CACHE_DIR / f"{name}.pkl", "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        logger.debug(f"Cache write failed for {name}: {e}")


if __name__ == "__main__":
    print(f"\nCBOE Put/Call Ratio Test ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print("=" * 60)

    result = fetch_put_call_ratio()
    if result:
        print(f"  Equity P/C:  {result['pcr_equity']}")
        print(f"  Total P/C:   {result['pcr_total']}")
        print(f"  Signal:      {result['signal']}")
        print(f"  Data date:   {result['date']}")
    else:
        print("  Failed to fetch P/C ratio")
