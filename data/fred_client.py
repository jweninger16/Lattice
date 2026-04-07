"""
data/fred_client.py
--------------------
FRED API client for VIX term structure analysis.

Fetches spot VIX (VIXCLS) and 3-month VIX (VIXCLS3M) to determine
contango vs backwardation — a key market fear/complacency signal.

Free tier: 120 requests/minute, no credit card required.
Get API key at: https://fred.stlouisfed.org/docs/api/api_key.html

Usage:
    from data.fred_client import FREDClient
    client = FREDClient()
    vix_data = client.get_vix_term_structure()

    # Standalone test:
    python data/fred_client.py
"""

import os
import pickle
import requests
from pathlib import Path
from datetime import datetime, date, timedelta
from loguru import logger


CACHE_DIR = Path("data/market_intel_cache")


class FREDClient:
    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self):
        self.api_key = os.getenv("FRED_API_KEY")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _get_series(self, series_id: str, lookback_days: int = 10) -> float | None:
        """Fetch the most recent observation for a FRED series."""
        if not self.api_key:
            return None

        start = (date.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        try:
            resp = requests.get(self.BASE_URL, params={
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "observation_start": start,
                "sort_order": "desc",
                "limit": 5,
            }, timeout=10)

            if resp.status_code != 200:
                logger.warning(f"FRED {series_id}: HTTP {resp.status_code}")
                return None

            data = resp.json()
            observations = data.get("observations", [])
            # Find most recent non-missing value
            for obs in observations:
                val = obs.get("value", ".")
                if val != ".":
                    return float(val)
            return None

        except requests.Timeout:
            logger.warning(f"FRED {series_id}: timeout")
            return None
        except Exception as e:
            logger.warning(f"FRED {series_id}: {e}")
            return None

    def get_vix_term_structure(self) -> dict | None:
        """
        Fetch VIX spot and 3-month VIX to compute term structure.

        Returns:
            {
                vix: float,         # Spot VIX (VIXCLS)
                vix3m: float,       # 3-month VIX (VIXCLS3M)
                term_slope: float,  # (vix3m - vix) / vix
                regime: str,        # "contango" / "backwardation" / "flat"
                signal: str,        # "calm" / "fear" / "neutral"
            }
        or None on failure.

        Interpretation:
            - Contango (slope > 0.05): Market expects current calm to persist.
              Normal state. Good for ORB — breakouts are cleaner.
            - Backwardation (slope < -0.05): Near-term fear exceeds long-term.
              Hedging demand is high. ORB breakouts may whipsaw.
            - Flat (-0.05 to 0.05): Ambiguous.
        """
        # Check cache first (6-hour TTL — VIX updates daily after close)
        cached = self._load_cache("fred_vix_term", ttl_hours=6)
        if cached is not None:
            return cached

        vix = self._get_series("VIXCLS")
        vix3m = self._get_series("VIXCLS3M")

        if vix is None or vix3m is None:
            logger.warning("FRED: Could not fetch VIX term structure "
                          f"(vix={vix}, vix3m={vix3m})")
            return None

        term_slope = (vix3m - vix) / vix if vix > 0 else 0

        if term_slope > 0.05:
            regime = "contango"
            signal = "calm"
        elif term_slope < -0.05:
            regime = "backwardation"
            signal = "fear"
        else:
            regime = "flat"
            signal = "neutral"

        result = {
            "vix": round(vix, 2),
            "vix3m": round(vix3m, 2),
            "term_slope": round(term_slope, 4),
            "regime": regime,
            "signal": signal,
        }

        self._save_cache("fred_vix_term", result)
        logger.info(f"VIX term structure: VIX={vix:.1f}, VIX3M={vix3m:.1f}, "
                    f"slope={term_slope:+.3f} ({regime}/{signal})")
        return result

    # ── Cache Helpers ─────────────────────────────────────────────

    def _load_cache(self, name: str, ttl_hours: int = 6):
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

    def _save_cache(self, name: str, data):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(CACHE_DIR / f"{name}.pkl", "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.debug(f"Cache write failed for {name}: {e}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    client = FREDClient()
    if not client.available:
        print("FRED_API_KEY not set in .env")
        print("Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        exit(1)

    print(f"\nFRED API Test ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print("=" * 60)

    result = client.get_vix_term_structure()
    if result:
        print(f"  VIX (spot):   {result['vix']:.2f}")
        print(f"  VIX3M:        {result['vix3m']:.2f}")
        print(f"  Term slope:   {result['term_slope']:+.4f}")
        print(f"  Regime:       {result['regime']}")
        print(f"  Signal:       {result['signal']}")
    else:
        print("  Failed to fetch VIX term structure")
