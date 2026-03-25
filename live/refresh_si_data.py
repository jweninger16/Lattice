"""
live/refresh_si_data.py
-------------------------
Refreshes short interest and float data for the scanner.

FINRA publishes short interest twice monthly (around 15th and last day).
Run this weekly to stay current — daily is unnecessary.

Can be scheduled via Windows Task Scheduler or run manually:
    python live/refresh_si_data.py

Data is saved to: data/gap_scanner_cache/float_short_data.json
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, ".")

CACHE_DIR = Path("data/gap_scanner_cache")


def get_universe():
    try:
        from data.universe import FALLBACK_TICKERS, ETF_TICKERS
        return [t for t in FALLBACK_TICKERS if t not in ETF_TICKERS]
    except ImportError:
        return [
            "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AMD","NFLX",
            "CRM","AVGO","ORCL","QCOM","AMAT","MU","INTC","UBER","SHOP",
            "COIN","PLTR","SOFI","DKNG","SNAP","PINS","RBLX",
            "NET","CRWD","ZS","DDOG","MDB","SNOW","ABNB","DASH","ROKU",
            "JPM","BAC","GS","MS","C","WFC","V","MA","AXP","PYPL","SCHW",
            "XOM","CVX","COP","OXY","SLB","HAL","MPC","VLO","PSX",
            "UNH","JNJ","PFE","ABBV","MRK","LLY","BMY","AMGN","GILD",
            "CAT","DE","GE","HON","BA","RTX","LMT","GD","NOC",
            "HD","LOW","TGT","COST","WMT","TJX","ROST","DG","DLTR",
            "DIS","CMCSA","T","TMUS","VZ","CHTR",
        ]


def refresh():
    import yfinance as yf

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "float_short_data.json"

    # Load existing data to preserve it if a ticker fails
    existing = {}
    if cache_path.exists():
        with open(cache_path) as f:
            existing = json.load(f)

    tickers = get_universe()
    print(f"Refreshing short interest data for {len(tickers)} tickers...")
    data = {}
    failed = []

    for i, ticker in enumerate(tickers):
        try:
            info = yf.Ticker(ticker).info
            data[ticker] = {
                "float_shares": info.get("floatShares", None),
                "short_interest": info.get("sharesShort", None),
                "short_pct_float": info.get("shortPercentOfFloat", None),
                "shares_outstanding": info.get("sharesOutstanding", None),
                "market_cap": info.get("marketCap", None),
                "avg_volume": info.get("averageVolume", None),
                "sector": info.get("sector", "Other"),
                "updated": str(datetime.now()),
            }
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(tickers)} done...")
            time.sleep(0.2)
        except Exception as e:
            failed.append(ticker)
            # Keep existing data if available
            if ticker in existing:
                data[ticker] = existing[ticker]

    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    # Summary
    si_stocks = sum(1 for t in data.values()
                    if (t.get("short_pct_float") or 0) > 0.05)
    low_float = sum(1 for t in data.values()
                    if (t.get("float_shares") or float('inf')) < 50_000_000)

    print(f"\nDone! {len(data)} tickers saved to {cache_path}")
    print(f"  High SI (>5%): {si_stocks} stocks")
    print(f"  Low float (<50M): {low_float} stocks")
    if failed:
        print(f"  Failed: {', '.join(failed)}")

    # Send Discord notification
    try:
        from live.alerts import send_discord
        send_discord(
            f"SI data refreshed: {len(data)} tickers | "
            f"High SI: {si_stocks} | Low float: {low_float}"
        )
    except Exception:
        pass


if __name__ == "__main__":
    refresh()
