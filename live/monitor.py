"""
live/monitor.py
---------------
Intraday position monitor. Runs during market hours and alerts
you immediately if any position hits its stop loss or profit target.

How it works:
  - Checks open positions every 30 minutes (configurable)
  - Fetches current prices via yfinance
  - Compares against each position's stop and target
  - Sends SMS only when action is needed (no spam)
  - Automatically sleeps outside market hours
  - Tracks which alerts have already been sent to avoid duplicates

Usage:
    python live/monitor.py              # Run manually
    python main.py monitor              # Via main entry point
    pythonw live/monitor.py             # Background on Windows

Schedule:
    Runs Mon-Fri 9:30 AM - 4:00 PM ET only.
    Checks every 30 minutes by default.
    Outside market hours, sleeps until next open.
"""

import sys
import os
import time
import json
import pandas as pd
import yfinance as yf
from datetime import datetime, date, timedelta
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")

CHECK_INTERVAL_MINUTES = 30
ALERTS_FILE = Path("live/alerts_sent_today.json")


def is_market_hours() -> bool:
    """Returns True if current time is during US market hours (ET)."""
    try:
        import zoneinfo
        et = zoneinfo.ZoneInfo("America/New_York")
    except ImportError:
        # Fallback: assume local time is ET or close enough
        # (will be slightly off for non-ET timezones)
        et = None

    if et:
        now = datetime.now(et)
    else:
        now = datetime.now()

    # Skip weekends
    if now.weekday() >= 5:
        return False

    # Market hours: 9:30 AM - 4:00 PM ET
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= now <= market_close


def minutes_until_market_open() -> int:
    """Returns minutes until next market open. Used for sleep scheduling."""
    try:
        import zoneinfo
        et = zoneinfo.ZoneInfo("America/New_York")
        now = datetime.now(et)
    except ImportError:
        now = datetime.now()

    # If it's before 9:30 today (weekday), wait until 9:30
    if now.weekday() < 5:
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        if now < market_open:
            delta = (market_open - now).total_seconds() / 60
            return int(delta) + 1

    # Otherwise wait until next weekday 9:30 AM
    days_ahead = 1
    while True:
        next_day = now + timedelta(days=days_ahead)
        if next_day.weekday() < 5:  # Monday-Friday
            next_open = next_day.replace(hour=9, minute=30, second=0, microsecond=0)
            delta = (next_open - now).total_seconds() / 60
            return int(delta) + 1
        days_ahead += 1


def load_sent_alerts() -> dict:
    """Loads today's already-sent alerts to avoid duplicates."""
    if not ALERTS_FILE.exists():
        return {"date": str(date.today()), "alerts": []}

    with open(ALERTS_FILE) as f:
        data = json.load(f)

    # Reset if it's a new day
    if data.get("date") != str(date.today()):
        return {"date": str(date.today()), "alerts": []}

    return data


def save_sent_alert(ticker: str, alert_type: str):
    """Records that an alert was sent to avoid resending."""
    data = load_sent_alerts()
    alert_key = f"{ticker}:{alert_type}"
    if alert_key not in data["alerts"]:
        data["alerts"].append(alert_key)
    ALERTS_FILE.parent.mkdir(exist_ok=True)
    with open(ALERTS_FILE, "w") as f:
        json.dump(data, f)


def was_alert_sent(ticker: str, alert_type: str) -> bool:
    """Checks if this alert was already sent today."""
    data = load_sent_alerts()
    return f"{ticker}:{alert_type}" in data["alerts"]


def fetch_current_prices(tickers: list) -> dict:
    """Fetches current prices for a list of tickers."""
    if not tickers:
        return {}

    prices = {}
    try:
        data = yf.download(
            tickers,
            period="1d",
            interval="1m",
            auto_adjust=True,
            progress=False,
        )

        if data.empty:
            # Fallback to daily data
            data = yf.download(
                tickers, period="2d", auto_adjust=True, progress=False
            )

        if data.empty:
            return prices

        # Flatten MultiIndex columns if yfinance returns them
        if isinstance(data.columns, pd.MultiIndex):
            if len(tickers) == 1:
                # Single ticker: drop the ticker level, keep field names
                data.columns = [c[0] for c in data.columns]
            # else: keep MultiIndex for multi-ticker handling below

        # Handle single vs multiple tickers
        if len(tickers) == 1:
            t = tickers[0]
            if "Close" in data.columns:
                close = data["Close"].dropna()
                high = data["High"].dropna()
                low = data["Low"].dropna()
                if len(close) > 0:
                    prices[t] = {
                        "close": float(close.iloc[-1]),
                        "high": float(high.max()),
                        "low": float(low.min()),
                    }
        else:
            for t in tickers:
                try:
                    if isinstance(data.columns, pd.MultiIndex) and t in data["Close"].columns:
                        close = data["Close"][t].dropna()
                        high = data["High"][t].dropna()
                        low = data["Low"][t].dropna()
                        if len(close) > 0:
                            prices[t] = {
                                "close": float(close.iloc[-1]),
                                "high": float(high.max()),
                                "low": float(low.min()),
                            }
                except Exception:
                    continue

    except Exception as e:
        logger.warning(f"Price fetch failed: {e}")

    return prices


def check_positions() -> list:
    """
    Checks all open positions against current prices.
    Returns list of alerts to send.
    """
    from live.positions import get_open_positions

    positions = get_open_positions()
    if not positions:
        return []

    tickers = [p["ticker"] for p in positions if p.get("ticker")]
    if not tickers:
        return []

    logger.info(f"Checking {len(tickers)} positions: {', '.join(tickers)}")
    prices = fetch_current_prices(tickers)

    alerts = []
    for p in positions:
        ticker = p["ticker"]
        if ticker not in prices:
            continue

        current = prices[ticker]
        entry = p["entry_price"]
        stop = p["stop_price"]
        target = p["target_price"]
        pnl_pct = (current["close"] / entry - 1) * 100

        # Check stop loss (use intraday low)
        if current["low"] <= stop and not was_alert_sent(ticker, "stop"):
            alerts.append({
                "ticker": ticker,
                "type": "STOP HIT",
                "message": (
                    f"⚠ STOP HIT: {ticker}\n"
                    f"Low ${current['low']:.2f} ≤ stop ${stop:.2f}\n"
                    f"Entry ${entry:.2f} | Now ${current['close']:.2f}\n"
                    f"P&L: {pnl_pct:+.1f}%\n"
                    f"ACTION: SELL at market"
                ),
            })
            save_sent_alert(ticker, "stop")

        # Check profit target (use intraday high)
        elif current["high"] >= target and not was_alert_sent(ticker, "target"):
            alerts.append({
                "ticker": ticker,
                "type": "TARGET HIT",
                "message": (
                    f"✓ TARGET HIT: {ticker}\n"
                    f"High ${current['high']:.2f} ≥ target ${target:.2f}\n"
                    f"Entry ${entry:.2f} | Now ${current['close']:.2f}\n"
                    f"P&L: {pnl_pct:+.1f}%\n"
                    f"ACTION: SELL at market"
                ),
            })
            save_sent_alert(ticker, "target")

        # Warning: approaching stop (within 1% of stop)
        elif current["close"] <= stop * 1.01 and not was_alert_sent(ticker, "near_stop"):
            alerts.append({
                "ticker": ticker,
                "type": "NEAR STOP",
                "message": (
                    f"⚡ NEAR STOP: {ticker}\n"
                    f"Price ${current['close']:.2f} approaching stop ${stop:.2f}\n"
                    f"P&L: {pnl_pct:+.1f}%\n"
                    f"Watch closely"
                ),
            })
            save_sent_alert(ticker, "near_stop")

        else:
            logger.debug(f"  {ticker}: ${current['close']:.2f} ({pnl_pct:+.1f}%) — OK")

    return alerts


def send_alert(message: str):
    """Sends an SMS alert."""
    try:
        from live.alerts import send_sms
        send_sms(message)
        logger.info(f"Alert sent: {message[:50]}...")
    except Exception as e:
        logger.error(f"Failed to send alert: {e}")
        # Still print to console as fallback
        print(f"\n{'='*40}")
        print(message)
        print(f"{'='*40}\n")


def run_monitor():
    """Main monitoring loop. Runs during market hours, sleeps otherwise."""
    logger.info("=" * 50)
    logger.info("  INTRADAY POSITION MONITOR")
    logger.info("  Checks every 30 min during market hours")
    logger.info("  Alerts on stop/target hits via SMS")
    logger.info("=" * 50)

    while True:
        if is_market_hours():
            try:
                alerts = check_positions()
                if alerts:
                    for alert in alerts:
                        send_alert(alert["message"])
                        logger.warning(f"ALERT: {alert['type']} on {alert['ticker']}")
                else:
                    logger.info("All positions OK")
            except Exception as e:
                logger.error(f"Monitor check failed: {e}")

            logger.info(f"Next check in {CHECK_INTERVAL_MINUTES} minutes...")
            time.sleep(CHECK_INTERVAL_MINUTES * 60)

        else:
            wait = minutes_until_market_open()
            if wait > 120:
                logger.info(f"Market closed. Sleeping {wait // 60} hours until next open...")
            else:
                logger.info(f"Market closed. Sleeping {wait} minutes until open...")
            time.sleep(min(wait, 60) * 60)  # Wake up at least every hour to recheck


def run_once():
    """Single check — useful for testing."""
    from live.positions import get_open_positions

    positions = get_open_positions()
    if not positions:
        print("No open positions to monitor.")
        return

    print(f"\nChecking {len(positions)} positions...")
    alerts = check_positions()

    if alerts:
        print(f"\n{len(alerts)} ALERT(S):")
        for alert in alerts:
            print(f"\n{alert['message']}")
    else:
        print("All positions within bounds.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true",
                        help="Run a single check and exit")
    args = parser.parse_args()

    if args.once:
        run_once()
    else:
        run_monitor()
