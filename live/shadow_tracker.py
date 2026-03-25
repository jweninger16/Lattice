"""
live/shadow_tracker.py
-----------------------
Shadow tracking for ML training data.

After each scan, records ALL candidates that passed the filter — not just
the one we bought. At end of day, fetches actual intraday data for each
shadow candidate and calculates what would have happened if we traded it.

This 3x-5x's the training data without risking a dollar.
"""

import csv
import logging
import time
from pathlib import Path
from datetime import date, datetime, timedelta

import yfinance as yf
import pandas as pd

logger = logging.getLogger("shadow_tracker")

SHADOW_LOG = Path("data/shadow_log.csv")

# CSV columns — mirrors the real trade log fields for easy merging
COLUMNS = [
    "date", "ticker", "rank", "score",
    "gap_pct", "rvol", "atr_14", "atr_pct",
    "mom_5d", "above_sma50", "si_pct", "float_shares",
    "call_vol_oi_ratio", "options_signal",
    "st_messages", "st_bullish", "st_bearish", "st_bull_ratio", "st_watchers",
    "entry_price", "exit_price", "exit_reason",
    "high_of_day", "low_of_day",
    "pnl_usd", "pnl_pct",
    "was_traded", "shadow",
]


class ShadowTracker:
    """Tracks all scan candidates for shadow P&L calculation."""

    def __init__(self):
        self.today_candidates = []  # List of candidate dicts
        self.traded_tickers = set()  # Tickers we actually traded
        self._ensure_csv()

    def _ensure_csv(self):
        """Create CSV with headers if it doesn't exist."""
        SHADOW_LOG.parent.mkdir(parents=True, exist_ok=True)
        if not SHADOW_LOG.exists():
            with open(SHADOW_LOG, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                writer.writeheader()

    def record_candidates(self, candidates_df, traded_ticker=None):
        """
        Record all candidates from a scan.
        Called after run_pre_market_scan() returns results.

        Args:
            candidates_df: DataFrame from run_pre_market_scan()
            traded_ticker: The ticker we actually bought (if any)
        """
        if candidates_df is None or candidates_df.empty:
            return

        if traded_ticker:
            self.traded_tickers.add(traded_ticker)

        for i, row in candidates_df.iterrows():
            ticker = row["ticker"]

            # Skip if we already recorded this ticker today
            if any(c["ticker"] == ticker for c in self.today_candidates):
                continue

            candidate = {
                "date": str(date.today()),
                "ticker": ticker,
                "rank": i + 1,
                "score": round(row.get("score", 0), 2),
                "gap_pct": round(row.get("gap_pct", 0), 2),
                "rvol": round(row.get("rvol", 0), 1),
                "atr_14": round(row.get("atr_14", 0), 2),
                "atr_pct": round(row.get("atr_pct", 0), 2),
                "mom_5d": round(row.get("mom_5d", 0), 2),
                "above_sma50": int(row.get("above_sma50", 0)),
                "si_pct": round(row.get("si_pct", 0), 1),
                "float_shares": row.get("float_shares", 0),
                "call_vol_oi_ratio": round(row.get("call_vol_oi_ratio", 0), 2),
                "options_signal": row.get("options_signal", ""),
                "st_messages": int(row.get("st_messages", 0)),
                "st_bullish": int(row.get("st_bullish", 0)),
                "st_bearish": int(row.get("st_bearish", 0)),
                "st_bull_ratio": round(row.get("st_bull_ratio", 0.5), 2),
                "st_watchers": int(row.get("st_watchers", 0)),
                "entry_price": round(row.get("open", 0), 2),
            }
            self.today_candidates.append(candidate)

        logger.info(f"Shadow tracker: {len(self.today_candidates)} candidates "
                    f"recorded ({len(self.traded_tickers)} traded)")

    def mark_traded(self, ticker):
        """Mark a ticker as actually traded (not shadow)."""
        self.traded_tickers.add(ticker)

    def resolve_shadows(self, trail_atr_mult=0.20):
        """
        Called at end of day. Fetches actual intraday data for all
        shadow candidates and calculates hypothetical P&L.

        Uses 5-minute bars to simulate trailing stop behavior.
        """
        if not self.today_candidates:
            logger.info("Shadow tracker: no candidates to resolve")
            return

        shadow_count = 0
        real_count = 0

        for candidate in self.today_candidates:
            ticker = candidate["ticker"]
            was_traded = ticker in self.traded_tickers
            entry_price = candidate["entry_price"]
            atr = candidate["atr_14"]

            if entry_price <= 0 or atr <= 0:
                continue

            try:
                # Fetch today's intraday data (5-min bars)
                data = yf.download(
                    ticker, period="1d", interval="5m",
                    progress=False, auto_adjust=True
                )

                if data is None or data.empty:
                    continue

                # Flatten MultiIndex if needed
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                high_of_day = float(data["High"].max())
                low_of_day = float(data["Low"].min())

                # Simulate trailing stop
                trail_amount = atr * trail_atr_mult
                exit_price, exit_reason = self._simulate_trailing_stop(
                    data, entry_price, trail_amount
                )

                pnl_usd_per_share = exit_price - entry_price
                pnl_pct = (pnl_usd_per_share / entry_price) * 100

                # Write to CSV
                row = {
                    "date": candidate["date"],
                    "ticker": ticker,
                    "rank": candidate["rank"],
                    "score": candidate["score"],
                    "gap_pct": candidate["gap_pct"],
                    "rvol": candidate["rvol"],
                    "atr_14": candidate["atr_14"],
                    "atr_pct": candidate["atr_pct"],
                    "mom_5d": candidate["mom_5d"],
                    "above_sma50": candidate["above_sma50"],
                    "si_pct": candidate["si_pct"],
                    "float_shares": candidate["float_shares"],
                    "call_vol_oi_ratio": candidate["call_vol_oi_ratio"],
                    "options_signal": candidate["options_signal"],
                    "st_messages": candidate.get("st_messages", 0),
                    "st_bullish": candidate.get("st_bullish", 0),
                    "st_bearish": candidate.get("st_bearish", 0),
                    "st_bull_ratio": candidate.get("st_bull_ratio", 0.5),
                    "st_watchers": candidate.get("st_watchers", 0),
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "exit_reason": exit_reason,
                    "high_of_day": round(high_of_day, 2),
                    "low_of_day": round(low_of_day, 2),
                    "pnl_usd": round(pnl_usd_per_share, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "was_traded": 1 if was_traded else 0,
                    "shadow": 0 if was_traded else 1,
                }

                with open(SHADOW_LOG, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=COLUMNS)
                    writer.writerow(row)

                label = "REAL" if was_traded else "SHADOW"
                if was_traded:
                    real_count += 1
                else:
                    shadow_count += 1

                logger.info(
                    f"  [{label}] #{candidate['rank']} {ticker}: "
                    f"entry=${entry_price:.2f} exit=${exit_price:.2f} "
                    f"({exit_reason}) pnl={pnl_pct:+.1f}%"
                )

            except Exception as e:
                logger.warning(f"  Shadow resolve failed for {ticker}: {e}")

            # Rate limit yfinance
            time.sleep(0.5)

        logger.info(
            f"Shadow tracker: resolved {real_count} real + "
            f"{shadow_count} shadow = "
            f"{real_count + shadow_count} total data points"
        )

    def _simulate_trailing_stop(self, data, entry_price, trail_amount):
        """
        Simulate a trailing stop on 5-min bar data.

        Returns (exit_price, exit_reason).
        """
        highest = entry_price
        stop_price = entry_price - trail_amount

        for _, bar in data.iterrows():
            high = float(bar["High"])
            low = float(bar["Low"])

            # Update trailing stop
            if high > highest:
                highest = high
                stop_price = highest - trail_amount

            # Check if stop was hit
            if low <= stop_price:
                return round(stop_price, 2), "trail"

        # Didn't hit stop — force close at last bar's close
        last_close = float(data["Close"].iloc[-1])
        return round(last_close, 2), "eod_close"

    def get_summary(self):
        """Read the shadow log and return summary stats."""
        if not SHADOW_LOG.exists():
            return None

        df = pd.read_csv(SHADOW_LOG)
        if df.empty:
            return None

        shadows = df[df["shadow"] == 1]
        reals = df[df["shadow"] == 0]

        summary = {
            "total_entries": len(df),
            "real_trades": len(reals),
            "shadow_trades": len(shadows),
            "shadow_win_rate": (
                (shadows["pnl_pct"] > 0).mean() * 100
                if len(shadows) > 0 else 0
            ),
            "real_win_rate": (
                (reals["pnl_pct"] > 0).mean() * 100
                if len(reals) > 0 else 0
            ),
            "shadow_avg_pnl": shadows["pnl_pct"].mean() if len(shadows) > 0 else 0,
            "real_avg_pnl": reals["pnl_pct"].mean() if len(reals) > 0 else 0,
            # Performance by rank
            "by_rank": {},
        }

        for rank in sorted(df["rank"].unique()):
            rank_df = df[df["rank"] == rank]
            summary["by_rank"][int(rank)] = {
                "count": len(rank_df),
                "win_rate": (rank_df["pnl_pct"] > 0).mean() * 100,
                "avg_pnl": round(rank_df["pnl_pct"].mean(), 2),
            }

        return summary
