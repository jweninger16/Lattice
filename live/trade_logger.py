"""
live/trade_logger.py
----------------------
Logs every trade with full context for training and analysis.

Captures everything needed to learn which signals predict wins:
  - All scoring inputs (gap, rvol, SI, float, options flow, momentum, trend)
  - Entry conditions (signal type, VWAP, PM high, price at entry)
  - Trade outcome (P&L, duration, high before exit, exit type)
  - Market context (SPY direction, VIX level, sector performance)

Data is appended to a CSV after every trade. After 50+ trades,
run the auto-tuner to retune scoring weights from live results.

Usage:
    from live.trade_logger import TradeLogger
    logger = TradeLogger()
    logger.log_entry(...)   # When entering a trade
    logger.log_exit(...)    # When exiting a trade
    logger.get_training_data()  # Returns DataFrame for analysis
"""

import csv
import json
from pathlib import Path
from datetime import datetime, date
import yfinance as yf
from loguru import logger


LOG_FILE = Path("data/trade_log.csv")

# All fields we capture — this is the training dataset
FIELDS = [
    # ── Trade identification ──────────────────────────────────────
    "date",
    "ticker",
    "trade_number",         # Which trade today (1st, 2nd, 3rd)

    # ── Scoring inputs (what the bot saw when picking this stock) ──
    "gap_pct",              # Gap from previous close
    "rvol",                 # Relative volume vs 20-day avg
    "atr_pct",              # ATR as % of price (volatility)
    "mom_5d",               # 5-day momentum into gap
    "above_sma50",          # 1 = uptrend, 0 = downtrend
    "si_pct",               # Short interest % of float
    "float_shares",         # Total float
    "float_category",       # low/mid/high
    "options_signal",       # VERY UNUSUAL / unusual / active / none
    "call_vol_oi_ratio",    # Call volume vs open interest
    "call_put_ratio",       # Call/put volume ratio
    "score",                # Final composite score
    "candidate_rank",       # Was this #1, #2, #3 pick?
    "num_candidates",       # How many stocks passed filters

    # ── Entry conditions ──────────────────────────────────────────
    "entry_signal",         # vwap_pullback / pm_breakout / timeout
    "entry_price",
    "pm_high",              # Pre-market high level
    "vwap_at_entry",        # VWAP when entered
    "float_rotation_at_entry",  # Float rotation when entered
    "minutes_after_open",   # How many minutes after 9:30 entry was
    "scan_number",          # Which scan found this (1st, 2nd, rescan?)

    # ── Stop/trail parameters ─────────────────────────────────────
    "hard_stop",            # Initial stop price
    "stop_type",            # vwap / gap / atr
    "trail_amount",         # Trailing stop distance in $
    "trail_atr_mult",       # Trail as multiple of ATR

    # ── Exit conditions ───────────────────────────────────────────
    "exit_price",
    "exit_reason",          # trail / stop / eod / target
    "pnl_usd",
    "pnl_pct",
    "trade_duration_min",   # How long the trade lasted
    "high_before_exit",     # Highest price reached during trade
    "max_unrealized_pct",   # Max unrealized gain during trade
    "drawdown_from_high",   # How much it pulled back from high

    # ── Market context ────────────────────────────────────────────
    "spy_change_pct",       # SPY's move on this day
    "vix_level",            # VIX at time of trade
    "sector",               # Stock's sector
    "day_of_week",          # Mon=0, Fri=4

    # ── Result ────────────────────────────────────────────────────
    "win",                  # 1 = win, 0 = loss
    "account_after",        # Account balance after trade
]


class TradeLogger:
    def __init__(self, log_file=None):
        self.log_file = log_file or LOG_FILE
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.current_trade = {}
        self.entry_time = None

        # Create CSV with headers if it doesn't exist
        if not self.log_file.exists():
            with open(self.log_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
            logger.info(f"Trade log created: {self.log_file}")

    def log_entry(self, ticker, entry_price, target_data, position,
                  entry_signal, scan_number=1, candidate_rank=1,
                  num_candidates=1, vwap=None, pm_high=None,
                  float_rotation=0, scans_completed=1):
        """
        Called when a trade is entered. Captures all the context.
        """
        import zoneinfo
        try:
            et = zoneinfo.ZoneInfo("America/New_York")
            now = datetime.now(et)
        except Exception:
            now = datetime.now()

        # Minutes after market open (9:30 ET)
        minutes_after = (now.hour * 60 + now.minute) - (9 * 60 + 30)

        self.entry_time = now
        self.current_trade = {
            "date": str(date.today()),
            "ticker": ticker,
            "trade_number": scan_number,

            # Scoring inputs
            "gap_pct": round(target_data.get("gap_pct", 0), 2),
            "rvol": round(target_data.get("rvol", 0), 2),
            "atr_pct": round(target_data.get("atr_pct", 0), 2),
            "mom_5d": round(target_data.get("mom_5d", 0), 2),
            "above_sma50": int(target_data.get("above_sma50", 0)),
            "si_pct": round(target_data.get("si_pct", 0), 2),
            "float_shares": target_data.get("float_shares", 0),
            "float_category": self._categorize_float(
                target_data.get("float_shares", 0)),
            "options_signal": target_data.get("options_signal", ""),
            "call_vol_oi_ratio": round(
                target_data.get("call_vol_oi_ratio", 0), 2),
            "call_put_ratio": round(
                target_data.get("call_put_ratio", 0), 2),
            "score": round(target_data.get("score", 0), 2),
            "candidate_rank": candidate_rank,
            "num_candidates": num_candidates,

            # Entry conditions
            "entry_signal": entry_signal or "immediate",
            "entry_price": round(entry_price, 2),
            "pm_high": round(pm_high, 2) if pm_high else 0,
            "vwap_at_entry": round(vwap, 2) if vwap else 0,
            "float_rotation_at_entry": round(float_rotation, 2),
            "minutes_after_open": max(0, minutes_after),
            "scan_number": scans_completed,

            # Stop/trail
            "hard_stop": round(position.get("stop", 0), 2),
            "stop_type": position.get("stop_type", "gap"),
            "trail_amount": round(position.get("trail_amount", 0), 2),
            "trail_atr_mult": 0.20,

            # Context
            "day_of_week": now.weekday(),
            "sector": target_data.get("sector", ""),
        }

        logger.info(f"  Trade logged (entry): {ticker} @ ${entry_price:.2f}")

    def log_exit(self, exit_price, exit_reason, pnl_usd, pnl_pct,
                 account_after, high_before_exit=None):
        """
        Called when a trade exits. Completes the record and writes to CSV.
        """
        if not self.current_trade:
            logger.warning("log_exit called with no active trade")
            return

        import zoneinfo
        try:
            et = zoneinfo.ZoneInfo("America/New_York")
            now = datetime.now(et)
        except Exception:
            now = datetime.now()

        # Duration
        duration = 0
        if self.entry_time:
            duration = int((now - self.entry_time).total_seconds() / 60)

        entry = self.current_trade.get("entry_price", exit_price)

        # High before exit and drawdown
        if high_before_exit is None:
            high_before_exit = max(entry, exit_price)
        max_unrealized = (high_before_exit - entry) / entry * 100 \
            if entry > 0 else 0
        drawdown = (high_before_exit - exit_price) / high_before_exit * 100 \
            if high_before_exit > 0 else 0

        # Market context
        spy_change, vix = self._get_market_context()

        self.current_trade.update({
            "exit_price": round(exit_price, 2),
            "exit_reason": exit_reason,
            "pnl_usd": round(pnl_usd, 2),
            "pnl_pct": round(pnl_pct, 2),
            "trade_duration_min": duration,
            "high_before_exit": round(high_before_exit, 2),
            "max_unrealized_pct": round(max_unrealized, 2),
            "drawdown_from_high": round(drawdown, 2),
            "spy_change_pct": round(spy_change, 2),
            "vix_level": round(vix, 2),
            "win": 1 if pnl_pct > 0 else 0,
            "account_after": round(account_after, 2),
        })

        # Write to CSV
        self._write_row(self.current_trade)
        self.current_trade = {}
        self.entry_time = None

        logger.info(f"  Trade logged (exit): {exit_reason}, "
                    f"P&L ${pnl_usd:+.2f}, duration {duration}min")

    def _write_row(self, data):
        """Appends a row to the CSV."""
        with open(self.log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            # Fill missing fields with empty string
            row = {k: data.get(k, "") for k in FIELDS}
            writer.writerow(row)

    def _categorize_float(self, float_shares):
        if not float_shares or float_shares <= 0:
            return "unknown"
        if float_shares < 50_000_000:
            return "low"
        if float_shares < 200_000_000:
            return "mid"
        return "high"

    def _get_market_context(self):
        """Gets SPY change and VIX for market context."""
        spy_change = 0
        vix = 0
        try:
            spy = yf.download("SPY", period="2d", auto_adjust=True,
                              progress=False)
            if spy is not None and len(spy) >= 2:
                spy_change = ((spy["Close"].iloc[-1] /
                               spy["Close"].iloc[-2]) - 1) * 100
        except Exception:
            pass

        try:
            vix_data = yf.download("^VIX", period="1d", auto_adjust=True,
                                   progress=False)
            if vix_data is not None and len(vix_data) >= 1:
                vix = float(vix_data["Close"].iloc[-1])
        except Exception:
            pass

        return spy_change, vix

    def get_training_data(self):
        """Returns the full trade log as a pandas DataFrame."""
        import pandas as pd
        if not self.log_file.exists():
            return pd.DataFrame()
        return pd.read_csv(self.log_file)

    def get_summary(self):
        """Returns a quick summary of logged trades."""
        import pandas as pd
        df = self.get_training_data()
        if df.empty:
            return "No trades logged yet."

        wins = df["win"].sum()
        losses = len(df) - wins
        total = len(df)
        wr = wins / total * 100 if total > 0 else 0
        avg_pnl = df["pnl_usd"].mean()
        total_pnl = df["pnl_usd"].sum()

        lines = [
            f"Trade log: {total} trades ({wins}W/{losses}L, {wr:.0f}%)",
            f"Avg P&L: ${avg_pnl:+.2f}/trade | Total: ${total_pnl:+.2f}",
        ]

        # Best signal breakdown
        if total >= 10:
            for signal in ["vwap_pullback", "pm_breakout", "timeout"]:
                subset = df[df["entry_signal"] == signal]
                if len(subset) >= 3:
                    sw = subset["win"].mean() * 100
                    sa = subset["pnl_usd"].mean()
                    lines.append(f"  {signal}: {len(subset)} trades, "
                                 f"{sw:.0f}% win, ${sa:+.2f}/trade")

        return "\n".join(lines)
