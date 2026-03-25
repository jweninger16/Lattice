"""
live/crypto_paper.py
---------------------
Crypto paper trading bot for Lattice.

Runs 24/7, monitors BTC and ETH for momentum breakout signals,
and logs hypothetical trades. No real money — pure data collection
to evaluate whether crypto trading is worth pursuing.

Strategy: Momentum breakout with trailing stop
- Monitors 15-min candles on BTC-USD and ETH-USD
- Entry: price breaks above the 20-bar high on rising volume
- Exit: trailing stop at 1.5x ATR
- Also tracks mean reversion: RSI < 30 buy, RSI > 70 sell

Run: python live/crypto_paper.py
  or via Lattice app "Beta Testing" tab
"""

import csv
import json
import time
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta, date
from collections import deque

import yfinance as yf
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crypto_paper")

# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

class CryptoConfig:
    # Tickers to monitor
    TICKERS = ["BTC-USD", "ETH-USD"]

    # Candle interval for analysis
    INTERVAL = "15m"

    # Lookback for breakout detection
    BREAKOUT_BARS = 20      # 20-bar high = 5 hours on 15m candles

    # RSI settings (used for logging only — NOT for entry)
    RSI_PERIOD = 14
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 80

    # ATR for trailing stop — backtested winner: 4.0x (wider = better for crypto)
    ATR_PERIOD = 14
    TRAIL_ATR_MULT = 4.0

    # Volume confirmation — backtested winner: 1.5x
    MIN_RVOL = 1.5

    # Strategy mode — backtest showed breakout_only is profitable, RSI loses
    STRATEGY = "breakout_only"  # "breakout_only", "rsi_only", or "both"

    # Position sizing (paper)
    STARTING_CAPITAL = 500   # Hypothetical starting capital
    POSITION_PCT = 0.50      # 50% per trade

    # Polling
    CHECK_INTERVAL = 300     # Check every 5 minutes
    MAX_POSITIONS = 2        # One per ticker

    # Data files
    TRADE_LOG = Path("data/crypto_paper_log.csv")
    STATE_FILE = Path("data/crypto_paper_state.json")


LOG_COLUMNS = [
    "date", "time", "ticker", "strategy", "direction",
    "entry_price", "exit_price", "exit_reason",
    "qty", "pnl_usd", "pnl_pct",
    "rsi_at_entry", "rvol_at_entry", "atr_at_entry",
    "hold_minutes", "balance_after",
]


# ═══════════════════════════════════════════════════════════════════════
# Technical Indicators
# ═══════════════════════════════════════════════════════════════════════

def compute_rsi(series, period=14):
    """Compute RSI from a price series."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_atr(high, low, close, period=14):
    """Compute Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


# ═══════════════════════════════════════════════════════════════════════
# Paper Trading Bot
# ═══════════════════════════════════════════════════════════════════════

class CryptoPaperBot:
    """Paper trades crypto 24/7 for data collection."""

    def __init__(self):
        self.state = self._load_state()
        self.positions = self.state.get("positions", {})
        self.balance = self.state.get("balance", CryptoConfig.STARTING_CAPITAL)
        self.total_trades = self.state.get("total_trades", 0)
        self.wins = self.state.get("wins", 0)
        self.losses = self.state.get("losses", 0)
        self.total_pnl = self.state.get("total_pnl", 0)
        self.running = False
        self.log_buffer = deque(maxlen=200)
        self._ensure_csv()

    def _ensure_csv(self):
        CryptoConfig.TRADE_LOG.parent.mkdir(parents=True, exist_ok=True)
        if not CryptoConfig.TRADE_LOG.exists():
            with open(CryptoConfig.TRADE_LOG, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
                writer.writeheader()

    def _load_state(self):
        if CryptoConfig.STATE_FILE.exists():
            try:
                with open(CryptoConfig.STATE_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "balance": CryptoConfig.STARTING_CAPITAL,
            "positions": {},
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl": 0,
            "trade_history": [],
        }

    def _save_state(self):
        CryptoConfig.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.state = {
            "balance": round(self.balance, 2),
            "positions": self.positions,
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "total_pnl": round(self.total_pnl, 2),
            "trade_history": self.state.get("trade_history", [])[-50:],
        }
        with open(CryptoConfig.STATE_FILE, "w") as f:
            json.dump(self.state, f, indent=2)

    def _log(self, msg):
        entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "msg": msg,
        }
        self.log_buffer.append(entry)
        logger.info(msg)

    def _log_trade(self, trade):
        """Write completed trade to CSV."""
        with open(CryptoConfig.TRADE_LOG, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            writer.writerow(trade)

        # Also append to state history
        history = self.state.get("trade_history", [])
        history.append({
            "date": trade["date"],
            "ticker": trade["ticker"],
            "strategy": trade["strategy"],
            "pnl_usd": trade["pnl_usd"],
            "pnl_pct": trade["pnl_pct"],
            "exit_reason": trade["exit_reason"],
            "balance_after": trade["balance_after"],
        })
        self.state["trade_history"] = history[-50:]

    def fetch_data(self, ticker):
        """Fetch recent candles for a ticker."""
        try:
            data = yf.download(
                ticker, period="5d", interval=CryptoConfig.INTERVAL,
                progress=False, auto_adjust=True,
            )
            if data is None or data.empty:
                return None

            # Flatten MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            return data
        except Exception as e:
            self._log(f"Data fetch failed for {ticker}: {e}")
            return None

    def analyze(self, ticker, data):
        """
        Analyze candle data and return signals.

        Returns dict with:
            - breakout_signal: True if price broke 20-bar high on volume
            - rsi_signal: "oversold", "overbought", or None
            - current_price, rsi, atr, rvol
        """
        if data is None or len(data) < CryptoConfig.BREAKOUT_BARS + 5:
            return None

        close = data["Close"]
        high = data["High"]
        low = data["Low"]
        volume = data["Volume"]

        # Current values
        current_price = float(close.iloc[-1])
        current_high = float(high.iloc[-1])

        # RSI
        rsi = compute_rsi(close, CryptoConfig.RSI_PERIOD)
        current_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50

        # ATR
        atr = compute_atr(high, low, close, CryptoConfig.ATR_PERIOD)
        current_atr = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0

        # Breakout: current high > max of previous N bars
        lookback_high = float(high.iloc[-(CryptoConfig.BREAKOUT_BARS+1):-1].max())
        breakout = current_high > lookback_high

        # Relative volume
        avg_vol = float(volume.iloc[-20:].mean()) if len(volume) >= 20 else float(volume.mean())
        current_vol = float(volume.iloc[-1])
        rvol = current_vol / avg_vol if avg_vol > 0 else 1.0

        # Volume confirmation for breakout
        breakout_signal = breakout and rvol >= CryptoConfig.MIN_RVOL

        # RSI signals
        rsi_signal = None
        if current_rsi <= CryptoConfig.RSI_OVERSOLD:
            rsi_signal = "oversold"
        elif current_rsi >= CryptoConfig.RSI_OVERBOUGHT:
            rsi_signal = "overbought"

        return {
            "ticker": ticker,
            "current_price": current_price,
            "rsi": round(current_rsi, 1),
            "atr": round(current_atr, 2),
            "rvol": round(rvol, 2),
            "breakout_signal": breakout_signal,
            "breakout_high": round(lookback_high, 2),
            "rsi_signal": rsi_signal,
        }

    def check_entry(self, analysis):
        """Decide whether to enter a paper trade."""
        ticker = analysis["ticker"]

        # Already have a position in this ticker
        if ticker in self.positions:
            return None

        strategy = CryptoConfig.STRATEGY

        # Check breakout signal
        if strategy in ("breakout_only", "both") and analysis["breakout_signal"]:
            return {
                "strategy": "momentum_breakout",
                "direction": "long",
                "price": analysis["current_price"],
                "rsi": analysis["rsi"],
                "rvol": analysis["rvol"],
                "atr": analysis["atr"],
            }

        # Check RSI oversold (mean reversion buy) — disabled by default
        if strategy in ("rsi_only", "both") and analysis["rsi_signal"] == "oversold":
            return {
                "strategy": "rsi_mean_reversion",
                "direction": "long",
                "price": analysis["current_price"],
                "rsi": analysis["rsi"],
                "rvol": analysis["rvol"],
                "atr": analysis["atr"],
            }

        return None

    def enter_position(self, ticker, signal):
        """Enter a paper position."""
        price = signal["price"]
        position_usd = self.balance * CryptoConfig.POSITION_PCT
        qty = position_usd / price
        trail_amount = signal["atr"] * CryptoConfig.TRAIL_ATR_MULT

        self.positions[ticker] = {
            "entry_price": round(price, 2),
            "entry_time": datetime.now().isoformat(),
            "qty": qty,
            "strategy": signal["strategy"],
            "direction": signal["direction"],
            "highest": price,
            "trail_stop": round(price - trail_amount, 2),
            "trail_amount": round(trail_amount, 2),
            "rsi_at_entry": signal["rsi"],
            "rvol_at_entry": signal["rvol"],
            "atr_at_entry": signal["atr"],
        }

        self._log(
            f"PAPER ENTRY: {ticker} @ ${price:,.2f} | "
            f"strategy={signal['strategy']} | "
            f"qty={qty:.6f} | trail=${trail_amount:,.2f} | "
            f"RSI={signal['rsi']}"
        )
        self._save_state()

    def check_exit(self, ticker, current_price):
        """Check if a position should be exited."""
        if ticker not in self.positions:
            return None

        pos = self.positions[ticker]
        entry = pos["entry_price"]

        # Update trailing stop
        if current_price > pos["highest"]:
            pos["highest"] = current_price
            pos["trail_stop"] = round(
                current_price - pos["trail_amount"], 2
            )

        # Trailing stop hit
        if current_price <= pos["trail_stop"]:
            return "trail"

        # RSI mean reversion: exit when RSI > 70 (overbought)
        if pos["strategy"] == "rsi_mean_reversion":
            # We'll check RSI in the main loop and pass it here
            pass

        # Time-based exit: close after 24 hours
        entry_time = datetime.fromisoformat(pos["entry_time"])
        if datetime.now() - entry_time > timedelta(hours=24):
            return "time_exit"

        return None

    def exit_position(self, ticker, exit_price, reason):
        """Exit a paper position and log it."""
        pos = self.positions[ticker]
        entry = pos["entry_price"]
        qty = pos["qty"]

        pnl_usd = (exit_price - entry) * qty
        pnl_pct = ((exit_price - entry) / entry) * 100

        self.balance += pnl_usd
        self.total_trades += 1
        self.total_pnl += pnl_usd
        if pnl_usd > 0:
            self.wins += 1
        else:
            self.losses += 1

        entry_time = datetime.fromisoformat(pos["entry_time"])
        hold_minutes = int((datetime.now() - entry_time).total_seconds() / 60)

        trade = {
            "date": str(date.today()),
            "time": datetime.now().strftime("%H:%M:%S"),
            "ticker": ticker,
            "strategy": pos["strategy"],
            "direction": pos["direction"],
            "entry_price": round(entry, 2),
            "exit_price": round(exit_price, 2),
            "exit_reason": reason,
            "qty": round(qty, 6),
            "pnl_usd": round(pnl_usd, 2),
            "pnl_pct": round(pnl_pct, 2),
            "rsi_at_entry": pos.get("rsi_at_entry", 0),
            "rvol_at_entry": pos.get("rvol_at_entry", 0),
            "atr_at_entry": pos.get("atr_at_entry", 0),
            "hold_minutes": hold_minutes,
            "balance_after": round(self.balance, 2),
        }
        self._log_trade(trade)

        self._log(
            f"PAPER EXIT: {ticker} @ ${exit_price:,.2f} | "
            f"reason={reason} | pnl={pnl_pct:+.2f}% "
            f"(${pnl_usd:+.2f}) | "
            f"balance=${self.balance:,.2f} | "
            f"record={self.wins}W/{self.losses}L"
        )

        del self.positions[ticker]
        self._save_state()

    def run_cycle(self):
        """Run one analysis cycle across all tickers."""
        for ticker in CryptoConfig.TICKERS:
            try:
                data = self.fetch_data(ticker)
                if data is None:
                    continue

                analysis = self.analyze(ticker, data)
                if analysis is None:
                    continue

                # Check exits first
                if ticker in self.positions:
                    current = analysis["current_price"]
                    reason = self.check_exit(ticker, current)

                    # RSI overbought exit for mean reversion
                    if (self.positions[ticker]["strategy"] == "rsi_mean_reversion"
                            and analysis["rsi"] >= CryptoConfig.RSI_OVERBOUGHT):
                        reason = "rsi_overbought"

                    if reason:
                        self.exit_position(ticker, current, reason)
                        continue

                # Check entries
                if len(self.positions) < CryptoConfig.MAX_POSITIONS:
                    signal = self.check_entry(analysis)
                    if signal:
                        self.enter_position(ticker, signal)

            except Exception as e:
                self._log(f"Error processing {ticker}: {e}")

    def run(self):
        """Main loop — runs until stopped."""
        self.running = True
        self._log("=" * 50)
        self._log("  Crypto Paper Trader - BETA")
        self._log(f"  Tickers: {', '.join(CryptoConfig.TICKERS)}")
        self._log(f"  Balance: ${self.balance:,.2f}")
        self._log(f"  Record: {self.wins}W/{self.losses}L")
        self._log(f"  Strategy: {CryptoConfig.STRATEGY}")
        self._log(f"  Trailing stop: {CryptoConfig.TRAIL_ATR_MULT}x ATR (backtested)")
        self._log(f"  Min volume: {CryptoConfig.MIN_RVOL}x relative")
        self._log(f"  Interval: {CryptoConfig.INTERVAL} candles")
        self._log(f"  Backtest: 40% WR, 1.10 PF, +8.3% over 60d")
        self._log("  THIS IS PAPER TRADING - NO REAL MONEY")
        self._log("=" * 50)

        while self.running:
            try:
                self.run_cycle()

                # Status update
                pos_str = ""
                for t, p in self.positions.items():
                    pos_str += f" | {t} @ ${p['entry_price']:,.2f}"
                if pos_str:
                    self._log(f"Open positions:{pos_str}")

            except Exception as e:
                self._log(f"Cycle error: {e}")

            # Sleep until next check
            time.sleep(CryptoConfig.CHECK_INTERVAL)

    def stop(self):
        self.running = False
        self._log("Crypto paper trader stopped")

    def get_status(self):
        """Return current status for API."""
        return {
            "running": self.running,
            "balance": round(self.balance, 2),
            "starting_capital": CryptoConfig.STARTING_CAPITAL,
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "total_pnl": round(self.total_pnl, 2),
            "positions": self.positions,
            "tickers": CryptoConfig.TICKERS,
            "strategy": "Momentum breakout + RSI mean reversion",
            "interval": CryptoConfig.INTERVAL,
            "logs": list(self.log_buffer)[-20:],
            "trade_history": self.state.get("trade_history", [])[-20:],
        }


# ═══════════════════════════════════════════════════════════════════════
# Singleton for server integration
# ═══════════════════════════════════════════════════════════════════════

_bot_instance = None
_bot_thread = None


def get_bot():
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = CryptoPaperBot()
    return _bot_instance


def start_bot():
    global _bot_thread
    bot = get_bot()
    if bot.running:
        return {"status": "already_running"}
    _bot_thread = threading.Thread(target=bot.run, daemon=True)
    _bot_thread.start()
    return {"status": "started"}


def stop_bot():
    bot = get_bot()
    bot.stop()
    return {"status": "stopped"}


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    bot = CryptoPaperBot()
    try:
        bot.run()
    except KeyboardInterrupt:
        bot.stop()
        print("\nStopped.")
