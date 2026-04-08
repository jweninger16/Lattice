"""
live/rsi2_trader.py
--------------------
RSI(2) Mean Reversion strategy for bear/choppy regimes.

Activated by the ORB trader when VIX >= 20 (unfavorable regime).
Buys SPY when RSI(2) < 10 (extremely oversold), sells when RSI(2) > 70.

Backtest results (2 years, $1,900 position, $5.50 RT):
  VIX > 20:  13 trades, 77% WR, 2.19 PF, +$178 total, +$13.73/trade
  VIX > 25:   6 trades, 83% WR, 3.28 PF, +$156 total, +$26.04/trade
  Below 200 SMA: 5 trades, 80% WR, 154 PF, +$164 total, +$32.88/trade

This module:
  - Checks daily RSI(2) on SPY via yfinance
  - Sends Discord alerts when entry/exit conditions are met
  - Auto-executes buy/sell via IBKR when called with an ib connection
  - Tracks position state in a JSON file (survives restarts)
  - Checks settled cash before buying (cash account safe)

Usage:
    Called by MultiORBTrader when VIX >= 20, OR standalone:
    python live/rsi2_trader.py           # Check signals
    python live/rsi2_trader.py --status  # Show current position
"""

import sys
import json
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, date, timedelta
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

RSI_PERIOD = 2
RSI_ENTRY = 10          # Buy when RSI(2) < 10
RSI_EXIT = 70           # Sell when RSI(2) > 70
MAX_HOLD_DAYS = 10      # Force exit after 10 days
VIX_THRESHOLD = 20      # Only activate when VIX >= 20
TICKER = "SPY"
POSITION_SIZE = 1900.0

STATE_FILE = Path("live/rsi2_state.json")


# ═══════════════════════════════════════════════════════════════════════
# RSI Calculation
# ═══════════════════════════════════════════════════════════════════════

def compute_rsi(series, period=RSI_PERIOD):
    """Compute RSI using exponential moving average."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ═══════════════════════════════════════════════════════════════════════
# State Management (survives restarts)
# ═══════════════════════════════════════════════════════════════════════

def load_state():
    """Load RSI(2) position state from disk."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "in_position": False,
        "entry_date": None,
        "entry_price": None,
        "entry_rsi": None,
        "qty": 0,
        "trade_history": [],
    }


def save_state(state):
    """Save RSI(2) position state to disk."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════
# Market Data
# ═══════════════════════════════════════════════════════════════════════

def get_spy_data():
    """Download recent SPY daily data with RSI(2)."""
    try:
        df = yf.download(TICKER, period="60d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df.columns = [c.lower() for c in df.columns]
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df["rsi2"] = compute_rsi(df["close"], RSI_PERIOD)
        df["sma200"] = df["close"].rolling(200, min_periods=50).mean()
        return df
    except Exception as e:
        logger.error(f"Failed to download SPY data: {e}")
        return None


def get_vix():
    """Get current VIX level."""
    try:
        vix = yf.download("^VIX", period="5d", auto_adjust=True, progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = [c[0] for c in vix.columns]
        vix.columns = [c.lower() for c in vix.columns]
        return float(vix["close"].iloc[-1])
    except Exception as e:
        logger.warning(f"Failed to get VIX: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
# Signal Generation
# ═══════════════════════════════════════════════════════════════════════

def check_signals():
    """
    Check RSI(2) entry/exit conditions.

    Returns dict:
        action: "buy", "sell", "hold", or "wait"
        reason: human-readable reason
        rsi: current RSI(2)
        vix: current VIX
        price: current SPY price
    """
    state = load_state()
    spy = get_spy_data()
    vix = get_vix()

    if spy is None or len(spy) < 10:
        return {"action": "error", "reason": "Could not fetch SPY data"}

    latest = spy.iloc[-1]
    current_rsi = float(latest["rsi2"])
    current_price = float(latest["close"])
    sma200 = float(latest["sma200"]) if not pd.isna(latest["sma200"]) else None

    result = {
        "rsi": round(current_rsi, 2),
        "vix": round(vix, 2) if vix else None,
        "price": round(current_price, 2),
        "sma200": round(sma200, 2) if sma200 else None,
        "in_position": state["in_position"],
    }

    if state["in_position"]:
        # Check exit conditions
        entry_date = datetime.strptime(state["entry_date"], "%Y-%m-%d").date()
        hold_days = (date.today() - entry_date).days
        entry_price = state["entry_price"]
        pnl_pct = (current_price - entry_price) / entry_price * 100

        result["hold_days"] = hold_days
        result["entry_price"] = entry_price
        result["pnl_pct"] = round(pnl_pct, 2)

        if current_rsi > RSI_EXIT:
            result["action"] = "sell"
            result["reason"] = f"RSI(2)={current_rsi:.1f} > {RSI_EXIT} (recovery). " \
                               f"P&L: {pnl_pct:+.2f}% after {hold_days}d"
        elif hold_days >= MAX_HOLD_DAYS:
            result["action"] = "sell"
            result["reason"] = f"Max hold {MAX_HOLD_DAYS}d reached. " \
                               f"P&L: {pnl_pct:+.2f}%, RSI(2)={current_rsi:.1f}"
        else:
            result["action"] = "hold"
            result["reason"] = f"Holding SPY day {hold_days}/{MAX_HOLD_DAYS}. " \
                               f"RSI(2)={current_rsi:.1f}, P&L: {pnl_pct:+.2f}%"
    else:
        # Check entry conditions
        if vix is None:
            result["action"] = "wait"
            result["reason"] = "Cannot read VIX — skipping"
        elif vix < VIX_THRESHOLD:
            result["action"] = "wait"
            result["reason"] = f"VIX {vix:.1f} < {VIX_THRESHOLD} — ORB regime, not RSI(2)"
        elif current_rsi < RSI_ENTRY:
            qty = max(1, int(POSITION_SIZE / current_price))
            result["action"] = "buy"
            result["qty"] = qty
            result["reason"] = f"RSI(2)={current_rsi:.1f} < {RSI_ENTRY} with VIX={vix:.1f}. " \
                               f"BUY {qty} SPY @ ~${current_price:.2f}"
        else:
            result["action"] = "wait"
            result["reason"] = f"RSI(2)={current_rsi:.1f} (need < {RSI_ENTRY}), " \
                               f"VIX={vix:.1f}. Waiting for oversold."

    return result


def record_entry(price, qty):
    """Record a buy entry."""
    state = load_state()
    state["in_position"] = True
    state["entry_date"] = str(date.today())
    state["entry_price"] = price
    state["entry_rsi"] = None
    state["qty"] = qty
    save_state(state)
    logger.info(f"RSI(2) ENTRY recorded: {qty} SPY @ ${price:.2f}")


def record_exit(price):
    """Record a sell exit."""
    state = load_state()
    if not state["in_position"]:
        logger.warning("No RSI(2) position to exit")
        return

    entry_price = state["entry_price"]
    qty = state["qty"]
    pnl_pct = (price - entry_price) / entry_price * 100
    pnl_usd = (price - entry_price) * qty
    entry_date = state["entry_date"]
    hold_days = (date.today() - datetime.strptime(entry_date, "%Y-%m-%d").date()).days

    state["trade_history"].append({
        "entry_date": entry_date,
        "exit_date": str(date.today()),
        "entry_price": entry_price,
        "exit_price": price,
        "qty": qty,
        "pnl_pct": round(pnl_pct, 2),
        "pnl_usd": round(pnl_usd, 2),
        "hold_days": hold_days,
    })
    state["in_position"] = False
    state["entry_date"] = None
    state["entry_price"] = None
    state["qty"] = 0
    save_state(state)
    logger.info(f"RSI(2) EXIT recorded: {qty} SPY @ ${price:.2f} | "
                f"P&L: {pnl_pct:+.2f}% (${pnl_usd:+.2f}) | {hold_days}d hold")


# ═══════════════════════════════════════════════════════════════════════
# Discord Alerts
# ═══════════════════════════════════════════════════════════════════════

def send_rsi2_alert(signal):
    """Send RSI(2) signal to Discord."""
    try:
        from live.alerts import send_discord

        action = signal["action"]
        if action == "buy":
            emoji = "\U0001f7e2"  # green circle
            msg = (f"{emoji} **RSI(2) BEAR REVERSAL — BUY SIGNAL**\n"
                   f"```\n"
                   f"Action:  BUY {signal.get('qty', '?')} SPY\n"
                   f"Price:   ${signal['price']:.2f}\n"
                   f"RSI(2):  {signal['rsi']:.1f} (oversold < {RSI_ENTRY})\n"
                   f"VIX:     {signal['vix']:.1f}\n"
                   f"Target:  Sell when RSI(2) > {RSI_EXIT}\n"
                   f"Max hold: {MAX_HOLD_DAYS} days\n"
                   f"```")
        elif action == "sell":
            emoji = "\U0001f534"  # red circle
            pnl = signal.get('pnl_pct', 0)
            pnl_emoji = "\U0001f4b0" if pnl > 0 else "\U0001f6a8"
            msg = (f"{emoji} **RSI(2) BEAR REVERSAL — SELL SIGNAL** {pnl_emoji}\n"
                   f"```\n"
                   f"Action:  SELL SPY\n"
                   f"Price:   ${signal['price']:.2f}\n"
                   f"RSI(2):  {signal['rsi']:.1f}\n"
                   f"P&L:     {pnl:+.2f}%\n"
                   f"Reason:  {signal['reason']}\n"
                   f"```")
        elif action == "hold":
            msg = (f"\U0001f4ca **RSI(2) Position Update**\n"
                   f"```\n"
                   f"SPY @ ${signal['price']:.2f} | "
                   f"RSI(2): {signal['rsi']:.1f} | "
                   f"P&L: {signal.get('pnl_pct', 0):+.2f}% | "
                   f"Day {signal.get('hold_days', '?')}/{MAX_HOLD_DAYS}\n"
                   f"```")
        else:
            return  # Don't alert on "wait"

        send_discord(msg)
    except Exception as e:
        logger.warning(f"RSI(2) Discord alert failed: {e}")


# ═══════════════════════════════════════════════════════════════════════
# IBKR Auto-Execution
# ═══════════════════════════════════════════════════════════════════════

def execute_rsi2_via_ibkr(ib, signal):
    """
    Auto-execute RSI(2) buy/sell using an existing IBKR connection.

    Args:
        ib: connected ib_insync.IB instance (from ORB trader)
        signal: dict from check_signals()

    Returns:
        True if order placed successfully, False otherwise.
    """
    from ib_insync import Stock, MarketOrder

    action = signal["action"]
    if action not in ("buy", "sell"):
        return False

    # Qualify SPY contract
    try:
        contract = Stock(TICKER, "SMART", "USD")
        ib.qualifyContracts(contract)
    except Exception as e:
        logger.error(f"RSI(2) failed to qualify {TICKER} contract: {e}")
        return False

    state = load_state()

    if action == "buy":
        # Check settled cash first
        try:
            summary = ib.accountSummary()
            settled = None
            for item in summary:
                if item.currency == "USD" and item.tag == "SettledCash":
                    settled = float(item.value)
                    break
            if settled is None:
                for item in summary:
                    if item.currency == "USD" and item.tag == "TotalCashValue":
                        settled = float(item.value)
                        break
        except Exception as e:
            logger.warning(f"RSI(2) could not check settled cash: {e}")
            settled = None

        price = signal["price"]
        qty = max(1, int(POSITION_SIZE / price))
        cost = qty * price

        if settled is not None and settled < price:
            logger.warning(f"RSI(2) SKIP: only ${settled:,.2f} settled cash, "
                           f"need ~${price:.2f} for 1 share of {TICKER}")
            try:
                from live.alerts import send_discord
                send_discord(f"RSI(2) BUY blocked — only ${settled:,.2f} settled cash "
                             f"(need ~${cost:,.2f} for {qty} {TICKER})")
            except Exception:
                pass
            return False

        if settled is not None and cost > settled:
            old_qty = qty
            qty = max(1, int(settled * 0.99 / price))
            logger.info(f"RSI(2) reducing qty {old_qty} -> {qty} "
                        f"to fit settled cash ${settled:,.2f}")

        # Place market buy
        try:
            order = MarketOrder("BUY", qty)
            trade = ib.placeOrder(contract, order)
            ib.sleep(3)  # Wait for fill

            fill_price = price  # default
            if trade.fills:
                fill_price = trade.fills[0].execution.price
            elif trade.orderStatus.avgFillPrice > 0:
                fill_price = trade.orderStatus.avgFillPrice

            record_entry(fill_price, qty)
            logger.info(f"RSI(2) AUTO-BUY executed: {qty} {TICKER} @ ${fill_price:.2f}")

            try:
                from live.alerts import send_discord
                send_discord(f"RSI(2) AUTO-BUY executed: {qty} {TICKER} @ ${fill_price:.2f}\n"
                             f"RSI(2)={signal['rsi']:.1f} | VIX={signal['vix']:.1f}\n"
                             f"Hold target: sell when RSI(2) > {RSI_EXIT}")
            except Exception:
                pass
            return True

        except Exception as e:
            logger.error(f"RSI(2) buy order failed: {e}")
            try:
                from live.alerts import send_discord
                send_discord(f"RSI(2) BUY ORDER FAILED: {e}\n"
                             f"Manual action needed: BUY {qty} {TICKER}")
            except Exception:
                pass
            return False

    elif action == "sell":
        # Get current qty from state
        qty = state.get("qty", 0)
        if qty <= 0:
            logger.warning("RSI(2) sell signal but no position recorded")
            return False

        # Place market sell
        try:
            order = MarketOrder("SELL", qty)
            trade = ib.placeOrder(contract, order)
            ib.sleep(3)

            fill_price = signal["price"]
            if trade.fills:
                fill_price = trade.fills[0].execution.price
            elif trade.orderStatus.avgFillPrice > 0:
                fill_price = trade.orderStatus.avgFillPrice

            record_exit(fill_price)
            logger.info(f"RSI(2) AUTO-SELL executed: {qty} {TICKER} @ ${fill_price:.2f}")

            entry_price = state.get("entry_price", fill_price)
            pnl_pct = (fill_price - entry_price) / entry_price * 100
            try:
                from live.alerts import send_discord
                send_discord(f"RSI(2) AUTO-SELL executed: {qty} {TICKER} @ ${fill_price:.2f}\n"
                             f"P&L: {pnl_pct:+.2f}% | RSI(2)={signal['rsi']:.1f}")
            except Exception:
                pass
            return True

        except Exception as e:
            logger.error(f"RSI(2) sell order failed: {e}")
            try:
                from live.alerts import send_discord
                send_discord(f"RSI(2) SELL ORDER FAILED: {e}\n"
                             f"Manual action needed: SELL {qty} {TICKER}")
            except Exception:
                pass
            return False

    return False


# ═══════════════════════════════════════════════════════════════════════
# Integration with ORB Trader
# ═══════════════════════════════════════════════════════════════════════

def check_and_alert(ib=None):
    """
    Main entry point — called by ORB trader or scheduler.
    Checks signals, sends Discord alert, and auto-executes if ib provided.

    Args:
        ib: optional ib_insync.IB connection for auto-execution.
            If None, sends alerts only (no auto-trade).

    Returns the signal dict (with 'executed' key if auto-traded).
    """
    signal = check_signals()
    action = signal["action"]

    if action in ("buy", "sell", "hold"):
        send_rsi2_alert(signal)

    if action in ("buy", "sell") and ib is not None:
        # Auto-execute via IBKR
        success = execute_rsi2_via_ibkr(ib, signal)
        signal["executed"] = success
        if success:
            logger.info(f"RSI(2) {action.upper()} auto-executed successfully")
        else:
            logger.warning(f"RSI(2) {action.upper()} auto-execution failed — "
                           f"check Discord for manual instructions")
    elif action == "buy":
        logger.info(f"RSI(2) BUY SIGNAL: {signal['reason']}")
    elif action == "sell":
        logger.info(f"RSI(2) SELL SIGNAL: {signal['reason']}")
    elif action == "hold":
        logger.info(f"RSI(2) HOLDING: {signal['reason']}")
    else:
        logger.info(f"RSI(2) WAITING: {signal['reason']}")

    return signal


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--status" in sys.argv:
        state = load_state()
        print(json.dumps(state, indent=2))
    elif "--record-buy" in sys.argv:
        # Manual: python live/rsi2_trader.py --record-buy 550.00 3
        price = float(sys.argv[sys.argv.index("--record-buy") + 1])
        qty = int(sys.argv[sys.argv.index("--record-buy") + 2])
        record_entry(price, qty)
    elif "--record-sell" in sys.argv:
        price = float(sys.argv[sys.argv.index("--record-sell") + 1])
        record_exit(price)
    else:
        signal = check_and_alert()
        print(f"\n  Action: {signal['action'].upper()}")
        print(f"  {signal['reason']}")
        print(f"  SPY: ${signal.get('price', '?')} | RSI(2): {signal.get('rsi', '?')} | "
              f"VIX: {signal.get('vix', '?')}")
