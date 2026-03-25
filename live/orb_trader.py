"""
live/orb_trader.py
-------------------
Opening Range Breakout (ORB) day trading bot.

Connects to IBKR (paper or live) and trades QQQ based on the
15-minute opening range breakout strategy.

Strategy (validated in research):
  - Wait for first 15 minutes (9:30-9:45 AM ET) to form opening range
  - Record OR high and OR low
  - If price breaks above OR high → BUY (long breakout)
  - If price breaks below OR low → SELL SHORT (short breakout)
  - Exit at: 1.5x OR range profit target, 1x OR range stop loss, or 3:55 PM
  - Skip days where gap from prior close > 0.5% (gap filter)

Late ORB (gap day second entry window):
  - When gap > 0.5% triggers the gap filter, instead of sitting out:
  - Wait for 10:30-10:45 AM to form a NEW "late" opening range
  - LONG-ONLY breakout above the late OR high
  - Same 1.5:1 R:R, same stop/target logic
  - Position size reduced to 50% (LATE_POSITION_SCALE) while proving out
  - Research: SPY 14 trades, 57% win, 2.29 PF, +1.80%, -0.29% max DD
  - Toggle: set LATE_ORB_ENABLED = False to disable

Best research results (QQQ, 60 days):
  - 62.5% win rate, 2.53 profit factor, +7.01% total
  - Max 3 consecutive losses, -1.44% max drawdown

Requirements:
  - IBKR TWS running with API enabled
  - Market data subscription for US equities
  - pip install ib_insync

Usage:
    python live/orb_trader.py                    # Paper trading (default)
    python live/orb_trader.py --live             # Live trading (careful!)
    python live/orb_trader.py --size 500         # Custom position size in USD
    python main.py orb                           # Via main entry point
"""

import sys
import asyncio
import time
from datetime import datetime, date, timedelta, time as dtime
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")

# Python 3.14 fix
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

from ib_insync import IB, Stock, MarketOrder, LimitOrder, StopOrder

try:
    from live.discord_format import (
        fmt_entry, fmt_exit, fmt_day_complete,
        fmt_gap_skip, fmt_error,
    )
except ImportError:
    fmt_entry = fmt_exit = fmt_day_complete = fmt_gap_skip = fmt_error = None


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

class ORBConfig:
    # Connection
    PAPER_PORT = 7497        # Paper trading
    LIVE_PORT = 7496         # Live trading
    HOST = "127.0.0.1"
    CLIENT_ID = 20           # Separate from swing trader (client 10)

    # Strategy — Regular ORB
    TICKER = "QQQ"
    OR_MINUTES = 15          # Opening range: first 15 minutes
    TARGET_MULT = 1.5        # Target = 1.5x opening range
    STOP_MULT = 1.0          # Stop = 1.0x opening range
    MAX_GAP_PCT = 0.5        # Skip if gap > 0.5%
    DIRECTION = "both"       # "long", "short", or "both"

    # Strategy — Late ORB (gap day second entry window)
    # Activates when regular ORB is skipped due to gap filter.
    # Research: SPY 10:30 long-only, 15min OR, 1.5:1 R:R
    #   14 trades, 57% win, 2.29 PF, +1.80%, -0.29% max DD
    LATE_ORB_ENABLED = True
    LATE_OR_START = dtime(10, 30)    # Late OR window starts at 10:30 AM
    LATE_OR_END = dtime(10, 45)      # 15-min late OR ends at 10:45 AM
    LATE_OR_MINUTES = 15
    LATE_TARGET_MULT = 1.5
    LATE_STOP_MULT = 1.0
    LATE_DIRECTION = "long"          # Long-only — shorts were 14% win rate
    LATE_LAST_ENTRY = dtime(14, 0)   # No late entries after 2 PM
    LATE_POSITION_SCALE = 0.5        # 50% of normal size while proving out

    # Risk
    POSITION_SIZE_USD = 500  # Default position size
    MAX_DAILY_LOSS_PCT = 1.0 # Stop trading if down 1% for the day
    MAX_TRADES_PER_DAY = 1   # Only one ORB trade per day

    # Timing (Eastern Time)
    MARKET_OPEN = dtime(9, 30)
    OR_END = dtime(9, 45)    # End of 15-min opening range
    LAST_ENTRY = dtime(14, 0)  # No new entries after 2 PM
    FORCE_EXIT = dtime(15, 55)  # Force close at 3:55 PM
    MARKET_CLOSE = dtime(16, 0)

    # Monitoring
    CHECK_INTERVAL = 10      # Check every 10 seconds during active trading


# ═══════════════════════════════════════════════════════════════════════
# ORB Trading Bot
# ═══════════════════════════════════════════════════════════════════════

class ORBTrader:
    def __init__(self, paper=True, position_size=None):
        self.paper = paper
        self.port = ORBConfig.PAPER_PORT if paper else ORBConfig.LIVE_PORT
        self.position_size = position_size or ORBConfig.POSITION_SIZE_USD
        self.ib = None

        # Daily state (resets each day)
        self.or_high = None
        self.or_low = None
        self.or_range = None
        self.prev_close = None
        self.gap_pct = None
        self.trade_taken = False
        self.entered_trade = False   # True only when a real order was placed
        self.gap_skipped = False     # True when gap filter sat us out
        self.daily_pnl = 0
        self.position = None  # {"direction": "long"/"short", "entry": float, "qty": int}
        self.bars_collected = []
        self.today = None

        # Late ORB state
        self.late_orb_mode = False   # True when gap filter fires and late ORB kicks in
        self.late_or_high = None
        self.late_or_low = None
        self.late_or_range = None
        self.late_or_computed = False

    def connect(self):
        """Connect to IBKR."""
        self.ib = IB()
        mode = "PAPER" if self.paper else "LIVE"
        try:
            self.ib.connect(ORBConfig.HOST, self.port, clientId=ORBConfig.CLIENT_ID)
            logger.info(f"Connected to IBKR ({mode}) on port {self.port}")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from IBKR."""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IBKR")

    def get_contract(self):
        """Returns qualified QQQ contract."""
        contract = Stock(ORBConfig.TICKER, "SMART", "USD")
        self.ib.qualifyContracts(contract)
        return contract

    def get_current_time_et(self):
        """Returns current Eastern Time."""
        try:
            import zoneinfo
            et = zoneinfo.ZoneInfo("America/New_York")
            return datetime.now(et)
        except ImportError:
            # Fallback: assume CST = ET - 1 hour (close enough for Arkansas)
            return datetime.now() + timedelta(hours=1)

    def is_market_day(self):
        """Check if today is a trading day."""
        now = self.get_current_time_et()
        return now.weekday() < 5  # Mon-Fri

    def get_previous_close(self, contract):
        """Gets previous day's closing price."""
        try:
            bars = self.ib.reqHistoricalData(
                contract, endDateTime="", durationStr="2 D",
                barSizeSetting="1 day", whatToShow="TRADES", useRTH=True
            )
            if bars and len(bars) >= 2:
                return bars[-2].close  # Previous day's close
            elif bars and len(bars) >= 1:
                return bars[-1].close
        except Exception as e:
            logger.warning(f"Failed to get previous close: {e}")
        return None

    def get_5min_bars(self, contract, duration="1800 S"):
        """Gets recent 5-minute bars."""
        try:
            bars = self.ib.reqHistoricalData(
                contract, endDateTime="", durationStr=duration,
                barSizeSetting="5 mins", whatToShow="TRADES", useRTH=True
            )
            return bars
        except Exception as e:
            logger.warning(f"Failed to get 5min bars: {e}")
            return []

    def get_current_price(self, contract):
        """Gets current price via snapshot."""
        try:
            self.ib.reqMktData(contract, "", False, False)
            self.ib.sleep(2)
            ticker = self.ib.ticker(contract)
            price = ticker.last
            if price != price:  # NaN check
                price = ticker.close
            if price != price:
                price = (ticker.bid + ticker.ask) / 2 if ticker.bid == ticker.bid else None
            self.ib.cancelMktData(contract)
            return price
        except Exception as e:
            logger.warning(f"Price fetch failed: {e}")
            return None

    # ── Core Strategy Logic ───────────────────────────────────────────

    def reset_daily_state(self):
        """Resets all daily tracking variables."""
        self.or_high = None
        self.or_low = None
        self.or_range = None
        self.prev_close = None
        self.gap_pct = None
        self.trade_taken = False
        self.entered_trade = False
        self.gap_skipped = False
        self.daily_pnl = 0
        self.position = None
        self.bars_collected = []
        self.today = date.today()
        # Late ORB
        self.late_orb_mode = False
        self.late_or_high = None
        self.late_or_low = None
        self.late_or_range = None
        self.late_or_computed = False
        logger.info("Daily state reset")

    def compute_opening_range(self, contract):
        """
        Computes the opening range from the first 15 minutes of trading.
        Uses 5-minute bars: first 3 bars = 15 minutes.
        """
        bars = self.get_5min_bars(contract, duration="1800 S")
        if not bars:
            logger.warning("No bars available for opening range")
            return False

        # Filter to today's bars only
        today_bars = []
        for bar in bars:
            bar_date = bar.date
            if hasattr(bar_date, 'date'):
                bar_date = bar_date.date()
            if bar_date == date.today():
                today_bars.append(bar)

        if len(today_bars) < 3:
            logger.info(f"Only {len(today_bars)} bars so far, need 3 for 15-min OR")
            return False

        # First 3 five-minute bars = 15-minute opening range
        or_bars = today_bars[:3]
        self.or_high = max(b.high for b in or_bars)
        self.or_low = min(b.low for b in or_bars)
        self.or_range = self.or_high - self.or_low

        if self.or_range <= 0:
            logger.warning("Opening range is zero — skipping today")
            return False

        logger.info(f"Opening Range: high=${self.or_high:.2f} low=${self.or_low:.2f} "
                    f"range=${self.or_range:.2f} ({self.or_range/self.or_high*100:.2f}%)")
        return True

    def check_gap_filter(self, contract):
        """Checks if today's gap is within acceptable range."""
        self.prev_close = self.get_previous_close(contract)
        if self.prev_close is None:
            logger.warning("Can't determine previous close — allowing trade")
            return True

        # Get today's open from first bar
        bars = self.get_5min_bars(contract, duration="1800 S")
        today_open = None
        for bar in bars:
            bar_date = bar.date
            if hasattr(bar_date, 'date'):
                bar_date = bar_date.date()
            if bar_date == date.today():
                today_open = bar.open
                break

        if today_open is None:
            return True

        self.gap_pct = abs(today_open / self.prev_close - 1) * 100

        if self.gap_pct > ORBConfig.MAX_GAP_PCT:
            logger.info(f"Gap filter triggered: {self.gap_pct:.2f}% > {ORBConfig.MAX_GAP_PCT}% — skipping today")
            return False

        logger.info(f"Gap: {self.gap_pct:.2f}% (within {ORBConfig.MAX_GAP_PCT}% limit)")
        return True

    # ── Late ORB Logic (gap day second window) ────────────────────────

    def compute_late_opening_range(self, contract):
        """
        Computes the late opening range from 10:30-10:45 AM.
        Uses 5-minute bars: 3 bars starting at the 10:30 bar.
        Called only on gap days when LATE_ORB_ENABLED is True.
        """
        bars = self.get_5min_bars(contract, duration="7200 S")  # 2 hours of bars
        if not bars:
            logger.warning("No bars available for late opening range")
            return False

        # Filter to today's bars in the late OR window
        late_bars = []
        for bar in bars:
            bar_date = bar.date
            if hasattr(bar_date, 'date'):
                bar_date = bar_date.date()
            bar_time = bar.date.time() if hasattr(bar.date, 'time') else None
            if bar_time is None:
                continue
            if (bar_date == date.today() and
                bar_time >= ORBConfig.LATE_OR_START and
                bar_time < ORBConfig.LATE_OR_END):
                late_bars.append(bar)

        n_needed = ORBConfig.LATE_OR_MINUTES // 5  # 3 bars for 15 min
        if len(late_bars) < n_needed:
            logger.info(f"Late OR: only {len(late_bars)} bars, need {n_needed}")
            return False

        late_bars = late_bars[:n_needed]
        self.late_or_high = max(b.high for b in late_bars)
        self.late_or_low = min(b.low for b in late_bars)
        self.late_or_range = self.late_or_high - self.late_or_low

        if self.late_or_range <= 0:
            logger.warning("Late OR range is zero — skipping late ORB")
            return False

        self.late_or_computed = True
        range_pct = self.late_or_range / self.late_or_high * 100

        logger.info(f"Late Opening Range (10:30-10:45): "
                    f"high=${self.late_or_high:.2f} low=${self.late_or_low:.2f} "
                    f"range=${self.late_or_range:.2f} ({range_pct:.2f}%)")
        return True

    def check_late_breakout(self, contract):
        """
        Checks for long-only breakout above the late opening range.
        Returns (direction, price) or (None, None).
        """
        price = self.get_current_price(contract)
        if price is None:
            return None, None

        direction = None
        if ORBConfig.LATE_DIRECTION in ("long", "both") and price > self.late_or_high:
            direction = "long"
        elif ORBConfig.LATE_DIRECTION in ("short", "both") and price < self.late_or_low:
            direction = "short"

        return direction, price

    def enter_late_trade(self, contract, direction, price):
        """
        Places a bracket order for the late ORB breakout.
        Uses reduced position size (LATE_POSITION_SCALE).
        """
        late_size = self.position_size * ORBConfig.LATE_POSITION_SCALE
        qty = max(1, int(late_size / price))

        entry_price = self.late_or_high  # Long-only for now
        target_price = round(entry_price + self.late_or_range * ORBConfig.LATE_TARGET_MULT, 2)
        stop_price = round(entry_price - self.late_or_range * ORBConfig.LATE_STOP_MULT, 2)

        if direction == "long":
            # Market buy
            parent = MarketOrder("BUY", qty)
            parent.tif = "DAY"
            parent.transmit = False
            parent_trade = self.ib.placeOrder(contract, parent)
            self.ib.sleep(2)

            # Verify parent order was accepted
            if parent_trade.orderStatus.status in ("Cancelled", "Inactive"):
                logger.error(f"Late ORB entry REJECTED: {parent_trade.orderStatus.status}")
                try:
                    from live.alerts import send_discord
                    send_discord(f"LATE ORB ORDER REJECTED: {parent_trade.orderStatus.status}")
                except Exception:
                    pass
                return False

            # Stop loss
            stop = StopOrder("SELL", qty, stop_price)
            stop.tif = "DAY"
            stop.parentId = parent_trade.order.orderId
            stop.transmit = False
            self.ib.placeOrder(contract, stop)

            # Profit target
            target = LimitOrder("SELL", qty, target_price)
            target.tif = "DAY"
            target.parentId = parent_trade.order.orderId
            target.transmit = True
            self.ib.placeOrder(contract, target)
        else:
            # Short (shouldn't happen with current config, but handle it)
            entry_price = self.late_or_low
            target_price = round(entry_price - self.late_or_range * ORBConfig.LATE_TARGET_MULT, 2)
            stop_price = round(entry_price + self.late_or_range * ORBConfig.LATE_STOP_MULT, 2)

            parent = MarketOrder("SELL", qty)
            parent.tif = "DAY"
            parent.transmit = False
            parent_trade = self.ib.placeOrder(contract, parent)
            self.ib.sleep(2)

            if parent_trade.orderStatus.status in ("Cancelled", "Inactive"):
                logger.error(f"Late ORB entry REJECTED: {parent_trade.orderStatus.status}")
                return False

            stop = StopOrder("BUY", qty, stop_price)
            stop.tif = "DAY"
            stop.parentId = parent_trade.order.orderId
            stop.transmit = False
            self.ib.placeOrder(contract, stop)

            target = LimitOrder("BUY", qty, target_price)
            target.tif = "DAY"
            target.parentId = parent_trade.order.orderId
            target.transmit = True
            self.ib.placeOrder(contract, target)

        self.ib.sleep(2)

        self.position = {
            "direction": direction,
            "entry": entry_price,
            "qty": qty,
            "stop": stop_price,
            "target": target_price,
            "order_id": parent_trade.order.orderId,
        }
        self.trade_taken = True
        self.entered_trade = True

        logger.info(f"LATE ORB {direction.upper()} ENTRY: "
                    f"{qty} {ORBConfig.TICKER} @ ~${price:.2f} "
                    f"(${late_size:.0f} = {ORBConfig.LATE_POSITION_SCALE:.0%} size) | "
                    f"stop=${stop_price:.2f} | target=${target_price:.2f}")

        # Discord alert
        try:
            from live.alerts import send_discord
            if fmt_entry:
                msg = fmt_entry(
                    "Late ORB", ORBConfig.TICKER, direction, qty,
                    price, stop_price, target_price,
                    position_size=late_size,
                    extra={
                        "Gap": f"{self.gap_pct:.2f}%",
                        "Late OR": f"${self.late_or_low:.2f} - ${self.late_or_high:.2f}",
                        "Size": f"{ORBConfig.LATE_POSITION_SCALE:.0%} of normal",
                    },
                )
            else:
                msg = (f"LATE ORB {direction.upper()}: {qty} {ORBConfig.TICKER} "
                       f"@ ${price:.2f}")
            send_discord(msg)
        except Exception:
            pass

        return True

    def check_breakout(self, contract):
        """Checks if price has broken out of the opening range."""
        price = self.get_current_price(contract)
        if price is None:
            return None, None

        direction = None
        if ORBConfig.DIRECTION in ("long", "both") and price > self.or_high:
            direction = "long"
        elif ORBConfig.DIRECTION in ("short", "both") and price < self.or_low:
            direction = "short"

        return direction, price

    def enter_trade(self, contract, direction, price):
        """Places a bracket order for the breakout."""
        qty = max(1, int(self.position_size / price))

        if direction == "long":
            entry_price = self.or_high
            target_price = round(entry_price + self.or_range * ORBConfig.TARGET_MULT, 2)
            stop_price = round(entry_price - self.or_range * ORBConfig.STOP_MULT, 2)

            # Market buy
            parent = MarketOrder("BUY", qty)
            parent.tif = "DAY"
            parent.transmit = False
            parent_trade = self.ib.placeOrder(contract, parent)
            self.ib.sleep(2)

            # Verify parent order was accepted
            if parent_trade.orderStatus.status in ("Cancelled", "Inactive"):
                logger.error(f"Entry order REJECTED: {parent_trade.orderStatus.status} — "
                             f"{parent_trade.log[-1].message if parent_trade.log else 'unknown'}")
                try:
                    from live.alerts import send_discord
                    send_discord(f"ORB ORDER REJECTED: {parent_trade.orderStatus.status} — "
                                 f"check TWS presets and permissions")
                except Exception:
                    pass
                return False

            # Stop loss
            stop = StopOrder("SELL", qty, stop_price)
            stop.tif = "DAY"
            stop.parentId = parent_trade.order.orderId
            stop.transmit = False
            self.ib.placeOrder(contract, stop)

            # Profit target
            target = LimitOrder("SELL", qty, target_price)
            target.tif = "DAY"
            target.parentId = parent_trade.order.orderId
            target.transmit = True
            self.ib.placeOrder(contract, target)

        else:  # short
            entry_price = self.or_low
            target_price = round(entry_price - self.or_range * ORBConfig.TARGET_MULT, 2)
            stop_price = round(entry_price + self.or_range * ORBConfig.STOP_MULT, 2)

            # Market sell (short)
            parent = MarketOrder("SELL", qty)
            parent.tif = "DAY"
            parent.transmit = False
            parent_trade = self.ib.placeOrder(contract, parent)
            self.ib.sleep(2)

            # Verify parent order was accepted
            if parent_trade.orderStatus.status in ("Cancelled", "Inactive"):
                logger.error(f"Entry order REJECTED: {parent_trade.orderStatus.status} — "
                             f"{parent_trade.log[-1].message if parent_trade.log else 'unknown'}")
                try:
                    from live.alerts import send_discord
                    send_discord(f"ORB ORDER REJECTED: {parent_trade.orderStatus.status} — "
                                 f"check TWS presets and permissions")
                except Exception:
                    pass
                return False

            # Stop loss (buy to cover)
            stop = StopOrder("BUY", qty, stop_price)
            stop.tif = "DAY"
            stop.parentId = parent_trade.order.orderId
            stop.transmit = False
            self.ib.placeOrder(contract, stop)

            # Profit target (buy to cover)
            target = LimitOrder("BUY", qty, target_price)
            target.tif = "DAY"
            target.parentId = parent_trade.order.orderId
            target.transmit = True
            self.ib.placeOrder(contract, target)

        self.ib.sleep(2)

        self.position = {
            "direction": direction,
            "entry": entry_price,
            "qty": qty,
            "stop": stop_price,
            "target": target_price,
            "order_id": parent_trade.order.orderId,
        }
        self.trade_taken = True
        self.entered_trade = True

        logger.info(f"{'LONG' if direction == 'long' else 'SHORT'} ENTRY: "
                    f"{qty} {ORBConfig.TICKER} @ ~${price:.2f} | "
                    f"stop=${stop_price:.2f} | target=${target_price:.2f}")

        # Discord alert
        try:
            from live.alerts import send_discord
            if fmt_entry:
                msg = fmt_entry(
                    "ORB", ORBConfig.TICKER, direction, qty,
                    price, stop_price, target_price,
                    position_size=self.position_size,
                    extra={
                        "OR range": f"${self.or_low:.2f} - ${self.or_high:.2f}",
                    },
                )
            else:
                msg = (f"ORB {direction.upper()}: {qty} {ORBConfig.TICKER} "
                       f"@ ${price:.2f}")
            send_discord(msg)
        except Exception:
            pass

        return True

    def force_close_position(self, contract):
        """Force closes any open position (end of day)."""
        if self.position is None:
            return

        qty = self.position["qty"]
        direction = self.position["direction"]

        # Cancel any open orders first
        open_orders = self.ib.openOrders()
        for order in open_orders:
            try:
                self.ib.cancelOrder(order)
            except Exception:
                pass
        self.ib.sleep(1)

        # Close position
        if direction == "long":
            order = MarketOrder("SELL", qty)
        else:
            order = MarketOrder("BUY", qty)
        order.tif = "DAY"

        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(2)

        price = self.get_current_price(contract)
        if price and self.position["entry"]:
            if direction == "long":
                pnl = (price - self.position["entry"]) / self.position["entry"] * 100
            else:
                pnl = (self.position["entry"] - price) / self.position["entry"] * 100
            self.daily_pnl += pnl
            logger.info(f"EOD CLOSE: {qty} {ORBConfig.TICKER} @ ${price:.2f} | P&L: {pnl:+.2f}%")

            try:
                from live.alerts import send_discord
                if fmt_exit:
                    msg = fmt_exit(
                        "ORB", ORBConfig.TICKER, direction, qty,
                        self.position["entry"], price, "eod",
                    )
                else:
                    msg = (f"ORB EOD CLOSE: {ORBConfig.TICKER} "
                           f"P&L: {pnl:+.2f}%")
                send_discord(msg)
            except Exception:
                pass

        self.position = None

    def check_position_status(self):
        """Checks if bracket order has been filled (stop or target hit)."""
        if self.position is None:
            return

        # Check IBKR positions
        positions = self.ib.positions()
        has_position = False
        for pos in positions:
            if pos.contract.symbol == ORBConfig.TICKER and pos.position != 0:
                has_position = True
                break

        if has_position:
            # Position confirmed open — mark it so we know the entry filled
            self.position["confirmed"] = True
            return

        if not has_position and self.position.get("confirmed"):
            # Entry was confirmed earlier, now position is gone = bracket filled
            entry = self.position["entry"]
            stop = self.position["stop"]
            target = self.position["target"]
            qty = self.position["qty"]
            direction = self.position["direction"]

            # Get current price to determine which side filled
            price = self.get_current_price(self.get_contract())
            if price is None:
                price = entry

            # Determine if stop or target was closer to current price
            dist_to_stop = abs(price - stop)
            dist_to_target = abs(price - target)

            if dist_to_stop < dist_to_target:
                exit_price = stop
                reason = "stop"
            else:
                exit_price = target
                reason = "target"

            if direction == "long":
                pnl = (exit_price - entry) / entry * 100
            else:
                pnl = (entry - exit_price) / entry * 100
            self.daily_pnl += pnl

            logger.info(f"Bracket filled ({reason}): {pnl:+.2f}%")

            try:
                from live.alerts import send_discord
                if fmt_exit:
                    msg = fmt_exit(
                        "ORB", ORBConfig.TICKER, direction, qty,
                        entry, exit_price, reason,
                    )
                else:
                    msg = (f"ORB {reason.upper()}: {ORBConfig.TICKER} "
                           f"{pnl:+.2f}%")
                send_discord(msg)
            except Exception:
                pass

            self.position = None

        elif not has_position and not self.position.get("confirmed"):
            # Entry was never confirmed — orders may have been rejected
            # Check if orders are still pending or were cancelled
            open_orders = self.ib.openOrders()
            has_pending = any(
                o.orderId == self.position.get("order_id")
                for o in open_orders
            )
            if not has_pending:
                logger.error("Entry order appears REJECTED — no position found, "
                             "no pending orders. Trade did not execute.")
                try:
                    from live.alerts import send_discord
                    if fmt_error:
                        msg = fmt_error(
                            "ORB",
                            f"Entry FAILED for {ORBConfig.TICKER}\n"
                            f"Orders were rejected or cancelled.\n"
                            f"No position opened. Check TWS."
                        )
                    else:
                        msg = (f"ORB ENTRY FAILED: {ORBConfig.TICKER} "
                               f"— orders rejected")
                    send_discord(msg)
                except Exception:
                    pass
                self.position = None
                self.entered_trade = False  # Correct the flag

    # ── Main Run Loop ─────────────────────────────────────────────────

    def run(self):
        """Main trading loop. Runs for one full trading day."""
        mode = "PAPER" if self.paper else "LIVE"
        logger.info(f"=" * 55)
        logger.info(f"  ORB Day Trader — {mode} MODE")
        logger.info(f"  Ticker: {ORBConfig.TICKER}")
        logger.info(f"  Position size: ${self.position_size:,.0f}")
        logger.info(f"  Strategy: {ORBConfig.OR_MINUTES}min OR, "
                    f"{ORBConfig.TARGET_MULT}:{ORBConfig.STOP_MULT} R:R, "
                    f"gap<{ORBConfig.MAX_GAP_PCT}%")
        if ORBConfig.LATE_ORB_ENABLED:
            late_size = self.position_size * ORBConfig.LATE_POSITION_SCALE
            logger.info(f"  Late ORB: ON — 10:30 {ORBConfig.LATE_DIRECTION}-only, "
                        f"{ORBConfig.LATE_OR_MINUTES}min OR, "
                        f"{ORBConfig.LATE_TARGET_MULT}:{ORBConfig.LATE_STOP_MULT} R:R, "
                        f"${late_size:.0f} ({ORBConfig.LATE_POSITION_SCALE:.0%} size)")
        else:
            logger.info(f"  Late ORB: OFF")
        logger.info(f"=" * 55)

        if not self.connect():
            return

        try:
            contract = self.get_contract()
            self.reset_daily_state()

            if not self.is_market_day():
                logger.info("Not a trading day (weekend). Exiting.")
                return

            # Main loop
            while True:
                now = self.get_current_time_et()
                current_time = now.time()

                # Before market open — wait
                if current_time < ORBConfig.MARKET_OPEN:
                    wait_mins = (datetime.combine(date.today(), ORBConfig.MARKET_OPEN) -
                                 datetime.combine(date.today(), current_time)).seconds // 60
                    logger.info(f"Market opens in {wait_mins} minutes. Waiting...")
                    self.ib.sleep(min(wait_mins * 60, 60))
                    continue

                # During opening range formation (9:30 - 9:45)
                if current_time < ORBConfig.OR_END:
                    logger.info(f"Opening range forming... ({current_time.strftime('%H:%M')})")
                    self.ib.sleep(30)  # Check every 30 seconds
                    continue

                # Opening range just ended — compute it (once)
                if self.or_high is None and not self.gap_skipped:
                    logger.info("Computing opening range...")

                    # Check gap filter first
                    if not self.check_gap_filter(contract):
                        self.gap_skipped = True
                        if ORBConfig.LATE_ORB_ENABLED:
                            self.late_orb_mode = True
                            logger.info(f"Gap too large ({self.gap_pct:.2f}%) — "
                                        f"LATE ORB mode activated. "
                                        f"Waiting for 10:30 range.")
                            try:
                                from live.alerts import send_discord
                                if fmt_gap_skip:
                                    msg = fmt_gap_skip(
                                        "ORB", ORBConfig.TICKER,
                                        self.gap_pct, late_orb=True,
                                    )
                                else:
                                    msg = (f"Gap {self.gap_pct:.2f}% — "
                                           f"late ORB at 10:30")
                                send_discord(msg)
                            except Exception:
                                pass
                        else:
                            logger.info("Gap too large — no trades today. "
                                        "Late ORB disabled.")
                            self.trade_taken = True

                    elif not self.compute_opening_range(contract):
                        logger.warning("Failed to compute OR — retrying in 30s")
                        self.ib.sleep(30)
                        continue

                # After market close
                if current_time >= ORBConfig.MARKET_CLOSE:
                    logger.info("Market closed. Daily summary:")
                    if self.gap_skipped and self.late_orb_mode:
                        if self.entered_trade:
                            logger.info("  Mode: Late ORB (gap day)")
                        else:
                            logger.info("  Mode: Late ORB attempted (no breakout)")
                    elif self.gap_skipped:
                        logger.info("  Skipped: gap filter (late ORB disabled)")
                    logger.info(f"  Trades taken: {'Yes' if self.entered_trade else 'No'}")
                    logger.info(f"  Daily P&L: {self.daily_pnl:+.2f}%")
                    break

                # Force close at 3:55 PM
                if current_time >= ORBConfig.FORCE_EXIT and self.position is not None:
                    logger.info("End of day — force closing position")
                    self.force_close_position(contract)
                    continue

                # No new entries after 2 PM (applies to both regular and late ORB)
                if current_time >= ORBConfig.LAST_ENTRY and not self.trade_taken:
                    if self.late_orb_mode:
                        logger.info("Past 2 PM — late ORB window expired, no trades today")
                    else:
                        logger.info("Past 2 PM with no breakout — no trades today")
                    self.trade_taken = True
                    self.ib.sleep(60)
                    continue

                # Check for regular breakout
                if not self.trade_taken and self.or_high is not None:
                    direction, price = self.check_breakout(contract)
                    if direction:
                        logger.info(f"BREAKOUT DETECTED: {direction.upper()} @ ${price:.2f}")
                        self.enter_trade(contract, direction, price)
                    else:
                        logger.debug(f"No breakout yet. Price: ${price:.2f} "
                                     f"(OR: ${self.or_low:.2f}-${self.or_high:.2f})")

                # ── Late ORB state machine ────────────────────────────
                if self.late_orb_mode and not self.trade_taken:

                    # Waiting for late OR window to start
                    if current_time < ORBConfig.LATE_OR_START:
                        logger.debug(f"Late ORB: waiting for 10:30 "
                                     f"({current_time.strftime('%H:%M')})")

                    # Late OR forming (10:30 - 10:45)
                    elif current_time < ORBConfig.LATE_OR_END:
                        logger.info(f"Late OR forming... "
                                    f"({current_time.strftime('%H:%M')})")

                    # Late OR just ended — compute it
                    elif not self.late_or_computed:
                        logger.info("Computing late opening range (10:30-10:45)...")
                        if not self.compute_late_opening_range(contract):
                            logger.warning("Failed to compute late OR — "
                                           "retrying in 30s")
                            self.ib.sleep(30)
                            continue

                    # Check for late breakout (long-only)
                    elif self.late_or_computed:
                        direction, price = self.check_late_breakout(contract)
                        if direction:
                            logger.info(f"LATE ORB BREAKOUT: "
                                        f"{direction.upper()} @ ${price:.2f}")
                            self.enter_late_trade(contract, direction, price)
                        else:
                            logger.debug(
                                f"Late ORB: no breakout. Price: ${price:.2f} "
                                f"(Late OR: ${self.late_or_low:.2f}"
                                f"-${self.late_or_high:.2f})"
                            )

                # Monitor open position
                if self.position is not None:
                    self.check_position_status()

                # Sleep between checks
                self.ib.sleep(ORBConfig.CHECK_INTERVAL)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            if self.position:
                logger.warning("You have an open position! Close it manually in TWS.")
        except Exception as e:
            logger.error(f"Error: {e}")
            if self.position:
                logger.warning("You may have an open position! Check TWS.")
        finally:
            self.disconnect()

            # Daily summary Discord
            try:
                from live.alerts import send_discord
                traded = self.entered_trade
                extra = {}
                if self.gap_pct is not None:
                    extra["Gap"] = f"{self.gap_pct:.2f}%"

                if not traded:
                    if self.gap_skipped and self.late_orb_mode:
                        extra["reason"] = "gap too large, late ORB no breakout"
                    elif self.gap_skipped:
                        extra["reason"] = "gap too large, sat out"
                    else:
                        extra["reason"] = "no breakout within OR range"

                ticker = ORBConfig.TICKER
                bot_name = "Late ORB" if self.late_orb_mode and traded else "ORB"

                if fmt_day_complete:
                    msg = fmt_day_complete(
                        bot_name,
                        traded=traded,
                        ticker=ticker if traded else None,
                        pnl_pct=self.daily_pnl,
                        extra=extra,
                    )
                else:
                    msg = (f"ORB Day complete | "
                           f"P&L: {self.daily_pnl:+.2f}%")
                send_discord(msg)
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def run_orb(args=None):
    """Entry point for ORB trader."""
    import argparse
    parser = argparse.ArgumentParser(description="ORB Day Trader")
    parser.add_argument("--live", action="store_true", help="Use live account (default: paper)")
    parser.add_argument("--size", type=float, default=500, help="Position size in USD")
    if args is not None:
        parsed = parser.parse_args(args)
    else:
        # Called from main.py: sys.argv = ['main.py', 'orb', '--live', ...]
        parsed = parser.parse_args(sys.argv[2:] if len(sys.argv) > 2 else [])

    trader = ORBTrader(paper=not parsed.live, position_size=parsed.size)
    trader.run()


if __name__ == "__main__":
    run_orb(sys.argv[1:])
