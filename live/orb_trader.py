"""
live/orb_trader.py
-------------------
Opening Range Breakout (ORB) day trading bot.

Default mode: Multi-ticker — scans 85 S&P 500 stocks from ORB universe,
computes opening ranges for all, enters the top 3 volume-confirmed
breakouts per day ranked by vol_ratio (breakout volume / OR avg volume).

Strategy:
  - 9:30-9:32 AM ET: opening range forms (first 2 one-minute bars)
  - Gap filter: skip tickers where gap > 0.5%
  - Volume confirmation: only enter if breakout bar volume > OR avg volume
  - Rank all breakouts by vol_ratio, take top MAX_TRADES_PER_DAY
  - Exit at: 1.5x OR range target, 1.0x OR range stop, or 3:55 PM
  - $1,900 per position, 1 trade/day (best signal by vol_ratio)
  - Sized for $1,944 settled capital (no unsettled fund usage)

Research results ($5.50 RT cost, 85 stocks):
  $1,900 x 1/day: 80% WR, 13.53 PF, +$9.36/day, -$7.88 max DD

Legacy single-ticker mode (QQQ only) available via --single flag.

Requirements:
  - IBKR TWS running with API enabled
  - Market data subscription for US equities
  - pip install ib_insync
  - data/orb_universe.csv (run: python research/orb_volume_stock_backtest.py --universe-only)

Usage:
    python main.py orb                           # Multi-ticker, paper (default)
    python main.py orb --live                    # Multi-ticker, live
    python main.py orb --max-trades 5            # Allow up to 5 trades/day
    python main.py orb --single                  # Legacy single-ticker QQQ mode
    python main.py orb --size 1000               # Custom position size
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

# Persistent log file (survives browser close, Lattice restart, etc.)
_log_dir = Path(__file__).resolve().parent.parent / "logs"
_log_dir.mkdir(exist_ok=True)
logger.add(
    _log_dir / "orb_trader.log",
    rotation="1 day",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
)

# Pre-market intelligence (optional — system works without it)
try:
    from data.market_intel import MarketIntelCollector, PreMarketBrief
except ImportError:
    MarketIntelCollector = None
    PreMarketBrief = None

# RSI(2) bear-regime mean reversion (optional)
try:
    from live.rsi2_trader import check_and_alert as rsi2_check, VIX_THRESHOLD as RSI2_VIX_THRESHOLD
except ImportError:
    rsi2_check = None
    RSI2_VIX_THRESHOLD = 20

# Python 3.14 fix
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

from ib_insync import IB, Stock, MarketOrder, LimitOrder, StopOrder, Order

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
    OR_MINUTES = 2           # Opening range: first 2 minutes (9:30-9:32)
    # Research (1-min bars, 85 stocks, costs): 65.2% WR, 1.99 PF, -5.87% DD
    # Shorter OR = tighter range = more decisive breakouts, fewer EOD exits
    STOP_MULT = 1.0          # Initial stop = 1.0x opening range
    TRAIL_MULT = 0.3         # Trailing stop distance = 0.3x opening range
    # Research: 0.3x trail = 6.69 PF, 67% WR, -1.29% DD (vs 1.64 PF fixed target)
    MAX_GAP_PCT = 1.0        # Skip if gap > 1.0% (backtest: 1.25 PF vs 1.15 at 0.5%)
    DIRECTION = "long"       # "long" only — cash account, no shorting

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

    # Volume confirmation — only enter breakouts where the breakout bar's
    # volume exceeds the average volume during the opening range period.
    # Research: 85 stocks, 1802 trades, 63.8% WR / 2.29 PF with costs.
    REQUIRE_VOLUME_CONFIRMATION = True

    # Risk — full capital on one high-conviction trade
    # $1,900 x 1/day: 80% WR, 13.53 PF, $9.36/day in backtest ($5.50 cost)
    # Commission drag at $1,900: 0.11% (vs 0.21% at $950)
    # One best signal per day via vol_ratio ranking
    POSITION_SIZE_USD = 1900
    MAX_DAILY_LOSS_PCT = 1.0  # Stop trading if down 1% for the day
    MAX_TRADES_PER_DAY = 1

    # Timing (Eastern Time)
    MARKET_OPEN = dtime(9, 30)
    OR_END = dtime(9, 32)    # End of 2-min opening range
    LAST_ENTRY = dtime(14, 0)  # No new entries after 2 PM
    FORCE_EXIT = dtime(15, 55)  # Force close at 3:55 PM
    MARKET_CLOSE = dtime(16, 0)

    # Pre-placed stop-limit entry (eliminates breakout detection latency)
    OCA_ENABLED = True           # False = fallback to poll-based scan entry
    OCA_CANDIDATE_LIMIT = 5      # Rank top N candidates (place 1, keep rest as backups)
    OCA_STOP_OFFSET = 0.01       # Trigger = or_high + $0.01
    OCA_LIMIT_SLIPPAGE = 0.001   # Max fill = or_high * 1.001 (0.1% slippage cap)
    OCA_ROTATION_MINUTES = 15    # Cancel unfilled order and rotate to best candidate after N min

    # Streaming market data (eliminates per-poll API calls)
    STREAMING_ENABLED = True     # False = fallback to snapshot pricing
    FAST_CHECK_INTERVAL = 1.5    # Trail update interval when positions open (seconds)
    SLOW_CHECK_INTERVAL = 5      # Scan interval when no positions (seconds)

    # Monitoring (legacy fallback)
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
        self.or_avg_volume = None    # Average volume during OR period (for confirmation)
        self.prev_close = None
        self.gap_pct = None
        self.trade_taken = False
        self.trades_today = 0        # Track number of trades taken today
        self.entered_trade = False   # True only when a real order was placed
        self._cash_alert_sent = False  # One Discord alert per day for cash issues
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
        self.or_avg_volume = None
        self.prev_close = None
        self.gap_pct = None
        self.trade_taken = False
        self.trades_today = 0
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
        Computes the opening range from the first OR_MINUTES of trading.
        Uses 1-minute bars for precision (critical for 2-min OR).
        """
        try:
            bars = self.ib.reqHistoricalData(
                contract, endDateTime="", durationStr="600 S",
                barSizeSetting="1 min", whatToShow="TRADES", useRTH=True
            )
        except Exception as e:
            logger.warning(f"Failed to get 1-min bars: {e}")
            return False

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

        n_bars = ORBConfig.OR_MINUTES
        if len(today_bars) < n_bars:
            logger.info(f"Only {len(today_bars)} bars so far, need {n_bars} for {ORBConfig.OR_MINUTES}-min OR")
            return False

        or_bars = today_bars[:n_bars]
        self.or_high = max(b.high for b in or_bars)
        self.or_low = min(b.low for b in or_bars)
        self.or_range = self.or_high - self.or_low
        self.or_avg_volume = sum(b.volume for b in or_bars) / len(or_bars)

        if self.or_range <= 0:
            logger.warning("Opening range is zero — skipping today")
            return False

        logger.info(f"Opening Range ({ORBConfig.OR_MINUTES}min): "
                    f"high=${self.or_high:.2f} low=${self.or_low:.2f} "
                    f"range=${self.or_range:.2f} ({self.or_range/self.or_high*100:.2f}%)")
        logger.info(f"OR avg volume: {self.or_avg_volume:,.0f} "
                    f"(volume confirmation: {'ON' if ORBConfig.REQUIRE_VOLUME_CONFIRMATION else 'OFF'})")
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

    def get_settled_cash(self):
        """
        Gets settled cash from IBKR.
        On a cash account, you can only trade with settled funds.
        Returns settled cash amount, or None if unavailable.
        """
        try:
            summary = self.ib.accountSummary()
            settled = None
            total = None
            for item in summary:
                if item.currency != "USD":
                    continue
                if item.tag == "SettledCash":
                    settled = float(item.value)
                elif item.tag == "TotalCashValue":
                    total = float(item.value)
            if settled is not None:
                return settled
            if total is not None:
                return total
            return None
        except Exception as e:
            logger.warning(f"Could not get settled cash: {e}")
            return None

    def _check_cash_and_size(self, size_usd, price, label="ORB"):
        """
        Check settled cash and clamp quantity to what we can afford.
        Returns qty (int) or 0 if we can't afford any shares.
        """
        qty = max(1, int(size_usd / price))
        cost = qty * price

        settled = self.get_settled_cash()
        if settled is None:
            return qty  # Can't check — proceed with calculated qty

        if settled < price:
            # Can't afford even 1 share
            logger.warning(f"Not enough settled cash: ${settled:,.2f} available, "
                           f"need ~${price:,.2f} for 1 share. Skipping trade.")
            if not self._cash_alert_sent:
                try:
                    from live.alerts import send_discord
                    send_discord(f"{label} — Skipping: only ${settled:,.2f} settled cash "
                                 f"(need ~${cost:,.2f}). Cash settles tomorrow.")
                except Exception:
                    pass
                self._cash_alert_sent = True
            return 0

        if cost > settled:
            # Reduce qty to fit available cash (with small buffer)
            old_qty = qty
            qty = max(1, int(settled * 0.99 / price))
            logger.info(f"Reducing qty {old_qty} → {qty} to fit settled cash ${settled:,.2f}")

        return qty

    def enter_late_trade(self, contract, direction, price):
        """
        Places a bracket order for the late ORB breakout.
        Uses reduced position size (LATE_POSITION_SCALE).
        """
        late_size = self.position_size * ORBConfig.LATE_POSITION_SCALE
        qty = self._check_cash_and_size(late_size, price, label="Late ORB")
        if qty == 0:
            return False

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
        self.trades_today += 1
        self.trade_taken = self.trades_today >= ORBConfig.MAX_TRADES_PER_DAY
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
        """Checks if price has broken out of the opening range.
        If REQUIRE_VOLUME_CONFIRMATION is True, only signals a breakout
        when the most recent 5-min bar's volume exceeds the OR average volume.
        """
        price = self.get_current_price(contract)
        if price is None:
            return None, None

        direction = None
        if ORBConfig.DIRECTION in ("long", "both") and price > self.or_high:
            direction = "long"
        elif ORBConfig.DIRECTION in ("short", "both") and price < self.or_low:
            direction = "short"

        # Volume confirmation: reject breakout if current bar volume is too low
        if direction and ORBConfig.REQUIRE_VOLUME_CONFIRMATION and self.or_avg_volume:
            bars = self.get_5min_bars(contract, duration="600 S")  # last ~10 min
            if bars:
                current_bar_volume = bars[-1].volume
                if current_bar_volume < self.or_avg_volume:
                    logger.debug(f"Breakout {direction} rejected: bar vol "
                                 f"{current_bar_volume:,.0f} < OR avg "
                                 f"{self.or_avg_volume:,.0f}")
                    return None, price
                else:
                    vol_ratio = current_bar_volume / self.or_avg_volume
                    logger.info(f"Volume confirmed: {current_bar_volume:,.0f} "
                                f"= {vol_ratio:.1f}x OR avg")

        return direction, price

    def enter_trade(self, contract, direction, price):
        """Places a bracket order for the breakout."""
        qty = self._check_cash_and_size(self.position_size, price, label="ORB")
        if qty == 0:
            return False

        if direction == "long":
            entry_price = self.or_high
            target_price = round(entry_price + self.or_range * ORBConfig.TRAIL_MULT, 2)
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
            target_price = round(entry_price - self.or_range * ORBConfig.TRAIL_MULT, 2)
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
        self.trades_today += 1
        self.trade_taken = self.trades_today >= ORBConfig.MAX_TRADES_PER_DAY
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
                    f"{ORBConfig.TRAIL_MULT}x trail, {ORBConfig.STOP_MULT}x initial stop, "
                    f"gap<{ORBConfig.MAX_GAP_PCT}%")
        logger.info(f"  Volume confirmation: "
                    f"{'ON' if ORBConfig.REQUIRE_VOLUME_CONFIRMATION else 'OFF'}")
        logger.info(f"  Max trades/day: {ORBConfig.MAX_TRADES_PER_DAY}")
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
                    self.ib.sleep(max(min(wait_mins * 60, 60), 10))
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
                    logger.info(f"  Trades taken: {self.trades_today}/{ORBConfig.MAX_TRADES_PER_DAY}")
                    logger.info(f"  Daily P&L: {self.daily_pnl:+.2f}%")
                    break

                # Force close at 3:55 PM
                if current_time >= ORBConfig.FORCE_EXIT and self.position is not None:
                    logger.info("End of day — force closing position")
                    self.force_close_position(contract)
                    continue

                # No new entries after 2 PM (applies to both regular and late ORB)
                has_capacity = self.trades_today < ORBConfig.MAX_TRADES_PER_DAY
                if current_time >= ORBConfig.LAST_ENTRY and has_capacity and not self.trade_taken:
                    if self.late_orb_mode:
                        logger.info("Past 2 PM — late ORB window expired, no trades today")
                    else:
                        logger.info(f"Past 2 PM — {self.trades_today} trades taken today")
                    self.trade_taken = True
                    self.ib.sleep(60)
                    continue

                # Check for regular breakout (if we haven't hit daily trade cap)
                if has_capacity and not self.trade_taken and self.or_high is not None:
                    direction, price = self.check_breakout(contract)
                    if direction:
                        logger.info(f"BREAKOUT DETECTED: {direction.upper()} @ ${price:.2f} "
                                    f"(trade {self.trades_today + 1}/{ORBConfig.MAX_TRADES_PER_DAY})")
                        self.enter_trade(contract, direction, price)
                    else:
                        logger.debug(f"No breakout yet. Price: ${price:.2f} "
                                     f"(OR: ${self.or_low:.2f}-${self.or_high:.2f})")

                # ── Late ORB state machine ────────────────────────────
                if self.late_orb_mode and has_capacity and not self.trade_taken:

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
# Multi-Ticker ORB Bot (S&P 500 universe)
#
# Research results (85 stocks, 60d, $2K positions, 0.02% slip, $1 comm):
#   3/day cap: 177 trades, 59.3% WR, 2.06 PF, $935, -3.70% DD
#   Uncapped:  1802 trades, 63.8% WR, 2.29 PF, $11.6K, -15.8% DD
# ═══════════════════════════════════════════════════════════════════════

class MultiORBTrader:
    """
    Scans the ORB universe for volume-confirmed breakouts.
    Ranks candidates by vol_ratio, enters the top MAX_TRADES_PER_DAY.
    Manages multiple concurrent bracket orders via IBKR.
    """

    def __init__(self, paper=True, position_size=None, max_trades=None):
        self.paper = paper
        self.port = ORBConfig.PAPER_PORT if paper else ORBConfig.LIVE_PORT
        self.position_size = position_size or ORBConfig.POSITION_SIZE_USD
        self.max_trades = max_trades or ORBConfig.MAX_TRADES_PER_DAY
        self.ib = None

        # Per-ticker state: {ticker: {or_high, or_low, or_range, or_avg_volume, ...}}
        self.ticker_state = {}
        # Active positions: {ticker: {direction, entry, qty, stop, target, order_id, confirmed, contract}}
        self.positions = {}
        self.trades_today = 0
        self.daily_pnl = 0
        self.today = None

        # Pre-market intelligence
        self.premarket_brief = None
        self.intel_collector = MarketIntelCollector() if MarketIntelCollector else None

        # Pre-placed stop-limit entry state
        self.oca_mode = False            # True when pre-placed order is active
        self.oca_trades = {}             # {ticker: Trade object from placeOrder}
        self.oca_candidates_ranked = []  # Full ranked list for rotation
        self.oca_placed_time = None      # When current order was placed (for rotation timer)

        # Streaming market data state
        self.streaming_tickers = {}      # {ticker: ib_insync Ticker object (updates in place)}
        self.streaming_contracts = {}    # {ticker: Contract} for cleanup

        # Account tracking (shared with Lattice dashboard)
        self.ACCOUNT_FILE = Path("live/gap_scanner_account.json")
        self.account = self._load_account()

    def _load_account(self):
        """Load or initialize account state for Lattice."""
        import json
        if self.ACCOUNT_FILE.exists():
            try:
                with open(self.ACCOUNT_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "starting_capital": 2000.0,
            "balance": 2000.0,
            "peak_balance": 2000.0,
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl_usd": 0.0,
            "total_pnl_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "trade_history": [],
        }

    def _save_account(self):
        """Save account state so Lattice can read it."""
        import json
        self.ACCOUNT_FILE.parent.mkdir(exist_ok=True)
        with open(self.ACCOUNT_FILE, "w") as f:
            json.dump(self.account, f, indent=2, default=str)

    def _sync_balance(self):
        """Sync real balance from IBKR."""
        if self.ib and self.ib.isConnected():
            try:
                self.ib.sleep(1)
                for item in self.ib.accountSummary():
                    if item.tag == "NetLiquidation" and item.currency == "USD":
                        self.account["balance"] = float(item.value)
                        return
            except Exception:
                pass

    def get_settled_cash(self):
        """
        Gets settled cash from IBKR.
        On a cash account, you can only trade with settled funds.
        """
        try:
            summary = self.ib.accountSummary()
            settled = None
            total = None
            for item in summary:
                if item.currency != "USD":
                    continue
                if item.tag == "SettledCash":
                    settled = float(item.value)
                elif item.tag == "TotalCashValue":
                    total = float(item.value)
            if settled is not None:
                return settled
            if total is not None:
                return total
            return None
        except Exception as e:
            logger.warning(f"Could not get settled cash: {e}")
            return None

    def record_trade(self, ticker, direction, entry, exit_price, qty, reason):
        """Record a completed trade to account JSON (read by Lattice)."""
        if direction == "long":
            pnl_pct = (exit_price - entry) / entry * 100
            pnl_usd = (exit_price - entry) * qty
        else:
            pnl_pct = (entry - exit_price) / entry * 100
            pnl_usd = (entry - exit_price) * qty

        self.account["total_trades"] += 1
        if pnl_pct > 0:
            self.account["wins"] += 1
        else:
            self.account["losses"] += 1

        self._sync_balance()
        if self.account["balance"] == 0:
            self.account["balance"] += pnl_usd

        self.account["total_pnl_usd"] = (
            self.account["balance"] - self.account["starting_capital"])
        self.account["total_pnl_pct"] = (
            (self.account["balance"] / self.account["starting_capital"]) - 1) * 100

        if self.account["balance"] > self.account["peak_balance"]:
            self.account["peak_balance"] = self.account["balance"]
        dd = ((self.account["balance"] - self.account["peak_balance"]) /
              self.account["peak_balance"] * 100)
        if dd < self.account["max_drawdown_pct"]:
            self.account["max_drawdown_pct"] = dd

        self.account["trade_history"].append({
            "date": str(date.today()),
            "ticker": ticker,
            "direction": direction,
            "entry": round(entry, 2),
            "exit": round(exit_price, 2),
            "qty": qty,
            "pnl_pct": round(pnl_pct, 2),
            "pnl_usd": round(pnl_usd, 2),
            "reason": reason,
            "balance_after": round(self.account["balance"], 2),
        })
        if len(self.account["trade_history"]) > 100:
            self.account["trade_history"] = self.account["trade_history"][-100:]

        self._save_account()
        logger.info(f"Trade recorded: {ticker} {reason} {pnl_pct:+.2f}% "
                    f"(${pnl_usd:+.2f}) | Balance: ${self.account['balance']:,.2f}")

    def connect(self):
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
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IBKR")

    def get_current_time_et(self):
        try:
            import zoneinfo
            et = zoneinfo.ZoneInfo("America/New_York")
            return datetime.now(et)
        except ImportError:
            return datetime.now() + timedelta(hours=1)

    def load_universe(self):
        """Load ORB universe tickers from data/orb_universe.csv."""
        import pandas as pd
        universe_path = Path("data/orb_universe.csv")
        if not universe_path.exists():
            logger.error("No ORB universe found. Run: python research/orb_volume_stock_backtest.py --universe-only")
            return []
        df = pd.read_csv(universe_path)
        tickers = df["ticker"].tolist()
        logger.info(f"Loaded {len(tickers)} tickers from ORB universe")
        return tickers

    def qualify_contract(self, ticker):
        """Returns a qualified IBKR contract for a ticker."""
        try:
            contract = Stock(ticker, "SMART", "USD")
            self.ib.qualifyContracts(contract)
            return contract
        except Exception as e:
            logger.warning(f"{ticker}: failed to qualify contract: {e}")
            return None

    def get_5min_bars(self, contract, duration="1800 S"):
        try:
            bars = self.ib.reqHistoricalData(
                contract, endDateTime="", durationStr=duration,
                barSizeSetting="5 mins", whatToShow="TRADES", useRTH=True
            )
            return bars
        except Exception as e:
            logger.debug(f"{contract.symbol}: bar fetch failed: {e}")
            return []

    def get_current_price(self, contract):
        try:
            self.ib.reqMktData(contract, "", False, False)
            self.ib.sleep(1)
            ticker = self.ib.ticker(contract)
            price = ticker.last
            if price != price:
                price = ticker.close
            if price != price:
                price = (ticker.bid + ticker.ask) / 2 if ticker.bid == ticker.bid else None
            self.ib.cancelMktData(contract)
            return price
        except Exception:
            return None

    # ── Opening Range Computation ─────────────────────────────────────

    def get_1min_bars(self, contract, duration="600 S"):
        """Gets recent 1-minute bars."""
        try:
            bars = self.ib.reqHistoricalData(
                contract, endDateTime="", durationStr=duration,
                barSizeSetting="1 min", whatToShow="TRADES", useRTH=True
            )
            return bars
        except Exception as e:
            logger.debug(f"{contract.symbol}: 1-min bar fetch failed: {e}")
            return []

    def compute_all_opening_ranges(self, tickers, contracts):
        """
        After OR_END, compute opening ranges for all tickers using 1-min bars.
        Filters out tickers with gaps > MAX_GAP_PCT.
        Stores results in self.ticker_state.
        """
        logger.info(f"Computing {ORBConfig.OR_MINUTES}-min opening ranges "
                    f"for {len(tickers)} tickers...")
        computed = 0
        skipped_gap = 0
        skipped_data = 0
        n_or_bars = ORBConfig.OR_MINUTES

        for ticker in tickers:
            contract = contracts.get(ticker)
            if contract is None:
                skipped_data += 1
                continue

            bars = self.get_1min_bars(contract, duration="7200 S")
            if not bars:
                skipped_data += 1
                continue

            # Filter to today's bars
            today_bars = [b for b in bars
                          if (b.date.date() if hasattr(b.date, 'date') else b.date) == date.today()]
            if len(today_bars) < n_or_bars:
                skipped_data += 1
                continue

            # Gap filter: compare today's open to yesterday's close
            prev_day_bars = [b for b in bars
                             if (b.date.date() if hasattr(b.date, 'date') else b.date) < date.today()]
            if prev_day_bars:
                prev_close = prev_day_bars[-1].close
                today_open = today_bars[0].open
                gap_pct = abs(today_open / prev_close - 1) * 100
                if gap_pct > ORBConfig.MAX_GAP_PCT:
                    skipped_gap += 1
                    continue

            or_bars = today_bars[:n_or_bars]
            or_high = max(b.high for b in or_bars)
            or_low = min(b.low for b in or_bars)
            or_range = or_high - or_low
            or_avg_volume = sum(b.volume for b in or_bars) / len(or_bars)

            if or_range <= 0:
                skipped_data += 1
                continue

            self.ticker_state[ticker] = {
                "or_high": or_high,
                "or_low": or_low,
                "or_range": or_range,
                "or_avg_volume": or_avg_volume,
                "contract": contract,
                "breakout_detected": False,
            }
            computed += 1

            # Throttle to avoid IBKR pacing violations (50 req/sec limit)
            if computed % 40 == 0:
                self.ib.sleep(2)

        logger.info(f"Opening ranges computed: {computed} tickers "
                    f"(gap-filtered: {skipped_gap}, no data: {skipped_data})")

    # ── OCA Pre-placed Entry ─────────────────────────────────────────

    def rank_oca_candidates(self):
        """
        Rank all tickers with computed ORs for OCA order placement.
        Uses OR volume + pre-market intel to select top N candidates.
        Returns list of dicts: [{ticker, state, qty, score}]
        """
        candidates = []
        for ticker, state in self.ticker_state.items():
            if state["breakout_detected"]:
                continue

            or_avg_vol = state["or_avg_volume"]
            if or_avg_vol <= 0:
                continue

            # Base score: OR average volume (proxy for breakout likelihood)
            vol_score = min(or_avg_vol / 50000, 1.0)  # Normalize

            # Intel score from pre-market brief
            intel_score = 0.0
            skip = False
            if self.premarket_brief and self.premarket_brief.gap_rankings is not None:
                rankings = self.premarket_brief.gap_rankings
                skip_set = self.premarket_brief.skip_tickers or set()
                if ticker in skip_set:
                    continue  # Skip earnings-day tickers
                match = rankings[rankings["ticker"] == ticker]
                if not match.empty:
                    intel_score = float(match.iloc[0]["intel_score"])

            # Composite: 70% volume quality + 30% intel
            composite = 0.7 * vol_score + 0.3 * intel_score

            candidates.append({
                "ticker": ticker,
                "state": state,
                "score": composite,
            })

        candidates.sort(key=lambda c: c["score"], reverse=True)
        top = candidates[:ORBConfig.OCA_CANDIDATE_LIMIT]

        if top:
            logger.info(f"OCA candidates (top {len(top)} of {len(candidates)}):")
            for c in top:
                logger.info(f"  {c['ticker']}: score={c['score']:.3f}, "
                            f"OR=${c['state']['or_high']:.2f}-${c['state']['or_low']:.2f} "
                            f"(range ${c['state']['or_range']:.2f})")
        return top

    def place_oca_orders(self, candidates):
        """
        Place BUY STOP LIMIT order for the #1 ranked candidate.

        Cash accounts: IBKR holds cash for ALL pending orders simultaneously,
        even in an OCA group. With ~$1,900 settled cash and $1,900 position size,
        we can only have ONE pending buy order at a time. The remaining candidates
        are stored as ranked backups — if the primary order doesn't trigger,
        we can rotate to the next candidate.
        """
        CASH_BUFFER = 1.05
        settled = self.get_settled_cash()

        # Store full ranked list for potential rotation
        self.oca_candidates_ranked = candidates

        placed = 0
        for candidate in candidates:
            ticker = candidate["ticker"]
            state = candidate["state"]
            contract = state["contract"]
            or_high = state["or_high"]

            qty = max(1, int(self.position_size / or_high))
            cost = qty * or_high

            # Verify cash for this position
            if settled is not None and cost * CASH_BUFFER > settled:
                old_qty = qty
                qty = max(1, int(settled / (or_high * CASH_BUFFER)))
                if qty < 1:
                    logger.warning(f"{ticker}: skipped — insufficient settled cash "
                                   f"${settled:,.2f} for even 1 share @ ${or_high:.2f}")
                    continue
                logger.info(f"{ticker}: qty {old_qty} → {qty} to fit settled cash")

            trigger_price = round(or_high + ORBConfig.OCA_STOP_OFFSET, 2)
            limit_price = round(or_high * (1 + ORBConfig.OCA_LIMIT_SLIPPAGE), 2)

            order = Order()
            order.action = "BUY"
            order.totalQuantity = qty
            order.orderType = "STP LMT"
            order.auxPrice = trigger_price    # Stop trigger
            order.lmtPrice = limit_price      # Max fill price (slippage cap)
            order.tif = "DAY"

            try:
                trade = self.ib.placeOrder(contract, order)
                self.ib.sleep(2)

                # Verify IBKR accepted the order
                if trade.orderStatus.status in ("Cancelled", "Inactive"):
                    logger.warning(f"  {ticker}: order REJECTED (status={trade.orderStatus.status}), "
                                   f"trying next candidate...")
                    continue

                self.oca_trades[ticker] = trade
                self.oca_placed_time = datetime.now()
                placed += 1
                logger.info(f"  Stop-limit placed: {ticker} BUY STP LMT "
                            f"trigger=${trigger_price:.2f} limit=${limit_price:.2f} "
                            f"qty={qty}")
                break  # Cash account: only one pending order at a time

            except Exception as e:
                logger.warning(f"  {ticker}: order failed: {e}, trying next candidate...")

        if placed:
            logger.info(f"Pre-placed entry active: {list(self.oca_trades.keys())[0]} "
                        f"({len(candidates)-1} backups ranked)")
        else:
            logger.warning("All candidates failed — falling back to scan mode")
        return placed > 0

    def check_rotation(self):
        """
        If the current stop-limit hasn't triggered within ROTATION_MINUTES,
        cancel it and place the best available candidate using streaming prices.
        """
        if not self.oca_trades or not self.oca_placed_time:
            return
        if not self.oca_candidates_ranked:
            return

        elapsed = (datetime.now() - self.oca_placed_time).total_seconds() / 60
        if elapsed < ORBConfig.OCA_ROTATION_MINUTES:
            return

        current_ticker = list(self.oca_trades.keys())[0]
        logger.info(f"{current_ticker}: no fill after {elapsed:.0f} min — rotating...")

        # Cancel current order
        trade = self.oca_trades[current_ticker]
        try:
            self.ib.cancelOrder(trade.order)
            self.ib.sleep(1)
        except Exception:
            pass
        del self.oca_trades[current_ticker]
        self.ticker_state[current_ticker]["breakout_detected"] = True  # Don't retry this one

        # Find the best remaining candidate using live streaming prices
        # Prefer tickers whose current price is close to (but below) their OR high
        # — these are coiling for a breakout. Skip tickers already past their OR high.
        best = None
        best_proximity = float('inf')

        for candidate in self.oca_candidates_ranked:
            t = candidate["ticker"]
            state = candidate["state"]
            if t == current_ticker:
                continue
            if state.get("breakout_detected"):
                continue

            price = self.get_streaming_price(t)
            if price is None:
                continue

            or_high = state["or_high"]
            or_low = state["or_low"]

            # Skip if already broken out (price above OR high)
            if price > or_high:
                logger.debug(f"  {t}: already above OR high ${or_high:.2f} (price=${price:.2f}), skip")
                state["breakout_detected"] = True
                continue

            # Skip if below OR low (broken down, not a long candidate)
            if price < or_low:
                logger.debug(f"  {t}: below OR low ${or_low:.2f} (price=${price:.2f}), skip")
                continue

            # Proximity: how close to breakout (lower = better)
            proximity = (or_high - price) / or_high
            if proximity < best_proximity:
                best_proximity = proximity
                best = candidate

        if best:
            logger.info(f"  Rotating to {best['ticker']} "
                        f"(${self.get_streaming_price(best['ticker']):.2f}, "
                        f"{best_proximity*100:.2f}% from breakout)")
            self.oca_candidates_ranked = [c for c in self.oca_candidates_ranked
                                          if c["ticker"] != current_ticker]
            self.place_oca_orders([best])
        else:
            logger.info("  No viable candidates remaining — falling back to scan mode")
            self.oca_mode = False

    def check_oca_fills(self):
        """
        Check if the pre-placed stop-limit has filled.
        When a fill is detected: place the initial stop, record the position,
        and begin trail management.
        """
        # Check for time-based rotation first
        self.check_rotation()

        for ticker, trade in list(self.oca_trades.items()):
            status = trade.orderStatus.status

            if status == "Filled":
                fill_price = trade.orderStatus.avgFillPrice
                if not fill_price or fill_price <= 0:
                    fill_price = trade.order.lmtPrice  # Fallback

                state = self.ticker_state[ticker]
                contract = state["contract"]
                or_range = state["or_range"]
                or_high = state["or_high"]
                qty = int(trade.orderStatus.filled) or trade.order.totalQuantity

                slippage = fill_price - or_high
                logger.info(f"STOP-LIMIT FILL: {ticker} {qty} shares @ ${fill_price:.2f} "
                            f"(breakout=${or_high:.2f}, slip={'+'if slippage>0 else ''}"
                            f"{slippage:.2f})")

                # Place initial stop (standalone — not bracket-linked)
                stop_price = round(fill_price - or_range * ORBConfig.STOP_MULT, 2)
                trail_amt = round(or_range * ORBConfig.TRAIL_MULT, 2)

                initial_stop = StopOrder("SELL", qty, stop_price)
                initial_stop.tif = "DAY"
                initial_stop.transmit = True

                try:
                    stop_trade = self.ib.placeOrder(contract, initial_stop)
                    self.ib.sleep(2)

                    # Verify stop was accepted
                    if stop_trade.orderStatus.status in ("Cancelled", "Inactive"):
                        logger.error(f"{ticker}: OCA stop REJECTED — "
                                     f"closing position immediately")
                        close = MarketOrder("SELL", qty)
                        close.tif = "DAY"
                        self.ib.placeOrder(contract, close)
                        del self.oca_trades[ticker]
                        self.trades_today += 1
                        continue
                except Exception as e:
                    logger.error(f"{ticker}: OCA stop placement failed: {e}")
                    del self.oca_trades[ticker]
                    continue

                # Record position (same structure as enter_trade)
                self.positions[ticker] = {
                    "direction": "long",
                    "entry": fill_price,
                    "qty": qty,
                    "stop": stop_price,
                    "trail_amt": trail_amt,
                    "trail_active": False,
                    "highest": fill_price,
                    "order_id": trade.order.orderId,
                    "stop_order_id": initial_stop.orderId,
                    "confirmed": True,  # OCA fill = position confirmed
                    "contract": contract,
                }
                self.trades_today += 1
                state["breakout_detected"] = True

                logger.info(f"ENTRY: LONG {qty} {ticker} @ ${fill_price:.2f} | "
                            f"stop=${stop_price:.2f} | trail=${trail_amt:.2f} "
                            f"({ORBConfig.TRAIL_MULT}x OR) | "
                            f"trade {self.trades_today}/{self.max_trades}")

                # Discord alert
                try:
                    from live.alerts import send_discord
                    slip_str = f", slip {'+'if slippage>0 else ''}{slippage:.2f}" if abs(slippage) > 0.001 else ""
                    if fmt_entry:
                        msg = fmt_entry("ORB", ticker, "long", qty,
                                        fill_price, stop_price, None,
                                        position_size=self.position_size,
                                        extra={"Entry": "stop-limit",
                                               "Slippage": f"{'+'if slippage>0 else ''}{slippage:.2f}" if abs(slippage) > 0.001 else "none",
                                               "Trail": f"${trail_amt:.2f}",
                                               "Trade": f"{self.trades_today}/{self.max_trades}"})
                    else:
                        msg = (f"ORB LONG: {qty} {ticker} @ ${fill_price:.2f}"
                               f"{slip_str} | stop=${stop_price:.2f} | "
                               f"trail=${trail_amt:.2f}")
                    send_discord(msg)
                except Exception:
                    pass

                del self.oca_trades[ticker]

            elif status in ("Cancelled", "Inactive"):
                logger.warning(f"{ticker}: stop-limit order {status.lower()}")
                del self.oca_trades[ticker]

                # Rotate to next ranked candidate
                if self.oca_candidates_ranked:
                    remaining = [c for c in self.oca_candidates_ranked
                                 if c["ticker"] != ticker
                                 and not self.ticker_state.get(c["ticker"], {}).get("breakout_detected")]
                    if remaining:
                        logger.info(f"Rotating to next candidate: {remaining[0]['ticker']}")
                        self.oca_candidates_ranked = remaining
                        self.place_oca_orders(remaining)

    def cancel_oca_orders(self):
        """Cancel all unfilled pre-placed orders (called at 2 PM cutoff or EOD)."""
        if not self.oca_trades:
            return
        logger.info(f"Cancelling {len(self.oca_trades)} unfilled pre-placed orders...")
        for ticker, trade in list(self.oca_trades.items()):
            try:
                self.ib.cancelOrder(trade.order)
                logger.info(f"  {ticker}: order cancelled")
            except Exception as e:
                logger.warning(f"  {ticker}: cancel failed: {e}")
        self.oca_trades.clear()

    # ── Streaming Market Data ────────────────────────────────────────

    def subscribe_streaming(self, contracts_dict):
        """
        Subscribe to real-time market data for candidates.
        Ticker objects update in-place from TWS data feed — no per-poll API calls.
        """
        count = 0
        for ticker, contract in contracts_dict.items():
            if ticker in self.streaming_tickers:
                continue  # Already subscribed
            try:
                ticker_obj = self.ib.reqMktData(contract, "", False, False)
                self.streaming_tickers[ticker] = ticker_obj
                self.streaming_contracts[ticker] = contract
                count += 1
            except Exception as e:
                logger.warning(f"{ticker}: streaming subscribe failed: {e}")

            if count % 20 == 0 and count > 0:
                self.ib.sleep(0.5)  # Throttle subscriptions

        if count > 0:
            logger.info(f"Streaming subscriptions: +{count} new "
                        f"({len(self.streaming_tickers)} total)")

    def cancel_all_streaming(self):
        """Cancel all streaming market data subscriptions."""
        if not self.streaming_contracts:
            return
        for ticker, contract in self.streaming_contracts.items():
            try:
                self.ib.cancelMktData(contract)
            except Exception:
                pass
        count = len(self.streaming_tickers)
        self.streaming_tickers.clear()
        self.streaming_contracts.clear()
        logger.info(f"Streaming subscriptions cancelled ({count} tickers)")

    def get_streaming_price(self, ticker):
        """
        Read price from streaming Ticker object (local read, no API call).
        Falls back to snapshot if streaming not available for this ticker.
        """
        ticker_obj = self.streaming_tickers.get(ticker)
        if ticker_obj is None:
            # Fallback to snapshot
            contract = self.positions.get(ticker, {}).get("contract")
            if contract:
                return self.get_current_price(contract)
            return None

        price = ticker_obj.last
        if price != price:  # NaN check
            price = ticker_obj.marketPrice()
        if price != price:
            # Try mid-point
            if ticker_obj.bid == ticker_obj.bid and ticker_obj.ask == ticker_obj.ask:
                price = (ticker_obj.bid + ticker_obj.ask) / 2
            else:
                price = None
        return price

    # ── Breakout Scanning (fallback when OCA disabled) ───────────────

    def scan_for_breakouts(self):
        """
        Scan all tickers with computed ORs for volume-confirmed breakouts.
        Returns list of candidates: [{ticker, direction, price, vol_ratio, state}]
        sorted by vol_ratio descending (highest conviction first).
        """
        candidates = []

        for ticker, state in self.ticker_state.items():
            if state["breakout_detected"]:
                continue
            if ticker in self.positions:
                continue  # Already have a position in this ticker

            contract = state["contract"]
            bars = self.get_1min_bars(contract, duration="120 S")
            if not bars:
                continue

            latest_bar = bars[-1]
            price = latest_bar.close
            bar_volume = latest_bar.volume

            # Volume confirmation
            if ORBConfig.REQUIRE_VOLUME_CONFIRMATION:
                if bar_volume < state["or_avg_volume"]:
                    continue

            vol_ratio = bar_volume / state["or_avg_volume"] if state["or_avg_volume"] > 0 else 1.0

            direction = None
            if ORBConfig.DIRECTION in ("long", "both") and latest_bar.high > state["or_high"]:
                direction = "long"
            elif ORBConfig.DIRECTION in ("short", "both") and latest_bar.low < state["or_low"]:
                direction = "short"

            if direction:
                candidates.append({
                    "ticker": ticker,
                    "direction": direction,
                    "price": price,
                    "vol_ratio": vol_ratio,
                    "state": state,
                    "contract": contract,
                })

            # Throttle
            if len(candidates) % 30 == 0 and len(candidates) > 0:
                self.ib.sleep(1)

        # ── Apply pre-market intelligence scoring ─────────────────
        if self.premarket_brief and self.premarket_brief.gap_rankings is not None:
            rankings = self.premarket_brief.gap_rankings
            skip_set = self.premarket_brief.skip_tickers or set()

            scored = []
            for c in candidates:
                ticker = c["ticker"]
                # Skip earnings-day tickers
                if ticker in skip_set:
                    logger.info(f"  {ticker}: skipped (earnings/skip list)")
                    continue

                # Get intel score
                match = rankings[rankings["ticker"] == ticker]
                intel = float(match.iloc[0]["intel_score"]) if not match.empty else 0.0

                # Composite: 70% vol_ratio + 30% intel
                vol_norm = min(c["vol_ratio"] / 5.0, 1.0)
                c["intel_score"] = intel
                c["composite_score"] = 0.7 * vol_norm + 0.3 * intel
                scored.append(c)

            candidates = scored
            candidates.sort(key=lambda c: c.get("composite_score", 0), reverse=True)
        else:
            # Fallback: pure vol_ratio ranking
            candidates.sort(key=lambda c: c["vol_ratio"], reverse=True)

        return candidates

    # ── Order Execution ───────────────────────────────────────────────

    def enter_trade(self, candidate):
        """Place a bracket order for a breakout candidate."""
        ticker = candidate["ticker"]
        direction = candidate["direction"]
        price = candidate["price"]
        state = candidate["state"]
        contract = candidate["contract"]
        vol_ratio = candidate["vol_ratio"]

        qty = max(1, int(self.position_size / price))
        cost = qty * price

        # ── Check settled cash before placing order ──────────
        # IBKR requires extra margin beyond stock cost for bracket orders
        # (stop order overhead, commissions, fees). Use 5% buffer.
        CASH_BUFFER = 1.05
        settled = self.get_settled_cash()
        if settled is not None:
            if settled < price * CASH_BUFFER:
                logger.warning(f"{ticker}: Not enough settled cash: ${settled:,.2f} available, "
                               f"need ~${price * CASH_BUFFER:,.2f} for 1 share + bracket overhead. Skipping.")
                try:
                    from live.alerts import send_discord
                    send_discord(f"ORB — Skipping {ticker}: only ${settled:,.2f} settled cash "
                                 f"(need ~${cost * CASH_BUFFER:,.2f} incl. bracket overhead). Cash settles tomorrow.")
                except Exception:
                    pass
                return False
            if cost * CASH_BUFFER > settled:
                old_qty = qty
                qty = max(1, int(settled / (price * CASH_BUFFER)))
                cost = qty * price
                logger.info(f"{ticker}: Reducing qty {old_qty} → {qty} "
                            f"to fit settled cash ${settled:,.2f} (with 5% bracket buffer)")

        or_range = state["or_range"]
        trail_amt = round(or_range * ORBConfig.TRAIL_MULT, 2)
        stop_price = round(state["or_high"] - or_range * ORBConfig.STOP_MULT, 2)
        planned_entry = state["or_high"]  # Theoretical breakout level

        # Entry: market buy
        parent = MarketOrder("BUY", qty)
        parent.tif = "DAY"
        parent.transmit = False

        try:
            parent_trade = self.ib.placeOrder(contract, parent)
            self.ib.sleep(2)

            if parent_trade.orderStatus.status in ("Cancelled", "Inactive"):
                logger.error(f"{ticker}: entry REJECTED: {parent_trade.orderStatus.status}")
                return False

            # Initial stop (protects against immediate reversal before trail kicks in)
            initial_stop = StopOrder("SELL", qty, stop_price)
            initial_stop.tif = "DAY"
            initial_stop.parentId = parent_trade.order.orderId
            initial_stop.transmit = True
            stop_trade = self.ib.placeOrder(contract, initial_stop)

            # Wait for IBKR to process the full bracket order
            self.ib.sleep(3)

            # ── Verify bracket was ACCEPTED before recording entry ──
            # The bracket rejection (Error 201) happens when transmit=True
            # triggers the full bracket submission. We must re-check BOTH orders.
            self.ib.sleep(1)  # Extra settle time for IBKR status propagation
            if parent_trade.orderStatus.status in ("Cancelled", "Inactive"):
                logger.error(f"{ticker}: bracket order REJECTED after transmit — "
                             f"parent status: {parent_trade.orderStatus.status}. "
                             f"Likely insufficient settled cash for bracket overhead.")
                try:
                    from live.alerts import send_discord
                    send_discord(f"⚠️ ORB — {ticker} bracket REJECTED by IBKR (Error 201). "
                                 f"Settled cash insufficient for bracket overhead.")
                except Exception:
                    pass
                return False
            if stop_trade.orderStatus.status in ("Cancelled", "Inactive"):
                logger.error(f"{ticker}: stop order REJECTED — "
                             f"status: {stop_trade.orderStatus.status}. "
                             f"Cancelling parent order.")
                try:
                    self.ib.cancelOrder(parent_trade.order)
                except Exception:
                    pass
                try:
                    from live.alerts import send_discord
                    send_discord(f"⚠️ ORB — {ticker} stop order REJECTED. Parent cancelled.")
                except Exception:
                    pass
                return False

        except Exception as e:
            logger.error(f"{ticker}: order placement failed: {e}")
            return False

        # ── Extract actual fill price from IBKR ──────────────────
        # Market orders fill at the ask, NOT at or_high. Use the real fill
        # for P&L tracking, trailing stop initialization, and Discord alerts.
        fill_price = parent_trade.orderStatus.avgFillPrice
        if fill_price and fill_price > 0:
            entry_price = fill_price
            slippage = entry_price - planned_entry
            if abs(slippage) > 0.001:
                logger.info(f"{ticker}: filled @ ${entry_price:.2f} "
                            f"(vs breakout ${planned_entry:.2f}, "
                            f"slip {'+'if slippage>0 else ''}{slippage:.2f})")
        else:
            # Fallback: use the scanner's latest price (better than or_high)
            entry_price = price
            logger.warning(f"{ticker}: no fill price from IBKR, using scanner price ${price:.2f}")

        # ── Only record position AFTER bracket is confirmed accepted ──
        self.positions[ticker] = {
            "direction": direction,
            "entry": entry_price,
            "qty": qty,
            "stop": stop_price,
            "trail_amt": trail_amt,
            "trail_active": False,  # Activated once price moves above entry + trail_amt
            "highest": entry_price,
            "order_id": parent_trade.order.orderId,
            "stop_order_id": initial_stop.orderId,
            "confirmed": False,
            "contract": contract,
        }
        self.trades_today += 1
        state["breakout_detected"] = True

        logger.info(f"ENTRY: LONG {qty} {ticker} @ ${entry_price:.2f} "
                    f"(vol {vol_ratio:.1f}x) | initial stop=${stop_price:.2f} | "
                    f"trail=${trail_amt:.2f} ({ORBConfig.TRAIL_MULT}x OR) | "
                    f"trade {self.trades_today}/{self.max_trades}")

        try:
            from live.alerts import send_discord
            slip_str = ""
            slippage = entry_price - planned_entry
            if abs(slippage) > 0.001:
                slip_str = f" | slip {'+'if slippage>0 else ''}{slippage:.2f}"
            if fmt_entry:
                msg = fmt_entry("ORB", ticker, direction, qty,
                                entry_price, stop_price, None,
                                position_size=self.position_size,
                                extra={"Vol ratio": f"{vol_ratio:.1f}x",
                                       "Trail": f"${trail_amt:.2f}",
                                       "Slippage": f"{'+'if slippage>0 else ''}{slippage:.2f}" if abs(slippage) > 0.001 else "none",
                                       "Trade": f"{self.trades_today}/{self.max_trades}"})
            else:
                msg = f"ORB LONG: {qty} {ticker} @ ${entry_price:.2f} (vol {vol_ratio:.1f}x, trail ${trail_amt:.2f}{slip_str})"
            send_discord(msg)
        except Exception:
            pass

        return True

    # ── Position Monitoring ───────────────────────────────────────────

    def check_all_positions(self):
        """Check positions, update trailing stops, detect exits."""
        ibkr_positions = {pos.contract.symbol: pos.position
                          for pos in self.ib.positions()}

        closed = []
        for ticker, pos in self.positions.items():
            has_ibkr_pos = ibkr_positions.get(ticker, 0) != 0

            if has_ibkr_pos:
                pos["confirmed"] = True

                # Update trailing stop: use streaming price (instant) or snapshot (fallback)
                if ticker in self.streaming_tickers:
                    price = self.get_streaming_price(ticker)
                else:
                    price = self.get_current_price(pos["contract"])
                if price and price > pos.get("highest", pos["entry"]):
                    pos["highest"] = price
                    new_stop = round(price - pos["trail_amt"], 2)
                    if new_stop > pos["stop"]:
                        # Modify the stop order to the new higher price
                        try:
                            for order in self.ib.openOrders():
                                if (order.orderId == pos.get("stop_order_id") or
                                    (order.parentId == pos["order_id"] and
                                     order.orderType in ("STP", "TRAIL"))):
                                    order.auxPrice = new_stop
                                    self.ib.placeOrder(pos["contract"], order)
                                    break
                            old_stop = pos["stop"]
                            pos["stop"] = new_stop
                            if not pos.get("trail_active"):
                                pos["trail_active"] = True
                                logger.info(f"{ticker}: trail activated | "
                                            f"stop ${old_stop:.2f} -> ${new_stop:.2f} "
                                            f"(high ${price:.2f})")
                            else:
                                logger.debug(f"{ticker}: trail updated ${old_stop:.2f} -> "
                                             f"${new_stop:.2f} (high ${price:.2f})")
                        except Exception as e:
                            logger.warning(f"{ticker}: failed to update stop: {e}")
                continue

            if not has_ibkr_pos and pos.get("confirmed"):
                # Position was confirmed, now gone = stop was hit
                entry = pos["entry"]
                exit_price = pos["stop"]  # Best estimate: the current stop level
                reason = "trail" if pos.get("trail_active") else "stop"

                pnl = (exit_price - entry) / entry * 100
                self.daily_pnl += pnl

                logger.info(f"{ticker}: {reason} filled @ ~${exit_price:.2f} | "
                            f"P&L: {pnl:+.2f}%")
                self.record_trade(ticker, pos["direction"], entry, exit_price,
                                  pos["qty"], reason)
                try:
                    from live.alerts import send_discord
                    if fmt_exit:
                        msg = fmt_exit("ORB", ticker, pos["direction"], pos["qty"],
                                       entry, exit_price, reason)
                    else:
                        msg = f"ORB {reason.upper()}: {ticker} {pnl:+.2f}%"
                    send_discord(msg)
                except Exception:
                    pass
                closed.append(ticker)

            elif not has_ibkr_pos and not pos.get("confirmed"):
                # Check if orders still pending
                open_orders = self.ib.openOrders()
                has_pending = any(o.orderId == pos.get("order_id") for o in open_orders)
                if not has_pending:
                    logger.error(f"{ticker}: entry REJECTED — no position, no pending orders")
                    closed.append(ticker)

        for ticker in closed:
            del self.positions[ticker]

    def force_close_all(self):
        """Force close all open positions at end of day."""
        # Cancel unfilled OCA orders first
        self.cancel_oca_orders()

        if not self.positions:
            # Clean up streaming even if no positions
            self.cancel_all_streaming()
            return

        logger.info(f"EOD: Force closing {len(self.positions)} positions...")

        # Cancel all open orders first
        for order in self.ib.openOrders():
            try:
                self.ib.cancelOrder(order)
            except Exception:
                pass
        self.ib.sleep(1)

        for ticker, pos in list(self.positions.items()):
            contract = pos["contract"]
            qty = pos["qty"]
            direction = pos["direction"]

            close_order = MarketOrder("SELL" if direction == "long" else "BUY", qty)
            close_order.tif = "DAY"
            self.ib.placeOrder(contract, close_order)
            self.ib.sleep(1)

            price = self.get_current_price(contract)
            if price and pos["entry"]:
                if direction == "long":
                    pnl = (price - pos["entry"]) / pos["entry"] * 100
                else:
                    pnl = (pos["entry"] - price) / pos["entry"] * 100
                self.daily_pnl += pnl
                logger.info(f"{ticker}: EOD close @ ${price:.2f} | P&L: {pnl:+.2f}%")
                self.record_trade(ticker, direction, pos["entry"], price, qty, "eod")

                try:
                    from live.alerts import send_discord
                    if fmt_exit:
                        msg = fmt_exit("ORB", ticker, direction, qty, pos["entry"], price, "eod")
                    else:
                        msg = f"ORB EOD: {ticker} {pnl:+.2f}%"
                    send_discord(msg)
                except Exception:
                    pass

        self.positions.clear()
        self.cancel_all_streaming()

    # ── Main Run Loop ─────────────────────────────────────────────────

    def run(self):
        """Main trading loop for multi-ticker ORB."""
        mode = "PAPER" if self.paper else "LIVE"
        logger.info("=" * 60)
        logger.info(f"  Multi-Ticker ORB Day Trader -- {mode} MODE")
        logger.info(f"  Position size: ${self.position_size:,.0f}")
        logger.info(f"  Max trades/day: {self.max_trades}")
        logger.info(f"  Max daily exposure: ${self.position_size * self.max_trades:,.0f}")
        logger.info(f"  Strategy: {ORBConfig.OR_MINUTES}min OR, "
                    f"{ORBConfig.TRAIL_MULT}x trail, {ORBConfig.STOP_MULT}x initial stop, "
                    f"gap<{ORBConfig.MAX_GAP_PCT}%")
        logger.info(f"  Volume confirmation: "
                    f"{'ON' if ORBConfig.REQUIRE_VOLUME_CONFIRMATION else 'OFF'}")
        logger.info(f"  Entry mode: "
                    f"{'OCA stop-limit' if ORBConfig.OCA_ENABLED else 'poll + market order'}")
        logger.info(f"  Streaming data: "
                    f"{'ON' if ORBConfig.STREAMING_ENABLED else 'OFF (snapshot polling)'}")
        logger.info("=" * 60)

        if not self.connect():
            return

        try:
            # Load universe and qualify contracts
            tickers = self.load_universe()
            if not tickers:
                return

            logger.info("Qualifying contracts...")
            contracts = {}
            for ticker in tickers:
                c = self.qualify_contract(ticker)
                if c:
                    contracts[ticker] = c
                if len(contracts) % 50 == 0:
                    self.ib.sleep(1)
            logger.info(f"Qualified {len(contracts)}/{len(tickers)} contracts")

            self.today = date.today()
            or_computed = False

            now = self.get_current_time_et()
            if now.weekday() >= 5:
                logger.info("Not a trading day (weekend). Exiting.")
                return

            while True:
                now = self.get_current_time_et()
                current_time = now.time()

                # Before market open — run pre-market intelligence
                if current_time < ORBConfig.MARKET_OPEN:
                    wait_mins = (datetime.combine(date.today(), ORBConfig.MARKET_OPEN) -
                                 datetime.combine(date.today(), current_time)).seconds // 60

                    # Run pre-market scan between 9:00 and 9:25 (once per day)
                    if (self.premarket_brief is None
                            and self.intel_collector is not None
                            and current_time >= dtime(9, 0)):
                        logger.info("Running pre-market intelligence scan...")
                        try:
                            self.premarket_brief = self.intel_collector.collect(tickers)
                            logger.info(f"Pre-market brief ready: {self.premarket_brief.summary()}")
                        except Exception as e:
                            logger.warning(f"Pre-market intel failed (non-fatal): {e}")
                            self.premarket_brief = None

                        # RSI(2) bear-regime check + auto-execute (once during pre-market)
                        if rsi2_check is not None:
                            try:
                                rsi2_signal = rsi2_check(ib=self.ib)
                                rsi2_action = rsi2_signal["action"]
                                if rsi2_action == "buy":
                                    if rsi2_signal.get("executed"):
                                        logger.info("RSI(2) BUY auto-executed — "
                                                    "cash tied up, ORB will skip today")
                                    else:
                                        logger.info(f"RSI(2) BUY SIGNAL — "
                                                    f"{rsi2_signal['reason']}")
                                elif rsi2_action == "sell":
                                    if rsi2_signal.get("executed"):
                                        logger.info("RSI(2) SELL auto-executed — "
                                                    "cash freed, ORB can trade")
                                    else:
                                        logger.info(f"RSI(2) SELL SIGNAL — "
                                                    f"{rsi2_signal['reason']}")
                                elif rsi2_action == "hold":
                                    logger.info(f"RSI(2) position open — "
                                                f"{rsi2_signal['reason']}")
                                    logger.info("Cash tied up in RSI(2) hold — "
                                                "ORB will check settled cash before trading")
                            except Exception as e:
                                logger.warning(f"RSI(2) check failed (non-fatal): {e}")
                    else:
                        logger.info(f"Market opens in {wait_mins} minutes...")

                    self.ib.sleep(max(min(wait_mins * 60, 60), 10))
                    continue

                # OR forming (9:30 - 9:45)
                if current_time < ORBConfig.OR_END:
                    logger.info(f"Opening range forming... ({current_time.strftime('%H:%M')})")
                    self.ib.sleep(30)
                    continue

                # Compute all opening ranges (once)
                if not or_computed:
                    # Use pre-market watchlist if available (narrows 85 -> ~30 tickers)
                    scan_tickers = tickers
                    scan_contracts = contracts
                    if (self.premarket_brief
                            and self.premarket_brief.watchlist):
                        wl = self.premarket_brief.watchlist
                        scan_tickers = [t for t in wl if t in contracts]
                        scan_contracts = {t: contracts[t] for t in scan_tickers}
                        logger.info(f"Using pre-market watchlist: {len(scan_tickers)} tickers "
                                    f"(narrowed from {len(tickers)})")
                    self.compute_all_opening_ranges(scan_tickers, scan_contracts)
                    or_computed = True
                    if not self.ticker_state:
                        logger.warning("No tickers passed OR computation. Sitting out today.")
                        break

                    # ── OCA + Streaming setup (immediately after OR computation) ──
                    if ORBConfig.OCA_ENABLED:
                        try:
                            oca_candidates = self.rank_oca_candidates()
                            if oca_candidates:
                                # Subscribe streaming for OCA candidates
                                if ORBConfig.STREAMING_ENABLED:
                                    oca_contracts = {c["ticker"]: c["state"]["contract"]
                                                     for c in oca_candidates}
                                    self.subscribe_streaming(oca_contracts)
                                    self.ib.sleep(2)  # Let streaming data populate

                                # Place OCA orders
                                if self.place_oca_orders(oca_candidates):
                                    self.oca_mode = True
                                    logger.info(f"Pre-placed stop-limit ACTIVE: "
                                                f"{list(self.oca_trades.keys())}, "
                                                f"{len(self.streaming_tickers)} streaming")
                                else:
                                    logger.warning("OCA placement failed — "
                                                   "falling back to scan mode")
                            else:
                                logger.info("No OCA candidates qualified — "
                                            "falling back to scan mode")
                        except Exception as e:
                            logger.error(f"OCA setup failed: {e} — "
                                         "falling back to scan mode")
                            self.oca_mode = False
                    elif ORBConfig.STREAMING_ENABLED:
                        # Streaming without OCA: subscribe for all computed tickers
                        # (improves trail accuracy even with poll-based entry)
                        all_contracts = {t: s["contract"]
                                         for t, s in self.ticker_state.items()}
                        self.subscribe_streaming(all_contracts)

                # After market close
                if current_time >= ORBConfig.MARKET_CLOSE:
                    logger.info("Market closed.")
                    break

                # Force close at 3:55 PM
                if current_time >= ORBConfig.FORCE_EXIT:
                    self.force_close_all()
                    self.ib.sleep(60)
                    continue

                # No new entries after 2 PM
                if current_time >= ORBConfig.LAST_ENTRY:
                    # Cancel unfilled OCA orders at cutoff
                    if self.oca_trades:
                        logger.info(f"2 PM cutoff: cancelling {len(self.oca_trades)} "
                                    f"unfilled OCA orders")
                        self.cancel_oca_orders()

                    # Just monitor existing positions
                    if self.positions:
                        self.check_all_positions()
                    self.ib.sleep(30)
                    continue

                # ── Entry: OCA mode (check fills) or scan mode (poll breakouts) ──
                if self.trades_today < self.max_trades:
                    if self.oca_mode and self.oca_trades:
                        # OCA mode: IBKR detects breakouts, we just check fills
                        self.check_oca_fills()
                    elif not self.oca_mode:
                        # Fallback: original poll-based scanning
                        candidates = self.scan_for_breakouts()
                        slots_available = self.max_trades - self.trades_today

                        for candidate in candidates[:slots_available]:
                            logger.info(f"BREAKOUT: {candidate['ticker']} "
                                        f"{candidate['direction'].upper()} "
                                        f"@ ${candidate['price']:.2f} "
                                        f"(vol {candidate['vol_ratio']:.1f}x)")
                            self.enter_trade(candidate)

                # Monitor existing positions (streaming = fast, snapshot = fallback)
                if self.positions:
                    self.check_all_positions()

                # Adaptive sleep: fast when actively trailing or watching OCA
                if self.positions or (self.oca_mode and self.oca_trades):
                    self.ib.sleep(ORBConfig.FAST_CHECK_INTERVAL)
                else:
                    self.ib.sleep(ORBConfig.SLOW_CHECK_INTERVAL)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            if self.positions:
                logger.warning(f"You have {len(self.positions)} open positions! "
                               "Close them manually in TWS.")
        except Exception as e:
            logger.error(f"Error: {e}")
            if self.positions:
                logger.warning(f"You may have {len(self.positions)} open positions! Check TWS.")
        finally:
            # Clean up OCA and streaming
            self.cancel_oca_orders()
            self.cancel_all_streaming()

            # Daily summary
            entry_mode = "OCA" if self.oca_mode else "scan"
            logger.info(f"Daily summary: {self.trades_today} trades, "
                        f"P&L: {self.daily_pnl:+.2f}% (entry: {entry_mode})")
            try:
                from live.alerts import send_discord
                if fmt_day_complete:
                    msg = fmt_day_complete(
                        "Multi-ORB",
                        traded=self.trades_today > 0,
                        ticker=None,
                        pnl_pct=self.daily_pnl,
                        extra={"Trades": f"{self.trades_today}/{self.max_trades}",
                               "Entry mode": entry_mode,
                               "Positions": ", ".join(self.positions.keys()) or "none"},
                    )
                else:
                    msg = (f"Multi-ORB complete | {self.trades_today} trades | "
                           f"P&L: {self.daily_pnl:+.2f}%")
                send_discord(msg)
            except Exception:
                pass

            self.disconnect()


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def run_orb(args=None):
    """Entry point for ORB trader."""
    import argparse
    parser = argparse.ArgumentParser(description="ORB Day Trader")
    parser.add_argument("--live", action="store_true", help="Use live account (default: paper)")
    parser.add_argument("--size", type=float, default=1900, help="Position size in USD")
    parser.add_argument("--single", action="store_true",
                        help="Single-ticker mode (QQQ only, legacy)")
    parser.add_argument("--max-trades", type=int, default=1,
                        help="Max trades per day (default: 1)")
    parser.add_argument("--no-oca", action="store_true",
                        help="Disable OCA stop-limit entry (use poll-based scan)")
    parser.add_argument("--no-streaming", action="store_true",
                        help="Disable streaming data (use snapshot polling)")
    if args is not None:
        parsed = parser.parse_args(args)
    else:
        parsed = parser.parse_args(sys.argv[2:] if len(sys.argv) > 2 else [])

    # Apply CLI overrides to config
    if parsed.no_oca:
        ORBConfig.OCA_ENABLED = False
    if parsed.no_streaming:
        ORBConfig.STREAMING_ENABLED = False

    if parsed.single:
        # Legacy single-ticker mode
        trader = ORBTrader(paper=not parsed.live, position_size=parsed.size)
        trader.run()
    else:
        # Multi-ticker mode (default)
        trader = MultiORBTrader(paper=not parsed.live, position_size=parsed.size,
                                max_trades=parsed.max_trades)
        trader.run()


if __name__ == "__main__":
    run_orb(sys.argv[1:])
