"""
live/swing_trader.py
--------------------
Fully automated swing trader with IBKR bracket orders.

Runs daily: generates ML signals pre-market, places bracket orders at open,
monitors positions for stop/target/time exits throughout the day.

Architecture:
  - 8:30 AM: Connect to TWS, download data, compute ML signals
  - 9:31 AM: Place bracket buy orders (market + stop + target) for top signals
  - 9:31-4:00 PM: Monitor positions (bracket fills, time exits)
  - 4:00 PM: Daily summary, Discord alert
  - Overnight: Bracket orders (GTC) remain active on IBKR servers

Usage:
    python main.py swing                  # Paper mode (default)
    python main.py swing --live           # Live trading
    python main.py swing --monitor-only   # Skip signal generation, just monitor

Requirements:
    - IBKR TWS or IB Gateway running with API enabled (port 7497 paper / 7496 live)
    - Market data subscription for US equities
    - pip install ib_insync
"""

import sys
import time
import asyncio
import yaml
import pandas as pd
import yfinance as yf
from datetime import datetime, date, timedelta, time as dtime
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")

# Persistent log file
_log_dir = Path(__file__).resolve().parent.parent / "logs"
_log_dir.mkdir(exist_ok=True)
logger.add(
    _log_dir / "swing_trader_live.log",
    rotation="1 day",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
)

# Python 3.14 fix
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

from ib_insync import IB, Stock, MarketOrder, LimitOrder, StopOrder


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

class SwingConfig:
    # Connection
    PAPER_PORT = 7497
    LIVE_PORT = 7496
    HOST = "127.0.0.1"
    CLIENT_ID = 10          # Separate from ORB trader (client 20)

    # Timing (Eastern Time)
    SIGNAL_TIME = dtime(8, 30)      # Start signal generation
    ENTRY_TIME = dtime(9, 31)       # Place orders 1 min after open
    LAST_ENTRY = dtime(10, 0)       # No new entries after 10 AM
    CHECK_INTERVAL = dtime(12, 0)   # Midday position check
    CLOSE_CHECK = dtime(15, 45)     # Pre-close time exit check
    MARKET_CLOSE = dtime(16, 0)

    # Monitoring
    POSITION_CHECK_SECONDS = 60     # Check positions every 60s during active hours
    IDLE_CHECK_SECONDS = 300        # Check every 5 min when no action needed


# ═══════════════════════════════════════════════════════════════════════
# Swing Trader
# ═══════════════════════════════════════════════════════════════════════

class SwingTrader:
    def __init__(self, paper: bool = True):
        self.paper = paper
        self.port = SwingConfig.PAPER_PORT if paper else SwingConfig.LIVE_PORT
        self.ib = None
        self.config = self._load_config()
        self.today_signals = []       # ML signals for today
        self.pending_orders = {}      # ticker -> {parent_id, stop_id, target_id}
        self.active_positions = {}    # ticker -> position dict
        self.orders_placed_today = False

    def _load_config(self) -> dict:
        with open("config/config.yaml") as f:
            return yaml.safe_load(f)

    # ── IBKR Connection ──────────────────────────────────────────────

    def connect(self) -> bool:
        """Connect to IBKR TWS."""
        self.ib = IB()
        mode = "PAPER" if self.paper else "LIVE"
        try:
            self.ib.connect(SwingConfig.HOST, self.port,
                            clientId=SwingConfig.CLIENT_ID)
            logger.info(f"Connected to IBKR ({mode}) on port {self.port}")
            return True
        except Exception as e:
            logger.error(f"IBKR connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from IBKR."""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IBKR")

    def ensure_connected(self) -> bool:
        """Reconnect if connection dropped."""
        if self.ib and self.ib.isConnected():
            return True
        logger.warning("IBKR connection lost, reconnecting...")
        return self.connect()

    # ── Account Info ─────────────────────────────────────────────────

    def get_account_balance(self) -> float:
        """Query settled cash from IBKR."""
        try:
            self.ib.sleep(1)
            summary = self.ib.accountSummary()
            for item in summary:
                if item.tag == "SettledCash":
                    return float(item.value)
            # Fallback to NetLiquidation
            for item in summary:
                if item.tag == "NetLiquidation":
                    return float(item.value)
        except Exception as e:
            logger.warning(f"Could not get account balance: {e}")
        return 0.0

    def get_ibkr_positions(self) -> dict:
        """Get current positions from IBKR."""
        try:
            positions = {}
            for pos in self.ib.positions():
                ticker = pos.contract.symbol
                if pos.position != 0:
                    positions[ticker] = {
                        "qty": int(pos.position),
                        "avg_cost": pos.avgCost,
                        "ticker": ticker,
                    }
            return positions
        except Exception as e:
            logger.warning(f"Could not get IBKR positions: {e}")
            return {}

    # ── Signal Generation ────────────────────────────────────────────

    def _build_live_df(self):
        """
        Live data path: yfinance → technical features → sector + earnings enrichment.
        Returns the enriched dataframe. Raises on unrecoverable error.
        """
        from data.universe import build_universe
        from data.pipeline import add_technical_features, add_cross_sectional_features

        universe = build_universe(self.config)
        logger.info(f"yfinance: downloading {len(universe)} tickers (120d)...")
        raw = yf.download(universe, period="120d", auto_adjust=True,
                          progress=False, threads=True)
        if raw.empty:
            raise RuntimeError("yfinance returned empty frame")

        raw.columns.names = ["Field", "Ticker"]
        stacked = raw.stack(level="Ticker", future_stack=True).reset_index()
        stacked.columns = [c.lower() for c in stacked.columns]
        stacked["date"] = pd.to_datetime(stacked["date"])
        stacked = stacked.dropna(subset=["close", "volume"])
        stacked = stacked[stacked["close"] > 0]
        logger.info(f"yfinance: {len(stacked):,} rows after cleanup")

        df = add_technical_features(stacked)
        df = add_cross_sectional_features(df)

        start_str = df["date"].min().strftime("%Y-%m-%d")
        end_str = df["date"].max().strftime("%Y-%m-%d")
        try:
            from data.sectors import add_sector_features
            df = add_sector_features(df, start_str, end_str)
        except Exception as e:
            logger.warning(f"Sector enrichment failed: {e}")
        try:
            from data.earnings import fetch_earnings_dates, add_earnings_features
            tickers = df["ticker"].unique().tolist()
            earnings_map = fetch_earnings_dates(tickers, use_cache=True)
            df = add_earnings_features(df, earnings_map)
        except Exception as e:
            logger.warning(f"Earnings enrichment failed: {e}")

        return df

    def _load_cached_df(self):
        """
        Fallback: load the enriched parquet built by `main.py pipeline && enrich`.
        Returns dataframe or None if no cache available.
        """
        from pathlib import Path
        processed = Path(self.config.get("data", {}).get("processed_dir", "data/processed"))
        cache_path = processed / "price_features_enriched.parquet"
        if not cache_path.exists():
            logger.error(f"No cached signal data at {cache_path} — run `python main.py pipeline && python main.py enrich`")
            return None
        df = pd.read_parquet(cache_path)
        latest = df["date"].max()
        age_hours = (pd.Timestamp.now().normalize() - latest.normalize()).total_seconds() / 3600
        logger.warning(
            f"USING CACHED SIGNALS: {len(df):,} rows, latest={latest.date()}, "
            f"age={age_hours/24:.1f} trading days"
        )
        return df

    def _build_signal_df(self, live_timeout: int = 180):
        """
        Build the enriched signal dataframe.
        Attempts live fetch (with overall timeout); falls back to cached parquet on
        timeout or any unrecoverable error. Returns None if both paths fail.
        """
        import concurrent.futures

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(self._build_live_df)
                try:
                    df = fut.result(timeout=live_timeout)
                    if df is not None and not df.empty:
                        return df
                    logger.warning("Live fetch returned empty — falling back to cache")
                except concurrent.futures.TimeoutError:
                    logger.warning(f"Live fetch exceeded {live_timeout}s — falling back to cache")
                    fut.cancel()  # best-effort; yfinance threads may linger but we move on
        except Exception as e:
            logger.warning(f"Live fetch failed: {e} — falling back to cache")

        return self._load_cached_df()

    def generate_signals(self) -> list:
        """
        Run the full ML signal pipeline and return today's buy candidates.
        Applies: ML scoring → score floor → VIX regime gate → sector/correlation filter.

        Data path: tries live yfinance (with 180s timeout); falls back to the cached
        enriched parquet if live fetch hangs or fails.
        """
        from models.predict import generate_ml_signals
        from utils.risk import compute_correlation_matrix, filter_candidates_by_risk

        logger.info("Generating ML signals...")

        df = self._build_signal_df(live_timeout=180)
        if df is None or df.empty:
            logger.error("No signal data available (live fetch and cache both failed)")
            return []

        # ML scoring (no regime gate — we apply VIX-aware gate below)
        df = generate_ml_signals(df, top_pct=0.10, apply_regime_gate=False)

        # Today's data
        latest_date = df["date"].max()
        today_df = df[df["date"] == latest_date].copy()

        # VIX-aware regime check
        try:
            from live.regime import get_todays_vix_context
            vix_ctx = get_todays_vix_context()
            pct_above = float(today_df["pct_above_sma50"].iloc[0]) if "pct_above_sma50" in today_df.columns else 0.5
            regime_ok = pct_above >= vix_ctx["threshold"]
            logger.info(f"Regime: {'OK' if regime_ok else 'UNFAVORABLE'} | "
                        f"VIX={vix_ctx.get('vix_current', '?')} | "
                        f"Breadth={pct_above:.0%} vs threshold={vix_ctx['threshold']:.0%}")
        except Exception as e:
            logger.warning(f"VIX check failed: {e}")
            regime_ok = True

        if not regime_ok:
            logger.info("Market regime unfavorable — no new entries today")
            self._send_alert("SWING TRADER: Regime UNFAVORABLE — no new trades today")
            return []

        # Get signals, filter out held tickers
        signals = today_df[today_df["signal"] == 1].sort_values("signal_score", ascending=False)
        held_tickers = list(self.active_positions.keys())
        signals = signals[~signals["ticker"].isin(held_tickers)]

        if signals.empty:
            logger.info("No ML signals today")
            return []

        # Sector + correlation filters
        open_positions = [{"ticker": t} for t in held_tickers]
        sector_map = dict(zip(today_df["ticker"], today_df.get("sector", ""))) if "sector" in today_df.columns else {}
        try:
            corr_matrix = compute_correlation_matrix(df)
        except Exception:
            corr_matrix = pd.DataFrame()

        filtered = filter_candidates_by_risk(
            candidates=signals,
            open_positions=open_positions,
            correlation_matrix=corr_matrix,
            sector_map=sector_map,
            config=self.config,
        )

        if filtered.empty:
            logger.info("All signals rejected by risk filters")
            return []

        # Available slots
        max_pos = self.config["universe"]["max_positions"]
        slots = max_pos - len(held_tickers)
        candidates = filtered.head(slots)

        result = []
        for _, row in candidates.iterrows():
            atr = row.get("atr_14", row["close"] * 0.02)
            stop_atr = self.config["backtest"]["stop_loss_atr"]
            target_atr = self.config["backtest"]["profit_target_atr"]
            hold_days = self.config["backtest"]["hold_days"]

            result.append({
                "ticker": row["ticker"],
                "price": row["close"],
                "ml_score": row.get("ml_score", 0),
                "atr": atr,
                "stop_price": round(row["close"] - atr * stop_atr, 2),
                "target_price": round(row["close"] + atr * target_atr, 2),
                "exit_date": (date.today() + timedelta(days=hold_days + 2)).strftime("%Y-%m-%d"),
            })

        logger.info(f"Signals ready: {len(result)} candidates — "
                    f"{', '.join(r['ticker'] for r in result)}")
        return result

    # ── Order Placement ──────────────────────────────────────────────

    def compute_position_size(self, price: float, atr: float) -> int:
        """Compute shares to buy based on vol-scaled sizing."""
        from utils.risk import compute_volatility_scaled_size
        balance = self.get_account_balance()
        if balance <= 0:
            balance = self.config["backtest"]["initial_capital"]

        atr_pct = atr / price if price > 0 else 0.02
        size_usd = compute_volatility_scaled_size(atr_pct, balance, self.config)
        shares = int(size_usd / price)
        return max(shares, 0)

    def place_bracket_entry(self, signal: dict) -> bool:
        """
        Place a bracket order: BUY parent + SELL stop + SELL target.
        Uses IBKR's parent-child OCA structure so one exit cancels the other.
        Stop and target are GTC to persist across days.
        """
        ticker = signal["ticker"]
        price = signal["price"]
        stop_price = signal["stop_price"]
        target_price = signal["target_price"]

        shares = self.compute_position_size(price, signal["atr"])
        if shares <= 0:
            logger.warning(f"{ticker}: Position size is 0 shares — skipping "
                           f"(price=${price:.2f}, account too small?)")
            return False

        contract = Stock(ticker, "SMART", "USD")
        self.ib.qualifyContracts(contract)

        # Parent: market buy
        parent = MarketOrder("BUY", shares)
        parent.transmit = False  # Don't send until children attached
        parent.tif = "DAY"

        # Child 1: stop loss (GTC — persists across days)
        stop = StopOrder("SELL", shares, stop_price)
        stop.parentId = parent.orderId
        stop.transmit = False
        stop.tif = "GTC"

        # Child 2: take profit (GTC)
        target = LimitOrder("SELL", shares, target_price)
        target.parentId = parent.orderId
        target.transmit = True  # This triggers the whole bracket
        target.tif = "GTC"

        try:
            parent_trade = self.ib.placeOrder(contract, parent)
            self.ib.sleep(1)
            stop_trade = self.ib.placeOrder(contract, stop)
            self.ib.sleep(1)
            target_trade = self.ib.placeOrder(contract, target)
            self.ib.sleep(2)

            # Verify parent accepted
            if parent_trade.orderStatus.status in ("Cancelled", "Inactive"):
                logger.error(f"{ticker}: Bracket REJECTED — {parent_trade.orderStatus.status}")
                return False

            logger.info(f"BRACKET PLACED: {ticker} | BUY {shares} shares "
                        f"@ market | stop=${stop_price:.2f} | "
                        f"target=${target_price:.2f} | exit={signal['exit_date']}")

            self.pending_orders[ticker] = {
                "parent_id": parent.orderId,
                "stop_id": stop.orderId,
                "target_id": target.orderId,
                "parent_trade": parent_trade,
                "stop_trade": stop_trade,
                "target_trade": target_trade,
                "contract": contract,
                "signal": signal,
                "shares": shares,
                "order_time": datetime.now(),
            }
            return True

        except Exception as e:
            logger.error(f"{ticker}: Bracket placement failed: {e}")
            return False

    def place_all_entries(self):
        """Place bracket orders for all today's signals."""
        if not self.today_signals:
            logger.info("No signals to place")
            return

        if not self.ensure_connected():
            logger.error("Cannot place orders — IBKR not connected")
            return

        placed = 0
        for signal in self.today_signals:
            if self.place_bracket_entry(signal):
                placed += 1
                self.ib.sleep(2)  # Brief pause between orders

        logger.info(f"Placed {placed}/{len(self.today_signals)} bracket orders")

        # Alert
        if placed > 0:
            tickers = [s["ticker"] for s in self.today_signals[:placed]]
            self._send_alert(
                f"SWING ENTRY: {placed} bracket orders placed\n" +
                "\n".join(f"  BUY {s['ticker']} ~${s['price']:.2f} | "
                          f"stop=${s['stop_price']:.2f} | target=${s['target_price']:.2f}"
                          for s in self.today_signals[:placed])
            )

    # ── Position Monitoring ──────────────────────────────────────────

    def check_entry_fills(self):
        """Check if pending buy orders have filled."""
        for ticker, order_info in list(self.pending_orders.items()):
            trade = order_info["parent_trade"]
            status = trade.orderStatus.status

            if status == "Filled":
                fill_price = trade.orderStatus.avgFillPrice
                if not fill_price or fill_price <= 0:
                    fill_price = order_info["signal"]["price"]

                shares = order_info["shares"]
                signal = order_info["signal"]

                # Record in positions DB
                from live.positions import add_position
                size_usd = fill_price * shares
                add_position(
                    ticker=ticker,
                    entry_price=fill_price,
                    size_usd=size_usd,
                    stop_price=signal["stop_price"],
                    target_price=signal["target_price"],
                    exit_date=signal["exit_date"],
                )

                self.active_positions[ticker] = {
                    "ticker": ticker,
                    "entry_price": fill_price,
                    "shares": shares,
                    "size_usd": size_usd,
                    "stop_price": signal["stop_price"],
                    "target_price": signal["target_price"],
                    "exit_date": signal["exit_date"],
                    "entry_date": str(date.today()),
                    "stop_id": order_info["stop_id"],
                    "target_id": order_info["target_id"],
                    "stop_trade": order_info["stop_trade"],
                    "target_trade": order_info["target_trade"],
                    "contract": order_info["contract"],
                }

                del self.pending_orders[ticker]
                logger.info(f"FILL: BUY {shares} {ticker} @ ${fill_price:.2f} | "
                            f"size=${size_usd:.0f} | stop=${signal['stop_price']:.2f} | "
                            f"target=${signal['target_price']:.2f}")

                self._send_alert(
                    f"SWING FILL: BUY {shares} {ticker} @ ${fill_price:.2f}\n"
                    f"  Stop: ${signal['stop_price']:.2f} | "
                    f"Target: ${signal['target_price']:.2f} | "
                    f"Exit: {signal['exit_date']}"
                )

            elif status in ("Cancelled", "Inactive"):
                logger.warning(f"{ticker}: Buy order {status}")
                del self.pending_orders[ticker]

    def check_exit_fills(self):
        """Check if any stop or target orders have filled."""
        for ticker, pos in list(self.active_positions.items()):
            stop_filled = False
            target_filled = False
            exit_price = 0
            reason = ""

            # Check stop fill
            try:
                stop_status = pos["stop_trade"].orderStatus.status
                if stop_status == "Filled":
                    stop_filled = True
                    exit_price = pos["stop_trade"].orderStatus.avgFillPrice or pos["stop_price"]
                    reason = "stop"
            except Exception:
                pass

            # Check target fill
            try:
                target_status = pos["target_trade"].orderStatus.status
                if target_status == "Filled":
                    target_filled = True
                    exit_price = pos["target_trade"].orderStatus.avgFillPrice or pos["target_price"]
                    reason = "target"
            except Exception:
                pass

            if stop_filled or target_filled:
                self._record_exit(ticker, exit_price, reason)

    def check_time_exits(self):
        """Check for positions that have exceeded hold_days."""
        hold_days = self.config["backtest"]["hold_days"]
        early_exit_days = self.config["backtest"].get("early_exit_days", 4)
        early_exit_threshold = self.config["backtest"].get("early_exit_threshold", -0.04)

        for ticker, pos in list(self.active_positions.items()):
            entry_date = datetime.strptime(pos["entry_date"], "%Y-%m-%d").date()
            days_held = (date.today() - entry_date).days

            # Get current price
            try:
                current = yf.download(ticker, period="1d", progress=False)
                if current.empty:
                    continue
                if isinstance(current.columns, pd.MultiIndex):
                    current.columns = [c[0] for c in current.columns]
                current_price = float(current["Close"].iloc[-1])
            except Exception:
                continue

            unrealized_pct = (current_price / pos["entry_price"] - 1)

            # Early exit: held >= N days AND losing > threshold
            if days_held >= early_exit_days and unrealized_pct < early_exit_threshold:
                logger.info(f"{ticker}: EARLY EXIT — {days_held}d held, "
                            f"{unrealized_pct:.1%} return < {early_exit_threshold:.1%} threshold")
                self._force_close(ticker, "early_exit")
                continue

            # Time exit: held >= hold_days
            if days_held >= hold_days:
                logger.info(f"{ticker}: TIME EXIT — {days_held}d held (limit={hold_days}d)")
                self._force_close(ticker, "time_exit")
                continue

    def _force_close(self, ticker: str, reason: str):
        """Cancel bracket orders and market sell."""
        pos = self.active_positions.get(ticker)
        if not pos:
            return

        contract = pos.get("contract")
        if not contract:
            contract = Stock(ticker, "SMART", "USD")
            self.ib.qualifyContracts(contract)

        # Cancel remaining bracket orders
        try:
            for trade in self.ib.openTrades():
                if (trade.contract.symbol == ticker and
                        trade.order.action == "SELL"):
                    self.ib.cancelOrder(trade.order)
                    self.ib.sleep(1)
        except Exception as e:
            logger.warning(f"{ticker}: Error cancelling brackets: {e}")

        # Market sell
        try:
            sell = MarketOrder("SELL", pos["shares"])
            sell.tif = "DAY"
            sell_trade = self.ib.placeOrder(contract, sell)
            self.ib.sleep(3)

            exit_price = sell_trade.orderStatus.avgFillPrice
            if not exit_price or exit_price <= 0:
                # Get last price as fallback
                self.ib.reqMktData(contract)
                self.ib.sleep(2)
                ticker_data = self.ib.reqTickers(contract)
                if ticker_data:
                    exit_price = ticker_data[0].last or ticker_data[0].close or pos["entry_price"]

            self._record_exit(ticker, exit_price, reason)

        except Exception as e:
            logger.error(f"{ticker}: Force close failed: {e}")

    def _record_exit(self, ticker: str, exit_price: float, reason: str):
        """Record a completed trade."""
        pos = self.active_positions.get(ticker)
        if not pos:
            return

        entry_price = pos["entry_price"]
        pnl_pct = (exit_price / entry_price - 1) * 100
        pnl_usd = (exit_price - entry_price) * pos["shares"]

        # Update positions DB
        from live.positions import close_position
        close_position(ticker, exit_price, reason)

        logger.info(f"EXIT: {ticker} @ ${exit_price:.2f} ({reason}) | "
                    f"P&L: {pnl_pct:+.1f}% (${pnl_usd:+.2f}) | "
                    f"Entry: ${entry_price:.2f}")

        self._send_alert(
            f"SWING EXIT: {ticker} @ ${exit_price:.2f} ({reason})\n"
            f"  P&L: {pnl_pct:+.1f}% (${pnl_usd:+.2f})\n"
            f"  Entry: ${entry_price:.2f} | Shares: {pos['shares']}"
        )

        del self.active_positions[ticker]

    # ── Sync with IBKR ───────────────────────────────────────────────

    def sync_positions(self):
        """
        Sync internal state with IBKR positions and the positions DB.
        Called at startup to recover state after restarts.
        """
        # Load from DB
        from live.positions import get_open_positions
        db_positions = get_open_positions()

        # Load from IBKR
        ibkr_positions = self.get_ibkr_positions()

        # Merge: trust IBKR as source of truth for what we actually hold
        for db_pos in db_positions:
            ticker = db_pos["ticker"]
            if ticker in ("SH", "SDS", "SPXU"):  # Skip hedge positions
                continue

            if ticker in ibkr_positions:
                ibkr_pos = ibkr_positions[ticker]
                contract = Stock(ticker, "SMART", "USD")
                self.ib.qualifyContracts(contract)

                self.active_positions[ticker] = {
                    "ticker": ticker,
                    "entry_price": db_pos["entry_price"],
                    "shares": ibkr_pos["qty"],
                    "size_usd": db_pos["size_usd"],
                    "stop_price": db_pos["stop_price"],
                    "target_price": db_pos["target_price"],
                    "exit_date": db_pos.get("planned_exit", db_pos.get("exit_date", "")),
                    "entry_date": db_pos["entry_date"],
                    "contract": contract,
                    # Bracket order IDs unknown after restart — find them
                    "stop_trade": None,
                    "target_trade": None,
                    "stop_id": None,
                    "target_id": None,
                }

                # Try to find existing bracket orders
                for trade in self.ib.openTrades():
                    if trade.contract.symbol == ticker and trade.order.action == "SELL":
                        if isinstance(trade.order, StopOrder) or getattr(trade.order, 'auxPrice', 0) > 0:
                            self.active_positions[ticker]["stop_trade"] = trade
                            self.active_positions[ticker]["stop_id"] = trade.order.orderId
                        elif isinstance(trade.order, LimitOrder) or getattr(trade.order, 'lmtPrice', 0) > 0:
                            self.active_positions[ticker]["target_trade"] = trade
                            self.active_positions[ticker]["target_id"] = trade.order.orderId

                logger.info(f"Synced position: {ticker} | {ibkr_pos['qty']} shares "
                            f"@ ${db_pos['entry_price']:.2f}")
            else:
                # In DB but not in IBKR — may have been closed manually
                logger.warning(f"{ticker}: In DB but not in IBKR — may need manual cleanup")

        logger.info(f"Position sync complete: {len(self.active_positions)} active positions")

    # ── Daily Snapshot ───────────────────────────────────────────────

    def record_daily_snapshot(self):
        """Record portfolio state to DB."""
        from live.positions import record_snapshot

        balance = self.get_account_balance()
        invested = sum(
            pos["shares"] * pos["entry_price"]
            for pos in self.active_positions.values()
        )

        record_snapshot(
            cash=balance - invested,
            invested_value=invested,
            n_positions=len(self.active_positions),
            notes=f"swing_trader auto-snapshot",
        )

    # ── Alerts ───────────────────────────────────────────────────────

    def _send_alert(self, message: str):
        """Send Discord alert (best effort)."""
        try:
            from live.alerts import send_discord
            send_discord(message)
        except Exception as e:
            logger.warning(f"Alert failed: {e}")

    # ── Main Run Loop ────────────────────────────────────────────────

    def run(self, monitor_only: bool = False):
        """
        Main entry point. Runs the full daily cycle:
          1. Connect to IBKR
          2. Sync existing positions
          3. Generate signals (unless monitor_only)
          4. Wait for market open → place orders
          5. Monitor positions until market close
          6. Daily summary
        """
        logger.info("=" * 60)
        logger.info(f"  SWING TRADER — {date.today()}")
        logger.info(f"  Mode: {'PAPER' if self.paper else 'LIVE'}")
        logger.info("=" * 60)

        if not self.connect():
            logger.error("Cannot connect to IBKR. Is TWS running?")
            return

        try:
            # Get account balance
            balance = self.get_account_balance()
            logger.info(f"Account balance: ${balance:,.2f}")

            # Sync existing positions
            self.sync_positions()

            if not monitor_only:
                # Generate signals
                logger.info("Running signal pipeline...")
                self.today_signals = self.generate_signals()

                if self.today_signals:
                    logger.info(f"Waiting for market open to place {len(self.today_signals)} orders...")

            # ── Main loop ────────────────────────────────────────
            while True:
                if not self.ensure_connected():
                    logger.error("Lost IBKR connection, waiting 60s...")
                    time.sleep(60)
                    continue

                now = datetime.now().time()

                # Before market open — wait
                if now < SwingConfig.ENTRY_TIME:
                    wait_secs = (datetime.combine(date.today(), SwingConfig.ENTRY_TIME) -
                                 datetime.now()).seconds
                    if wait_secs > 120:
                        logger.info(f"Market opens in {wait_secs // 60}m — sleeping...")
                        self.ib.sleep(min(wait_secs - 60, 300))
                    else:
                        self.ib.sleep(10)
                    continue

                # Place orders at 9:31 (once)
                if (not self.orders_placed_today and
                        now >= SwingConfig.ENTRY_TIME and
                        now < SwingConfig.LAST_ENTRY and
                        self.today_signals):
                    logger.info("Market open — placing bracket orders...")
                    self.place_all_entries()
                    self.orders_placed_today = True

                # Check entry fills
                if self.pending_orders:
                    self.check_entry_fills()

                # Check exit fills (stop/target)
                if self.active_positions:
                    self.check_exit_fills()

                # Check time exits at 3:45 PM
                if now >= SwingConfig.CLOSE_CHECK and now < SwingConfig.MARKET_CLOSE:
                    self.check_time_exits()

                # Market closed — done for today
                if now >= SwingConfig.MARKET_CLOSE:
                    logger.info("Market closed — wrapping up")
                    break

                # Adaptive sleep
                if self.pending_orders or self.active_positions:
                    self.ib.sleep(SwingConfig.POSITION_CHECK_SECONDS)
                else:
                    self.ib.sleep(SwingConfig.IDLE_CHECK_SECONDS)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            self._send_alert(f"SWING TRADER ERROR: {e}")
        finally:
            # Daily summary
            self.record_daily_snapshot()

            n_pos = len(self.active_positions)
            balance = self.get_account_balance() if self.ib and self.ib.isConnected() else 0
            summary = (f"SWING TRADER — End of Day\n"
                       f"  Positions: {n_pos}\n"
                       f"  Balance: ${balance:,.2f}\n"
                       f"  Orders placed today: {self.orders_placed_today}")

            for ticker, pos in self.active_positions.items():
                days = (date.today() - datetime.strptime(pos["entry_date"], "%Y-%m-%d").date()).days
                summary += f"\n  {ticker}: day {days}/{self.config['backtest']['hold_days']}"

            logger.info(summary)
            self._send_alert(summary)
            self.disconnect()


# ═══════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════

def run_swing():
    """CLI entry point for swing trader."""
    paper = "--live" not in sys.argv
    monitor_only = "--monitor-only" in sys.argv

    if not paper:
        logger.warning("=" * 40)
        logger.warning("  LIVE TRADING MODE — REAL MONEY")
        logger.warning("=" * 40)

    trader = SwingTrader(paper=paper)
    trader.run(monitor_only=monitor_only)


if __name__ == "__main__":
    run_swing()
