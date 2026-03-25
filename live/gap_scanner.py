"""
live/gap_scanner.py
---------------------
Gap Scanner Day Trading Bot — v3 (Smart Entry).

Scans the market after open, finds the best gap-up stock, and enters
via VWAP pullback or pre-market high breakout with a trailing stop.

Strategy:
  - 9:35 AM ET: scan 200+ stocks for gap-ups 1.5-8% on 1.5x+ volume
  - Score by: relative volume (35%), short interest bonus (88% win on
    >10% SI), low float bonus, gap sweet spot (1.5-3% = 80% win)
  - Smart entry (waits for best price instead of buying immediately):
      a) VWAP pullback: price pulls back to VWAP and bounces → tight stop
      b) PM high breakout: price clears pre-market high → momentum confirmed
      c) Timeout: if neither triggers in 15 min, enter if gap still valid
  - Exit: trailing stop 0.2x ATR (locks in profits, 72% win rate)
  - Tracks: VWAP, pre-market high, float rotation, short interest
  - Dynamic position sizing: grows with wins, shrinks with losses

Usage:
    python live/gap_scanner.py --dry-run             # Scan only
    python live/gap_scanner.py --capital 2000         # First run
    python live/gap_scanner.py --live                 # Live trading
    python live/gap_scanner.py --deposit 1000         # Record deposit
"""

import sys
import asyncio
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, date, timedelta, time as dtime
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")

# Python 3.14 fix
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

from ib_insync import IB, Stock, MarketOrder, LimitOrder, StopOrder, Order

try:
    from live.discord_format import (
        fmt_entry, fmt_exit, fmt_day_complete, fmt_scan_results,
        fmt_gap_skip, fmt_error, fmt_abort,
    )
except ImportError:
    fmt_entry = fmt_exit = fmt_day_complete = fmt_scan_results = None
    fmt_gap_skip = fmt_error = fmt_abort = None

try:
    from live.trade_logger import TradeLogger
except ImportError:
    TradeLogger = None


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

class ScannerConfig:
    # Connection
    PAPER_PORT = 7497
    LIVE_PORT = 7496
    HOST = "127.0.0.1"
    CLIENT_ID = 30           # Separate from swing (10) and ORB (20)

    # Scanner filters (AGG: Gap 1.5-8%, 1.5x vol — best backtest config)
    # 365 trades, 60.8% win, 2.58 PF, +463% total, -10.3% max DD
    MIN_GAP_PCT = 1.5        # Sweet spot: 1.5-3% gaps = 80% win, $22/trade
    MAX_GAP_PCT = 8.0        # Maximum gap (big gaps underperform)
    MIN_RVOL = 1.5           # Higher is better: 4x+ = $44/trade, 79% win
    MIN_PRICE = 10.0         # No penny stocks
    MAX_PRICE = 500.0        # Stay within reasonable range
    MIN_DOLLAR_VOL = 5_000_000  # Minimum daily dollar volume
    MIN_ATR_PCT = 1.0        # Stock must move enough to trade
    DIRECTION = "long"       # "long" only for now

    # Trade parameters — trailing stop strategy
    # Research: 0.20x ATR trail = $18.13/trade, 71.9% win, PF 3.49
    #   Tighter trail locks in profits faster, higher win rate
    EXIT_MODE = "trailing"
    TRAIL_ATR_MULT = 0.20        # Tightened from 0.3 → 0.2 (research: +$2.66/trade)
    STOP_ATR_MULT = 1.0          # Initial hard stop at 1x ATR (backup)
    TARGET_ATR_MULT = 4.0        # Safety ceiling target (rarely hit)
    USE_GAP_STOP = True          # Initial stop at previous close

    # Entry mode — VWAP pullback + pre-market high breakout
    # Instead of buying at open, wait for a better entry:
    #   (a) Price pulls back to VWAP and bounces → tighter stop
    #   (b) Price breaks above pre-market high → momentum confirmed
    ENTRY_MODE = "smart"         # "smart" (VWAP/PM high) or "immediate"
    VWAP_BOUNCE_BARS = 2         # Bars price must hold above VWAP to confirm
    PM_HIGH_BUFFER_PCT = 0.10    # Buy 0.10% above pre-market high
    VWAP_STOP_BUFFER_PCT = 0.20  # Stop 0.20% below VWAP (much tighter)

    # Position sizing — dynamic, grows/shrinks with account
    SIZING_MODE = "dynamic"  # "fixed" = flat dollar amount, "dynamic" = % of account
    POSITION_SIZE_USD = 500  # Used when SIZING_MODE = "fixed"
    POSITION_PCT = 0.33      # Used when SIZING_MODE = "dynamic" (33% of account)
    MIN_POSITION_USD = 100   # Floor — never go below this even on drawdown
    MAX_POSITION_USD = 10000 # Ceiling — cap even on big accounts
    MAX_RISK_PCT = 0.02      # Risk 2% of account per trade

    # Timing (Eastern Time)
    SCAN_TIME = dtime(9, 35)       # First scan at 9:35 AM ET
    RESCAN_TIMES = [dtime(10, 0), dtime(10, 30), dtime(11, 0),
                    dtime(12, 0), dtime(13, 0), dtime(14, 0)]
    MARKET_OPEN = dtime(9, 30)
    ENTRY_WINDOW_END = dtime(9, 40)
    LAST_ENTRY = dtime(15, 0)       # Trade all day until 3:00 PM ET
    FORCE_EXIT = dtime(15, 55)
    MARKET_CLOSE = dtime(16, 0)

    # Monitoring
    CHECK_INTERVAL = 15      # Check every 15 seconds
    MAX_TRADES_PER_DAY = 5   # No hard limit — take what the market gives you

    # Candidate display
    SHOW_TOP_N = 5           # Show top 5 candidates in the scan

    # Settings file (editable from Lattice app)
    SETTINGS_FILE = Path("lattice/lattice_settings.json")

    @classmethod
    def load_settings(cls):
        """Load overrides from lattice_settings.json (set via the app)."""
        if not cls.SETTINGS_FILE.exists():
            return
        try:
            with open(cls.SETTINGS_FILE) as f:
                s = json.load(f)
            if "position_pct" in s:
                cls.POSITION_PCT = s["position_pct"]
            if "max_trades_per_day" in s:
                cls.MAX_TRADES_PER_DAY = s["max_trades_per_day"]
            if "trail_atr_mult" in s:
                cls.TRAIL_ATR_MULT = s["trail_atr_mult"]
            if "entry_mode" in s:
                cls.ENTRY_MODE = s["entry_mode"]
            if "min_gap_pct" in s:
                cls.MIN_GAP_PCT = s["min_gap_pct"]
            if "max_gap_pct" in s:
                cls.MAX_GAP_PCT = s["max_gap_pct"]
            if "min_rvol" in s:
                cls.MIN_RVOL = s["min_rvol"]
            if "last_entry_hour" in s and "last_entry_minute" in s:
                cls.LAST_ENTRY = dtime(s["last_entry_hour"],
                                       s["last_entry_minute"])
            logger.info(f"  Settings loaded from {cls.SETTINGS_FILE}")
        except Exception as e:
            logger.warning(f"  Could not load settings: {e}")


# ═══════════════════════════════════════════════════════════════════════
# Universe
# ═══════════════════════════════════════════════════════════════════════

def get_universe():
    """Returns the tradeable universe."""
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


# ═══════════════════════════════════════════════════════════════════════
# Scanner
# ═══════════════════════════════════════════════════════════════════════

def run_pre_market_scan():
    """
    Downloads latest data and finds today's best gap-up candidates.
    Must run AFTER 9:30 AM ET so today's open price exists.

    Two-step approach:
      1. Daily data (30d) → historical features (ATR, SMA50, avg volume)
      2. Intraday data (1d, 5m) → today's actual open price and volume

    Returns a DataFrame of scored candidates, best first.
    """
    tickers = get_universe()
    logger.info(f"Scanning {len(tickers)} stocks for gap-up candidates...")

    # ── Load short interest data (cached from research) ───────────────
    si_data = {}
    si_cache = Path("data/gap_scanner_cache/float_short_data.json")
    if si_cache.exists():
        import json
        with open(si_cache) as f:
            si_data = json.load(f)
        logger.info(f"  Loaded short interest data for {len(si_data)} tickers")
    else:
        logger.info("  No short interest cache — run research/advanced_selection_backtest.py to build it")

    # ── Step 1: Historical daily data for features ────────────────────
    logger.info("  Downloading historical daily data...")
    try:
        daily = yf.download(
            tickers, period="30d", auto_adjust=True,
            progress=False, threads=True,
        )
    except Exception as e:
        logger.error(f"Daily data download failed: {e}")
        return pd.DataFrame()

    if daily.empty:
        logger.error("No daily data returned")
        return pd.DataFrame()

    # ── Step 2: Today's intraday data for actual open ─────────────────
    logger.info("  Downloading today's intraday data...")
    try:
        intraday = yf.download(
            tickers, period="1d", interval="5m", auto_adjust=True,
            progress=False, threads=True, prepost=False,
        )
    except Exception as e:
        logger.error(f"Intraday data download failed: {e}")
        return pd.DataFrame()

    if intraday.empty:
        logger.error("No intraday data — market may not be open yet")
        return pd.DataFrame()

    # ── Reshape daily data ────────────────────────────────────────────
    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns.names = ["Field", "Ticker"]
        df_daily = daily.stack(level="Ticker", future_stack=True).reset_index()
        df_daily.columns = [c.lower() if isinstance(c, str) else c
                            for c in df_daily.columns]
        if "level_1" in df_daily.columns:
            df_daily = df_daily.rename(columns={"level_1": "ticker"})
    else:
        # Single ticker fallback
        df_daily = daily.reset_index()
        df_daily.columns = [c.lower() for c in df_daily.columns]

    df_daily["date"] = pd.to_datetime(df_daily["date"])
    df_daily = df_daily.dropna(subset=["close", "volume"])
    df_daily = df_daily[df_daily["close"] > 0].copy()

    # ── Extract today's open from intraday data ───────────────────────
    today_opens = {}
    today_volumes = {}

    if isinstance(intraday.columns, pd.MultiIndex):
        for ticker in tickers:
            try:
                if ticker in intraday["Open"].columns:
                    opens = intraday["Open"][ticker].dropna()
                    vols = intraday["Volume"][ticker].dropna()
                    if len(opens) > 0:
                        today_opens[ticker] = float(opens.iloc[0])
                        today_volumes[ticker] = float(vols.sum())
            except Exception:
                continue
    else:
        # Single ticker
        if "Open" in intraday.columns:
            opens = intraday["Open"].dropna()
            vols = intraday["Volume"].dropna()
            if len(opens) > 0 and len(tickers) == 1:
                today_opens[tickers[0]] = float(opens.iloc[0])
                today_volumes[tickers[0]] = float(vols.sum())

    logger.info(f"  Got today's open price for {len(today_opens)} tickers")

    if not today_opens:
        logger.error("Could not get any open prices — market may not be open")
        return pd.DataFrame()

    # ── Compute features per ticker ───────────────────────────────────
    candidates = []

    for ticker, g in df_daily.groupby("ticker"):
        g = g.sort_values("date")
        if len(g) < 21:
            continue
        if ticker not in today_opens:
            continue

        today_open = today_opens[ticker]
        today_vol = today_volumes.get(ticker, 0)

        # Get yesterday's close — careful: during market hours,
        # yfinance daily data may include today as the latest row
        today_date = pd.Timestamp(date.today())
        latest_date = g.iloc[-1]["date"]
        if hasattr(latest_date, "date"):
            latest_date = pd.Timestamp(latest_date.date())
        else:
            latest_date = pd.Timestamp(latest_date)

        if latest_date >= today_date and len(g) >= 2:
            # Latest row is today → use second-to-last for prev close
            prev_close = g.iloc[-2]["close"]
        else:
            # Latest row is yesterday → use it directly
            prev_close = g.iloc[-1]["close"]

        # Gap: today's actual open vs yesterday's close
        gap_pct = (today_open - prev_close) / prev_close * 100

        # Historical data excluding today's partial bar
        if latest_date >= today_date and len(g) >= 2:
            hist = g.iloc[:-1]  # Exclude today
        else:
            hist = g

        # Volume ratio (today's volume so far vs 20-day average)
        vol_sma20 = hist["volume"].iloc[-20:].mean()
        # Project full-day volume from first 5 min (rough estimate)
        rvol = (today_vol * 78) / vol_sma20 if vol_sma20 > 0 else 0

        # More conservative: compare raw partial volume
        rvol_raw = today_vol / vol_sma20 if vol_sma20 > 0 else 0
        # Scale up early volume — if first bars are already hot, it's notable
        rvol = max(rvol_raw * 10, rvol_raw)

        # ATR (14-day) from historical data only
        g_recent = hist.tail(15).copy()
        tr_data = pd.DataFrame({
            "hl": g_recent["high"] - g_recent["low"],
            "hc": (g_recent["high"] - g_recent["close"].shift(1)).abs(),
            "lc": (g_recent["low"] - g_recent["close"].shift(1)).abs(),
        })
        g_recent = g_recent.copy()
        g_recent["tr"] = tr_data.max(axis=1)
        atr_14 = g_recent["tr"].iloc[-14:].mean()
        atr_pct = atr_14 / prev_close * 100

        # Momentum (5-day return into yesterday)
        mom_5d = (prev_close / hist.iloc[-6]["close"] - 1) * 100 if len(hist) >= 6 else 0

        # Trend (above 50 SMA)
        sma_50 = hist["close"].iloc[-50:].mean() if len(hist) >= 50 else hist["close"].mean()
        above_sma50 = 1 if prev_close > sma_50 else 0

        # Dollar volume (estimated from historical avg)
        dollar_vol = prev_close * vol_sma20

        # Short interest and float (from cached data)
        ticker_si = si_data.get(ticker, {})
        si_pct = (ticker_si.get("short_pct_float") or 0)
        if isinstance(si_pct, (int, float)) and si_pct > 0:
            si_pct = si_pct * 100  # Convert to percentage
        else:
            si_pct = 0
        float_shares = ticker_si.get("float_shares") or 0

        candidates.append({
            "ticker": ticker,
            "date": str(date.today()),
            "open": today_open,
            "prev_close": prev_close,
            "close": prev_close,
            "high": today_open,
            "low": today_open,
            "gap_pct": gap_pct,
            "rvol": rvol,
            "atr_14": atr_14,
            "atr_pct": atr_pct,
            "mom_5d": mom_5d,
            "above_sma50": above_sma50,
            "dollar_volume": dollar_vol,
            "volume": today_vol,
            "si_pct": si_pct,
            "float_shares": float_shares,
        })

    cdf = pd.DataFrame(candidates)
    if cdf.empty:
        logger.warning("No candidates found")
        return cdf

    logger.info(f"  Computed features for {len(cdf)} tickers")
    logger.info(f"  Gap range: {cdf['gap_pct'].min():+.1f}% to "
                f"{cdf['gap_pct'].max():+.1f}%")
    gappers = cdf[cdf["gap_pct"] >= ScannerConfig.MIN_GAP_PCT]
    logger.info(f"  Stocks gapping >{ScannerConfig.MIN_GAP_PCT}%: {len(gappers)}")

    # ── Apply scanner filters ────────────────────────────────────────
    cdf = cdf[cdf["prev_close"] >= ScannerConfig.MIN_PRICE]
    cdf = cdf[cdf["prev_close"] <= ScannerConfig.MAX_PRICE]
    cdf = cdf[cdf["dollar_volume"] >= ScannerConfig.MIN_DOLLAR_VOL]
    cdf = cdf[cdf["atr_pct"] >= ScannerConfig.MIN_ATR_PCT]
    cdf = cdf[cdf["gap_pct"] >= ScannerConfig.MIN_GAP_PCT]
    cdf = cdf[cdf["gap_pct"] <= ScannerConfig.MAX_GAP_PCT]
    # Relaxed volume filter for early morning (partial day volume)
    cdf = cdf[cdf["rvol"] >= ScannerConfig.MIN_RVOL * 0.5]

    if cdf.empty:
        logger.info("No stocks passed the scanner filters today")
        return cdf

    # ── Check unusual options activity (call volume vs open interest) ─
    logger.info(f"  Checking options flow for {len(cdf)} candidates...")
    cdf["call_vol_oi_ratio"] = 0.0
    cdf["call_put_ratio"] = 0.0
    cdf["options_signal"] = ""

    for idx, row in cdf.iterrows():
        try:
            tk = yf.Ticker(row["ticker"])
            # Get the nearest expiration
            exps = tk.options
            if not exps:
                continue

            # Check the next 1-2 expirations (near-term = most signal)
            total_call_vol = 0
            total_call_oi = 0
            total_put_vol = 0

            for exp in exps[:2]:
                chain = tk.option_chain(exp)
                if chain and chain.calls is not None and not chain.calls.empty:
                    total_call_vol += chain.calls["volume"].sum()
                    total_call_oi += chain.calls["openInterest"].sum()
                if chain and chain.puts is not None and not chain.puts.empty:
                    total_put_vol += chain.puts["volume"].sum()

            # Call volume / open interest ratio
            # > 1.0 means more contracts traded today than existed yesterday
            # That's new positions being opened = unusual activity
            if total_call_oi > 0:
                ratio = total_call_vol / total_call_oi
                cdf.at[idx, "call_vol_oi_ratio"] = round(ratio, 2)

                # Tag unusual activity
                if ratio >= 3.0:
                    cdf.at[idx, "options_signal"] = "VERY UNUSUAL"
                elif ratio >= 1.5:
                    cdf.at[idx, "options_signal"] = "unusual"
                elif ratio >= 0.8:
                    cdf.at[idx, "options_signal"] = "active"

            # Call/put ratio (bullish bias)
            if total_put_vol > 0:
                cdf.at[idx, "call_put_ratio"] = round(
                    total_call_vol / total_put_vol, 2)
            else:
                cdf.at[idx, "call_put_ratio"] = 0

        except Exception:
            continue

    unusual_count = len(cdf[cdf["call_vol_oi_ratio"] >= 1.5])
    if unusual_count > 0:
        logger.info(f"  Unusual options activity: {unusual_count} stocks")
    else:
        logger.info(f"  No unusual options activity detected")

    # ── Score candidates ─────────────────────────────────────────────
    # Research-backed scoring (advanced_selection_backtest.py):
    #   - High SI (>10%): 88% win, $37.48/trade → big bonus
    #   - RVol 4x+: 79% win, $44.67/trade → heavy weight
    #   - Gap 1.5-3%: 80% win, $22/trade → sweet spot bonus
    #   - Trend aligned: 66% win vs 43% → solid signal
    #   - Volume acceleration: best scoring model overall

    # Base score
    cdf["score"] = (
        cdf["rvol"].clip(upper=10) * 0.35 +          # Volume is king
        cdf["gap_pct"] * 0.20 +                       # Gap size
        cdf["mom_5d"].clip(-5, 10) * 0.10 +            # Prior momentum
        cdf["above_sma50"] * 0.10 * 5 +                # Trend aligned
        (cdf["atr_pct"] * 0.10).clip(upper=1.5)        # Moves enough
    )

    # Short interest bonus (biggest edge in the data)
    cdf["score"] += cdf["si_pct"].apply(
        lambda x: 3.0 if x >= 10 else (1.5 if x >= 5 else 0)
    )

    # Low float bonus
    cdf["score"] += cdf["float_shares"].apply(
        lambda x: 2.0 if 0 < x < 50_000_000 else
                  (0.5 if 0 < x < 200_000_000 else 0)
    )

    # Gap sweet spot bonus (1.5-3% outperforms bigger gaps)
    cdf["score"] += cdf["gap_pct"].apply(
        lambda x: 1.0 if 1.5 <= x <= 3.0 else 0
    )

    # Unusual options activity bonus (smart money signal)
    # Call vol/OI > 3.0 = very unusual → big bonus
    # Call vol/OI > 1.5 = unusual → moderate bonus
    if "call_vol_oi_ratio" in cdf.columns:
        cdf["score"] += cdf["call_vol_oi_ratio"].apply(
            lambda x: 3.0 if x >= 3.0 else (1.5 if x >= 1.5 else
                      (0.5 if x >= 0.8 else 0))
        )

    cdf = cdf.sort_values("score", ascending=False).reset_index(drop=True)

    # Log top candidates
    logger.info(f"Found {len(cdf)} candidates. Top {ScannerConfig.SHOW_TOP_N}:")
    for i, row in cdf.head(ScannerConfig.SHOW_TOP_N).iterrows():
        trend = "↑" if row["above_sma50"] else "↓"
        si_str = f"SI={row['si_pct']:.0f}%" if row["si_pct"] > 0 else ""
        opt_str = ""
        if row.get("options_signal"):
            opt_str = f"opts={row['options_signal']}({row['call_vol_oi_ratio']:.1f}x)"
        logger.info(
            f"  #{i+1} {row['ticker']:<6} gap={row['gap_pct']:+.1f}% "
            f"rvol={row['rvol']:.1f}x atr={row['atr_pct']:.1f}% "
            f"trend={trend} {si_str} {opt_str} "
            f"score={row['score']:.2f}"
        )

    return cdf


# ═══════════════════════════════════════════════════════════════════════
# Trading Bot
# ═══════════════════════════════════════════════════════════════════════

class GapScannerBot:
    ACCOUNT_FILE = Path("live/gap_scanner_account.json")

    def __init__(self, paper=True, position_size=None, dry_run=False,
                 starting_capital=None):
        self.paper = paper
        self.port = ScannerConfig.PAPER_PORT if paper else ScannerConfig.LIVE_PORT
        self.fixed_size = position_size  # Override from CLI (None = use dynamic)
        self.dry_run = dry_run
        self.ib = None

        # Load or initialize account state
        self.account = self._load_account(starting_capital)

        # Load settings from Lattice app (if they exist)
        self._load_settings()

        # Trade logger for ML training
        self.trade_logger = TradeLogger() if TradeLogger else None

        # Daily state
        self.candidates = None
        self.target_ticker = None
        self.target_data = None
        self.trade_taken = False
        self.entered_trade = False
        self.position = None
        self.daily_pnl = 0
        self.today = None

    # ── Account tracking (persists across sessions) ───────────────────

    def _load_account(self, starting_capital=None):
        """Loads account state from disk, or initializes fresh."""
        if self.ACCOUNT_FILE.exists():
            import json
            with open(self.ACCOUNT_FILE) as f:
                acct = json.load(f)
            logger.info(f"Account loaded: ${acct['balance']:,.2f} "
                        f"({acct['total_trades']} trades, "
                        f"{acct['wins']}W/{acct['losses']}L)")
            return acct

        # First run — initialize
        balance = starting_capital or ScannerConfig.POSITION_SIZE_USD * 2
        acct = {
            "starting_capital": balance,
            "balance": balance,
            "peak_balance": balance,
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl_usd": 0.0,
            "total_pnl_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "trade_history": [],
        }
        self._save_account(acct)
        logger.info(f"Account initialized: ${balance:,.2f}")
        return acct

    def _save_account(self, acct=None):
        """Saves account state to disk."""
        import json
        if acct is None:
            acct = self.account
        self.ACCOUNT_FILE.parent.mkdir(exist_ok=True)
        with open(self.ACCOUNT_FILE, "w") as f:
            json.dump(acct, f, indent=2, default=str)

    SETTINGS_FILE = Path("lattice/bot_settings.json")

    def _load_settings(self):
        """
        Loads settings from Lattice app config file.
        These override ScannerConfig defaults so the app can
        adjust position sizing, trailing stop, etc. without editing code.
        """
        if not self.SETTINGS_FILE.exists():
            return

        try:
            with open(self.SETTINGS_FILE) as f:
                settings = json.load(f)

            # Map settings to ScannerConfig
            mapping = {
                "position_pct": "POSITION_PCT",
                "trail_atr_mult": "TRAIL_ATR_MULT",
                "max_trades_per_day": "MAX_TRADES_PER_DAY",
                "min_gap_pct": "MIN_GAP_PCT",
                "max_gap_pct": "MAX_GAP_PCT",
                "min_rvol": "MIN_RVOL",
                "entry_mode": "ENTRY_MODE",
            }

            changed = []
            for key, attr in mapping.items():
                if key in settings:
                    old_val = getattr(ScannerConfig, attr)
                    new_val = settings[key]
                    if old_val != new_val:
                        setattr(ScannerConfig, attr, new_val)
                        changed.append(f"{attr}: {old_val} -> {new_val}")

            if changed:
                logger.info(f"  Loaded {len(changed)} settings from Lattice app:")
                for c in changed:
                    logger.info(f"    {c}")
        except Exception as e:
            logger.warning(f"  Could not load settings: {e}")

    def record_trade(self, ticker, direction, entry, exit_price, qty, reason):
        """Records a completed trade. Syncs balance from IBKR (source of truth)."""
        if direction == "long":
            pnl_pct = (exit_price - entry) / entry * 100
        else:
            pnl_pct = (entry - exit_price) / entry * 100

        pnl_usd = (exit_price - entry) * qty if direction == "long" else \
                   (entry - exit_price) * qty

        self.account["total_trades"] += 1
        if pnl_pct > 0:
            self.account["wins"] += 1
        else:
            self.account["losses"] += 1

        # Sync real balance from IBKR instead of calculating ourselves
        if self.ib and self.ib.isConnected():
            try:
                self.ib.sleep(1)
                summary = self.ib.accountSummary()
                for item in summary:
                    if item.tag == "NetLiquidation" and \
                       item.currency == "USD":
                        self.account["balance"] = float(item.value)
                        break
            except Exception:
                # Fallback to calculated balance
                self.account["balance"] += pnl_usd
        else:
            self.account["balance"] += pnl_usd

        self.account["total_pnl_usd"] = (
            self.account["balance"] - self.account["starting_capital"]
        )

        # Track peak and drawdown
        if self.account["balance"] > self.account["peak_balance"]:
            self.account["peak_balance"] = self.account["balance"]
        dd = ((self.account["balance"] - self.account["peak_balance"]) /
              self.account["peak_balance"] * 100)
        if dd < self.account["max_drawdown_pct"]:
            self.account["max_drawdown_pct"] = dd

        # Update total return
        self.account["total_pnl_pct"] = (
            (self.account["balance"] / self.account["starting_capital"]) - 1
        ) * 100

        # Append to history (keep last 100 trades)
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

        self.daily_pnl = pnl_pct
        win_rate = (self.account["wins"] / self.account["total_trades"] * 100
                    if self.account["total_trades"] > 0 else 0)

        logger.info(f"TRADE RECORDED: {ticker} {pnl_pct:+.2f}% (${pnl_usd:+.2f})")
        logger.info(f"  Account (IBKR): ${self.account['balance']:,.2f} | "
                    f"Record: {self.account['wins']}W/{self.account['losses']}L "
                    f"({win_rate:.0f}%) | "
                    f"Total P&L: ${self.account['total_pnl_usd']:+,.2f}")

    def get_position_size(self):
        """
        Calculates position size based on current account balance.
        Grows with wins, shrinks with losses — automatic compounding.
        """
        # CLI override: fixed dollar amount
        if self.fixed_size is not None:
            return self.fixed_size

        if ScannerConfig.SIZING_MODE == "fixed":
            return ScannerConfig.POSITION_SIZE_USD

        # Dynamic: percentage of current balance
        size = self.account["balance"] * ScannerConfig.POSITION_PCT

        # Apply floor and ceiling
        size = max(size, ScannerConfig.MIN_POSITION_USD)
        size = min(size, ScannerConfig.MAX_POSITION_USD)

        return round(size, 2)

    def _advance_to_next_candidate(self, dummy=None):
        """
        When a candidate's gap fades, move to the next one.
        If no more candidates, sit out.
        """
        if self.candidates is None or self.candidates.empty:
            self.trade_taken = True
            return

        self.next_candidate_idx += 1
        if self.next_candidate_idx >= len(self.candidates):
            logger.info("No more candidates to try. Sitting out.")
            self.trade_taken = True
            return

        # Pick next candidate
        next_row = self.candidates.iloc[self.next_candidate_idx]
        self.target_ticker = next_row["ticker"]
        self.target_data = next_row

        # Reset entry state for new candidate
        self.pm_high = None
        self.vwap = None
        self.vwap_cum_vol = 0
        self.vwap_cum_pv = 0
        self.bars_above_vwap = 0
        self.entry_signal = None
        self.float_rotation = 0
        self.intraday_high = 0
        self.intraday_volume = 0

        logger.info(f"Next candidate: #{self.next_candidate_idx + 1} "
                    f"{self.target_ticker} "
                    f"(gap={next_row['gap_pct']:+.1f}%, "
                    f"rvol={next_row['rvol']:.1f}x)")

        # Qualify new contract
        if not self.dry_run:
            try:
                self._contract = self.get_contract(self.target_ticker)
            except Exception as e:
                logger.error(f"Failed to qualify {self.target_ticker}: {e}")
                self._advance_to_next_candidate()

    def _prepare_for_next_trade(self):
        """
        After a trade closes, reset entry state and advance to the
        next candidate so the bot can take another trade.
        """
        self.trades_today += 1
        now = self.get_current_time_et()

        # Check limits
        if self.trades_today >= ScannerConfig.MAX_TRADES_PER_DAY:
            logger.info(f"Max trades reached ({self.trades_today}). Done.")
            self.trade_taken = True
            return

        if now.time() >= ScannerConfig.LAST_ENTRY:
            logger.info("Past entry window — no more trades today.")
            self.trade_taken = True
            return

        # Reset entry state
        self.trade_taken = False
        self.position = None
        self.pm_high = None
        self.vwap = None
        self.vwap_cum_vol = 0
        self.vwap_cum_pv = 0
        self.bars_above_vwap = 0
        self.entry_signal = None
        self.float_rotation = 0
        self.intraday_high = 0
        self.intraday_volume = 0

        # Fresh re-scan instead of walking stale list
        logger.info(f"Trade #{self.trades_today} done. "
                    f"Re-scanning for trade #{self.trades_today + 1}...")

        # Track which tickers we already traded today (don't repeat)
        traded_tickers = [t.get("ticker") for t in
                          self.account.get("trade_history", [])
                          if t.get("date") == str(date.today())]

        self.candidates = run_pre_market_scan()

        if self.candidates is not None and not self.candidates.empty:
            # Filter out stocks we already traded today
            self.candidates = self.candidates[
                ~self.candidates["ticker"].isin(traded_tickers)
            ].reset_index(drop=True)

        if self.candidates is None or self.candidates.empty:
            logger.info("No new candidates on re-scan. Done for today.")
            self.trade_taken = True
            return

        self.next_candidate_idx = 0
        top = self.candidates.iloc[0]
        self.target_ticker = top["ticker"]
        self.target_data = top

        logger.info(f"New target: {self.target_ticker} "
                    f"(gap={top['gap_pct']:+.1f}%, "
                    f"rvol={top['rvol']:.1f}x, "
                    f"score={top['score']:.2f})")

        # Qualify and fetch PM high
        if not self.dry_run:
            try:
                self._contract = self.get_contract(self.target_ticker)
                self.pm_high = self.fetch_premarket_high(self.target_ticker)
            except Exception as e:
                logger.error(f"Failed to set up {self.target_ticker}: {e}")
                self.trade_taken = True

        try:
            from live.alerts import send_discord
            send_discord(
                f"Trade #{self.trades_today} complete. "
                f"Re-scanned market — new target: {self.target_ticker} "
                f"(gap={top['gap_pct']:+.1f}%, "
                f"rvol={top['rvol']:.1f}x)"
            )
        except Exception:
            pass

    # ── VWAP, Pre-Market High, Float Rotation, Entry Signals ─────────

    def fetch_premarket_high(self, ticker):
        """
        Gets the pre-market high from yfinance intraday data.
        Pre-market = 4:00-9:30 AM ET.
        """
        try:
            intra = yf.download(
                ticker, period="1d", interval="5m",
                auto_adjust=True, progress=False, prepost=True,
            )
            if intra.empty:
                return None

            # Flatten MultiIndex if needed
            if isinstance(intra.columns, pd.MultiIndex):
                intra.columns = intra.columns.get_level_values(0)

            # Filter to pre-market (before 9:30 ET)
            try:
                import zoneinfo
                et = zoneinfo.ZoneInfo("America/New_York")
                intra.index = intra.index.tz_convert(et)
                pre_market = intra[intra.index.time < dtime(9, 30)]
            except Exception:
                # Fallback: take first few bars as pre-market
                pre_market = intra.head(6)

            if pre_market.empty:
                return None

            pm_high = float(pre_market["High"].max())
            pm_vol = int(pre_market["Volume"].sum())
            logger.info(f"  Pre-market high: ${pm_high:.2f} "
                        f"(vol: {pm_vol:,})")
            return pm_high
        except Exception as e:
            logger.warning(f"  Pre-market high fetch failed: {e}")
            return None

    def update_vwap(self, price, volume):
        """
        Updates running VWAP calculation with a new bar.
        VWAP = cumulative(price * volume) / cumulative(volume)
        """
        if volume <= 0:
            return self.vwap

        self.vwap_cum_pv += price * volume
        self.vwap_cum_vol += volume
        self.vwap = self.vwap_cum_pv / self.vwap_cum_vol
        return self.vwap

    def update_float_rotation(self, volume):
        """
        Tracks how many times the float has turned over today.
        High rotation = extreme momentum.
        """
        self.intraday_volume += volume
        float_shares = self.target_data.get("float_shares", 0) if \
            self.target_data is not None else 0
        if float_shares and float_shares > 0:
            self.float_rotation = self.intraday_volume / float_shares
        return self.float_rotation

    def check_entry_signal(self, contract):
        """
        Monitors price for entry signals:
          (a) VWAP pullback: price at VWAP and bouncing
          (b) PM high breakout: price broke above pre-market high
          (c) Timeout: after 15 min, buy if gap still valid

        Returns: ("signal_type", price) or (None, None)
        """
        price = self.get_current_price(contract)
        if price is None:
            return None, None

        # Track intraday high
        if price > self.intraday_high:
            self.intraday_high = price

        # Estimate bar volume for VWAP
        avg_bar_vol = max(1, (self.target_data.get("volume", 100000) or
                              100000) / 78)
        self.update_vwap(price, avg_bar_vol)
        self.update_float_rotation(avg_bar_vol)

        if self.vwap is None:
            return None, None

        vwap = self.vwap

        # ── Signal A: VWAP Pullback ──────────────────────────────────
        vwap_zone = vwap * 1.003  # Within 0.3% of VWAP
        if vwap < price <= vwap_zone:
            self.bars_above_vwap += 1
        elif price > vwap_zone:
            self.bars_above_vwap = 0
        else:
            self.bars_above_vwap = 0

        if self.bars_above_vwap >= ScannerConfig.VWAP_BOUNCE_BARS:
            logger.info(f"  VWAP PULLBACK: ${price:.2f} "
                        f"(VWAP=${vwap:.2f})")
            return "vwap_pullback", price

        # ── Signal B: Pre-Market High Breakout ───────────────────────
        if self.pm_high and self.pm_high > 0:
            breakout = self.pm_high * (
                1 + ScannerConfig.PM_HIGH_BUFFER_PCT / 100)
            if price > breakout:
                logger.info(f"  PM HIGH BREAKOUT: ${price:.2f} "
                            f"(PM high=${self.pm_high:.2f})")
                return "pm_breakout", price

        # ── Signal C: Timeout after 15 min ───────────────────────────
        now = self.get_current_time_et()
        scan_min = (ScannerConfig.SCAN_TIME.hour * 60 +
                    ScannerConfig.SCAN_TIME.minute)
        now_min = now.hour * 60 + now.minute
        if (now_min - scan_min) >= 15:
            if price > self.target_data["prev_close"]:
                logger.info(f"  TIMEOUT entry: ${price:.2f} "
                            f"(15 min elapsed, gap valid)")
                return "timeout", price

        return None, None

    def connect(self):
        """Connect to IBKR."""
        if self.dry_run:
            logger.info("DRY RUN mode — no IBKR connection")
            return True

        self.ib = IB()
        mode = "PAPER" if self.paper else "LIVE"
        try:
            self.ib.connect(ScannerConfig.HOST, self.port,
                            clientId=ScannerConfig.CLIENT_ID)
            logger.info(f"Connected to IBKR ({mode}) on port {self.port}")

            # Sync account balance from IBKR (source of truth)
            self.sync_account_from_ibkr()
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def sync_account_from_ibkr(self):
        """
        Pulls the real account balance from IBKR.
        IBKR is the source of truth — our JSON file just tracks history.
        """
        try:
            self.ib.sleep(1)
            summary = self.ib.accountSummary()
            for item in summary:
                if item.tag == "NetLiquidation" and item.currency == "USD":
                    ibkr_balance = float(item.value)
                    old_balance = self.account["balance"]

                    if abs(ibkr_balance - old_balance) > 0.50:
                        logger.info(f"  IBKR balance: ${ibkr_balance:,.2f} "
                                    f"(was ${old_balance:,.2f} in tracker)")
                        self.account["balance"] = ibkr_balance
                        if ibkr_balance > self.account["peak_balance"]:
                            self.account["peak_balance"] = ibkr_balance
                        self._save_account()
                    else:
                        logger.info(f"  IBKR balance: ${ibkr_balance:,.2f} (in sync)")
                    break
        except Exception as e:
            logger.warning(f"  Could not sync IBKR balance: {e}")

    def get_ibkr_fill_price(self, ticker, side="SLD"):
        """
        Gets the actual fill price from IBKR executions.
        Returns (price, qty) or (None, None).
        """
        try:
            fills = self.ib.fills()
            matching = [
                f for f in fills
                if f.contract.symbol == ticker and
                f.execution.side == side
            ]
            if matching:
                last = matching[-1]
                return last.execution.price, int(last.execution.shares)
            return None, None
        except Exception as e:
            logger.warning(f"  Could not query fills: {e}")
            return None, None

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

    def is_market_day(self):
        return self.get_current_time_et().weekday() < 5

    def reset_daily_state(self):
        self.candidates = None
        self.target_ticker = None
        self.target_data = None
        self.trade_taken = False
        self.entered_trade = False
        self.position = None
        self.daily_pnl = 0
        self.today = date.today()

        # Intraday tracking (VWAP, pre-market high, float rotation)
        self.pm_high = None
        self.vwap = None
        self.vwap_cum_vol = 0
        self.vwap_cum_pv = 0
        self.bars_above_vwap = 0
        self.entry_ready = False
        self.entry_signal = None
        self.float_rotation = 0
        self.intraday_high = 0
        self.intraday_volume = 0

        # Multi-trade tracking
        self.trades_today = 0
        self.next_candidate_idx = 0

        # Re-scan tracking
        self.scans_completed = 0
        self.last_scan_time = None

        logger.info("Daily state reset")

    def get_contract(self, ticker):
        """Returns a qualified contract for the given ticker."""
        contract = Stock(ticker, "SMART", "USD")
        self.ib.qualifyContracts(contract)
        return contract

    def get_current_price(self, contract):
        """Gets current price via snapshot."""
        try:
            self.ib.reqMktData(contract, "", False, False)
            self.ib.sleep(2)
            ticker_data = self.ib.ticker(contract)
            price = ticker_data.last
            if price != price:  # NaN
                price = ticker_data.close
            if price != price:
                price = (ticker_data.bid + ticker_data.ask) / 2 \
                    if ticker_data.bid == ticker_data.bid else None
            self.ib.cancelMktData(contract)
            return price
        except Exception as e:
            logger.warning(f"Price fetch failed: {e}")
            return None

    # ── Entry logic ───────────────────────────────────────────────────

    def enter_trade(self, contract):
        """
        Places a trailing stop order for the gap-and-go trade.

        Trailing stop (0.3x ATR):
          - As price rises, the stop rises with it
          - Locks in profits automatically
          - Never gives back more than 0.3x ATR from the high
          - 74% of trades exit via trail (vs 4% target hit with old bracket)
        """
        data = self.target_data
        ticker = self.target_ticker
        pos_size = self.get_position_size()
        qty = max(1, int(pos_size / data["open"]))

        entry_price = data["open"]
        atr = data["atr_14"]

        # Trailing distance = 0.3x ATR in dollars
        trail_amount = round(atr * ScannerConfig.TRAIL_ATR_MULT, 2)

        # Initial hard stop: VWAP stop (tighter) or gap stop (fallback)
        vwap_stop = data.get("vwap_stop")
        gap_stop = data["prev_close"] * 0.998
        atr_stop = entry_price - atr * ScannerConfig.STOP_ATR_MULT

        if vwap_stop and vwap_stop < entry_price:
            # VWAP pullback entry → use tighter VWAP-based stop
            hard_stop = round(vwap_stop, 2)
            stop_type = "vwap"
        elif ScannerConfig.USE_GAP_STOP:
            hard_stop = round(max(gap_stop, atr_stop), 2)
            stop_type = "gap"
        else:
            hard_stop = round(atr_stop, 2)
            stop_type = "atr"

        # Safety ceiling target (rarely hit, just a cap)
        target_price = round(entry_price + atr * ScannerConfig.TARGET_ATR_MULT, 2)

        if hard_stop >= entry_price:
            logger.error(f"Stop ${hard_stop} >= entry ${entry_price} — aborting")
            return False

        risk_per_share = entry_price - hard_stop
        risk_total = risk_per_share * qty

        entry_type = self.entry_signal or "immediate"
        logger.info(f"ORDER PLAN: BUY {qty} {ticker} @ ~${entry_price:.2f}")
        logger.info(f"  Entry: {entry_type} | Stop: {stop_type} ${hard_stop:.2f} "
                    f"(risk ${risk_total:.2f})")
        logger.info(f"  Trail: ${trail_amount:.2f} | Ceiling: ${target_price:.2f}")
        if self.float_rotation > 0:
            logger.info(f"  Float rotation: {self.float_rotation:.1f}x")

        if self.dry_run:
            logger.info("DRY RUN — no order placed")
            self.trade_taken = True
            return True

        # ── Place orders ──────────────────────────────────────────────
        # IBKR doesn't allow trailing stops as bracket children of
        # market orders. So we: buy first, wait for fill, then place
        # the trailing stop as a standalone order.

        # Step 1: Market buy (transmit immediately)
        parent = MarketOrder("BUY", qty)
        parent.tif = "DAY"
        parent.transmit = True
        parent_trade = self.ib.placeOrder(contract, parent)
        self.ib.sleep(3)

        # Verify order accepted and filled
        if parent_trade.orderStatus.status in ("Cancelled", "Inactive"):
            logger.error(f"Entry REJECTED: {parent_trade.orderStatus.status}")
            try:
                from live.alerts import send_discord
                if fmt_error:
                    msg = fmt_error(
                        "Gap Scanner",
                        f"Order REJECTED for {ticker}\n"
                        f"Status: {parent_trade.orderStatus.status}\n"
                        f"Action: Check TWS presets and permissions"
                    )
                else:
                    msg = f"ORDER REJECTED: {ticker}"
                send_discord(msg)
            except Exception:
                pass
            return False

        # Wait for fill (up to 10 seconds)
        for _ in range(5):
            if parent_trade.orderStatus.status == "Filled":
                break
            self.ib.sleep(2)

        # Get actual fill price
        if parent_trade.orderStatus.avgFillPrice > 0:
            entry_price = parent_trade.orderStatus.avgFillPrice
            logger.info(f"  Fill price: ${entry_price:.2f}")

        # Step 2: Place trailing stop as standalone order
        trail_order = Order()
        trail_order.action = "SELL"
        trail_order.orderType = "TRAIL"
        trail_order.totalQuantity = qty
        trail_order.auxPrice = trail_amount
        trail_order.tif = "DAY"
        trail_order.transmit = True
        trail_trade = self.ib.placeOrder(contract, trail_order)
        self.ib.sleep(2)

        # Verify trailing stop was accepted
        if trail_trade.orderStatus.status in ("Cancelled", "Inactive"):
            logger.error(f"TRAILING STOP REJECTED — placing hard stop instead")
            # Fallback: place a regular stop order
            fallback_stop = StopOrder("SELL", qty, hard_stop)
            fallback_stop.tif = "DAY"
            fallback_stop.transmit = True
            self.ib.placeOrder(contract, fallback_stop)
            self.ib.sleep(1)
            logger.info(f"  Fallback stop placed at ${hard_stop:.2f}")

            try:
                from live.alerts import send_discord
                send_discord(
                    f"WARNING: Trailing stop rejected for {ticker}. "
                    f"Hard stop at ${hard_stop:.2f} placed instead."
                )
            except Exception:
                pass
        else:
            logger.info(f"  Trailing stop active: ${trail_amount:.2f} trail")

        self.ib.sleep(2)

        self.position = {
            "ticker": ticker,
            "direction": "long",
            "entry": entry_price,
            "qty": qty,
            "stop": hard_stop,
            "stop_type": stop_type,
            "target": target_price,
            "trail_amount": trail_amount,
            "order_id": parent_trade.order.orderId,
            "trail_order_id": trail_trade.order.orderId if trail_trade else None,
            "confirmed": False,
        }
        self.trade_taken = True
        self.entered_trade = True

        # Log entry for ML training
        if self.trade_logger:
            try:
                self.trade_logger.log_entry(
                    ticker=ticker,
                    entry_price=entry_price,
                    target_data=data,
                    position=self.position,
                    entry_signal=self.entry_signal,
                    scan_number=self.trades_today + 1,
                    candidate_rank=self.next_candidate_idx + 1,
                    num_candidates=len(self.candidates) if self.candidates is not None else 0,
                    vwap=self.vwap,
                    pm_high=self.pm_high,
                    float_rotation=self.float_rotation,
                    scans_completed=self.scans_completed,
                )
            except Exception as e:
                logger.warning(f"  Trade logger entry failed: {e}")

        logger.info(f"LONG ENTRY: {qty} {ticker} @ ~${entry_price:.2f} | "
                    f"size=${pos_size:.0f} | trail=${trail_amount:.2f} | "
                    f"hard stop=${hard_stop:.2f}")

        # Discord
        try:
            from live.alerts import send_discord
            if fmt_entry:
                extra_items = [
                    ("Entry", entry_type.replace("_", " ").title()),
                    ("Gap", f"{data['gap_pct']:+.1f}%"),
                    ("Volume", f"{data['rvol']:.1f}x normal"),
                ]
                if data.get('si_pct', 0) > 0:
                    extra_items.append(
                        ("Short int", f"{data['si_pct']:.0f}% of float"))
                if data.get('options_signal'):
                    extra_items.append(
                        ("Options", f"{data['options_signal']} "
                                    f"(calls {data['call_vol_oi_ratio']:.1f}x OI)"))
                if self.float_rotation > 0.1:
                    extra_items.append(
                        ("Float rot", f"{self.float_rotation:.1f}x"))
                if self.vwap:
                    extra_items.append(("VWAP", f"${self.vwap:.2f}"))
                extra_items.append(
                    ("Stop", f"${hard_stop:.2f} ({stop_type})"))
                extra_items.append(
                    ("Trail", f"${trail_amount:.2f} "
                              f"({ScannerConfig.TRAIL_ATR_MULT}x ATR)"))

                msg = fmt_entry(
                    "Gap Scanner", ticker, "long", qty, entry_price,
                    hard_stop, target_price,
                    position_size=pos_size,
                    account_balance=self.account["balance"],
                    extra=dict(extra_items),
                )
            else:
                msg = (f"BUY {qty} {ticker} @ ${entry_price:.2f} | "
                       f"{entry_type} | trail=${trail_amount:.2f}")
            send_discord(msg)
        except Exception:
            pass

        return True

    def force_close_position(self, contract):
        """Force closes any open position (end of day)."""
        if self.position is None:
            return

        qty = self.position["qty"]
        ticker = self.position["ticker"]

        # Cancel open orders
        open_orders = self.ib.openOrders()
        for order in open_orders:
            try:
                self.ib.cancelOrder(order)
            except Exception:
                pass
        self.ib.sleep(1)

        order = MarketOrder("SELL", qty)
        order.tif = "DAY"
        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(2)

        price = self.get_current_price(self._contract)
        if price and self.position["entry"]:
            self.record_trade(
                ticker, self.position["direction"],
                self.position["entry"], price, qty, "eod"
            )

            # Log exit for ML training
            if self.trade_logger:
                try:
                    entry = self.position["entry"]
                    pnl_usd = (price - entry) * qty
                    pnl_pct = (price - entry) / entry * 100
                    self.trade_logger.log_exit(
                        exit_price=price,
                        exit_reason="eod",
                        pnl_usd=pnl_usd,
                        pnl_pct=pnl_pct,
                        account_after=self.account["balance"],
                        high_before_exit=self.intraday_high if self.intraday_high > 0 else None,
                    )
                except Exception as e:
                    logger.warning(f"  Trade logger exit failed: {e}")

            logger.info(f"EOD CLOSE: {qty} {ticker} @ ${price:.2f} | "
                        f"P&L: {self.daily_pnl:+.2f}% | "
                        f"Account: ${self.account['balance']:,.2f}")

            try:
                from live.alerts import send_discord
                if fmt_exit:
                    msg = fmt_exit(
                        "Gap Scanner", ticker,
                        self.position["direction"], qty,
                        self.position["entry"], price, "eod",
                        account_balance=self.account["balance"],
                        record=self.account,
                    )
                else:
                    msg = (f"EOD CLOSE: {ticker} @ ${price:.2f} | "
                           f"P&L: {self.daily_pnl:+.2f}%")
                send_discord(msg)
            except Exception:
                pass

        self.position = None

    def check_position_status(self, contract=None):
        """Checks if position has been closed by trailing stop."""
        if self.position is None:
            return

        positions = self.ib.positions()
        has_position = False
        for pos in positions:
            if pos.contract.symbol == self.position["ticker"] and \
               pos.position != 0:
                has_position = True
                break

        if has_position:
            self.position["confirmed"] = True
            return

        if not has_position and self.position.get("confirmed"):
            # Position gone — get real fill price from IBKR
            ticker = self.position["ticker"]
            entry = self.position["entry"]
            qty = self.position["qty"]

            # Get actual exit price from IBKR executions
            exit_price, fill_qty = self.get_ibkr_fill_price(ticker, "SLD")

            if exit_price:
                logger.info(f"  IBKR sell fill: ${exit_price:.2f} "
                            f"× {fill_qty} shares")
            else:
                # Fallback: current price
                exit_price = self.get_current_price(self._contract)
                if exit_price:
                    logger.info(f"  Using current price: ${exit_price:.2f}")
                else:
                    exit_price = entry
                    logger.warning("  Could not determine exit price")

            # Determine reason
            if exit_price > entry:
                reason = "trail"
            elif exit_price <= self.position["stop"]:
                reason = "stop"
            else:
                reason = "trail"

            self.record_trade(ticker, "long", entry, exit_price, qty, reason)

            # Log exit for ML training
            if self.trade_logger:
                try:
                    pnl_usd = (exit_price - entry) * qty
                    pnl_pct = (exit_price - entry) / entry * 100
                    self.trade_logger.log_exit(
                        exit_price=exit_price,
                        exit_reason=reason,
                        pnl_usd=pnl_usd,
                        pnl_pct=pnl_pct,
                        account_after=self.account["balance"],
                        high_before_exit=self.intraday_high if self.intraday_high > 0 else None,
                    )
                except Exception as e:
                    logger.warning(f"  Trade logger exit failed: {e}")

            logger.info(f"Position closed ({reason}) @ ${exit_price:.2f} | "
                        f"Account (IBKR): ${self.account['balance']:,.2f}")

            try:
                from live.alerts import send_discord
                if fmt_exit:
                    msg = fmt_exit(
                        "Gap Scanner", ticker, "long", qty,
                        entry, exit_price, reason,
                        account_balance=self.account["balance"],
                        record=self.account,
                    )
                else:
                    msg = (f"{reason.upper()}: {ticker} @ ${exit_price:.2f}")
                send_discord(msg)
            except Exception:
                pass

            self.position = None
            # Look for next trade if time permits
            self._prepare_for_next_trade()

        elif not has_position and not self.position.get("confirmed"):
            open_orders = self.ib.openOrders()
            has_pending = any(
                o.orderId == self.position.get("order_id")
                for o in open_orders
            )
            if not has_pending:
                logger.error("Entry FAILED — no position, no pending orders")
                try:
                    from live.alerts import send_discord
                    if fmt_error:
                        msg = fmt_error(
                            "Gap Scanner",
                            f"Entry FAILED for {self.position['ticker']}\n"
                            f"Orders were rejected or cancelled.\n"
                            f"No position opened. Check TWS."
                        )
                    else:
                        msg = (f"ENTRY FAILED: {self.position['ticker']} "
                               f"— orders rejected")
                    send_discord(msg)
                except Exception:
                    pass
                self.position = None
                self.entered_trade = False

    # ── Main loop ─────────────────────────────────────────────────────

    def run(self):
        """Main trading loop for one day."""
        # Load settings from Lattice app (if configured)
        ScannerConfig.load_settings()

        mode = "DRY RUN" if self.dry_run else \
               ("PAPER" if self.paper else "LIVE")

        pos_size = self.get_position_size()
        logger.info(f"{'='*60}")
        logger.info(f"  Gap Scanner Day Trader — {mode}")
        logger.info(f"  Account: ${self.account['balance']:,.2f} | "
                    f"Record: {self.account['wins']}W/"
                    f"{self.account['losses']}L | "
                    f"P&L: ${self.account['total_pnl_usd']:+,.2f}")
        if self.fixed_size:
            logger.info(f"  Position size: ${self.fixed_size:,.0f} (fixed)")
        else:
            logger.info(f"  Position size: ${pos_size:,.0f} "
                        f"({ScannerConfig.POSITION_PCT:.0%} of "
                        f"${self.account['balance']:,.0f}) — DYNAMIC")
        logger.info(f"  Strategy: gap-up {ScannerConfig.MIN_GAP_PCT}-"
                    f"{ScannerConfig.MAX_GAP_PCT}%, "
                    f"{ScannerConfig.MIN_RVOL}x+ vol, "
                    f"trailing stop {ScannerConfig.TRAIL_ATR_MULT}x ATR")
        logger.info(f"  Entry: {ScannerConfig.ENTRY_MODE} "
                    f"(VWAP pullback / PM high breakout) | "
                    f"Max trades/day: {ScannerConfig.MAX_TRADES_PER_DAY}")
        logger.info(f"{'='*60}")

        if not self.connect():
            return

        try:
            self.reset_daily_state()

            if not self.is_market_day():
                logger.info("Not a trading day. Exiting.")
                return

            self._contract = None  # Will be set after scan picks a ticker

            while True:
                now = self.get_current_time_et()
                current_time = now.time()

                # ── Before scan time: wait for market to open ─────────
                if current_time < ScannerConfig.SCAN_TIME:
                    wait = (datetime.combine(date.today(), ScannerConfig.SCAN_TIME) -
                            datetime.combine(date.today(), current_time)).seconds // 60
                    logger.info(f"Waiting for 9:35 ET to scan "
                                f"(need today's open prices). "
                                f"{wait} min remaining...")
                    if not self.dry_run:
                        self.ib.sleep(min(wait * 60, 60))
                    else:
                        time.sleep(min(wait * 60, 60))
                    continue

                # ── Run scanner (with re-scans if empty) ─────────────────
                if self.candidates is None or \
                   (self.candidates is not None and self.candidates.empty and
                    not self.trade_taken):

                    # Check if it's time for a scan or re-scan
                    should_scan = False

                    if self.scans_completed == 0:
                        # First scan
                        should_scan = True
                    else:
                        # Check re-scan schedule
                        for rescan_time in ScannerConfig.RESCAN_TIMES:
                            if (current_time >= rescan_time and
                                (self.last_scan_time is None or
                                 self.last_scan_time < rescan_time)):
                                should_scan = True
                                break

                    if not should_scan:
                        # Not time for a re-scan yet, keep waiting
                        if not self.dry_run:
                            self.ib.sleep(ScannerConfig.CHECK_INTERVAL)
                        else:
                            time.sleep(5)
                        continue

                    scan_label = ("Re-scanning" if self.scans_completed > 0
                                  else "Running gap scan")
                    logger.info(f"{scan_label} (market is open, "
                                f"scan #{self.scans_completed + 1})...")
                    self.candidates = run_pre_market_scan()
                    self.scans_completed += 1
                    self.last_scan_time = current_time

                    if self.candidates is None or self.candidates.empty:
                        # Check if more re-scans are scheduled
                        remaining = [t for t in ScannerConfig.RESCAN_TIMES
                                     if t > current_time]
                        if remaining:
                            next_scan = remaining[0]
                            logger.info(f"No candidates yet. "
                                        f"Will re-scan at "
                                        f"{next_scan.strftime('%H:%M')} ET")
                            try:
                                from live.alerts import send_discord
                                send_discord(
                                    f"Gap Scanner — No candidates at "
                                    f"{current_time.strftime('%H:%M')} ET. "
                                    f"Re-scanning at "
                                    f"{next_scan.strftime('%H:%M')} ET."
                                )
                            except Exception:
                                pass
                        else:
                            logger.info("No candidates after all scans. "
                                        "Sitting out.")
                            self.trade_taken = True
                            try:
                                from live.alerts import send_discord
                                if fmt_day_complete:
                                    msg = fmt_day_complete(
                                        "Gap Scanner", traded=False,
                                        account_balance=self.account["balance"],
                                        extra={"reason":
                                               "no candidates after "
                                               f"{self.scans_completed} scans"},
                                    )
                                else:
                                    msg = "No candidates today."
                                send_discord(msg)
                            except Exception:
                                pass
                    else:
                        # Pick the top candidate
                        top = self.candidates.iloc[0]
                        self.target_ticker = top["ticker"]
                        self.target_data = top

                        logger.info(f"TARGET: {self.target_ticker} "
                                    f"(gap={top['gap_pct']:+.1f}%, "
                                    f"rvol={top['rvol']:.1f}x, "
                                    f"score={top['score']:.2f})")

                        # Send scan results to Discord
                        try:
                            from live.alerts import send_discord
                            pos_size = self.get_position_size()
                            if fmt_scan_results:
                                msg = fmt_scan_results(
                                    "Gap Scanner",
                                    self.candidates.head(ScannerConfig.SHOW_TOP_N),
                                    self.target_ticker,
                                    position_size=pos_size,
                                )
                            else:
                                msg = (f"Trading: {self.target_ticker} "
                                       f"(${pos_size:,.0f} position)")
                            send_discord(msg)
                        except Exception:
                            pass

                        # Qualify the contract and fetch pre-market high
                        if not self.dry_run:
                            self._contract = self.get_contract(self.target_ticker)
                            # Get pre-market high for breakout entry
                            self.pm_high = self.fetch_premarket_high(
                                self.target_ticker)
                            if self.pm_high:
                                logger.info(f"  PM high breakout level: "
                                            f"${self.pm_high:.2f}")
                            else:
                                logger.info(f"  No PM high data — "
                                            f"using VWAP pullback only")

                # ── Wait for market open ──────────────────────────────
                if current_time < ScannerConfig.MARKET_OPEN:
                    logger.debug(f"Waiting for market open "
                                 f"({current_time.strftime('%H:%M')})")
                    if not self.dry_run:
                        self.ib.sleep(15)
                    else:
                        time.sleep(15)
                    continue

                # ── Market close ──────────────────────────────────────
                if current_time >= ScannerConfig.MARKET_CLOSE:
                    logger.info("Market closed.")
                    logger.info(f"  Traded: {'Yes' if self.entered_trade else 'No'}")
                    logger.info(f"  Ticker: {self.target_ticker or 'None'}")
                    logger.info(f"  P&L: {self.daily_pnl:+.2f}%")
                    break

                # ── Force close at 3:55 PM ────────────────────────────
                if current_time >= ScannerConfig.FORCE_EXIT and \
                   self.position is not None:
                    logger.info("End of day — force closing position")
                    self.force_close_position(self._contract)
                    continue

                # ── No entries after 11:30 AM ─────────────────────────
                if current_time >= ScannerConfig.LAST_ENTRY and \
                   not self.trade_taken:
                    logger.info("Past 11:30 AM — entry window closed")
                    self.trade_taken = True
                    continue

                # ── Smart entry: wait for VWAP pullback or PM breakout ────
                if not self.trade_taken and self.target_ticker and \
                   current_time >= ScannerConfig.MARKET_OPEN and \
                   self.position is None:

                    if ScannerConfig.ENTRY_MODE == "smart" and not self.dry_run:
                        signal, price = self.check_entry_signal(self._contract)

                        if signal:
                            # Verify gap hasn't completely faded
                            live_gap = (price / self.target_data["prev_close"]
                                        - 1) * 100
                            if live_gap < ScannerConfig.MIN_GAP_PCT * 0.5:
                                logger.warning(
                                    f"{self.target_ticker} gap faded to "
                                    f"{live_gap:+.1f}% — trying next")
                                try:
                                    from live.alerts import send_discord
                                    if fmt_abort:
                                        msg = fmt_abort(
                                            "Gap Scanner",
                                            self.target_ticker,
                                            f"Gap faded to {live_gap:+.1f}%"
                                        )
                                    else:
                                        msg = f"ABORT: gap faded"
                                    send_discord(msg)
                                except Exception:
                                    pass
                                self._advance_to_next_candidate()
                                continue

                            # Update entry price and signal type
                            self.target_data = self.target_data.copy()
                            self.target_data["open"] = price
                            self.entry_signal = signal

                            # Use VWAP-based stop if pullback entry
                            if signal == "vwap_pullback" and self.vwap:
                                vwap_stop = self.vwap * (
                                    1 - ScannerConfig.VWAP_STOP_BUFFER_PCT / 100)
                                self.target_data["vwap_stop"] = vwap_stop
                                logger.info(f"  VWAP stop: ${vwap_stop:.2f} "
                                            f"(tighter than gap stop)")

                            fr_str = (f" | float rotation: {self.float_rotation:.1f}x"
                                      if self.float_rotation > 0 else "")
                            logger.info(f"ENTERING via {signal}: "
                                        f"{self.target_ticker} @ ${price:.2f}"
                                        f"{fr_str}")

                            self.enter_trade(self._contract)
                        # else: no signal yet, keep monitoring

                    else:
                        # Immediate mode or dry run — enter now
                        logger.info(f"Entering {self.target_ticker}...")
                        if not self.dry_run:
                            price = self.get_current_price(self._contract)
                            if price:
                                live_gap = (price /
                                            self.target_data["prev_close"]
                                            - 1) * 100
                                if live_gap < ScannerConfig.MIN_GAP_PCT * 0.5:
                                    self._advance_to_next_candidate()
                                    continue
                                self.target_data = self.target_data.copy()
                                self.target_data["open"] = price
                        self.enter_trade(self._contract)

                # ── Monitor position ──────────────────────────────────
                if self.position is not None and not self.dry_run:
                    self.check_position_status(self._contract)

                # ── Sleep ─────────────────────────────────────────────
                if not self.dry_run:
                    self.ib.sleep(ScannerConfig.CHECK_INTERVAL)
                else:
                    # Dry run: just show what would happen and exit
                    if self.trade_taken:
                        break
                    time.sleep(5)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            if self.position:
                logger.warning("You have an open position! Close in TWS.")
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            if self.position:
                logger.warning("You may have an open position! Check TWS.")
        finally:
            # Daily summary Discord
            try:
                from live.alerts import send_discord
                traded = self.entered_trade
                extra = {}
                if not traded:
                    if self.candidates is not None and len(self.candidates) == 0:
                        extra["reason"] = "no candidates passed filters"
                    else:
                        extra["reason"] = "no entry triggered"

                if fmt_day_complete:
                    if self.trades_today > 0:
                        extra["Trades"] = str(self.trades_today)
                    msg = fmt_day_complete(
                        "Gap Scanner",
                        traded=traded,
                        ticker=self.target_ticker,
                        pnl_pct=self.daily_pnl,
                        account_balance=self.account["balance"],
                        record=self.account,
                        extra=extra,
                    )
                else:
                    msg = (f"Day complete | P&L: {self.daily_pnl:+.2f}% | "
                           f"Account: ${self.account['balance']:,.2f}")
                send_discord(msg)
            except Exception:
                pass

            self.disconnect()


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def run_gap_scanner(args=None):
    """Entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Gap Scanner Day Trader")
    parser.add_argument("--live", action="store_true",
                        help="Use live account")
    parser.add_argument("--size", type=float, default=None,
                        help="Fixed position size in USD (overrides dynamic)")
    parser.add_argument("--capital", type=float, default=None,
                        help="Starting capital (first run only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Scan only, no IBKR connection or orders")
    parser.add_argument("--deposit", type=float, default=None,
                        help="Record a deposit to the account")

    if args is not None:
        parsed = parser.parse_args(args)
    else:
        parsed = parser.parse_args(sys.argv[1:])

    # Handle deposit command
    if parsed.deposit:
        bot = GapScannerBot(dry_run=True)
        bot.account["balance"] += parsed.deposit
        bot.account["starting_capital"] += parsed.deposit
        bot.account["peak_balance"] = max(
            bot.account["peak_balance"],
            bot.account["balance"]
        )
        bot._save_account()
        logger.info(f"Deposit recorded: +${parsed.deposit:,.2f} → "
                    f"balance: ${bot.account['balance']:,.2f}")
        return

    bot = GapScannerBot(
        paper=not parsed.live,
        position_size=parsed.size,
        dry_run=parsed.dry_run,
        starting_capital=parsed.capital,
    )
    bot.run()


if __name__ == "__main__":
    run_gap_scanner()
