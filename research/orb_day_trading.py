"""
research/orb_day_trading.py  (v2)
-----------------------------------
RESEARCH — Opening Range Breakout (ORB) day trading strategy.

Phase A: Daily OHLC approximation (2018-2026)
  - Fixed: uses close vs open direction + magnitude, not stops/targets
  - Tests whether breakout days produce directional follow-through

Phase B: Intraday 5-min ORB (last 60 days)
  - True bar-by-bar simulation with precise entries/exits
  - Enhanced with: gap filter, day-of-week, time-of-exit, streak analysis

Usage:
    python research/orb_day_trading.py
    python research/orb_day_trading.py --collect     # Cache intraday data
    python research/orb_day_trading.py --intraday-only
"""

import sys
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from datetime import datetime, timedelta, time as dtime

sys.path.insert(0, ".")

CACHE_DIR = Path("data/intraday_cache")
TICKERS = ["SPY", "QQQ"]


# ═══════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════

def collect_intraday_data(tickers=TICKERS, interval="5m"):
    """Download and cache intraday data."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Collecting {interval} intraday data for {tickers}...")

    for ticker in tickers:
        cache_path = CACHE_DIR / f"{ticker}_{interval}.csv"
        raw = yf.download(ticker, period="60d", interval=interval,
                           auto_adjust=True, progress=False, prepost=False)
        if raw.empty:
            logger.warning(f"No intraday data for {ticker}")
            continue

        raw = raw.reset_index()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] if c[1] == '' else c[0] for c in raw.columns]
        raw.columns = [c.lower() if isinstance(c, str) else c for c in raw.columns]
        if "datetime" in raw.columns:
            raw = raw.rename(columns={"datetime": "timestamp"})
        elif "date" in raw.columns:
            raw = raw.rename(columns={"date": "timestamp"})
        raw["timestamp"] = pd.to_datetime(raw["timestamp"])

        if cache_path.exists():
            existing = pd.read_csv(cache_path, parse_dates=["timestamp"])
            combined = pd.concat([existing, raw]).drop_duplicates(subset="timestamp").sort_values("timestamp")
            logger.info(f"{ticker}: merged → {len(combined)} total bars")
        else:
            combined = raw
            logger.info(f"{ticker}: {len(combined)} bars (new cache)")

        combined.to_csv(cache_path, index=False)
    logger.info("Intraday collection complete.")


def load_intraday(ticker, interval="5m"):
    cache_path = CACHE_DIR / f"{ticker}_{interval}.csv"
    if not cache_path.exists():
        return pd.DataFrame()
    return pd.read_csv(cache_path, parse_dates=["timestamp"])


# ═══════════════════════════════════════════════════════════════════════
# Phase A: Daily OHLC — Breakout Follow-Through Analysis
# ═══════════════════════════════════════════════════════════════════════

def run_daily_orb_test():
    """
    Fixed daily approximation. Instead of simulating intraday stops/targets
    with daily bars (which doesn't work), we ask a simpler question:

    "On days where price breaks meaningfully above/below the open,
     does it tend to close in the breakout direction?"

    This measures whether ORB-style breakouts have follow-through,
    which is the statistical foundation the strategy needs.
    """
    logger.info("=" * 65)
    logger.info("  PHASE A: Daily Breakout Follow-Through Analysis")
    logger.info("=" * 65)

    for ticker in TICKERS:
        raw = yf.download(ticker, start="2018-01-01", auto_adjust=True, progress=False)
        if raw.empty:
            continue
        raw = raw.reset_index()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] if c[1] == '' else c[0] for c in raw.columns]
        raw.columns = [c.lower() if isinstance(c, str) else c for c in raw.columns]
        df = raw.sort_values("date").copy()

        # ATR for dynamic thresholds
        df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
        df["atr_pct"] = df["atr"] / df["close"]

        # Previous close for gap calculation
        df["prev_close"] = df["close"].shift(1)
        df["gap_pct"] = (df["open"] - df["prev_close"]) / df["prev_close"] * 100

        # How far did price move from open?
        df["open_to_high_pct"] = (df["high"] - df["open"]) / df["open"] * 100
        df["open_to_low_pct"] = (df["open"] - df["low"]) / df["open"] * 100
        df["open_to_close_pct"] = (df["close"] - df["open"]) / df["open"] * 100

        df = df.dropna(subset=["atr"])

        print(f"\n  {ticker} — {len(df)} days")
        print(f"  {'='*65}")

        # Test multiple breakout thresholds
        for threshold_name, threshold in [
            ("0.2% from open", 0.002),
            ("0.3% from open", 0.003),
            ("0.5% from open", 0.005),
            ("0.3x ATR from open", None),  # dynamic
        ]:
            if threshold is None:
                # Dynamic: 0.3x ATR
                breakout_level = df["open"] + df["atr"] * 0.3
                breakdown_level = df["open"] - df["atr"] * 0.3
                long_breakout = df["high"] >= breakout_level
                short_breakout = df["low"] <= breakdown_level
            else:
                long_breakout = df["open_to_high_pct"] >= threshold * 100
                short_breakout = df["open_to_low_pct"] >= threshold * 100

            # Long breakout days: did they close above open?
            long_days = df[long_breakout & ~short_breakout]  # clean breakouts only
            short_days = df[short_breakout & ~long_breakout]
            both_days = df[long_breakout & short_breakout]

            if len(long_days) > 0:
                long_close_above = (long_days["close"] > long_days["open"]).mean() * 100
                long_avg_move = long_days["open_to_close_pct"].mean()
                long_median_move = long_days["open_to_close_pct"].median()
            else:
                long_close_above = long_avg_move = long_median_move = 0

            if len(short_days) > 0:
                short_close_below = (short_days["close"] < short_days["open"]).mean() * 100
                short_avg_move = -short_days["open_to_close_pct"].mean()  # positive = follow-through
                short_median_move = -short_days["open_to_close_pct"].median()
            else:
                short_close_below = short_avg_move = short_median_move = 0

            print(f"\n  Threshold: {threshold_name}")
            print(f"    LONG breakout days:  {len(long_days):>5} | "
                  f"Close above open: {long_close_above:.1f}% | "
                  f"Avg follow-through: {long_avg_move:+.3f}% | "
                  f"Median: {long_median_move:+.3f}%")
            print(f"    SHORT breakout days: {len(short_days):>5} | "
                  f"Close below open: {short_close_below:.1f}% | "
                  f"Avg follow-through: {short_avg_move:+.3f}% | "
                  f"Median: {short_median_move:+.3f}%")
            print(f"    Both-direction days: {len(both_days):>5} (noisy — skip these)")

        # Gap analysis: do gapped-up days continue or fade?
        print(f"\n  GAP ANALYSIS:")
        for gap_label, gap_min, gap_max in [
            ("Gap up 0.3-1.0%", 0.3, 1.0),
            ("Gap up >1.0%", 1.0, 10.0),
            ("Gap down 0.3-1.0%", -1.0, -0.3),
            ("Gap down >1.0%", -10.0, -1.0),
            ("Small gap (<0.3%)", -0.3, 0.3),
        ]:
            mask = (df["gap_pct"] >= gap_min) & (df["gap_pct"] < gap_max)
            gap_days = df[mask]
            if len(gap_days) < 10:
                continue
            # For gap ups: does price continue higher from open?
            avg_otc = gap_days["open_to_close_pct"].mean()
            pct_continues = (gap_days["open_to_close_pct"] > 0).mean() * 100 if gap_min > 0 else \
                            (gap_days["open_to_close_pct"] < 0).mean() * 100
            print(f"    {gap_label:<25} {len(gap_days):>4} days | "
                  f"Continues: {pct_continues:.1f}% | "
                  f"Avg open→close: {avg_otc:+.3f}%")

        # Day-of-week
        df["dow"] = pd.to_datetime(df["date"]).dt.dayofweek
        print(f"\n  DAY-OF-WEEK (avg open→close move):")
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        for d in range(5):
            day_data = df[df["dow"] == d]
            avg = day_data["open_to_close_pct"].mean()
            win = (day_data["open_to_close_pct"] > 0).mean() * 100
            vol = day_data["open_to_close_pct"].std()
            print(f"    {dow_names[d]}: avg {avg:+.3f}% | win {win:.1f}% | vol {vol:.3f}%")


# ═══════════════════════════════════════════════════════════════════════
# Phase B: Intraday 5-Min ORB (Enhanced)
# ═══════════════════════════════════════════════════════════════════════

def _find_exit(bars, entry, target, stop, direction):
    """Scan bars for first exit."""
    for _, bar in bars.iterrows():
        if direction == "long":
            if bar["low"] <= stop:
                return stop, "stop"
            if bar["high"] >= target:
                return target, "target"
        else:
            if bar["high"] >= stop:
                return stop, "stop"
            if bar["low"] <= target:
                return target, "target"
        if bar["time"] >= dtime(15, 55):
            return bar["close"], "eod"
    if len(bars) > 0:
        return bars.iloc[-1]["close"], "eod"
    return entry, "flat"


def backtest_orb_intraday(df, params):
    """True ORB backtest on 5-min bars with enhanced tracking."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None)

    df["date"] = df["timestamp"].dt.date
    df["time"] = df["timestamp"].dt.time

    or_bars = params["or_bars"]
    target_mult = params["target_mult"]
    stop_mult = params["stop_mult"]
    direction = params["direction"]
    max_gap_pct = params.get("max_gap_pct", None)  # Skip days with gaps > this
    skip_both_break = params.get("skip_both_break", False)  # Skip if OR broken both ways

    trades = []

    for day, day_df in df.groupby("date"):
        mkt = day_df[(day_df["time"] >= dtime(9, 30)) & (day_df["time"] <= dtime(15, 55))]
        if len(mkt) < or_bars + 5:
            continue

        or_data = mkt.iloc[:or_bars]
        or_high = or_data["high"].max()
        or_low = or_data["low"].min()
        or_range = or_high - or_low
        or_mid = (or_high + or_low) / 2

        if or_range <= 0 or or_mid <= 0:
            continue

        or_range_pct = or_range / or_mid * 100

        # Gap filter
        if max_gap_pct is not None:
            day_open = mkt.iloc[0]["open"]
            # Get previous day's close from the full df
            prev_days = sorted(df["date"].unique())
            day_idx = list(prev_days).index(day) if day in prev_days else -1
            if day_idx > 0:
                prev_day = prev_days[day_idx - 1]
                prev_close_data = df[df["date"] == prev_day]
                if len(prev_close_data) > 0:
                    prev_close = prev_close_data.iloc[-1]["close"]
                    gap = abs(day_open / prev_close - 1) * 100
                    if gap > max_gap_pct:
                        continue

        target_pts = or_range * target_mult
        stop_pts = or_range * stop_mult

        remaining = mkt.iloc[or_bars:]
        trade_taken = False

        for _, bar in remaining.iterrows():
            if trade_taken:
                break

            # Long breakout
            if direction in ("long", "both") and bar["high"] > or_high:
                if skip_both_break and bar["low"] < or_low:
                    continue  # Both sides broken in same bar — skip

                entry = or_high
                target = entry + target_pts
                stop = entry - stop_pts
                future_bars = remaining[remaining["timestamp"] >= bar["timestamp"]]
                exit_price, reason = _find_exit(future_bars, entry, target, stop, "long")
                pnl_pct = (exit_price - entry) / entry * 100

                trades.append({
                    "date": day, "direction": "long", "entry": entry,
                    "exit": exit_price, "pnl_pct": pnl_pct, "reason": reason,
                    "or_range_pct": or_range_pct,
                    "entry_time": bar["time"],
                    "dow": pd.Timestamp(day).dayofweek,
                })
                trade_taken = True

            # Short breakout
            elif direction in ("short", "both") and bar["low"] < or_low:
                if skip_both_break and bar["high"] > or_high:
                    continue

                entry = or_low
                target = entry - target_pts
                stop = entry + stop_pts
                future_bars = remaining[remaining["timestamp"] >= bar["timestamp"]]
                exit_price, reason = _find_exit(future_bars, entry, target, stop, "short")
                pnl_pct = (entry - exit_price) / entry * 100

                trades.append({
                    "date": day, "direction": "short", "entry": entry,
                    "exit": exit_price, "pnl_pct": pnl_pct, "reason": reason,
                    "or_range_pct": or_range_pct,
                    "entry_time": bar["time"],
                    "dow": pd.Timestamp(day).dayofweek,
                })
                trade_taken = True

    return pd.DataFrame(trades)


def analyze_trades(trades, label):
    """Detailed trade analysis."""
    if trades.empty:
        return

    n = len(trades)
    wins = trades[trades["pnl_pct"] > 0]
    losses = trades[trades["pnl_pct"] <= 0]
    win_rate = len(wins) / n * 100
    avg_pnl = trades["pnl_pct"].mean()
    avg_win = wins["pnl_pct"].mean() if len(wins) > 0 else 0
    avg_loss = losses["pnl_pct"].mean() if len(losses) > 0 else 0
    gross_wins = wins["pnl_pct"].sum() if len(wins) > 0 else 0
    gross_losses = abs(losses["pnl_pct"].sum()) if len(losses) > 0 else 0
    pf = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    # Max consecutive losses
    is_loss = (trades["pnl_pct"] <= 0).astype(int)
    max_consec_loss = 0
    current_streak = 0
    for val in is_loss:
        if val:
            current_streak += 1
            max_consec_loss = max(max_consec_loss, current_streak)
        else:
            current_streak = 0

    # Cumulative P&L and drawdown
    cum = trades["pnl_pct"].cumsum()
    max_dd = (cum - cum.cummax()).min()

    # Exit reason breakdown
    reason_counts = trades["reason"].value_counts()

    print(f"\n  ┌─ {label}")
    print(f"  │  Trades: {n}  |  Win Rate: {win_rate:.1f}%  |  PF: {pf:.2f}  |  Total: {cum.iloc[-1]:+.2f}%")
    print(f"  │  Avg Win: {avg_win:+.3f}%  |  Avg Loss: {avg_loss:+.3f}%  |  Avg Trade: {avg_pnl:+.3f}%")
    print(f"  │  Max Consec Loss: {max_consec_loss}  |  Max DD: {max_dd:+.2f}%")
    print(f"  │  Exits: {', '.join(f'{r}={c}' for r, c in reason_counts.items())}")

    # Day of week
    if "dow" in trades.columns and n >= 20:
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        print(f"  │  Day-of-week:")
        for d in range(5):
            dt = trades[trades["dow"] == d]
            if len(dt) >= 3:
                dwr = len(dt[dt["pnl_pct"] > 0]) / len(dt) * 100
                davg = dt["pnl_pct"].mean()
                print(f"  │    {dow_names[d]}: {len(dt):>3} trades, {dwr:.0f}% win, {davg:+.3f}% avg")

    # Entry time analysis
    if "entry_time" in trades.columns and n >= 10:
        trades_c = trades.copy()
        trades_c["entry_hour"] = pd.to_datetime(trades_c["entry_time"].astype(str)).dt.hour
        early = trades_c[trades_c["entry_hour"] < 11]  # Before 11 AM
        late = trades_c[trades_c["entry_hour"] >= 11]   # 11 AM+
        if len(early) >= 3 and len(late) >= 3:
            early_wr = (early["pnl_pct"] > 0).mean() * 100
            late_wr = (late["pnl_pct"] > 0).mean() * 100
            print(f"  │  Entry timing:")
            print(f"  │    Before 11AM: {len(early)} trades, {early_wr:.0f}% win, {early['pnl_pct'].mean():+.3f}% avg")
            print(f"  │    After 11AM:  {len(late)} trades, {late_wr:.0f}% win, {late['pnl_pct'].mean():+.3f}% avg")

    # OR range size analysis
    if "or_range_pct" in trades.columns and n >= 10:
        median_or = trades["or_range_pct"].median()
        tight = trades[trades["or_range_pct"] <= median_or]
        wide = trades[trades["or_range_pct"] > median_or]
        if len(tight) >= 3 and len(wide) >= 3:
            tight_wr = (tight["pnl_pct"] > 0).mean() * 100
            wide_wr = (wide["pnl_pct"] > 0).mean() * 100
            print(f"  │  OR range size (median {median_or:.2f}%):")
            print(f"  │    Tight OR (≤{median_or:.2f}%): {len(tight)} trades, {tight_wr:.0f}% win, {tight['pnl_pct'].mean():+.3f}% avg")
            print(f"  │    Wide OR (>{median_or:.2f}%):  {len(wide)} trades, {wide_wr:.0f}% win, {wide['pnl_pct'].mean():+.3f}% avg")

    # Long vs Short breakdown
    if "direction" in trades.columns:
        for d in ["long", "short"]:
            dt = trades[trades["direction"] == d]
            if len(dt) >= 3:
                dwr = (dt["pnl_pct"] > 0).mean() * 100
                print(f"  │  {d.upper()}: {len(dt)} trades, {dwr:.0f}% win, {dt['pnl_pct'].mean():+.3f}% avg")

    print(f"  └{'─'*50}")


def run_intraday_orb_test():
    """Enhanced intraday ORB test."""
    logger.info("\n" + "=" * 65)
    logger.info("  PHASE B: Intraday 5-Min ORB (Enhanced)")
    logger.info("=" * 65)

    # Core parameter grid
    param_combos = [
        {"or_bars": 3, "target_mult": 1.5, "stop_mult": 1.0, "direction": "long",
         "label": "Long 15min OR, 1.5:1 R:R"},
        {"or_bars": 3, "target_mult": 2.0, "stop_mult": 1.0, "direction": "long",
         "label": "Long 15min OR, 2:1 R:R"},
        {"or_bars": 6, "target_mult": 1.5, "stop_mult": 1.0, "direction": "long",
         "label": "Long 30min OR, 1.5:1 R:R"},
        {"or_bars": 3, "target_mult": 1.5, "stop_mult": 1.0, "direction": "both",
         "label": "Both 15min OR, 1.5:1 R:R"},
        {"or_bars": 6, "target_mult": 1.5, "stop_mult": 1.0, "direction": "both",
         "label": "Both 30min OR, 1.5:1 R:R"},
        {"or_bars": 3, "target_mult": 1.0, "stop_mult": 1.0, "direction": "long",
         "label": "Long 15min OR, 1:1 R:R"},
    ]

    # Enhanced combos with filters
    enhanced_combos = [
        {"or_bars": 3, "target_mult": 1.5, "stop_mult": 1.0, "direction": "both",
         "max_gap_pct": 0.5, "label": "Both 15min, 1.5:1, gap<0.5%"},
        {"or_bars": 3, "target_mult": 1.5, "stop_mult": 1.0, "direction": "both",
         "max_gap_pct": 1.0, "label": "Both 15min, 1.5:1, gap<1.0%"},
        {"or_bars": 3, "target_mult": 1.5, "stop_mult": 1.0, "direction": "both",
         "skip_both_break": True, "label": "Both 15min, 1.5:1, clean breaks"},
        {"or_bars": 3, "target_mult": 2.0, "stop_mult": 1.0, "direction": "both",
         "max_gap_pct": 0.5, "label": "Both 15min, 2:1, gap<0.5%"},
    ]

    has_data = False

    for ticker in TICKERS:
        df = load_intraday(ticker)
        if df.empty:
            continue
        has_data = True
        n_days = df["timestamp"].dt.date.nunique()

        print(f"\n{'='*65}")
        print(f"  {ticker} — {n_days} days of 5-min data")
        print(f"{'='*65}")

        # Summary table first
        print(f"\n  {'Strategy':<35} {'Trades':>6} {'Win%':>6} {'PF':>5} {'Total':>7} {'MaxDD':>7}")
        print(f"  {'-'*70}")

        all_trades = {}

        for params in param_combos + enhanced_combos:
            trades = backtest_orb_intraday(df, params)
            if trades.empty:
                continue

            label = params["label"]
            all_trades[label] = trades

            n = len(trades)
            wr = (trades["pnl_pct"] > 0).mean() * 100
            gw = trades[trades["pnl_pct"] > 0]["pnl_pct"].sum()
            gl = abs(trades[trades["pnl_pct"] <= 0]["pnl_pct"].sum())
            pf = gw / gl if gl > 0 else float('inf')
            total = trades["pnl_pct"].sum()
            cum = trades["pnl_pct"].cumsum()
            max_dd = (cum - cum.cummax()).min()

            print(f"  {label:<35} {n:>6} {wr:>5.1f}% {pf:>5.2f} {total:>+6.2f}% {max_dd:>+6.2f}%")

        # Detailed analysis on top strategies
        print(f"\n  DETAILED ANALYSIS — Top Strategies:")
        # Pick top 3 by profit factor
        scored = []
        for label, trades in all_trades.items():
            if len(trades) < 10:
                continue
            gw = trades[trades["pnl_pct"] > 0]["pnl_pct"].sum()
            gl = abs(trades[trades["pnl_pct"] <= 0]["pnl_pct"].sum())
            pf = gw / gl if gl > 0 else 0
            scored.append((label, pf, trades))

        scored.sort(key=lambda x: x[1], reverse=True)
        for label, pf, trades in scored[:3]:
            analyze_trades(trades, f"{ticker} — {label}")

    if not has_data:
        print("\n  No intraday data. Run: python research/orb_day_trading.py --collect")


# ═══════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════

def plot_orb_results(ticker, trades, label):
    """Plot cumulative P&L curve for a strategy."""
    if trades.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d1117")
    fig.suptitle(f"ORB Research: {ticker} — {label}", color="white", fontsize=13, fontweight="bold")

    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="gray")
        ax.spines[:].set_color("#333")

    # Cumulative P&L
    ax1 = axes[0]
    cum = trades["pnl_pct"].cumsum()
    colors = ["#00e676" if v >= 0 else "#ff5252" for v in trades["pnl_pct"]]
    ax1.plot(range(len(cum)), cum, color="#448aff", linewidth=1.5)
    ax1.fill_between(range(len(cum)), 0, cum, alpha=0.1, color="#448aff")
    ax1.axhline(0, color="#555", linewidth=0.5)
    ax1.set_xlabel("Trade #", color="gray")
    ax1.set_ylabel("Cumulative P&L (%)", color="gray")
    ax1.set_title("Equity Curve", color="white")

    # Trade distribution
    ax2 = axes[1]
    bins = np.linspace(trades["pnl_pct"].min(), trades["pnl_pct"].max(), 25)
    ax2.hist(trades["pnl_pct"], bins=bins, color="#448aff", alpha=0.7, edgecolor="#333")
    ax2.axvline(0, color="#ff5252", linewidth=1, linestyle="--")
    ax2.axvline(trades["pnl_pct"].mean(), color="#00e676", linewidth=1, linestyle="--", label=f"Mean: {trades['pnl_pct'].mean():+.3f}%")
    ax2.set_xlabel("P&L (%)", color="gray")
    ax2.set_ylabel("Count", color="gray")
    ax2.set_title("Trade Distribution", color="white")
    ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    plt.tight_layout()
    Path("research").mkdir(exist_ok=True)
    save_path = f"research/orb_{ticker.lower()}_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    logger.info(f"Chart saved to {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true", help="Collect intraday data only")
    parser.add_argument("--daily-only", action="store_true")
    parser.add_argument("--intraday-only", action="store_true")
    args = parser.parse_args()

    if args.collect:
        collect_intraday_data()
        sys.exit(0)

    if not args.intraday_only:
        run_daily_orb_test()

    if not args.daily_only:
        collect_intraday_data()
        run_intraday_orb_test()

        # Plot best strategy per ticker
        for ticker in TICKERS:
            df = load_intraday(ticker)
            if df.empty:
                continue
            # Run best config and plot
            best_params = {"or_bars": 3, "target_mult": 1.5, "stop_mult": 1.0,
                           "direction": "both", "label": "Both 15min OR, 1.5:1"}
            trades = backtest_orb_intraday(df, best_params)
            if not trades.empty:
                plot_orb_results(ticker, trades, best_params["label"])

    print("\n  Run --collect weekly to build more intraday history.")
    print("  More data = more confidence in the results.")
