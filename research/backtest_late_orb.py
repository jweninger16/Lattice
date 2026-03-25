"""
research/backtest_late_orb.py
-------------------------------
RESEARCH — Late ORB strategy for gap days.

When the regular ORB bot sits out due to gap > 0.5%, this tests whether
a *delayed* opening range (letting the gap settle) produces tradeable
breakouts on those exact days.

Concept:
  - Normal ORB uses 9:30-9:45 opening range, skips if gap > 0.5%
  - Late ORB: on gap days, wait for price to settle, then form a NEW
    range starting at 10:00, 10:15, or 10:30 and trade that breakout
  - Hypothesis: initial gap volatility fades, a later range is cleaner

Tests:
  1. Multiple late-OR start times (10:00, 10:15, 10:30)
  2. Multiple OR durations (15min, 30min)
  3. Various R:R ratios (1:1, 1.5:1, 2:1)
  4. Gap size buckets (0.5-1%, 1-2%, 2%+)
  5. Gap direction (gap up vs gap down) and trade direction alignment
  6. Comparison: late ORB on gap days vs regular ORB on non-gap days

Usage:
    python research/backtest_late_orb.py
    python research/backtest_late_orb.py --collect   # Cache intraday data first
"""

import sys
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta, time as dtime
from itertools import product as iter_product

sys.path.insert(0, ".")

CACHE_DIR = Path("data/intraday_cache")
TICKERS = ["SPY", "QQQ"]
GAP_THRESHOLD = 0.5  # Minimum gap % to qualify as a "gap day"


# ═══════════════════════════════════════════════════════════════════════
# Data (reuses existing cache from orb_day_trading.py)
# ═══════════════════════════════════════════════════════════════════════

def collect_intraday_data(tickers=TICKERS, interval="5m"):
    """Download and cache intraday data."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Collecting {interval} intraday data for {tickers}...")

    for ticker in tickers:
        cache_path = CACHE_DIR / f"{ticker}_{interval}.csv"
        raw = yf.download(ticker, period="60d", interval=interval,
                          auto_adjust=True, progress=False, prepost=False)
        if raw.empty:
            print(f"  WARNING: No intraday data for {ticker}")
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
            combined = pd.concat([existing, raw]).drop_duplicates(
                subset="timestamp"
            ).sort_values("timestamp")
            print(f"  {ticker}: merged → {len(combined)} total bars")
        else:
            combined = raw
            print(f"  {ticker}: {len(combined)} bars (new cache)")

        combined.to_csv(cache_path, index=False)
    print("  Intraday collection complete.\n")


def load_intraday(ticker, interval="5m"):
    cache_path = CACHE_DIR / f"{ticker}_{interval}.csv"
    if not cache_path.exists():
        return pd.DataFrame()
    return pd.read_csv(cache_path, parse_dates=["timestamp"])


# ═══════════════════════════════════════════════════════════════════════
# Core backtest engine
# ═══════════════════════════════════════════════════════════════════════

def _find_exit(bars, entry, target, stop, direction, deadline=dtime(15, 55)):
    """Scan bars for first exit hit."""
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
        if bar["time"] >= deadline:
            return bar["close"], "eod"
    if len(bars) > 0:
        return bars.iloc[-1]["close"], "eod"
    return entry, "flat"


def classify_gap(gap_pct):
    """Bucket gap size for analysis."""
    absg = abs(gap_pct)
    if absg < 0.5:
        return "no_gap"
    elif absg < 1.0:
        return "small_gap"
    elif absg < 2.0:
        return "medium_gap"
    else:
        return "large_gap"


def backtest_late_orb(df, params):
    """
    Late ORB backtest: on gap days, form an opening range starting LATER
    than 9:30 and trade the breakout from that delayed range.

    params:
        or_start_time  : time  — when the late OR window begins (e.g., 10:00)
        or_bars        : int   — number of 5-min bars for the late OR (3=15min, 6=30min)
        target_mult    : float — target as multiple of OR range
        stop_mult      : float — stop as multiple of OR range
        direction      : str   — "long", "short", "both"
        min_gap_pct    : float — minimum abs gap to qualify (default 0.5)
        max_gap_pct    : float — maximum abs gap (default 999)
        gap_direction  : str   — "any", "up", "down" — filter by gap direction
        fade_only      : bool  — only trade AGAINST the gap (fade the gap)
        last_entry     : time  — no new entries after this time
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert(
            "America/New_York"
        ).dt.tz_localize(None)

    df["date"] = df["timestamp"].dt.date
    df["time"] = df["timestamp"].dt.time

    or_start = params["or_start_time"]
    or_bars = params["or_bars"]
    target_mult = params["target_mult"]
    stop_mult = params["stop_mult"]
    direction = params.get("direction", "both")
    min_gap_pct = params.get("min_gap_pct", GAP_THRESHOLD)
    max_gap_pct = params.get("max_gap_pct", 999.0)
    gap_dir = params.get("gap_direction", "any")
    fade_only = params.get("fade_only", False)
    last_entry = params.get("last_entry", dtime(14, 0))

    # Compute OR end time from start + bars
    or_start_minutes = or_start.hour * 60 + or_start.minute
    or_end_minutes = or_start_minutes + or_bars * 5
    or_end_time = dtime(or_end_minutes // 60, or_end_minutes % 60)

    trades = []
    gap_days_seen = 0
    gap_days_traded = 0

    sorted_dates = sorted(df["date"].unique())

    for i, day in enumerate(sorted_dates):
        day_df = df[df["date"] == day]
        mkt = day_df[
            (day_df["time"] >= dtime(9, 30)) & (day_df["time"] <= dtime(15, 55))
        ]
        if len(mkt) < 6:
            continue

        # ── Gap calculation ──────────────────────────────────────────
        day_open = mkt.iloc[0]["open"]
        if i == 0:
            continue  # Need prior day

        prev_day = sorted_dates[i - 1]
        prev_close_data = df[df["date"] == prev_day]
        if prev_close_data.empty:
            continue
        prev_close = prev_close_data.iloc[-1]["close"]

        gap_pct = (day_open / prev_close - 1) * 100  # Signed gap
        abs_gap = abs(gap_pct)

        # ── Gap filter: only trade days where gap is in our range ────
        if abs_gap < min_gap_pct or abs_gap > max_gap_pct:
            continue

        # Gap direction filter
        if gap_dir == "up" and gap_pct < 0:
            continue
        if gap_dir == "down" and gap_pct > 0:
            continue

        gap_days_seen += 1

        # ── Form the LATE opening range ──────────────────────────────
        late_bars = mkt[
            (mkt["time"] >= or_start) & (mkt["time"] < or_end_time)
        ]
        if len(late_bars) < or_bars:
            continue  # Not enough bars in the late window

        late_bars = late_bars.iloc[:or_bars]  # Exact number
        or_high = late_bars["high"].max()
        or_low = late_bars["low"].min()
        or_range = or_high - or_low
        or_mid = (or_high + or_low) / 2

        if or_range <= 0 or or_mid <= 0:
            continue

        or_range_pct = or_range / or_mid * 100

        # How much has price already moved from open to late-OR start?
        pre_move = (late_bars.iloc[0]["open"] / day_open - 1) * 100

        target_pts = or_range * target_mult
        stop_pts = or_range * stop_mult

        # ── Scan for breakout after late OR ends ─────────────────────
        remaining = mkt[mkt["time"] >= or_end_time]
        if remaining.empty:
            continue

        # Filter out bars past last_entry for new entries
        entry_window = remaining[remaining["time"] <= last_entry]
        if entry_window.empty:
            continue

        trade_taken = False

        for _, bar in entry_window.iterrows():
            if trade_taken:
                break

            # Determine allowed directions
            allowed_long = direction in ("long", "both")
            allowed_short = direction in ("short", "both")

            # Fade logic: only trade against the gap
            if fade_only:
                if gap_pct > 0:
                    allowed_long = False  # Gap up → only short (fade)
                else:
                    allowed_short = False  # Gap down → only long (fade)

            # Long breakout
            if allowed_long and bar["high"] > or_high:
                entry = or_high
                target = entry + target_pts
                stop = entry - stop_pts
                future = remaining[remaining["timestamp"] >= bar["timestamp"]]
                exit_price, reason = _find_exit(
                    future, entry, target, stop, "long"
                )
                pnl_pct = (exit_price - entry) / entry * 100

                trades.append({
                    "date": day,
                    "direction": "long",
                    "entry": entry,
                    "exit": exit_price,
                    "pnl_pct": pnl_pct,
                    "reason": reason,
                    "or_range_pct": or_range_pct,
                    "entry_time": bar["time"],
                    "gap_pct": gap_pct,
                    "gap_bucket": classify_gap(gap_pct),
                    "pre_move_pct": pre_move,
                    "or_start": str(or_start),
                    "dow": pd.Timestamp(day).dayofweek,
                    "is_fade": gap_pct > 0,  # long on gap-up = continuation
                })
                trade_taken = True
                gap_days_traded += 1

            # Short breakout
            elif allowed_short and bar["low"] < or_low:
                entry = or_low
                target = entry - target_pts
                stop = entry + stop_pts
                future = remaining[remaining["timestamp"] >= bar["timestamp"]]
                exit_price, reason = _find_exit(
                    future, entry, target, stop, "short"
                )
                pnl_pct = (entry - exit_price) / entry * 100

                trades.append({
                    "date": day,
                    "direction": "short",
                    "entry": entry,
                    "exit": exit_price,
                    "pnl_pct": pnl_pct,
                    "reason": reason,
                    "or_range_pct": or_range_pct,
                    "entry_time": bar["time"],
                    "gap_pct": gap_pct,
                    "gap_bucket": classify_gap(gap_pct),
                    "pre_move_pct": pre_move,
                    "or_start": str(or_start),
                    "dow": pd.Timestamp(day).dayofweek,
                    "is_fade": gap_pct < 0,  # short on gap-down = continuation
                })
                trade_taken = True
                gap_days_traded += 1

    result = pd.DataFrame(trades)
    result.attrs["gap_days_seen"] = gap_days_seen
    result.attrs["gap_days_traded"] = gap_days_traded
    return result


# ═══════════════════════════════════════════════════════════════════════
# Regular ORB backtest (for comparison baseline)
# ═══════════════════════════════════════════════════════════════════════

def backtest_regular_orb(df, params):
    """Standard ORB on non-gap days (the current production strategy)."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert(
            "America/New_York"
        ).dt.tz_localize(None)
    df["date"] = df["timestamp"].dt.date
    df["time"] = df["timestamp"].dt.time

    or_bars = params.get("or_bars", 3)
    target_mult = params.get("target_mult", 1.5)
    stop_mult = params.get("stop_mult", 1.0)
    direction = params.get("direction", "both")
    max_gap = params.get("max_gap_pct", 0.5)

    trades = []
    sorted_dates = sorted(df["date"].unique())

    for i, day in enumerate(sorted_dates):
        day_df = df[df["date"] == day]
        mkt = day_df[
            (day_df["time"] >= dtime(9, 30)) & (day_df["time"] <= dtime(15, 55))
        ]
        if len(mkt) < or_bars + 5:
            continue

        # Gap filter — skip gap days (those belong to late ORB)
        if i > 0:
            day_open = mkt.iloc[0]["open"]
            prev_day = sorted_dates[i - 1]
            prev_data = df[df["date"] == prev_day]
            if not prev_data.empty:
                prev_close = prev_data.iloc[-1]["close"]
                gap = abs(day_open / prev_close - 1) * 100
                if gap > max_gap:
                    continue

        or_data = mkt.iloc[:or_bars]
        or_high = or_data["high"].max()
        or_low = or_data["low"].min()
        or_range = or_high - or_low
        if or_range <= 0:
            continue

        target_pts = or_range * target_mult
        stop_pts = or_range * stop_mult
        remaining = mkt.iloc[or_bars:]
        trade_taken = False

        for _, bar in remaining.iterrows():
            if trade_taken:
                break
            if bar["time"] > dtime(14, 0):
                break

            if direction in ("long", "both") and bar["high"] > or_high:
                entry = or_high
                target = entry + target_pts
                stop = entry - stop_pts
                future = remaining[remaining["timestamp"] >= bar["timestamp"]]
                exit_price, reason = _find_exit(future, entry, target, stop, "long")
                pnl_pct = (exit_price - entry) / entry * 100
                trades.append({
                    "date": day, "direction": "long", "pnl_pct": pnl_pct,
                    "reason": reason, "entry_time": bar["time"],
                })
                trade_taken = True

            elif direction in ("short", "both") and bar["low"] < or_low:
                entry = or_low
                target = entry - target_pts
                stop = entry + stop_pts
                future = remaining[remaining["timestamp"] >= bar["timestamp"]]
                exit_price, reason = _find_exit(future, entry, target, stop, "short")
                pnl_pct = (entry - exit_price) / entry * 100
                trades.append({
                    "date": day, "direction": "short", "pnl_pct": pnl_pct,
                    "reason": reason, "entry_time": bar["time"],
                })
                trade_taken = True

    return pd.DataFrame(trades)


# ═══════════════════════════════════════════════════════════════════════
# Analysis helpers
# ═══════════════════════════════════════════════════════════════════════

def calc_stats(trades_df):
    """Calculate summary stats for a trade list."""
    if trades_df.empty or len(trades_df) == 0:
        return None
    t = trades_df
    n = len(t)
    wins = t[t["pnl_pct"] > 0]
    losses = t[t["pnl_pct"] <= 0]
    wr = len(wins) / n * 100
    gw = wins["pnl_pct"].sum() if len(wins) > 0 else 0
    gl = abs(losses["pnl_pct"].sum()) if len(losses) > 0 else 0.001
    pf = gw / gl
    total = t["pnl_pct"].sum()
    cum = t["pnl_pct"].cumsum()
    max_dd = (cum - cum.cummax()).min()
    avg = t["pnl_pct"].mean()
    return {
        "n": n, "win_rate": wr, "pf": pf, "total": total,
        "max_dd": max_dd, "avg": avg,
    }


def print_strategy_table(rows, title):
    """Print a formatted comparison table."""
    print(f"\n  {'='*80}")
    print(f"  {title}")
    print(f"  {'='*80}")
    print(f"  {'Strategy':<42} {'N':>5} {'Win%':>6} {'PF':>6} {'Total':>8} {'MaxDD':>8} {'Avg':>8}")
    print(f"  {'-'*85}")
    for label, stats in rows:
        if stats is None:
            print(f"  {label:<42} {'—':>5} {'—':>6} {'—':>6} {'—':>8} {'—':>8} {'—':>8}")
        else:
            print(f"  {label:<42} {stats['n']:>5} {stats['win_rate']:>5.1f}% "
                  f"{stats['pf']:>6.2f} {stats['total']:>+7.2f}% "
                  f"{stats['max_dd']:>+7.2f}% {stats['avg']:>+7.3f}%")


def analyze_gap_slices(trades, label):
    """Break down trades by gap size, direction, fade vs continuation."""
    if trades.empty:
        return

    print(f"\n  ┌─ Deep Dive: {label}")
    print(f"  │  Total: {len(trades)} trades, "
          f"{(trades['pnl_pct'] > 0).mean()*100:.1f}% win, "
          f"{trades['pnl_pct'].sum():+.2f}% total")

    # By gap bucket
    if "gap_bucket" in trades.columns:
        print(f"  │")
        print(f"  │  By gap size:")
        for bucket in ["small_gap", "medium_gap", "large_gap"]:
            bt = trades[trades["gap_bucket"] == bucket]
            if len(bt) >= 2:
                wr = (bt["pnl_pct"] > 0).mean() * 100
                print(f"  │    {bucket:<15} {len(bt):>3} trades, "
                      f"{wr:.0f}% win, {bt['pnl_pct'].mean():+.3f}% avg, "
                      f"{bt['pnl_pct'].sum():+.2f}% total")

    # Gap up vs gap down
    if "gap_pct" in trades.columns:
        print(f"  │")
        print(f"  │  Gap direction:")
        for label_dir, mask in [
            ("Gap UP", trades["gap_pct"] > 0),
            ("Gap DOWN", trades["gap_pct"] < 0),
        ]:
            bt = trades[mask]
            if len(bt) >= 2:
                wr = (bt["pnl_pct"] > 0).mean() * 100
                print(f"  │    {label_dir:<15} {len(bt):>3} trades, "
                      f"{wr:.0f}% win, {bt['pnl_pct'].mean():+.3f}% avg")

    # Fade vs continuation
    if "is_fade" in trades.columns:
        print(f"  │")
        print(f"  │  Fade vs Continuation:")
        for lbl, val in [("Fade gap", True), ("With gap", False)]:
            bt = trades[trades["is_fade"] == val]
            if len(bt) >= 2:
                wr = (bt["pnl_pct"] > 0).mean() * 100
                print(f"  │    {lbl:<15} {len(bt):>3} trades, "
                      f"{wr:.0f}% win, {bt['pnl_pct'].mean():+.3f}% avg")

    # Trade direction
    if "direction" in trades.columns:
        print(f"  │")
        print(f"  │  Trade direction:")
        for d in ["long", "short"]:
            bt = trades[trades["direction"] == d]
            if len(bt) >= 2:
                wr = (bt["pnl_pct"] > 0).mean() * 100
                print(f"  │    {d.upper():<15} {len(bt):>3} trades, "
                      f"{wr:.0f}% win, {bt['pnl_pct'].mean():+.3f}% avg")

    # Exit reasons
    if "reason" in trades.columns:
        rc = trades["reason"].value_counts()
        print(f"  │")
        print(f"  │  Exits: {', '.join(f'{r}={c}' for r, c in rc.items())}")

    # Day of week
    if "dow" in trades.columns and len(trades) >= 10:
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        print(f"  │")
        print(f"  │  Day of week:")
        for d in range(5):
            dt = trades[trades["dow"] == d]
            if len(dt) >= 2:
                wr = (dt["pnl_pct"] > 0).mean() * 100
                print(f"  │    {dow_names[d]}: {len(dt):>3} trades, "
                      f"{wr:.0f}% win, {dt['pnl_pct'].mean():+.3f}% avg")

    print(f"  └{'─'*55}")


# ═══════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════

def plot_late_orb_results(ticker, results_dict, baseline_trades=None):
    """
    Multi-panel plot comparing late ORB configs.
    results_dict: {label: trades_df}
    baseline_trades: regular ORB trades for comparison
    """
    # Filter to configs with enough trades
    valid = {k: v for k, v in results_dict.items()
             if not v.empty and len(v) >= 3}
    if not valid:
        print(f"  No plottable results for {ticker}")
        return

    n_panels = min(len(valid) + (1 if baseline_trades is not None else 0), 9)
    rows = 3
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(18, 14), facecolor="#0d1117")
    fig.suptitle(f"Late ORB Research: {ticker} — Gap Day Strategies (10:30 Long Focus)",
                 color="white", fontsize=14, fontweight="bold")

    all_axes = axes.flatten()
    for ax in all_axes:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="gray", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#333")

    panel_idx = 0

    # Baseline
    if baseline_trades is not None and not baseline_trades.empty:
        ax = all_axes[panel_idx]
        cum = baseline_trades["pnl_pct"].cumsum()
        ax.plot(range(len(cum)), cum, color="#888888", linewidth=1.5)
        ax.fill_between(range(len(cum)), 0, cum, alpha=0.08, color="#888")
        ax.axhline(0, color="#555", linewidth=0.5)
        ax.set_title(f"Baseline: Regular ORB (non-gap)\n"
                     f"{len(baseline_trades)} trades, "
                     f"{(baseline_trades['pnl_pct']>0).mean()*100:.0f}% win, "
                     f"PF {calc_stats(baseline_trades)['pf']:.2f}",
                     color="white", fontsize=9)
        ax.set_xlabel("Trade #", color="gray", fontsize=8)
        ax.set_ylabel("Cum P&L (%)", color="gray", fontsize=8)
        panel_idx += 1

    # Late ORB configs
    colors = ["#448aff", "#00e676", "#ff9800", "#e040fb", "#00bcd4",
              "#ff5252", "#76ff03", "#ffd740", "#40c4ff"]
    for i, (label, trades) in enumerate(valid.items()):
        if panel_idx >= 9:
            break
        ax = all_axes[panel_idx]
        cum = trades["pnl_pct"].cumsum()
        c = colors[i % len(colors)]
        ax.plot(range(len(cum)), cum, color=c, linewidth=1.5)
        ax.fill_between(range(len(cum)), 0, cum, alpha=0.1, color=c)
        ax.axhline(0, color="#555", linewidth=0.5)

        stats = calc_stats(trades)
        ax.set_title(f"{label}\n"
                     f"{stats['n']} trades, {stats['win_rate']:.0f}% win, "
                     f"PF {stats['pf']:.2f}, {stats['total']:+.2f}%",
                     color="white", fontsize=9)
        ax.set_xlabel("Trade #", color="gray", fontsize=8)
        ax.set_ylabel("Cum P&L (%)", color="gray", fontsize=8)
        panel_idx += 1

    # Hide unused panels
    for j in range(panel_idx, rows * cols):
        all_axes[j].set_visible(False)

    plt.tight_layout()
    save_path = f"late_orb_{ticker.lower()}_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"\n  Chart saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main research runner
# ═══════════════════════════════════════════════════════════════════════

def run_late_orb_research():
    print("=" * 80)
    print("  LATE ORB RESEARCH — Gap Day Second Entry Window")
    print("  Hypothesis: delayed OR on gap days can recover skipped days")
    print("=" * 80)

    # ── Parameter grid ───────────────────────────────────────────────
    late_or_configs = [
        # Core: vary the start time
        {"or_start_time": dtime(10, 0),  "or_bars": 3, "target_mult": 1.5,
         "stop_mult": 1.0, "direction": "both",
         "label": "10:00 start, 15min OR, 1.5:1"},
        {"or_start_time": dtime(10, 15), "or_bars": 3, "target_mult": 1.5,
         "stop_mult": 1.0, "direction": "both",
         "label": "10:15 start, 15min OR, 1.5:1"},
        {"or_start_time": dtime(10, 30), "or_bars": 3, "target_mult": 1.5,
         "stop_mult": 1.0, "direction": "both",
         "label": "10:30 start, 15min OR, 1.5:1"},

        # Wider late OR (30 min)
        {"or_start_time": dtime(10, 0),  "or_bars": 6, "target_mult": 1.5,
         "stop_mult": 1.0, "direction": "both",
         "label": "10:00 start, 30min OR, 1.5:1"},
        {"or_start_time": dtime(10, 15), "or_bars": 6, "target_mult": 1.5,
         "stop_mult": 1.0, "direction": "both",
         "label": "10:15 start, 30min OR, 1.5:1"},

        # Different R:R
        {"or_start_time": dtime(10, 0),  "or_bars": 3, "target_mult": 2.0,
         "stop_mult": 1.0, "direction": "both",
         "label": "10:00, 15min, 2:1 R:R"},
        {"or_start_time": dtime(10, 0),  "or_bars": 3, "target_mult": 1.0,
         "stop_mult": 1.0, "direction": "both",
         "label": "10:00, 15min, 1:1 R:R"},

        # Fade-only (trade against the gap)
        {"or_start_time": dtime(10, 0),  "or_bars": 3, "target_mult": 1.5,
         "stop_mult": 1.0, "direction": "both", "fade_only": True,
         "label": "10:00, 15min, 1.5:1, FADE only"},
        {"or_start_time": dtime(10, 15), "or_bars": 3, "target_mult": 1.5,
         "stop_mult": 1.0, "direction": "both", "fade_only": True,
         "label": "10:15, 15min, 1.5:1, FADE only"},

        # Long-only on gap-down days (mean reversion)
        {"or_start_time": dtime(10, 0),  "or_bars": 3, "target_mult": 1.5,
         "stop_mult": 1.0, "direction": "long", "gap_direction": "down",
         "label": "10:00, LONG on gap-DOWN days"},

        # Short-only on gap-up days (mean reversion)
        {"or_start_time": dtime(10, 0),  "or_bars": 3, "target_mult": 1.5,
         "stop_mult": 1.0, "direction": "short", "gap_direction": "up",
         "label": "10:00, SHORT on gap-UP days"},

        # ── 10:30 LONG-ONLY focused block ────────────────────────────
        # All gap days, long-only
        {"or_start_time": dtime(10, 30), "or_bars": 3, "target_mult": 1.5,
         "stop_mult": 1.0, "direction": "long",
         "label": "10:30, 15min, 1.5:1, LONG only"},
        {"or_start_time": dtime(10, 30), "or_bars": 3, "target_mult": 2.0,
         "stop_mult": 1.0, "direction": "long",
         "label": "10:30, 15min, 2:1, LONG only"},
        {"or_start_time": dtime(10, 30), "or_bars": 3, "target_mult": 1.0,
         "stop_mult": 1.0, "direction": "long",
         "label": "10:30, 15min, 1:1, LONG only"},

        # 10:30 long-only with 30min OR (more settling time)
        {"or_start_time": dtime(10, 30), "or_bars": 6, "target_mult": 1.5,
         "stop_mult": 1.0, "direction": "long",
         "label": "10:30, 30min, 1.5:1, LONG only"},
        {"or_start_time": dtime(10, 30), "or_bars": 6, "target_mult": 2.0,
         "stop_mult": 1.0, "direction": "long",
         "label": "10:30, 30min, 2:1, LONG only"},

        # 10:30 long-only, gap-down only (strongest signal from prev run)
        {"or_start_time": dtime(10, 30), "or_bars": 3, "target_mult": 1.5,
         "stop_mult": 1.0, "direction": "long", "gap_direction": "down",
         "label": "10:30, 15min, 1.5:1, LONG gap-DOWN"},
        {"or_start_time": dtime(10, 30), "or_bars": 3, "target_mult": 2.0,
         "stop_mult": 1.0, "direction": "long", "gap_direction": "down",
         "label": "10:30, 15min, 2:1, LONG gap-DOWN"},
        {"or_start_time": dtime(10, 30), "or_bars": 6, "target_mult": 1.5,
         "stop_mult": 1.0, "direction": "long", "gap_direction": "down",
         "label": "10:30, 30min, 1.5:1, LONG gap-DOWN"},

        # 10:30 long-only, gap-up only (continuation — does it work?)
        {"or_start_time": dtime(10, 30), "or_bars": 3, "target_mult": 1.5,
         "stop_mult": 1.0, "direction": "long", "gap_direction": "up",
         "label": "10:30, 15min, 1.5:1, LONG gap-UP"},

        # Tighter stop variant (0.75x)
        {"or_start_time": dtime(10, 30), "or_bars": 3, "target_mult": 1.5,
         "stop_mult": 0.75, "direction": "long",
         "label": "10:30, 15min, 1.5:0.75, LONG tight stop"},

        # Extended entry window (allow entries until 2:30 PM)
        {"or_start_time": dtime(10, 30), "or_bars": 3, "target_mult": 1.5,
         "stop_mult": 1.0, "direction": "long", "last_entry": dtime(14, 30),
         "label": "10:30, 15min, 1.5:1, LONG late entry"},
    ]

    has_data = False

    for ticker in TICKERS:
        df = load_intraday(ticker)
        if df.empty:
            continue
        has_data = True
        n_days = df["timestamp"].dt.date.nunique()

        # Count gap days in the dataset
        df_tmp = df.copy()
        df_tmp["timestamp"] = pd.to_datetime(df_tmp["timestamp"])
        if df_tmp["timestamp"].dt.tz is not None:
            df_tmp["timestamp"] = df_tmp["timestamp"].dt.tz_convert(
                "America/New_York"
            ).dt.tz_localize(None)
        df_tmp["date"] = df_tmp["timestamp"].dt.date
        sorted_d = sorted(df_tmp["date"].unique())
        gap_count = 0
        for idx in range(1, len(sorted_d)):
            d = sorted_d[idx]
            pd_data = df_tmp[df_tmp["date"] == sorted_d[idx - 1]]
            cd_data = df_tmp[df_tmp["date"] == d]
            if pd_data.empty or cd_data.empty:
                continue
            pc = pd_data.iloc[-1]["close"]
            co = cd_data[cd_data["timestamp"].dt.time >= dtime(9, 30)]
            if co.empty:
                continue
            g = abs(co.iloc[0]["open"] / pc - 1) * 100
            if g > GAP_THRESHOLD:
                gap_count += 1

        print(f"\n{'='*80}")
        print(f"  {ticker} — {n_days} trading days, {gap_count} gap days "
              f"(>{GAP_THRESHOLD}%) = {gap_count/max(n_days,1)*100:.0f}% of days")
        print(f"{'='*80}")

        # ── Baseline: regular ORB on non-gap days ────────────────────
        baseline = backtest_regular_orb(df, {
            "or_bars": 3, "target_mult": 1.5, "stop_mult": 1.0,
            "direction": "both", "max_gap_pct": 0.5,
        })

        # ── Run all late ORB configs ─────────────────────────────────
        results = {}
        table_rows = []

        # Baseline row
        bs = calc_stats(baseline)
        table_rows.append(("BASELINE: Regular ORB (gap<0.5%)", bs))

        for cfg in late_or_configs:
            label = cfg["label"]
            trades = backtest_late_orb(df, cfg)
            results[label] = trades
            stats = calc_stats(trades)
            gap_seen = trades.attrs.get("gap_days_seen", "?")
            gap_traded = trades.attrs.get("gap_days_traded", "?")
            table_rows.append((label, stats))

        print_strategy_table(table_rows, f"{ticker} — Late ORB Comparison")

        # ── Deep dive on top 5 configs ───────────────────────────────
        scored = []
        for label, trades in results.items():
            s = calc_stats(trades)
            if s and s["n"] >= 3:
                scored.append((label, s["pf"], trades))
        scored.sort(key=lambda x: x[1], reverse=True)

        print(f"\n  TOP CONFIGS — Detailed Breakdown:")
        for label, pf, trades in scored[:5]:
            analyze_gap_slices(trades, f"{ticker} — {label}")

        # ── 10:30 Long-Only Head-to-Head ─────────────────────────────
        long_1030_rows = []
        for label, trades in results.items():
            if "10:30" in label and "LONG" in label:
                long_1030_rows.append((label, calc_stats(trades)))
        if long_1030_rows:
            # Also include the original 10:30 both-direction for reference
            if "10:30 start, 15min OR, 1.5:1" in results:
                ref = results["10:30 start, 15min OR, 1.5:1"]
                long_1030_rows.insert(0, ("10:30 both-dir (reference)", calc_stats(ref)))
            print_strategy_table(long_1030_rows,
                                 f"{ticker} — 10:30 LONG-ONLY Head-to-Head")

        # ── Combined strategy estimate ───────────────────────────────
        # What if we run regular ORB on non-gap days + best late ORB on gap days?
        if scored and not baseline.empty:
            best_label, best_pf, best_trades = scored[0]
            combined_pnl = (
                baseline["pnl_pct"].sum() + best_trades["pnl_pct"].sum()
            )
            combined_n = len(baseline) + len(best_trades)
            combined_days = n_days

            print(f"\n  {'─'*70}")
            print(f"  COMBINED STRATEGY ESTIMATE (regular + best late ORB):")
            print(f"    Regular ORB:  {len(baseline)} trades → "
                  f"{baseline['pnl_pct'].sum():+.2f}%")
            print(f"    + Late ORB ({best_label}):  "
                  f"{len(best_trades)} trades → "
                  f"{best_trades['pnl_pct'].sum():+.2f}%")
            print(f"    = Combined:   {combined_n} trades → "
                  f"{combined_pnl:+.2f}% over {combined_days} days")
            print(f"    (vs regular-only: {baseline['pnl_pct'].sum():+.2f}% "
                  f"from {len(baseline)} trades)")
            print(f"  {'─'*70}")

            # Also show best 10:30 long-only combined if different
            long_1030_scored = [(l, pf, t) for l, pf, t in scored
                                if "10:30" in l and "LONG" in l]
            if long_1030_scored:
                bl, bpf, bt = long_1030_scored[0]
                if bl != best_label:
                    cp = baseline["pnl_pct"].sum() + bt["pnl_pct"].sum()
                    print(f"\n  COMBINED with best 10:30 LONG-ONLY:")
                    print(f"    Regular ORB + {bl}:")
                    print(f"    {len(baseline)} + {len(bt)} = "
                          f"{len(baseline)+len(bt)} trades → {cp:+.2f}%")
                    print(f"  {'─'*70}")

        # ── Plot ─────────────────────────────────────────────────────
        top_results = {l: t for l, _, t in scored[:8]}
        plot_late_orb_results(ticker, top_results, baseline)

    if not has_data:
        print("\n  No intraday data found.")
        print("  Run: python research/backtest_late_orb.py --collect")
        print("  (or: python research/orb_day_trading.py --collect)")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Late ORB backtest for gap days"
    )
    parser.add_argument(
        "--collect", action="store_true",
        help="Collect/refresh intraday data before running"
    )
    parser.add_argument(
        "--no-collect", action="store_true",
        help="Skip data collection (use cached data only)"
    )
    args = parser.parse_args()

    if args.collect:
        collect_intraday_data()

    if not args.no_collect and not args.collect:
        # Default: collect then run
        collect_intraday_data()

    run_late_orb_research()

    print("\n  NEXT STEPS:")
    print("  • If a late ORB config shows PF > 1.2 with 10+ trades, consider live testing")
    print("  • Add winning config to orb_trader.py as a secondary entry window")
    print("  • Run --collect weekly to accumulate more gap-day samples")
    print("  • 60 days of data → ~10-15 gap days — watch for small-sample noise")
