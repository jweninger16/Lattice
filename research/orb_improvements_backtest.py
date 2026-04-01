"""
research/orb_improvements_backtest.py
--------------------------------------
Backtests three ORB entry improvements against out-of-sample data.

Improvement 1: Volume-confirmed breakout
  - Only enter if the breakout bar's volume exceeds the OR-period average volume
  - Filters out low-conviction fakeouts

Improvement 2: OR-width filter
  - Skip days where the opening range is too narrow (<0.15%) or too wide (>1.0%)
  - Tight ranges produce targets eaten by slippage; wide ranges mean the move already happened

Improvement 3: Pullback re-test entry
  - After breakout, wait for price to pull back to within 0.2x OR range of the boundary
  - Enter on the re-test rather than chasing the initial spike
  - Must happen within 30 minutes of breakout or skip

All three are compared against the BASELINE (current logic) on the same out-of-sample data.
Data is split: first 60% = in-sample (not used), last 40% = out-of-sample.

Usage:
    python research/orb_improvements_backtest.py
    python research/orb_improvements_backtest.py --collect   # refresh intraday cache first
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

sys.path.insert(0, ".")

CACHE_DIR = Path("data/intraday_cache")
TICKERS = ["SPY", "QQQ"]
OOS_FRACTION = 0.40  # last 40% of days = out-of-sample


# ═══════════════════════════════════════════════════════════════════════
# Data
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
                subset="timestamp").sort_values("timestamp")
            print(f"  {ticker}: merged -> {len(combined)} total bars")
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


def prepare_data(df):
    """Prepare dataframe with date/time columns and timezone handling."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    df["date"] = df["timestamp"].dt.date
    df["time"] = df["timestamp"].dt.time
    return df


def split_oos(df):
    """Split into in-sample / out-of-sample by date."""
    all_days = sorted(df["date"].unique())
    n_days = len(all_days)
    split_idx = int(n_days * (1 - OOS_FRACTION))
    oos_start = all_days[split_idx]
    oos_df = df[df["date"] >= oos_start].copy()
    return oos_df, all_days[:split_idx], all_days[split_idx:]


# ═══════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════

def find_exit(bars, entry, target, stop, direction):
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
        if bar["time"] >= dtime(15, 55):
            return bar["close"], "eod"
    if len(bars) > 0:
        return bars.iloc[-1]["close"], "eod"
    return entry, "flat"


def get_prev_close(df, day, prev_days):
    """Get previous day's closing price."""
    day_idx = list(prev_days).index(day) if day in prev_days else -1
    if day_idx > 0:
        prev_day = prev_days[day_idx - 1]
        prev_data = df[df["date"] == prev_day]
        if len(prev_data) > 0:
            return prev_data.iloc[-1]["close"]
    return None


# ═══════════════════════════════════════════════════════════════════════
# BASELINE: Current ORB logic (from orb_day_trading.py)
# ═══════════════════════════════════════════════════════════════════════

def backtest_baseline(df, direction="both", max_gap_pct=0.5):
    """Current ORB: breakout on first cross of OR high/low."""
    or_bars = 3  # 15 min
    target_mult = 1.5
    stop_mult = 1.0
    all_days = sorted(df["date"].unique())
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

        # Gap filter
        prev_close = get_prev_close(df, day, all_days)
        if prev_close is not None and max_gap_pct is not None:
            gap = abs(mkt.iloc[0]["open"] / prev_close - 1) * 100
            if gap > max_gap_pct:
                continue

        target_pts = or_range * target_mult
        stop_pts = or_range * stop_mult
        remaining = mkt.iloc[or_bars:]
        trade_taken = False

        for _, bar in remaining.iterrows():
            if trade_taken:
                break

            if direction in ("long", "both") and bar["high"] > or_high:
                entry = or_high
                target = entry + target_pts
                stop = entry - stop_pts
                future = remaining[remaining["timestamp"] >= bar["timestamp"]]
                exit_price, reason = find_exit(future, entry, target, stop, "long")
                pnl_pct = (exit_price - entry) / entry * 100
                trades.append({"date": day, "direction": "long", "entry": entry,
                               "exit": exit_price, "pnl_pct": pnl_pct, "reason": reason,
                               "or_range_pct": or_range / or_mid * 100})
                trade_taken = True

            elif direction in ("short", "both") and bar["low"] < or_low:
                entry = or_low
                target = entry - target_pts
                stop = entry + stop_pts
                future = remaining[remaining["timestamp"] >= bar["timestamp"]]
                exit_price, reason = find_exit(future, entry, target, stop, "short")
                pnl_pct = (entry - exit_price) / entry * 100
                trades.append({"date": day, "direction": "short", "entry": entry,
                               "exit": exit_price, "pnl_pct": pnl_pct, "reason": reason,
                               "or_range_pct": or_range / or_mid * 100})
                trade_taken = True

    return pd.DataFrame(trades)


# ═══════════════════════════════════════════════════════════════════════
# IMPROVEMENT 1: Volume-confirmed breakout
# ═══════════════════════════════════════════════════════════════════════

def backtest_volume_confirmed(df, direction="both", max_gap_pct=0.5):
    """Only enter if breakout bar volume > average OR bar volume."""
    or_bars = 3
    target_mult = 1.5
    stop_mult = 1.0
    all_days = sorted(df["date"].unique())
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

        # Average volume during OR period (threshold for confirmation)
        or_avg_volume = or_data["volume"].mean()

        prev_close = get_prev_close(df, day, all_days)
        if prev_close is not None and max_gap_pct is not None:
            gap = abs(mkt.iloc[0]["open"] / prev_close - 1) * 100
            if gap > max_gap_pct:
                continue

        target_pts = or_range * target_mult
        stop_pts = or_range * stop_mult
        remaining = mkt.iloc[or_bars:]
        trade_taken = False

        for _, bar in remaining.iterrows():
            if trade_taken:
                break

            # IMPROVEMENT: require breakout bar volume > OR average
            if bar["volume"] < or_avg_volume:
                continue

            if direction in ("long", "both") and bar["high"] > or_high:
                entry = or_high
                target = entry + target_pts
                stop = entry - stop_pts
                future = remaining[remaining["timestamp"] >= bar["timestamp"]]
                exit_price, reason = find_exit(future, entry, target, stop, "long")
                pnl_pct = (exit_price - entry) / entry * 100
                trades.append({"date": day, "direction": "long", "entry": entry,
                               "exit": exit_price, "pnl_pct": pnl_pct, "reason": reason,
                               "or_range_pct": or_range / or_mid * 100})
                trade_taken = True

            elif direction in ("short", "both") and bar["low"] < or_low:
                entry = or_low
                target = entry - target_pts
                stop = entry + stop_pts
                future = remaining[remaining["timestamp"] >= bar["timestamp"]]
                exit_price, reason = find_exit(future, entry, target, stop, "short")
                pnl_pct = (entry - exit_price) / entry * 100
                trades.append({"date": day, "direction": "short", "entry": entry,
                               "exit": exit_price, "pnl_pct": pnl_pct, "reason": reason,
                               "or_range_pct": or_range / or_mid * 100})
                trade_taken = True

    return pd.DataFrame(trades)


# ═══════════════════════════════════════════════════════════════════════
# IMPROVEMENT 2: OR-width filter
# ═══════════════════════════════════════════════════════════════════════

def backtest_or_width_filter(df, direction="both", max_gap_pct=0.5,
                              min_or_pct=0.15, max_or_pct=1.0):
    """Skip days where OR range is too narrow or too wide."""
    or_bars = 3
    target_mult = 1.5
    stop_mult = 1.0
    all_days = sorted(df["date"].unique())
    trades = []
    skipped_narrow = 0
    skipped_wide = 0

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

        # IMPROVEMENT: skip extreme OR widths
        if or_range_pct < min_or_pct:
            skipped_narrow += 1
            continue
        if or_range_pct > max_or_pct:
            skipped_wide += 1
            continue

        prev_close = get_prev_close(df, day, all_days)
        if prev_close is not None and max_gap_pct is not None:
            gap = abs(mkt.iloc[0]["open"] / prev_close - 1) * 100
            if gap > max_gap_pct:
                continue

        target_pts = or_range * target_mult
        stop_pts = or_range * stop_mult
        remaining = mkt.iloc[or_bars:]
        trade_taken = False

        for _, bar in remaining.iterrows():
            if trade_taken:
                break

            if direction in ("long", "both") and bar["high"] > or_high:
                entry = or_high
                target = entry + target_pts
                stop = entry - stop_pts
                future = remaining[remaining["timestamp"] >= bar["timestamp"]]
                exit_price, reason = find_exit(future, entry, target, stop, "long")
                pnl_pct = (exit_price - entry) / entry * 100
                trades.append({"date": day, "direction": "long", "entry": entry,
                               "exit": exit_price, "pnl_pct": pnl_pct, "reason": reason,
                               "or_range_pct": or_range_pct})
                trade_taken = True

            elif direction in ("short", "both") and bar["low"] < or_low:
                entry = or_low
                target = entry - target_pts
                stop = entry + stop_pts
                future = remaining[remaining["timestamp"] >= bar["timestamp"]]
                exit_price, reason = find_exit(future, entry, target, stop, "short")
                pnl_pct = (entry - exit_price) / entry * 100
                trades.append({"date": day, "direction": "short", "entry": entry,
                               "exit": exit_price, "pnl_pct": pnl_pct, "reason": reason,
                               "or_range_pct": or_range_pct})
                trade_taken = True

    return pd.DataFrame(trades), skipped_narrow, skipped_wide


# ═══════════════════════════════════════════════════════════════════════
# IMPROVEMENT 3: Pullback re-test entry
# ═══════════════════════════════════════════════════════════════════════

def backtest_pullback_retest(df, direction="both", max_gap_pct=0.5,
                              pullback_threshold=0.2, max_wait_bars=6):
    """
    After breakout, wait for price to pull back to within (pullback_threshold * OR range)
    of the OR boundary, then enter. Must happen within max_wait_bars (30 min at 5-min bars).
    """
    or_bars = 3
    target_mult = 1.5
    stop_mult = 1.0
    all_days = sorted(df["date"].unique())
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

        prev_close = get_prev_close(df, day, all_days)
        if prev_close is not None and max_gap_pct is not None:
            gap = abs(mkt.iloc[0]["open"] / prev_close - 1) * 100
            if gap > max_gap_pct:
                continue

        target_pts = or_range * target_mult
        stop_pts = or_range * stop_mult
        remaining = mkt.iloc[or_bars:]
        trade_taken = False
        breakout_detected = None  # ("long"/"short", bar_index)
        bars_since_breakout = 0

        remaining_list = list(remaining.iterrows())

        for i, (idx, bar) in enumerate(remaining_list):
            if trade_taken:
                break

            # Phase 1: Detect breakout (but don't enter yet)
            if breakout_detected is None:
                if direction in ("long", "both") and bar["high"] > or_high:
                    breakout_detected = "long"
                    bars_since_breakout = 0
                elif direction in ("short", "both") and bar["low"] < or_low:
                    breakout_detected = "short"
                    bars_since_breakout = 0
                continue

            # Phase 2: Wait for pullback re-test
            bars_since_breakout += 1
            if bars_since_breakout > max_wait_bars:
                # Missed the re-test window — skip this day
                break

            pullback_zone = or_range * pullback_threshold

            if breakout_detected == "long":
                # Price must dip back to within pullback_zone of OR high
                if bar["low"] <= or_high + pullback_zone:
                    # Re-test confirmed — enter long
                    entry = or_high
                    target = entry + target_pts
                    stop = entry - stop_pts
                    # Don't enter if price broke below stop during pullback
                    if bar["low"] < stop:
                        break
                    future = remaining[remaining["timestamp"] >= bar["timestamp"]]
                    exit_price, reason = find_exit(future, entry, target, stop, "long")
                    pnl_pct = (exit_price - entry) / entry * 100
                    trades.append({"date": day, "direction": "long", "entry": entry,
                                   "exit": exit_price, "pnl_pct": pnl_pct, "reason": reason,
                                   "or_range_pct": or_range / or_mid * 100,
                                   "bars_to_retest": bars_since_breakout})
                    trade_taken = True

            elif breakout_detected == "short":
                # Price must bounce back up to within pullback_zone of OR low
                if bar["high"] >= or_low - pullback_zone:
                    entry = or_low
                    target = entry - target_pts
                    stop = entry + stop_pts
                    if bar["high"] > stop:
                        break
                    future = remaining[remaining["timestamp"] >= bar["timestamp"]]
                    exit_price, reason = find_exit(future, entry, target, stop, "short")
                    pnl_pct = (entry - exit_price) / entry * 100
                    trades.append({"date": day, "direction": "short", "entry": entry,
                                   "exit": exit_price, "pnl_pct": pnl_pct, "reason": reason,
                                   "or_range_pct": or_range / or_mid * 100,
                                   "bars_to_retest": bars_since_breakout})
                    trade_taken = True

    return pd.DataFrame(trades)


# ═══════════════════════════════════════════════════════════════════════
# Analysis & Reporting
# ═══════════════════════════════════════════════════════════════════════

def compute_stats(trades, label):
    """Compute and print stats for a set of trades."""
    if trades.empty:
        print(f"\n  {label}: NO TRADES")
        return {}

    n = len(trades)
    wins = trades[trades["pnl_pct"] > 0]
    losses = trades[trades["pnl_pct"] <= 0]
    win_rate = len(wins) / n * 100
    avg_pnl = trades["pnl_pct"].mean()
    avg_win = wins["pnl_pct"].mean() if len(wins) > 0 else 0
    avg_loss = losses["pnl_pct"].mean() if len(losses) > 0 else 0
    gross_wins = wins["pnl_pct"].sum() if len(wins) > 0 else 0
    gross_losses = abs(losses["pnl_pct"].sum()) if len(losses) > 0 else 0.001
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")
    total = trades["pnl_pct"].sum()

    cum = trades["pnl_pct"].cumsum()
    max_dd = (cum - cum.cummax()).min()

    # Max consecutive losses
    is_loss = (trades["pnl_pct"] <= 0).astype(int).values
    max_consec = 0
    streak = 0
    for v in is_loss:
        if v:
            streak += 1
            max_consec = max(max_consec, streak)
        else:
            streak = 0

    # Exit breakdown
    reason_counts = trades["reason"].value_counts().to_dict()

    stats = {
        "label": label, "trades": n, "win_rate": win_rate, "pf": pf,
        "total_pnl": total, "avg_pnl": avg_pnl, "avg_win": avg_win,
        "avg_loss": avg_loss, "max_dd": max_dd, "max_consec_loss": max_consec,
        "exits": reason_counts,
    }

    print(f"\n  {'='*60}")
    print(f"  {label}")
    print(f"  {'='*60}")
    print(f"  Trades: {n}  |  Win Rate: {win_rate:.1f}%  |  Profit Factor: {pf:.2f}")
    print(f"  Total P&L: {total:+.2f}%  |  Avg Trade: {avg_pnl:+.3f}%")
    print(f"  Avg Win: {avg_win:+.3f}%  |  Avg Loss: {avg_loss:+.3f}%")
    print(f"  Max Drawdown: {max_dd:+.2f}%  |  Max Consec Loss: {max_consec}")
    exits_str = ", ".join(f"{r}={c}" for r, c in reason_counts.items())
    print(f"  Exits: {exits_str}")

    return stats


def plot_comparison(all_results, ticker, save_path):
    """Plot equity curves for all strategies side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#0d1117")
    fig.suptitle(f"ORB Improvements — {ticker} (Out-of-Sample)",
                 color="white", fontsize=14, fontweight="bold")

    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="gray")
        for spine in ax.spines.values():
            spine.set_color("#333")

    # Equity curves
    ax1 = axes[0]
    colors = ["#888888", "#00e676", "#448aff", "#ff9800"]
    for i, (label, trades) in enumerate(all_results.items()):
        if trades.empty:
            continue
        cum = trades["pnl_pct"].cumsum()
        ax1.plot(range(len(cum)), cum, color=colors[i % len(colors)],
                 linewidth=1.8, label=label, alpha=0.9)
    ax1.axhline(0, color="#555", linewidth=0.5)
    ax1.set_xlabel("Trade #", color="gray")
    ax1.set_ylabel("Cumulative P&L (%)", color="gray")
    ax1.set_title("Equity Curves", color="white")
    ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8, loc="upper left")

    # Bar chart comparison
    ax2 = axes[1]
    labels = []
    win_rates = []
    pfs = []
    totals = []
    for label, trades in all_results.items():
        if trades.empty:
            continue
        n = len(trades)
        wr = (trades["pnl_pct"] > 0).mean() * 100
        gw = trades[trades["pnl_pct"] > 0]["pnl_pct"].sum()
        gl = abs(trades[trades["pnl_pct"] <= 0]["pnl_pct"].sum()) or 0.001
        labels.append(label.replace("Improvement ", "Imp "))
        win_rates.append(wr)
        pfs.append(gw / gl)
        totals.append(trades["pnl_pct"].sum())

    x = np.arange(len(labels))
    width = 0.25
    ax2.bar(x - width, win_rates, width, label="Win Rate %", color="#00e676", alpha=0.7)
    ax2.bar(x, [p * 20 for p in pfs], width, label="PF (x20 scale)", color="#448aff", alpha=0.7)
    ax2.bar(x + width, totals, width, label="Total P&L %", color="#ff9800", alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, color="gray", fontsize=8, rotation=15)
    ax2.set_title("Strategy Comparison", color="white")
    ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    ax2.axhline(0, color="#555", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"\n  Chart saved to {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def run_all():
    has_data = False

    for ticker in TICKERS:
        df = load_intraday(ticker)
        if df.empty:
            print(f"  No data for {ticker}. Run with --collect first.")
            continue
        has_data = True
        df = prepare_data(df)

        # Split into OOS
        oos_df, is_days, oos_days = split_oos(df)
        n_is = len(is_days)
        n_oos = len(oos_days)

        print(f"\n{'='*65}")
        print(f"  {ticker} — {n_is + n_oos} total days")
        print(f"  In-sample: {n_is} days (not used)  |  Out-of-sample: {n_oos} days")
        print(f"  OOS period: {oos_days[0]} to {oos_days[-1]}")
        print(f"{'='*65}")

        # Run all strategies on OOS data only
        baseline = backtest_baseline(oos_df)
        stats_baseline = compute_stats(baseline, "BASELINE (current logic)")

        vol_confirmed = backtest_volume_confirmed(oos_df)
        stats_vol = compute_stats(vol_confirmed, "Improvement 1: Volume-Confirmed Breakout")

        width_filtered, skip_narrow, skip_wide = backtest_or_width_filter(oos_df)
        stats_width = compute_stats(width_filtered, "Improvement 2: OR-Width Filter")
        print(f"  (Skipped: {skip_narrow} narrow + {skip_wide} wide = {skip_narrow + skip_wide} days filtered)")

        pullback = backtest_pullback_retest(oos_df)
        stats_pb = compute_stats(pullback, "Improvement 3: Pullback Re-Test Entry")

        # Summary table
        print(f"\n  {'-'*60}")
        print(f"  SUMMARY -- {ticker} Out-of-Sample ({n_oos} days)")
        print(f"  {'-'*60}")
        print(f"  {'Strategy':<35} {'Trades':>6} {'Win%':>6} {'PF':>6} {'Total':>8} {'MaxDD':>7}")
        print(f"  {'-'*70}")
        for s in [stats_baseline, stats_vol, stats_width, stats_pb]:
            if not s:
                continue
            print(f"  {s['label']:<35} {s['trades']:>6} {s['win_rate']:>5.1f}% "
                  f"{s['pf']:>5.2f} {s['total_pnl']:>+7.2f}% {s['max_dd']:>+6.2f}%")

        # Plot
        all_results = {
            "Baseline": baseline,
            "Improvement 1: Volume": vol_confirmed,
            "Improvement 2: Width Filter": width_filtered,
            "Improvement 3: Pullback": pullback,
        }
        Path("research").mkdir(exist_ok=True)
        plot_comparison(all_results, ticker,
                        f"research/orb_improvements_{ticker.lower()}.png")

    if not has_data:
        print("\n  No intraday data found. Run:")
        print("    python research/orb_improvements_backtest.py --collect")


def monthly_breakdown(trades, label):
    """Print month-by-month stats for a set of trades."""
    if trades.empty:
        print(f"\n  {label}: NO TRADES")
        return pd.DataFrame()

    t = trades.copy()
    t["month"] = pd.to_datetime(t["date"]).dt.to_period("M")

    rows = []
    for month, grp in t.groupby("month"):
        n = len(grp)
        wins = grp[grp["pnl_pct"] > 0]
        losses = grp[grp["pnl_pct"] <= 0]
        wr = len(wins) / n * 100 if n else 0
        gw = wins["pnl_pct"].sum() if len(wins) else 0
        gl = abs(losses["pnl_pct"].sum()) if len(losses) else 0.001
        pf = gw / gl if gl > 0 else float("inf")
        total = grp["pnl_pct"].sum()
        avg = grp["pnl_pct"].mean()
        cum = grp["pnl_pct"].cumsum()
        dd = (cum - cum.cummax()).min()
        rows.append({
            "month": str(month), "trades": n, "wins": len(wins),
            "losses": len(losses), "win_rate": wr, "pf": pf,
            "total_pnl": total, "avg_pnl": avg, "max_dd": dd,
        })

    mdf = pd.DataFrame(rows)
    return mdf


def run_volume_deep_dive():
    """Run baseline vs volume-confirmed on full + IS + OOS with monthly breakdowns."""

    for ticker in TICKERS:
        df = load_intraday(ticker)
        if df.empty:
            print(f"  No data for {ticker}. Run with --collect first.")
            continue
        df = prepare_data(df)

        all_days = sorted(df["date"].unique())
        n_days = len(all_days)
        split_idx = int(n_days * (1 - OOS_FRACTION))
        is_days = all_days[:split_idx]
        oos_days = all_days[split_idx:]
        is_df = df[df["date"] < oos_days[0]].copy()
        oos_df = df[df["date"] >= oos_days[0]].copy()

        print(f"\n{'='*70}")
        print(f"  {ticker} -- VOLUME CONFIRMATION DEEP DIVE")
        print(f"  Total: {n_days} days | IS: {len(is_days)} days ({is_days[0]} to {is_days[-1]})")
        print(f"  OOS: {len(oos_days)} days ({oos_days[0]} to {oos_days[-1]})")
        print(f"{'='*70}")

        # Run on all three splits
        for split_label, split_df in [
            ("IN-SAMPLE", is_df),
            ("OUT-OF-SAMPLE", oos_df),
            ("FULL DATASET", df),
        ]:
            print(f"\n  {'='*65}")
            print(f"  {split_label} ({split_df['date'].nunique()} days)")
            print(f"  {'='*65}")

            baseline = backtest_baseline(split_df)
            vol_conf = backtest_volume_confirmed(split_df)

            compute_stats(baseline, f"Baseline ({split_label})")
            compute_stats(vol_conf, f"Volume-Confirmed ({split_label})")

            # Monthly breakdowns side by side
            base_monthly = monthly_breakdown(baseline, "Baseline")
            vol_monthly = monthly_breakdown(vol_conf, "Volume-Confirmed")

            if base_monthly.empty and vol_monthly.empty:
                continue

            print(f"\n  MONTHLY BREAKDOWN -- {split_label}")
            print(f"  {'-'*80}")
            print(f"  {'Month':<10} | {'--- Baseline ---':^30} | {'--- Volume-Confirmed ---':^30}")
            print(f"  {'':10} | {'Trades':>6} {'Win%':>6} {'PF':>6} {'P&L':>8} | "
                  f"{'Trades':>6} {'Win%':>6} {'PF':>6} {'P&L':>8}")
            print(f"  {'-'*80}")

            all_months = sorted(set(
                list(base_monthly["month"].values if not base_monthly.empty else []) +
                list(vol_monthly["month"].values if not vol_monthly.empty else [])
            ))

            for m in all_months:
                b = base_monthly[base_monthly["month"] == m]
                v = vol_monthly[vol_monthly["month"] == m]

                if not b.empty:
                    br = b.iloc[0]
                    b_str = f"{br['trades']:>6} {br['win_rate']:>5.1f}% {br['pf']:>5.2f} {br['total_pnl']:>+7.2f}%"
                else:
                    b_str = f"{'--':>6} {'--':>6} {'--':>6} {'--':>8}"

                if not v.empty:
                    vr = v.iloc[0]
                    v_str = f"{vr['trades']:>6} {vr['win_rate']:>5.1f}% {vr['pf']:>5.2f} {vr['total_pnl']:>+7.2f}%"
                else:
                    v_str = f"{'--':>6} {'--':>6} {'--':>6} {'--':>8}"

                print(f"  {m:<10} | {b_str} | {v_str}")

            # Totals row
            if not base_monthly.empty and not vol_monthly.empty:
                bt = base_monthly
                vt = vol_monthly
                b_tot = f"{bt['trades'].sum():>6} {(bt['wins'].sum()/max(bt['trades'].sum(),1)*100):>5.1f}% " \
                        f"{'':>6} {bt['total_pnl'].sum():>+7.2f}%"
                v_tot = f"{vt['trades'].sum():>6} {(vt['wins'].sum()/max(vt['trades'].sum(),1)*100):>5.1f}% " \
                        f"{'':>6} {vt['total_pnl'].sum():>+7.2f}%"
                print(f"  {'-'*80}")
                print(f"  {'TOTAL':<10} | {b_tot} | {v_tot}")

        # Trade-level detail for volume-confirmed on full dataset
        vol_full = backtest_volume_confirmed(df)
        if not vol_full.empty:
            print(f"\n  {'='*65}")
            print(f"  INDIVIDUAL TRADES -- Volume-Confirmed ({ticker}, full dataset)")
            print(f"  {'='*65}")
            print(f"  {'Date':<12} {'Dir':<6} {'Entry':>8} {'Exit':>8} {'P&L':>8} {'Reason':<8} {'OR%':>6}")
            print(f"  {'-'*65}")
            for _, t in vol_full.iterrows():
                print(f"  {str(t['date']):<12} {t['direction']:<6} "
                      f"${t['entry']:>7.2f} ${t['exit']:>7.2f} "
                      f"{t['pnl_pct']:>+7.3f}% {t['reason']:<8} "
                      f"{t['or_range_pct']:>5.2f}%")

            # Flag potential outliers (> 2 stdev from mean)
            mean_pnl = vol_full["pnl_pct"].mean()
            std_pnl = vol_full["pnl_pct"].std()
            outliers = vol_full[abs(vol_full["pnl_pct"] - mean_pnl) > 2 * std_pnl]
            if not outliers.empty:
                print(f"\n  OUTLIER TRADES (>2 stdev from mean {mean_pnl:+.3f}%):")
                for _, t in outliers.iterrows():
                    print(f"    {t['date']} {t['direction']} {t['pnl_pct']:+.3f}%")
            else:
                print(f"\n  No outlier trades (all within 2 stdev of mean {mean_pnl:+.3f}%)")

            # Winning months vs losing months
            mdf = monthly_breakdown(vol_full, "")
            if not mdf.empty:
                win_months = len(mdf[mdf["total_pnl"] > 0])
                lose_months = len(mdf[mdf["total_pnl"] <= 0])
                print(f"\n  Monthly consistency: {win_months} winning months, "
                      f"{lose_months} losing months")


if __name__ == "__main__":
    if "--collect" in sys.argv:
        collect_intraday_data()

    if "--volume-deep-dive" in sys.argv or "--volume" in sys.argv:
        print("\n" + "=" * 70)
        print("  VOLUME CONFIRMATION -- Deep Dive (IS + OOS + Monthly)")
        print("=" * 70)
        run_volume_deep_dive()
    else:
        print("\n" + "=" * 65)
        print("  ORB ENTRY IMPROVEMENTS -- Out-of-Sample Backtest")
        print("=" * 65)
        run_all()
