"""
research/orb_exit_method_backtest.py
--------------------------------------
Compares exit strategies for the volume-confirmed 2-min ORB:

1. BASELINE: Fixed 1.5x target + 1.0x stop (current)
2. Trailing stop at various distances (no fixed target)
3. Hybrid: take profit at target, then trail the remainder

All use 1-min bars, 2-min OR, volume confirmation, long-only, with costs.

Usage:
    python research/orb_exit_method_backtest.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import time as dtime
from tqdm import tqdm

sys.path.insert(0, ".")

CACHE_DIR = Path("data/intraday_cache")
UNIVERSE_PATH = Path("data/orb_universe.csv")


def load_1min(ticker):
    path = CACHE_DIR / f"{ticker}_1m.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def prepare(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    df["date"] = df["timestamp"].dt.date
    df["time"] = df["timestamp"].dt.time
    return df


# =====================================================================
# Exit methods
# =====================================================================

def exit_fixed(bars, entry, or_range, target_mult=1.5, stop_mult=1.0):
    """Current method: fixed target and stop."""
    target = entry + or_range * target_mult
    stop = entry - or_range * stop_mult
    for _, bar in bars.iterrows():
        if bar["low"] <= stop:
            return stop, "stop"
        if bar["high"] >= target:
            return target, "target"
        if bar["time"] >= dtime(15, 55):
            return bar["close"], "eod"
    if len(bars) > 0:
        return bars.iloc[-1]["close"], "eod"
    return entry, "flat"


def exit_trailing(bars, entry, or_range, trail_mult=0.5, stop_mult=1.0):
    """Trailing stop only, no fixed target. Trail distance = trail_mult * OR range."""
    stop = entry - or_range * stop_mult  # initial stop
    trail_dist = or_range * trail_mult
    highest = entry

    for _, bar in bars.iterrows():
        if bar["low"] <= stop:
            return stop, "stop"
        if bar["high"] > highest:
            highest = bar["high"]
            trail_stop = highest - trail_dist
            if trail_stop > stop:
                stop = trail_stop
        if bar["time"] >= dtime(15, 55):
            return bar["close"], "eod"
    if len(bars) > 0:
        return bars.iloc[-1]["close"], "eod"
    return entry, "flat"


def exit_hybrid(bars, entry, or_range, target_mult=1.5, trail_mult=0.5, stop_mult=1.0):
    """
    Fixed target for first portion, then trail the rest.
    Returns a blended P&L: 50% at target, 50% trailed.
    We simulate by tracking both exits separately.
    """
    target = entry + or_range * target_mult
    stop = entry - or_range * stop_mult
    trail_dist = or_range * trail_mult
    highest = entry

    target_hit = False
    target_price = None
    trail_exit_price = None

    for _, bar in bars.iterrows():
        if bar["low"] <= stop:
            if not target_hit:
                # Both halves stopped out
                return stop, "stop", stop, "stop"
            else:
                # First half already took profit, second half stopped
                return target_price, "target", stop, "trail_stop"

        if not target_hit and bar["high"] >= target:
            target_hit = True
            target_price = target
            # Reset stop to breakeven for trailing portion
            stop = entry
            highest = bar["high"]

        if target_hit:
            if bar["high"] > highest:
                highest = bar["high"]
                trail_stop = highest - trail_dist
                if trail_stop > stop:
                    stop = trail_stop

        if bar["time"] >= dtime(15, 55):
            if target_hit:
                return target_price, "target", bar["close"], "eod"
            else:
                return bar["close"], "eod", bar["close"], "eod"

    if len(bars) > 0:
        last = bars.iloc[-1]["close"]
        if target_hit:
            return target_price, "target", last, "eod"
        return last, "eod", last, "eod"
    return entry, "flat", entry, "flat"


# =====================================================================
# Backtest engine
# =====================================================================

def backtest_exit_method(df, exit_fn, or_minutes=2, max_gap_pct=0.5,
                          slippage_pct=0.02, commission_usd=1.0,
                          position_size_usd=950.0):
    """Run ORB backtest with a custom exit function. Long-only, volume-confirmed."""
    all_days = sorted(df["date"].unique())
    trades = []
    cost_pct = (slippage_pct * 2) + (commission_usd / position_size_usd * 100)
    or_end = dtime(9, 30 + or_minutes)

    for day, day_df in df.groupby("date"):
        mkt = day_df[(day_df["time"] >= dtime(9, 30)) & (day_df["time"] <= dtime(15, 55))]
        if len(mkt) < or_minutes + 10:
            continue

        or_data = mkt[mkt["time"] < or_end]
        if len(or_data) < or_minutes:
            continue

        or_high = or_data["high"].max()
        or_low = or_data["low"].min()
        or_range = or_high - or_low
        or_mid = (or_high + or_low) / 2
        if or_range <= 0 or or_mid <= 0:
            continue

        or_avg_volume = or_data["volume"].mean()

        # Gap filter
        day_idx = list(all_days).index(day) if day in all_days else -1
        if day_idx > 0 and max_gap_pct is not None:
            prev_day = all_days[day_idx - 1]
            prev_data = df[df["date"] == prev_day]
            if len(prev_data) > 0:
                prev_close = prev_data.iloc[-1]["close"]
                gap = abs(mkt.iloc[0]["open"] / prev_close - 1) * 100
                if gap > max_gap_pct:
                    continue

        remaining = mkt[mkt["time"] >= or_end]
        trade_taken = False

        for _, bar in remaining.iterrows():
            if trade_taken:
                break

            # Volume confirmation
            if bar["volume"] < or_avg_volume:
                continue

            # Long breakout only
            if bar["high"] > or_high:
                entry = or_high
                future = remaining[remaining["timestamp"] >= bar["timestamp"]]
                result = exit_fn(future, entry, or_range)

                # Handle hybrid (returns 4 values) vs simple (returns 2)
                if len(result) == 4:
                    p1, r1, p2, r2 = result
                    pnl1 = (p1 - entry) / entry * 100
                    pnl2 = (p2 - entry) / entry * 100
                    pnl_pct = (pnl1 + pnl2) / 2 - cost_pct  # 50/50 blend
                    reason = f"{r1}/{r2}"
                else:
                    exit_price, reason = result
                    pnl_pct = (exit_price - entry) / entry * 100 - cost_pct

                trades.append({
                    "date": day, "entry": entry, "pnl_pct": pnl_pct,
                    "reason": reason,
                })
                trade_taken = True

    return pd.DataFrame(trades)


def stats_line(trades, label):
    if trades.empty:
        print(f"  {label:<50} -- no trades --")
        return {}
    n = len(trades)
    wins = trades[trades["pnl_pct"] > 0]
    losses = trades[trades["pnl_pct"] <= 0]
    wr = len(wins) / n * 100
    gw = wins["pnl_pct"].sum() if len(wins) else 0
    gl = abs(losses["pnl_pct"].sum()) if len(losses) else 0.001
    pf = gw / gl
    total = trades["pnl_pct"].sum()
    avg = trades["pnl_pct"].mean()
    cum = trades["pnl_pct"].cumsum()
    dd = (cum - cum.cummax()).min()
    avg_win = wins["pnl_pct"].mean() if len(wins) else 0
    avg_loss = losses["pnl_pct"].mean() if len(losses) else 0

    reasons = trades["reason"].value_counts().to_dict()
    r_str = " ".join(f"{k}:{v}" for k, v in reasons.items())

    print(f"  {label:<50} {n:>4} tr  {wr:>5.1f}% WR  {pf:>5.2f} PF  "
          f"{total:>+8.2f}%  {avg:>+6.3f}% avg  {dd:>+6.2f}% DD")
    print(f"  {'':50} avg win: {avg_win:+.3f}%  avg loss: {avg_loss:+.3f}%  exits: {r_str}")

    return {"label": label, "trades": n, "wr": wr, "pf": pf,
            "total": total, "avg": avg, "dd": dd,
            "avg_win": avg_win, "avg_loss": avg_loss}


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  EXIT METHOD COMPARISON (2-min OR, volume confirmed, long-only, with costs)")
    print("=" * 80)

    universe = pd.read_csv(UNIVERSE_PATH)
    tickers = universe["ticker"].tolist()

    # Define exit strategies to test
    strategies = [
        ("BASELINE: fixed 1.5x target + 1.0x stop",
         lambda bars, entry, orr: exit_fixed(bars, entry, orr, 1.5, 1.0)),
        ("Fixed 2.0x target + 1.0x stop",
         lambda bars, entry, orr: exit_fixed(bars, entry, orr, 2.0, 1.0)),
        ("Fixed 1.0x target + 1.0x stop (1:1 R:R)",
         lambda bars, entry, orr: exit_fixed(bars, entry, orr, 1.0, 1.0)),
        ("Trail 0.3x OR (tight) + 1.0x initial stop",
         lambda bars, entry, orr: exit_trailing(bars, entry, orr, 0.3, 1.0)),
        ("Trail 0.5x OR + 1.0x initial stop",
         lambda bars, entry, orr: exit_trailing(bars, entry, orr, 0.5, 1.0)),
        ("Trail 0.75x OR + 1.0x initial stop",
         lambda bars, entry, orr: exit_trailing(bars, entry, orr, 0.75, 1.0)),
        ("Trail 1.0x OR + 1.0x initial stop",
         lambda bars, entry, orr: exit_trailing(bars, entry, orr, 1.0, 1.0)),
        ("Trail 1.5x OR (wide) + 1.0x initial stop",
         lambda bars, entry, orr: exit_trailing(bars, entry, orr, 1.5, 1.0)),
        ("Hybrid: 50% at 1.5x target, trail rest 0.3x",
         lambda bars, entry, orr: exit_hybrid(bars, entry, orr, 1.5, 0.3, 1.0)),
        ("Hybrid: 50% at 1.5x target, trail rest 0.5x",
         lambda bars, entry, orr: exit_hybrid(bars, entry, orr, 1.5, 0.5, 1.0)),
        ("Hybrid: 50% at 1.0x target, trail rest 0.5x",
         lambda bars, entry, orr: exit_hybrid(bars, entry, orr, 1.0, 0.5, 1.0)),
    ]

    # Aggregate across all tickers
    agg = {label: [] for label, _ in strategies}

    print(f"\n  Backtesting {len(strategies)} exit strategies across {len(tickers)} tickers...\n")
    for ticker in tqdm(tickers, desc="  Backtesting"):
        df = load_1min(ticker)
        if df.empty:
            continue
        try:
            df = prepare(df)
        except Exception:
            continue
        if df["date"].nunique() < 5:
            continue

        for label, exit_fn in strategies:
            trades = backtest_exit_method(df, exit_fn)
            if not trades.empty:
                trades["ticker"] = ticker
                agg[label].append(trades)

    # Results
    print(f"\n{'='*80}")
    print(f"  RESULTS (85 stocks pooled, $950 positions, long-only)")
    print(f"{'='*80}\n")

    all_stats = []
    for label, _ in strategies:
        if agg[label]:
            pooled = pd.concat(agg[label]).reset_index(drop=True)
        else:
            pooled = pd.DataFrame()
        s = stats_line(pooled, label)
        if s:
            all_stats.append(s)
        print()

    # Summary table
    print(f"{'='*80}")
    print(f"  RANKED BY PROFIT FACTOR")
    print(f"{'='*80}")
    print(f"  {'Strategy':<50} {'Trades':>5} {'WR':>7} {'PF':>6} {'Total':>9} {'MaxDD':>7}")
    print(f"  {'-'*85}")
    for s in sorted(all_stats, key=lambda x: x["pf"], reverse=True):
        print(f"  {s['label']:<50} {s['trades']:>5} {s['wr']:>6.1f}% {s['pf']:>5.2f} "
              f"{s['total']:>+8.2f}% {s['dd']:>+6.2f}%")
