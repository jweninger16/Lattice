"""
research/orb_exit_method_backtest.py
------------------------------------
Head-to-head comparison of ORB exit strategies on 87-stock universe.
Uses 1-min bars for realistic simulation.

Now that PDT rule is gone and unsettled funds can be reused:
- Tests multiple trades per day
- Tests different exit methods
- Uses $1,900 account with $5.50 round-trip cost

Exit methods tested:
  A. Fixed bracket: stop at OR low, target at R:R multiple (1.5x, 2x, 3x)
  B. Fixed bracket + 30-min time stop
  C. Trailing stop (current live logic, 0.3x OR range)
  D. Trailing stop with activation threshold (new logic)
  E. Time-based exit only (exit after N minutes regardless)

Also tests: 1 trade/day vs up to 3 trades/day (sequential, reuse funds)

Usage:
    python research/orb_exit_method_backtest.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import time as dtime, timedelta

sys.path.insert(0, ".")

CACHE_DIR = Path("data/intraday_cache")
UNIVERSE = Path("data/orb_universe.csv")
POSITION_SIZE = 1900
RT_COST = 5.50  # round-trip commission + SEC fees
MAX_GAP_PCT = 1.0
OR_MINUTES = 2  # 2 one-minute bars
MIN_OR_RANGE_PCT = 0.40  # filter narrow ranges
SLIPPAGE = 0.01  # $0.01 per share slippage on entry


def load_universe():
    df = pd.read_csv(UNIVERSE)
    return df["ticker"].tolist()


def load_1m(ticker):
    path = CACHE_DIR / f"{ticker}_1m.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def compute_or(day_bars):
    """Compute opening range from first OR_MINUTES 1-min bars."""
    market_open = dtime(13, 30)  # 9:30 ET in UTC
    or_end = dtime(13, 30 + OR_MINUTES)

    or_bars = day_bars[
        (day_bars["timestamp"].dt.time >= market_open) &
        (day_bars["timestamp"].dt.time < or_end)
    ]
    if len(or_bars) < OR_MINUTES:
        return None

    or_high = or_bars["high"].max()
    or_low = or_bars["low"].min()
    or_range = or_high - or_low
    or_close = or_bars.iloc[-1]["close"]
    or_avg_vol = or_bars["volume"].mean()

    if or_range <= 0:
        return None

    return {
        "or_high": or_high,
        "or_low": or_low,
        "or_range": or_range,
        "or_close": or_close,
        "or_avg_vol": or_avg_vol,
    }


def get_prev_close(ticker, date, all_data):
    """Get previous day's close for gap filter."""
    prev_days = all_data[all_data["timestamp"].dt.date < date]
    if prev_days.empty:
        return None
    return prev_days.iloc[-1]["close"]


def simulate_trade(post_or_bars, entry_price, or_data, method, params):
    """
    Simulate a single trade bar-by-bar using 1-min data.

    Returns dict with: exit_price, exit_time, exit_reason, bars_held
    """
    or_range = or_data["or_range"]
    or_low = or_data["or_low"]

    # Common: initial stop at OR low
    stop = or_low

    if method == "fixed_bracket":
        rr_mult = params.get("rr_mult", 1.5)
        time_stop_min = params.get("time_stop_min", None)
        risk = entry_price - or_low
        target = entry_price + risk * rr_mult

        for i, (_, bar) in enumerate(post_or_bars.iterrows()):
            # Check time stop first
            if time_stop_min and i >= time_stop_min:
                return {
                    "exit_price": bar["open"],
                    "exit_reason": f"time_{time_stop_min}m",
                    "bars_held": i,
                }
            # Check stop
            if bar["low"] <= stop:
                return {
                    "exit_price": stop,
                    "exit_reason": "stop",
                    "bars_held": i,
                }
            # Check target
            if bar["high"] >= target:
                return {
                    "exit_price": target,
                    "exit_reason": "target",
                    "bars_held": i,
                }

        # EOD exit
        return {
            "exit_price": post_or_bars.iloc[-1]["close"],
            "exit_reason": "eod",
            "bars_held": len(post_or_bars),
        }

    elif method == "trail_old":
        # Current live logic: trail ratchets on any new high
        trail_mult = params.get("trail_mult", 0.3)
        trail_amt = or_range * trail_mult
        # Apply floor
        floor = entry_price * 0.15 / 100
        trail_amt = max(trail_amt, floor)
        highest = entry_price

        for i, (_, bar) in enumerate(post_or_bars.iterrows()):
            # Update high and ratchet stop
            if bar["high"] > highest:
                highest = bar["high"]
                new_stop = round(highest - trail_amt, 2)
                if new_stop > stop:
                    stop = new_stop

            if bar["low"] <= stop:
                return {
                    "exit_price": stop,
                    "exit_reason": "trail",
                    "bars_held": i,
                }

        return {
            "exit_price": post_or_bars.iloc[-1]["close"],
            "exit_reason": "eod",
            "bars_held": len(post_or_bars),
        }

    elif method == "trail_threshold":
        # New logic: trail activates only after entry + trail_amt
        trail_mult = params.get("trail_mult", 0.3)
        trail_amt = or_range * trail_mult
        floor = entry_price * 0.15 / 100
        trail_amt = max(trail_amt, floor)
        activation = entry_price + trail_amt
        highest = entry_price
        trail_active = False

        for i, (_, bar) in enumerate(post_or_bars.iterrows()):
            if bar["high"] > highest:
                highest = bar["high"]

            if highest >= activation:
                trail_active = True
                new_stop = round(highest - trail_amt, 2)
                if new_stop > stop:
                    stop = new_stop

            if bar["low"] <= stop:
                reason = "trail" if trail_active else "stop"
                return {
                    "exit_price": stop,
                    "exit_reason": reason,
                    "bars_held": i,
                }

        return {
            "exit_price": post_or_bars.iloc[-1]["close"],
            "exit_reason": "eod",
            "bars_held": len(post_or_bars),
        }

    elif method == "time_exit":
        # Pure time-based: exit after N minutes, stop at OR low for protection
        exit_min = params.get("exit_min", 15)

        for i, (_, bar) in enumerate(post_or_bars.iterrows()):
            if bar["low"] <= stop:
                return {
                    "exit_price": stop,
                    "exit_reason": "stop",
                    "bars_held": i,
                }
            if i >= exit_min:
                return {
                    "exit_price": bar["open"],
                    "exit_reason": f"time_{exit_min}m",
                    "bars_held": i,
                }

        return {
            "exit_price": post_or_bars.iloc[-1]["close"],
            "exit_reason": "eod",
            "bars_held": len(post_or_bars),
        }


def rank_candidates(candidates):
    """Rank by proximity (same as live bot)."""
    for c in candidates:
        or_data = c["or_data"]
        or_range = or_data["or_range"]
        or_high = or_data["or_high"]
        or_low = or_data["or_low"]
        or_close = or_data["or_close"]
        or_mid = (or_high + or_low) / 2

        proximity = (or_close - or_low) / or_range if or_range > 0 else 0
        proximity = max(0, min(1, proximity))

        or_range_pct = or_range / or_mid * 100 if or_mid > 0 else 0
        width_score = min(or_range_pct / 1.5, 1.0)

        vol_score = min(or_data["or_avg_vol"] / 50000, 1.0)

        c["score"] = 0.60 * proximity + 0.20 * width_score + 0.20 * vol_score
        c["proximity"] = proximity

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def run_backtest():
    tickers = load_universe()
    print(f"Universe: {len(tickers)} tickers")

    # Load all data
    print("Loading 1-min data...")
    all_data = {}
    for t in tickers:
        df = load_1m(t)
        if not df.empty:
            all_data[t] = df
    print(f"Loaded: {len(all_data)} tickers with data")

    # Get all trading days
    sample = next(iter(all_data.values()))
    dates = sorted(sample["timestamp"].dt.date.unique())
    print(f"Trading days: {len(dates)} ({dates[0]} to {dates[-1]})")
    print()

    # Define methods to test
    methods = [
        ("Bracket 1.5R", "fixed_bracket", {"rr_mult": 1.5}),
        ("Bracket 2R", "fixed_bracket", {"rr_mult": 2.0}),
        ("Bracket 3R", "fixed_bracket", {"rr_mult": 3.0}),
        ("Bracket 1.5R+30m", "fixed_bracket", {"rr_mult": 1.5, "time_stop_min": 30}),
        ("Bracket 2R+30m", "fixed_bracket", {"rr_mult": 2.0, "time_stop_min": 30}),
        ("Bracket 2R+60m", "fixed_bracket", {"rr_mult": 2.0, "time_stop_min": 60}),
        ("Trail 0.3x (old)", "trail_old", {"trail_mult": 0.3}),
        ("Trail 0.3x (threshold)", "trail_threshold", {"trail_mult": 0.3}),
        ("Trail 0.5x (threshold)", "trail_threshold", {"trail_mult": 0.5}),
        ("Time 15m", "time_exit", {"exit_min": 15}),
        ("Time 30m", "time_exit", {"exit_min": 30}),
        ("Time 60m", "time_exit", {"exit_min": 60}),
    ]

    # Run for each trade-per-day count
    for max_trades in [1, 3]:
        print(f"\n{'='*90}")
        print(f"  MAX {max_trades} TRADE(S) PER DAY  |  $1,900 account  |  $5.50 RT cost  |  "
              f"${SLIPPAGE} slippage")
        print(f"{'='*90}")

        results = {name: [] for name, _, _ in methods}

        for day_idx, date in enumerate(dates[1:], 1):  # skip first day (need prev close)
            # Collect candidates for this day
            candidates = []
            for ticker, df in all_data.items():
                day_bars = df[df["timestamp"].dt.date == date].copy()
                if len(day_bars) < 10:
                    continue

                # Gap filter
                prev_close = get_prev_close(ticker, date, df)
                if prev_close is None:
                    continue
                today_open = day_bars.iloc[0]["open"]
                gap_pct = abs(today_open / prev_close - 1) * 100
                if gap_pct > MAX_GAP_PCT:
                    continue

                # Compute OR
                or_data = compute_or(day_bars)
                if or_data is None:
                    continue

                # Min range filter
                or_mid = (or_data["or_high"] + or_data["or_low"]) / 2
                or_range_pct = or_data["or_range"] / or_mid * 100
                if or_range_pct < MIN_OR_RANGE_PCT:
                    continue

                # Get post-OR bars (where trading happens)
                or_end_time = dtime(13, 30 + OR_MINUTES)
                post_or = day_bars[day_bars["timestamp"].dt.time >= or_end_time].copy()
                if post_or.empty:
                    continue

                candidates.append({
                    "ticker": ticker,
                    "or_data": or_data,
                    "post_or_bars": post_or,
                    "day_bars": day_bars,
                })

            if not candidates:
                continue

            # Rank candidates
            candidates = rank_candidates(candidates)

            # For each method, simulate top N trades
            for method_name, method_type, params in methods:
                day_trades = []
                used_tickers = set()

                for cand in candidates:
                    if len(day_trades) >= max_trades:
                        break
                    ticker = cand["ticker"]
                    if ticker in used_tickers:
                        continue

                    or_data = cand["or_data"]
                    post_or = cand["post_or_bars"]

                    # Check for breakout: find the bar where price crosses OR high
                    entry_bar_idx = None
                    for j, (_, bar) in enumerate(post_or.iterrows()):
                        if bar["high"] > or_data["or_high"]:
                            entry_bar_idx = j
                            break

                    if entry_bar_idx is None:
                        continue  # Never broke out

                    entry_price = or_data["or_high"] + SLIPPAGE
                    remaining = post_or.iloc[entry_bar_idx + 1:]  # bars after entry
                    if remaining.empty:
                        continue

                    result = simulate_trade(remaining, entry_price, or_data,
                                            method_type, params)

                    qty = max(1, int(POSITION_SIZE / entry_price))
                    pnl_per_share = result["exit_price"] - entry_price
                    pnl_usd = pnl_per_share * qty - RT_COST
                    pnl_pct = pnl_per_share / entry_price * 100

                    day_trades.append({
                        "date": date,
                        "ticker": ticker,
                        "entry": entry_price,
                        "exit": result["exit_price"],
                        "reason": result["exit_reason"],
                        "bars_held": result["bars_held"],
                        "pnl_pct": pnl_pct,
                        "pnl_usd": pnl_usd,
                        "qty": qty,
                        "score": cand["score"],
                    })
                    used_tickers.add(ticker)

                results[method_name].extend(day_trades)

        # Print results
        print(f"\n{'Method':<25} {'Trades':>6} {'WR':>6} {'Avg%':>7} "
              f"{'PF':>6} {'Total$':>8} {'MaxDD$':>8} {'Avg$':>7} {'AvgBars':>7}")
        print("-" * 90)

        for method_name, _, _ in methods:
            trades = results[method_name]
            if not trades:
                print(f"{method_name:<25} {'N/A':>6}")
                continue

            df = pd.DataFrame(trades)
            n = len(df)
            wins = (df["pnl_usd"] > 0).sum()
            wr = wins / n * 100
            avg_pct = df["pnl_pct"].mean()
            total_usd = df["pnl_usd"].sum()
            avg_usd = df["pnl_usd"].mean()
            avg_bars = df["bars_held"].mean()

            # Profit factor
            gross_profit = df.loc[df["pnl_usd"] > 0, "pnl_usd"].sum()
            gross_loss = abs(df.loc[df["pnl_usd"] < 0, "pnl_usd"].sum())
            pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

            # Max drawdown
            cumsum = df["pnl_usd"].cumsum()
            running_max = cumsum.cummax()
            dd = cumsum - running_max
            max_dd = dd.min()

            print(f"{method_name:<25} {n:>6} {wr:>5.1f}% {avg_pct:>+6.2f}% "
                  f"{pf:>6.2f} {total_usd:>+7.0f} {max_dd:>+7.0f} "
                  f"{avg_usd:>+6.1f} {avg_bars:>7.1f}")

        # Exit reason breakdown for 1-trade mode
        if max_trades == 1:
            print(f"\n--- Exit reason breakdown ---")
            for method_name, _, _ in methods:
                trades = results[method_name]
                if not trades:
                    continue
                df = pd.DataFrame(trades)
                reasons = df.groupby("reason").agg(
                    count=("pnl_usd", "count"),
                    avg_pnl=("pnl_pct", "mean"),
                    total=("pnl_usd", "sum"),
                ).reset_index()
                reason_str = " | ".join(
                    f"{r['reason']}: {r['count']}x ({r['avg_pnl']:+.2f}%, ${r['total']:+.1f})"
                    for _, r in reasons.iterrows()
                )
                print(f"  {method_name:<25} {reason_str}")


if __name__ == "__main__":
    run_backtest()
