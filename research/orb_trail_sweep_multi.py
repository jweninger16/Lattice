"""
research/orb_trail_sweep_multi.py
-----------------------------------
Trail multiplier + floor sweep on the full 87-stock ORB universe.
Uses 1-min bars, 2-min OR, matching live ORB config exactly.

Usage:
    python research/orb_trail_sweep_multi.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import time as dtime

sys.path.insert(0, ".")

CACHE_DIR = Path("data/intraday_cache")
UNIVERSE_PATH = Path("data/orb_universe.csv")

# -- Match live ORB config --
POSITION_SIZE_USD = 1900.0
COMMISSION_RT_USD = 5.50
SLIPPAGE_PCT      = 0.02
STOP_MULT         = 1.0
OR_MINUTES        = 2
MAX_GAP_PCT       = 1.0
LAST_ENTRY        = dtime(14, 0)


def load_1min(ticker):
    path = CACHE_DIR / f"{ticker}_1m.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert(
            "America/New_York").dt.tz_localize(None)
    df["date"] = df["timestamp"].dt.date
    df["time"] = df["timestamp"].dt.time
    return df


def exit_trailing(bars, entry, or_range, trail_mult, min_trail_pct=0.0):
    trail_dist = or_range * trail_mult
    if min_trail_pct > 0:
        floor = entry * min_trail_pct / 100
        trail_dist = max(trail_dist, floor)

    stop = entry - or_range * STOP_MULT
    highest = entry
    trail_active = False

    for _, bar in bars.iterrows():
        if bar["low"] <= stop:
            reason = "trail" if trail_active else "stop"
            return stop, reason, highest
        if bar["high"] > highest:
            highest = bar["high"]
            trail_stop = highest - trail_dist
            if trail_stop > stop:
                stop = trail_stop
                trail_active = True
        if bar["time"] >= dtime(15, 55):
            return bar["close"], "eod", highest
    if len(bars) > 0:
        return bars.iloc[-1]["close"], "eod", highest
    return entry, "flat", highest


def backtest_ticker(df, trail_mult, min_trail_pct=0.0):
    """Run ORB backtest on one ticker's 1-min data. Returns list of trade dicts."""
    cost_pct = (SLIPPAGE_PCT * 2) + (COMMISSION_RT_USD / POSITION_SIZE_USD * 100)
    or_end = dtime(9, 30 + OR_MINUTES)
    all_days = sorted(df["date"].unique())
    trades = []

    for day_idx, day in enumerate(all_days):
        day_df = df[df["date"] == day]
        mkt = day_df[(day_df["time"] >= dtime(9, 30)) & (day_df["time"] <= dtime(15, 55))]
        if len(mkt) < OR_MINUTES + 10:
            continue

        or_data = mkt[mkt["time"] < or_end]
        if len(or_data) < OR_MINUTES:
            continue

        or_high = or_data["high"].max()
        or_low = or_data["low"].min()
        or_range = or_high - or_low
        or_mid = (or_high + or_low) / 2
        if or_range <= 0 or or_mid <= 0:
            continue

        or_avg_volume = or_data["volume"].mean()

        # Gap filter
        if day_idx > 0:
            prev_day = all_days[day_idx - 1]
            prev_data = df[df["date"] == prev_day]
            if len(prev_data) > 0:
                prev_close = prev_data.iloc[-1]["close"]
                gap = abs(mkt.iloc[0]["open"] / prev_close - 1) * 100
                if gap > MAX_GAP_PCT:
                    continue

        remaining = mkt[(mkt["time"] >= or_end) & (mkt["time"] <= LAST_ENTRY)]

        for _, bar in remaining.iterrows():
            if bar["volume"] < or_avg_volume:
                continue
            if bar["high"] > or_high:
                entry = or_high
                vol_ratio = bar["volume"] / or_avg_volume
                future = mkt[mkt["timestamp"] >= bar["timestamp"]]
                exit_price, reason, peak = exit_trailing(
                    future, entry, or_range, trail_mult, min_trail_pct)

                pnl_pct = (exit_price - entry) / entry * 100 - cost_pct
                peak_exc = (peak - entry) / entry * 100

                trades.append({
                    "date": day, "entry": entry, "exit": exit_price,
                    "pnl_pct": pnl_pct, "reason": reason,
                    "vol_ratio": vol_ratio, "price": entry,
                    "or_range_pct": or_range / or_mid * 100,
                    "peak_excursion": peak_exc,
                })
                break

    return trades


def compute_stats(trades_df):
    if trades_df.empty:
        return {"n": 0, "wr": 0, "pf": 0, "total": 0, "avg": 0, "med": 0,
                "dd": 0, "avg_win": 0, "avg_loss": 0, "stop_rate": 0,
                "trail_rate": 0, "eod_rate": 0, "avg_peak": 0,
                "captured": 0}
    n = len(trades_df)
    wins = trades_df[trades_df["pnl_pct"] > 0]
    losses = trades_df[trades_df["pnl_pct"] <= 0]
    gw = wins["pnl_pct"].sum() if len(wins) else 0
    gl = abs(losses["pnl_pct"].sum()) if len(losses) else 0.001
    cum = trades_df["pnl_pct"].cumsum()
    avg_peak = trades_df["peak_excursion"].mean()
    avg_pnl = trades_df["pnl_pct"].mean()
    # "Captured" = what % of peak excursion is kept as profit
    captured = (avg_pnl / avg_peak * 100) if avg_peak > 0 else 0

    return {
        "n": n,
        "wr": len(wins) / n * 100,
        "pf": gw / gl if gl > 0 else 0,
        "total": trades_df["pnl_pct"].sum(),
        "avg": avg_pnl,
        "med": trades_df["pnl_pct"].median(),
        "dd": (cum - cum.cummax()).min(),
        "avg_win": wins["pnl_pct"].mean() if len(wins) else 0,
        "avg_loss": losses["pnl_pct"].mean() if len(losses) else 0,
        "stop_rate": (trades_df["reason"] == "stop").sum() / n * 100,
        "trail_rate": (trades_df["reason"] == "trail").sum() / n * 100,
        "eod_rate": (trades_df["reason"] == "eod").sum() / n * 100,
        "avg_peak": avg_peak,
        "captured": captured,
    }


def filter_best_per_day(all_trades_df):
    """Keep only the best trade per day by vol_ratio."""
    if all_trades_df.empty:
        return all_trades_df
    return (all_trades_df
            .sort_values("vol_ratio", ascending=False)
            .groupby("date").head(1)
            .sort_values("date")
            .reset_index(drop=True))


if __name__ == "__main__":
    cost_pct = (SLIPPAGE_PCT * 2) + (COMMISSION_RT_USD / POSITION_SIZE_USD * 100)

    print()
    print("=" * 90)
    print("  TRAIL SWEEP: 87-stock ORB universe, 1-min bars")
    print(f"  {OR_MINUTES}-min OR | {STOP_MULT}x stop | vol confirmed | long-only | gap<{MAX_GAP_PCT}%")
    print(f"  ${POSITION_SIZE_USD:,.0f} position | ${COMMISSION_RT_USD:.2f} RT cost | "
          f"{SLIPPAGE_PCT}% slip/side | total cost: {cost_pct:.3f}%/trade")
    print("=" * 90)

    # Load data
    universe = pd.read_csv(UNIVERSE_PATH)
    tickers = universe["ticker"].tolist()

    print(f"\n  Loading 1-min data for {len(tickers)} tickers...")
    ticker_data = {}
    for t in tickers:
        df = load_1min(t)
        if not df.empty and df["date"].nunique() >= 5:
            ticker_data[t] = df
    print(f"  Loaded: {len(ticker_data)} tickers with >= 5 days")

    # ================================================================
    # Part 1: Trail multiplier sweep (all trades)
    # ================================================================
    trail_mults = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.25, 1.5]

    print(f"\n  Part 1: Trail multiplier sweep (all trades)...")
    all_results = {}
    for m in trail_mults:
        all_trades = []
        for ticker, df in ticker_data.items():
            trades = backtest_ticker(df, trail_mult=m)
            for t in trades:
                t["ticker"] = ticker
            all_trades.extend(trades)
        all_results[m] = pd.DataFrame(all_trades)
        sys.stdout.write(f"    {m:.2f}x: {len(all_trades)} trades\n")

    print(f"\n{'='*90}")
    print(f"  ALL TRADES (every breakout, not filtered to best-per-day)")
    print(f"{'='*90}")
    print(f"  {'Trail':>6} {'N':>5} {'Win%':>6} {'PF':>6} {'Total':>9} "
          f"{'Avg':>8} {'Med':>8} {'Stop%':>6} {'Trail%':>7} {'EOD%':>5} "
          f"{'Peak':>6} {'Capt':>6} {'DD':>7}")
    print(f"  {'-'*88}")

    for m in trail_mults:
        s = compute_stats(all_results[m])
        if s["n"] == 0:
            continue
        marker = " <-- current" if m == 0.3 else ""
        print(f"  {m:>5.2f}x {s['n']:>5} {s['wr']:>5.1f}% {s['pf']:>5.2f} "
              f"{s['total']:>+8.1f}% {s['avg']:>+7.3f}% {s['med']:>+7.3f}% "
              f"{s['stop_rate']:>5.1f}% {s['trail_rate']:>6.1f}% {s['eod_rate']:>4.1f}% "
              f"{s['avg_peak']:>5.2f}% {s['captured']:>5.0f}% {s['dd']:>+6.2f}%{marker}")

    # ================================================================
    # Part 2: Best-per-day (matching live: 1 trade/day, best vol_ratio)
    # ================================================================
    print(f"\n{'='*90}")
    print(f"  BEST 1 TRADE/DAY (top vol_ratio -- matches live bot)")
    print(f"{'='*90}")
    print(f"  {'Trail':>6} {'N':>5} {'Win%':>6} {'PF':>6} {'Total':>9} "
          f"{'Avg':>8} {'Med':>8} {'Stop%':>6} {'Trail%':>7} {'EOD%':>5} "
          f"{'Peak':>6} {'Capt':>6} {'DD':>7}")
    print(f"  {'-'*88}")

    best_results = {}
    for m in trail_mults:
        best = filter_best_per_day(all_results[m])
        best_results[m] = best
        s = compute_stats(best)
        if s["n"] == 0:
            continue
        marker = " <-- current" if m == 0.3 else ""
        print(f"  {m:>5.2f}x {s['n']:>5} {s['wr']:>5.1f}% {s['pf']:>5.2f} "
              f"{s['total']:>+8.1f}% {s['avg']:>+7.3f}% {s['med']:>+7.3f}% "
              f"{s['stop_rate']:>5.1f}% {s['trail_rate']:>6.1f}% {s['eod_rate']:>4.1f}% "
              f"{s['avg_peak']:>5.2f}% {s['captured']:>5.0f}% {s['dd']:>+6.2f}%{marker}")

    # ================================================================
    # Part 3: Floor sweep at promising trail mults
    # ================================================================
    # Find best mult by PF from best-per-day
    best_mult_pf = max(trail_mults,
                       key=lambda m: compute_stats(best_results[m])["pf"]
                       if not best_results[m].empty else 0)
    best_mult_total = max(trail_mults,
                          key=lambda m: compute_stats(best_results[m])["total"]
                          if not best_results[m].empty else -999)

    print(f"\n  Best by PF: {best_mult_pf}x | Best by total P&L: {best_mult_total}x")

    test_mults = sorted(set([0.3, 0.5, 0.7, 1.0, best_mult_pf, best_mult_total]))
    floors = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]

    print(f"\n  Part 3: Floor sweep at trail = {test_mults}...")
    floor_results = {}
    for m in test_mults:
        for f in floors:
            key = (m, f)
            all_trades = []
            for ticker, df in ticker_data.items():
                trades = backtest_ticker(df, trail_mult=m, min_trail_pct=f)
                for t in trades:
                    t["ticker"] = ticker
                all_trades.extend(trades)
            floor_results[key] = pd.DataFrame(all_trades)
        sys.stdout.write(f"    {m:.2f}x done\n")

    print(f"\n{'='*90}")
    print(f"  FLOOR SWEEP (best 1/day)")
    print(f"{'='*90}")

    for m in test_mults:
        print(f"\n  Trail {m}x:")
        print(f"  {'Floor':>10} {'N':>5} {'Win%':>6} {'PF':>6} {'Total':>9} "
              f"{'Avg':>8} {'Med':>8} {'Stop%':>6} {'Peak':>6} {'Capt':>6} {'DD':>7}")
        print(f"  {'-'*82}")
        for f in floors:
            key = (m, f)
            best = filter_best_per_day(floor_results[key])
            s = compute_stats(best)
            if s["n"] == 0:
                continue
            flabel = "none" if f == 0 else f"{f:.2f}%"
            marker = " <-- current" if m == 0.3 and f == 0 else ""
            print(f"  {flabel:>10} {s['n']:>5} {s['wr']:>5.1f}% {s['pf']:>5.2f} "
                  f"{s['total']:>+8.1f}% {s['avg']:>+7.3f}% {s['med']:>+7.3f}% "
                  f"{s['stop_rate']:>5.1f}% {s['avg_peak']:>5.2f}% {s['captured']:>5.0f}% "
                  f"{s['dd']:>+6.2f}%{marker}")

    # ================================================================
    # Part 4: Low-price stocks (< $50) -- where the pain is
    # ================================================================
    print(f"\n{'='*90}")
    print(f"  LOW-PRICE STOCKS (entry < $50) -- best 1/day")
    print(f"{'='*90}")

    for m in test_mults:
        print(f"\n  Trail {m}x:")
        print(f"  {'Floor':>10} {'N':>5} {'Win%':>6} {'PF':>6} {'Avg':>8} {'Med':>8} {'DD':>7}")
        print(f"  {'-'*60}")
        for f in floors:
            key = (m, f)
            best = filter_best_per_day(floor_results[key])
            if best.empty:
                continue
            low = best[best["price"] < 50]
            if len(low) < 3:
                continue
            s = compute_stats(low)
            flabel = "none" if f == 0 else f"{f:.2f}%"
            print(f"  {flabel:>10} {s['n']:>5} {s['wr']:>5.1f}% {s['pf']:>5.2f} "
                  f"{s['avg']:>+7.3f}% {s['med']:>+7.3f}% {s['dd']:>+6.2f}%")

    # ================================================================
    # Part 5: Top 10 configs ranked by PF (best-per-day)
    # ================================================================
    print(f"\n{'='*90}")
    print(f"  TOP 10 CONFIGS (best 1/day, ranked by PF)")
    print(f"{'='*90}")

    ranked = []
    for (m, f), df in floor_results.items():
        best = filter_best_per_day(df)
        s = compute_stats(best)
        if s["n"] >= 10:  # minimum trades
            ranked.append((m, f, s))

    ranked.sort(key=lambda x: x[2]["pf"], reverse=True)
    print(f"  {'Trail':>6} {'Floor':>8} {'N':>5} {'Win%':>6} {'PF':>6} "
          f"{'Total':>9} {'Avg':>8} {'Stop%':>6} {'DD':>7}")
    print(f"  {'-'*68}")
    for m, f, s in ranked[:10]:
        flabel = "none" if f == 0 else f"{f:.2f}%"
        print(f"  {m:>5.2f}x {flabel:>8} {s['n']:>5} {s['wr']:>5.1f}% {s['pf']:>5.2f} "
              f"{s['total']:>+8.1f}% {s['avg']:>+7.3f}% "
              f"{s['stop_rate']:>5.1f}% {s['dd']:>+6.2f}%")

    # Also rank by total P&L
    ranked_total = sorted(ranked, key=lambda x: x[2]["total"], reverse=True)
    print(f"\n  TOP 10 CONFIGS (best 1/day, ranked by Total P&L)")
    print(f"  {'-'*68}")
    for m, f, s in ranked_total[:10]:
        flabel = "none" if f == 0 else f"{f:.2f}%"
        print(f"  {m:>5.2f}x {flabel:>8} {s['n']:>5} {s['wr']:>5.1f}% {s['pf']:>5.2f} "
              f"{s['total']:>+8.1f}% {s['avg']:>+7.3f}% "
              f"{s['stop_rate']:>5.1f}% {s['dd']:>+6.2f}%")
