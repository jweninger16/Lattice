"""
research/orb_ranking_test.py
------------------------------
Tests alternative candidate ranking methods for ORB stock selection.
Currently the bot picks the highest vol_ratio trade each day.
This script tests whether other criteria produce better results.

Usage:
    python research/orb_ranking_test.py
"""

import sys
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import time as dtime

sys.path.insert(0, ".")

CACHE_DIR = Path("data/intraday_cache")

POSITION_SIZE_USD = 1900.0
COMMISSION_RT_USD = 5.50
SLIPPAGE_PCT = 0.02
STOP_MULT = 1.0
TRAIL_MULT = 0.3
OR_MINUTES = 2
MAX_GAP_PCT = 1.0
LAST_ENTRY = dtime(14, 0)


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


def exit_trailing(bars, entry, or_range):
    stop = entry - or_range * STOP_MULT
    trail_dist = or_range * TRAIL_MULT
    highest = entry
    trail_active = False
    for _, bar in bars.iterrows():
        if bar["low"] <= stop:
            return stop, "trail" if trail_active else "stop", highest
        if bar["high"] > highest:
            highest = bar["high"]
            ts = highest - trail_dist
            if ts > stop:
                stop = ts
                trail_active = True
        if bar["time"] >= dtime(15, 55):
            return bar["close"], "eod", highest
    if len(bars) > 0:
        return bars.iloc[-1]["close"], "eod", highest
    return entry, "flat", highest


def collect_all_trades(ticker_data):
    """Run ORB backtest and collect all trades with ranking features."""
    cost_pct = (SLIPPAGE_PCT * 2) + (COMMISSION_RT_USD / POSITION_SIZE_USD * 100)
    or_end = dtime(9, 32)
    trades = []

    for ticker, df in ticker_data.items():
        all_days = sorted(df["date"].unique())
        for day_idx, day in enumerate(all_days):
            day_df = df[df["date"] == day]
            mkt = day_df[(day_df["time"] >= dtime(9, 30)) &
                         (day_df["time"] <= dtime(15, 55))]
            if len(mkt) < 12:
                continue

            or_data = mkt[mkt["time"] < or_end]
            if len(or_data) < 2:
                continue

            or_high = or_data["high"].max()
            or_low = or_data["low"].min()
            or_range = or_high - or_low
            or_mid = (or_high + or_low) / 2
            if or_range <= 0 or or_mid <= 0:
                continue

            or_avg_vol = or_data["volume"].mean()

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
                if bar["volume"] < or_avg_vol:
                    continue
                if bar["high"] > or_high:
                    entry = or_high
                    vol_ratio = bar["volume"] / or_avg_vol
                    future = mkt[mkt["timestamp"] >= bar["timestamp"]]
                    exit_price, reason, peak = exit_trailing(
                        future, entry, or_range)
                    pnl_pct = (exit_price - entry) / entry * 100 - cost_pct

                    or_range_pct = or_range / or_mid * 100
                    or_tightness = 1.0 / or_range_pct if or_range_pct > 0 else 0
                    dollar_volume = or_avg_vol * or_mid

                    # Price coiling: how close was bar open to OR high?
                    bar_open_proximity = ((bar["open"] - or_low) / or_range
                                         if or_range > 0 else 0)

                    # Breakout bar strength: bullish close relative to bar range
                    bar_range = bar["high"] - bar["low"]
                    bar_strength = ((bar["close"] - bar["open"]) / bar_range
                                   if bar_range > 0 else 0)

                    trades.append({
                        "date": day, "ticker": ticker, "entry": entry,
                        "exit": exit_price, "pnl_pct": pnl_pct,
                        "reason": reason, "price": entry,
                        "vol_ratio": vol_ratio,
                        "or_range_pct": or_range_pct,
                        "or_tightness": or_tightness,
                        "dollar_volume": dollar_volume,
                        "bar_open_proximity": bar_open_proximity,
                        "bar_strength": bar_strength,
                        "peak_excursion": (peak - entry) / entry * 100,
                    })
                    break

    return pd.DataFrame(trades)


def test_ranking(df, rank_col, ascending=False, label=""):
    """Pick best 1/day by rank_col and compute stats."""
    ranked = df.sort_values(rank_col, ascending=ascending)
    best = (ranked.groupby("date").head(1)
            .sort_values("date").reset_index(drop=True))
    n = len(best)
    if n == 0:
        return None
    wins = best[best["pnl_pct"] > 0]
    losses = best[best["pnl_pct"] <= 0]
    wr = len(wins) / n * 100
    gw = wins["pnl_pct"].sum() if len(wins) else 0
    gl = abs(losses["pnl_pct"].sum()) if len(losses) else 0.001
    pf = gw / gl
    total = best["pnl_pct"].sum()
    avg = best["pnl_pct"].mean()
    med = best["pnl_pct"].median()
    cum = best["pnl_pct"].cumsum()
    dd = (cum - cum.cummax()).min()
    avg_peak = best["peak_excursion"].mean()
    return {"label": label, "n": n, "wr": wr, "pf": pf, "total": total,
            "avg": avg, "med": med, "dd": dd, "avg_peak": avg_peak}


def print_result(r):
    if r is None:
        return
    print(f"  {r['label']:<38} {r['n']:>3}  {r['wr']:>5.1f}%  {r['pf']:>5.2f}  "
          f"{r['total']:>+6.1f}%  {r['avg']:>+7.3f}%  {r['med']:>+7.3f}%  "
          f"{r['dd']:>+6.2f}%  {r['avg_peak']:>5.2f}%")


if __name__ == "__main__":
    print()
    print("=" * 105)
    print("  RANKING METHOD COMPARISON")
    print(f"  2-min OR | 0.3x trail | 1.0x stop | vol confirmed | long-only | gap<{MAX_GAP_PCT}%")
    print(f"  Best 1 trade/day | cost: {(SLIPPAGE_PCT*2 + COMMISSION_RT_USD/POSITION_SIZE_USD*100):.3f}%/trade")
    print("=" * 105)

    # Load data
    universe = pd.read_csv("data/orb_universe.csv")
    tickers = universe["ticker"].tolist()
    print(f"\n  Loading {len(tickers)} tickers...")
    ticker_data = {}
    for t in tickers:
        d = load_1min(t)
        if not d.empty and d["date"].nunique() >= 5:
            ticker_data[t] = d
    print(f"  Loaded: {len(ticker_data)} tickers")

    print("  Running ORB backtest on all tickers...")
    df = collect_all_trades(ticker_data)
    print(f"  Total breakout trades: {len(df)} across {df['date'].nunique()} days")

    # Random baseline
    random.seed(42)
    random_avgs = []
    for _ in range(500):
        picks = df.groupby("date").apply(
            lambda x: x.sample(1)).reset_index(drop=True)
        random_avgs.append(picks["pnl_pct"].mean())
    rand_avg = np.mean(random_avgs)
    rand_std = np.std(random_avgs)

    # Compute composite scores
    for col in ["vol_ratio", "or_tightness", "dollar_volume",
                "bar_open_proximity", "bar_strength"]:
        df[f"{col}_rank"] = df.groupby("date")[col].rank(pct=True)

    df["score_vol_tight"] = (df["vol_ratio_rank"] * 0.5 +
                              df["or_tightness_rank"] * 0.5)
    df["score_vol_prox"] = (df["vol_ratio_rank"] * 0.4 +
                             df["bar_open_proximity_rank"] * 0.4 +
                             df["bar_strength_rank"] * 0.2)
    df["score_tight_prox"] = (df["or_tightness_rank"] * 0.4 +
                               df["bar_open_proximity_rank"] * 0.4 +
                               df["bar_strength_rank"] * 0.2)
    df["score_all_equal"] = (df["vol_ratio_rank"] * 0.25 +
                              df["or_tightness_rank"] * 0.25 +
                              df["bar_open_proximity_rank"] * 0.25 +
                              df["bar_strength_rank"] * 0.25)
    df["score_dvol_tight"] = (df["dollar_volume_rank"] * 0.5 +
                               df["or_tightness_rank"] * 0.5)
    df["score_prox_heavy"] = (df["bar_open_proximity_rank"] * 0.6 +
                               df["bar_strength_rank"] * 0.2 +
                               df["vol_ratio_rank"] * 0.2)

    print(f"\n  Random baseline (500 runs): {rand_avg:+.3f}% avg "
          f"(+/- {rand_std:.3f}%)")

    header = (f"  {'Method':<38} {'N':>3}  {'Win%':>5}  {'PF':>5}  "
              f"{'Total':>6}  {'Avg':>8}  {'Med':>8}  {'DD':>7}  {'Peak':>6}")
    sep = "  " + "-" * 100

    # -- Single feature rankings --
    print(f"\n  SINGLE FEATURE RANKINGS")
    print(header)
    print(sep)
    for col, asc, label in [
        ("vol_ratio", False, "vol_ratio (CURRENT)"),
        ("or_tightness", False, "OR tightness (narrow range)"),
        ("or_range_pct", False, "OR width (wide range)"),
        ("dollar_volume", False, "Dollar volume"),
        ("bar_open_proximity", False, "Proximity (coiling near high)"),
        ("bar_strength", False, "Breakout bar strength"),
        ("vol_ratio", True, "LOWEST vol_ratio (contrarian)"),
    ]:
        print_result(test_ranking(df, col, asc, label))

    # -- Composite rankings --
    print(f"\n  COMPOSITE RANKINGS")
    print(header)
    print(sep)
    for col, label in [
        ("score_vol_tight", "50% vol + 50% tightness"),
        ("score_vol_prox", "40% vol + 40% prox + 20% strength"),
        ("score_tight_prox", "40% tight + 40% prox + 20% strength"),
        ("score_all_equal", "25% each (vol/tight/prox/str)"),
        ("score_dvol_tight", "50% dollar-vol + 50% tightness"),
        ("score_prox_heavy", "60% prox + 20% str + 20% vol"),
    ]:
        print_result(test_ranking(df, col, False, label))

    # -- Multi-trade per day --
    print(f"\n  MULTI-TRADE PER DAY (does taking 2-3 help?)")
    print(f"  {'Method':<30} {'Top':>4}  {'N':>3}  {'Win%':>5}  {'PF':>5}  "
          f"{'Total':>7}  {'Avg':>8}")
    print("  " + "-" * 75)

    for col, label in [("vol_ratio", "vol_ratio"),
                       ("score_tight_prox", "tight+prox+str"),
                       ("score_prox_heavy", "prox-heavy")]:
        for top_n in [1, 2, 3, 5]:
            ranked = df.sort_values(col, ascending=False)
            best = (ranked.groupby("date").head(top_n)
                    .sort_values("date").reset_index(drop=True))
            n = len(best)
            wins = best[best["pnl_pct"] > 0]
            losses = best[best["pnl_pct"] <= 0]
            wr = len(wins) / n * 100
            gw = wins["pnl_pct"].sum() if len(wins) else 0
            gl = abs(losses["pnl_pct"].sum()) if len(losses) else 0.001
            pf = gw / gl
            total = best["pnl_pct"].sum()
            avg = best["pnl_pct"].mean()
            print(f"  {label:<30} top-{top_n}  {n:>3}  {wr:>5.1f}%  {pf:>5.2f}  "
                  f"{total:>+6.1f}%  {avg:>+7.3f}%")

    # -- Best by P&L (oracle / ceiling) --
    print(f"\n  ORACLE (what's the ceiling?)")
    oracle = (df.sort_values("pnl_pct", ascending=False)
              .groupby("date").head(1).sort_values("date").reset_index(drop=True))
    r = test_ranking(df, "pnl_pct", False, "Best trade each day (hindsight)")
    print(header)
    print(sep)
    print_result(r)

    worst = test_ranking(df, "pnl_pct", True, "Worst trade each day (anti-oracle)")
    print_result(worst)
