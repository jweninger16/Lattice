"""
research/orb_entry_time_backtest.py
-------------------------------------
Tests different OR durations using 1-minute bars to find optimal entry time.

Compares: 2min (9:32), 5min (9:35), 10min (9:40), 15min (9:45)
All with volume confirmation, 1.5:1 R:R, 0.5% gap filter.
Uses 1-minute bars from yfinance (30 day max history).

Usage:
    python research/orb_entry_time_backtest.py
"""

import sys
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import time as dtime
from tqdm import tqdm

sys.path.insert(0, ".")

CACHE_DIR = Path("data/intraday_cache")
UNIVERSE_PATH = Path("data/orb_universe.csv")


def collect_1min_data(tickers):
    """Download 1-min data per ticker in 7-day chunks (yfinance 8-day limit)."""
    from datetime import datetime, timedelta
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  Downloading 1-min data for {len(tickers)} tickers (7-day chunks)...")
    success = 0

    # Build date ranges: 4 chunks of 7 days covers ~28 days
    today = datetime.now()
    chunks = []
    for i in range(4):
        end = today - timedelta(days=i * 7)
        start = end - timedelta(days=7)
        chunks.append((start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
    chunks.reverse()  # oldest first

    for ticker in tqdm(tickers, desc="  Downloading"):
        cache_path = CACHE_DIR / f"{ticker}_1m.parquet"
        frames = []
        for start, end in chunks:
            try:
                raw = yf.download(ticker, start=start, end=end, interval="1m",
                                  auto_adjust=True, progress=False, prepost=False)
                if not raw.empty:
                    frames.append(raw)
            except Exception:
                pass

        if not frames:
            continue

        combined = pd.concat(frames)
        combined = combined.reset_index()
        if isinstance(combined.columns, pd.MultiIndex):
            combined.columns = [c[0] if c[1] == '' else c[0] for c in combined.columns]
        combined.columns = [c.lower() if isinstance(c, str) else c for c in combined.columns]
        if "datetime" in combined.columns:
            combined = combined.rename(columns={"datetime": "timestamp"})
        elif "date" in combined.columns:
            combined = combined.rename(columns={"date": "timestamp"})
        combined["timestamp"] = pd.to_datetime(combined["timestamp"])
        combined = combined.drop_duplicates(subset="timestamp").sort_values("timestamp")
        combined.to_parquet(cache_path, index=False)
        success += 1

    print(f"  Downloaded: {success}/{len(tickers)}")


def load_1min(ticker):
    cache_path = CACHE_DIR / f"{ticker}_1m.parquet"
    if not cache_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(cache_path)


def prepare(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    df["date"] = df["timestamp"].dt.date
    df["time"] = df["timestamp"].dt.time
    return df


def find_exit(bars, entry, target, stop, direction):
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


def backtest_or_duration(df, or_minutes=15, require_volume=True,
                          max_gap_pct=0.5, slippage_pct=0.02,
                          commission_usd=1.0, position_size_usd=950.0):
    """
    Backtest ORB with a specific OR duration in minutes using 1-min bars.
    or_minutes=2 means OR forms from 9:30-9:32 (first 2 bars).
    """
    target_mult = 1.5
    stop_mult = 1.0
    all_days = sorted(df["date"].unique())
    trades = []
    cost_pct = (slippage_pct * 2) + (commission_usd / position_size_usd * 100)

    total_mins = 9 * 60 + 30 + or_minutes
    or_end_time = dtime(total_mins // 60, total_mins % 60)

    for day, day_df in df.groupby("date"):
        mkt = day_df[(day_df["time"] >= dtime(9, 30)) & (day_df["time"] <= dtime(15, 55))]
        if len(mkt) < or_minutes + 10:
            continue

        # OR bars = first or_minutes 1-minute bars
        or_data = mkt[mkt["time"] < or_end_time]
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

        target_pts = or_range * target_mult
        stop_pts = or_range * stop_mult
        remaining = mkt[mkt["time"] >= or_end_time]
        trade_taken = False

        for _, bar in remaining.iterrows():
            if trade_taken:
                break

            if require_volume and bar["volume"] < or_avg_volume:
                continue

            vol_ratio = bar["volume"] / or_avg_volume if or_avg_volume > 0 else 1.0

            if bar["high"] > or_high:
                entry = or_high
                target = entry + target_pts
                stop = entry - stop_pts
                future = remaining[remaining["timestamp"] >= bar["timestamp"]]
                exit_price, reason = find_exit(future, entry, target, stop, "long")
                pnl_pct = (exit_price - entry) / entry * 100 - cost_pct
                trades.append({"date": day, "direction": "long", "entry": entry,
                               "exit": exit_price, "pnl_pct": pnl_pct, "reason": reason,
                               "or_range_pct": or_range / or_mid * 100,
                               "vol_ratio": vol_ratio,
                               "entry_time": bar["time"]})
                trade_taken = True

            elif bar["low"] < or_low:
                entry = or_low
                target = entry - target_pts
                stop = entry + stop_pts
                future = remaining[remaining["timestamp"] >= bar["timestamp"]]
                exit_price, reason = find_exit(future, entry, target, stop, "short")
                pnl_pct = (entry - exit_price) / entry * 100 - cost_pct
                trades.append({"date": day, "direction": "short", "entry": entry,
                               "exit": exit_price, "pnl_pct": pnl_pct, "reason": reason,
                               "or_range_pct": or_range / or_mid * 100,
                               "vol_ratio": vol_ratio,
                               "entry_time": bar["time"]})
                trade_taken = True

    return pd.DataFrame(trades)


def stats_line(trades, label):
    if trades.empty:
        print(f"  {label:<45} -- no trades --")
        return
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

    # Median OR range
    med_or = trades["or_range_pct"].median() if "or_range_pct" in trades.columns else 0

    # Exit breakdown
    reasons = trades["reason"].value_counts().to_dict()
    r_str = " ".join(f"{r[0]}:{c}" for r, c in reasons.items())

    print(f"  {label:<45} {n:>5} tr  {wr:>5.1f}% WR  {pf:>5.2f} PF  "
          f"{total:>+8.2f}%  {avg:>+6.3f}% avg  {dd:>+6.2f}% DD  "
          f"OR:{med_or:.2f}%  {r_str}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  OR DURATION TEST (1-minute bars, volume confirmed, with costs)")
    print("=" * 80)

    # Load universe
    if not UNIVERSE_PATH.exists():
        print("  No ORB universe. Run orb_volume_stock_backtest.py --universe-only first.")
        sys.exit(1)
    universe = pd.read_csv(UNIVERSE_PATH)
    tickers = universe["ticker"].tolist()

    # Download 1-min data
    collect_1min_data(tickers)

    # Test these OR durations
    or_configs = [
        (2,  "9:32 AM (2-min OR)"),
        (5,  "9:35 AM (5-min OR)"),
        (10, "9:40 AM (10-min OR)"),
        (15, "9:45 AM (15-min OR)"),
        (20, "9:50 AM (20-min OR)"),
        (30, "10:00 AM (30-min OR)"),
    ]

    # Aggregate trades across all tickers for each duration
    agg = {mins: [] for mins, _ in or_configs}

    print(f"\n  Running backtests across {len(tickers)} tickers...")
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

        for mins, label in or_configs:
            trades = backtest_or_duration(df, or_minutes=mins)
            if not trades.empty:
                trades["ticker"] = ticker
                agg[mins].append(trades)

    # Report
    print(f"\n{'='*80}")
    print(f"  RESULTS (all tickers pooled, with costs: 0.02%/side + $1 comm on $950)")
    print(f"{'='*80}")
    print(f"  {'Entry Time':<45} {'Trades':>5}    {'WR':>5}    {'PF':>5}  "
          f"{'Total':>8}  {'AvgTrd':>7}  {'MaxDD':>7}  {'OR%':>5}  Exits")
    print(f"  {'-'*110}")

    all_results = {}
    for mins, label in or_configs:
        if agg[mins]:
            pooled = pd.concat(agg[mins]).reset_index(drop=True)
        else:
            pooled = pd.DataFrame()
        all_results[label] = pooled
        stats_line(pooled, label)

    # Detailed comparison of top 2
    print(f"\n{'='*80}")
    print(f"  DETAIL: Entry time breakdown")
    print(f"{'='*80}")

    for mins, label in or_configs:
        pooled = all_results.get(label, pd.DataFrame())
        if pooled.empty or len(pooled) < 10:
            continue

        # Median entry time
        if "entry_time" in pooled.columns:
            entry_times = pd.to_datetime(pooled["entry_time"].astype(str))
            med_entry = entry_times.dt.strftime("%H:%M").mode().iloc[0] if len(entry_times) > 0 else "?"

            before_11 = pooled[pd.to_datetime(pooled["entry_time"].astype(str)).dt.hour < 11]
            after_11 = pooled[pd.to_datetime(pooled["entry_time"].astype(str)).dt.hour >= 11]

            b_wr = (before_11["pnl_pct"] > 0).mean() * 100 if len(before_11) > 0 else 0
            a_wr = (after_11["pnl_pct"] > 0).mean() * 100 if len(after_11) > 0 else 0

            print(f"\n  {label}:")
            print(f"    Entries before 11AM: {len(before_11)} trades, {b_wr:.1f}% WR, "
                  f"{before_11['pnl_pct'].mean():+.3f}% avg")
            if len(after_11) > 0:
                print(f"    Entries after 11AM:  {len(after_11)} trades, {a_wr:.1f}% WR, "
                      f"{after_11['pnl_pct'].mean():+.3f}% avg")

        # Tickers with most trades
        top_tickers = pooled.groupby("ticker").agg(
            n=("pnl_pct", "count"),
            wr=("pnl_pct", lambda x: (x > 0).mean() * 100),
            total=("pnl_pct", "sum"),
        ).sort_values("total", ascending=False).head(5)

        print(f"    Top 5 tickers: ", end="")
        parts = []
        for t, r in top_tickers.iterrows():
            parts.append(f"{t}({r['n']:.0f}tr,{r['wr']:.0f}%WR,{r['total']:+.1f}%)")
        print(", ".join(parts))
