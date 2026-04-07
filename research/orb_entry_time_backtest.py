"""
research/orb_entry_time_backtest.py
-------------------------------------
Tests different OR durations using 1-minute bars to find optimal entry time.

Compares: 2min (9:32), 5min (9:35), 10min (9:40), 15min (9:45), 20min, 30min
Uses the CURRENT live exit method: 0.3x trailing stop + 1.0x initial stop.
Multi-ticker: ranks candidates by vol_ratio, takes top N per day.
Long-only, volume confirmed, 0.5% gap filter, with costs ($5.50 RT on $1,900).

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

# ── Match live config ─────────────────────────────────────────────────
POSITION_SIZE_USD = 1900.0
COMMISSION_RT_USD = 5.50       # Round-trip commission + fees
SLIPPAGE_PCT = 0.02            # Per side
STOP_MULT = 1.0                # Initial stop = 1.0x OR range below entry
TRAIL_MULT = 0.3               # Trailing stop distance = 0.3x OR range
MAX_GAP_PCT = 0.5              # Skip if gap > 0.5%
MAX_TRADES_PER_DAY_OPTIONS = [1, 2, 3]  # Test 1, 2, and 3 trades/day


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


def exit_trailing(bars, entry, or_range, trail_mult=TRAIL_MULT, stop_mult=STOP_MULT):
    """
    Trailing stop exit matching the live ORB system.
    Initial stop at stop_mult * OR range below entry.
    Trail distance = trail_mult * OR range, ratchets up with price.
    """
    stop = entry - or_range * stop_mult
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


def find_candidates_for_day(day_df, or_end_time, or_minutes, or_high, or_low,
                             or_range, or_avg_volume):
    """
    Scan post-OR bars for volume-confirmed long breakouts.
    Returns list of candidate dicts with vol_ratio and entry info.
    """
    remaining = day_df[day_df["time"] >= or_end_time]
    candidates = []

    for _, bar in remaining.iterrows():
        if bar["volume"] < or_avg_volume:
            continue
        vol_ratio = bar["volume"] / or_avg_volume if or_avg_volume > 0 else 1.0

        if bar["high"] > or_high:
            future = remaining[remaining["timestamp"] >= bar["timestamp"]]
            exit_price, reason = exit_trailing(future, or_high, or_range)
            candidates.append({
                "entry": or_high,
                "exit": exit_price,
                "reason": reason,
                "vol_ratio": vol_ratio,
                "entry_time": bar["time"],
                "or_range_pct": or_range / ((or_high + or_low) / 2) * 100,
            })
            break  # One candidate per ticker per day

    return candidates


def backtest_or_duration_multi(all_ticker_data, or_minutes=2, max_trades=1):
    """
    Multi-ticker ORB backtest with trailing stop exit.
    For each day: collect breakout candidates across all tickers,
    rank by vol_ratio, take top max_trades.
    """
    cost_pct = (SLIPPAGE_PCT * 2) + (COMMISSION_RT_USD / POSITION_SIZE_USD * 100)

    total_mins = 9 * 60 + 30 + or_minutes
    or_end_time = dtime(total_mins // 60, total_mins % 60)

    # Gather all trading dates across all tickers
    all_dates = set()
    for ticker, df in all_ticker_data.items():
        all_dates.update(df["date"].unique())
    all_dates = sorted(all_dates)

    # Build prev_close lookup per ticker
    prev_closes = {}
    for ticker, df in all_ticker_data.items():
        daily = df.groupby("date").agg(
            first_open=("open", "first"),
            last_close=("close", "last"),
        )
        prev_closes[ticker] = daily

    trades = []

    for day in all_dates:
        day_candidates = []

        for ticker, df in all_ticker_data.items():
            day_df = df[df["date"] == day]
            mkt = day_df[(day_df["time"] >= dtime(9, 30)) & (day_df["time"] <= dtime(15, 55))]
            if len(mkt) < or_minutes + 10:
                continue

            # Compute OR
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
            if ticker in prev_closes and MAX_GAP_PCT is not None:
                pc = prev_closes[ticker]
                day_dates = sorted(pc.index)
                day_idx = day_dates.index(day) if day in day_dates else -1
                if day_idx > 0:
                    prev_close = pc.loc[day_dates[day_idx - 1], "last_close"]
                    gap = abs(mkt.iloc[0]["open"] / prev_close - 1) * 100
                    if gap > MAX_GAP_PCT:
                        continue

            cands = find_candidates_for_day(
                mkt, or_end_time, or_minutes, or_high, or_low, or_range, or_avg_volume
            )
            for c in cands:
                c["ticker"] = ticker
                c["date"] = day
                day_candidates.append(c)

        # Rank by vol_ratio, take top max_trades
        day_candidates.sort(key=lambda x: x["vol_ratio"], reverse=True)
        for c in day_candidates[:max_trades]:
            pnl_pct = (c["exit"] - c["entry"]) / c["entry"] * 100 - cost_pct
            pnl_usd = pnl_pct / 100 * POSITION_SIZE_USD
            c["pnl_pct"] = pnl_pct
            c["pnl_usd"] = pnl_usd
            c["direction"] = "long"
            trades.append(c)

    return pd.DataFrame(trades)


def stats_line(trades, label, show_usd=True):
    if trades.empty:
        print(f"  {label:<55} -- no trades --")
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

    # Daily P&L
    n_days = trades["date"].nunique() if "date" in trades.columns else 1
    daily_usd = trades["pnl_usd"].sum() / n_days if "pnl_usd" in trades.columns else 0

    # Exit breakdown
    reasons = trades["reason"].value_counts().to_dict()
    r_str = " ".join(f"{r[0]}:{c}" for r, c in reasons.items())

    usd_str = f"  ${daily_usd:>+6.2f}/day" if show_usd else ""
    print(f"  {label:<55} {n:>5} tr  {wr:>5.1f}% WR  {pf:>5.2f} PF  "
          f"{total:>+8.2f}%  {avg:>+6.3f}% avg  {dd:>+6.2f}% DD  "
          f"OR:{med_or:.2f}%{usd_str}  {r_str}")


if __name__ == "__main__":
    print("\n" + "=" * 90)
    print("  OR DURATION TEST — trailing stop exit (0.3x trail, 1.0x stop)")
    print("  Multi-ticker ranked by vol_ratio | long-only | volume confirmed | with costs")
    print(f"  Position: ${POSITION_SIZE_USD:,.0f} | Commission: ${COMMISSION_RT_USD:.2f} RT | "
          f"Slippage: {SLIPPAGE_PCT:.2f}%/side")
    print("=" * 90)

    # Load universe
    if not UNIVERSE_PATH.exists():
        print("  No ORB universe. Run orb_volume_stock_backtest.py --universe-only first.")
        sys.exit(1)
    universe = pd.read_csv(UNIVERSE_PATH)
    tickers = universe["ticker"].tolist()

    # Download 1-min data
    collect_1min_data(tickers)

    # Load and prepare all ticker data
    print(f"\n  Loading data for {len(tickers)} tickers...")
    all_data = {}
    for ticker in tqdm(tickers, desc="  Loading"):
        df = load_1min(ticker)
        if df.empty:
            continue
        try:
            df = prepare(df)
        except Exception:
            continue
        if df["date"].nunique() < 5:
            continue
        all_data[ticker] = df

    print(f"  Loaded {len(all_data)} tickers with sufficient data")

    # Test these OR durations
    or_configs = [
        (2,  "2-min OR (9:32)"),
        (5,  "5-min OR (9:35)"),
        (10, "10-min OR (9:40)"),
        (15, "15-min OR (9:45)"),
        (20, "20-min OR (9:50)"),
        (30, "30-min OR (10:00)"),
    ]

    # ── Run all combinations ──────────────────────────────────────────
    for max_trades in MAX_TRADES_PER_DAY_OPTIONS:
        print(f"\n{'='*90}")
        print(f"  MAX {max_trades} TRADE{'S' if max_trades > 1 else ''}/DAY "
              f"(top {max_trades} by vol_ratio)")
        print(f"{'='*90}")
        print(f"  {'Config':<55} {'Trades':>5}    {'WR':>5}    {'PF':>5}  "
              f"{'Total':>8}  {'AvgTrd':>7}  {'MaxDD':>7}  {'OR%':>5}  {'$/day':>8}  Exits")
        print(f"  {'-'*120}")

        best_pf = 0
        best_label = ""

        for mins, label in or_configs:
            full_label = f"{label} | {max_trades}/day"
            result = backtest_or_duration_multi(all_data, or_minutes=mins, max_trades=max_trades)
            stats_line(result, full_label)

            if not result.empty:
                wins = result[result["pnl_pct"] > 0]
                losses = result[result["pnl_pct"] <= 0]
                gw = wins["pnl_pct"].sum() if len(wins) else 0
                gl = abs(losses["pnl_pct"].sum()) if len(losses) else 0.001
                pf = gw / gl
                if pf > best_pf:
                    best_pf = pf
                    best_label = full_label

        print(f"\n  >>> Best PF: {best_label} ({best_pf:.2f})")

    # ── Detailed breakdown for top configs ────────────────────────────
    print(f"\n{'='*90}")
    print(f"  DETAIL: Per-ticker breakdown for 2-min OR (1/day and 2/day)")
    print(f"{'='*90}")

    for max_trades in [1, 2]:
        result = backtest_or_duration_multi(all_data, or_minutes=2, max_trades=max_trades)
        if result.empty:
            continue

        print(f"\n  --- 2-min OR, {max_trades}/day ---")

        # Time-of-day analysis
        if "entry_time" in result.columns:
            entry_times = pd.to_datetime(result["entry_time"].astype(str))
            before_11 = result[entry_times.dt.hour < 11]
            after_11 = result[entry_times.dt.hour >= 11]

            b_wr = (before_11["pnl_pct"] > 0).mean() * 100 if len(before_11) > 0 else 0
            a_wr = (after_11["pnl_pct"] > 0).mean() * 100 if len(after_11) > 0 else 0

            print(f"    Entries before 11AM: {len(before_11)} trades, {b_wr:.1f}% WR, "
                  f"{before_11['pnl_pct'].mean():+.3f}% avg, "
                  f"${before_11['pnl_usd'].sum() / max(1, before_11['date'].nunique()):+.2f}/day")
            if len(after_11) > 0:
                print(f"    Entries after 11AM:  {len(after_11)} trades, {a_wr:.1f}% WR, "
                      f"{after_11['pnl_pct'].mean():+.3f}% avg, "
                      f"${after_11['pnl_usd'].sum() / max(1, after_11['date'].nunique()):+.2f}/day")

        # Top tickers
        top_tickers = result.groupby("ticker").agg(
            n=("pnl_pct", "count"),
            wr=("pnl_pct", lambda x: (x > 0).mean() * 100),
            total_pct=("pnl_pct", "sum"),
            total_usd=("pnl_usd", "sum"),
        ).sort_values("total_usd", ascending=False).head(10)

        print(f"    Top 10 tickers:")
        for t, r in top_tickers.iterrows():
            print(f"      {t:<6} {r['n']:>3} trades  {r['wr']:>5.1f}% WR  "
                  f"{r['total_pct']:>+7.2f}%  ${r['total_usd']:>+7.2f}")

    # ── Summary recommendation ────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  SUMMARY: Compare 1/day vs 2/day across all OR durations")
    print(f"{'='*90}")
    print(f"  {'Config':<40} {'Trades':>5}  {'WR':>5}  {'PF':>5}  {'$/day':>8}  {'MaxDD':>7}")
    print(f"  {'-'*80}")

    for mins, label in or_configs:
        for mt in [1, 2]:
            result = backtest_or_duration_multi(all_data, or_minutes=mins, max_trades=mt)
            if result.empty:
                continue
            n = len(result)
            wr = (result["pnl_pct"] > 0).mean() * 100
            wins = result[result["pnl_pct"] > 0]
            losses = result[result["pnl_pct"] <= 0]
            gw = wins["pnl_pct"].sum() if len(wins) else 0
            gl = abs(losses["pnl_pct"].sum()) if len(losses) else 0.001
            pf = gw / gl
            n_days = result["date"].nunique()
            daily = result["pnl_usd"].sum() / n_days
            cum = result["pnl_pct"].cumsum()
            dd = (cum - cum.cummax()).min()
            tag = f"{label} | {mt}/day"
            print(f"  {tag:<40} {n:>5}  {wr:>5.1f}%  {pf:>5.2f}  ${daily:>+7.2f}  {dd:>+6.2f}%")
