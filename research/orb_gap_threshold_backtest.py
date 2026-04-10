"""
research/orb_gap_threshold_backtest.py
---------------------------------------
Tests different gap filter thresholds to find optimal setting.
Also tests VIX-adaptive gap thresholds.

Current live setting: MAX_GAP_PCT = 0.5% (too tight in volatile markets).
Tests: 0.5%, 1.0%, 1.5%, 2.0%, 2.5%, 3.0%, no limit.

Uses 2-min OR, 0.3x trail, 1.0x stop, volume confirmed, 1 trade/day.
"""

import sys
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import time as dtime, datetime, timedelta
from tqdm import tqdm

sys.path.insert(0, ".")

CACHE_DIR = Path("data/intraday_cache")
UNIVERSE_PATH = Path("data/orb_universe.csv")

# ── Match live config ─────────────────────────────────────────────────
POSITION_SIZE_USD = 1900.0
COMMISSION_RT_USD = 5.50
SLIPPAGE_PCT = 0.02
STOP_MULT = 1.0
TRAIL_MULT = 0.3
OR_MINUTES = 2
MAX_TRADES_PER_DAY = 1

# ── Gap thresholds to test ────────────────────────────────────────────
GAP_THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, None]  # None = no limit

# ── VIX-adaptive thresholds ───────────────────────────────────────────
VIX_ADAPTIVE = {
    "calm":     {"vix_max": 15, "gap_max": 0.5},
    "normal":   {"vix_max": 20, "gap_max": 1.0},
    "elevated": {"vix_max": 25, "gap_max": 1.5},
    "stressed": {"vix_max": 35, "gap_max": 2.5},
    "extreme":  {"vix_max": 999, "gap_max": 3.0},
}


def collect_1min_data(tickers):
    """Download 1-min data per ticker in 7-day chunks."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  Downloading 1-min data for {len(tickers)} tickers (7-day chunks)...")
    success = 0
    today = datetime.now()
    chunks = []
    for i in range(4):
        end = today - timedelta(days=i * 7)
        start = end - timedelta(days=7)
        chunks.append((start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
    chunks.reverse()

    for ticker in tqdm(tickers, desc="  Downloading"):
        cache_path = CACHE_DIR / f"{ticker}_1m.parquet"
        if cache_path.exists():
            # Use cache if fresh (< 12 hours old)
            mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
            if (datetime.now() - mtime).total_seconds() < 43200:
                success += 1
                continue
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
        combined = pd.concat(frames).reset_index()
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

    print(f"  Downloaded/cached: {success}/{len(tickers)}")


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


def exit_trailing(bars, entry, or_range):
    """Trailing stop exit matching live system."""
    stop = entry - or_range * STOP_MULT
    trail_dist = or_range * TRAIL_MULT
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


def get_vix_data():
    """Download daily VIX for the backtest period."""
    try:
        vix = yf.download("^VIX", period="60d", auto_adjust=True, progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = [c[0] for c in vix.columns]
        vix = vix.reset_index()
        vix.columns = [c.lower() for c in vix.columns]
        vix["date"] = pd.to_datetime(vix["date"]).dt.date
        return vix.set_index("date")["close"].to_dict()
    except Exception as e:
        print(f"  WARNING: Could not fetch VIX data: {e}")
        return {}


def adaptive_gap_for_vix(vix_level):
    """Return gap threshold based on VIX level."""
    for regime, params in sorted(VIX_ADAPTIVE.items(), key=lambda x: x[1]["vix_max"]):
        if vix_level <= params["vix_max"]:
            return params["gap_max"]
    return 3.0


def backtest_gap_threshold(all_ticker_data, gap_max, vix_data=None, adaptive=False):
    """
    Run ORB backtest with a specific gap threshold.
    If adaptive=True, use VIX-based adaptive thresholds (ignores gap_max).
    """
    cost_pct = (SLIPPAGE_PCT * 2) + (COMMISSION_RT_USD / POSITION_SIZE_USD * 100)
    total_mins = 9 * 60 + 30 + OR_MINUTES
    or_end_time = dtime(total_mins // 60, total_mins % 60)

    all_dates = set()
    for ticker, df in all_ticker_data.items():
        all_dates.update(df["date"].unique())
    all_dates = sorted(all_dates)

    # Build prev_close lookup
    prev_closes = {}
    for ticker, df in all_ticker_data.items():
        daily = df.groupby("date").agg(
            first_open=("open", "first"),
            last_close=("close", "last"),
        )
        prev_closes[ticker] = daily

    trades = []
    days_with_candidates = 0
    days_all_filtered = 0
    gap_filter_counts = []

    for day in all_dates:
        day_candidates = []
        tickers_checked = 0
        tickers_gap_filtered = 0

        # Determine gap threshold for this day
        if adaptive and vix_data:
            # Find most recent VIX close
            vix_val = None
            for offset in range(5):
                check_day = day - timedelta(days=offset) if isinstance(day, datetime) else \
                    datetime.combine(day, datetime.min.time()).date() - timedelta(days=offset)
                if check_day in vix_data:
                    vix_val = vix_data[check_day]
                    break
            day_gap_max = adaptive_gap_for_vix(vix_val) if vix_val else 1.5
        else:
            day_gap_max = gap_max

        for ticker, df in all_ticker_data.items():
            day_df = df[df["date"] == day]
            mkt = day_df[(day_df["time"] >= dtime(9, 30)) & (day_df["time"] <= dtime(15, 55))]
            if len(mkt) < OR_MINUTES + 10:
                continue

            tickers_checked += 1
            or_data = mkt[mkt["time"] < or_end_time]
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
            if day_gap_max is not None and ticker in prev_closes:
                pc = prev_closes[ticker]
                day_dates = sorted(pc.index)
                day_idx = day_dates.index(day) if day in day_dates else -1
                if day_idx > 0:
                    prev_close = pc.loc[day_dates[day_idx - 1], "last_close"]
                    gap = abs(mkt.iloc[0]["open"] / prev_close - 1) * 100
                    if gap > day_gap_max:
                        tickers_gap_filtered += 1
                        continue

            # Find volume-confirmed breakout
            remaining = mkt[mkt["time"] >= or_end_time]
            for _, bar in remaining.iterrows():
                if bar["time"] >= dtime(14, 0):
                    break
                if bar["volume"] < or_avg_volume:
                    continue
                vol_ratio = bar["volume"] / or_avg_volume if or_avg_volume > 0 else 1.0

                if bar["high"] > or_high:
                    future = remaining[remaining["timestamp"] >= bar["timestamp"]]
                    exit_price, reason = exit_trailing(future, or_high, or_range)
                    day_candidates.append({
                        "ticker": ticker,
                        "date": day,
                        "entry": or_high,
                        "exit": exit_price,
                        "reason": reason,
                        "vol_ratio": vol_ratio,
                        "entry_time": bar["time"],
                        "or_range_pct": or_range / or_mid * 100,
                        "gap_pct": gap if 'gap' in dir() else 0,
                    })
                    break

        gap_filter_counts.append(tickers_gap_filtered)

        if day_candidates:
            days_with_candidates += 1
            day_candidates.sort(key=lambda x: x["vol_ratio"], reverse=True)
            for c in day_candidates[:MAX_TRADES_PER_DAY]:
                pnl_pct = (c["exit"] - c["entry"]) / c["entry"] * 100 - cost_pct
                pnl_usd = pnl_pct / 100 * POSITION_SIZE_USD
                c["pnl_pct"] = pnl_pct
                c["pnl_usd"] = pnl_usd
                trades.append(c)
        else:
            if tickers_checked > 0:
                days_all_filtered += 1

    return pd.DataFrame(trades), days_with_candidates, days_all_filtered, gap_filter_counts


def stats_line(trades, label, days_traded, days_filtered):
    if trades.empty:
        print(f"  {label:<50} -- no trades -- ({days_filtered} days filtered out)")
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
    total_days = days_traded + days_filtered
    usd_per_day = total / 100 * POSITION_SIZE_USD / max(total_days, 1)

    print(f"  {label:<50} {n:>3} tr   {wr:5.1f}% WR   {pf:5.2f} PF   "
          f"{total:+7.2f}%  {avg:+.3f}% avg   {dd:+6.2f}% DD  "
          f"${usd_per_day:+.2f}/day  traded:{days_traded} sat-out:{days_filtered}")


if __name__ == "__main__":
    # Load universe
    if UNIVERSE_PATH.exists():
        uni = pd.read_csv(UNIVERSE_PATH)
        tickers = uni["ticker"].tolist() if "ticker" in uni.columns else uni.iloc[:, 0].tolist()
    else:
        print("ERROR: No ORB universe. Run orb_volume_stock_backtest.py --universe-only")
        sys.exit(1)

    print(f"\n{'=' * 90}")
    print(f"  GAP FILTER THRESHOLD TEST")
    print(f"  2-min OR | 0.3x trail | vol confirmed | 1 trade/day | $1,900 | $5.50 RT")
    print(f"{'=' * 90}")

    # Download data (uses cache if fresh)
    collect_1min_data(tickers)

    # Load all ticker data
    print(f"\n  Loading data...")
    all_ticker_data = {}
    for ticker in tickers:
        df = load_1min(ticker)
        if df.empty:
            continue
        df = prepare(df)
        if len(df) > 100:
            all_ticker_data[ticker] = df
    print(f"  Loaded {len(all_ticker_data)} tickers")

    # Get VIX data for adaptive test
    print(f"  Fetching VIX data...")
    vix_data = get_vix_data()
    print(f"  VIX data: {len(vix_data)} days")

    # ── Test fixed thresholds ─────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print(f"  FIXED GAP THRESHOLDS")
    print(f"{'=' * 90}")

    results = {}
    for gap_max in GAP_THRESHOLDS:
        label = f"Gap <= {gap_max}%" if gap_max else "No gap filter"
        trades, days_traded, days_filtered, _ = backtest_gap_threshold(
            all_ticker_data, gap_max
        )
        stats_line(trades, label, days_traded, days_filtered)
        results[gap_max] = {
            "trades": trades, "days_traded": days_traded,
            "days_filtered": days_filtered
        }

    # ── Test VIX-adaptive threshold ───────────────────────────────────
    print(f"\n{'=' * 90}")
    print(f"  VIX-ADAPTIVE GAP THRESHOLD")
    print(f"  VIX < 15: 0.5% | VIX 15-20: 1.0% | VIX 20-25: 1.5% | VIX 25-35: 2.5% | VIX 35+: 3.0%")
    print(f"{'=' * 90}")

    trades, days_traded, days_filtered, gap_counts = backtest_gap_threshold(
        all_ticker_data, None, vix_data=vix_data, adaptive=True
    )
    stats_line(trades, "VIX-adaptive gap filter", days_traded, days_filtered)

    # ── Comparison summary ────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print(f"  SUMMARY: RANKED BY PROFIT FACTOR")
    print(f"{'=' * 90}")
    print(f"  {'Config':<50} {'Trades':>6}  {'WR':>7}  {'PF':>7}  {'Total':>9}  {'MaxDD':>8}  {'$/day':>8}")
    print(f"  {'-' * 95}")

    all_results = []
    for gap_max, r in results.items():
        t = r["trades"]
        if t.empty:
            continue
        n = len(t)
        wr = len(t[t["pnl_pct"] > 0]) / n * 100
        gw = t[t["pnl_pct"] > 0]["pnl_pct"].sum()
        gl = abs(t[t["pnl_pct"] <= 0]["pnl_pct"].sum()) or 0.001
        pf = gw / gl
        total = t["pnl_pct"].sum()
        cum = t["pnl_pct"].cumsum()
        dd = (cum - cum.cummax()).min()
        total_days = r["days_traded"] + r["days_filtered"]
        usd = total / 100 * POSITION_SIZE_USD / max(total_days, 1)
        label = f"Gap <= {gap_max}%" if gap_max else "No gap filter"
        all_results.append((pf, label, n, wr, total, dd, usd))

    # Add adaptive
    if not trades.empty:
        t = trades
        n = len(t)
        wr = len(t[t["pnl_pct"] > 0]) / n * 100
        gw = t[t["pnl_pct"] > 0]["pnl_pct"].sum()
        gl = abs(t[t["pnl_pct"] <= 0]["pnl_pct"].sum()) or 0.001
        pf = gw / gl
        total = t["pnl_pct"].sum()
        cum = t["pnl_pct"].cumsum()
        dd = (cum - cum.cummax()).min()
        total_days = days_traded + days_filtered
        usd = total / 100 * POSITION_SIZE_USD / max(total_days, 1)
        all_results.append((pf, "VIX-ADAPTIVE", n, wr, total, dd, usd))

    all_results.sort(key=lambda x: x[0], reverse=True)
    for pf, label, n, wr, total, dd, usd in all_results:
        marker = " <<<" if label == "VIX-ADAPTIVE" else ""
        print(f"  {label:<50} {n:>4}    {wr:5.1f}%   {pf:5.2f}   {total:+7.2f}%   {dd:+6.2f}%   ${usd:+.2f}{marker}")

    print()
