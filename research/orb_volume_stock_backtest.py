"""
research/orb_volume_stock_backtest.py
---------------------------------------
Tests whether the volume-confirmed ORB edge holds across individual
S&P 500 stocks, not just SPY/QQQ ETFs.

Phase 1: Build ORB universe (avg vol > 5M, avg range > 1%, price > $20)
Phase 2: Collect 60d of 5-min data, backtest baseline vs volume-confirmed
Phase 3: Aggregate results, sector breakdown, statistical significance test

Usage:
    python research/orb_volume_stock_backtest.py
    python research/orb_volume_stock_backtest.py --universe-only   # just build the universe
    python research/orb_volume_stock_backtest.py --skip-download   # use cached 5m data only
"""

import sys
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import time as dtime
from tqdm import tqdm
from loguru import logger
from scipy import stats

sys.path.insert(0, ".")

CACHE_DIR = Path("data/intraday_cache")
UNIVERSE_PATH = Path("data/orb_universe.csv")
RESULTS_PATH = Path("research/orb_volume_stock_results.csv")
OOS_FRACTION = 0.40


# =====================================================================
# Phase 1: Build ORB Universe
# =====================================================================

def build_orb_universe(min_volume=5_000_000, min_range_pct=1.0, min_price=20.0):
    """
    Pull S&P 500, filter to stocks suitable for ORB trading.
    Returns DataFrame with ticker, avg_volume, avg_range_pct, avg_price, sector.
    """
    from data.universe import get_sp500_tickers

    print("\n  Phase 1: Building ORB Universe")
    print("  " + "-" * 50)

    tickers = get_sp500_tickers()
    print(f"  S&P 500: {len(tickers)} tickers")

    # Download 30 days of daily data in chunks (reuse pipeline.py pattern)
    print("  Downloading 30 days of daily data...")
    chunk_size = 100
    chunks = [tickers[i:i+chunk_size] for i in range(0, len(tickers), chunk_size)]
    frames = []
    for i, chunk in enumerate(chunks):
        try:
            raw = yf.download(chunk, period="30d", auto_adjust=True,
                              progress=False, threads=True)
            if not raw.empty:
                frames.append(raw)
        except Exception as e:
            print(f"  WARNING: Chunk {i+1} failed: {e}")
    if not frames:
        print("  ERROR: No data downloaded")
        return pd.DataFrame()

    # Combine (all chunks have same date index, different ticker columns)
    raw = frames[0] if len(frames) == 1 else pd.concat(frames, axis=1)

    # Handle MultiIndex columns from yfinance
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
        high = raw["High"]
        low = raw["Low"]
        volume = raw["Volume"]
    else:
        close = raw[["Close"]]
        high = raw[["High"]]
        low = raw[["Low"]]
        volume = raw[["Volume"]]

    # Compute stats per ticker
    results = []
    for ticker in tickers:
        try:
            if ticker not in close.columns:
                continue
            c = close[ticker].dropna()
            h = high[ticker].dropna()
            l = low[ticker].dropna()
            v = volume[ticker].dropna()
            if len(c) < 10:
                continue

            avg_price = c.mean()
            avg_volume = v.mean()
            # Average daily range as % of close
            # Align indices for range calc
            common = c.index.intersection(h.index).intersection(l.index)
            if len(common) < 10:
                continue
            avg_range_pct = ((h[common] - l[common]) / c[common] * 100).mean()

            results.append({
                "ticker": ticker,
                "avg_price": round(avg_price, 2),
                "avg_volume": int(avg_volume),
                "avg_range_pct": round(avg_range_pct, 2),
            })
        except Exception:
            continue

    df = pd.DataFrame(results)
    print(f"  Stats computed for {len(df)} tickers")

    # Apply filters
    filtered = df[
        (df["avg_volume"] >= min_volume) &
        (df["avg_range_pct"] >= min_range_pct) &
        (df["avg_price"] >= min_price)
    ].copy()
    filtered = filtered.sort_values("avg_volume", ascending=False).reset_index(drop=True)

    print(f"\n  Filters: volume >= {min_volume/1e6:.0f}M, range >= {min_range_pct:.1f}%, price >= ${min_price:.0f}")
    print(f"  Result: {len(filtered)} stocks pass")

    # Try to add sector info from yfinance
    print("  Looking up sectors...")
    sector_map = {}
    for t in filtered["ticker"].tolist():
        try:
            info = yf.Ticker(t).info
            sector_map[t] = info.get("sector", "Unknown")
        except Exception:
            sector_map[t] = "Unknown"
    filtered["sector"] = filtered["ticker"].map(sector_map).fillna("Unknown")

    # Save
    UNIVERSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(UNIVERSE_PATH, index=False)
    print(f"  Saved to {UNIVERSE_PATH}")

    # Print summary
    print(f"\n  {'Ticker':<8} {'AvgVol':>12} {'Range%':>8} {'Price':>8} {'Sector'}")
    print(f"  {'-'*70}")
    for _, row in filtered.iterrows():
        print(f"  {row['ticker']:<8} {row['avg_volume']:>12,.0f} {row['avg_range_pct']:>7.2f}% "
              f"${row['avg_price']:>7.2f} {row.get('sector', '')}")

    if "sector" in filtered.columns:
        print(f"\n  Sector breakdown:")
        for sector, grp in filtered.groupby("sector"):
            print(f"    {sector}: {len(grp)} stocks")

    return filtered


# =====================================================================
# Phase 2: Collect 5-Minute Intraday Data
# =====================================================================

def collect_intraday_data(tickers, interval="5m"):
    """Download 60d of 5-min data per ticker. Cache as parquet."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  Phase 2a: Collecting {interval} data for {len(tickers)} tickers")
    print("  " + "-" * 50)

    success = 0
    failed = []

    for i, ticker in enumerate(tqdm(tickers, desc="  Downloading")):
        cache_path = CACHE_DIR / f"{ticker}_{interval}.parquet"
        try:
            raw = yf.download(ticker, period="60d", interval=interval,
                              auto_adjust=True, progress=False, prepost=False)
            if raw.empty:
                failed.append(ticker)
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

            # Merge with existing cache
            if cache_path.exists():
                existing = pd.read_parquet(cache_path)
                combined = pd.concat([existing, raw]).drop_duplicates(
                    subset="timestamp").sort_values("timestamp")
            else:
                combined = raw

            combined.to_parquet(cache_path, index=False)
            success += 1
        except Exception as e:
            failed.append(ticker)

    print(f"\n  Downloaded: {success}/{len(tickers)} tickers")
    if failed:
        print(f"  Failed ({len(failed)}): {', '.join(failed[:20])}")

    return failed


def load_intraday(ticker, interval="5m"):
    """Load cached 5m data for a ticker. Tries parquet first, then CSV."""
    parquet_path = CACHE_DIR / f"{ticker}_{interval}.parquet"
    csv_path = CACHE_DIR / f"{ticker}_{interval}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    elif csv_path.exists():
        return pd.read_csv(csv_path, parse_dates=["timestamp"])
    return pd.DataFrame()


# =====================================================================
# Phase 2b: Backtest (reuses logic from orb_improvements_backtest.py)
# =====================================================================

def prepare_data(df):
    """Prepare dataframe with date/time columns and timezone handling."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    df["date"] = df["timestamp"].dt.date
    df["time"] = df["timestamp"].dt.time
    return df


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


def backtest_orb(df, require_volume=False, direction="both", max_gap_pct=0.5,
                 slippage_pct=0.0, commission_usd=0.0, position_size_usd=500.0):
    """
    Run ORB backtest. If require_volume=True, only enter when breakout bar
    volume exceeds the OR-period average volume.

    Transaction costs:
      slippage_pct: applied to each side (entry and exit), e.g. 0.02 = 0.02%
      commission_usd: flat $ per trade (round trip), converted to % via position_size_usd
    """
    or_bars = 3  # 15 min
    target_mult = 1.5
    stop_mult = 1.0
    all_days = sorted(df["date"].unique())
    trades = []

    # Round-trip cost as % of position
    cost_pct = (slippage_pct * 2) + (commission_usd / position_size_usd * 100)

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
        remaining = mkt.iloc[or_bars:]
        trade_taken = False

        for _, bar in remaining.iterrows():
            if trade_taken:
                break

            if require_volume and bar["volume"] < or_avg_volume:
                continue

            vol_ratio = bar["volume"] / or_avg_volume if or_avg_volume > 0 else 1.0

            if direction in ("long", "both") and bar["high"] > or_high:
                entry = or_high
                target = entry + target_pts
                stop = entry - stop_pts
                future = remaining[remaining["timestamp"] >= bar["timestamp"]]
                exit_price, reason = find_exit(future, entry, target, stop, "long")
                pnl_pct = (exit_price - entry) / entry * 100 - cost_pct
                trades.append({"date": day, "direction": "long", "entry": entry,
                               "exit": exit_price, "pnl_pct": pnl_pct, "reason": reason,
                               "vol_ratio": vol_ratio})
                trade_taken = True

            elif direction in ("short", "both") and bar["low"] < or_low:
                entry = or_low
                target = entry - target_pts
                stop = entry + stop_pts
                future = remaining[remaining["timestamp"] >= bar["timestamp"]]
                exit_price, reason = find_exit(future, entry, target, stop, "short")
                pnl_pct = (entry - exit_price) / entry * 100 - cost_pct
                trades.append({"date": day, "direction": "short", "entry": entry,
                               "exit": exit_price, "pnl_pct": pnl_pct, "reason": reason,
                               "vol_ratio": vol_ratio})
                trade_taken = True

    return pd.DataFrame(trades)


def filter_top_n_per_day(all_trades_df, n=5):
    """
    Given a DataFrame of trades across all tickers (must have 'ticker', 'date',
    'vol_ratio' columns), keep only the top N per day ranked by vol_ratio
    (highest conviction first).
    """
    if all_trades_df.empty or "vol_ratio" not in all_trades_df.columns:
        return all_trades_df
    ranked = all_trades_df.sort_values("vol_ratio", ascending=False)
    return ranked.groupby("date").head(n).sort_values(["date", "vol_ratio"],
                                                       ascending=[True, False]).reset_index(drop=True)


def compute_stats(trades):
    """Compute summary stats for a set of trades. Returns dict."""
    if trades.empty:
        return {"trades": 0, "win_rate": 0, "pf": 0, "total_pnl": 0,
                "avg_pnl": 0, "max_dd": 0}
    n = len(trades)
    wins = trades[trades["pnl_pct"] > 0]
    losses = trades[trades["pnl_pct"] <= 0]
    gw = wins["pnl_pct"].sum() if len(wins) else 0
    gl = abs(losses["pnl_pct"].sum()) if len(losses) else 0.001
    cum = trades["pnl_pct"].cumsum()
    return {
        "trades": n,
        "win_rate": len(wins) / n * 100,
        "pf": gw / gl if gl > 0 else 0,
        "total_pnl": trades["pnl_pct"].sum(),
        "avg_pnl": trades["pnl_pct"].mean(),
        "max_dd": (cum - cum.cummax()).min(),
    }


# =====================================================================
# Phase 3: Run Backtest Across All Tickers
# =====================================================================

def run_stock_backtest(universe_df, slippage_pct=0.0, commission_usd=0.0,
                       position_size_usd=500.0):
    """Run baseline vs volume-confirmed on each ticker. Return results DataFrame."""
    tickers = universe_df["ticker"].tolist()
    sector_map = dict(zip(universe_df["ticker"], universe_df.get("sector", "Unknown")))

    cost_label = ""
    if slippage_pct > 0 or commission_usd > 0:
        cost_label = f" (slip={slippage_pct:.2f}%/side, comm=${commission_usd:.0f}/trade)"

    print(f"\n  Backtesting {len(tickers)} tickers{cost_label}")
    print("  " + "-" * 50)

    rows = []
    all_baseline_trades = []
    all_vol_trades = []
    skipped = []

    cost_kwargs = dict(slippage_pct=slippage_pct, commission_usd=commission_usd,
                       position_size_usd=position_size_usd)

    for ticker in tqdm(tickers, desc="  Backtesting"):
        df = load_intraday(ticker)
        if df.empty:
            skipped.append(ticker)
            continue

        try:
            df = prepare_data(df)
        except Exception:
            skipped.append(ticker)
            continue

        all_days = sorted(df["date"].unique())
        n_days = len(all_days)
        if n_days < 10:
            skipped.append(ticker)
            continue

        # Split IS/OOS
        split_idx = int(n_days * (1 - OOS_FRACTION))
        oos_start = all_days[split_idx]
        oos_df = df[df["date"] >= oos_start].copy()
        is_df = df[df["date"] < oos_start].copy()

        # Run both strategies on full, IS, and OOS
        for label, split_df in [("full", df), ("is", is_df), ("oos", oos_df)]:
            baseline = backtest_orb(split_df, require_volume=False, **cost_kwargs)
            vol_conf = backtest_orb(split_df, require_volume=True, **cost_kwargs)
            b = compute_stats(baseline)
            v = compute_stats(vol_conf)

            row = {
                "ticker": ticker,
                "sector": sector_map.get(ticker, "Unknown"),
                "split": label,
                "days": split_df["date"].nunique(),
                "base_trades": b["trades"],
                "base_win_rate": round(b["win_rate"], 1),
                "base_pf": round(b["pf"], 2),
                "base_pnl": round(b["total_pnl"], 2),
                "base_max_dd": round(b["max_dd"], 2),
                "vol_trades": v["trades"],
                "vol_win_rate": round(v["win_rate"], 1),
                "vol_pf": round(v["pf"], 2),
                "vol_pnl": round(v["total_pnl"], 2),
                "vol_max_dd": round(v["max_dd"], 2),
                "delta_wr": round(v["win_rate"] - b["win_rate"], 1),
                "delta_pf": round(v["pf"] - b["pf"], 2),
                "delta_pnl": round(v["total_pnl"] - b["total_pnl"], 2),
            }
            rows.append(row)

            if label == "full":
                if not baseline.empty:
                    baseline["ticker"] = ticker
                    all_baseline_trades.append(baseline)
                if not vol_conf.empty:
                    vol_conf["ticker"] = ticker
                    all_vol_trades.append(vol_conf)

    if skipped:
        print(f"\n  Skipped ({len(skipped)}): {', '.join(skipped[:15])}"
              f"{'...' if len(skipped) > 15 else ''}")

    results = pd.DataFrame(rows)
    agg_base = pd.concat(all_baseline_trades) if all_baseline_trades else pd.DataFrame()
    agg_vol = pd.concat(all_vol_trades) if all_vol_trades else pd.DataFrame()

    return results, agg_base, agg_vol


# =====================================================================
# Phase 3: Reporting
# =====================================================================

def print_report(results, agg_base, agg_vol, universe_df):
    """Print full analysis report."""

    # --- Cross-ticker aggregate ---
    print(f"\n{'='*75}")
    print(f"  AGGREGATE RESULTS (all stocks pooled)")
    print(f"{'='*75}")

    for label, name in [("full", "FULL DATASET"), ("is", "IN-SAMPLE"), ("oos", "OUT-OF-SAMPLE")]:
        split = results[results["split"] == label]
        active = split[split["base_trades"] > 0]
        print(f"\n  {name} ({len(active)} tickers with trades)")
        print(f"  {'-'*65}")

        bt = active["base_trades"].sum()
        bw = (active["base_win_rate"] * active["base_trades"] / 100).sum()
        vt = active["vol_trades"].sum()
        vw = (active["vol_win_rate"] * active["vol_trades"] / 100).sum()

        b_wr = bw / bt * 100 if bt else 0
        v_wr = vw / vt * 100 if vt else 0

        # Aggregate PF from pooled trades
        if label == "full" and not agg_base.empty and not agg_vol.empty:
            b_gw = agg_base[agg_base["pnl_pct"] > 0]["pnl_pct"].sum()
            b_gl = abs(agg_base[agg_base["pnl_pct"] <= 0]["pnl_pct"].sum()) or 0.001
            v_gw = agg_vol[agg_vol["pnl_pct"] > 0]["pnl_pct"].sum()
            v_gl = abs(agg_vol[agg_vol["pnl_pct"] <= 0]["pnl_pct"].sum()) or 0.001
            b_pf = b_gw / b_gl
            v_pf = v_gw / v_gl
        else:
            # Weighted average PF as approximation for IS/OOS
            b_pf = active["base_pf"].mean() if len(active) else 0
            v_pf = active["vol_pf"].mean() if len(active) else 0

        b_pnl = active["base_pnl"].sum()
        v_pnl = active["vol_pnl"].sum()

        print(f"  {'':30} {'Baseline':>12} {'Vol-Confirmed':>14} {'Delta':>8}")
        print(f"  {'Total trades':<30} {bt:>12} {vt:>14} {vt-bt:>+8}")
        print(f"  {'Aggregate win rate':<30} {b_wr:>11.1f}% {v_wr:>13.1f}% {v_wr-b_wr:>+7.1f}%")
        print(f"  {'Aggregate profit factor':<30} {b_pf:>12.2f} {v_pf:>14.2f} {v_pf-b_pf:>+8.2f}")
        print(f"  {'Total P&L (sum of %)':<30} {b_pnl:>+11.2f}% {v_pnl:>+13.2f}% {v_pnl-b_pnl:>+7.2f}%")

    # --- Consistency check ---
    full = results[results["split"] == "full"]
    active = full[full["base_trades"] > 0].copy()

    improved = active[active["delta_wr"] > 0]
    degraded = active[active["delta_wr"] < 0]
    unchanged = active[active["delta_wr"] == 0]

    print(f"\n{'='*75}")
    print(f"  CONSISTENCY CHECK (full dataset, {len(active)} tickers with trades)")
    print(f"{'='*75}")
    print(f"  Win rate improved:   {len(improved):>4} tickers ({len(improved)/len(active)*100:.0f}%)")
    print(f"  Win rate degraded:   {len(degraded):>4} tickers ({len(degraded)/len(active)*100:.0f}%)")
    print(f"  Win rate unchanged:  {len(unchanged):>4} tickers ({len(unchanged)/len(active)*100:.0f}%)")

    pf_improved = active[active["delta_pf"] > 0]
    pf_degraded = active[active["delta_pf"] < 0]
    print(f"  PF improved:         {len(pf_improved):>4} tickers ({len(pf_improved)/len(active)*100:.0f}%)")
    print(f"  PF degraded:         {len(pf_degraded):>4} tickers ({len(pf_degraded)/len(active)*100:.0f}%)")

    # --- Statistical test ---
    # Paired sign test: is volume-confirmed win rate systematically higher?
    paired = active[active["vol_trades"] > 0].copy()
    if len(paired) >= 5:
        diffs = paired["delta_wr"].values
        n_pos = (diffs > 0).sum()
        n_neg = (diffs < 0).sum()
        n_total = n_pos + n_neg
        if n_total > 0:
            # Binomial test: under null, P(improve) = 0.5
            p_value = stats.binom_test(n_pos, n_total, 0.5) if hasattr(stats, 'binom_test') else \
                      stats.binomtest(n_pos, n_total, 0.5).pvalue
            print(f"\n  Sign test (win rate improvement):")
            print(f"    {n_pos} improved, {n_neg} degraded")
            print(f"    p-value = {p_value:.4f} {'(significant at 5%)' if p_value < 0.05 else '(not significant)'}")

        # Paired t-test on P&L
        if len(paired) >= 10:
            t_stat, t_pval = stats.ttest_rel(paired["vol_pnl"], paired["base_pnl"])
            print(f"\n  Paired t-test (total P&L):")
            print(f"    t = {t_stat:.3f}, p = {t_pval:.4f} "
                  f"{'(significant at 5%)' if t_pval < 0.05 else '(not significant)'}")

    # --- Sector breakdown ---
    if "sector" in active.columns:
        print(f"\n{'='*75}")
        print(f"  SECTOR BREAKDOWN (full dataset)")
        print(f"{'='*75}")
        print(f"  {'Sector':<28} {'N':>3} {'Base WR':>9} {'Vol WR':>9} {'Delta':>7} {'Base PnL':>10} {'Vol PnL':>10}")
        print(f"  {'-'*78}")

        for sector, grp in active.groupby("sector"):
            n = len(grp)
            if n < 2:
                continue
            b_wr = (grp["base_win_rate"] * grp["base_trades"]).sum() / max(grp["base_trades"].sum(), 1)
            v_wr = (grp["vol_win_rate"] * grp["vol_trades"]).sum() / max(grp["vol_trades"].sum(), 1)
            b_pnl = grp["base_pnl"].sum()
            v_pnl = grp["vol_pnl"].sum()
            print(f"  {sector:<28} {n:>3} {b_wr:>8.1f}% {v_wr:>8.1f}% {v_wr-b_wr:>+6.1f}% "
                  f"{b_pnl:>+9.2f}% {v_pnl:>+9.2f}%")

    # --- Per-ticker table (top and bottom) ---
    print(f"\n{'='*75}")
    print(f"  PER-TICKER RESULTS (full dataset, sorted by WR improvement)")
    print(f"{'='*75}")
    active_sorted = active.sort_values("delta_wr", ascending=False)
    print(f"  {'Ticker':<7} {'Sector':<22} {'BaseTr':>6} {'VolTr':>6} "
          f"{'BaseWR':>7} {'VolWR':>7} {'dWR':>6} "
          f"{'BasePnL':>8} {'VolPnL':>8}")
    print(f"  {'-'*82}")
    for _, row in active_sorted.iterrows():
        print(f"  {row['ticker']:<7} {str(row.get('sector',''))[:21]:<22} "
              f"{row['base_trades']:>6} {row['vol_trades']:>6} "
              f"{row['base_win_rate']:>6.1f}% {row['vol_win_rate']:>6.1f}% {row['delta_wr']:>+5.1f}% "
              f"{row['base_pnl']:>+7.2f}% {row['vol_pnl']:>+7.2f}%")

    # Save full results
    results.to_csv(RESULTS_PATH, index=False)
    print(f"\n  Full results saved to {RESULTS_PATH}")


# =====================================================================
# Main
# =====================================================================

def print_aggregate_line(label, trades):
    """Print one-line aggregate stats for a set of pooled trades."""
    if trades.empty:
        print(f"  {label:<40} -- no trades --")
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
    print(f"  {label:<40} {n:>5} trades  {wr:>5.1f}% WR  {pf:>5.2f} PF  "
          f"{total:>+8.2f}% total  {avg:>+6.3f}% avg  {dd:>+6.2f}% DD")


def run_costs_analysis(slip=0.02, comm=1.00, size=2000.0, max_exposure=10000.0):
    """Compare no-cost vs realistic costs, and sweep position caps for exposure limit."""

    # Load universe
    if not UNIVERSE_PATH.exists():
        print("  No ORB universe found. Run without --costs first.")
        sys.exit(1)
    universe_df = pd.read_csv(UNIVERSE_PATH)
    tickers = universe_df["ticker"].tolist()
    print(f"  Loaded {len(tickers)} tickers from {UNIVERSE_PATH}")

    rt_cost = slip * 2 + comm / size * 100
    max_positions = int(max_exposure / size)

    print(f"\n  Cost assumptions:")
    print(f"    Slippage:       {slip:.2f}% per side")
    print(f"    Commission:     ${comm:.2f} per trade")
    print(f"    Position size:  ${size:,.0f}")
    print(f"    Round-trip cost: {rt_cost:.3f}% per trade")
    print(f"    Max exposure:   ${max_exposure:,.0f} = {max_positions} positions @ ${size:,.0f}")

    # -- Run 1: Vol-confirmed, no costs --
    print(f"\n{'='*75}")
    print(f"  RUN 1: Volume-Confirmed, NO costs")
    print(f"{'='*75}")
    _, _, vol_nocost = run_stock_backtest(universe_df, slippage_pct=0, commission_usd=0)

    # -- Run 2: Baseline with costs --
    print(f"\n{'='*75}")
    print(f"  RUN 2: Baseline (no volume filter), WITH costs")
    print(f"{'='*75}")
    _, base_costs, _ = run_stock_backtest(universe_df, slippage_pct=slip,
                                           commission_usd=comm, position_size_usd=size)

    # -- Run 3: Vol-confirmed with costs --
    print(f"\n{'='*75}")
    print(f"  RUN 3: Volume-Confirmed, WITH costs")
    print(f"{'='*75}")
    _, _, vol_costs = run_stock_backtest(universe_df, slippage_pct=slip,
                                         commission_usd=comm, position_size_usd=size)

    # ── Summary comparison ──
    print(f"\n{'='*75}")
    print(f"  COMPARISON: ${size:,.0f} positions, {slip:.2f}%/side slip, ${comm:.0f} comm")
    print(f"  Round-trip cost: {rt_cost:.3f}% per trade")
    print(f"{'='*75}")
    print(f"  {'Strategy':<40} {'Trades':>6} {'WR':>7} {'PF':>6} "
          f"{'Total':>9} {'AvgTrd':>8} {'MaxDD':>7}")
    print(f"  {'-'*82}")
    scenarios = [
        ("Baseline + costs", base_costs),
        ("Vol-confirmed, no costs", vol_nocost),
        ("Vol-confirmed + costs", vol_costs),
    ]
    for label, trades in scenarios:
        print_aggregate_line(label, trades)

    # ── Position cap sweep ──
    # For each cap N, keep top N trades per day by vol_ratio, then compute
    # daily P&L in dollar terms (each position = $size) and total exposure
    if vol_costs.empty:
        return

    print(f"\n{'='*75}")
    print(f"  POSITION CAP SWEEP (max daily exposure = ${max_exposure:,.0f})")
    print(f"  Each position = ${size:,.0f}  |  Max positions = {max_positions}")
    print(f"{'='*75}")

    # Show results for caps from 1 to the natural max (where uncapped = all)
    day_counts = vol_costs.groupby("date").size()
    natural_max = int(day_counts.max())
    caps_to_test = list(range(1, min(natural_max + 1, 16)))  # up to 15

    print(f"\n  {'Cap':>4} {'Trades':>7} {'WR':>7} {'PF':>6} {'TotalPnL':>10} "
          f"{'AvgTrd':>8} {'MaxDD':>8} {'Exposure':>10} {'$ P&L':>10}")
    print(f"  {'-'*82}")

    for cap in caps_to_test:
        capped = filter_top_n_per_day(vol_costs, n=cap)
        if capped.empty:
            continue
        n = len(capped)
        wins = capped[capped["pnl_pct"] > 0]
        losses = capped[capped["pnl_pct"] <= 0]
        wr = len(wins) / n * 100
        gw = wins["pnl_pct"].sum() if len(wins) else 0
        gl = abs(losses["pnl_pct"].sum()) if len(losses) else 0.001
        pf = gw / gl
        total_pnl = capped["pnl_pct"].sum()
        avg_pnl = capped["pnl_pct"].mean()
        cum = capped["pnl_pct"].cumsum()
        dd = (cum - cum.cummax()).min()
        daily_exposure = cap * size
        # Approximate $ P&L: each trade's pnl_pct applies to $size
        dollar_pnl = (capped["pnl_pct"] / 100 * size).sum()

        marker = " <-- max exposure" if cap == max_positions else ""
        print(f"  {cap:>4} {n:>7} {wr:>6.1f}% {pf:>5.2f} {total_pnl:>+9.2f}% "
              f"{avg_pnl:>+7.3f}% {dd:>+7.2f}% ${daily_exposure:>9,.0f} ${dollar_pnl:>+9.2f}{marker}")

    # ── Detailed look at the target cap ──
    target_cap = max_positions
    target_trades = filter_top_n_per_day(vol_costs, n=target_cap)

    if not target_trades.empty:
        print(f"\n{'='*75}")
        print(f"  DETAIL: {target_cap} positions/day (${target_cap * size:,.0f} exposure)")
        print(f"{'='*75}")

        n = len(target_trades)
        wins = target_trades[target_trades["pnl_pct"] > 0]
        losses = target_trades[target_trades["pnl_pct"] <= 0]
        wr = len(wins) / n * 100
        gw = wins["pnl_pct"].sum() if len(wins) else 0
        gl = abs(losses["pnl_pct"].sum()) if len(losses) else 0.001
        pf = gw / gl
        total_pnl_pct = target_trades["pnl_pct"].sum()
        dollar_pnl = (target_trades["pnl_pct"] / 100 * size).sum()

        print(f"  Trades: {n}  |  Win Rate: {wr:.1f}%  |  PF: {pf:.2f}")
        print(f"  Total P&L: {total_pnl_pct:+.2f}% (${dollar_pnl:+,.2f} on ${size:,.0f} positions)")

        # Daily P&L distribution
        daily_pnl = target_trades.groupby("date")["pnl_pct"].sum()
        daily_dollar = daily_pnl / 100 * size * target_cap  # approx: cap positions * size
        # More precise: sum each trade's $ pnl per day
        target_trades_d = target_trades.copy()
        target_trades_d["dollar_pnl"] = target_trades_d["pnl_pct"] / 100 * size
        daily_dollar = target_trades_d.groupby("date")["dollar_pnl"].sum()

        n_days = len(daily_dollar)
        win_days = (daily_dollar > 0).sum()
        lose_days = (daily_dollar <= 0).sum()

        print(f"\n  Daily P&L distribution ({n_days} trading days):")
        print(f"    Winning days: {win_days} ({win_days/n_days*100:.0f}%)")
        print(f"    Losing days:  {lose_days} ({lose_days/n_days*100:.0f}%)")
        print(f"    Avg daily $:  ${daily_dollar.mean():+,.2f}")
        print(f"    Best day:     ${daily_dollar.max():+,.2f}")
        print(f"    Worst day:    ${daily_dollar.min():+,.2f}")
        print(f"    Daily $ std:  ${daily_dollar.std():,.2f}")

        # Trades per day histogram
        day_counts = target_trades.groupby("date").size()
        print(f"\n  Trades per day: min={day_counts.min()}, "
              f"median={day_counts.median():.0f}, mean={day_counts.mean():.1f}, "
              f"max={day_counts.max()}")

        # Exit reason breakdown
        reasons = target_trades["reason"].value_counts()
        print(f"\n  Exit reasons: {', '.join(f'{r}={c}' for r, c in reasons.items())}")


if __name__ == "__main__":
    print("\n" + "=" * 75)
    print("  VOLUME-CONFIRMED ORB -- Cross-Stock Validation")
    print("=" * 75)

    # Phase 1: Build or load universe
    if UNIVERSE_PATH.exists() and "--rebuild" not in sys.argv:
        print(f"\n  Loading existing ORB universe from {UNIVERSE_PATH}")
        universe_df = pd.read_csv(UNIVERSE_PATH)
        print(f"  {len(universe_df)} tickers loaded")
    else:
        universe_df = build_orb_universe()

    if universe_df.empty:
        print("  ERROR: No tickers in ORB universe. Exiting.")
        sys.exit(1)

    if "--universe-only" in sys.argv:
        sys.exit(0)

    tickers = universe_df["ticker"].tolist()

    # Phase 2a: Collect intraday data
    if "--skip-download" not in sys.argv:
        failed = collect_intraday_data(tickers)
    else:
        print("\n  Skipping download (--skip-download)")

    if "--costs" in sys.argv:
        # Parse optional --size and --exposure flags
        pos_size = 2000.0
        max_exp = 10000.0
        for i, arg in enumerate(sys.argv):
            if arg == "--size" and i + 1 < len(sys.argv):
                pos_size = float(sys.argv[i + 1])
            if arg == "--exposure" and i + 1 < len(sys.argv):
                max_exp = float(sys.argv[i + 1])
        run_costs_analysis(slip=0.02, comm=1.00, size=pos_size, max_exposure=max_exp)
    else:
        # Phase 2b + 3: Backtest and report (original zero-cost run)
        results, agg_base, agg_vol = run_stock_backtest(universe_df)

        if results.empty:
            print("  ERROR: No backtest results. Check data collection.")
            sys.exit(1)

        print_report(results, agg_base, agg_vol, universe_df)
