"""
research/orb_relief_bounce_study.py
-------------------------------------
Tests whether ORB breakout performance degrades on "relief bounce" days --
days where SPY gaps up after 3+ consecutive down days.

Also slices ORB results by VIX level and SPY prior-3-day return to quantify
regime effects.

Usage:
    python research/orb_relief_bounce_study.py
"""

import sys
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import time as dtime, date as Date, timedelta

sys.path.insert(0, ".")

CACHE_DIR = Path("data/intraday_cache")
RESULTS_DIR = Path("research/results")

# -- Match live ORB config (orb_trader.py ORBConfig) ----------------
POSITION_SIZE_USD = 1900.0
COMMISSION_RT_USD = 5.50       # round-trip commission
SLIPPAGE_PCT      = 0.02       # per side
STOP_MULT         = 1.0        # initial stop = 1.0x OR range
TRAIL_MULT        = 0.3        # trailing stop = 0.3x OR range
OR_MINUTES        = 2          # 2-min opening range
MAX_GAP_PCT       = 1.0        # skip if gap > 1.0%
LAST_ENTRY        = dtime(14, 0)   # no new entries after 2 PM

# -- Relief bounce thresholds --------------------------------------
MIN_CONSECUTIVE_DOWN = 3       # SPY must be down N+ consecutive days
MIN_GAP_UP_PCT       = 0.3    # ... and today gaps up at least 0.3%


# =======================================================================
# Data Loading
# =======================================================================

def load_all_intraday():
    """Load all 1-min parquet files from cache. Returns {ticker: DataFrame}."""
    data = {}
    for path in sorted(CACHE_DIR.glob("*_1m.parquet")):
        ticker = path.stem.replace("_1m", "")
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_convert(
                "America/New_York").dt.tz_localize(None)
        df["date"] = df["timestamp"].dt.date
        df["time"] = df["timestamp"].dt.time
        data[ticker] = df
    print(f"  Loaded {len(data)} tickers from 1-min cache")
    return data


def build_market_context():
    """
    Download SPY + VIX daily data and compute per-day market context.
    All values use prior-day data (available pre-market).
    """
    print("  Downloading SPY + VIX daily data...")
    spy = yf.download("SPY", period="120d", auto_adjust=True, progress=False)
    vix = yf.download("^VIX", period="120d", auto_adjust=True, progress=False)

    for df in (spy, vix):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

    spy = spy.reset_index()
    spy.columns = [c.lower() for c in spy.columns]
    spy["date"] = pd.to_datetime(spy["date"]).dt.date
    spy = spy.sort_values("date").reset_index(drop=True)

    vix = vix.reset_index()
    vix.columns = [c.lower() for c in vix.columns]
    vix["date"] = pd.to_datetime(vix["date"]).dt.date
    vix = vix.sort_values("date").reset_index(drop=True)

    # SPY features
    spy["prev_close"] = spy["close"].shift(1)
    spy["gap_pct"] = (spy["open"] / spy["prev_close"] - 1) * 100
    spy["ret_1d"] = spy["close"].pct_change() * 100
    spy["ret_3d"] = spy["close"].pct_change(3).shift(0) * 100  # last 3 days ending yesterday

    # Consecutive down days (close < prev close)
    spy["down_day"] = spy["close"] < spy["prev_close"]
    consec_down = []
    streak = 0
    for is_down in spy["down_day"]:
        if is_down:
            streak += 1
        else:
            streak = 0
        consec_down.append(streak)
    spy["consec_down_raw"] = consec_down
    # Shift so day T sees streak as of T-1 (pre-market knowledge)
    spy["consec_down"] = spy["consec_down_raw"].shift(1)

    # Shift SPY 3d return so it's T-3 to T-1 (known pre-market)
    spy["spy_3d_return"] = spy["ret_3d"].shift(1)

    # VIX features (use T-1 close, known pre-market)
    vix_lookup = vix.set_index("date")["close"].to_dict()
    spy["vix"] = spy["date"].map(vix_lookup)
    spy["vix"] = spy["vix"].ffill()  # fill weekends/gaps
    spy["vix_prev"] = spy["vix"].shift(1)
    spy["vix_change_pct"] = (spy["vix"] - spy["vix_prev"]) / spy["vix_prev"] * 100

    # Relief bounce flag
    spy["relief_bounce"] = (
        (spy["consec_down"] >= MIN_CONSECUTIVE_DOWN) &
        (spy["gap_pct"] > MIN_GAP_UP_PCT)
    )

    ctx = spy[["date", "open", "close", "prev_close", "gap_pct",
               "spy_3d_return", "consec_down", "vix", "vix_change_pct",
               "relief_bounce"]].copy()
    ctx = ctx.rename(columns={"open": "spy_open", "close": "spy_close",
                              "prev_close": "spy_prev_close"})
    print(f"  Market context: {len(ctx)} trading days")
    return ctx


# =======================================================================
# ORB Backtest Engine
# =======================================================================

def exit_trailing_detailed(bars, entry, or_range):
    """
    Trailing stop exit matching the live bot.
    Returns (exit_price, reason, peak_price, exit_time).
    """
    stop = entry - or_range * STOP_MULT
    trail_dist = or_range * TRAIL_MULT
    highest = entry
    trail_active = False

    for _, bar in bars.iterrows():
        if bar["low"] <= stop:
            reason = "trail" if trail_active else "stop"
            return stop, reason, highest, bar["time"]
        if bar["high"] > highest:
            highest = bar["high"]
            trail_stop = highest - trail_dist
            if trail_stop > stop:
                stop = trail_stop
                trail_active = True
        if bar["time"] >= dtime(15, 55):
            return bar["close"], "eod", highest, bar["time"]
    if len(bars) > 0:
        return bars.iloc[-1]["close"], "eod", highest, bars.iloc[-1]["time"]
    return entry, "flat", entry, None


def run_all_orb_trades(all_ticker_data):
    """
    Run ORB backtest across all tickers and days.
    Returns (all_trades_df, best_per_day_df).
    """
    cost_pct = (SLIPPAGE_PCT * 2) + (COMMISSION_RT_USD / POSITION_SIZE_USD * 100)
    or_end = dtime(9, 30 + OR_MINUTES)  # 9:32

    # Collect all unique dates
    all_dates = set()
    for df in all_ticker_data.values():
        all_dates.update(df["date"].unique())
    all_dates = sorted(all_dates)

    # Build per-ticker prev_close lookup
    prev_closes = {}
    for ticker, df in all_ticker_data.items():
        daily = df.groupby("date").agg(
            first_open=("open", "first"),
            last_close=("close", "last"),
        )
        prev_closes[ticker] = daily

    trades = []
    for day in all_dates:
        for ticker, df in all_ticker_data.items():
            day_df = df[df["date"] == day]
            mkt = day_df[(day_df["time"] >= dtime(9, 30)) &
                         (day_df["time"] <= dtime(15, 55))]
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
            if ticker in prev_closes:
                pc = prev_closes[ticker]
                prev_days = [d for d in pc.index if d < day]
                if prev_days:
                    prev_close = pc.loc[prev_days[-1], "last_close"]
                    gap = abs(mkt.iloc[0]["open"] / prev_close - 1) * 100
                    if gap > MAX_GAP_PCT:
                        continue

            # Scan for breakout
            remaining = mkt[(mkt["time"] >= or_end) & (mkt["time"] <= LAST_ENTRY)]
            for _, bar in remaining.iterrows():
                # Volume confirmation
                if bar["volume"] < or_avg_volume:
                    continue
                if bar["high"] > or_high:
                    entry = or_high
                    vol_ratio = bar["volume"] / or_avg_volume
                    entry_time = bar["time"]

                    # Exit simulation on all bars after entry
                    future = mkt[mkt["timestamp"] >= bar["timestamp"]]
                    exit_price, reason, peak, exit_time = \
                        exit_trailing_detailed(future, entry, or_range)

                    pnl_pct = (exit_price - entry) / entry * 100 - cost_pct
                    peak_excursion = (peak - entry) / entry * 100

                    time_to_exit = None
                    if exit_time and entry_time:
                        t0 = pd.Timestamp(f"2000-01-01 {entry_time}")
                        t1 = pd.Timestamp(f"2000-01-01 {exit_time}")
                        time_to_exit = (t1 - t0).total_seconds() / 60

                    trades.append({
                        "date": day,
                        "ticker": ticker,
                        "entry": entry,
                        "exit": exit_price,
                        "pnl_pct": pnl_pct,
                        "reason": reason,
                        "vol_ratio": vol_ratio,
                        "or_range_pct": or_range / or_mid * 100,
                        "peak_excursion_pct": peak_excursion,
                        "entry_time": entry_time,
                        "exit_time": exit_time,
                        "time_to_exit_mins": time_to_exit,
                    })
                    break  # one trade per ticker per day

    all_trades = pd.DataFrame(trades)
    if all_trades.empty:
        return all_trades, all_trades

    # Best per day: top 1 by vol_ratio (matching live bot's MAX_TRADES_PER_DAY=1)
    best_per_day = (all_trades
                    .sort_values("vol_ratio", ascending=False)
                    .groupby("date").head(1)
                    .sort_values("date")
                    .reset_index(drop=True))

    print(f"  ORB trades: {len(all_trades)} total, "
          f"{len(best_per_day)} best-per-day, "
          f"{all_trades['date'].nunique()} trading days")
    return all_trades, best_per_day


# =======================================================================
# Analysis
# =======================================================================

def compute_stats(trades_df):
    """Compute summary stats for a set of trades."""
    if trades_df.empty or len(trades_df) == 0:
        return {"n": 0, "win_rate": 0, "pf": 0, "total_pnl": 0,
                "avg_pnl": 0, "median_pnl": 0, "max_dd": 0,
                "stop_rate": 0, "trail_rate": 0, "eod_rate": 0,
                "avg_peak_excursion": 0, "avg_time_to_exit": 0}

    n = len(trades_df)
    wins = trades_df[trades_df["pnl_pct"] > 0]
    losses = trades_df[trades_df["pnl_pct"] <= 0]
    gw = wins["pnl_pct"].sum() if len(wins) else 0
    gl = abs(losses["pnl_pct"].sum()) if len(losses) else 0.001
    cum = trades_df["pnl_pct"].cumsum()

    return {
        "n": n,
        "win_rate": len(wins) / n * 100,
        "pf": gw / gl if gl > 0 else 0,
        "total_pnl": trades_df["pnl_pct"].sum(),
        "avg_pnl": trades_df["pnl_pct"].mean(),
        "median_pnl": trades_df["pnl_pct"].median(),
        "max_dd": (cum - cum.cummax()).min(),
        "stop_rate": (trades_df["reason"] == "stop").sum() / n * 100,
        "trail_rate": (trades_df["reason"] == "trail").sum() / n * 100,
        "eod_rate": (trades_df["reason"] == "eod").sum() / n * 100,
        "avg_peak_excursion": trades_df["peak_excursion_pct"].mean(),
        "avg_time_to_exit": trades_df["time_to_exit_mins"].dropna().mean(),
    }


def fmt_stats_row(label, s, width=16):
    """Format a single stats dict as a row."""
    if s["n"] == 0:
        return f"  {label:<{width}}  {'(no trades)':>10}"
    return (f"  {label:<{width}}  {s['n']:>4}  {s['win_rate']:>6.1f}%  "
            f"{s['pf']:>5.2f}  {s['avg_pnl']:>+7.3f}%  {s['median_pnl']:>+7.3f}%  "
            f"{s['stop_rate']:>5.1f}%  {s['trail_rate']:>5.1f}%  "
            f"{s['avg_peak_excursion']:>6.2f}%  {s['max_dd']:>7.2f}%")


def analyze_and_report(best_trades, all_trades, context):
    """Run all analyses and print results."""
    lines = []

    def out(s=""):
        lines.append(s)
        print(s)

    # Merge trades with context
    best = best_trades.merge(context[["date", "spy_3d_return", "gap_pct",
                                      "consec_down", "vix", "vix_change_pct",
                                      "relief_bounce"]],
                             on="date", how="left")
    all_t = all_trades.merge(context[["date", "spy_3d_return", "gap_pct",
                                      "consec_down", "vix", "vix_change_pct",
                                      "relief_bounce"]],
                             on="date", how="left")

    n_days = best["date"].nunique()
    date_min = best["date"].min()
    date_max = best["date"].max()
    relief_days = best[best["relief_bounce"] == True]["date"].unique()

    out("=" * 78)
    out("  ORB PERFORMANCE vs MARKET CONTEXT")
    out(f"  {OR_MINUTES}-min OR | {TRAIL_MULT}x trail | {STOP_MULT}x stop | "
        f"vol confirmed | long-only")
    out(f"  1 trade/day (best vol_ratio) | ${POSITION_SIZE_USD:,.0f} | "
        f"${COMMISSION_RT_USD:.2f} RT cost")
    out(f"  {len(all_trades['ticker'].unique())} tickers | "
        f"{n_days} trading days | {date_min} to {date_max}")
    out("=" * 78)

    out()
    out("  DATA SUMMARY")
    out("  " + "-" * 40)
    out(f"  Trading days:       {n_days}")
    out(f"  Relief bounce days: {len(relief_days)}"
        f"  {list(relief_days) if len(relief_days) <= 10 else ''}")
    out(f"  Non-relief days:    {n_days - len(relief_days)}")
    out(f"  Total trade candidates: {len(all_trades)}")
    out(f"  Best-per-day trades:    {len(best_trades)}")
    out()
    if n_days < 30:
        out("  CAVEAT: Small sample. Results are directional indicators only.")
    out()

    # -- A. Relief Bounce vs Normal ------------------------------------
    header = (f"  {'':16}  {'N':>4}  {'Win%':>7}  {'PF':>5}  {'Avg':>8}  "
              f"{'Med':>8}  {'Stop%':>6}  {'Trail%':>7}  "
              f"{'PkExc':>7}  {'MaxDD':>8}")
    out("  -- A. RELIEF BOUNCE vs NORMAL (best 1/day) " + "-" * 33)
    out(header)
    out("  " + "-" * 76)

    normal = best[best["relief_bounce"] != True]
    relief = best[best["relief_bounce"] == True]
    s_normal = compute_stats(normal)
    s_relief = compute_stats(relief)
    out(fmt_stats_row("Normal days", s_normal))
    out(fmt_stats_row("Relief bounce", s_relief))
    out()

    # -- B. VIX Level Buckets ------------------------------------------
    out("  -- B. VIX LEVEL BUCKETS (best 1/day) " + "-" * 39)
    out(header)
    out("  " + "-" * 76)

    vix_bins = [0, 15, 20, 25, 100]
    vix_labels = ["< 15", "15-20", "20-25", "25+"]
    best["vix_bucket"] = pd.cut(best["vix"], bins=vix_bins, labels=vix_labels,
                                right=False)
    for label in vix_labels:
        subset = best[best["vix_bucket"] == label]
        s = compute_stats(subset)
        out(fmt_stats_row(f"VIX {label}", s))
    out()

    # -- C. SPY Prior 3-Day Return Quartiles ---------------------------
    out("  -- C. SPY PRIOR 3-DAY RETURN QUARTILES (best 1/day) " + "-" * 24)
    out(header)
    out("  " + "-" * 76)

    valid_3d = best.dropna(subset=["spy_3d_return"])
    if len(valid_3d) >= 4:
        try:
            valid_3d = valid_3d.copy()
            valid_3d["spy_3d_q"] = pd.qcut(valid_3d["spy_3d_return"], 4,
                                           labels=["Q1 (worst)", "Q2", "Q3",
                                                   "Q4 (best)"],
                                           duplicates="drop")
            for q in ["Q1 (worst)", "Q2", "Q3", "Q4 (best)"]:
                subset = valid_3d[valid_3d["spy_3d_q"] == q]
                s = compute_stats(subset)
                rng = subset["spy_3d_return"]
                desc = f"({rng.min():+.1f}% to {rng.max():+.1f}%)" if len(rng) else ""
                out(fmt_stats_row(f"{q} {desc}", s, width=30))
        except Exception as e:
            out(f"  (Could not compute quartiles: {e})")
    else:
        out("  (Not enough data for quartile analysis)")
    out()

    # -- D. Continuous Correlation -------------------------------------
    out("  -- D. CORRELATION: SPY 3d return vs ORB daily P&L " + "-" * 26)
    daily_pnl = best.groupby("date")["pnl_pct"].mean()
    daily_ctx = context.set_index("date")["spy_3d_return"]
    merged = pd.DataFrame({"orb_pnl": daily_pnl, "spy_3d": daily_ctx}).dropna()

    if len(merged) >= 5:
        from scipy.stats import pearsonr, spearmanr
        r, p = pearsonr(merged["spy_3d"], merged["orb_pnl"])
        rs, ps = spearmanr(merged["spy_3d"], merged["orb_pnl"])
        out(f"  Pearson  r = {r:+.3f}, p = {p:.3f}  (n={len(merged)})")
        out(f"  Spearman r = {rs:+.3f}, p = {ps:.3f}")
        if p < 0.05:
            direction = "positive" if r > 0 else "negative"
            out(f"  -> Significant {direction} relationship (p<0.05)")
        elif p < 0.20:
            direction = "positive" if r > 0 else "negative"
            out(f"  -> Weak {direction} trend (p<0.20) -- suggestive but not conclusive")
        else:
            out(f"  -> No significant relationship detected (p={p:.2f})")
    else:
        out("  (Not enough data)")
    out()

    # -- E. SPY Consecutive Down Days ----------------------------------
    out("  -- E. SPY CONSECUTIVE DOWN DAYS vs ORB (best 1/day) " + "-" * 24)
    out(header)
    out("  " + "-" * 76)

    for n_down in sorted(best["consec_down"].dropna().unique()):
        n_down = int(n_down)
        subset = best[best["consec_down"] == n_down]
        s = compute_stats(subset)
        out(fmt_stats_row(f"{n_down} down days", s))
    out()

    # -- F. VIX Change Buckets -----------------------------------------
    out("  -- F. VIX 1-DAY CHANGE BUCKETS (best 1/day) " + "-" * 32)
    out(header)
    out("  " + "-" * 76)

    vix_chg_bins = [-100, -10, -5, 0, 5, 10, 100]
    vix_chg_labels = ["< -10%", "-10 to -5%", "-5 to 0%",
                      "0 to +5%", "+5 to +10%", "> +10%"]
    best["vix_chg_bucket"] = pd.cut(best["vix_change_pct"],
                                    bins=vix_chg_bins, labels=vix_chg_labels,
                                    right=False)
    for label in vix_chg_labels:
        subset = best[best["vix_chg_bucket"] == label]
        if len(subset) > 0:
            s = compute_stats(subset)
            out(fmt_stats_row(label, s))
    out()

    # -- G. All Trades (not filtered) Relief vs Normal -----------------
    out("  -- G. ALL TRADES (not best-per-day): Relief vs Normal " + "-" * 22)
    out(header)
    out("  " + "-" * 76)

    all_normal = all_t[all_t["relief_bounce"] != True]
    all_relief = all_t[all_t["relief_bounce"] == True]
    out(fmt_stats_row("Normal (all)", compute_stats(all_normal)))
    out(fmt_stats_row("Relief (all)", compute_stats(all_relief)))
    out()

    # -- H. Per-Day Detail ---------------------------------------------
    out("  -- H. PER-DAY DETAIL (best trade) " + "-" * 42)
    out(f"  {'Date':>12}  {'Ticker':>6}  {'P&L':>8}  {'Reason':>6}  "
        f"{'VIX':>5}  {'SPY3d':>7}  {'Gap':>6}  {'Down':>4}  {'Relief':>7}")
    out("  " + "-" * 76)

    for _, row in best.sort_values("date").iterrows():
        rb = "YES" if row.get("relief_bounce") else ""
        out(f"  {row['date']}  {row['ticker']:>6}  {row['pnl_pct']:>+7.2f}%  "
            f"{row['reason']:>6}  {row.get('vix', 0):>5.1f}  "
            f"{row.get('spy_3d_return', 0):>+6.1f}%  "
            f"{row.get('gap_pct', 0):>+5.1f}%  "
            f"{int(row.get('consec_down', 0)):>4}  {rb:>7}")
    out()

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "relief_bounce_analysis.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Results saved to {out_path}")


# =======================================================================
# Main
# =======================================================================

if __name__ == "__main__":
    print()
    print("  Loading intraday data...")
    all_ticker_data = load_all_intraday()

    print("  Building market context...")
    context = build_market_context()

    print("  Running ORB backtest...")
    all_trades, best_trades = run_all_orb_trades(all_ticker_data)

    if best_trades.empty:
        print("  No trades generated. Check data/intraday_cache.")
        sys.exit(1)

    print()
    analyze_and_report(best_trades, all_trades, context)
