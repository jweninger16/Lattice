"""
research/gap_scanner_backtest.py
----------------------------------
RESEARCH — Market-wide gap scanner day trading strategy.

Instead of trading one fixed ticker, this scans the entire market each
morning for stocks that are "in play" — gapping on unusual volume with
momentum conditions that historically lead to intraday follow-through.

The core insight: every day, 5-10 stocks gap significantly on volume.
These are the stocks day traders should focus on. The edge comes from
PICKING THE RIGHT STOCK on the right day, not from trading a pattern
on a random ticker.

Strategy candidates tested:
  1. Gap-and-Go (continuation): Buy stocks gapping up 2-8% on 2x+ volume
  2. Gap reversal fade (long): Buy stocks that gap DOWN 2-5%, bounce off open
  3. Relative strength breakout: Buy strongest gapper that holds above VWAP
  4. Earnings gap continuation: Buy stocks gapping on earnings with volume
  5. Sector momentum: Buy gappers in the day's leading sector

Risk framework:
  - ATR-based position sizing (risk $X per trade, stop at 1 ATR)
  - Max 1-2 trades per day (cash account, T+1 settlement)
  - Only trade first 2 hours (9:30-11:30) — that's where momentum lives
  - Hard stop at 1x ATR, target at 2-3x ATR (asymmetric R:R)

Data: Uses daily OHLC to approximate intraday behavior across 500+ stocks
over 2+ years. Follow-through is measured by open→high (long) and
open→close (did it hold?). This gives hundreds of trade samples.

Usage:
    python research/gap_scanner_backtest.py
"""

import sys
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger

sys.path.insert(0, ".")

CACHE_DIR = Path("data/gap_scanner_cache")


# ═══════════════════════════════════════════════════════════════════════
# Data — download daily OHLCV for a broad universe
# ═══════════════════════════════════════════════════════════════════════

def get_universe():
    """Returns a broad tradeable universe."""
    try:
        from data.universe import FALLBACK_TICKERS, ETF_TICKERS
        tickers = [t for t in FALLBACK_TICKERS if t not in ETF_TICKERS]
    except ImportError:
        tickers = [
            "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AMD","NFLX",
            "CRM","AVGO","ORCL","QCOM","AMAT","MU","INTC","UBER","SQ","SHOP",
            "COIN","MARA","RIOT","PLTR","SOFI","DKNG","SNAP","PINS","RBLX",
            "NET","CRWD","ZS","DDOG","MDB","SNOW","ABNB","DASH","ROKU",
            "JPM","BAC","GS","MS","C","WFC","V","MA","AXP","PYPL","SCHW",
            "XOM","CVX","COP","OXY","SLB","HAL","MPC","VLO","PSX",
            "UNH","JNJ","PFE","ABBV","MRK","LLY","BMY","AMGN","GILD",
            "CAT","DE","GE","HON","BA","RTX","LMT","GD","NOC",
            "HD","LOW","TGT","COST","WMT","TJX","ROST","DG","DLTR",
            "DIS","CMCSA","T","TMUS","VZ","CHTR",
        ]
    return tickers


def download_daily_data(period="2y"):
    """Download daily OHLCV for the full universe."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "daily_universe.parquet"

    # Use cache if fresh (less than 12 hours old)
    if cache_path.exists():
        age = datetime.now().timestamp() - cache_path.stat().st_mtime
        if age < 43200:  # 12 hours
            print(f"  Using cached data ({age/3600:.1f}h old)")
            return pd.read_parquet(cache_path)

    tickers = get_universe()
    print(f"  Downloading {len(tickers)} tickers ({period})...")

    raw = yf.download(
        tickers, period=period, auto_adjust=True,
        progress=True, threads=True,
    )

    if raw.empty:
        print("  ERROR: No data downloaded")
        return pd.DataFrame()

    # Reshape to long format
    raw.columns.names = ["Field", "Ticker"]
    df = raw.stack(level="Ticker", future_stack=True).reset_index()
    df.columns = [c.lower() if isinstance(c, str) else c
                  for c in df.columns]
    if "date" not in df.columns:
        df = df.rename(columns={"level_0": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close", "volume"])
    df = df[df["close"] > 0].copy()

    # Rename if needed
    col_map = {}
    for c in df.columns:
        cl = c.lower() if isinstance(c, str) else c
        col_map[c] = cl
    df = df.rename(columns=col_map)

    if "ticker" not in df.columns:
        # Check for level_1
        if "level_1" in df.columns:
            df = df.rename(columns={"level_1": "ticker"})

    print(f"  Downloaded: {df['ticker'].nunique()} tickers, "
          f"{df['date'].nunique()} days, {len(df):,} rows")

    df.to_parquet(cache_path, index=False)
    return df


# ═══════════════════════════════════════════════════════════════════════
# Feature engineering — compute gap, volume, momentum for each stock-day
# ═══════════════════════════════════════════════════════════════════════

def build_scanner_features(df):
    """
    For each stock-day, compute the features a gap scanner would use
    to decide whether to trade it.
    """
    df = df.sort_values(["ticker", "date"]).copy()

    # Per-ticker rolling features
    out = []
    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date").copy()

        # Previous close
        g["prev_close"] = g["close"].shift(1)
        g["prev_volume"] = g["volume"].shift(1)

        # Gap
        g["gap_pct"] = (g["open"] - g["prev_close"]) / g["prev_close"] * 100

        # Volume ratio (today's volume vs 20-day average)
        g["vol_sma20"] = g["volume"].rolling(20).mean()
        g["rvol"] = g["volume"] / g["vol_sma20"]

        # ATR (14-day)
        tr = pd.DataFrame({
            "hl": g["high"] - g["low"],
            "hc": (g["high"] - g["close"].shift(1)).abs(),
            "lc": (g["low"] - g["close"].shift(1)).abs(),
        })
        g["tr"] = tr.max(axis=1)
        g["atr_14"] = g["tr"].rolling(14).mean()
        g["atr_pct"] = g["atr_14"] / g["close"] * 100

        # Intraday follow-through metrics (what we're trying to predict)
        g["open_to_high_pct"] = (g["high"] - g["open"]) / g["open"] * 100
        g["open_to_low_pct"] = (g["open"] - g["low"]) / g["open"] * 100
        g["open_to_close_pct"] = (g["close"] - g["open"]) / g["open"] * 100
        g["intraday_range_pct"] = (g["high"] - g["low"]) / g["open"] * 100

        # Prior momentum (5-day return going into today)
        g["mom_5d"] = g["close"].pct_change(5) * 100

        # Prior momentum (20-day return)
        g["mom_20d"] = g["close"].pct_change(20) * 100

        # 50-day SMA trend
        g["sma_50"] = g["close"].rolling(50).mean()
        g["above_sma50"] = (g["close"] > g["sma_50"]).astype(int)

        # Recent high (is this stock near 20-day high?)
        g["high_20d"] = g["high"].rolling(20).max()
        g["near_20d_high"] = (g["close"] / g["high_20d"]) * 100

        # Dollar volume (liquidity)
        g["dollar_volume"] = g["close"] * g["volume"]

        out.append(g)

    result = pd.concat(out, ignore_index=True)
    result = result.dropna(subset=["gap_pct", "atr_14", "vol_sma20"])

    return result


# ═══════════════════════════════════════════════════════════════════════
# Scanner: find the best candidates each day
# ═══════════════════════════════════════════════════════════════════════

def scan_gap_candidates(day_df, config):
    """
    Applies the morning scanner filters to find tradeable gap stocks.
    Returns ranked candidates for the day.
    """
    min_gap = config.get("min_gap_pct", 2.0)
    max_gap = config.get("max_gap_pct", 10.0)
    min_price = config.get("min_price", 10.0)
    max_price = config.get("max_price", 500.0)
    min_rvol = config.get("min_rvol", 1.5)
    min_dollar_vol = config.get("min_dollar_vol", 5_000_000)
    direction = config.get("direction", "long")  # long, short, both
    min_atr_pct = config.get("min_atr_pct", 1.0)

    c = day_df.copy()

    # Price filter
    c = c[(c["prev_close"] >= min_price) & (c["prev_close"] <= max_price)]

    # Liquidity filter
    c = c[c["dollar_volume"] >= min_dollar_vol]

    # ATR filter (stock must move enough to be worth trading)
    c = c[c["atr_pct"] >= min_atr_pct]

    # Gap filter
    if direction == "long":
        c = c[(c["gap_pct"] >= min_gap) & (c["gap_pct"] <= max_gap)]
    elif direction == "short":
        c = c[(c["gap_pct"] <= -min_gap) & (c["gap_pct"] >= -max_gap)]
    else:  # both
        c = c[c["gap_pct"].abs().between(min_gap, max_gap)]

    # Relative volume filter
    c = c[c["rvol"] >= min_rvol]

    if c.empty:
        return c

    # Score candidates: bigger gap + higher rvol + momentum alignment = better
    if direction == "long":
        c["score"] = (
            c["gap_pct"] * 0.4 +          # Bigger gap = more momentum
            c["rvol"].clip(upper=5) * 0.3 + # Higher volume = more conviction
            c["mom_5d"].clip(-5, 10) * 0.15 + # Prior momentum alignment
            c["above_sma50"] * 0.15 * 5    # Trend alignment
        )
    elif direction == "short":
        c["score"] = (
            c["gap_pct"].abs() * 0.4 +
            c["rvol"].clip(upper=5) * 0.3 +
            (-c["mom_5d"]).clip(-5, 10) * 0.15 +
            (1 - c["above_sma50"]) * 0.15 * 5
        )
    else:
        c["score"] = (
            c["gap_pct"].abs() * 0.4 +
            c["rvol"].clip(upper=5) * 0.3 +
            c["atr_pct"] * 0.3
        )

    c = c.sort_values("score", ascending=False)
    return c


# ═══════════════════════════════════════════════════════════════════════
# Backtest engine
# ═══════════════════════════════════════════════════════════════════════

def simulate_day_trade(row, config):
    """
    Simulates a day trade using daily OHLC data.

    For a LONG gap-and-go:
      - Entry: at the open price (we're buying the gap)
      - Stop: open - (ATR * stop_atr_mult)
      - Target: open + (ATR * target_atr_mult)
      - Check: did intraday low hit stop? did intraday high hit target?
      - If neither: close at end of day (use close price)

    Returns trade result dict.
    """
    direction = config.get("trade_direction", "long")
    stop_mult = config.get("stop_atr_mult", 1.0)
    target_mult = config.get("target_atr_mult", 2.5)
    use_gap_stop = config.get("use_gap_stop", False)

    entry = row["open"]
    atr = row["atr_14"]

    if direction == "long":
        stop = entry - atr * stop_mult
        target = entry + atr * target_mult

        if use_gap_stop:
            # Stop at previous close (if gap fails completely)
            stop = max(stop, row["prev_close"] * 0.998)

        # Check stop first (conservative: assume stop checked before target)
        if row["low"] <= stop:
            exit_price = stop
            reason = "stop"
        elif row["high"] >= target:
            exit_price = target
            reason = "target"
        else:
            exit_price = row["close"]
            reason = "eod"

        pnl_pct = (exit_price - entry) / entry * 100

    else:  # short
        stop = entry + atr * stop_mult
        target = entry - atr * target_mult

        if row["high"] >= stop:
            exit_price = stop
            reason = "stop"
        elif row["low"] <= target:
            exit_price = target
            reason = "target"
        else:
            exit_price = row["close"]
            reason = "eod"

        pnl_pct = (entry - exit_price) / entry * 100

    return {
        "date": row["date"],
        "ticker": row["ticker"],
        "direction": direction,
        "entry": entry,
        "exit": exit_price,
        "stop": stop,
        "target": target,
        "pnl_pct": pnl_pct,
        "reason": reason,
        "gap_pct": row["gap_pct"],
        "rvol": row["rvol"],
        "atr_pct": row["atr_pct"],
        "score": row.get("score", 0),
        "mom_5d": row.get("mom_5d", 0),
        "above_sma50": row.get("above_sma50", 0),
    }


def run_backtest(df, scanner_config, trade_config, max_trades_per_day=1):
    """
    Full backtest: for each day, scan for candidates, pick the top N,
    simulate day trades on each.
    """
    trades = []
    days_scanned = 0
    days_with_candidates = 0

    for day, day_df in df.groupby("date"):
        days_scanned += 1
        candidates = scan_gap_candidates(day_df, scanner_config)

        if candidates.empty:
            continue

        days_with_candidates += 1

        # Take top N candidates
        top = candidates.head(max_trades_per_day)

        for _, row in top.iterrows():
            trade = simulate_day_trade(row, trade_config)
            trades.append(trade)

    result = pd.DataFrame(trades)
    result.attrs["days_scanned"] = days_scanned
    result.attrs["days_with_candidates"] = days_with_candidates
    return result


# ═══════════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════════

def calc_stats(trades):
    """Comprehensive stats for a set of trades."""
    if trades.empty:
        return None
    t = trades
    n = len(t)
    wins = t[t["pnl_pct"] > 0]
    losses = t[t["pnl_pct"] <= 0]
    wr = len(wins) / n * 100
    avg_win = wins["pnl_pct"].mean() if len(wins) > 0 else 0
    avg_loss = losses["pnl_pct"].mean() if len(losses) > 0 else 0
    gw = wins["pnl_pct"].sum() if len(wins) > 0 else 0
    gl = abs(losses["pnl_pct"].sum()) if len(losses) > 0 else 0.001
    pf = gw / gl
    total = t["pnl_pct"].sum()
    avg = t["pnl_pct"].mean()
    cum = t["pnl_pct"].cumsum()
    max_dd = (cum - cum.cummax()).min()

    # Per-day stats
    daily = t.groupby("date")["pnl_pct"].sum()
    daily_avg = daily.mean()
    daily_std = daily.std()
    sharpe_approx = (daily_avg / daily_std * np.sqrt(252)) if daily_std > 0 else 0

    # Consecutive losses
    is_loss = (t["pnl_pct"] <= 0).astype(int)
    max_consec_loss = 0
    streak = 0
    for val in is_loss:
        if val:
            streak += 1
            max_consec_loss = max(max_consec_loss, streak)
        else:
            streak = 0

    return {
        "n": n, "win_rate": wr, "pf": pf, "total": total,
        "avg": avg, "avg_win": avg_win, "avg_loss": avg_loss,
        "max_dd": max_dd, "max_consec_loss": max_consec_loss,
        "sharpe": sharpe_approx, "daily_avg": daily_avg,
    }


def print_strategy_table(rows, title):
    print(f"\n  {'='*95}")
    print(f"  {title}")
    print(f"  {'='*95}")
    print(f"  {'Strategy':<45} {'N':>5} {'Win%':>6} {'PF':>6} "
          f"{'Total':>8} {'MaxDD':>8} {'Avg':>8} {'Sharpe':>7}")
    print(f"  {'-'*95}")
    for label, stats in rows:
        if stats is None:
            print(f"  {label:<45} {'—':>5}")
        else:
            print(f"  {label:<45} {stats['n']:>5} {stats['win_rate']:>5.1f}% "
                  f"{stats['pf']:>6.2f} {stats['total']:>+7.1f}% "
                  f"{stats['max_dd']:>+7.1f}% {stats['avg']:>+7.3f}% "
                  f"{stats['sharpe']:>6.2f}")


def analyze_trades_deep(trades, label):
    """Detailed breakdown of trade results."""
    if trades.empty:
        return

    stats = calc_stats(trades)
    if stats is None:
        return

    print(f"\n  ┌─ {label}")
    print(f"  │  {stats['n']} trades | {stats['win_rate']:.1f}% win | "
          f"PF {stats['pf']:.2f} | Total {stats['total']:+.1f}%")
    print(f"  │  Avg win: {stats['avg_win']:+.2f}% | "
          f"Avg loss: {stats['avg_loss']:+.2f}% | "
          f"R:R {abs(stats['avg_win']/stats['avg_loss']) if stats['avg_loss'] != 0 else 0:.2f}")
    print(f"  │  Max DD: {stats['max_dd']:+.1f}% | "
          f"Max consec loss: {stats['max_consec_loss']} | "
          f"Sharpe: {stats['sharpe']:.2f}")

    # Exit reasons
    rc = trades["reason"].value_counts()
    print(f"  │  Exits: {', '.join(f'{r}={c}' for r, c in rc.items())}")

    # By gap size
    trades_c = trades.copy()
    trades_c["gap_bucket"] = pd.cut(
        trades_c["gap_pct"].abs(),
        bins=[0, 3, 5, 8, 100],
        labels=["2-3%", "3-5%", "5-8%", "8%+"]
    )
    print(f"  │")
    print(f"  │  By gap size:")
    for bucket in ["2-3%", "3-5%", "5-8%", "8%+"]:
        bt = trades_c[trades_c["gap_bucket"] == bucket]
        if len(bt) >= 3:
            wr = (bt["pnl_pct"] > 0).mean() * 100
            print(f"  │    {bucket:<8} {len(bt):>4} trades, "
                  f"{wr:.0f}% win, {bt['pnl_pct'].mean():+.3f}% avg, "
                  f"{bt['pnl_pct'].sum():+.1f}% total")

    # By relative volume
    trades_c["rvol_bucket"] = pd.cut(
        trades_c["rvol"],
        bins=[0, 2, 3, 5, 100],
        labels=["1.5-2x", "2-3x", "3-5x", "5x+"]
    )
    print(f"  │")
    print(f"  │  By relative volume:")
    for bucket in ["1.5-2x", "2-3x", "3-5x", "5x+"]:
        bt = trades_c[trades_c["rvol_bucket"] == bucket]
        if len(bt) >= 3:
            wr = (bt["pnl_pct"] > 0).mean() * 100
            print(f"  │    {bucket:<8} {len(bt):>4} trades, "
                  f"{wr:.0f}% win, {bt['pnl_pct'].mean():+.3f}% avg")

    # Above/below 50 SMA
    if "above_sma50" in trades.columns:
        print(f"  │")
        print(f"  │  Trend alignment:")
        for val, lbl in [(1, "Above SMA50"), (0, "Below SMA50")]:
            bt = trades[trades["above_sma50"] == val]
            if len(bt) >= 3:
                wr = (bt["pnl_pct"] > 0).mean() * 100
                print(f"  │    {lbl:<15} {len(bt):>4} trades, "
                      f"{wr:.0f}% win, {bt['pnl_pct'].mean():+.3f}% avg")

    # Top performing tickers
    ticker_stats = trades.groupby("ticker")["pnl_pct"].agg(["count", "sum", "mean"])
    ticker_stats = ticker_stats[ticker_stats["count"] >= 3].sort_values("sum", ascending=False)
    if len(ticker_stats) > 0:
        print(f"  │")
        print(f"  │  Top tickers (3+ trades):")
        for tkr, row in ticker_stats.head(5).iterrows():
            print(f"  │    {tkr:<6} {int(row['count']):>3} trades, "
                  f"{row['mean']:+.3f}% avg, {row['sum']:+.1f}% total")

    # Day of week
    trades_c["dow"] = pd.to_datetime(trades_c["date"]).dt.dayofweek
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    print(f"  │")
    print(f"  │  Day of week:")
    for d in range(5):
        bt = trades_c[trades_c["dow"] == d]
        if len(bt) >= 3:
            wr = (bt["pnl_pct"] > 0).mean() * 100
            print(f"  │    {dow_names[d]}: {len(bt):>4} trades, "
                  f"{wr:.0f}% win, {bt['pnl_pct'].mean():+.3f}% avg")

    # Monthly performance
    trades_c["month"] = pd.to_datetime(trades_c["date"]).dt.to_period("M")
    monthly = trades_c.groupby("month")["pnl_pct"].agg(["count", "sum"])
    print(f"  │")
    print(f"  │  Monthly P&L:")
    for month, row in monthly.iterrows():
        bar = "+" * int(max(0, row["sum"])) + "-" * int(max(0, -row["sum"]))
        print(f"  │    {str(month):<8} {int(row['count']):>3} trades  "
              f"{row['sum']:>+6.1f}%  {bar}")

    print(f"  └{'─'*60}")


# ═══════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════

def plot_results(results_dict, title="Gap Scanner Day Trading Research"):
    """Plot equity curves for multiple strategies."""
    valid = {k: v for k, v in results_dict.items()
             if not v.empty and len(v) >= 10}
    if not valid:
        print("  No plottable results")
        return

    n = min(len(valid), 12)
    rows = (n + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(18, rows * 4.5), facecolor="#0d1117")
    fig.suptitle(title, color="white", fontsize=14, fontweight="bold")

    all_axes = axes.flatten() if rows > 1 else axes
    for ax in all_axes:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="gray", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#333")

    colors = ["#448aff", "#00e676", "#ff9800", "#e040fb", "#00bcd4",
              "#ff5252", "#76ff03", "#ffd740", "#40c4ff", "#ea80fc",
              "#ff8a65", "#80cbc4"]
    for i, (label, trades) in enumerate(valid.items()):
        if i >= rows * 3:
            break
        ax = all_axes[i]
        cum = trades["pnl_pct"].cumsum()
        c = colors[i % len(colors)]
        ax.plot(range(len(cum)), cum, color=c, linewidth=1.5)
        ax.fill_between(range(len(cum)), 0, cum, alpha=0.1, color=c)
        ax.axhline(0, color="#555", linewidth=0.5)
        stats = calc_stats(trades)
        ax.set_title(f"{label}\n{stats['n']} trades, {stats['win_rate']:.0f}% win, "
                     f"PF {stats['pf']:.2f}, {stats['total']:+.1f}%",
                     color="white", fontsize=8)
        ax.set_xlabel("Trade #", color="gray", fontsize=7)
        ax.set_ylabel("Cum P&L (%)", color="gray", fontsize=7)

    for j in range(i + 1, rows * 3):
        all_axes[j].set_visible(False)

    plt.tight_layout()
    save_path = "research/gap_scanner_results.png"
    Path("research").mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"\n  Chart saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main research runner
# ═══════════════════════════════════════════════════════════════════════

def run_research():
    print("=" * 95)
    print("  GAP SCANNER DAY TRADING RESEARCH")
    print("  Scanning entire market for the best day trade each morning")
    print("=" * 95)

    # ── Download data ────────────────────────────────────────────────
    df = download_daily_data("2y")
    if df.empty:
        return

    # ── Build features ───────────────────────────────────────────────
    print("\n  Building scanner features...")
    df = build_scanner_features(df)
    n_tickers = df["ticker"].nunique()
    n_days = df["date"].nunique()
    print(f"  Universe: {n_tickers} tickers × {n_days} days = "
          f"{len(df):,} stock-days")

    # Count gap days across the universe
    gap_days = df[df["gap_pct"].abs() >= 2.0]
    print(f"  Stock-days with ≥2% gap: {len(gap_days):,} "
          f"({len(gap_days)/len(df)*100:.1f}%)")
    gap_with_vol = gap_days[gap_days["rvol"] >= 1.5]
    print(f"  ... with 1.5x+ relative volume: {len(gap_with_vol):,}")

    # ── Strategy grid ────────────────────────────────────────────────
    strategies = {}

    # ── 1. Gap-and-Go Long: buy stocks gapping up on volume ─────────
    configs = [
        # (label, scanner_config, trade_config, max_per_day)
        (
            "Gap-up 2-8%, 1.5x vol, 2.5:1 R:R",
            {"min_gap_pct": 2.0, "max_gap_pct": 8.0, "min_rvol": 1.5,
             "direction": "long", "min_price": 10, "max_price": 500,
             "min_dollar_vol": 5_000_000, "min_atr_pct": 1.0},
            {"trade_direction": "long", "stop_atr_mult": 1.0,
             "target_atr_mult": 2.5},
            1,
        ),
        (
            "Gap-up 2-8%, 2x vol, 2:1 R:R",
            {"min_gap_pct": 2.0, "max_gap_pct": 8.0, "min_rvol": 2.0,
             "direction": "long", "min_price": 10, "max_price": 500,
             "min_dollar_vol": 5_000_000, "min_atr_pct": 1.0},
            {"trade_direction": "long", "stop_atr_mult": 1.0,
             "target_atr_mult": 2.0},
            1,
        ),
        (
            "Gap-up 3-8%, 2x vol, 3:1 R:R",
            {"min_gap_pct": 3.0, "max_gap_pct": 8.0, "min_rvol": 2.0,
             "direction": "long", "min_price": 10, "max_price": 500,
             "min_dollar_vol": 5_000_000, "min_atr_pct": 1.5},
            {"trade_direction": "long", "stop_atr_mult": 1.0,
             "target_atr_mult": 3.0},
            1,
        ),
        (
            "Gap-up 2-5%, 1.5x vol, trend aligned",
            {"min_gap_pct": 2.0, "max_gap_pct": 5.0, "min_rvol": 1.5,
             "direction": "long", "min_price": 10, "max_price": 500,
             "min_dollar_vol": 5_000_000, "min_atr_pct": 1.0},
            {"trade_direction": "long", "stop_atr_mult": 1.0,
             "target_atr_mult": 2.0},
            1,
        ),
        # Tighter stop
        (
            "Gap-up 2-8%, 2x vol, tight stop (0.5 ATR)",
            {"min_gap_pct": 2.0, "max_gap_pct": 8.0, "min_rvol": 2.0,
             "direction": "long", "min_price": 10, "max_price": 500,
             "min_dollar_vol": 5_000_000, "min_atr_pct": 1.0},
            {"trade_direction": "long", "stop_atr_mult": 0.5,
             "target_atr_mult": 2.0},
            1,
        ),
        # Use previous close as stop (gap stop)
        (
            "Gap-up 2-8%, 2x vol, gap stop",
            {"min_gap_pct": 2.0, "max_gap_pct": 8.0, "min_rvol": 2.0,
             "direction": "long", "min_price": 10, "max_price": 500,
             "min_dollar_vol": 5_000_000, "min_atr_pct": 1.0},
            {"trade_direction": "long", "stop_atr_mult": 1.0,
             "target_atr_mult": 2.5, "use_gap_stop": True},
            1,
        ),
    ]

    # ── 2. Multi-trade configs (take top 2 candidates) ──────────────
    configs.append((
        "Gap-up 2-8%, 2x vol, TOP 2 trades/day",
        {"min_gap_pct": 2.0, "max_gap_pct": 8.0, "min_rvol": 2.0,
         "direction": "long", "min_price": 10, "max_price": 500,
         "min_dollar_vol": 5_000_000, "min_atr_pct": 1.0},
        {"trade_direction": "long", "stop_atr_mult": 1.0,
         "target_atr_mult": 2.5},
        2,
    ))

    # ── 3. Gap-down fade (long): buy oversold gap-downs ─────────────
    configs.append((
        "Gap-DOWN 2-5% fade (buy the dip)",
        {"min_gap_pct": 2.0, "max_gap_pct": 5.0, "min_rvol": 1.5,
         "direction": "short", "min_price": 10, "max_price": 500,
         "min_dollar_vol": 5_000_000, "min_atr_pct": 1.0},
        {"trade_direction": "long", "stop_atr_mult": 1.5,
         "target_atr_mult": 2.0},
        1,
    ))

    # ── 4. Small gap, huge volume (stealth momentum) ────────────────
    configs.append((
        "Small gap 1-2%, 3x+ volume (stealth)",
        {"min_gap_pct": 1.0, "max_gap_pct": 2.0, "min_rvol": 3.0,
         "direction": "long", "min_price": 10, "max_price": 500,
         "min_dollar_vol": 5_000_000, "min_atr_pct": 1.0},
        {"trade_direction": "long", "stop_atr_mult": 1.0,
         "target_atr_mult": 2.0},
        1,
    ))

    # ═════════════════════════════════════════════════════════════════
    #  AGGRESSIVE STRATEGIES
    # ═════════════════════════════════════════════════════════════════

    # ── 5. Trade more often: lower gap threshold ─────────────────────
    configs.append((
        "AGG: Gap 1.5-8%, 1.5x vol, 2.5:1",
        {"min_gap_pct": 1.5, "max_gap_pct": 8.0, "min_rvol": 1.5,
         "direction": "long", "min_price": 10, "max_price": 500,
         "min_dollar_vol": 5_000_000, "min_atr_pct": 1.0},
        {"trade_direction": "long", "stop_atr_mult": 1.0,
         "target_atr_mult": 2.5, "use_gap_stop": True},
        1,
    ))

    # ── 6. Multi-trade: top 2 per day with gap stop ──────────────────
    configs.append((
        "AGG: TOP 2 trades/day, gap stop",
        {"min_gap_pct": 2.0, "max_gap_pct": 8.0, "min_rvol": 2.0,
         "direction": "long", "min_price": 10, "max_price": 500,
         "min_dollar_vol": 5_000_000, "min_atr_pct": 1.0},
        {"trade_direction": "long", "stop_atr_mult": 1.0,
         "target_atr_mult": 2.5, "use_gap_stop": True},
        2,
    ))

    # ── 7. Multi-trade: top 3 per day ────────────────────────────────
    configs.append((
        "AGG: TOP 3 trades/day, gap stop",
        {"min_gap_pct": 2.0, "max_gap_pct": 8.0, "min_rvol": 2.0,
         "direction": "long", "min_price": 10, "max_price": 500,
         "min_dollar_vol": 5_000_000, "min_atr_pct": 1.0},
        {"trade_direction": "long", "stop_atr_mult": 1.0,
         "target_atr_mult": 2.5, "use_gap_stop": True},
        3,
    ))

    # ── 8. Let winners run: 4x ATR target ────────────────────────────
    configs.append((
        "AGG: 4x ATR target (let winners run)",
        {"min_gap_pct": 2.0, "max_gap_pct": 8.0, "min_rvol": 2.0,
         "direction": "long", "min_price": 10, "max_price": 500,
         "min_dollar_vol": 5_000_000, "min_atr_pct": 1.0},
        {"trade_direction": "long", "stop_atr_mult": 1.0,
         "target_atr_mult": 4.0, "use_gap_stop": True},
        1,
    ))

    # ── 9. High conviction only: 3%+ gap, 3x+ vol, trend aligned ────
    configs.append((
        "AGG: High conviction (3%+, 3x vol, trend)",
        {"min_gap_pct": 3.0, "max_gap_pct": 10.0, "min_rvol": 3.0,
         "direction": "long", "min_price": 10, "max_price": 500,
         "min_dollar_vol": 5_000_000, "min_atr_pct": 1.5},
        {"trade_direction": "long", "stop_atr_mult": 1.0,
         "target_atr_mult": 3.0, "use_gap_stop": True},
        1,
    ))

    # ── 10. Combine: lower threshold + 2 trades/day ──────────────────
    configs.append((
        "AGG: Gap 1.5%+, TOP 2/day, gap stop",
        {"min_gap_pct": 1.5, "max_gap_pct": 8.0, "min_rvol": 1.5,
         "direction": "long", "min_price": 10, "max_price": 500,
         "min_dollar_vol": 5_000_000, "min_atr_pct": 1.0},
        {"trade_direction": "long", "stop_atr_mult": 1.0,
         "target_atr_mult": 2.5, "use_gap_stop": True},
        2,
    ))

    # ── 11. Max frequency: 1.5%+ gap, 1.5x vol, 3 trades/day ───────
    configs.append((
        "AGG: MAX FREQ 1.5%+, 3/day, gap stop",
        {"min_gap_pct": 1.5, "max_gap_pct": 8.0, "min_rvol": 1.5,
         "direction": "long", "min_price": 10, "max_price": 500,
         "min_dollar_vol": 5_000_000, "min_atr_pct": 1.0},
        {"trade_direction": "long", "stop_atr_mult": 1.0,
         "target_atr_mult": 2.5, "use_gap_stop": True},
        3,
    ))

    # ── 12. Volatile movers only: high ATR + gap ─────────────────────
    configs.append((
        "AGG: Volatile movers (ATR>2.5%, gap 2%+)",
        {"min_gap_pct": 2.0, "max_gap_pct": 10.0, "min_rvol": 1.5,
         "direction": "long", "min_price": 10, "max_price": 500,
         "min_dollar_vol": 5_000_000, "min_atr_pct": 2.5},
        {"trade_direction": "long", "stop_atr_mult": 1.0,
         "target_atr_mult": 3.0, "use_gap_stop": True},
        1,
    ))

    # ── 13. Tight stop aggressive: 0.5 ATR stop, 3x target ──────────
    configs.append((
        "AGG: Tight stop 0.5 ATR, 3x target",
        {"min_gap_pct": 2.0, "max_gap_pct": 8.0, "min_rvol": 2.0,
         "direction": "long", "min_price": 10, "max_price": 500,
         "min_dollar_vol": 5_000_000, "min_atr_pct": 1.0},
        {"trade_direction": "long", "stop_atr_mult": 0.5,
         "target_atr_mult": 3.0, "use_gap_stop": False},
        1,
    ))

    # ── Run all strategies ───────────────────────────────────────────
    table_rows = []
    all_results = {}

    for label, scanner_cfg, trade_cfg, max_per_day in configs:
        trades = run_backtest(df, scanner_cfg, trade_cfg, max_per_day)
        all_results[label] = trades
        stats = calc_stats(trades)
        table_rows.append((label, stats))

        days_scanned = trades.attrs.get("days_scanned", "?")
        days_traded = trades.attrs.get("days_with_candidates", "?")
        if stats:
            print(f"  ✓ {label}: {stats['n']} trades from "
                  f"{days_traded}/{days_scanned} days")

    # ── Results ──────────────────────────────────────────────────────
    print_strategy_table(table_rows, "Gap Scanner — Strategy Comparison")

    # ── Deep dive on top 3 by profit factor ──────────────────────────
    scored = [(l, calc_stats(t), t) for l, t in all_results.items()
              if calc_stats(t) and calc_stats(t)["n"] >= 20]
    scored.sort(key=lambda x: x[1]["pf"], reverse=True)

    print(f"\n  TOP STRATEGIES — Deep Dive:")
    for label, stats, trades in scored[:5]:
        analyze_trades_deep(trades, label)

    # ── Dollar returns estimate ──────────────────────────────────────
    if scored:
        print(f"\n  {'─'*80}")
        print(f"  DOLLAR RETURN ESTIMATES ($2,000 account)")
        print(f"  {'─'*80}")
        print(f"  {'Strategy':<45} {'Trades':>6} {'$/trade':>8} "
              f"{'$/month':>8} {'$/year':>9}")
        print(f"  {'-'*80}")

        for label, stats, trades in scored[:8]:
            avg_atr_pct = trades["atr_pct"].mean()
            trades_per_month = stats["n"] / 24  # ~24 months of data

            # Conservative sizing: 25% of $2K = $500 per position
            pos_size = 500
            pnl_per_trade = stats["avg"] / 100 * pos_size
            monthly = pnl_per_trade * trades_per_month
            yearly = monthly * 12

            print(f"  {label:<45} {stats['n']:>6} "
                  f"${pnl_per_trade:>+6.2f} ${monthly:>+7.0f} "
                  f"${yearly:>+8,.0f}")

        print(f"\n  (Assumes $500 position size = 25% of $2,000 account)")

        # Also show what happens at 50% sizing
        print(f"\n  AT 50% SIZING ($1,000 per position):")
        print(f"  {'Strategy':<45} {'$/trade':>8} {'$/month':>8} {'$/year':>9}")
        print(f"  {'-'*75}")
        for label, stats, trades in scored[:5]:
            trades_per_month = stats["n"] / 24
            pos_size = 1000
            pnl_per_trade = stats["avg"] / 100 * pos_size
            monthly = pnl_per_trade * trades_per_month
            yearly = monthly * 12
            print(f"  {label:<45} ${pnl_per_trade:>+6.2f} "
                  f"${monthly:>+7.0f} ${yearly:>+8,.0f}")

        print(f"  {'─'*80}")

    # ── Plot ─────────────────────────────────────────────────────────
    plot_results(all_results)

    # ── Cash account feasibility ─────────────────────────────────────
    print(f"\n  CASH ACCOUNT NOTES:")
    print(f"    • T+1 settlement: cash available next day after selling")
    print(f"    • 1 trade/day with $2K = use $500-$1,000 per position")
    print(f"    • 2 trades/day requires $2K+ (split across positions)")
    print(f"    • 3 trades/day requires $3K+ or settled funds from prior days")
    print(f"    • The more aggressive strategies trade more often = more $")
    print(f"    • But also more risk — bigger drawdowns")

    print(f"\n  NEXT STEPS:")
    print(f"    • If PF > 1.3 with 50+ trades → worth live testing")
    print(f"    • Compare aggressive vs conservative at your risk tolerance")
    print(f"    • Start at 25% sizing, scale up after 20 winning trades")
    print(f"    • Phase 2: intraday 5-min entry timing (VWAP pullback)")
    print(f"    • Phase 3: live pre-market scanner with IBKR real-time data")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Gap Scanner Day Trading Research"
    )
    parser.add_argument("--period", default="2y",
                        help="Data period (default: 2y)")
    args = parser.parse_args()

    run_research()
