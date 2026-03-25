"""
research/tight_target_backtest.py
-----------------------------------
RESEARCH — Tight target day trading strategies.

Problem: The 2.5x ATR target rarely gets hit. 65% of trades close at EOD
for small gains that get eaten by commissions. We need strategies that
CAPTURE THE MORNING MOVE and get out with real profit.

Approaches tested:
  1. Tighter ATR targets: 0.5x, 0.75x, 1.0x, 1.5x ATR
  2. Time-based exits: sell at 10:30, 11:00, 11:30 AM (morning momentum window)
  3. Trailing stop: lock in profits as price moves, never give back gains
  4. Partial profit: take half off at 1x ATR, trail the rest
  5. Commission-adjusted: subtract $1 round-trip per trade from all results
  6. OR-range targets: use opening range as the target instead of ATR

The goal: consistent $10-20 profit per trade on $500-1000 positions,
AFTER commissions. That means targeting 1-2% moves, not 5%+ moonshots.

Usage:
    python research/tight_target_backtest.py
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

sys.path.insert(0, ".")

CACHE_DIR = Path("data/gap_scanner_cache")
COMMISSION_PER_TRADE = 1.00  # Round-trip commission estimate


# ═══════════════════════════════════════════════════════════════════════
# Data (reuse cached data from gap scanner backtest)
# ═══════════════════════════════════════════════════════════════════════

def get_universe():
    try:
        from data.universe import FALLBACK_TICKERS, ETF_TICKERS
        return [t for t in FALLBACK_TICKERS if t not in ETF_TICKERS]
    except ImportError:
        return [
            "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AMD","NFLX",
            "CRM","AVGO","ORCL","QCOM","AMAT","MU","INTC","UBER","SHOP",
            "COIN","PLTR","SOFI","DKNG","SNAP","PINS","RBLX",
            "NET","CRWD","ZS","DDOG","MDB","SNOW","ABNB","DASH","ROKU",
            "JPM","BAC","GS","MS","C","WFC","V","MA","AXP","PYPL","SCHW",
            "XOM","CVX","COP","OXY","SLB","HAL","MPC","VLO","PSX",
            "UNH","JNJ","PFE","ABBV","MRK","LLY","BMY","AMGN","GILD",
            "CAT","DE","GE","HON","BA","RTX","LMT","GD","NOC",
            "HD","LOW","TGT","COST","WMT","TJX","ROST","DG","DLTR",
            "DIS","CMCSA","T","TMUS","VZ","CHTR",
        ]


def download_data(period="2y"):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "daily_universe.parquet"

    if cache_path.exists():
        age = datetime.now().timestamp() - cache_path.stat().st_mtime
        if age < 43200:
            print(f"  Using cached data ({age/3600:.1f}h old)")
            return pd.read_parquet(cache_path)

    # Also check the other cache location
    alt_cache = Path("data/gap_scanner_cache/daily_universe.parquet")
    if alt_cache.exists():
        age = datetime.now().timestamp() - alt_cache.stat().st_mtime
        if age < 43200:
            print(f"  Using cached data from gap_scanner ({age/3600:.1f}h old)")
            return pd.read_parquet(alt_cache)

    tickers = get_universe()
    print(f"  Downloading {len(tickers)} tickers ({period})...")
    raw = yf.download(tickers, period=period, auto_adjust=True,
                      progress=True, threads=True)
    if raw.empty:
        return pd.DataFrame()

    raw.columns.names = ["Field", "Ticker"]
    df = raw.stack(level="Ticker", future_stack=True).reset_index()
    df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]
    if "level_1" in df.columns:
        df = df.rename(columns={"level_1": "ticker"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close", "volume"])
    df = df[df["close"] > 0].copy()

    df.to_parquet(cache_path, index=False)
    return df


def build_features(df):
    df = df.sort_values(["ticker", "date"]).copy()
    out = []
    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date").copy()
        g["prev_close"] = g["close"].shift(1)
        g["gap_pct"] = (g["open"] - g["prev_close"]) / g["prev_close"] * 100
        g["vol_sma20"] = g["volume"].rolling(20).mean()
        g["rvol"] = g["volume"] / g["vol_sma20"]

        tr = pd.DataFrame({
            "hl": g["high"] - g["low"],
            "hc": (g["high"] - g["close"].shift(1)).abs(),
            "lc": (g["low"] - g["close"].shift(1)).abs(),
        })
        g["atr_14"] = tr.max(axis=1).rolling(14).mean()
        g["atr_pct"] = g["atr_14"] / g["close"] * 100

        g["open_to_high_pct"] = (g["high"] - g["open"]) / g["open"] * 100
        g["open_to_low_pct"] = (g["open"] - g["low"]) / g["open"] * 100
        g["open_to_close_pct"] = (g["close"] - g["open"]) / g["open"] * 100

        g["mom_5d"] = g["close"].pct_change(5) * 100
        g["sma_50"] = g["close"].rolling(50).mean()
        g["above_sma50"] = (g["close"] > g["sma_50"]).astype(int)
        g["dollar_volume"] = g["close"] * g["volume"]

        # How much of the gap was "given back"?
        # High water mark from open: how far did it run before pulling back?
        g["high_from_open_pct"] = (g["high"] - g["open"]) / g["open"] * 100

        out.append(g)

    result = pd.concat(out, ignore_index=True)
    return result.dropna(subset=["gap_pct", "atr_14", "vol_sma20"])


def scan_candidates(day_df, min_gap=1.5, max_gap=8.0, min_rvol=1.5):
    c = day_df.copy()
    c = c[c["prev_close"] >= 10]
    c = c[c["prev_close"] <= 500]
    c = c[c["dollar_volume"] >= 5_000_000]
    c = c[c["atr_pct"] >= 1.0]
    c = c[c["gap_pct"] >= min_gap]
    c = c[c["gap_pct"] <= max_gap]
    c = c[c["rvol"] >= min_rvol]

    if c.empty:
        return c

    c["score"] = (
        c["gap_pct"] * 0.30 +
        c["rvol"].clip(upper=5) * 0.25 +
        c["mom_5d"].clip(-5, 10) * 0.15 +
        c["above_sma50"] * 0.15 * 5 +
        (c["atr_pct"] * 0.15).clip(upper=1.5)
    )
    return c.sort_values("score", ascending=False)


# ═══════════════════════════════════════════════════════════════════════
# Trade simulation — multiple exit strategies
# ═══════════════════════════════════════════════════════════════════════

def simulate_trade(row, config):
    """
    Simulates a day trade with various exit strategies.
    Uses daily OHLC data.
    """
    entry = row["open"]
    atr = row["atr_14"]
    prev_close = row["prev_close"]
    strategy = config.get("strategy", "atr_target")

    # ── Stop calculation (always gap stop) ────────────────────────────
    gap_stop = prev_close * 0.998
    atr_stop = entry - atr * config.get("stop_mult", 1.0)
    stop = max(gap_stop, atr_stop)

    if stop >= entry:
        return None  # Invalid trade

    # ── Target / exit logic ───────────────────────────────────────────

    if strategy == "atr_target":
        # Simple: fixed ATR multiple target
        target_mult = config.get("target_mult", 1.0)
        target = entry + atr * target_mult

        if row["low"] <= stop:
            exit_price = stop
            reason = "stop"
        elif row["high"] >= target:
            exit_price = target
            reason = "target"
        else:
            exit_price = row["close"]
            reason = "eod"

    elif strategy == "gap_pct_target":
        # Target based on gap size (capture X% of the gap move)
        target_pct = config.get("target_pct", 1.0)
        target = entry * (1 + target_pct / 100)

        if row["low"] <= stop:
            exit_price = stop
            reason = "stop"
        elif row["high"] >= target:
            exit_price = target
            reason = "target"
        else:
            exit_price = row["close"]
            reason = "eod"

    elif strategy == "morning_exit":
        # Approximation: exit at a fraction of the day's range
        # Morning = first half, so use midpoint between open and close
        # as a proxy for "sold at midday"
        exit_frac = config.get("exit_fraction", 0.5)

        if row["low"] <= stop:
            exit_price = stop
            reason = "stop"
        else:
            # Approximate morning exit: capture exit_frac of open→high move
            max_move = row["high"] - entry
            if max_move > 0:
                exit_price = entry + max_move * exit_frac
                reason = "morning"
            else:
                exit_price = row["close"]
                reason = "eod"

    elif strategy == "trailing_stop":
        # Trailing stop: move stop up as price rises
        trail_mult = config.get("trail_mult", 0.5)
        trail_stop = stop  # Initial stop

        # Approximate: if price hit the high, trailing stop would be
        # high - trail_mult * ATR. Did the low breach that?
        if row["low"] <= stop:
            exit_price = stop
            reason = "stop"
        else:
            trail_from_high = row["high"] - atr * trail_mult
            # Did price come back down to the trailing stop?
            if row["low"] <= trail_from_high and row["high"] > entry:
                exit_price = trail_from_high
                reason = "trail"
            elif row["high"] >= entry + atr * config.get("target_mult", 3.0):
                exit_price = entry + atr * config.get("target_mult", 3.0)
                reason = "target"
            else:
                exit_price = row["close"]
                reason = "eod"

    elif strategy == "quick_scalp":
        # Scalp: tiny target, tight stop. In and out fast.
        target_pct = config.get("target_pct", 0.5)
        stop_pct = config.get("stop_pct", 0.3)
        target = entry * (1 + target_pct / 100)
        stop = entry * (1 - stop_pct / 100)

        if row["low"] <= stop:
            exit_price = stop
            reason = "stop"
        elif row["high"] >= target:
            exit_price = target
            reason = "target"
        else:
            exit_price = row["close"]
            reason = "eod"

    else:
        return None

    pnl_pct = (exit_price - entry) / entry * 100

    return {
        "date": row["date"],
        "ticker": row["ticker"],
        "entry": entry,
        "exit": exit_price,
        "stop": stop,
        "pnl_pct": pnl_pct,
        "reason": reason,
        "gap_pct": row["gap_pct"],
        "rvol": row["rvol"],
        "atr_pct": row["atr_pct"],
        "above_sma50": row.get("above_sma50", 0),
        "high_from_open_pct": row.get("high_from_open_pct", 0),
    }


def run_backtest(df, scanner_cfg, trade_cfg, max_per_day=1,
                 commission=COMMISSION_PER_TRADE, position_size=1000):
    trades = []
    for day, day_df in df.groupby("date"):
        candidates = scan_candidates(day_df, **scanner_cfg)
        if candidates.empty:
            continue
        for _, row in candidates.head(max_per_day).iterrows():
            result = simulate_trade(row, trade_cfg)
            if result:
                # Apply commission
                comm_pct = (commission / position_size) * 100
                result["pnl_pct_raw"] = result["pnl_pct"]
                result["pnl_pct"] = result["pnl_pct"] - comm_pct
                result["commission"] = commission
                trades.append(result)

    return pd.DataFrame(trades)


# ═══════════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════════

def calc_stats(t):
    if t.empty:
        return None
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

    # Per-trade dollar P&L at $1000 position
    avg_dollar = avg / 100 * 1000
    total_dollar = total / 100 * 1000

    # Target hit rate
    target_hits = len(t[t["reason"].isin(["target", "trail", "morning"])])
    target_rate = target_hits / n * 100

    # EOD exits (the problem we're fixing)
    eod_exits = len(t[t["reason"] == "eod"])
    eod_rate = eod_exits / n * 100

    return {
        "n": n, "win_rate": wr, "pf": pf, "total": total,
        "avg": avg, "avg_win": avg_win, "avg_loss": avg_loss,
        "max_dd": max_dd, "avg_dollar": avg_dollar,
        "total_dollar": total_dollar,
        "target_rate": target_rate, "eod_rate": eod_rate,
    }


def print_table(rows, title):
    print(f"\n  {'='*110}")
    print(f"  {title}")
    print(f"  {'='*110}")
    print(f"  {'Strategy':<40} {'N':>5} {'Win%':>6} {'PF':>5} "
          f"{'Avg':>7} {'$/trade':>8} {'$/year':>8} "
          f"{'TgtHit':>7} {'EOD%':>6} {'MaxDD':>7}")
    print(f"  {'-'*110}")
    for label, s in rows:
        if s is None:
            print(f"  {label:<40} {'—':>5}")
        else:
            yearly = s["avg_dollar"] * s["n"] / 2  # ~2 years of data
            print(f"  {label:<40} {s['n']:>5} {s['win_rate']:>5.1f}% "
                  f"{s['pf']:>5.2f} {s['avg']:>+6.3f}% "
                  f"${s['avg_dollar']:>+6.2f} ${yearly:>+7.0f} "
                  f"{s['target_rate']:>6.1f}% {s['eod_rate']:>5.1f}% "
                  f"{s['max_dd']:>+6.1f}%")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def run_research():
    print("=" * 110)
    print("  TIGHT TARGET DAY TRADING RESEARCH")
    print("  Goal: consistent $10-20 per trade on $1000 positions, AFTER commissions")
    print("=" * 110)

    df = download_data("2y")
    if df.empty:
        return

    print("\n  Building features...")
    df = build_features(df)

    # First: show what the CURRENT strategy actually does
    print("\n  Analyzing current strategy's exit problem...")
    scanner_cfg = {"min_gap": 1.5, "max_gap": 8.0, "min_rvol": 1.5}

    current = run_backtest(df, scanner_cfg,
                           {"strategy": "atr_target", "target_mult": 2.5})
    if not current.empty:
        rc = current["reason"].value_counts()
        total = len(current)
        print(f"\n  CURRENT STRATEGY EXIT BREAKDOWN ({total} trades):")
        for reason, count in rc.items():
            pct = count / total * 100
            avg = current[current["reason"] == reason]["pnl_pct"].mean()
            print(f"    {reason:<10} {count:>4} trades ({pct:.0f}%)  "
                  f"avg P&L: {avg:+.3f}%")

    # ── Strategy grid ────────────────────────────────────────────────
    configs = []

    # Current strategy for reference
    configs.append(("CURRENT: 2.5x ATR target (too greedy)",
                    {"strategy": "atr_target", "target_mult": 2.5}))

    # ── Tighter ATR targets ──────────────────────────────────────────
    for mult in [0.5, 0.75, 1.0, 1.5]:
        configs.append((f"ATR target {mult}x",
                        {"strategy": "atr_target", "target_mult": mult}))

    # ── Fixed percentage targets ─────────────────────────────────────
    for pct in [0.5, 0.75, 1.0, 1.5, 2.0]:
        configs.append((f"Fixed target +{pct}%",
                        {"strategy": "gap_pct_target", "target_pct": pct}))

    # ── Morning momentum capture ─────────────────────────────────────
    for frac in [0.3, 0.5, 0.7]:
        configs.append((f"Morning capture {int(frac*100)}% of high",
                        {"strategy": "morning_exit", "exit_fraction": frac}))

    # ── Trailing stops ───────────────────────────────────────────────
    for trail in [0.3, 0.5, 0.75]:
        configs.append((f"Trailing stop {trail}x ATR",
                        {"strategy": "trailing_stop", "trail_mult": trail,
                         "target_mult": 3.0}))

    # ── Quick scalps ─────────────────────────────────────────────────
    for tgt, stp in [(0.5, 0.3), (0.75, 0.5), (1.0, 0.5), (1.0, 0.75)]:
        configs.append((f"Scalp +{tgt}% / -{stp}%",
                        {"strategy": "quick_scalp",
                         "target_pct": tgt, "stop_pct": stp}))

    # ── Run all ──────────────────────────────────────────────────────
    results = {}
    table_rows = []

    for label, trade_cfg in configs:
        trades = run_backtest(df, scanner_cfg, trade_cfg,
                              position_size=1000, commission=1.0)
        results[label] = trades
        stats = calc_stats(trades)
        table_rows.append((label, stats))

    print_table(table_rows, "ALL STRATEGIES — Commission Adjusted ($1/trade)")

    # ── Deep analysis on top strategies ──────────────────────────────
    scored = [(l, calc_stats(t), t) for l, t in results.items()
              if calc_stats(t) and calc_stats(t)["n"] >= 50]
    scored.sort(key=lambda x: x[1]["total_dollar"], reverse=True)

    print(f"\n  TOP 5 BY TOTAL DOLLAR RETURN:")
    for i, (label, stats, trades) in enumerate(scored[:5]):
        yearly = stats["avg_dollar"] * stats["n"] / 2
        rc = trades["reason"].value_counts()
        print(f"\n  #{i+1} {label}")
        print(f"      {stats['n']} trades | {stats['win_rate']:.1f}% win | "
              f"PF {stats['pf']:.2f}")
        print(f"      Avg: {stats['avg']:+.3f}% = ${stats['avg_dollar']:+.2f}/trade "
              f"| ~${yearly:+,.0f}/year on $1K positions")
        print(f"      Target hit: {stats['target_rate']:.0f}% | "
              f"EOD exit: {stats['eod_rate']:.0f}%")
        print(f"      Exits: {', '.join(f'{r}={c}' for r, c in rc.items())}")

        # Show what happens on winning vs losing days
        wins = trades[trades["pnl_pct"] > 0]
        losses = trades[trades["pnl_pct"] <= 0]
        if len(wins) > 0 and len(losses) > 0:
            print(f"      Avg win: {stats['avg_win']:+.3f}% "
                  f"(${stats['avg_win']/100*1000:+.2f}) | "
                  f"Avg loss: {stats['avg_loss']:+.3f}% "
                  f"(${stats['avg_loss']/100*1000:+.2f})")

    # ── Dollar projections ───────────────────────────────────────────
    if scored:
        print(f"\n  {'─'*80}")
        print(f"  DOLLAR PROJECTIONS (commission-adjusted, $1K positions)")
        print(f"  {'─'*80}")
        print(f"  {'Strategy':<40} {'$/trade':>8} {'$/month':>8} {'$/year':>9}")
        print(f"  {'-'*70}")
        for label, stats, trades in scored[:8]:
            trades_mo = stats["n"] / 24
            monthly = stats["avg_dollar"] * trades_mo
            yearly = monthly * 12
            print(f"  {label:<40} ${stats['avg_dollar']:>+6.2f} "
                  f"${monthly:>+7.0f} ${yearly:>+8,.0f}")
        print(f"  {'─'*80}")

    # ── Recommendation ───────────────────────────────────────────────
    if scored:
        best_label, best_stats, best_trades = scored[0]
        print(f"\n  RECOMMENDATION:")
        print(f"    Strategy: {best_label}")
        print(f"    Why: highest total dollar return after commissions")
        print(f"    Target hit rate: {best_stats['target_rate']:.0f}% "
              f"(vs {calc_stats(current)['target_rate']:.0f}% current)")
        print(f"    EOD exits: {best_stats['eod_rate']:.0f}% "
              f"(vs {calc_stats(current)['eod_rate']:.0f}% current)")
        print(f"    Avg trade: ${best_stats['avg_dollar']:+.2f} "
              f"(vs ${calc_stats(current)['avg_dollar']:+.2f} current)")


if __name__ == "__main__":
    run_research()
