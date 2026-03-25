"""
research/advanced_selection_backtest.py
-----------------------------------------
ADVANCED STOCK SELECTION RESEARCH

Tests which metrics actually improve day trade stock picking:
  1. Float size — low float stocks move faster on volume
  2. Short interest — high SI + gap = short squeeze fuel
  3. VWAP-based exits — academic research shows Sharpe 3.0+
  4. Pre-market volume acceleration — early volume predicts follow-through
  5. Sector momentum — are peers also gapping? (sector confirmation)
  6. Enhanced scoring models — weighted combinations of all factors
  7. Multi-trade per day — 2-3 trades when conditions are strong

Data sources:
  - Price/volume: yfinance daily (cached)
  - Float/short interest: yfinance .info (fetched once, cached)

Usage:
    python research/advanced_selection_backtest.py
"""

import sys
import json
import time
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

sys.path.insert(0, ".")

CACHE_DIR = Path("data/gap_scanner_cache")
COMMISSION = 1.00  # Round-trip
POSITION_SIZE = 1000


# ═══════════════════════════════════════════════════════════════════════
# Universe & Data
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

# Sector mapping for sector momentum
SECTOR_MAP = {
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "AMZN": "Consumer",
    "META": "Tech", "GOOGL": "Tech", "TSLA": "Consumer", "AMD": "Tech",
    "NFLX": "Consumer", "CRM": "Tech", "AVGO": "Tech", "ORCL": "Tech",
    "QCOM": "Tech", "AMAT": "Tech", "MU": "Tech", "INTC": "Tech",
    "UBER": "Tech", "SHOP": "Tech", "COIN": "Fintech", "PLTR": "Tech",
    "SOFI": "Fintech", "DKNG": "Consumer", "SNAP": "Tech", "PINS": "Tech",
    "RBLX": "Tech", "NET": "Tech", "CRWD": "Tech", "ZS": "Tech",
    "DDOG": "Tech", "MDB": "Tech", "SNOW": "Tech", "ABNB": "Consumer",
    "DASH": "Consumer", "ROKU": "Tech",
    "JPM": "Finance", "BAC": "Finance", "GS": "Finance", "MS": "Finance",
    "C": "Finance", "WFC": "Finance", "V": "Finance", "MA": "Finance",
    "AXP": "Finance", "PYPL": "Fintech", "SCHW": "Finance",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "OXY": "Energy",
    "SLB": "Energy", "HAL": "Energy", "MPC": "Energy", "VLO": "Energy",
    "PSX": "Energy",
    "UNH": "Health", "JNJ": "Health", "PFE": "Health", "ABBV": "Health",
    "MRK": "Health", "LLY": "Health", "BMY": "Health", "AMGN": "Health",
    "GILD": "Health",
    "CAT": "Industrial", "DE": "Industrial", "GE": "Industrial",
    "HON": "Industrial", "BA": "Industrial", "RTX": "Industrial",
    "LMT": "Industrial", "GD": "Industrial", "NOC": "Industrial",
    "HD": "Retail", "LOW": "Retail", "TGT": "Retail", "COST": "Retail",
    "WMT": "Retail", "TJX": "Retail", "ROST": "Retail", "DG": "Retail",
    "DLTR": "Retail",
    "DIS": "Media", "CMCSA": "Media", "T": "Telecom", "TMUS": "Telecom",
    "VZ": "Telecom", "CHTR": "Telecom",
}


def download_price_data(period="2y"):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "daily_universe.parquet"

    if cache_path.exists():
        age = datetime.now().timestamp() - cache_path.stat().st_mtime
        if age < 43200:
            print(f"  Using cached price data ({age/3600:.1f}h old)")
            return pd.read_parquet(cache_path)

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


def fetch_float_short_data():
    """Fetch float and short interest data from yfinance .info endpoint."""
    cache_path = CACHE_DIR / "float_short_data.json"

    if cache_path.exists():
        age = datetime.now().timestamp() - cache_path.stat().st_mtime
        if age < 86400 * 7:  # Cache for 7 days (doesn't change fast)
            print(f"  Using cached float/short data ({age/3600:.0f}h old)")
            with open(cache_path) as f:
                return json.load(f)

    tickers = get_universe()
    print(f"  Fetching float & short interest for {len(tickers)} tickers...")
    data = {}

    for i, ticker in enumerate(tickers):
        try:
            info = yf.Ticker(ticker).info
            data[ticker] = {
                "float_shares": info.get("floatShares", None),
                "short_interest": info.get("sharesShort", None),
                "short_pct_float": info.get("shortPercentOfFloat", None),
                "shares_outstanding": info.get("sharesOutstanding", None),
                "market_cap": info.get("marketCap", None),
                "avg_volume": info.get("averageVolume", None),
                "sector": info.get("sector", SECTOR_MAP.get(ticker, "Other")),
            }
            if (i + 1) % 20 == 0:
                print(f"    {i+1}/{len(tickers)} fetched...")
            time.sleep(0.2)  # Rate limit
        except Exception as e:
            data[ticker] = {}

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"  Fetched float data for {len(data)} tickers")
    return data


# ═══════════════════════════════════════════════════════════════════════
# Feature Engineering
# ═══════════════════════════════════════════════════════════════════════

def build_features(df, float_data):
    """Build all features including advanced metrics."""
    df = df.sort_values(["ticker", "date"]).copy()
    out = []

    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date").copy()
        if len(g) < 50:
            continue

        # ── Basic features ────────────────────────────────────────────
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

        g["mom_5d"] = g["close"].pct_change(5) * 100
        g["mom_10d"] = g["close"].pct_change(10) * 100
        g["sma_50"] = g["close"].rolling(50).mean()
        g["above_sma50"] = (g["close"] > g["sma_50"]).astype(int)
        g["dollar_volume"] = g["close"] * g["volume"]

        # Intraday metrics from daily OHLC
        g["open_to_high_pct"] = (g["high"] - g["open"]) / g["open"] * 100
        g["open_to_close_pct"] = (g["close"] - g["open"]) / g["open"] * 100
        g["day_range_pct"] = (g["high"] - g["low"]) / g["open"] * 100

        # ── VWAP proxy (daily VWAP approximation) ────────────────────
        # True VWAP needs intraday data, but we can approximate using
        # typical price as a proxy
        g["typical_price"] = (g["high"] + g["low"] + g["close"]) / 3
        g["vwap_proxy"] = (g["typical_price"] * g["volume"]).rolling(5).sum() / \
                          g["volume"].rolling(5).sum()
        g["price_vs_vwap"] = (g["close"] - g["vwap_proxy"]) / g["vwap_proxy"] * 100

        # ── Volume acceleration ──────────────────────────────────────
        # Is volume increasing day over day? (pre-breakout signal)
        g["vol_accel"] = g["volume"] / g["volume"].shift(1)
        g["vol_accel_3d"] = g["volume"].rolling(3).mean() / \
                            g["volume"].rolling(10).mean()

        # ── Price compression (tighter range → bigger breakout) ──────
        g["range_5d"] = g["day_range_pct"].rolling(5).mean()
        g["range_20d"] = g["day_range_pct"].rolling(20).mean()
        g["compression"] = g["range_20d"] / (g["range_5d"] + 0.001)
        # High compression = range was big, now tight → ready to break out

        # ── Consecutive green/red days ───────────────────────────────
        green = (g["close"] > g["open"]).astype(int)
        streak = green.copy()
        for j in range(1, 5):
            streak += green.shift(j).fillna(0)
        g["green_streak"] = streak

        # ── Float and short interest (static, from .info) ────────────
        fdata = float_data.get(ticker, {})
        g["float_shares"] = fdata.get("float_shares", None)
        g["short_pct_float"] = fdata.get("short_pct_float", None)
        g["sector"] = fdata.get("sector", SECTOR_MAP.get(ticker, "Other"))

        # Float category
        fl = fdata.get("float_shares")
        if fl and fl > 0:
            if fl < 50_000_000:
                g["float_cat"] = "low"      # < 50M float
            elif fl < 200_000_000:
                g["float_cat"] = "mid"      # 50-200M
            else:
                g["float_cat"] = "high"     # > 200M
        else:
            g["float_cat"] = "unknown"

        # Short interest category
        si = fdata.get("short_pct_float")
        if si and si > 0:
            g["si_pct"] = si * 100  # Convert to percentage
            if si > 0.10:
                g["si_cat"] = "high"    # >10% short
            elif si > 0.05:
                g["si_cat"] = "mid"     # 5-10%
            else:
                g["si_cat"] = "low"     # <5%
        else:
            g["si_pct"] = 0
            g["si_cat"] = "unknown"

        out.append(g)

    result = pd.concat(out, ignore_index=True)
    return result.dropna(subset=["gap_pct", "atr_14", "vol_sma20"])


def add_sector_features(df):
    """Add sector-level momentum and confirmation features."""
    # Sector gap average per day
    sector_gap = df.groupby(["date", "sector"])["gap_pct"].mean().reset_index()
    sector_gap.columns = ["date", "sector", "sector_avg_gap"]
    df = df.merge(sector_gap, on=["date", "sector"], how="left")

    # How many stocks in the sector are gapping up?
    sector_gappers = df[df["gap_pct"] >= 1.5].groupby(
        ["date", "sector"]).size().reset_index(name="sector_gapper_count")
    df = df.merge(sector_gappers, on=["date", "sector"], how="left")
    df["sector_gapper_count"] = df["sector_gapper_count"].fillna(0)

    # Sector confirmation: is this stock's gap supported by its sector?
    df["sector_confirmed"] = (df["sector_avg_gap"] > 0.5).astype(int)

    return df


# ═══════════════════════════════════════════════════════════════════════
# Trade Simulation (trailing stop — the winning exit strategy)
# ═══════════════════════════════════════════════════════════════════════

def simulate_trailing_trade(row, trail_mult=0.3):
    """Simulate a trade with trailing stop exit."""
    entry = row["open"]
    atr = row["atr_14"]
    prev_close = row["prev_close"]

    # Gap stop
    gap_stop = prev_close * 0.998
    atr_stop = entry - atr * 1.0
    stop = max(gap_stop, atr_stop)

    if stop >= entry:
        return None

    trail_dist = atr * trail_mult

    # Check if initial stop was hit
    if row["low"] <= stop:
        exit_price = stop
        reason = "stop"
    else:
        # Trailing stop from the high
        trail_from_high = row["high"] - trail_dist
        if trail_from_high > entry and row["low"] <= trail_from_high:
            exit_price = trail_from_high
            reason = "trail"
        else:
            exit_price = row["close"]
            reason = "eod"

    pnl_pct = (exit_price - entry) / entry * 100
    # Subtract commission
    pnl_pct -= (COMMISSION / POSITION_SIZE) * 100

    return {
        "date": row["date"],
        "ticker": row["ticker"],
        "entry": entry,
        "exit": exit_price,
        "pnl_pct": pnl_pct,
        "reason": reason,
        "gap_pct": row["gap_pct"],
        "rvol": row["rvol"],
        "atr_pct": row["atr_pct"],
        "above_sma50": row.get("above_sma50", 0),
        "float_cat": row.get("float_cat", "unknown"),
        "si_cat": row.get("si_cat", "unknown"),
        "sector": row.get("sector", "Other"),
        "sector_confirmed": row.get("sector_confirmed", 0),
    }


# ═══════════════════════════════════════════════════════════════════════
# Scoring Models
# ═══════════════════════════════════════════════════════════════════════

def score_basic(c):
    """Current scoring model."""
    return (
        c["gap_pct"] * 0.30 +
        c["rvol"].clip(upper=5) * 0.25 +
        c["mom_5d"].clip(-5, 10) * 0.15 +
        c["above_sma50"] * 0.15 * 5 +
        (c["atr_pct"] * 0.15).clip(upper=1.5)
    )


def score_float_weighted(c):
    """Boost low-float stocks."""
    base = score_basic(c)
    float_bonus = c["float_cat"].map({
        "low": 2.0, "mid": 0.5, "high": 0.0, "unknown": 0.0
    }).fillna(0)
    return base + float_bonus


def score_short_squeeze(c):
    """Boost high short interest + low float combo."""
    base = score_basic(c)
    si_bonus = c["si_cat"].map({
        "high": 2.5, "mid": 1.0, "low": 0.0, "unknown": 0.0
    }).fillna(0)
    float_bonus = c["float_cat"].map({
        "low": 1.5, "mid": 0.5, "high": 0.0, "unknown": 0.0
    }).fillna(0)
    return base + si_bonus + float_bonus


def score_sector_confirmed(c):
    """Boost stocks where the whole sector is gapping."""
    base = score_basic(c)
    sector_bonus = c["sector_confirmed"] * 2.0
    return base + sector_bonus


def score_volume_accel(c):
    """Boost stocks with accelerating volume into the gap."""
    base = score_basic(c)
    vol_bonus = c["vol_accel_3d"].clip(0, 3) * 0.5
    return base + vol_bonus


def score_compression_breakout(c):
    """Boost stocks breaking out of tight ranges."""
    base = score_basic(c)
    comp_bonus = c["compression"].clip(0, 5) * 0.4
    return base + comp_bonus


def score_kitchen_sink(c):
    """Everything combined — the mega model."""
    base = score_basic(c)
    float_bonus = c["float_cat"].map({
        "low": 1.5, "mid": 0.3, "high": 0.0, "unknown": 0.0
    }).fillna(0)
    si_bonus = c["si_cat"].map({
        "high": 2.0, "mid": 0.8, "low": 0.0, "unknown": 0.0
    }).fillna(0)
    sector_bonus = c["sector_confirmed"] * 1.5
    vol_bonus = c["vol_accel_3d"].clip(0, 3) * 0.3
    comp_bonus = c["compression"].clip(0, 5) * 0.3
    return base + float_bonus + si_bonus + sector_bonus + vol_bonus + comp_bonus


SCORING_MODELS = {
    "CURRENT: Basic scoring": score_basic,
    "Float-weighted (low float boost)": score_float_weighted,
    "Short squeeze (SI + float)": score_short_squeeze,
    "Sector confirmation": score_sector_confirmed,
    "Volume acceleration": score_volume_accel,
    "Compression breakout": score_compression_breakout,
    "Kitchen sink (all factors)": score_kitchen_sink,
}


# ═══════════════════════════════════════════════════════════════════════
# Backtest Runner
# ═══════════════════════════════════════════════════════════════════════

def scan_and_score(day_df, score_fn, min_gap=1.5, max_gap=8.0,
                   min_rvol=1.5):
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

    c["score"] = score_fn(c)
    return c.sort_values("score", ascending=False)


def run_backtest(df, score_fn, max_per_day=1, trail_mult=0.3):
    trades = []
    for day, day_df in df.groupby("date"):
        candidates = scan_and_score(day_df, score_fn)
        if candidates.empty:
            continue
        for _, row in candidates.head(max_per_day).iterrows():
            result = simulate_trailing_trade(row, trail_mult)
            if result:
                trades.append(result)
    return pd.DataFrame(trades)


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
    avg_dollar = avg / 100 * POSITION_SIZE
    total_dollar = total / 100 * POSITION_SIZE

    # Exit breakdown
    reasons = t["reason"].value_counts()
    trail_pct = reasons.get("trail", 0) / n * 100
    eod_pct = reasons.get("eod", 0) / n * 100

    return {
        "n": n, "win_rate": wr, "pf": pf, "total": total,
        "avg": avg, "avg_win": avg_win, "avg_loss": avg_loss,
        "max_dd": max_dd, "avg_dollar": avg_dollar,
        "total_dollar": total_dollar,
        "trail_pct": trail_pct, "eod_pct": eod_pct,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main Research
# ═══════════════════════════════════════════════════════════════════════

def run_research():
    print("=" * 110)
    print("  ADVANCED STOCK SELECTION RESEARCH")
    print("  Testing: float, short interest, VWAP, sector momentum, ")
    print("  compression breakouts, volume acceleration, multi-trade")
    print("=" * 110)

    # ── Data ──────────────────────────────────────────────────────────
    df = download_price_data("2y")
    if df.empty:
        print("  No data!")
        return

    float_data = fetch_float_short_data()

    print("\n  Building features...")
    df = build_features(df, float_data)
    df = add_sector_features(df)
    print(f"  Universe: {df['ticker'].nunique()} tickers × "
          f"{df['date'].nunique()} days")

    # ── Float & SI distribution ───────────────────────────────────────
    print("\n  FLOAT & SHORT INTEREST DISTRIBUTION:")
    latest = df.groupby("ticker").last()
    for cat in ["low", "mid", "high", "unknown"]:
        count = (latest["float_cat"] == cat).sum()
        print(f"    Float {cat}: {count} stocks")
    for cat in ["low", "mid", "high", "unknown"]:
        count = (latest["si_cat"] == cat).sum()
        print(f"    Short interest {cat}: {count} stocks")

    # ── Test all scoring models ───────────────────────────────────────
    print(f"\n  {'='*110}")
    print(f"  SCORING MODEL COMPARISON (trailing stop 0.3x ATR, 1 trade/day)")
    print(f"  {'='*110}")
    print(f"  {'Model':<42} {'N':>5} {'Win%':>6} {'PF':>5} "
          f"{'Avg':>7} {'$/trade':>8} {'$/year':>8} "
          f"{'Trail%':>7} {'MaxDD':>7}")
    print(f"  {'-'*110}")

    results = {}
    for label, score_fn in SCORING_MODELS.items():
        trades = run_backtest(df, score_fn, max_per_day=1, trail_mult=0.3)
        results[label] = trades
        s = calc_stats(trades)
        if s:
            yearly = s["avg_dollar"] * s["n"] / 2
            print(f"  {label:<42} {s['n']:>5} {s['win_rate']:>5.1f}% "
                  f"{s['pf']:>5.2f} {s['avg']:>+6.3f}% "
                  f"${s['avg_dollar']:>+6.2f} ${yearly:>+7.0f} "
                  f"{s['trail_pct']:>6.1f}% {s['max_dd']:>+6.1f}%")

    # ── Analyze what float and SI actually do to returns ──────────────
    print(f"\n  {'='*110}")
    print(f"  FACTOR ANALYSIS — Does each factor actually help?")
    print(f"  {'='*110}")

    # Use basic scoring for factor analysis
    all_trades = run_backtest(df, score_basic, max_per_day=1, trail_mult=0.3)

    if not all_trades.empty:
        # Float analysis
        print(f"\n  BY FLOAT SIZE:")
        for cat in ["low", "mid", "high", "unknown"]:
            subset = all_trades[all_trades["float_cat"] == cat]
            if len(subset) >= 5:
                avg = subset["pnl_pct"].mean()
                wr = len(subset[subset["pnl_pct"] > 0]) / len(subset) * 100
                print(f"    {cat:<10} {len(subset):>4} trades, "
                      f"{wr:.0f}% win, {avg:+.3f}% avg "
                      f"(${avg/100*POSITION_SIZE:+.2f}/trade)")

        # Short interest analysis
        print(f"\n  BY SHORT INTEREST:")
        for cat in ["low", "mid", "high", "unknown"]:
            subset = all_trades[all_trades["si_cat"] == cat]
            if len(subset) >= 5:
                avg = subset["pnl_pct"].mean()
                wr = len(subset[subset["pnl_pct"] > 0]) / len(subset) * 100
                print(f"    {cat:<10} {len(subset):>4} trades, "
                      f"{wr:.0f}% win, {avg:+.3f}% avg "
                      f"(${avg/100*POSITION_SIZE:+.2f}/trade)")

        # Sector confirmation
        print(f"\n  BY SECTOR CONFIRMATION:")
        for val, label in [(1, "confirmed"), (0, "solo")]:
            subset = all_trades[all_trades["sector_confirmed"] == val]
            if len(subset) >= 5:
                avg = subset["pnl_pct"].mean()
                wr = len(subset[subset["pnl_pct"] > 0]) / len(subset) * 100
                print(f"    {label:<15} {len(subset):>4} trades, "
                      f"{wr:.0f}% win, {avg:+.3f}% avg "
                      f"(${avg/100*POSITION_SIZE:+.2f}/trade)")

        # Gap size buckets
        print(f"\n  BY GAP SIZE:")
        bins = [(1.5, 2.5), (2.5, 4.0), (4.0, 6.0), (6.0, 10.0)]
        for lo, hi in bins:
            subset = all_trades[(all_trades["gap_pct"] >= lo) &
                               (all_trades["gap_pct"] < hi)]
            if len(subset) >= 5:
                avg = subset["pnl_pct"].mean()
                wr = len(subset[subset["pnl_pct"] > 0]) / len(subset) * 100
                print(f"    {lo:.1f}-{hi:.1f}%   {len(subset):>4} trades, "
                      f"{wr:.0f}% win, {avg:+.3f}% avg "
                      f"(${avg/100*POSITION_SIZE:+.2f}/trade)")

        # RVol buckets
        print(f"\n  BY RELATIVE VOLUME:")
        bins = [(1.5, 2.5), (2.5, 4.0), (4.0, 7.0), (7.0, 100.0)]
        for lo, hi in bins:
            subset = all_trades[(all_trades["rvol"] >= lo) &
                               (all_trades["rvol"] < hi)]
            if len(subset) >= 5:
                avg = subset["pnl_pct"].mean()
                wr = len(subset[subset["pnl_pct"] > 0]) / len(subset) * 100
                print(f"    {lo:.1f}-{hi:.1f}x   {len(subset):>4} trades, "
                      f"{wr:.0f}% win, {avg:+.3f}% avg "
                      f"(${avg/100*POSITION_SIZE:+.2f}/trade)")

    # ── Multi-trade per day ───────────────────────────────────────────
    print(f"\n  {'='*110}")
    print(f"  MULTI-TRADE PER DAY (using best scoring model)")
    print(f"  {'='*110}")

    # Find best model
    scored = [(l, calc_stats(t), t) for l, t in results.items()
              if calc_stats(t) and calc_stats(t)["n"] >= 50]
    scored.sort(key=lambda x: x[1]["total_dollar"], reverse=True)
    best_label = scored[0][0] if scored else "CURRENT: Basic scoring"
    best_fn = SCORING_MODELS[best_label]

    print(f"  Best model: {best_label}\n")
    print(f"  {'Trades/day':<20} {'N':>5} {'Win%':>6} {'PF':>5} "
          f"{'$/trade':>8} {'$/month':>8} {'$/year':>9} {'MaxDD':>7}")
    print(f"  {'-'*80}")

    for max_trades in [1, 2, 3, 5]:
        trades = run_backtest(df, best_fn, max_per_day=max_trades,
                              trail_mult=0.3)
        s = calc_stats(trades)
        if s:
            trades_mo = s["n"] / 24
            monthly = s["avg_dollar"] * trades_mo
            yearly = monthly * 12
            print(f"  Top {max_trades:<17} {s['n']:>5} {s['win_rate']:>5.1f}% "
                  f"{s['pf']:>5.2f} ${s['avg_dollar']:>+6.2f} "
                  f"${monthly:>+7.0f} ${yearly:>+8,.0f} {s['max_dd']:>+6.1f}%")

    # ── Trailing stop sensitivity ─────────────────────────────────────
    print(f"\n  {'='*110}")
    print(f"  TRAILING STOP SENSITIVITY (best model, 1 trade/day)")
    print(f"  {'='*110}")
    print(f"  {'Trail mult':<15} {'N':>5} {'Win%':>6} {'PF':>5} "
          f"{'$/trade':>8} {'$/year':>8} {'Trail%':>7} {'EOD%':>6}")
    print(f"  {'-'*70}")

    for trail in [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]:
        trades = run_backtest(df, best_fn, max_per_day=1, trail_mult=trail)
        s = calc_stats(trades)
        if s:
            yearly = s["avg_dollar"] * s["n"] / 2
            print(f"  {trail:.2f}x ATR       {s['n']:>5} {s['win_rate']:>5.1f}% "
                  f"{s['pf']:>5.2f} ${s['avg_dollar']:>+6.2f} "
                  f"${yearly:>+7.0f} {s['trail_pct']:>6.1f}% "
                  f"{s['eod_pct']:>5.1f}%")

    # ── Dollar projections ────────────────────────────────────────────
    if scored:
        print(f"\n  {'─'*80}")
        print(f"  DOLLAR PROJECTIONS — Best models on $2,000 account")
        print(f"  {'─'*80}")
        print(f"  {'Strategy':<42} {'$/trade':>8} {'$/month':>8} {'$/year':>9}")
        print(f"  {'-'*70}")
        for label, stats, trades in scored[:5]:
            trades_mo = stats["n"] / 24
            monthly = stats["avg_dollar"] * trades_mo
            yearly = monthly * 12
            print(f"  {label:<42} ${stats['avg_dollar']:>+6.2f} "
                  f"${monthly:>+7.0f} ${yearly:>+8,.0f}")
        print(f"  {'─'*80}")

    # ── Final recommendation ──────────────────────────────────────────
    if scored:
        best = scored[0]
        print(f"\n  RECOMMENDATION:")
        print(f"    Scoring model:  {best[0]}")
        print(f"    Trades/day:     1 (until account > $4K, then 2)")
        print(f"    Exit:           Trailing stop 0.3x ATR")
        print(f"    Expected:       ${best[1]['avg_dollar']:+.2f}/trade, "
              f"~${best[1]['avg_dollar'] * best[1]['n'] / 24:+.0f}/month")
        print(f"    Win rate:       {best[1]['win_rate']:.1f}%")
        print(f"    Profit factor:  {best[1]['pf']:.2f}")


if __name__ == "__main__":
    run_research()
