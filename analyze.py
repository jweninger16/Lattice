"""
analyze.py - Signal quality diagnosis
Run: python analyze.py
"""
import sys
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
from data.pipeline import load_processed
from signals.signals import generate_signals

print("Loading data...")
df = load_processed()
df = generate_signals(df)

# Recompute forward returns cleanly (shift by -N within each ticker)
print("Computing forward returns...")
df = df.sort_values(["ticker","date"])
for days in [1, 3, 5, 8, 10]:
    df[f"fwd_{days}d"] = df.groupby("ticker")["close"].transform(
        lambda x: x.shift(-days) / x - 1
    )

signals = df[df["signal"] == 1].copy()
nosig   = df[df["signal"] == 0].copy()
print(f"Total signal observations: {len(signals):,}")

# ── 1. Raw edge ────────────────────────────────────────────────────
print(f"\n--- Raw Edge (properly computed forward returns) ---")
for days in [1, 3, 5, 8, 10]:
    col = f"fwd_{days}d"
    s_mean = signals[col].mean() * 100
    b_mean = df[col].mean() * 100
    s_wr   = (signals[col] > 0).mean() * 100
    print(f"  {days:>2}d: signal avg={s_mean:.3f}%  baseline={b_mean:.3f}%  "
          f"edge={s_mean-b_mean:+.3f}%  win_rate={s_wr:.1f}%")

# ── 2. By year ─────────────────────────────────────────────────────
print(f"\n--- Signal 5d Forward Return by Year ---")
signals["year"] = signals["date"].dt.year
yearly = signals.groupby("year")["fwd_5d"].agg(
    avg_ret=lambda x: x.mean()*100,
    win_rate=lambda x: (x>0).mean()*100,
    count="count"
)
print(yearly.to_string())

# ── 3. Volatility regime using cross-sectional vol ─────────────────
print(f"\n--- Signal Performance by Market Volatility Regime ---")
# Use avg realized vol across all stocks as market vol proxy
mkt_vol = df.groupby("date")["realized_vol_21d"].median().reset_index()
mkt_vol.columns = ["date", "mkt_vol"]
signals = signals.merge(mkt_vol, on="date", how="left")
signals["regime"] = pd.cut(signals["mkt_vol"],
    bins=[0, 0.15, 0.22, 0.35, 999],
    labels=["Low vol", "Normal", "Elevated", "Crisis"])
regime_perf = signals.groupby("regime", observed=True)["fwd_5d"].agg(
    avg_ret=lambda x: x.mean()*100,
    win_rate=lambda x: (x>0).mean()*100,
    count="count"
)
print(regime_perf.to_string())

# ── 4. Signal score quintiles ──────────────────────────────────────
print(f"\n--- 5d Return by Signal Score Quintile ---")
signals["score_q"] = pd.qcut(signals["signal_score"], 5,
    labels=["Q1 weak","Q2","Q3","Q4","Q5 strong"])
score_perf = signals.groupby("score_q", observed=True)["fwd_5d"].agg(
    avg_ret=lambda x: x.mean()*100,
    win_rate=lambda x: (x>0).mean()*100,
    count="count"
)
print(score_perf.to_string())

# ── 5. Excluding crisis years ──────────────────────────────────────
print(f"\n--- Edge Excluding 2022 and 2026 (crisis/selloff years) ---")
good_years = signals[~signals["year"].isin([2022, 2026])]
bad_years  = signals[signals["year"].isin([2022, 2026])]
for days in [1, 3, 5, 8]:
    col = f"fwd_{days}d"
    g = good_years[col].mean() * 100
    b = bad_years[col].mean() * 100
    print(f"  {days}d: good years avg={g:.3f}%   bad years avg={b:.3f}%")

print("\nDone.")
