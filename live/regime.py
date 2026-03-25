"""
live/regime.py
--------------
Dynamic regime threshold using VIX.

The problem with a fixed 50% threshold:
  - 48% breadth during a calm market = mild pullback, probably fine to trade
  - 48% breadth during VIX=35 panic = dangerous, stay out

This module adjusts the regime threshold based on current volatility:
  - VIX < 15 (calm):      threshold = 0.45 (looser, more trading days)
  - VIX 15-20 (normal):   threshold = 0.50 (baseline)
  - VIX 20-25 (elevated): threshold = 0.55 (tighter)
  - VIX 25-30 (stressed): threshold = 0.60 (much tighter)
  - VIX > 30 (panic):     threshold = 0.65 (very tight, near lockout)

Also tracks VIX trend — a rising VIX is more dangerous than the same
level with VIX falling.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from loguru import logger
from datetime import date


def fetch_vix(period: str = "60d") -> pd.DataFrame:
    """Downloads VIX data."""
    vix = yf.download("^VIX", period=period, auto_adjust=True, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = [col[0] for col in vix.columns]
    vix = vix[["Close"]].rename(columns={"Close": "vix"})
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    vix.index.name = "date"
    return vix.reset_index()


def get_dynamic_threshold(vix_current: float, vix_5d_avg: float = None) -> dict:
    """
    Returns dynamic regime threshold and description based on VIX level.
    Also considers VIX trend (rising vs falling).
    """
    # Base threshold by VIX level
    if vix_current < 15:
        base_threshold = 0.45
        vix_regime = "CALM"
    elif vix_current < 20:
        base_threshold = 0.50
        vix_regime = "NORMAL"
    elif vix_current < 25:
        base_threshold = 0.55
        vix_regime = "ELEVATED"
    elif vix_current < 30:
        base_threshold = 0.60
        vix_regime = "STRESSED"
    else:
        base_threshold = 0.65
        vix_regime = "PANIC"

    # Adjust for VIX trend
    trend_adj = 0.0
    vix_trend = "stable"
    if vix_5d_avg is not None:
        vix_change = (vix_current - vix_5d_avg) / vix_5d_avg
        if vix_change > 0.15:    # VIX rising fast — tighten
            trend_adj = 0.03
            vix_trend = "rising fast"
        elif vix_change > 0.05:  # VIX rising — slight tighten
            trend_adj = 0.01
            vix_trend = "rising"
        elif vix_change < -0.10: # VIX falling — loosen slightly
            trend_adj = -0.02
            vix_trend = "falling"

    final_threshold = round(base_threshold + trend_adj, 3)

    return {
        "threshold": final_threshold,
        "vix_current": vix_current,
        "vix_5d_avg": vix_5d_avg,
        "vix_regime": vix_regime,
        "vix_trend": vix_trend,
        "base_threshold": base_threshold,
        "trend_adjustment": trend_adj,
    }


def compute_dynamic_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces the fixed 50% regime filter with a dynamic VIX-adjusted threshold.
    Adds regime_ok column using the dynamic threshold.
    """
    # Get VIX data covering the same date range
    start = df["date"].min().strftime("%Y-%m-%d")
    end   = df["date"].max().strftime("%Y-%m-%d")

    logger.info("Fetching VIX data for dynamic regime threshold...")
    try:
        vix_raw = yf.download("^VIX", start=start, end=end,
                              auto_adjust=True, progress=False)
        if isinstance(vix_raw.columns, pd.MultiIndex):
            vix_raw.columns = [col[0] for col in vix_raw.columns]
        vix_df = vix_raw[["Close"]].rename(columns={"Close": "vix"})
        vix_df.index = pd.to_datetime(vix_df.index).tz_localize(None)
        vix_df.index.name = "date"
        vix_df = vix_df.reset_index()
        vix_df["vix_5d_avg"] = vix_df["vix"].rolling(5).mean()
        vix_df["threshold"] = vix_df.apply(
            lambda r: get_dynamic_threshold(r["vix"], r["vix_5d_avg"])["threshold"],
            axis=1
        )
    except Exception as e:
        logger.warning(f"VIX download failed: {e}. Using fixed 0.50 threshold.")
        vix_df = pd.DataFrame({"date": df["date"].unique(), "threshold": 0.50})

    # Compute breadth metrics per day
    daily = df.groupby("date").agg(
        pct_above_sma50=("above_sma50", "mean"),
        pct_above_sma20=("above_sma20", "mean"),
        median_momentum_21d=("momentum_21d", "median"),
    ).reset_index()

    # Merge VIX threshold
    vix_df["date"] = pd.to_datetime(vix_df["date"])
    daily["date"]  = pd.to_datetime(daily["date"])
    daily = daily.merge(vix_df[["date","vix","threshold"]], on="date", how="left")
    daily["threshold"] = daily["threshold"].fillna(0.50)
    daily["vix"]       = daily["vix"].fillna(20.0)

    # Dynamic regime: use VIX-adjusted threshold
    daily["regime_ok"] = (
        (daily["pct_above_sma50"] >= daily["threshold"]) &
        (daily["pct_above_sma20"] >= daily["threshold"] - 0.10) &
        (daily["median_momentum_21d"] >= -0.02)
    ).astype(int)

    logger.info(f"Dynamic regime: favorable on "
                f"{daily['regime_ok'].sum()}/{len(daily)} days "
                f"({daily['regime_ok'].mean()*100:.0f}%)")

    return daily[["date","regime_ok","pct_above_sma50","pct_above_sma20",
                  "vix","threshold"]]


def get_todays_vix_context() -> dict:
    """Gets current VIX and dynamic threshold for today's brief."""
    try:
        vix_df = fetch_vix(period="10d")
        vix_current = float(vix_df["vix"].iloc[-1])
        vix_5d_avg  = float(vix_df["vix"].tail(5).mean())
        return get_dynamic_threshold(vix_current, vix_5d_avg)
    except Exception as e:
        logger.warning(f"Could not fetch VIX: {e}")
        return {"threshold": 0.50, "vix_current": None, "vix_regime": "UNKNOWN"}


if __name__ == "__main__":
    ctx = get_todays_vix_context()
    print(f"\nToday's VIX Context:")
    print(f"  VIX:              {ctx['vix_current']:.1f}")
    print(f"  5-day avg:        {ctx['vix_5d_avg']:.1f}")
    print(f"  VIX regime:       {ctx['vix_regime']}")
    print(f"  VIX trend:        {ctx['vix_trend']}")
    print(f"  Base threshold:   {ctx['base_threshold']:.2f}")
    print(f"  Trend adjustment: {ctx['trend_adjustment']:+.2f}")
    print(f"  Final threshold:  {ctx['threshold']:.2f}")
    print(f"  Meaning: market needs {ctx['threshold']*100:.0f}%+ stocks above SMA50 to be FAVORABLE")
