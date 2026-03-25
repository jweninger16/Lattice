"""
signals.py — v6 (improved)
Market regime filter with smoothing + better signal scoring.

Improvements over v5:
  - Regime smoothing: requires 2 consecutive days to flip state
    (avoids whipsawing in/out on noisy boundary days)
  - Signal score uses more features and better weighting
  - Added earnings avoidance in signal filter (if column present)
  - Gap detection: avoids buying into large gap-ups
"""

import pandas as pd
import numpy as np
from loguru import logger


def compute_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes daily market regime (favorable/unfavorable).
    Uses breadth + momentum with smoothing to avoid whipsaw.
    """
    daily = df.groupby("date").agg(
        pct_above_sma50=("above_sma50", "mean"),
        pct_above_sma20=("above_sma20", "mean"),
        median_momentum_21d=("momentum_21d", "median"),
    ).reset_index()

    # Raw regime signal
    daily["regime_raw"] = (
        (daily["pct_above_sma50"] >= 0.50) &
        (daily["pct_above_sma20"] >= 0.40) &
        (daily["median_momentum_21d"] >= -0.02)
    ).astype(int)

    # Smoothing: require 2 consecutive days to change state
    # This prevents flipping on noisy boundary days
    daily["regime_ok"] = daily["regime_raw"].copy()
    prev = daily["regime_raw"].shift(1).fillna(1)
    for i in range(1, len(daily)):
        raw = daily.iloc[i]["regime_raw"]
        prev_smooth = daily.iloc[i-1]["regime_ok"]
        if raw != prev_smooth:
            # Only flip if previous raw also agreed
            if daily.iloc[i-1]["regime_raw"] == raw:
                daily.iloc[i, daily.columns.get_loc("regime_ok")] = raw
            else:
                daily.iloc[i, daily.columns.get_loc("regime_ok")] = prev_smooth

    return daily[["date", "regime_ok", "pct_above_sma50", "pct_above_sma20"]]


def generate_signals(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Generates buy signals using rule-based filters.

    Improvements:
      - Avoids stocks with earnings within 5 days (if data available)
      - Avoids large gap-up entries (gap > 3%)
      - Better signal scoring with more features
    """
    logger.info("Generating signals (v6 improved)...")
    df = df.copy()

    regime = compute_market_regime(df)
    df = df.merge(regime, on="date", how="left")

    regime_days = regime["regime_ok"].sum()
    logger.info(f"Regime: favorable on {regime_days}/{len(regime)} days "
                f"({regime_days/len(regime)*100:.0f}%)")

    # Core filters
    strong_trend = (
        (df["rs_rank_63d"] >= 0.60) &
        (df["above_sma50"] == 1) &
        (df["momentum_63d"] > 0.05)
    )
    pulled_back  = df["ret_5d"] < -0.01
    near_support = df["dist_sma20"].between(-0.05, 0.03)
    volume_dry   = df["volume_ratio"] < 0.90
    reversal     = df.groupby("ticker")["close"].pct_change(1) > 0.001
    vol_filter   = df["realized_vol_21d"] < 0.55
    regime_gate  = df["regime_ok"] == 1

    # NEW: Avoid gap-up entries (buying into euphoria)
    if "gap_pct" in df.columns:
        no_gap = df["gap_pct"] < 0.03  # skip if gapped up >3%
    else:
        no_gap = True

    # NEW: Avoid earnings (if column present)
    if "earnings_soon" in df.columns:
        no_earnings = df["earnings_soon"] == 0
    else:
        no_earnings = True

    all_filters = (
        strong_trend & pulled_back & near_support &
        volume_dry & reversal & vol_filter & regime_gate &
        no_gap & no_earnings
    )
    df["signal"] = all_filters.astype(int)

    # Improved signal score with more features
    df["signal_score"] = _compute_signal_score(df) * df["signal"]

    n = df["signal"].sum()
    logger.info(f"Signals: {n:,} across {df['date'].nunique()} days ({n/df['date'].nunique():.1f}/day)")
    return df


def _compute_signal_score(df: pd.DataFrame) -> pd.Series:
    """
    Multi-factor signal quality score.
    Higher = more confident signal.
    """
    score = pd.Series(0.0, index=df.index)

    # Relative strength (most important)
    score += 0.30 * df["rs_rank_63d"].fillna(0)

    # Volume dryup strength (lower ratio = better setup)
    score += 0.20 * (1 - df["volume_ratio"].clip(0, 1.5) / 1.5).fillna(0)

    # Low volatility preference
    score += 0.15 * df["vol_rank"].fillna(0.5)

    # Pullback depth (deeper pullback near support = better entry)
    dist_score = (df["dist_sma20"].clip(-0.05, 0) / -0.05).fillna(0)
    score += 0.10 * dist_score

    # Momentum acceleration (if available)
    if "mom_accel" in df.columns:
        # Positive acceleration = momentum turning up
        accel_norm = df["mom_accel"].clip(-0.1, 0.1) / 0.1 * 0.5 + 0.5
        score += 0.10 * accel_norm.fillna(0.5)
    else:
        score += 0.10 * 0.5

    # Sector strength bonus (if available)
    if "in_leading_sector" in df.columns:
        score += 0.10 * df["in_leading_sector"].fillna(0)
    else:
        score += 0.10 * 0.5

    # Bollinger squeeze bonus (if available) — tight bands = potential breakout
    if "bb_width" in df.columns:
        bb_rank = df.groupby("date")["bb_width"].rank(pct=True, ascending=True)
        score += 0.05 * bb_rank.fillna(0.5)
    else:
        score += 0.05 * 0.5

    return score


def get_todays_signals(df: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    latest_date = df["date"].max()
    today = df[df["date"] == latest_date].copy()
    if "regime_ok" in today.columns and len(today) > 0:
        status = "FAVORABLE" if today["regime_ok"].iloc[0] == 1 else "UNFAVORABLE - sitting out"
        pct50  = today["pct_above_sma50"].iloc[0] * 100 if "pct_above_sma50" in today.columns else 0
        logger.info(f"Market regime: {status} | {pct50:.0f}% stocks above SMA50")
    signals = today[today["signal"] == 1].sort_values("signal_score", ascending=False).head(top_n)
    cols = ["ticker", "date", "close", "signal_score", "rs_rank_63d",
            "volume_ratio", "ret_5d", "dist_sma20", "realized_vol_21d", "atr_pct"]
    return signals[[c for c in cols if c in signals.columns]].reset_index(drop=True)


def signal_summary(df: pd.DataFrame) -> dict:
    signals = df[df["signal"] == 1]
    return {
        "total_signals": len(signals),
        "unique_tickers": signals["ticker"].nunique(),
        "avg_signals_per_day": len(signals) / max(df["date"].nunique(), 1),
        "avg_forward_5d_return": signals["target_5d"].mean() if "target_5d" in signals.columns else None,
        "signal_win_rate": (signals["target_5d"] > 0).mean() if "target_5d" in signals.columns else None,
    }
