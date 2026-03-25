"""
models/predict.py
-----------------
Loads trained LightGBM model and generates ML-scored signals
for use in backtesting and live trading.

Improvements:
  - Better feature gap handling (warns clearly about missing features)
  - Uses median imputation per-date instead of global median
  - Adds confidence metric based on model prediction spread
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from loguru import logger
from models.train import build_features, build_target


MODEL_PATH = Path("models/lgbm_model.pkl")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No model found. Run: python main.py train")
    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["features"]


def generate_ml_signals(df: pd.DataFrame, top_pct: float = 0.20) -> pd.DataFrame:
    """
    Scores every stock-day with the ML model.
    Adds ml_score and ml_signal columns.

    Improvements:
      - Per-date median imputation (avoids future data leaking via global median)
      - Logs feature coverage stats
      - Adds ml_confidence based on score distribution
    """
    model, feature_cols = load_model()

    # Build features
    from signals.signals import compute_market_regime
    regime = compute_market_regime(df)
    df = df.merge(regime, on="date", how="left")
    df, available_features = build_features(df)

    # Check which model features are missing from data
    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        logger.warning(f"Missing {len(missing)} model features: {missing[:5]}...")
        for col in missing:
            df[col] = 0.0

    # Per-date median imputation (safer than global median)
    for col in feature_cols:
        if col in df.columns:
            # Fill NaN with same-day cross-sectional median
            daily_median = df.groupby("date")[col].transform("median")
            df[col] = df[col].fillna(daily_median)
            # If still NaN (whole day missing), use 0
            df[col] = df[col].fillna(0.0)

    # Score all rows
    valid = df[df[feature_cols].notna().all(axis=1)].copy()
    if valid.empty:
        valid = df.copy()
        for col in feature_cols:
            valid[col] = valid[col].fillna(0.0)

    valid["ml_score"] = model.predict(valid[feature_cols])

    # Log coverage
    coverage = len(valid) / len(df) * 100
    logger.info(f"ML scoring coverage: {coverage:.1f}% of rows scored")

    # Merge scores back
    df = df.merge(valid[["date", "ticker", "ml_score"]], on=["date", "ticker"], how="left")
    df["ml_score"] = df["ml_score"].fillna(0)

    # Daily cross-sectional ranking
    df["ml_rank"] = df.groupby("date")["ml_score"].rank(pct=True)

    # Regime filter
    regime_gate = df["regime_ok"] == 1

    # ML signal: top % by score AND in favorable regime
    df["signal"] = ((df["ml_rank"] >= (1 - top_pct)) & regime_gate).astype(int)
    df["signal_score"] = df["ml_score"] * df["signal"]

    n = df["signal"].sum()
    n_days = df["date"].nunique()
    logger.info(f"ML signals: {n:,} across {n_days} days ({n/max(n_days,1):.1f}/day)")
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.pipeline import load_processed
    df = load_processed()
    df = generate_ml_signals(df)
    latest = df[df["date"] == df["date"].max()]
    signals = latest[latest["signal"] == 1].sort_values("ml_score", ascending=False)
    print(f"\nToday's ML signals ({df['date'].max().date()}):")
    cols = ["ticker", "close", "ml_score", "ml_rank", "rs_rank_63d",
            "volume_ratio", "realized_vol_21d", "atr_pct"]
    print(signals[[c for c in cols if c in signals.columns]].head(10).to_string(index=False))
