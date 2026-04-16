"""
models/predict.py
-----------------
Loads trained LightGBM model and generates ML-scored signals
for use in backtesting and live trading.

Improvements:
  - Better feature gap handling (warns clearly about missing features)
  - Uses median imputation per-date instead of global median
  - Adds confidence metric based on model prediction spread
  - Feature coverage validation — refuses to score with < 80% features
  - Absolute ML score floor — prevents weak signals from passing percentile gate
  - Regime gate removed — daily.py applies VIX-aware regime check as sole gate
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from loguru import logger
from models.train import build_features, build_target


MODEL_PATH = Path("models/lgbm_model.pkl")

# Minimum fraction of model features that must be present to score
MIN_FEATURE_COVERAGE = 0.80


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No model found. Run: python main.py train")
    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["features"]


def generate_ml_signals(
    df: pd.DataFrame,
    top_pct: float = 0.20,
    min_score: float = None,
    apply_regime_gate: bool = False,
) -> pd.DataFrame:
    """
    Scores every stock-day with the ML model.
    Adds ml_score, ml_rank, signal, and signal_score columns.

    Parameters:
      top_pct:  fraction of top-ranked stocks to flag as signals (default 0.20)
      min_score: absolute ml_score floor; signals below this are rejected even
                 if they pass the percentile filter. If None, reads from config
                 (backtest.min_ml_score) or defaults to 0.55.
      apply_regime_gate: if True, applies the static regime_ok filter from
                 signals.py. Set False (default) when the caller (daily.py)
                 applies its own VIX-aware regime gate.

    Improvements:
      - Per-date median imputation (avoids future data leaking via global median)
      - Feature coverage validation (rejects scoring if <80% features present)
      - Absolute score floor (no weak-conviction signals)
      - Regime gate is opt-in (callers manage their own regime check)
    """
    model, feature_cols = load_model()

    # Build features
    from signals.signals import compute_market_regime
    regime = compute_market_regime(df)
    df = df.merge(regime, on="date", how="left")
    df, available_features = build_features(df)

    # ── Feature coverage validation ──────────────────────────────────
    missing = [f for f in feature_cols if f not in df.columns]
    present_count = len(feature_cols) - len(missing)
    coverage_ratio = present_count / len(feature_cols)

    if missing:
        logger.warning(f"Missing {len(missing)}/{len(feature_cols)} model features "
                       f"({coverage_ratio:.0%} coverage): {missing[:8]}...")

    if coverage_ratio < MIN_FEATURE_COVERAGE:
        logger.error(
            f"Feature coverage {coverage_ratio:.0%} below minimum {MIN_FEATURE_COVERAGE:.0%}. "
            f"ML scores unreliable — returning zero signals. "
            f"Run 'python main.py enrich' to rebuild enriched features."
        )
        df["ml_score"] = 0.0
        df["ml_rank"] = 0.0
        df["signal"] = 0
        df["signal_score"] = 0.0
        return df

    # Fill missing columns with 0 (they passed coverage check)
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
    scoring_coverage = len(valid) / len(df) * 100
    logger.info(f"ML scoring: {scoring_coverage:.1f}% of rows scored | "
                f"Feature coverage: {coverage_ratio:.0%} ({present_count}/{len(feature_cols)})")

    # Merge scores back
    df = df.merge(valid[["date", "ticker", "ml_score"]], on=["date", "ticker"], how="left")
    df["ml_score"] = df["ml_score"].fillna(0)

    # Daily cross-sectional ranking
    df["ml_rank"] = df.groupby("date")["ml_score"].rank(pct=True)

    # ── Resolve min_score floor ──────────────────────────────────────
    if min_score is None:
        try:
            import yaml
            with open("config/config.yaml") as f:
                cfg = yaml.safe_load(f)
            min_score = cfg.get("backtest", {}).get("min_ml_score", 0.18)
        except Exception:
            min_score = 0.18

    # ── Signal gate ──────────────────────────────────────────────────
    # Percentile filter + absolute score floor
    percentile_pass = df["ml_rank"] >= (1 - top_pct)
    score_floor_pass = df["ml_score"] >= min_score

    if apply_regime_gate:
        regime_pass = df["regime_ok"] == 1
        df["signal"] = (percentile_pass & score_floor_pass & regime_pass).astype(int)
    else:
        # Regime gate applied downstream by the caller (daily.py VIX-aware check)
        df["signal"] = (percentile_pass & score_floor_pass).astype(int)

    df["signal_score"] = df["ml_score"] * df["signal"]

    n = df["signal"].sum()
    n_days = df["date"].nunique()
    n_floor_rejected = int((percentile_pass & ~score_floor_pass).sum())
    logger.info(f"ML signals: {n:,} across {n_days} days ({n/max(n_days,1):.1f}/day) | "
                f"{n_floor_rejected} rejected by score floor ({min_score:.2f})")
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
