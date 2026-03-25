"""
models/train.py
---------------
Phase 2: LightGBM model for predicting 5-day forward returns.

Improvements:
  - Includes new features: gap_pct, bb_width, intraday_range, obv_slope
  - Better hyperparameter defaults (lower learning rate, more regularization)
  - Quantile-based target with finer bins option
  - Feature validation (warns about missing features)
  - Cross-validation AUC variance tracking
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from pathlib import Path
from loguru import logger
import pickle
import yaml


MODEL_DIR = Path("models")

FEATURE_COLS = [
    "ret_1d", "ret_5d", "ret_21d", "ret_63d",
    "momentum_63d", "momentum_21d",
    "rs_rank_63d", "rs_rank_21d", "rs_score",
    "volume_ratio",
    "atr_pct", "realized_vol_21d", "vol_rank",
    "above_sma20", "above_sma50", "dist_sma20", "dist_sma50",
    "pct_above_sma50", "pct_above_sma20",
]

# New features from improved pipeline
EXTENDED_FEATURES = [
    "rsi_14", "dist_52w_high", "mom_accel", "vol_trend",
    # New pipeline features
    "gap_pct", "bb_width", "intraday_range", "obv_slope_20d",
    "volume_rank",
    # Sector features
    "sector_momentum_21d", "sector_momentum_63d",
    "sector_above_sma50", "sector_rs", "in_leading_sector",
    "spy_momentum_21d", "spy_above_sma50",
    "qqq_momentum_21d", "qqq_above_sma50",
    # Earnings features
    "days_to_earnings", "earnings_soon", "earnings_just_passed",
]


def build_features(df: pd.DataFrame) -> tuple:
    df = df.copy().sort_values(["ticker", "date"])

    # RSI 14
    def rsi(series, p=14):
        d = series.diff()
        g = d.clip(lower=0).rolling(p).mean()
        l = (-d.clip(upper=0)).rolling(p).mean()
        return 100 - (100 / (1 + g / l.replace(0, np.nan)))

    if "rsi_14" not in df.columns:
        df["rsi_14"] = df.groupby("ticker")["close"].transform(rsi)

    if "high_52w" not in df.columns:
        df["high_52w"] = df.groupby("ticker")["high"].transform(lambda x: x.rolling(252).max())
        df["dist_52w_high"] = (df["close"] - df["high_52w"]) / df["high_52w"].replace(0, np.nan)

    if "mom_accel" not in df.columns:
        df["mom_accel"] = df["momentum_21d"] - df.groupby("ticker")["momentum_21d"].shift(10)

    if "vol_trend" not in df.columns:
        df["vol_trend"] = df.groupby("ticker")["volume_ratio"].transform(lambda x: x.rolling(5).mean())

    # Build feature list from what's actually available
    all_candidates = FEATURE_COLS + EXTENDED_FEATURES
    available = [f for f in all_candidates if f in df.columns]

    missing = [f for f in all_candidates if f not in df.columns]
    if missing:
        logger.debug(f"Features not in data (OK): {missing[:10]}...")

    return df, available


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # v3: Use 3-day forward returns (validated improvement over 5-day)
    if "fwd_3d" not in df.columns:
        df["fwd_3d"] = df.groupby("ticker")["close"].transform(
            lambda x: x.shift(-3) / x - 1
        )
    # Keep fwd_5d for evaluation
    if "fwd_5d" not in df.columns:
        df["fwd_5d"] = df.groupby("ticker")["close"].transform(
            lambda x: x.shift(-5) / x - 1
        )
    df["target"] = (
        df.groupby("date")["fwd_3d"].rank(pct=True) >= 0.80
    ).astype(int)
    return df


def walk_forward_train(df: pd.DataFrame, config: dict, feature_cols: list) -> dict:
    vc         = config["validation"]
    dates      = pd.Series(sorted(df["date"].unique())).reset_index(drop=True)
    all_preds  = []
    fold_stats = []
    last_model = None
    fold       = 0

    # v3 conservative params (validated walk-forward improvement)
    params = {
        "objective": "binary", "metric": "auc",
        "learning_rate": 0.01,        # slow learning for better generalization
        "num_leaves": 15,             # simpler trees
        "min_child_samples": 300,     # high to prevent overfitting
        "feature_fraction": 0.5,
        "bagging_fraction": 0.75,
        "bagging_freq": 5,
        "reg_alpha": 1.0,             # strong L1 regularization
        "reg_lambda": 1.0,            # strong L2 regularization
        "max_depth": 4,               # shallow trees
        "verbose": -1, "n_jobs": -1,
    }

    train_start = dates.iloc[0]

    while True:
        train_end  = train_start + pd.DateOffset(months=vc["train_months"])
        test_start = train_end   + pd.Timedelta(days=vc["embargo_days"])
        test_end   = test_start  + pd.DateOffset(months=vc["test_months"])

        if test_start > dates.iloc[-1]:
            logger.info("Walk-forward complete: test window past end of data")
            break

        train_dates = dates[(dates >= train_start) & (dates < train_end)]
        test_dates  = dates[(dates >= test_start)  & (dates <= test_end)]

        logger.info(f"Fold {fold+1}: train {train_start.date()}>{train_end.date()} | "
                    f"test {test_start.date()}>{test_end.date()} ({len(test_dates)} days)")

        if len(test_dates) < 10 or len(train_dates) < 100:
            train_start = train_start + pd.DateOffset(months=vc["test_months"])
            continue

        tr = df[df["date"].isin(train_dates)].dropna(subset=feature_cols + ["target"])
        te = df[df["date"].isin(test_dates)].dropna(subset=feature_cols)

        if len(tr) < 500 or len(te) < 50:
            train_start = train_start + pd.DateOffset(months=vc["test_months"])
            continue

        split      = int(len(tr) * 0.80)
        train_data = lgb.Dataset(tr.iloc[:split][feature_cols], label=tr.iloc[:split]["target"])
        val_data   = lgb.Dataset(tr.iloc[split:][feature_cols], label=tr.iloc[split:]["target"],
                                 reference=train_data)
        model = lgb.train(
            params, train_data, num_boost_round=800,    # more rounds for LR=0.01
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)]
        )

        te = te.copy()
        te["ml_score"] = model.predict(te[feature_cols])
        all_preds.append(te[["date", "ticker", "ml_score", "fwd_5d", "target"]])

        auc     = roc_auc_score(te["target"].fillna(0), te["ml_score"])
        top20   = te.nlargest(max(1, int(len(te)*0.20)), "ml_score")
        top20_r = top20["fwd_5d"].mean() * 100
        bot20   = te.nsmallest(max(1, int(len(te)*0.20)), "ml_score")
        bot20_r = bot20["fwd_5d"].mean() * 100

        logger.info(f"  AUC={auc:.3f} | Top20%={top20_r:.3f}% | Bot20%={bot20_r:.3f}% | Spread={top20_r-bot20_r:.3f}%")
        fold_stats.append(dict(
            fold=fold+1, test_start=test_start,
            test_end=test_dates.iloc[-1], auc=auc,
            top20_ret=top20_r, bot20_ret=bot20_r,
            spread=top20_r - bot20_r,
            n_train=len(tr), n_test=len(te),
        ))
        last_model = model
        fold += 1
        train_start = train_start + pd.DateOffset(months=vc["test_months"])

    if not all_preds:
        raise RuntimeError("No folds completed.")

    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / "lgbm_model.pkl", "wb") as f:
        pickle.dump({"model": last_model, "features": feature_cols}, f)
    logger.info(f"Model saved. {fold} folds completed.")

    fold_df = pd.DataFrame(fold_stats)
    logger.info(f"Cross-fold AUC: {fold_df['auc'].mean():.3f} ± {fold_df['auc'].std():.3f}")
    logger.info(f"Cross-fold Spread: {fold_df['spread'].mean():.3f}% ± {fold_df['spread'].std():.3f}%")

    return {
        "oos_predictions": pd.concat(all_preds, ignore_index=True),
        "fold_stats": fold_df,
        "model": last_model,
        "feature_cols": feature_cols,
    }


def evaluate_ml_signal(oos_df: pd.DataFrame):
    oos_df = oos_df.copy()
    oos_df["score_q"] = pd.qcut(oos_df["ml_score"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    qperf = oos_df.groupby("score_q", observed=True)["fwd_5d"].agg(
        avg_ret=lambda x: x.mean()*100,
        win_rate=lambda x: (x>0).mean()*100,
        count="count"
    )
    top20 = oos_df.nlargest(max(1, int(len(oos_df)*0.20)), "ml_score")
    bot20 = oos_df.nsmallest(max(1, int(len(oos_df)*0.20)), "ml_score")
    auc   = roc_auc_score(oos_df["target"].fillna(0), oos_df["ml_score"])

    # NEW: monotonicity check (do quintiles rank correctly?)
    q_means = qperf["avg_ret"].values
    monotonic = all(q_means[i] <= q_means[i+1] for i in range(len(q_means)-1))

    print("\n" + "=" * 55)
    print("  ML MODEL — OUT OF SAMPLE RESULTS")
    print("=" * 55)
    print(f"  Overall AUC:            {auc:.3f}  (0.5=random, 1.0=perfect)")
    print(f"  Top 20% avg 5d return:  {top20['fwd_5d'].mean()*100:.3f}%")
    print(f"  Bottom 20% avg 5d ret:  {bot20['fwd_5d'].mean()*100:.3f}%")
    print(f"  Spread (top - bottom):  {(top20['fwd_5d'].mean()-bot20['fwd_5d'].mean())*100:.3f}%")
    print(f"  Quintile monotonic:     {'YES ✓' if monotonic else 'NO ✗'}")
    print(f"\n  Score Quintile Breakdown:")
    print(qperf.to_string())
    print("=" * 55 + "\n")
    return {"auc": auc, "quintile_perf": qperf, "monotonic": monotonic}


def print_feature_importance(model, feature_cols: list, top_n: int = 15):
    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    max_imp = imp["importance"].max()
    print(f"  Top {top_n} Features:")
    for _, r in imp.head(top_n).iterrows():
        bar = "█" * int(r["importance"] / max_imp * 25)
        print(f"  {r['feature']:<25} {bar}")


if __name__ == "__main__":
    import sys, yaml
    sys.path.insert(0, ".")
    from data.pipeline import load_processed
    from signals.signals import compute_market_regime

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    df = load_processed()
    regime = compute_market_regime(df)
    df = df.merge(regime, on="date", how="left")
    df, feature_cols = build_features(df)
    df = build_target(df)

    logger.info(f"Training on {len(df):,} rows, {len(feature_cols)} features...")
    results = walk_forward_train(df, config, feature_cols)
    evaluate_ml_signal(results["oos_predictions"])
    print_feature_importance(results["model"], results["feature_cols"])
    print("\nFold results:")
    print(results["fold_stats"].to_string(index=False))
