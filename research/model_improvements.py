"""
research/model_improvements.py
-------------------------------
STANDALONE RESEARCH — Tests three approaches to improve model performance
WITHOUT adding new features:

1. TARGET ENGINEERING — Better targets for the model to predict
   - Risk-adjusted returns (Sharpe-like target)
   - Asymmetric target (penalize drawdowns, reward smooth gains)
   - Wider top percentile (top 30% vs top 20%)
   - Shorter horizon (3-day vs 5-day)

2. REGIME-ADAPTIVE MODEL — Different models for different market states
   - Train separate models for high-vol vs low-vol environments
   - Weight features differently based on VIX regime
   - Use regime as a meta-feature for model switching

3. ENSEMBLE METHODS — Combine multiple model types
   - LightGBM + simple momentum score blend
   - LightGBM trained on different time horizons
   - Stacked model with meta-learner

Usage:
    python research/model_improvements.py              # All three tests
    python research/model_improvements.py --target     # Target engineering only
    python research/model_improvements.py --regime     # Regime-adaptive only
    python research/model_improvements.py --ensemble   # Ensemble only
    python research/model_improvements.py --full       # Walk-forward all
"""

import sys
import yaml
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from pathlib import Path
from loguru import logger

sys.path.insert(0, ".")
from models.train import build_features, FEATURE_COLS, EXTENDED_FEATURES


# ═══════════════════════════════════════════════════════════════════════
# Shared Utilities
# ═══════════════════════════════════════════════════════════════════════

LGB_PARAMS = {
    "objective": "binary", "metric": "auc",
    "learning_rate": 0.03, "num_leaves": 31,
    "min_child_samples": 150, "feature_fraction": 0.65,
    "bagging_fraction": 0.75, "bagging_freq": 5,
    "reg_alpha": 0.3, "reg_lambda": 0.3,
    "max_depth": 6, "verbose": -1, "n_jobs": -1,
}


def load_and_prepare():
    """Load data and build base features."""
    from data.pipeline import load_processed
    try:
        df = load_processed("price_features_enriched.parquet")
    except FileNotFoundError:
        df = load_processed("price_features.parquet")

    df, feature_cols = build_features(df)

    # Ensure forward returns exist
    if "fwd_5d" not in df.columns:
        df["fwd_5d"] = df.groupby("ticker")["close"].transform(
            lambda x: x.shift(-5) / x - 1
        )
    if "fwd_3d" not in df.columns:
        df["fwd_3d"] = df.groupby("ticker")["close"].transform(
            lambda x: x.shift(-3) / x - 1
        )
    if "fwd_10d" not in df.columns:
        df["fwd_10d"] = df.groupby("ticker")["close"].transform(
            lambda x: x.shift(-10) / x - 1
        )

    # Forward max drawdown over 5 days (for risk-adjusted targets)
    def fwd_max_dd(prices, horizon=5):
        result = pd.Series(index=prices.index, dtype=float)
        vals = prices.values
        for i in range(len(vals) - horizon):
            window = vals[i:i+horizon+1]
            peak = window[0]
            max_dd = 0
            for v in window[1:]:
                peak = max(peak, v)
                dd = (v - peak) / peak
                max_dd = min(max_dd, dd)
            result.iloc[i] = max_dd
        return result

    if "fwd_max_dd_5d" not in df.columns:
        logger.info("Computing forward max drawdown (this takes a minute)...")
        df["fwd_max_dd_5d"] = df.groupby("ticker")["close"].transform(
            lambda x: fwd_max_dd(x, 5)
        )

    return df, feature_cols


def train_and_evaluate(tr, te, features, params=None):
    """Train LightGBM and evaluate on test set."""
    if params is None:
        params = LGB_PARAMS.copy()

    split = int(len(tr) * 0.8)
    train_data = lgb.Dataset(tr.iloc[:split][features], label=tr.iloc[:split]["target"])
    val_data = lgb.Dataset(tr.iloc[split:][features], label=tr.iloc[split:]["target"],
                            reference=train_data)

    model = lgb.train(params, train_data, num_boost_round=500,
                       valid_sets=[val_data],
                       callbacks=[lgb.early_stopping(30, verbose=False),
                                  lgb.log_evaluation(-1)])

    te = te.copy()
    te["ml_score"] = model.predict(te[features])
    auc = roc_auc_score(te["target"], te["ml_score"])

    te["quintile"] = pd.qcut(te["ml_score"], 5, labels=[1,2,3,4,5])
    q_returns = te.groupby("quintile")["fwd_5d"].mean() * 100

    top_q = te[te["quintile"] == 5]["fwd_5d"].mean() * 100
    bot_q = te[te["quintile"] == 1]["fwd_5d"].mean() * 100

    q_vals = q_returns.values
    monotonic = all(q_vals[i] <= q_vals[i+1] for i in range(len(q_vals)-1))

    return {
        "auc": auc, "top_q": top_q, "bot_q": bot_q,
        "spread": top_q - bot_q, "monotonic": monotonic,
        "q_returns": q_returns, "model": model, "predictions": te,
    }


def walkforward_evaluate(df, features, target_col="target", params=None):
    """Run walk-forward evaluation."""
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    if params is None:
        params = LGB_PARAMS.copy()

    vc = config["validation"]
    dates = pd.Series(sorted(df["date"].unique())).reset_index(drop=True)
    results = []
    train_start = dates.iloc[0]

    while True:
        train_end = train_start + pd.DateOffset(months=vc["train_months"])
        test_start = train_end + pd.Timedelta(days=vc["embargo_days"])
        test_end = test_start + pd.DateOffset(months=vc["test_months"])

        if test_start > dates.iloc[-1]:
            break

        train_dates = dates[(dates >= train_start) & (dates < train_end)]
        test_dates = dates[(dates >= test_start) & (dates <= test_end)]

        if len(test_dates) < 10 or len(train_dates) < 100:
            train_start += pd.DateOffset(months=vc["test_months"])
            continue

        tr = df[df["date"].isin(train_dates)].dropna(subset=features + [target_col])
        te = df[df["date"].isin(test_dates)].dropna(subset=features + [target_col])

        if len(tr) < 500 or len(te) < 50:
            train_start += pd.DateOffset(months=vc["test_months"])
            continue

        split = int(len(tr) * 0.8)
        train_data = lgb.Dataset(tr.iloc[:split][features], label=tr.iloc[:split][target_col])
        val_data = lgb.Dataset(tr.iloc[split:][features], label=tr.iloc[split:][target_col],
                                reference=train_data)

        model = lgb.train(params, train_data, num_boost_round=500,
                           valid_sets=[val_data],
                           callbacks=[lgb.early_stopping(30, verbose=False),
                                      lgb.log_evaluation(-1)])

        te = te.copy()
        te["ml_score"] = model.predict(te[features])

        # AUC uses the same target column
        auc = roc_auc_score(te[target_col], te["ml_score"])

        # But returns are always measured on fwd_5d
        top20 = te.nlargest(max(1, int(len(te) * 0.2)), "ml_score")
        top20_r = top20["fwd_5d"].mean() * 100 if "fwd_5d" in te.columns else 0
        bot20 = te.nsmallest(max(1, int(len(te) * 0.2)), "ml_score")
        bot20_r = bot20["fwd_5d"].mean() * 100 if "fwd_5d" in te.columns else 0

        results.append({
            "auc": auc, "top20_ret": top20_r,
            "bot20_ret": bot20_r, "spread": top20_r - bot20_r,
        })

        train_start += pd.DateOffset(months=vc["test_months"])

    return results


# ═══════════════════════════════════════════════════════════════════════
# 1. TARGET ENGINEERING
# ═══════════════════════════════════════════════════════════════════════

def test_target_engineering(df, features, full_wf=False):
    """Tests different target definitions."""
    logger.info("\n" + "=" * 65)
    logger.info("  TEST 1: TARGET ENGINEERING")
    logger.info("=" * 65)

    df = df.copy()

    # Define different targets
    targets = {}

    # A) Current: top 20% of 5-day returns
    df["target_base"] = (
        df.groupby("date")["fwd_5d"].rank(pct=True) >= 0.80
    ).astype(int)
    targets["Base (top 20%, 5d)"] = "target_base"

    # B) Top 30% — wider net, more samples
    df["target_top30"] = (
        df.groupby("date")["fwd_5d"].rank(pct=True) >= 0.70
    ).astype(int)
    targets["Top 30%, 5d"] = "target_top30"

    # C) Top 10% — more selective
    df["target_top10"] = (
        df.groupby("date")["fwd_5d"].rank(pct=True) >= 0.90
    ).astype(int)
    targets["Top 10%, 5d"] = "target_top10"

    # D) 3-day horizon
    df["target_3d"] = (
        df.groupby("date")["fwd_3d"].rank(pct=True) >= 0.80
    ).astype(int)
    targets["Top 20%, 3d"] = "target_3d"

    # E) 10-day horizon
    df["target_10d"] = (
        df.groupby("date")["fwd_10d"].rank(pct=True) >= 0.80
    ).astype(int)
    targets["Top 20%, 10d"] = "target_10d"

    # F) Risk-adjusted: top 20% of return / max_drawdown ratio
    df["risk_adj_return"] = df["fwd_5d"] / (-df["fwd_max_dd_5d"]).replace(0, np.nan).clip(lower=0.001)
    df["target_risk_adj"] = (
        df.groupby("date")["risk_adj_return"].rank(pct=True) >= 0.80
    ).astype(int)
    targets["Risk-Adjusted, 5d"] = "target_risk_adj"

    # G) Asymmetric: positive return AND small drawdown
    # Stock must be in top 30% return AND not in bottom 30% drawdown
    dd_rank = df.groupby("date")["fwd_max_dd_5d"].rank(pct=True)  # higher rank = less drawdown
    ret_rank = df.groupby("date")["fwd_5d"].rank(pct=True)
    df["target_asymmetric"] = (
        (ret_rank >= 0.70) & (dd_rank >= 0.30)
    ).astype(int)
    targets["Asymmetric (ret+dd)"] = "target_asymmetric"

    # H) Smooth gains: top 20% of return with above-median drawdown rank
    df["target_smooth"] = (
        (ret_rank >= 0.80) & (dd_rank >= 0.50)
    ).astype(int)
    targets["Smooth Gains"] = "target_smooth"

    if full_wf:
        print("\n  WALK-FORWARD TARGET COMPARISON:")
        print(f"  {'Target':<25} {'Folds':>6} {'AUC':>12} {'Top20% Ret':>14} {'Spread':>14}")
        print(f"  {'-'*70}")
        for name, target_col in targets.items():
            results = walkforward_evaluate(df, features, target_col=target_col)
            if results:
                aucs = [r["auc"] for r in results]
                rets = [r["top20_ret"] for r in results]
                spreads = [r["spread"] for r in results]
                print(f"  {name:<25} {len(aucs):>6} "
                      f"{np.mean(aucs):.4f}±{np.std(aucs):.3f} "
                      f"{np.mean(rets):>+.3f}±{np.std(rets):.3f}% "
                      f"{np.mean(spreads):>+.3f}±{np.std(spreads):.3f}%")
    else:
        # Single split
        dates = sorted(df["date"].unique())
        split_date = dates[int(len(dates) * 0.75)]

        print("\n  SINGLE-SPLIT TARGET COMPARISON:")
        print(f"  {'Target':<25} {'AUC':>8} {'Top Q':>9} {'Bot Q':>9} "
              f"{'Spread':>9} {'Mono':>6}")
        print(f"  {'-'*65}")

        for name, target_col in targets.items():
            df_t = df.copy()
            df_t["target"] = df_t[target_col]
            tr = df_t[df_t["date"] < split_date].dropna(subset=features + ["target"])
            te = df_t[df_t["date"] >= split_date].dropna(subset=features + ["target"])

            result = train_and_evaluate(tr, te, features)
            mono = "✓" if result["monotonic"] else "✗"
            print(f"  {name:<25} {result['auc']:>8.4f} {result['top_q']:>+8.3f}% "
                  f"{result['bot_q']:>+8.3f}% {result['spread']:>8.3f}% {mono:>6}")

    return df


# ═══════════════════════════════════════════════════════════════════════
# 2. REGIME-ADAPTIVE MODEL
# ═══════════════════════════════════════════════════════════════════════

def test_regime_adaptive(df, features, full_wf=False):
    """Tests regime-adaptive model approaches."""
    logger.info("\n" + "=" * 65)
    logger.info("  TEST 2: REGIME-ADAPTIVE MODEL")
    logger.info("=" * 65)

    df = df.copy()

    # Build target
    df["target"] = (
        df.groupby("date")["fwd_5d"].rank(pct=True) >= 0.80
    ).astype(int)

    # Define regimes based on realized volatility
    spy_vol = df.groupby("date")["realized_vol_21d"].mean()
    vol_median = spy_vol.median()

    # Map regime to each row
    date_regime = (spy_vol > vol_median).astype(int)  # 1 = high vol
    date_regime.name = "vol_regime"
    df = df.merge(date_regime.reset_index(), on="date", how="left")

    dates = sorted(df["date"].unique())
    split_date = dates[int(len(dates) * 0.75)]

    results = {}

    # A) Single model (baseline)
    tr = df[df["date"] < split_date].dropna(subset=features + ["target"])
    te = df[df["date"] >= split_date].dropna(subset=features + ["target"])
    results["Single Model"] = train_and_evaluate(tr, te, features)

    # B) Regime as feature
    features_with_regime = features + ["vol_regime"]
    tr_r = df[df["date"] < split_date].dropna(subset=features_with_regime + ["target"])
    te_r = df[df["date"] >= split_date].dropna(subset=features_with_regime + ["target"])
    results["+ Regime Feature"] = train_and_evaluate(tr_r, te_r, features_with_regime)

    # C) Separate models per regime
    te_combined = []
    for regime_val, regime_name in [(0, "low_vol"), (1, "high_vol")]:
        tr_regime = tr[tr["vol_regime"] == regime_val]
        te_regime = te[te["vol_regime"] == regime_val]

        if len(tr_regime) < 500 or len(te_regime) < 50:
            continue

        split = int(len(tr_regime) * 0.8)
        train_data = lgb.Dataset(tr_regime.iloc[:split][features],
                                  label=tr_regime.iloc[:split]["target"])
        val_data = lgb.Dataset(tr_regime.iloc[split:][features],
                                label=tr_regime.iloc[split:]["target"],
                                reference=train_data)

        model = lgb.train(LGB_PARAMS, train_data, num_boost_round=500,
                           valid_sets=[val_data],
                           callbacks=[lgb.early_stopping(30, verbose=False),
                                      lgb.log_evaluation(-1)])

        te_regime = te_regime.copy()
        te_regime["ml_score"] = model.predict(te_regime[features])
        te_combined.append(te_regime)

    if te_combined:
        te_all = pd.concat(te_combined)
        auc = roc_auc_score(te_all["target"], te_all["ml_score"])
        te_all["quintile"] = pd.qcut(te_all["ml_score"], 5, labels=[1,2,3,4,5],
                                      duplicates="drop")
        q_returns = te_all.groupby("quintile")["fwd_5d"].mean() * 100
        top_q = te_all[te_all["quintile"] == 5]["fwd_5d"].mean() * 100
        bot_q = te_all[te_all["quintile"] == 1]["fwd_5d"].mean() * 100
        q_vals = q_returns.values
        monotonic = all(q_vals[i] <= q_vals[i+1] for i in range(len(q_vals)-1))
        results["Separate Models"] = {
            "auc": auc, "top_q": top_q, "bot_q": bot_q,
            "spread": top_q - bot_q, "monotonic": monotonic,
            "q_returns": q_returns,
        }

    # D) VIX-regime interaction features
    # Create interaction between key features and regime
    interaction_features = features.copy()
    top_features = ["vol_rank", "atr_pct", "rs_rank_63d", "dist_52w_high", "rsi_14"]
    for f in top_features:
        if f in df.columns:
            col_name = f"{f}_x_regime"
            df[col_name] = df[f] * df["vol_regime"]
            interaction_features.append(col_name)

    tr_i = df[df["date"] < split_date].dropna(subset=interaction_features + ["target"])
    te_i = df[df["date"] >= split_date].dropna(subset=interaction_features + ["target"])
    results["+ Regime Interactions"] = train_and_evaluate(tr_i, te_i, interaction_features)

    # Print results
    print("\n  REGIME-ADAPTIVE COMPARISON:")
    print(f"  {'Model':<25} {'AUC':>8} {'Top Q':>9} {'Bot Q':>9} "
          f"{'Spread':>9} {'Mono':>6}")
    print(f"  {'-'*65}")
    for name, r in results.items():
        mono = "✓" if r.get("monotonic", False) else "✗"
        print(f"  {name:<25} {r['auc']:>8.4f} {r['top_q']:>+8.3f}% "
              f"{r['bot_q']:>+8.3f}% {r['spread']:>8.3f}% {mono:>6}")

    if full_wf:
        print("\n  WALK-FORWARD REGIME COMPARISON:")
        print(f"  {'Model':<25} {'Folds':>6} {'AUC':>12} {'Top20% Ret':>14} {'Spread':>14}")
        print(f"  {'-'*70}")

        for name, feat_set, tgt in [
            ("Single Model", features, "target"),
            ("+ Regime Feature", features_with_regime, "target"),
            ("+ Regime Interactions", interaction_features, "target"),
        ]:
            wf_results = walkforward_evaluate(df, feat_set, target_col=tgt)
            if wf_results:
                aucs = [r["auc"] for r in wf_results]
                rets = [r["top20_ret"] for r in wf_results]
                spreads = [r["spread"] for r in wf_results]
                print(f"  {name:<25} {len(aucs):>6} "
                      f"{np.mean(aucs):.4f}±{np.std(aucs):.3f} "
                      f"{np.mean(rets):>+.3f}±{np.std(rets):.3f}% "
                      f"{np.mean(spreads):>+.3f}±{np.std(spreads):.3f}%")

    return results


# ═══════════════════════════════════════════════════════════════════════
# 3. ENSEMBLE METHODS
# ═══════════════════════════════════════════════════════════════════════

def test_ensemble(df, features, full_wf=False):
    """Tests ensemble model approaches."""
    logger.info("\n" + "=" * 65)
    logger.info("  TEST 3: ENSEMBLE METHODS")
    logger.info("=" * 65)

    df = df.copy()
    df["target"] = (
        df.groupby("date")["fwd_5d"].rank(pct=True) >= 0.80
    ).astype(int)

    dates = sorted(df["date"].unique())
    split_date = dates[int(len(dates) * 0.75)]

    tr = df[df["date"] < split_date].dropna(subset=features + ["target"])
    te = df[df["date"] >= split_date].dropna(subset=features + ["target"])

    results = {}

    # A) Base LightGBM
    base_result = train_and_evaluate(tr, te, features)
    results["LightGBM Base"] = base_result

    # B) Simple momentum score (no ML)
    te_mom = te.copy()
    # Rank by composite of RS score + volume confirmation
    mom_features = ["rs_score", "volume_ratio", "above_sma50", "above_sma20"]
    available_mom = [f for f in mom_features if f in te.columns]
    if available_mom:
        for f in available_mom:
            te_mom[f"{f}_rank"] = te_mom.groupby("date")[f].rank(pct=True)
        rank_cols = [f"{f}_rank" for f in available_mom]
        te_mom["momentum_score"] = te_mom[rank_cols].mean(axis=1)

        # Evaluate momentum score
        te_mom["ml_score"] = te_mom["momentum_score"]
        auc_mom = roc_auc_score(te_mom["target"], te_mom["ml_score"])
        te_mom["quintile"] = pd.qcut(te_mom["ml_score"], 5, labels=[1,2,3,4,5],
                                      duplicates="drop")
        q_returns_mom = te_mom.groupby("quintile")["fwd_5d"].mean() * 100
        top_q = te_mom[te_mom["quintile"] == 5]["fwd_5d"].mean() * 100
        bot_q = te_mom[te_mom["quintile"] == 1]["fwd_5d"].mean() * 100
        q_vals = q_returns_mom.values
        monotonic = all(q_vals[i] <= q_vals[i+1] for i in range(len(q_vals)-1))
        results["Pure Momentum"] = {
            "auc": auc_mom, "top_q": top_q, "bot_q": bot_q,
            "spread": top_q - bot_q, "monotonic": monotonic,
            "q_returns": q_returns_mom,
        }

    # C) Blend: 70% LightGBM + 30% Momentum
    if "momentum_score" in te_mom.columns:
        te_blend = te.copy()
        te_blend["momentum_score"] = te_mom["momentum_score"].values
        # Normalize both to 0-1 range
        lgb_norm = (base_result["predictions"]["ml_score"] -
                    base_result["predictions"]["ml_score"].min()) / \
                   (base_result["predictions"]["ml_score"].max() -
                    base_result["predictions"]["ml_score"].min())
        mom_norm = (te_blend["momentum_score"] - te_blend["momentum_score"].min()) / \
                   (te_blend["momentum_score"].max() - te_blend["momentum_score"].min())

        for w_lgb, w_name in [(0.7, "70/30"), (0.5, "50/50"), (0.8, "80/20")]:
            w_mom = 1 - w_lgb
            te_blend["ml_score"] = w_lgb * lgb_norm.values + w_mom * mom_norm.values
            auc_blend = roc_auc_score(te_blend["target"], te_blend["ml_score"])
            te_blend["quintile"] = pd.qcut(te_blend["ml_score"], 5, labels=[1,2,3,4,5],
                                            duplicates="drop")
            q_rets = te_blend.groupby("quintile")["fwd_5d"].mean() * 100
            top_q = te_blend[te_blend["quintile"] == 5]["fwd_5d"].mean() * 100
            bot_q = te_blend[te_blend["quintile"] == 1]["fwd_5d"].mean() * 100
            q_vals = q_rets.values
            monotonic = all(q_vals[i] <= q_vals[i+1] for i in range(len(q_vals)-1))
            results[f"Blend {w_name} LGB/Mom"] = {
                "auc": auc_blend, "top_q": top_q, "bot_q": bot_q,
                "spread": top_q - bot_q, "monotonic": monotonic,
                "q_returns": q_rets,
            }

    # D) Multi-horizon ensemble: train on 3d, 5d, 10d targets and average scores
    horizon_models = {}
    for horizon, target_name in [("3d", "fwd_3d"), ("5d", "fwd_5d"), ("10d", "fwd_10d")]:
        df_h = df.copy()
        if target_name not in df_h.columns:
            continue
        df_h["target_h"] = (
            df_h.groupby("date")[target_name].rank(pct=True) >= 0.80
        ).astype(int)

        tr_h = df_h[df_h["date"] < split_date].dropna(subset=features + ["target_h"])
        te_h = df_h[df_h["date"] >= split_date].dropna(subset=features + ["target_h"])

        split = int(len(tr_h) * 0.8)
        train_data = lgb.Dataset(tr_h.iloc[:split][features], label=tr_h.iloc[:split]["target_h"])
        val_data = lgb.Dataset(tr_h.iloc[split:][features], label=tr_h.iloc[split:]["target_h"],
                                reference=train_data)

        model = lgb.train(LGB_PARAMS, train_data, num_boost_round=500,
                           valid_sets=[val_data],
                           callbacks=[lgb.early_stopping(30, verbose=False),
                                      lgb.log_evaluation(-1)])

        horizon_models[horizon] = model.predict(te[features])

    if len(horizon_models) == 3:
        te_multi = te.copy()
        te_multi["ml_score"] = (horizon_models["3d"] + horizon_models["5d"] +
                                 horizon_models["10d"]) / 3
        auc_multi = roc_auc_score(te_multi["target"], te_multi["ml_score"])
        te_multi["quintile"] = pd.qcut(te_multi["ml_score"], 5, labels=[1,2,3,4,5],
                                        duplicates="drop")
        q_rets = te_multi.groupby("quintile")["fwd_5d"].mean() * 100
        top_q = te_multi[te_multi["quintile"] == 5]["fwd_5d"].mean() * 100
        bot_q = te_multi[te_multi["quintile"] == 1]["fwd_5d"].mean() * 100
        q_vals = q_rets.values
        monotonic = all(q_vals[i] <= q_vals[i+1] for i in range(len(q_vals)-1))
        results["Multi-Horizon (3/5/10d)"] = {
            "auc": auc_multi, "top_q": top_q, "bot_q": bot_q,
            "spread": top_q - bot_q, "monotonic": monotonic,
            "q_returns": q_rets,
        }

    # E) Different LGB hyperparameters — more aggressive
    params_aggressive = LGB_PARAMS.copy()
    params_aggressive.update({
        "learning_rate": 0.05, "num_leaves": 63,
        "min_child_samples": 80, "feature_fraction": 0.8,
        "max_depth": 8,
    })
    results["LGB Aggressive Params"] = train_and_evaluate(tr, te, features, params_aggressive)

    # F) Conservative params — more regularized
    params_conservative = LGB_PARAMS.copy()
    params_conservative.update({
        "learning_rate": 0.01, "num_leaves": 15,
        "min_child_samples": 300, "feature_fraction": 0.5,
        "max_depth": 4, "reg_alpha": 1.0, "reg_lambda": 1.0,
    })
    results["LGB Conservative Params"] = train_and_evaluate(tr, te, features, params_conservative)

    # Print results
    print("\n  ENSEMBLE COMPARISON:")
    print(f"  {'Model':<25} {'AUC':>8} {'Top Q':>9} {'Bot Q':>9} "
          f"{'Spread':>9} {'Mono':>6}")
    print(f"  {'-'*65}")
    for name, r in results.items():
        mono = "✓" if r.get("monotonic", False) else "✗"
        print(f"  {name:<25} {r['auc']:>8.4f} {r['top_q']:>+8.3f}% "
              f"{r['bot_q']:>+8.3f}% {r['spread']:>8.3f}% {mono:>6}")

    if full_wf:
        print("\n  WALK-FORWARD ENSEMBLE COMPARISON:")
        print(f"  {'Model':<25} {'Folds':>6} {'AUC':>12} {'Top20% Ret':>14} {'Spread':>14}")
        print(f"  {'-'*70}")

        for name, params_variant in [
            ("LGB Base", LGB_PARAMS),
            ("LGB Aggressive", params_aggressive),
            ("LGB Conservative", params_conservative),
        ]:
            wf_results = walkforward_evaluate(df, features, params=params_variant)
            if wf_results:
                aucs = [r["auc"] for r in wf_results]
                rets = [r["top20_ret"] for r in wf_results]
                spreads = [r["spread"] for r in wf_results]
                print(f"  {name:<25} {len(aucs):>6} "
                      f"{np.mean(aucs):.4f}±{np.std(aucs):.3f} "
                      f"{np.mean(rets):>+.3f}±{np.std(rets):.3f}% "
                      f"{np.mean(spreads):>+.3f}±{np.std(spreads):.3f}%")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", action="store_true", help="Target engineering only")
    parser.add_argument("--regime", action="store_true", help="Regime-adaptive only")
    parser.add_argument("--ensemble", action="store_true", help="Ensemble only")
    parser.add_argument("--full", action="store_true", help="Full walk-forward for all")
    args = parser.parse_args()

    run_all = not (args.target or args.regime or args.ensemble)

    df, features = load_and_prepare()

    if args.target or run_all:
        test_target_engineering(df, features, full_wf=args.full)

    if args.regime or run_all:
        test_regime_adaptive(df, features, full_wf=args.full)

    if args.ensemble or run_all:
        test_ensemble(df, features, full_wf=args.full)

    print("\n" + "=" * 65)
    print("  RESEARCH COMPLETE")
    print("=" * 65)
    print("\nTo run full walk-forward validation on all tests:")
    print("  python research/model_improvements.py --full")
