"""
research/integrate_improvements.py
-----------------------------------
Integrates the two validated model improvements:

1. Conservative LightGBM hyperparameters (AUC 0.5951, +0.422% spread)
2. 3-day forward return target (AUC 0.5899, +0.423% spread)

This script:
  A) Tests the COMBINED effect (both changes together)
  B) If results are good, generates updated config and model files
  C) Runs a full walk-forward backtest with the new settings
  D) Compares equity curves: old vs new

Usage:
    python research/integrate_improvements.py           # Test combined effect
    python research/integrate_improvements.py --apply   # Apply changes to live system
"""

import sys
import yaml
import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from pathlib import Path
from loguru import logger
from datetime import datetime

sys.path.insert(0, ".")


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

# Current params
OLD_PARAMS = {
    "objective": "binary", "metric": "auc",
    "learning_rate": 0.03, "num_leaves": 31,
    "min_child_samples": 150, "feature_fraction": 0.65,
    "bagging_fraction": 0.75, "bagging_freq": 5,
    "reg_alpha": 0.3, "reg_lambda": 0.3,
    "max_depth": 6, "verbose": -1, "n_jobs": -1,
}

# New conservative params (validated in walk-forward)
NEW_PARAMS = {
    "objective": "binary", "metric": "auc",
    "learning_rate": 0.01, "num_leaves": 15,
    "min_child_samples": 300, "feature_fraction": 0.5,
    "bagging_fraction": 0.75, "bagging_freq": 5,
    "reg_alpha": 1.0, "reg_lambda": 1.0,
    "max_depth": 4, "verbose": -1, "n_jobs": -1,
}

# New hold days (3-day target means shorter holds)
NEW_HOLD_DAYS = 5  # keep 5 for live trading (3d is the prediction target, not the hold)
# The model predicts best-performing stocks over NEXT 3 DAYS
# But we still hold up to 5 days to let the trade play out (ATR stops/targets handle exits)
# The key change is WHAT the model optimizes for, not how long we hold

OLD_HOLD_DAYS = 7
OLD_EARLY_EXIT_DAYS = 4
NEW_EARLY_EXIT_DAYS = 3  # tighter since we're optimizing for shorter window


# ═══════════════════════════════════════════════════════════════════════
# Step 1: Walk-Forward Comparison (Old vs New Combined)
# ═══════════════════════════════════════════════════════════════════════

def run_comparison():
    """Run walk-forward comparing old model vs new (3d target + conservative params)."""
    from data.pipeline import load_processed
    from models.train import build_features

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    try:
        df = load_processed("price_features_enriched.parquet")
    except FileNotFoundError:
        df = load_processed("price_features.parquet")

    df, feature_cols = build_features(df)

    # Build multiple targets
    df["fwd_3d"] = df.groupby("ticker")["close"].transform(lambda x: x.shift(-3) / x - 1)
    df["fwd_5d"] = df.groupby("ticker")["close"].transform(lambda x: x.shift(-5) / x - 1)

    df["target_old"] = (df.groupby("date")["fwd_5d"].rank(pct=True) >= 0.80).astype(int)
    df["target_new"] = (df.groupby("date")["fwd_3d"].rank(pct=True) >= 0.80).astype(int)

    vc = config["validation"]
    dates = pd.Series(sorted(df["date"].unique())).reset_index(drop=True)

    configs = {
        "Current (5d target, standard params)": {
            "params": OLD_PARAMS, "target": "target_old",
        },
        "3d target + standard params": {
            "params": OLD_PARAMS, "target": "target_new",
        },
        "5d target + conservative params": {
            "params": NEW_PARAMS, "target": "target_old",
        },
        "COMBINED (3d + conservative)": {
            "params": NEW_PARAMS, "target": "target_new",
        },
    }

    all_results = {name: [] for name in configs}
    all_predictions = {name: [] for name in configs}

    train_start = dates.iloc[0]
    fold = 0

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

        fold += 1

        for name, cfg in configs.items():
            target_col = cfg["target"]
            params = cfg["params"]

            tr = df[df["date"].isin(train_dates)].dropna(subset=feature_cols + [target_col])
            te = df[df["date"].isin(test_dates)].dropna(subset=feature_cols + [target_col])

            if len(tr) < 500 or len(te) < 50:
                continue

            split = int(len(tr) * 0.8)
            train_data = lgb.Dataset(tr.iloc[:split][feature_cols], label=tr.iloc[:split][target_col])
            val_data = lgb.Dataset(tr.iloc[split:][feature_cols], label=tr.iloc[split:][target_col],
                                    reference=train_data)

            model = lgb.train(params, train_data, num_boost_round=800,
                               valid_sets=[val_data],
                               callbacks=[lgb.early_stopping(40, verbose=False),
                                          lgb.log_evaluation(-1)])

            te = te.copy()
            te["ml_score"] = model.predict(te[feature_cols])
            auc = roc_auc_score(te[target_col], te["ml_score"])

            top20 = te.nlargest(max(1, int(len(te) * 0.2)), "ml_score")
            top20_r = top20["fwd_5d"].mean() * 100
            bot20 = te.nsmallest(max(1, int(len(te) * 0.2)), "ml_score")
            bot20_r = bot20["fwd_5d"].mean() * 100

            all_results[name].append({
                "fold": fold, "auc": auc,
                "top20_ret": top20_r, "bot20_ret": bot20_r,
                "spread": top20_r - bot20_r,
            })

            all_predictions[name].append(te[["date", "ticker", "ml_score", "fwd_5d"]])

        train_start += pd.DateOffset(months=vc["test_months"])

    # Print comparison
    print("\n" + "=" * 80)
    print("  COMBINED WALK-FORWARD COMPARISON")
    print("=" * 80)
    print(f"  {'Model':<40} {'Folds':>6} {'AUC':>12} {'Top20%':>12} {'Spread':>12}")
    print(f"  {'-'*78}")

    for name in configs:
        if all_results[name]:
            aucs = [r["auc"] for r in all_results[name]]
            rets = [r["top20_ret"] for r in all_results[name]]
            spreads = [r["spread"] for r in all_results[name]]
            print(f"  {name:<40} {len(aucs):>6} "
                  f"{np.mean(aucs):.4f}±{np.std(aucs):.3f} "
                  f"{np.mean(rets):>+.3f}±{np.std(rets):.3f}% "
                  f"{np.mean(spreads):>+.3f}±{np.std(spreads):.3f}%")

    print("=" * 80)

    # Per-fold comparison: Current vs Combined
    if all_results["Current (5d target, standard params)"] and all_results["COMBINED (3d + conservative)"]:
        old = all_results["Current (5d target, standard params)"]
        new = all_results["COMBINED (3d + conservative)"]
        n = min(len(old), len(new))
        wins = sum(1 for i in range(n) if new[i]["spread"] > old[i]["spread"])
        avg_delta_auc = np.mean([new[i]["auc"] - old[i]["auc"] for i in range(n)])
        avg_delta_spread = np.mean([new[i]["spread"] - old[i]["spread"] for i in range(n)])

        print(f"\n  COMBINED vs CURRENT:")
        print(f"    Avg AUC improvement:    {avg_delta_auc:+.4f}")
        print(f"    Avg spread improvement: {avg_delta_spread:+.3f}%")
        print(f"    Wins {wins}/{n} folds ({wins/n*100:.0f}%)")

    return all_results, all_predictions, df, feature_cols


# ═══════════════════════════════════════════════════════════════════════
# Step 2: Full Backtest with New Settings
# ═══════════════════════════════════════════════════════════════════════

def run_full_backtest(df, feature_cols, old_predictions, new_predictions):
    """Runs a side-by-side backtest simulation using OOS predictions."""
    from backtest.backtest import SwingBacktester

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    # Run backtest with old settings
    old_config = config.copy()
    old_config["backtest"] = dict(config["backtest"])

    # Run backtest with new settings
    new_config = config.copy()
    new_config["backtest"] = dict(config["backtest"])
    new_config["backtest"]["hold_days"] = NEW_HOLD_DAYS
    new_config["backtest"]["early_exit_days"] = NEW_EARLY_EXIT_DAYS

    # Merge OOS predictions into df for both
    old_preds_df = pd.concat(old_predictions, ignore_index=True)
    new_preds_df = pd.concat(new_predictions, ignore_index=True)

    results = {}
    for name, preds_df, bt_config in [
        ("Current System", old_preds_df, old_config),
        ("Improved System", new_preds_df, new_config),
    ]:
        merged = df.merge(
            preds_df[["date", "ticker", "ml_score"]],
            on=["date", "ticker"], how="left", suffixes=("", "_oos")
        )
        if "ml_score_oos" in merged.columns:
            merged["ml_score"] = merged["ml_score_oos"]
            merged.drop(columns=["ml_score_oos"], inplace=True)
        merged["ml_score"] = merged["ml_score"].fillna(0)
        merged["ml_rank"] = merged.groupby("date")["ml_score"].rank(pct=True)

        # Regime filter
        from signals.signals import compute_market_regime
        regime = compute_market_regime(df)
        merged = merged.merge(regime, on="date", how="left", suffixes=("", "_r"))
        if "regime_ok_r" in merged.columns:
            merged["regime_ok"] = merged["regime_ok_r"]
        merged["signal"] = ((merged["ml_rank"] >= 0.80) &
                             (merged["regime_ok"] == 1)).astype(int)

        bt = SwingBacktester(bt_config)
        bt_result = bt.run(merged)
        results[name] = bt_result

        stats = bt_result["stats"]
        print(f"\n  {name} Backtest:")
        print(f"    Total Return:   {stats.get('total_return', 0)*100:+.1f}%")
        print(f"    Sharpe Ratio:   {stats.get('sharpe', 0):.2f}")
        print(f"    Max Drawdown:   {stats.get('max_drawdown', 0)*100:.1f}%")
        print(f"    Win Rate:       {stats.get('win_rate', 0)*100:.1f}%")
        print(f"    Total Trades:   {stats.get('n_trades', 0)}")
        print(f"    Profit Factor:  {stats.get('profit_factor', 0):.2f}")
        if "avg_hold" in stats:
            print(f"    Avg Hold Days:  {stats['avg_hold']:.1f}")

    # Plot comparison
    _plot_equity_comparison(results)

    return results


def _plot_equity_comparison(results):
    """Plot equity curves side by side."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="gray")
    ax.spines[:].set_color("#333")

    colors = {"Current System": "#4a9eff", "Improved System": "#00e676"}

    for name, res in results.items():
        eq = res["equity_curve"]
        if isinstance(eq, list):
            eq_df = pd.DataFrame(eq)
        else:
            eq_df = eq

        if "date" in eq_df.columns:
            ax.plot(eq_df["date"], eq_df["equity"], label=name,
                    color=colors.get(name, "#fff"), linewidth=1.5)
        elif "equity" in eq_df.columns:
            ax.plot(eq_df["equity"].values, label=name,
                    color=colors.get(name, "#fff"), linewidth=1.5)

    ax.set_xlabel("Date", color="gray")
    ax.set_ylabel("Equity ($)", color="gray")
    ax.set_title("Backtest Equity Curve: Current vs Improved", color="white", fontweight="bold")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")

    plt.tight_layout()
    save_path = "research/model_improvement_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    logger.info(f"Equity comparison chart saved to {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# Step 3: Apply Changes to Live System
# ═══════════════════════════════════════════════════════════════════════

def apply_changes():
    """
    Applies the validated improvements to the live system:
    1. Updates config.yaml with new hold_days and early_exit_days
    2. Retrains model with new params and 3-day target
    3. Saves new model as lgbm_model.pkl (backs up old one)
    """
    from data.pipeline import load_processed
    from models.train import build_features
    from signals.signals import compute_market_regime

    logger.info("=" * 60)
    logger.info("  APPLYING MODEL IMPROVEMENTS TO LIVE SYSTEM")
    logger.info("=" * 60)

    # ── 1. Backup current model ──────────────────────────────────────
    model_path = Path("models/lgbm_model.pkl")
    if model_path.exists():
        backup_name = f"models/lgbm_model_backup_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
        shutil.copy(model_path, backup_name)
        logger.info(f"Backed up current model to {backup_name}")

    # ── 2. Update config.yaml ────────────────────────────────────────
    config_path = Path("config/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    old_hold = config["backtest"]["hold_days"]
    old_early = config["backtest"].get("early_exit_days", 4)

    config["backtest"]["hold_days"] = NEW_HOLD_DAYS
    config["backtest"]["early_exit_days"] = NEW_EARLY_EXIT_DAYS

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Updated config: hold_days {old_hold} → {NEW_HOLD_DAYS}, "
                f"early_exit_days {old_early} → {NEW_EARLY_EXIT_DAYS}")

    # ── 3. Retrain model with new params and 3-day target ────────────
    try:
        df = load_processed("price_features_enriched.parquet")
    except FileNotFoundError:
        df = load_processed("price_features.parquet")

    regime = compute_market_regime(df)
    df = df.merge(regime, on="date", how="left")
    df, feature_cols = build_features(df)

    # Build 3-day target
    df["fwd_3d"] = df.groupby("ticker")["close"].transform(lambda x: x.shift(-3) / x - 1)
    df["target"] = (df.groupby("date")["fwd_3d"].rank(pct=True) >= 0.80).astype(int)

    # Also keep fwd_5d for evaluation
    if "fwd_5d" not in df.columns:
        df["fwd_5d"] = df.groupby("ticker")["close"].transform(lambda x: x.shift(-5) / x - 1)

    logger.info(f"Training with conservative params + 3-day target on {len(df):,} rows...")

    # Walk-forward with new settings
    vc = config["validation"]
    dates = pd.Series(sorted(df["date"].unique())).reset_index(drop=True)
    all_preds = []
    fold_stats = []
    last_model = None
    fold = 0

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

        tr = df[df["date"].isin(train_dates)].dropna(subset=feature_cols + ["target"])
        te = df[df["date"].isin(test_dates)].dropna(subset=feature_cols)

        if len(tr) < 500 or len(te) < 50:
            train_start += pd.DateOffset(months=vc["test_months"])
            continue

        split = int(len(tr) * 0.80)
        train_data = lgb.Dataset(tr.iloc[:split][feature_cols], label=tr.iloc[:split]["target"])
        val_data = lgb.Dataset(tr.iloc[split:][feature_cols], label=tr.iloc[split:]["target"],
                                reference=train_data)

        model = lgb.train(
            NEW_PARAMS, train_data, num_boost_round=800,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)]
        )

        te = te.copy()
        te["ml_score"] = model.predict(te[feature_cols])
        all_preds.append(te[["date", "ticker", "ml_score", "fwd_5d", "target"]])

        auc = roc_auc_score(te["target"].fillna(0), te["ml_score"])
        top20 = te.nlargest(max(1, int(len(te) * 0.20)), "ml_score")
        top20_r = top20["fwd_5d"].mean() * 100

        logger.info(f"  Fold {fold+1}: AUC={auc:.3f} | Top20%={top20_r:+.3f}%")
        fold_stats.append({"fold": fold+1, "auc": auc, "top20_ret": top20_r})
        last_model = model
        fold += 1
        train_start += pd.DateOffset(months=vc["test_months"])

    # Save new model
    model_data = {
        "model": last_model,
        "features": feature_cols,
        "target_horizon": "3d",
        "params": NEW_PARAMS,
        "trained_at": datetime.now().isoformat(),
        "n_folds": fold,
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    fold_df = pd.DataFrame(fold_stats)
    avg_auc = fold_df["auc"].mean()
    avg_ret = fold_df["top20_ret"].mean()

    logger.info(f"\nNew model saved: {fold} folds")
    logger.info(f"  Avg AUC:       {avg_auc:.4f}")
    logger.info(f"  Avg Top20% ret: {avg_ret:+.3f}%")

    # ── 4. Update train.py params ────────────────────────────────────
    # Write a model metadata file for reference
    meta = {
        "version": "v3_improved",
        "changes": [
            "Conservative LightGBM params (LR=0.01, leaves=15, depth=4)",
            "3-day forward return target (was 5-day)",
            "Shorter early exit window (3 days vs 4)",
            "Hold days reduced to 5 (was 7)",
        ],
        "walk_forward_results": {
            "avg_auc": float(avg_auc),
            "avg_top20_ret": float(avg_ret),
            "n_folds": fold,
        },
        "old_settings": {
            "params": OLD_PARAMS,
            "hold_days": OLD_HOLD_DAYS,
            "early_exit_days": OLD_EARLY_EXIT_DAYS,
            "target": "5-day forward return, top 20%",
        },
        "new_settings": {
            "params": NEW_PARAMS,
            "hold_days": NEW_HOLD_DAYS,
            "early_exit_days": NEW_EARLY_EXIT_DAYS,
            "target": "3-day forward return, top 20%",
        },
        "applied_at": datetime.now().isoformat(),
    }
    with open("models/model_v3_metadata.yaml", "w") as f:
        yaml.dump(meta, f, default_flow_style=False)

    print("\n" + "=" * 60)
    print("  CHANGES APPLIED SUCCESSFULLY")
    print("=" * 60)
    print(f"\n  Model:   models/lgbm_model.pkl (v3 improved)")
    print(f"  Backup:  {backup_name}")
    print(f"  Config:  hold_days={NEW_HOLD_DAYS}, early_exit={NEW_EARLY_EXIT_DAYS}")
    print(f"  Target:  3-day forward return, top 20%")
    print(f"  Params:  conservative (LR=0.01, leaves=15, depth=4)")
    print(f"\n  Walk-Forward Results:")
    print(f"    Avg AUC:       {avg_auc:.4f}")
    print(f"    Avg Top20% ret: {avg_ret:+.3f}%")
    print(f"    Folds:         {fold}")
    print("=" * 60)

    print("\n  IMPORTANT: Also update models/train.py to use new params")
    print("  for future retraining. The retrain module will use the saved")
    print("  model's params automatically, but manual retrains should use:")
    print(f"    learning_rate: {NEW_PARAMS['learning_rate']}")
    print(f"    num_leaves:    {NEW_PARAMS['num_leaves']}")
    print(f"    max_depth:     {NEW_PARAMS['max_depth']}")
    print(f"    min_child:     {NEW_PARAMS['min_child_samples']}")
    print(f"    reg_alpha:     {NEW_PARAMS['reg_alpha']}")
    print(f"    reg_lambda:    {NEW_PARAMS['reg_lambda']}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="Apply changes to live system (retrain + update config)")
    args = parser.parse_args()

    if args.apply:
        apply_changes()
    else:
        # Run comparison first
        results, predictions, df, features = run_comparison()

        # Run backtest comparison if we have predictions
        old_key = "Current (5d target, standard params)"
        new_key = "COMBINED (3d + conservative)"
        if predictions.get(old_key) and predictions.get(new_key):
            run_full_backtest(df, features, predictions[old_key], predictions[new_key])

        print("\nTo apply these changes to the live system:")
        print("  python research/integrate_improvements.py --apply")
