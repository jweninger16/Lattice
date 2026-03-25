"""
live/retrain.py
---------------
Automated monthly retraining pipeline.

Runs on the 1st Sunday of each month:
1. Downloads latest price data
2. Refreshes sector + earnings features
3. Retrains LightGBM model with fresh data
4. Compares new model AUC vs current model
5. Swaps in new model only if it's better (or within tolerance)
6. Sends SMS with retraining results
7. Logs everything for review

Safety checks:
- Never swaps in a model with AUC < 0.55
- Never swaps if new model is more than 0.02 worse than current
- Keeps last 3 model versions as backups
"""

import sys
import yaml
import pickle
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date
from loguru import logger

sys.path.insert(0, ".")

MODEL_DIR    = Path("models")
LOG_DIR      = Path("logs")
RETRAIN_LOG  = LOG_DIR / "retrain_history.csv"
MIN_AUC      = 0.55      # Never deploy a model below this
MAX_DEGRADATION = 0.02   # Never deploy if more than 2% worse than current


def get_current_model_auc() -> float:
    """Gets the AUC of the currently deployed model from retrain log."""
    if not RETRAIN_LOG.exists():
        return 0.55  # Default if no history
    log = pd.read_csv(RETRAIN_LOG)
    if len(log) == 0:
        return 0.55
    return float(log.iloc[-1]["new_auc"])


def backup_current_model():
    """Keeps last 3 model versions."""
    model_path = MODEL_DIR / "lgbm_model.pkl"
    if not model_path.exists():
        return

    # Rotate backups
    for i in range(2, 0, -1):
        src  = MODEL_DIR / f"lgbm_model_backup_{i}.pkl"
        dst  = MODEL_DIR / f"lgbm_model_backup_{i+1}.pkl"
        if src.exists():
            shutil.copy(src, dst)

    shutil.copy(model_path, MODEL_DIR / "lgbm_model_backup_1.pkl")
    logger.info("Current model backed up")


def log_retrain(new_auc: float, current_auc: float, deployed: bool,
                n_folds: int, reason: str):
    """Logs retraining results."""
    LOG_DIR.mkdir(exist_ok=True)
    row = pd.DataFrame([{
        "date":         str(date.today()),
        "new_auc":      new_auc,
        "current_auc":  current_auc,
        "deployed":     deployed,
        "n_folds":      n_folds,
        "reason":       reason,
    }])
    if RETRAIN_LOG.exists():
        log = pd.read_csv(RETRAIN_LOG)
        log = pd.concat([log, row], ignore_index=True)
    else:
        log = row
    log.to_csv(RETRAIN_LOG, index=False)


def run_retrain(force: bool = False, notify: bool = True) -> dict:
    """
    Full retraining pipeline.
    force=True skips the AUC comparison and always deploys.
    """
    start_time = datetime.now()
    logger.info("="*50)
    logger.info(f"MONTHLY RETRAIN — {date.today()}")
    logger.info("="*50)

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    # ── 1. Download fresh data ──────────────────────────────────────────
    logger.info("Step 1/5: Downloading fresh market data...")
    from data.universe import build_universe
    from data.pipeline import run_pipeline, save_processed
    universe = build_universe(config)
    df = run_pipeline(config, universe)
    logger.info(f"Data: {len(df):,} rows, {df['date'].max().date()} latest")

    # ── 2. Enrich with sector + earnings ───────────────────────────────
    logger.info("Step 2/5: Adding sector and earnings features...")
    from data.sectors import add_sector_features
    from data.earnings import fetch_earnings_dates, add_earnings_features

    start = df["date"].min().strftime("%Y-%m-%d")
    end   = df["date"].max().strftime("%Y-%m-%d")
    df = add_sector_features(df, start, end)

    tickers = df["ticker"].unique().tolist()
    earnings_map = fetch_earnings_dates(tickers, use_cache=False)  # Fresh data
    df = add_earnings_features(df, earnings_map)

    save_processed(df, "price_features_enriched.parquet")
    logger.info(f"Enriched dataset: {df.shape[1]} columns")

    # ── 3. Train new model ──────────────────────────────────────────────
    logger.info("Step 3/5: Training new model...")
    from signals.signals import compute_market_regime
    from models.train import build_features, build_target, walk_forward_train
    from models.train import evaluate_ml_signal

    regime = compute_market_regime(df)
    df = df.merge(regime, on="date", how="left")
    df, feature_cols = build_features(df)
    df = build_target(df)

    results = walk_forward_train(df, config, feature_cols)
    new_auc = float(results["fold_stats"]["auc"].mean())
    n_folds = len(results["fold_stats"])

    logger.info(f"New model: AUC={new_auc:.3f} across {n_folds} folds")
    evaluate_ml_signal(results["oos_predictions"])

    # ── 4. Compare vs current model ────────────────────────────────────
    logger.info("Step 4/5: Comparing vs current model...")
    current_auc = get_current_model_auc()
    auc_delta   = new_auc - current_auc

    deploy  = False
    reason  = ""

    if new_auc < MIN_AUC:
        reason = f"New AUC {new_auc:.3f} below minimum {MIN_AUC}"
        logger.warning(f"NOT deploying: {reason}")
    elif auc_delta < -MAX_DEGRADATION and not force:
        reason = f"New AUC {new_auc:.3f} is {abs(auc_delta):.3f} worse than current {current_auc:.3f}"
        logger.warning(f"NOT deploying: {reason}")
    else:
        deploy = True
        reason = f"New AUC {new_auc:.3f} vs current {current_auc:.3f} ({auc_delta:+.3f})"
        logger.info(f"Deploying new model: {reason}")

    # ── 5. Deploy if approved ───────────────────────────────────────────
    if deploy:
        logger.info("Step 5/5: Deploying new model...")
        backup_current_model()
        new_model_path = MODEL_DIR / "lgbm_model_new.pkl"
        with open(new_model_path, "wb") as f:
            pickle.dump({"model": results["model"], "features": feature_cols}, f)
        shutil.copy(new_model_path, MODEL_DIR / "lgbm_model.pkl")
        logger.info("New model deployed successfully")
    else:
        logger.info("Step 5/5: Keeping current model")

    # Log results
    log_retrain(new_auc, current_auc, deploy, n_folds, reason)

    elapsed = (datetime.now() - start_time).seconds // 60
    summary = {
        "new_auc":      new_auc,
        "current_auc":  current_auc,
        "auc_delta":    auc_delta,
        "deployed":     deploy,
        "reason":       reason,
        "elapsed_min":  elapsed,
        "n_folds":      n_folds,
    }

    # ── 6. Send SMS notification ────────────────────────────────────────
    if notify:
        try:
            from live.alerts import send_sms
            status = "DEPLOYED" if deploy else "KEPT OLD"
            msg = (f"SWING TRADER — Monthly Retrain\n"
                   f"New AUC: {new_auc:.3f} ({auc_delta:+.3f})\n"
                   f"Model: {status}\n"
                   f"Folds: {n_folds} | Time: {elapsed}min\n"
                   f"{reason[:80]}")
            send_sms(msg)
            logger.info("Retrain notification sent")
        except Exception as e:
            logger.warning(f"SMS notification failed: {e}")

    logger.info(f"Retrain complete in {elapsed} minutes")
    return summary


def should_retrain_today() -> bool:
    """Returns True if today is the 1st Monday of the month."""
    today = date.today()
    if today.weekday() != 0:  # Not Monday
        return False
    # Is it the first Monday? (day <= 7)
    return today.day <= 7


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Force retrain even if AUC is worse")
    parser.add_argument("--check", action="store_true",
                        help="Just check if retrain is due today")
    args = parser.parse_args()

    if args.check:
        due = should_retrain_today()
        print(f"Retrain due today: {due}")
        if RETRAIN_LOG.exists():
            log = pd.read_csv(RETRAIN_LOG)
            print(f"Last retrain: {log.iloc[-1]['date']} | AUC: {log.iloc[-1]['new_auc']:.3f}")
    else:
        results = run_retrain(force=args.force)
        print(f"\nRetrain complete:")
        for k, v in results.items():
            print(f"  {k}: {v}")
