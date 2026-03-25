"""
research/cross_asset_features.py
---------------------------------
STANDALONE RESEARCH — Tests whether cross-asset features improve the ML model.

Downloads and engineers features from:
  1. Credit/Bond market: HYG (high yield), TLT (long bonds), IEF (intermediate)
  2. Commodities: GLD (gold), COPX (copper miners as copper proxy)
  3. Volatility structure: VIX, VIX3M (via ^VIX and ^VIX3M)
  4. Options sentiment: Put/Call ratio approximated from vol term structure
  5. Calendar/behavioral: month-end, OpEx week, day-of-week, January effect
  6. Sector ETF flow signals: XLK, XLF, XLE, XLV volume anomalies

Compares:
  - Current model (existing features only)
  - Enhanced model (existing + cross-asset features)
  - Feature importance analysis

Usage:
    python research/cross_asset_features.py
    python research/cross_asset_features.py --full   # Run full walk-forward comparison
"""

import sys
import yaml
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from datetime import timedelta

sys.path.insert(0, ".")


# ═══════════════════════════════════════════════════════════════════════
# Cross-Asset Data Download
# ═══════════════════════════════════════════════════════════════════════

CROSS_ASSET_TICKERS = {
    # Credit / Bonds
    "HYG": "high_yield",       # High yield corporate bonds
    "TLT": "treasury_long",    # 20+ year treasuries
    "IEF": "treasury_mid",     # 7-10 year treasuries

    # Commodities
    "GLD": "gold",
    # Use COPX (copper miners ETF) as copper proxy since futures aren't in yfinance
    "COPX": "copper",

    # Equity benchmarks (for relative signals)
    "SPY": "spy",
    "QQQ": "qqq",
    "IWM": "russell2000",     # Small caps — risk appetite signal

    # Sector ETFs (for flow signals)
    "XLK": "tech",
    "XLF": "financials",
    "XLE": "energy",
    "XLV": "healthcare",
    "XLI": "industrials",
}


def download_cross_asset_data(start_date: str = "2017-01-01") -> pd.DataFrame:
    """Downloads daily price data for all cross-asset instruments."""
    logger.info("Downloading cross-asset data...")

    tickers = list(CROSS_ASSET_TICKERS.keys())
    raw = yf.download(tickers, start=start_date, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
        volume = raw["Volume"]
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
        volume = raw[["Volume"]].rename(columns={"Volume": tickers[0]})

    close.index = pd.to_datetime(close.index).tz_localize(None)
    volume.index = pd.to_datetime(volume.index).tz_localize(None)

    logger.info(f"Cross-asset data: {len(close)} days, {len(close.columns)} instruments")
    return close, volume


def download_vix_term_structure(start_date: str = "2017-01-01") -> pd.DataFrame:
    """Downloads VIX and VIX3M for term structure analysis."""
    logger.info("Downloading VIX term structure...")
    try:
        vix_data = yf.download(["^VIX", "^VIX3M"], start=start_date,
                                auto_adjust=True, progress=False)
        if isinstance(vix_data.columns, pd.MultiIndex):
            vix_close = vix_data["Close"]
        else:
            vix_close = vix_data[["Close"]]

        vix_close.index = pd.to_datetime(vix_close.index).tz_localize(None)

        result = pd.DataFrame(index=vix_close.index)
        if "^VIX" in vix_close.columns:
            result["vix"] = vix_close["^VIX"]
        if "^VIX3M" in vix_close.columns:
            result["vix3m"] = vix_close["^VIX3M"]

        logger.info(f"VIX term structure: {len(result)} days")
        return result
    except Exception as e:
        logger.warning(f"VIX term structure download failed: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════
# Feature Engineering — Cross-Asset
# ═══════════════════════════════════════════════════════════════════════

def engineer_cross_asset_features(close: pd.DataFrame, volume: pd.DataFrame,
                                    vix_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers cross-asset features from raw price/volume data.
    Returns a DataFrame indexed by date with one row per trading day.
    """
    logger.info("Engineering cross-asset features...")
    features = pd.DataFrame(index=close.index)

    # ── 1. Credit Spread Signal ─────────────────────────────────────
    # HYG/SPY divergence: when credit weakens vs equities, trouble ahead
    if "HYG" in close.columns and "SPY" in close.columns:
        hyg_ret5 = close["HYG"].pct_change(5)
        spy_ret5 = close["SPY"].pct_change(5)
        features["credit_equity_div"] = hyg_ret5 - spy_ret5  # negative = credit weaker

        # HYG momentum (5-day) — credit market trend
        features["hyg_momentum_5d"] = hyg_ret5
        features["hyg_momentum_21d"] = close["HYG"].pct_change(21)

        # HYG relative to its own SMA — credit health
        hyg_sma50 = close["HYG"].rolling(50).mean()
        features["hyg_above_sma50"] = (close["HYG"] > hyg_sma50).astype(int)

    # ── 2. Treasury Signals ──────────────────────────────────────────
    # Yield curve proxy: TLT vs IEF relative performance
    if "TLT" in close.columns and "IEF" in close.columns:
        tlt_ief_ratio = close["TLT"] / close["IEF"]
        features["yield_curve_slope"] = tlt_ief_ratio.pct_change(21)

        # Flight to safety: TLT momentum (bonds rallying = fear)
        features["tlt_momentum_5d"] = close["TLT"].pct_change(5)
        features["tlt_momentum_21d"] = close["TLT"].pct_change(21)

    # ── 3. Commodity Signals ─────────────────────────────────────────
    # Copper/Gold ratio — economic health indicator
    if "GLD" in close.columns and "COPX" in close.columns:
        copper_gold = close["COPX"] / close["GLD"]
        features["copper_gold_ratio"] = copper_gold
        features["copper_gold_momentum"] = copper_gold.pct_change(21)

        # Gold momentum — fear/inflation signal
        features["gold_momentum_21d"] = close["GLD"].pct_change(21)

    # ── 4. Risk Appetite Signals ─────────────────────────────────────
    # IWM/SPY ratio: small caps outperforming = risk-on
    if "IWM" in close.columns and "SPY" in close.columns:
        risk_appetite = close["IWM"] / close["SPY"]
        features["risk_appetite_5d"] = risk_appetite.pct_change(5)
        features["risk_appetite_21d"] = risk_appetite.pct_change(21)

        # IWM relative strength
        iwm_sma50 = close["IWM"].rolling(50).mean()
        features["iwm_above_sma50"] = (close["IWM"] > iwm_sma50).astype(int)

    # ── 5. Volatility Term Structure ─────────────────────────────────
    if not vix_df.empty and "vix" in vix_df.columns:
        # Align VIX data with close dates
        vix_aligned = vix_df.reindex(close.index, method="ffill")

        features["vix_level"] = vix_aligned["vix"]
        features["vix_5d_change"] = vix_aligned["vix"].pct_change(5)
        features["vix_21d_change"] = vix_aligned["vix"].pct_change(21)

        # VIX mean reversion signal: distance from 21-day mean
        vix_sma21 = vix_aligned["vix"].rolling(21).mean()
        features["vix_zscore"] = (vix_aligned["vix"] - vix_sma21) / \
                                  vix_aligned["vix"].rolling(21).std()

        if "vix3m" in vix_aligned.columns:
            # Term structure: VIX / VIX3M
            # < 1 = contango (normal), > 1 = backwardation (fear)
            features["vix_term_structure"] = vix_aligned["vix"] / \
                                              vix_aligned["vix3m"].replace(0, np.nan)
            features["vix_contango"] = (features["vix_term_structure"] < 0.95).astype(int)
            features["vix_backwardation"] = (features["vix_term_structure"] > 1.05).astype(int)

    # ── 6. Sector Rotation Signals ───────────────────────────────────
    # Detect institutional rotation via sector ETF relative momentum
    sector_etfs = ["XLK", "XLF", "XLE", "XLV", "XLI"]
    available_sectors = [s for s in sector_etfs if s in close.columns]

    if len(available_sectors) >= 3 and "SPY" in close.columns:
        # Sector dispersion: high dispersion = stock picker's market
        sector_rets = pd.DataFrame({
            s: close[s].pct_change(21) for s in available_sectors
        })
        features["sector_dispersion"] = sector_rets.std(axis=1)

        # Defensive vs cyclical rotation
        defensive = [s for s in ["XLV"] if s in close.columns]
        cyclical = [s for s in ["XLK", "XLE", "XLI"] if s in close.columns]
        if defensive and cyclical:
            def_ret = pd.DataFrame({s: close[s].pct_change(21) for s in defensive}).mean(axis=1)
            cyc_ret = pd.DataFrame({s: close[s].pct_change(21) for s in cyclical}).mean(axis=1)
            features["cyclical_vs_defensive"] = cyc_ret - def_ret

    # Sector ETF volume anomaly (institutional flow proxy)
    for etf in available_sectors:
        if etf in volume.columns:
            avg_vol = volume[etf].rolling(20).mean()
            features[f"{etf.lower()}_vol_anomaly"] = (volume[etf] / avg_vol.replace(0, np.nan)) - 1

    # ── 7. Calendar / Behavioral Features ────────────────────────────
    dates = pd.Series(close.index)

    # Day of week (Monday=0, Friday=4)
    features["day_of_week"] = close.index.dayofweek

    # Month-end proximity (last 3 trading days of month = pension rebalancing)
    features["month_end"] = 0
    for i, dt in enumerate(close.index):
        # Check if this is one of last 3 trading days of the month
        month_dates = close.index[(close.index.month == dt.month) &
                                   (close.index.year == dt.year)]
        if len(month_dates) >= 3 and dt >= month_dates[-3]:
            features.iloc[i, features.columns.get_loc("month_end")] = 1

    # January effect
    features["is_january"] = (close.index.month == 1).astype(int)

    # Options expiration week (3rd Friday of each month)
    features["opex_week"] = 0
    for i, dt in enumerate(close.index):
        # Find 3rd Friday of this month
        first_day = dt.replace(day=1)
        # Days until first Friday
        days_to_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_to_friday)
        third_friday = first_friday + timedelta(weeks=2)
        # OpEx week = Mon-Fri containing the 3rd Friday
        opex_monday = third_friday - timedelta(days=third_friday.weekday())
        opex_friday = opex_monday + timedelta(days=4)
        if opex_monday <= dt <= opex_friday:
            features.iloc[i, features.columns.get_loc("opex_week")] = 1

    # Quarter end (extra rebalancing pressure)
    features["quarter_end"] = ((close.index.month % 3 == 0) &
                                (features["month_end"] == 1)).astype(int)

    # ── 8. Cross-Asset Momentum Composite ────────────────────────────
    # Combines multiple risk signals into a single "macro risk" score
    risk_signals = []
    if "hyg_above_sma50" in features.columns:
        risk_signals.append(features["hyg_above_sma50"])
    if "iwm_above_sma50" in features.columns:
        risk_signals.append(features["iwm_above_sma50"])
    if "vix_contango" in features.columns:
        risk_signals.append(features["vix_contango"])
    if "copper_gold_momentum" in features.columns:
        risk_signals.append((features["copper_gold_momentum"] > 0).astype(int))

    if risk_signals:
        features["macro_risk_score"] = sum(risk_signals) / len(risk_signals)

    # Clean up
    features = features.replace([np.inf, -np.inf], np.nan)
    features.index.name = "date"

    n_features = len([c for c in features.columns if features[c].notna().sum() > 100])
    logger.info(f"Engineered {n_features} cross-asset features over {len(features)} days")

    return features


# ═══════════════════════════════════════════════════════════════════════
# Merge with Stock Data
# ═══════════════════════════════════════════════════════════════════════

def merge_cross_asset_with_stocks(stock_df: pd.DataFrame,
                                    cross_features: pd.DataFrame) -> pd.DataFrame:
    """
    Merges cross-asset features into the stock-level DataFrame.
    Cross-asset features are the same for all stocks on a given date.
    """
    logger.info("Merging cross-asset features with stock data...")

    # Ensure date types match
    stock_df = stock_df.copy()
    stock_df["date"] = pd.to_datetime(stock_df["date"])
    cross_features = cross_features.copy()
    cross_features.index = pd.to_datetime(cross_features.index)

    # Merge on date
    cross_features_reset = cross_features.reset_index()
    merged = stock_df.merge(cross_features_reset, on="date", how="left")

    n_new = len(cross_features.columns)
    n_filled = merged[cross_features.columns].notna().mean().mean()
    logger.info(f"Merged {n_new} cross-asset features (avg {n_filled*100:.0f}% coverage)")

    return merged


# ═══════════════════════════════════════════════════════════════════════
# Model Comparison Test
# ═══════════════════════════════════════════════════════════════════════

def compare_models(stock_df: pd.DataFrame, cross_features: pd.DataFrame,
                    config: dict, full_walkforward: bool = False):
    """
    Compares model performance with and without cross-asset features.

    Quick mode: single train/test split
    Full mode: walk-forward validation (slower but more robust)
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from models.train import build_features, build_target, FEATURE_COLS, EXTENDED_FEATURES

    logger.info("=" * 60)
    logger.info("  MODEL COMPARISON: Base vs Cross-Asset Enhanced")
    logger.info("=" * 60)

    # Prepare data
    stock_df = build_target(stock_df)
    stock_df, base_features = build_features(stock_df)

    # Merge cross-asset
    enhanced_df = merge_cross_asset_with_stocks(stock_df, cross_features)
    cross_cols = [c for c in cross_features.columns
                  if c in enhanced_df.columns and enhanced_df[c].notna().sum() > 1000]
    enhanced_features = base_features + cross_cols

    logger.info(f"Base features: {len(base_features)}")
    logger.info(f"Cross-asset features: {len(cross_cols)}")
    logger.info(f"Enhanced total: {len(enhanced_features)}")

    # LightGBM params
    params = {
        "objective": "binary", "metric": "auc",
        "learning_rate": 0.03, "num_leaves": 31,
        "min_child_samples": 150, "feature_fraction": 0.65,
        "bagging_fraction": 0.75, "bagging_freq": 5,
        "reg_alpha": 0.3, "reg_lambda": 0.3,
        "max_depth": 6, "verbose": -1, "n_jobs": -1,
    }

    if full_walkforward:
        results = _walkforward_comparison(stock_df, enhanced_df, base_features,
                                           enhanced_features, cross_cols, params, config)
    else:
        results = _single_split_comparison(stock_df, enhanced_df, base_features,
                                            enhanced_features, cross_cols, params)

    return results


def _single_split_comparison(stock_df, enhanced_df, base_features,
                              enhanced_features, cross_cols, params):
    """Quick single train/test split comparison."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    dates = sorted(stock_df["date"].unique())
    split_idx = int(len(dates) * 0.75)
    split_date = dates[split_idx]

    train_mask = stock_df["date"] < split_date
    test_mask = stock_df["date"] >= split_date

    results = {}

    for name, df, features in [
        ("Base Model", stock_df, base_features),
        ("Enhanced Model", enhanced_df, enhanced_features),
    ]:
        tr = df[train_mask].dropna(subset=features + ["target"])
        te = df[test_mask].dropna(subset=features + ["target"])

        split = int(len(tr) * 0.8)
        train_data = lgb.Dataset(tr.iloc[:split][features], label=tr.iloc[:split]["target"])
        val_data = lgb.Dataset(tr.iloc[split:][features], label=tr.iloc[split:]["target"],
                                reference=train_data)

        model = lgb.train(params, train_data, num_boost_round=500,
                           valid_sets=[val_data],
                           callbacks=[lgb.early_stopping(30, verbose=False),
                                      lgb.log_evaluation(-1)])

        preds = model.predict(te[features])
        auc = roc_auc_score(te["target"], preds)

        # Quintile analysis
        te = te.copy()
        te["ml_score"] = preds
        te["quintile"] = pd.qcut(te["ml_score"], 5, labels=[1,2,3,4,5])
        q_returns = te.groupby("quintile")["fwd_5d"].mean() * 100

        # Top vs bottom quintile spread
        top_q = te[te["quintile"] == 5]["fwd_5d"].mean() * 100
        bot_q = te[te["quintile"] == 1]["fwd_5d"].mean() * 100

        results[name] = {
            "auc": auc,
            "top_quintile": top_q,
            "bot_quintile": bot_q,
            "spread": top_q - bot_q,
            "q_returns": q_returns,
            "model": model,
            "n_features": len(features),
        }

        logger.info(f"\n  {name}:")
        logger.info(f"    AUC:           {auc:.4f}")
        logger.info(f"    Top quintile:  {top_q:+.3f}%")
        logger.info(f"    Bot quintile:  {bot_q:+.3f}%")
        logger.info(f"    Spread:        {top_q - bot_q:.3f}%")

    # Print comparison
    base = results["Base Model"]
    enhanced = results["Enhanced Model"]

    print("\n" + "=" * 65)
    print("  MODEL COMPARISON — Single Split")
    print("=" * 65)
    print(f"  {'Metric':<25} {'Base':>15} {'Enhanced':>15} {'Delta':>10}")
    print(f"  {'-'*60}")
    print(f"  {'AUC':<25} {base['auc']:>15.4f} {enhanced['auc']:>15.4f} "
          f"{enhanced['auc']-base['auc']:>+10.4f}")
    print(f"  {'Top Quintile Ret':<25} {base['top_quintile']:>+14.3f}% {enhanced['top_quintile']:>+14.3f}% "
          f"{enhanced['top_quintile']-base['top_quintile']:>+9.3f}%")
    print(f"  {'Bot Quintile Ret':<25} {base['bot_quintile']:>+14.3f}% {enhanced['bot_quintile']:>+14.3f}% "
          f"{enhanced['bot_quintile']-base['bot_quintile']:>+9.3f}%")
    print(f"  {'Q1-Q5 Spread':<25} {base['spread']:>14.3f}% {enhanced['spread']:>14.3f}% "
          f"{enhanced['spread']-base['spread']:>+9.3f}%")
    print(f"  {'Features Used':<25} {base['n_features']:>15} {enhanced['n_features']:>15}")
    print("=" * 65)

    # Feature importance for cross-asset features
    print("\n  TOP CROSS-ASSET FEATURES BY IMPORTANCE:")
    print(f"  {'-'*50}")
    importance = enhanced["model"].feature_importance(importance_type="gain")
    feat_imp = pd.DataFrame({
        "feature": enhanced_features,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    cross_imp = feat_imp[feat_imp["feature"].isin(cross_cols)].head(15)
    total_imp = feat_imp["importance"].sum()
    for _, row in cross_imp.iterrows():
        pct = row["importance"] / total_imp * 100
        bar = "█" * int(pct * 2)
        print(f"  {row['feature']:<30} {pct:>5.1f}% {bar}")

    cross_total = feat_imp[feat_imp["feature"].isin(cross_cols)]["importance"].sum()
    print(f"\n  Cross-asset features: {cross_total/total_imp*100:.1f}% of total importance")

    # Quintile comparison chart
    _plot_comparison(results, cross_imp, total_imp)

    return results


def _walkforward_comparison(stock_df, enhanced_df, base_features,
                             enhanced_features, cross_cols, params, config):
    """Full walk-forward comparison (more robust but slower)."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    vc = config["validation"]
    dates = pd.Series(sorted(stock_df["date"].unique())).reset_index(drop=True)

    all_results = {"Base Model": [], "Enhanced Model": []}

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

        for name, df, features in [
            ("Base Model", stock_df, base_features),
            ("Enhanced Model", enhanced_df, enhanced_features),
        ]:
            tr = df[df["date"].isin(train_dates)].dropna(subset=features + ["target"])
            te = df[df["date"].isin(test_dates)].dropna(subset=features)

            if len(tr) < 500 or len(te) < 50:
                continue

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
            auc = roc_auc_score(te["target"].fillna(0), te["ml_score"])

            top20 = te.nlargest(max(1, int(len(te) * 0.2)), "ml_score")
            top20_r = top20["fwd_5d"].mean() * 100

            all_results[name].append({
                "test_start": test_start,
                "auc": auc,
                "top20_ret": top20_r,
            })

        train_start += pd.DateOffset(months=vc["test_months"])

    # Summarize
    print("\n" + "=" * 65)
    print("  WALK-FORWARD COMPARISON")
    print("=" * 65)

    for name in ["Base Model", "Enhanced Model"]:
        if all_results[name]:
            aucs = [r["auc"] for r in all_results[name]]
            rets = [r["top20_ret"] for r in all_results[name]]
            print(f"\n  {name} ({len(aucs)} folds):")
            print(f"    AUC:  {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
            print(f"    Top20% return: {np.mean(rets):+.3f}% ± {np.std(rets):.3f}%")

    # Delta
    if all_results["Base Model"] and all_results["Enhanced Model"]:
        base_auc = np.mean([r["auc"] for r in all_results["Base Model"]])
        enh_auc = np.mean([r["auc"] for r in all_results["Enhanced Model"]])
        base_ret = np.mean([r["top20_ret"] for r in all_results["Base Model"]])
        enh_ret = np.mean([r["top20_ret"] for r in all_results["Enhanced Model"]])

        print(f"\n  AUC improvement:    {enh_auc - base_auc:+.4f}")
        print(f"  Return improvement: {enh_ret - base_ret:+.3f}%")
        print("=" * 65)

    return all_results


# ═══════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════

def _plot_comparison(results, cross_imp, total_imp):
    """Plots comparison charts."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d1117")
    fig.suptitle("Cross-Asset Feature Analysis", color="white",
                 fontsize=14, fontweight="bold")

    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="gray")
        ax.spines[:].set_color("#333")

    # Quintile returns comparison
    ax1 = axes[0]
    x = np.arange(5)
    width = 0.35
    base_q = results["Base Model"]["q_returns"].values
    enh_q = results["Enhanced Model"]["q_returns"].values

    bars1 = ax1.bar(x - width/2, base_q, width, label="Base", color="#4a9eff", alpha=0.8)
    bars2 = ax1.bar(x + width/2, enh_q, width, label="Enhanced", color="#ffd700", alpha=0.8)
    ax1.set_xlabel("Quintile", color="gray")
    ax1.set_ylabel("Avg 5-Day Return (%)", color="gray")
    ax1.set_title("Quintile Returns", color="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Q1\n(worst)", "Q2", "Q3", "Q4", "Q5\n(best)"])
    ax1.legend(facecolor="#1a1a2e", labelcolor="white")
    ax1.axhline(0, color="#555", linewidth=0.5)

    # Feature importance
    ax2 = axes[1]
    top_cross = cross_imp.head(10)
    y_pos = range(len(top_cross))
    pcts = [row["importance"] / total_imp * 100 for _, row in top_cross.iterrows()]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_cross)))
    ax2.barh(y_pos, pcts, color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_cross["feature"].values, fontsize=9, color="gray")
    ax2.set_xlabel("% of Total Importance", color="gray")
    ax2.set_title("Top Cross-Asset Features", color="white")
    ax2.invert_yaxis()

    plt.tight_layout()
    Path("research").mkdir(exist_ok=True)
    save_path = "research/cross_asset_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    logger.info(f"Chart saved to {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Run full walk-forward comparison (slower)")
    args = parser.parse_args()

    # Load config
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    # Load existing stock data
    from data.pipeline import load_processed
    try:
        stock_df = load_processed("price_features_enriched.parquet")
    except FileNotFoundError:
        try:
            stock_df = load_processed("price_features.parquet")
        except FileNotFoundError:
            print("No processed data found. Run: python main.py pipeline")
            sys.exit(1)

    # Download cross-asset data
    start_date = config["data"]["start_date"]
    close, volume = download_cross_asset_data(start_date)
    vix_df = download_vix_term_structure(start_date)

    # Engineer features
    cross_features = engineer_cross_asset_features(close, volume, vix_df)

    # Save cross-asset features for later use
    cross_path = Path("data/processed/cross_asset_features.parquet")
    cross_features.to_parquet(cross_path)
    logger.info(f"Saved cross-asset features to {cross_path}")

    # Compare models
    results = compare_models(stock_df, cross_features, config,
                              full_walkforward=args.full)

    print("\nDone. Chart saved to research/cross_asset_analysis.png")
    print("\nTo run full walk-forward validation:")
    print("  python research/cross_asset_features.py --full")
