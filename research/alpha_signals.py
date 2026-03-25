"""
research/alpha_signals.py
--------------------------
STANDALONE RESEARCH — Tests unconventional stock-level alpha signals.

Unlike cross-asset features (same value for all stocks on a date), these
are stock-specific signals that help the model pick WHICH stocks will
outperform. This is where the real edge lives.

Signal Categories:
  1. Institutional Footprint — detect big money accumulation/distribution
     before it shows up in price
  2. Supply/Demand Imbalance — order flow proxies from price/volume
  3. Volatility Regime Shifts — stocks transitioning from low to high vol
  4. Mean Reversion Timing — identify oversold bounces with precision
  5. Relative Value — cheap vs expensive within sector
  6. Smart Money Divergence — price vs volume disagreements
  7. Microstructure — intraday patterns from OHLC data
  8. Cross-stock contagion — momentum spillover from sector leaders

Plus the best 2-3 cross-asset features from the previous research.

Usage:
    python research/alpha_signals.py           # Quick single-split test
    python research/alpha_signals.py --full    # Full walk-forward validation
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

sys.path.insert(0, ".")


# ═══════════════════════════════════════════════════════════════════════
# Alpha Signal Engineering
# ═══════════════════════════════════════════════════════════════════════

def engineer_alpha_signals(df: pd.DataFrame) -> tuple:
    """
    Engineers stock-level alpha signals from OHLCV data.
    Returns the enhanced DataFrame and list of new feature names.
    """
    logger.info("Engineering alpha signals...")
    df = df.copy().sort_values(["ticker", "date"])

    new_features = []

    results = []
    for ticker, grp in df.groupby("ticker"):
        grp = grp.sort_values("date").copy()
        c = grp["close"]
        o = grp["open"]
        h = grp["high"]
        l = grp["low"]
        v = grp["volume"]
        ret = c.pct_change()

        # ── 1. INSTITUTIONAL FOOTPRINT ──────────────────────────────
        # Idea: institutions accumulate over multiple days with above-average
        # volume but controlled price movement. They try to hide their buying.

        # Accumulation/Distribution Line (classic but enhanced)
        # Money Flow Multiplier: where close falls in the day's range
        mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
        mfv = mfm * v  # Money Flow Volume
        grp["adl"] = mfv.cumsum()
        grp["adl_slope_10d"] = grp["adl"].rolling(10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0,
            raw=True
        )

        # Volume-Price Divergence: price flat but volume accumulating
        # If ADL is rising but price is flat, institutions are quietly buying
        price_chg_10d = c.pct_change(10)
        adl_chg_10d = grp["adl"].diff(10)
        adl_norm = adl_chg_10d / v.rolling(10).sum().replace(0, np.nan)
        grp["vol_price_divergence"] = adl_norm - price_chg_10d

        # Institutional volume signature: high volume on up days, low on down days
        up_vol = v.where(ret > 0, 0).rolling(20).mean()
        down_vol = v.where(ret <= 0, 0).rolling(20).mean()
        grp["up_down_vol_ratio"] = (up_vol / down_vol.replace(0, np.nan))

        # Quiet accumulation: volume above average but price change small
        avg_vol = v.rolling(20).mean()
        vol_ratio = v / avg_vol.replace(0, np.nan)
        abs_ret = ret.abs()
        avg_abs_ret = abs_ret.rolling(20).mean()
        grp["quiet_accumulation"] = vol_ratio / (abs_ret / avg_abs_ret.replace(0, np.nan)).replace(0, np.nan)
        # High value = lots of volume relative to price movement = institutional

        # ── 2. SUPPLY/DEMAND IMBALANCE ──────────────────────────────
        # Closing position within bar: where does the close fall in the range?
        # Consistently closing near highs = demand > supply
        grp["close_position"] = (c - l) / (h - l).replace(0, np.nan)
        grp["close_position_5d"] = grp["close_position"].rolling(5).mean()

        # Buying pressure: close > midpoint of range on above-avg volume
        midpoint = (h + l) / 2
        grp["buying_pressure"] = ((c > midpoint).astype(int) *
                                   (v > avg_vol).astype(int)).rolling(10).mean()

        # Price rejection: long lower wicks = buyers defending a level
        body = (c - o).abs()
        lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l
        upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
        full_range = (h - l).replace(0, np.nan)

        grp["lower_wick_ratio"] = (lower_wick / full_range).rolling(5).mean()
        grp["upper_wick_ratio"] = (upper_wick / full_range).rolling(5).mean()

        # Hammer pattern frequency (long lower wick + small body near high)
        is_hammer = ((lower_wick > 2 * body) & (upper_wick < body)).astype(int)
        grp["hammer_freq_10d"] = is_hammer.rolling(10).sum()

        # ── 3. VOLATILITY REGIME SHIFT ──────────────────────────────
        # Stocks breaking out of low-vol consolidation have strong momentum
        vol_5d = ret.rolling(5).std()
        vol_20d = ret.rolling(20).std()
        vol_60d = ret.rolling(60).std()

        # Vol contraction: current vol much lower than recent = coiled spring
        grp["vol_contraction"] = vol_5d / vol_60d.replace(0, np.nan)

        # Vol breakout: 5-day vol suddenly exceeds 20-day
        grp["vol_expansion"] = vol_5d / vol_20d.replace(0, np.nan)

        # Bollinger Band squeeze then expand
        bb_width = grp.get("bb_width")
        if bb_width is not None:
            bb_min_20 = bb_width.rolling(20).min()
            grp["bb_squeeze_release"] = bb_width / bb_min_20.replace(0, np.nan)

        # ── 4. MEAN REVERSION TIMING ────────────────────────────────
        # Not just "oversold" but "oversold AND showing signs of reversal"

        # RSI with momentum confirmation
        rsi = grp.get("rsi_14")
        if rsi is not None:
            grp["rsi_oversold_reversing"] = (
                (rsi < 35) & (rsi > rsi.shift(1)) & (rsi.shift(1) < rsi.shift(2))
            ).astype(int).rolling(5).max()  # Was oversold and turning up recently

        # Distance from 20-day low as % — how far has it bounced?
        low_20d = l.rolling(20).min()
        grp["bounce_from_low"] = (c - low_20d) / low_20d.replace(0, np.nan)

        # Consecutive down days then up — reversal pattern
        down_days = (ret < 0).astype(int)
        consec_down = down_days.rolling(5).sum()
        grp["reversal_setup"] = (
            (consec_down.shift(1) >= 3) & (ret > 0)
        ).astype(int).rolling(5).max()

        # Mean reversion Z-score: how many std devs from 20-day mean
        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        grp["price_zscore_20d"] = (c - sma20) / std20.replace(0, np.nan)

        # ── 5. RELATIVE VALUE ────────────────────────────────────────
        # Price relative to 52-week range — are we near the bottom or top?
        high_52w = h.rolling(252).max()
        low_52w = l.rolling(252).min()
        range_52w = (high_52w - low_52w).replace(0, np.nan)
        grp["position_in_52w_range"] = (c - low_52w) / range_52w

        # Distance from 52w high (we might already have this but computed differently)
        grp["pct_from_52w_high"] = (c - high_52w) / high_52w.replace(0, np.nan)

        # Recent range compression: last 10 days range vs last 60 days
        range_10d = h.rolling(10).max() - l.rolling(10).min()
        range_60d = h.rolling(60).max() - l.rolling(60).min()
        grp["range_compression"] = range_10d / range_60d.replace(0, np.nan)

        # ── 6. SMART MONEY DIVERGENCE ────────────────────────────────
        # Volume-weighted average price (VWAP) proxy
        typical_price = (h + l + c) / 3
        vwap_20d = (typical_price * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
        grp["price_vs_vwap"] = (c - vwap_20d) / vwap_20d.replace(0, np.nan)

        # On-balance volume divergence from price
        obv = (v * np.sign(ret.fillna(0))).cumsum()
        # Normalize OBV to compare across stocks
        obv_pct = obv.pct_change(20)
        price_pct = c.pct_change(20)
        grp["obv_price_divergence"] = obv_pct - price_pct

        # Chaikin Money Flow (20-day)
        cmf_raw = mfv.rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
        grp["cmf_20d"] = cmf_raw

        # ── 7. MICROSTRUCTURE FROM OHLC ──────────────────────────────
        # Gap analysis: overnight gaps that hold vs fade
        gap = o / c.shift(1) - 1
        grp["gap_fill_ratio"] = np.where(
            gap > 0,
            1 - (o - l) / (gap * c.shift(1)).replace(0, np.nan),  # How much gap filled
            np.where(
                gap < 0,
                1 - (h - o) / (-gap * c.shift(1)).replace(0, np.nan),
                0.5
            )
        )
        grp["gap_fill_ratio"] = pd.Series(grp["gap_fill_ratio"], index=grp.index)
        grp["avg_gap_fill_5d"] = grp["gap_fill_ratio"].rolling(5).mean()

        # Opening drive: first move direction (open vs prior close) predicting close
        open_move = (o - c.shift(1)) / c.shift(1).replace(0, np.nan)
        close_move = ret
        grp["open_drive_consistency"] = (
            (np.sign(open_move) == np.sign(close_move)).astype(int)
        ).rolling(10).mean()

        # Bar efficiency: how much of the day's range turns into close-to-close move
        grp["bar_efficiency"] = (c - c.shift(1)).abs() / full_range.replace(0, np.nan)
        grp["avg_bar_efficiency_10d"] = grp["bar_efficiency"].rolling(10).mean()

        # ── 8. TREND QUALITY ─────────────────────────────────────────
        # Not just "is it trending" but "how clean is the trend"

        # R-squared of price over last 20 days (linearity of trend)
        def rolling_r2(series, window=20):
            def r2(x):
                if len(x) < window:
                    return np.nan
                y = np.arange(len(x))
                corr = np.corrcoef(y, x)[0, 1]
                return corr ** 2
            return series.rolling(window).apply(r2, raw=True)

        grp["trend_r2_20d"] = rolling_r2(c, 20)

        # Efficiency ratio: net move / sum of absolute moves
        net_move = (c - c.shift(20)).abs()
        sum_abs_moves = ret.abs().rolling(20).sum() * c
        grp["efficiency_ratio_20d"] = net_move / sum_abs_moves.replace(0, np.nan)

        # Higher highs / higher lows streak
        hh = (h > h.shift(1)).astype(int)
        hl = (l > l.shift(1)).astype(int)
        grp["hh_hl_streak"] = (hh * hl).rolling(5).sum()

        results.append(grp)

    out = pd.concat(results, ignore_index=True)

    # Collect new feature names
    original_cols = set(df.columns)
    new_features = [c for c in out.columns if c not in original_cols
                    and c not in ["adl"]]  # exclude intermediate calcs

    # Clean infinities
    for col in new_features:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)

    logger.info(f"Engineered {len(new_features)} alpha signals")
    return out, new_features


def add_best_cross_asset(df: pd.DataFrame) -> tuple:
    """
    Adds just the 2-3 best cross-asset features from previous research.
    These are date-level features merged onto stock data.
    """
    logger.info("Adding top cross-asset features...")

    cross_path = Path("data/processed/cross_asset_features.parquet")
    if not cross_path.exists():
        logger.warning("No cross-asset features found. Run cross_asset_features.py first.")
        return df, []

    cross = pd.read_parquet(cross_path)
    cross.index = pd.to_datetime(cross.index)

    # Cherry-pick only the best performers from walk-forward
    keep_cols = []
    available = cross.columns.tolist()

    # These showed consistent importance across folds:
    best = ["copper_gold_ratio", "copper_gold_momentum", "vix_term_structure",
            "vix_zscore", "risk_appetite_21d", "macro_risk_score"]
    keep_cols = [c for c in best if c in available]

    if not keep_cols:
        return df, []

    cross_subset = cross[keep_cols].reset_index()
    cross_subset.columns = ["date"] + [f"xa_{c}" for c in keep_cols]

    df["date"] = pd.to_datetime(df["date"])
    merged = df.merge(cross_subset, on="date", how="left")

    new_cols = [f"xa_{c}" for c in keep_cols]
    logger.info(f"Added {len(new_cols)} cross-asset features: {new_cols}")

    return merged, new_cols


# ═══════════════════════════════════════════════════════════════════════
# Model Comparison
# ═══════════════════════════════════════════════════════════════════════

def compare_models(stock_df: pd.DataFrame, full_walkforward: bool = False):
    """
    Trains and compares:
      1. Base model (current features)
      2. Alpha model (current + alpha signals)
      3. Alpha + Cross-Asset model (current + alpha + best cross-asset)
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from models.train import build_features, build_target

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    # Build base features and target
    stock_df = build_target(stock_df)
    stock_df, base_features = build_features(stock_df)

    # Add alpha signals
    alpha_df, alpha_features = engineer_alpha_signals(stock_df)

    # Add cross-asset
    full_df, cross_features = add_best_cross_asset(alpha_df)

    # Feature sets to compare
    alpha_all = base_features + alpha_features
    full_all = alpha_all + cross_features

    # Filter to features with enough data
    for feat_list_name in ["alpha_features", "cross_features"]:
        feat_list = locals()[feat_list_name]
        valid = [f for f in feat_list if f in full_df.columns
                 and full_df[f].notna().sum() > 1000]
        locals()[feat_list_name] = valid

    alpha_features = [f for f in alpha_features if f in full_df.columns
                      and full_df[f].notna().sum() > 1000]
    cross_features = [f for f in cross_features if f in full_df.columns
                      and full_df[f].notna().sum() > 1000]
    alpha_all = base_features + alpha_features
    full_all = alpha_all + cross_features

    logger.info(f"Base features:    {len(base_features)}")
    logger.info(f"Alpha features:   {len(alpha_features)}")
    logger.info(f"Cross-asset:      {len(cross_features)}")
    logger.info(f"Total enhanced:   {len(full_all)}")

    params = {
        "objective": "binary", "metric": "auc",
        "learning_rate": 0.03, "num_leaves": 31,
        "min_child_samples": 150, "feature_fraction": 0.65,
        "bagging_fraction": 0.75, "bagging_freq": 5,
        "reg_alpha": 0.3, "reg_lambda": 0.3,
        "max_depth": 6, "verbose": -1, "n_jobs": -1,
    }

    models_to_test = [
        ("Base (38 feat)", stock_df, base_features),
        ("+ Alpha Signals", alpha_df, alpha_all),
        ("+ Alpha + CrossAsset", full_df, full_all),
    ]

    if full_walkforward:
        _run_walkforward(models_to_test, params, config)
    else:
        _run_single_split(models_to_test, params, config)


def _run_single_split(models_to_test, params, config):
    """Quick single train/test split comparison."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    # Use first model's df for date split
    ref_df = models_to_test[0][1]
    dates = sorted(ref_df["date"].unique())
    split_date = dates[int(len(dates) * 0.75)]

    results = {}

    for name, df, features in models_to_test:
        train_mask = df["date"] < split_date
        test_mask = df["date"] >= split_date
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

        te = te.copy()
        te["ml_score"] = model.predict(te[features])
        auc = roc_auc_score(te["target"], te["ml_score"])

        te["quintile"] = pd.qcut(te["ml_score"], 5, labels=[1,2,3,4,5])
        q_returns = te.groupby("quintile")["fwd_5d"].mean() * 100

        top_q = te[te["quintile"] == 5]["fwd_5d"].mean() * 100
        bot_q = te[te["quintile"] == 1]["fwd_5d"].mean() * 100

        # Monotonicity check
        q_vals = q_returns.values
        is_monotonic = all(q_vals[i] <= q_vals[i+1] for i in range(len(q_vals)-1))

        results[name] = {
            "auc": auc, "top_q": top_q, "bot_q": bot_q,
            "spread": top_q - bot_q, "q_returns": q_returns,
            "monotonic": is_monotonic, "model": model,
            "n_features": len(features), "features": features,
        }

    # Print results
    print("\n" + "=" * 75)
    print("  ALPHA SIGNAL COMPARISON — Single Split")
    print("=" * 75)
    print(f"  {'Model':<25} {'AUC':>8} {'Top Q':>9} {'Bot Q':>9} "
          f"{'Spread':>9} {'Mono':>6} {'Feat':>5}")
    print(f"  {'-'*70}")
    for name, r in results.items():
        mono = "✓" if r["monotonic"] else "✗"
        print(f"  {name:<25} {r['auc']:>8.4f} {r['top_q']:>+8.3f}% "
              f"{r['bot_q']:>+8.3f}% {r['spread']:>8.3f}% {mono:>6} {r['n_features']:>5}")
    print("=" * 75)

    # Feature importance for best model
    best_name = max(results, key=lambda k: results[k]["auc"])
    best = results[best_name]
    print(f"\n  TOP 20 FEATURES — {best_name}:")
    print(f"  {'-'*55}")

    importance = best["model"].feature_importance(importance_type="gain")
    feat_imp = pd.DataFrame({
        "feature": best["features"],
        "importance": importance,
    }).sort_values("importance", ascending=False)

    total_imp = feat_imp["importance"].sum()
    for _, row in feat_imp.head(20).iterrows():
        pct = row["importance"] / total_imp * 100
        # Mark feature type
        if row["feature"] in models_to_test[0][2]:
            tag = "     "
        elif row["feature"].startswith("xa_"):
            tag = " [XA]"
        else:
            tag = " [α] "
        bar = "█" * int(pct * 2)
        print(f"  {tag} {row['feature']:<30} {pct:>5.1f}% {bar}")

    # Quintile return chart
    _plot_results(results)


def _run_walkforward(models_to_test, params, config):
    """Full walk-forward comparison."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    vc = config["validation"]
    ref_df = models_to_test[0][1]
    dates = pd.Series(sorted(ref_df["date"].unique())).reset_index(drop=True)

    all_results = {name: [] for name, _, _ in models_to_test}

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

        for name, df, features in models_to_test:
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
            bot20 = te.nsmallest(max(1, int(len(te) * 0.2)), "ml_score")
            bot20_r = bot20["fwd_5d"].mean() * 100

            all_results[name].append({
                "fold": fold, "auc": auc,
                "top20_ret": top20_r, "bot20_ret": bot20_r,
                "spread": top20_r - bot20_r,
            })

        train_start += pd.DateOffset(months=vc["test_months"])

    # Print results
    print("\n" + "=" * 75)
    print("  WALK-FORWARD COMPARISON — Alpha Signals")
    print("=" * 75)
    print(f"  {'Model':<25} {'Folds':>6} {'AUC':>12} {'Top20%':>14} {'Spread':>14}")
    print(f"  {'-'*70}")

    for name in [n for n, _, _ in models_to_test]:
        if all_results[name]:
            aucs = [r["auc"] for r in all_results[name]]
            rets = [r["top20_ret"] for r in all_results[name]]
            spreads = [r["spread"] for r in all_results[name]]
            print(f"  {name:<25} {len(aucs):>6} "
                  f"{np.mean(aucs):>.4f}±{np.std(aucs):.3f} "
                  f"{np.mean(rets):>+.3f}±{np.std(rets):.3f}% "
                  f"{np.mean(spreads):>+.3f}±{np.std(spreads):.3f}%")
    print("=" * 75)

    # Per-fold detail
    base_name = models_to_test[0][0]
    best_name = models_to_test[-1][0]
    if all_results[base_name] and all_results[best_name]:
        print(f"\n  Per-fold AUC comparison ({base_name} vs {best_name}):")
        base_wins = 0
        for i, (b, e) in enumerate(zip(all_results[base_name], all_results[best_name])):
            delta = e["auc"] - b["auc"]
            winner = "+" if delta > 0 else "-"
            if delta > 0:
                base_wins += 1
            print(f"    Fold {b['fold']:>2}: base={b['auc']:.4f}  enhanced={e['auc']:.4f}  "
                  f"Δ={delta:+.4f} {winner}")
        n = len(all_results[base_name])
        print(f"\n  Enhanced wins: {base_wins}/{n} folds ({base_wins/n*100:.0f}%)")


def _plot_results(results):
    """Plot quintile returns for all models."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="gray")
    ax.spines[:].set_color("#333")

    x = np.arange(5)
    n_models = len(results)
    width = 0.8 / n_models
    colors = ["#4a9eff", "#ffd700", "#00e676"]

    for i, (name, r) in enumerate(results.items()):
        offset = (i - n_models/2 + 0.5) * width
        q_vals = r["q_returns"].values
        ax.bar(x + offset, q_vals, width, label=f"{name} (AUC={r['auc']:.3f})",
               color=colors[i % len(colors)], alpha=0.85)

    ax.set_xlabel("Quintile", color="gray")
    ax.set_ylabel("Avg 5-Day Return (%)", color="gray")
    ax.set_title("Quintile Returns by Model", color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Q1\n(worst)", "Q2", "Q3", "Q4", "Q5\n(best)"])
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    ax.axhline(0, color="#555", linewidth=0.5)

    plt.tight_layout()
    Path("research").mkdir(exist_ok=True)
    save_path = "research/alpha_signals_analysis.png"
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
                        help="Run full walk-forward comparison")
    args = parser.parse_args()

    # Load stock data
    from data.pipeline import load_processed
    try:
        stock_df = load_processed("price_features_enriched.parquet")
    except FileNotFoundError:
        try:
            stock_df = load_processed("price_features.parquet")
        except FileNotFoundError:
            print("No processed data found. Run: python main.py pipeline")
            sys.exit(1)

    compare_models(stock_df, full_walkforward=args.full)

    print("\nDone. Chart saved to research/alpha_signals_analysis.png")
