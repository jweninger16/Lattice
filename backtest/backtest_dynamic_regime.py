"""
backtest/backtest_dynamic_regime.py
------------------------------------
Compares three regime approaches:
  1. Fixed 50% threshold (original)
  2. Dynamic VIX-adjusted threshold (new)
  3. Dynamic VIX threshold + SH hedge

This tells us whether the VIX adjustment genuinely improves performance
or just adds complexity without benefit.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger


def get_vix_thresholds(start: str, end: str) -> pd.DataFrame:
    """Downloads VIX and computes dynamic thresholds for each day."""
    logger.info("Downloading VIX data...")
    vix = yf.download("^VIX", start=start, end=end,
                      auto_adjust=True, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = [col[0] for col in vix.columns]

    vix = vix[["Close"]].rename(columns={"Close": "vix"})
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    vix.index.name = "date"
    vix = vix.reset_index()
    vix["vix_5d_avg"] = vix["vix"].rolling(5).mean()

    def vix_to_threshold(row):
        v = row["vix"]
        avg = row["vix_5d_avg"] if not pd.isna(row["vix_5d_avg"]) else v
        if v < 15:   base = 0.45
        elif v < 20: base = 0.50
        elif v < 25: base = 0.55
        elif v < 30: base = 0.60
        else:        base = 0.65
        # Trend adjustment
        change = (v - avg) / avg if avg > 0 else 0
        if change > 0.15:   adj = 0.03
        elif change > 0.05: adj = 0.01
        elif change < -0.10: adj = -0.02
        else:               adj = 0.0
        return round(base + adj, 3)

    vix["dynamic_threshold"] = vix.apply(vix_to_threshold, axis=1)
    return vix


def run_dynamic_regime_backtest(ml_signals_df: pd.DataFrame, config: dict) -> dict:
    from backtest.backtest import SwingBacktester, print_stats

    dates = sorted(ml_signals_df["date"].unique())
    start = pd.Timestamp(dates[0]).strftime("%Y-%m-%d")
    end   = pd.Timestamp(dates[-1]).strftime("%Y-%m-%d")

    # Get VIX thresholds
    vix_df = get_vix_thresholds(start, end)
    vix_df["date"] = pd.to_datetime(vix_df["date"])

    # Get SH data
    logger.info("Downloading SH data...")
    sh = yf.download("SH", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(sh.columns, pd.MultiIndex):
        sh.columns = [col[0] for col in sh.columns]
    sh = sh[["Close"]].rename(columns={"Close": "sh_close"})
    sh.index = pd.to_datetime(sh.index).tz_localize(None)
    sh.index.name = "date"
    sh = sh.reset_index()
    sh["sh_ret"] = sh["sh_close"].pct_change().fillna(0)
    sh["date"] = pd.to_datetime(sh["date"])

    # ── Run 1: Fixed 50% threshold (baseline) ───────────────────────────
    logger.info("Running backtest 1: Fixed 50% threshold...")
    df_fixed = ml_signals_df.copy()
    bt1 = SwingBacktester(config)
    result1 = bt1.run(df_fixed)
    stats1 = result1["stats"]
    eq1 = result1["equity"].reset_index()

    # ── Run 2: Dynamic VIX threshold ────────────────────────────────────
    logger.info("Running backtest 2: Dynamic VIX threshold...")
    df_dynamic = ml_signals_df.copy()
    df_dynamic["date"] = pd.to_datetime(df_dynamic["date"])

    # Recompute regime_ok using dynamic threshold
    daily_breadth = df_dynamic.groupby("date").agg(
        pct_above_sma50=("above_sma50", "mean"),
        pct_above_sma20=("above_sma20", "mean"),
        median_momentum_21d=("momentum_21d", "median"),
    ).reset_index()
    daily_breadth["date"] = pd.to_datetime(daily_breadth["date"])
    daily_breadth = daily_breadth.merge(vix_df[["date","dynamic_threshold","vix"]], on="date", how="left")
    daily_breadth["dynamic_threshold"] = daily_breadth["dynamic_threshold"].fillna(0.50)

    daily_breadth["regime_ok_dynamic"] = (
        (daily_breadth["pct_above_sma50"] >= daily_breadth["dynamic_threshold"]) &
        (daily_breadth["pct_above_sma20"] >= daily_breadth["dynamic_threshold"] - 0.10) &
        (daily_breadth["median_momentum_21d"] >= -0.02)
    ).astype(int)

    # Merge back
    df_dynamic = df_dynamic.drop(columns=["regime_ok"], errors="ignore")
    df_dynamic = df_dynamic.merge(
        daily_breadth[["date","regime_ok_dynamic","dynamic_threshold","vix"]],
        on="date", how="left"
    )
    df_dynamic["regime_ok"] = df_dynamic["regime_ok_dynamic"].fillna(0)
    df_dynamic["signal"] = (
        (df_dynamic["signal"] == 1) & (df_dynamic["regime_ok"] == 1)
    ).astype(int)

    favorable_fixed   = (daily_breadth["pct_above_sma50"] >= 0.50).sum()
    favorable_dynamic = daily_breadth["regime_ok_dynamic"].sum()
    logger.info(f"Favorable days - Fixed: {favorable_fixed} | Dynamic: {favorable_dynamic}")

    bt2 = SwingBacktester(config)
    result2 = bt2.run(df_dynamic)
    stats2 = result2["stats"]
    eq2 = result2["equity"].reset_index()

    # ── Run 3: Dynamic + SH hedge ────────────────────────────────────────
    logger.info("Running backtest 3: Dynamic threshold + SH hedge...")
    initial = config["backtest"]["initial_capital"]

    # Build hedged equity from dynamic backtest
    eq2_merged = eq2.copy()
    eq2_merged["date"] = pd.to_datetime(eq2_merged["date"])
    eq2_merged = eq2_merged.merge(
        daily_breadth[["date","regime_ok_dynamic"]], on="date", how="left"
    )
    eq2_merged = eq2_merged.merge(sh[["date","sh_ret"]], on="date", how="left")
    eq2_merged["sh_ret"] = eq2_merged["sh_ret"].fillna(0)

    hedged_values = []
    prev_base = initial
    prev_hedged = initial

    for _, row in eq2_merged.iterrows():
        base_val = row["equity"]
        if row.get("regime_ok_dynamic", 1) == 0 and not pd.isna(row["sh_ret"]):
            hedge_gain = prev_hedged * 0.50 * row["sh_ret"]
            hedged_val = prev_hedged + (base_val - prev_base) + hedge_gain
        else:
            hedged_val = prev_hedged + (base_val - prev_base)
        hedged_values.append(hedged_val)
        prev_base = base_val
        prev_hedged = hedged_val

    eq2_merged["hedged_equity"] = hedged_values

    # Compute hedged stats
    n_years = len(eq2_merged) / 252
    final_h = eq2_merged["hedged_equity"].iloc[-1]
    h_return = (final_h / initial - 1) * 100
    h_cagr   = ((final_h / initial) ** (1/n_years) - 1) * 100
    h_rets   = eq2_merged["hedged_equity"].pct_change().dropna()
    h_sharpe = h_rets.mean() / h_rets.std() * np.sqrt(252)
    roll_max = eq2_merged["hedged_equity"].cummax()
    h_maxdd  = ((eq2_merged["hedged_equity"] - roll_max) / roll_max * 100).min()

    # ── Print comparison ─────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  DYNAMIC REGIME — BACKTEST COMPARISON")
    print("="*65)
    print(f"  {'Metric':<22} {'Fixed 50%':>12} {'Dynamic VIX':>13} {'Dyn+SH':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Total Return':<22} {stats1.get('total_return_pct',0):>11.1f}% "
          f"{stats2.get('total_return_pct',0):>12.1f}% {h_return:>9.1f}%")
    print(f"  {'CAGR':<22} {stats1.get('cagr_pct',0):>11.1f}% "
          f"{stats2.get('cagr_pct',0):>12.1f}% {h_cagr:>9.1f}%")
    print(f"  {'Sharpe Ratio':<22} {stats1.get('sharpe',0):>12.2f} "
          f"{stats2.get('sharpe',0):>13.2f} {h_sharpe:>10.2f}")
    print(f"  {'Max Drawdown':<22} {stats1.get('max_drawdown_pct',0):>11.1f}% "
          f"{stats2.get('max_drawdown_pct',0):>12.1f}% {h_maxdd:>9.1f}%")
    print(f"  {'Final Capital':<22} ${stats1.get('final_capital',initial):>10,.0f} "
          f"${stats2.get('final_capital',initial):>11,.0f} ${final_h:>8,.0f}")
    print(f"  {'Favorable Days':<22} {favorable_fixed:>12} {favorable_dynamic:>13}")
    print("="*65 + "\n")

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                              facecolor="#0d1117",
                              gridspec_kw={"height_ratios": [3,1]})
    fig.suptitle("Fixed vs Dynamic VIX Regime Threshold", color="white",
                 fontsize=14, fontweight="bold")

    ax1, ax2 = axes
    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="gray")
        ax.spines[:].set_color("#333")

    eq1["date"] = pd.to_datetime(eq1["date"])
    eq2_merged["date"] = pd.to_datetime(eq2_merged["date"])

    ax1.plot(eq1["date"], eq1["equity"], color="#4a9eff", linewidth=1.5,
             label=f"Fixed 50% (+{stats1.get('total_return_pct',0):.1f}%)")
    ax1.plot(eq2_merged["date"], eq2_merged["equity"], color="#ffd700", linewidth=1.5,
             label=f"Dynamic VIX (+{stats2.get('total_return_pct',0):.1f}%)")
    ax1.plot(eq2_merged["date"], eq2_merged["hedged_equity"], color="#00e676", linewidth=1.5,
             label=f"Dynamic+SH (+{h_return:.1f}%)")
    ax1.axhline(initial, color="#555", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Portfolio Value ($)", color="gray")
    ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Drawdowns
    dd1 = (eq1["equity"] - eq1["equity"].cummax()) / eq1["equity"].cummax() * 100
    dd2 = (eq2_merged["equity"] - eq2_merged["equity"].cummax()) / eq2_merged["equity"].cummax() * 100
    ddh = (eq2_merged["hedged_equity"] - eq2_merged["hedged_equity"].cummax()) / eq2_merged["hedged_equity"].cummax() * 100

    ax2.fill_between(eq1["date"], dd1, 0, alpha=0.4, color="#4a9eff", label="Fixed")
    ax2.fill_between(eq2_merged["date"], dd2, 0, alpha=0.4, color="#ffd700", label="Dynamic")
    ax2.fill_between(eq2_merged["date"], ddh, 0, alpha=0.4, color="#00e676", label="Dyn+SH")
    ax2.set_ylabel("Drawdown %", color="gray")
    ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    plt.tight_layout()
    Path("backtest").mkdir(exist_ok=True)
    plt.savefig("backtest/backtest_dynamic_regime.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    logger.info("Chart saved to backtest/backtest_dynamic_regime.png")

    return {
        "fixed_stats": stats1,
        "dynamic_stats": stats2,
        "hedged_return": h_return,
        "hedged_sharpe": h_sharpe,
        "hedged_maxdd": h_maxdd,
    }


if __name__ == "__main__":
    import sys, yaml
    sys.path.insert(0, ".")
    from data.pipeline import load_processed
    from models.predict import generate_ml_signals
    from pathlib import Path

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    if Path("data/processed/price_features_enriched.parquet").exists():
        df = load_processed("price_features_enriched.parquet")
    else:
        df = load_processed()

    df = generate_ml_signals(df, top_pct=0.20)
    run_dynamic_regime_backtest(df, config)
