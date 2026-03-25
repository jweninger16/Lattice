"""
backtest/backtest_inverse.py
-----------------------------
Tests the "inverse ETF on bad regime days" strategy.
On unfavorable regime days, allocates a portion of cash to SH (inverse S&P 500).
Compares:
  - ML strategy alone (sit out on bad days)
  - ML strategy + SH on bad regime days
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from loguru import logger


def get_sh_data(start: str, end: str) -> pd.DataFrame:
    """Downloads SH (ProShares Short S&P 500) daily returns."""
    logger.info("Downloading SH (inverse S&P 500) data...")
    sh = yf.download("SH", start=start, end=end, auto_adjust=True, progress=False)
    # Flatten multi-level columns if present
    if isinstance(sh.columns, pd.MultiIndex):
        sh.columns = [col[0] for col in sh.columns]
    sh = sh[["Close"]].rename(columns={"Close": "sh_close"})
    sh.index.name = "date"
    sh = sh.reset_index()
    sh["date"] = pd.to_datetime(sh["date"]).dt.tz_localize(None)
    sh["sh_ret"] = sh["sh_close"].pct_change()
    return sh


def run_inverse_backtest(
    ml_signals_df: pd.DataFrame,
    config: dict,
    sh_allocation: float = 0.50,   # % of portfolio in SH on bad days
) -> dict:
    """
    Runs two backtests side by side:
    1. ML only (sit out on bad regime days)
    2. ML + SH hedge on bad regime days
    """
    from backtest.backtest import SwingBacktester, print_stats

    # ── Run baseline ML backtest ────────────────────────────────────────
    logger.info("Running baseline ML backtest (no hedge)...")
    bt = SwingBacktester(config)
    baseline = bt.run(ml_signals_df)
    baseline_equity = baseline["equity"].copy().reset_index()

    # ── Get SH data ─────────────────────────────────────────────────────
    dates = sorted(ml_signals_df["date"].unique())
    start = pd.Timestamp(dates[0]).strftime("%Y-%m-%d")
    end   = pd.Timestamp(dates[-1]).strftime("%Y-%m-%d")
    sh_df = get_sh_data(start, end)

    # ── Get regime days ─────────────────────────────────────────────────
    regime = ml_signals_df.groupby("date")["regime_ok"].first().reset_index()
    regime["date"] = pd.to_datetime(regime["date"])

    # ── Build hedged equity curve ────────────────────────────────────────
    # On unfavorable regime days: uninvested cash earns SH return * allocation
    # On favorable regime days: normal ML strategy

    equity_curve = baseline_equity.copy()
    hedged_equity = [config["backtest"]["initial_capital"]]
    dates_list = sorted(equity_curve["date"].unique())

    # Merge baseline daily returns with regime and SH
    eq = equity_curve.copy()
    eq["date"] = pd.to_datetime(eq["date"]).dt.tz_localize(None)
    eq = eq.merge(regime, on="date", how="left")
    eq = eq.merge(sh_df[["date","sh_ret"]], on="date", how="left")
    eq["sh_ret"] = eq["sh_ret"].fillna(0)
    eq["regime_ok"] = eq["regime_ok"].fillna(1)

    # Calculate daily portfolio value for hedged strategy
    # Baseline already handles favorable days correctly
    # On unfavorable days: add SH return on the uninvested portion
    capital = config["backtest"]["initial_capital"]
    hedged_values = []

    prev_baseline = capital
    prev_hedged   = capital

    for _, row in eq.iterrows():
        # How much was invested vs cash in baseline on this day?
        baseline_val = row["equity"] if "equity" in eq.columns else prev_baseline

        # On unfavorable days, apply SH return to sh_allocation of portfolio
        if row["regime_ok"] == 0 and not pd.isna(row["sh_ret"]):
            hedge_gain = prev_hedged * sh_allocation * row["sh_ret"]
            hedged_val = prev_hedged + (baseline_val - prev_baseline) + hedge_gain
        else:
            hedged_val = prev_hedged + (baseline_val - prev_baseline)

        hedged_values.append(hedged_val)
        prev_baseline = baseline_val
        prev_hedged   = hedged_val

    eq["hedged_value"] = hedged_values

    # ── Calculate hedged stats ───────────────────────────────────────────
    final_hedged  = eq["hedged_value"].iloc[-1]
    initial       = config["backtest"]["initial_capital"]
    n_years       = len(eq) / 252

    hedged_return = (final_hedged / initial - 1) * 100
    hedged_cagr   = ((final_hedged / initial) ** (1/n_years) - 1) * 100

    daily_rets    = eq["hedged_value"].pct_change().dropna()
    hedged_sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252)

    roll_max      = eq["hedged_value"].cummax()
    drawdowns     = (eq["hedged_value"] - roll_max) / roll_max * 100
    hedged_maxdd  = drawdowns.min()

    baseline_return = (eq["equity"].iloc[-1] / initial - 1) * 100 if "equity" in eq.columns else baseline["stats"]["total_return_pct"]

    # ── Count regime days ────────────────────────────────────────────────
    n_unfavorable = (regime["regime_ok"] == 0).sum()
    n_favorable   = (regime["regime_ok"] == 1).sum()

    # ── Print comparison ─────────────────────────────────────────────────
    b = baseline["stats"]
    print("\n" + "="*60)
    print("  INVERSE ETF HEDGE — BACKTEST COMPARISON")
    print("="*60)
    print(f"  SH allocation on bad regime days: {sh_allocation*100:.0f}% of portfolio")
    print(f"  Unfavorable regime days: {n_unfavorable} ({n_unfavorable/(n_favorable+n_unfavorable)*100:.0f}% of time)")
    print()
    print(f"  {'Metric':<22} {'ML Only':>12} {'ML + SH Hedge':>14}")
    print(f"  {'-'*50}")
    print(f"  {'Total Return':<22} {b.get('total_return_pct',0):>11.1f}% {hedged_return:>13.1f}%")
    print(f"  {'CAGR':<22} {b.get('cagr_pct',0):>11.1f}% {hedged_cagr:>13.1f}%")
    print(f"  {'Sharpe Ratio':<22} {b.get('sharpe',0):>12.2f} {hedged_sharpe:>14.2f}")
    print(f"  {'Max Drawdown':<22} {b.get('max_drawdown_pct',0):>11.1f}% {hedged_maxdd:>13.1f}%")
    print(f"  {'Final Capital':<22} ${b.get('final_capital',initial):>10,.0f} ${final_hedged:>12,.0f}")
    print("="*60 + "\n")

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                              facecolor="#0d1117", gridspec_kw={"height_ratios": [3,1]})
    fig.suptitle("ML Strategy vs ML + SH Hedge", color="white", fontsize=14, fontweight="bold")

    ax1, ax2 = axes
    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="gray")
        ax.spines[:].set_color("#333")

    eq["date"] = pd.to_datetime(eq["date"]).dt.tz_localize(None)

    # Equity curves
    if "equity" in eq.columns:
        ax1.plot(eq["date"], eq["equity"], color="#4a9eff", linewidth=1.5,
                 label=f"ML Only (+{b.get('total_return_pct',0):.1f}%)")
    ax1.plot(eq["date"], eq["hedged_value"], color="#00e676", linewidth=1.5,
             label=f"ML + SH Hedge (+{hedged_return:.1f}%)")
    ax1.axhline(initial, color="#555", linewidth=0.8, linestyle="--")

    # Shade unfavorable regime periods
    unfav = regime[regime["regime_ok"] == 0]["date"]
    for d in unfav:
        ax1.axvspan(d, d + pd.Timedelta(days=1), alpha=0.08, color="red")

    ax1.set_ylabel("Portfolio Value ($)", color="gray")
    ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Drawdown comparison
    if "equity" in eq.columns:
        bl_dd = (eq["equity"] - eq["equity"].cummax()) / eq["equity"].cummax() * 100
        ax2.fill_between(eq["date"], bl_dd, 0, alpha=0.5, color="#4a9eff", label="ML Only")
    ax2.fill_between(eq["date"], drawdowns, 0, alpha=0.5, color="#00e676", label="ML + SH")
    ax2.set_ylabel("Drawdown %", color="gray")
    ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    plt.tight_layout()
    Path("backtest").mkdir(exist_ok=True)
    plt.savefig("backtest/backtest_inverse_comparison.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    logger.info("Chart saved to backtest/backtest_inverse_comparison.png")

    return {
        "baseline_stats": b,
        "hedged_return": hedged_return,
        "hedged_cagr": hedged_cagr,
        "hedged_sharpe": hedged_sharpe,
        "hedged_maxdd": hedged_maxdd,
        "equity_df": eq,
    }


if __name__ == "__main__":
    import sys, yaml
    sys.path.insert(0, ".")
    from data.pipeline import load_processed
    from models.predict import generate_ml_signals

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    df = load_processed()
    df = generate_ml_signals(df, top_pct=0.20)

    run_inverse_backtest(df, config, sh_allocation=0.50)
