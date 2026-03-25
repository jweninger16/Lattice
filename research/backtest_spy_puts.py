"""
research/backtest_spy_puts.py
------------------------------
STANDALONE RESEARCH — Does not affect the live trading system.

Backtests a SPY protective put strategy during unfavorable regime periods.
Uses Black-Scholes synthetic pricing since historical options data isn't free.

Strategy:
  - When regime turns UNFAVORABLE: buy a SPY put (5% OTM, 30-day expiry)
  - Allocate a fixed % of portfolio to put premium
  - Hold until expiry or regime turns FAVORABLE (whichever comes first)
  - At exit: put value = max(strike - SPY_price, 0)

Compares:
  1. Cash only during bad regimes (current system)
  2. Cash + SPY puts during bad regimes
  3. Cash + SH during bad regimes (existing comparison)

This uses the regime data from your trained model, so run it after
you've already done: python main.py pipeline && python main.py train

Usage:
    python research/backtest_spy_puts.py
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
from scipy.stats import norm
from datetime import timedelta

sys.path.insert(0, ".")


# ═══════════════════════════════════════════════════════════════════════
# Black-Scholes Option Pricing
# ═══════════════════════════════════════════════════════════════════════

def black_scholes_put(S, K, T, r, sigma):
    """
    Black-Scholes put option price.
    S: current price
    K: strike price
    T: time to expiry in years
    r: risk-free rate
    sigma: annualized volatility
    """
    if T <= 0:
        return max(K - S, 0)
    if sigma <= 0:
        sigma = 0.01

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return max(put, 0)


def put_delta(S, K, T, r, sigma):
    """Put option delta (for P&L attribution)."""
    if T <= 0 or sigma <= 0:
        return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1


# ═══════════════════════════════════════════════════════════════════════
# Regime Detection (reuses your model's regime logic)
# ═══════════════════════════════════════════════════════════════════════

def get_regime_periods(spy_df: pd.DataFrame, breadth_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Determines favorable/unfavorable regime periods.
    Uses SMA50 breadth if available, otherwise uses SPY's own SMA.
    """
    spy = spy_df.copy()
    spy["sma50"] = spy["close"].rolling(50).mean()
    spy["sma20"] = spy["close"].rolling(20).mean()
    spy["above_sma50"] = (spy["close"] > spy["sma50"]).astype(int)

    # Simple regime: SPY above its 50-day SMA = favorable
    # This is a simplified version since we don't have full breadth data here
    spy["regime_ok"] = spy["above_sma50"]

    # Smooth regime (require 2 consecutive days to flip)
    spy["regime_smooth"] = spy["regime_ok"].copy()
    for i in range(2, len(spy)):
        raw = spy.iloc[i]["regime_ok"]
        prev_smooth = spy.iloc[i-1]["regime_smooth"]
        if raw != prev_smooth:
            if spy.iloc[i-1]["regime_ok"] == raw:
                spy.iloc[i, spy.columns.get_loc("regime_smooth")] = raw
            else:
                spy.iloc[i, spy.columns.get_loc("regime_smooth")] = prev_smooth

    spy["regime_ok"] = spy["regime_smooth"]
    return spy


def identify_unfavorable_periods(regime_df: pd.DataFrame) -> list:
    """
    Returns list of (start_date, end_date) for unfavorable regime periods.
    """
    periods = []
    in_bad = False
    start = None

    for i, row in regime_df.iterrows():
        if row["regime_ok"] == 0 and not in_bad:
            in_bad = True
            start = row["date"]
        elif row["regime_ok"] == 1 and in_bad:
            in_bad = False
            periods.append((start, row["date"]))

    # Close open period
    if in_bad:
        periods.append((start, regime_df["date"].iloc[-1]))

    return periods


# ═══════════════════════════════════════════════════════════════════════
# Put Hedge Backtest
# ═══════════════════════════════════════════════════════════════════════

def backtest_put_hedge(
    spy_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    initial_capital: float = 20000,
    put_allocation_pct: float = 0.02,   # spend 2% of portfolio on puts
    otm_pct: float = 0.05,             # 5% out of the money
    put_duration_days: int = 30,        # 30-day puts
    risk_free_rate: float = 0.05,       # ~5% fed funds
) -> dict:
    """
    Backtests three strategies during unfavorable regimes:
    1. Cash only (do nothing)
    2. Cash + SPY puts
    3. Cash + SH (inverse ETF)
    """
    logger.info("Running put hedge backtest...")
    logger.info(f"  Put allocation: {put_allocation_pct*100:.1f}% of portfolio")
    logger.info(f"  Strike: {otm_pct*100:.0f}% OTM")
    logger.info(f"  Duration: {put_duration_days} days")

    spy = spy_df.copy().sort_values("date").reset_index(drop=True)
    regime = regime_df[["date", "regime_ok"]].copy()
    spy = spy.merge(regime, on="date", how="left")
    spy["regime_ok"] = spy["regime_ok"].fillna(1)

    # Compute realized vol for pricing
    spy["ret"] = spy["close"].pct_change()
    spy["realized_vol"] = spy["ret"].rolling(21).std() * np.sqrt(252)
    spy["realized_vol"] = spy["realized_vol"].fillna(0.20)

    # Download SH for comparison
    logger.info("Downloading SH data for comparison...")
    try:
        sh_raw = yf.download("SH",
            start=spy["date"].min().strftime("%Y-%m-%d"),
            end=spy["date"].max().strftime("%Y-%m-%d"),
            auto_adjust=True, progress=False)
        if isinstance(sh_raw.columns, pd.MultiIndex):
            sh_raw.columns = [c[0] for c in sh_raw.columns]
        sh_raw.index = pd.to_datetime(sh_raw.index).tz_localize(None)
        sh_raw = sh_raw[["Close"]].rename(columns={"Close": "sh_close"}).reset_index()
        sh_raw.columns = ["date", "sh_close"]
        sh_raw["sh_ret"] = sh_raw["sh_close"].pct_change().fillna(0)
        spy = spy.merge(sh_raw[["date", "sh_ret"]], on="date", how="left")
        spy["sh_ret"] = spy["sh_ret"].fillna(0)
    except Exception as e:
        logger.warning(f"SH download failed: {e}")
        spy["sh_ret"] = 0

    # ── Run three strategies ────────────────────────────────────────────

    # Strategy 1: Cash only (current system behavior)
    cash_equity = [initial_capital]

    # Strategy 2: Cash + puts during bad regimes
    put_equity = [initial_capital]
    put_trades = []
    active_put = None

    # Strategy 3: Cash + SH during bad regimes
    sh_equity = [initial_capital]
    sh_active = False

    for i in range(1, len(spy)):
        row = spy.iloc[i]
        prev = spy.iloc[i-1]
        date = row["date"]
        price = row["close"]
        vol = row["realized_vol"]
        regime_ok = row["regime_ok"]
        prev_regime = prev["regime_ok"]

        # ── Strategy 1: Cash only ──
        # During bad regime: flat. During good regime: market return.
        if regime_ok == 1:
            cash_ret = row["ret"] if not np.isnan(row["ret"]) else 0
            cash_equity.append(cash_equity[-1] * (1 + cash_ret))
        else:
            cash_equity.append(cash_equity[-1])

        # ── Strategy 2: Cash + puts ──
        put_val = put_equity[-1]

        # Enter put when regime turns bad
        if regime_ok == 0 and prev_regime == 1 and active_put is None:
            premium_budget = put_val * put_allocation_pct
            strike = price * (1 - otm_pct)
            T = put_duration_days / 252

            # Price the put using Black-Scholes
            # Add a vol premium (implied vol is usually higher than realized)
            implied_vol = vol * 1.15  # ~15% vol premium
            put_price_per_share = black_scholes_put(price, strike, T, risk_free_rate, implied_vol)

            if put_price_per_share > 0:
                n_contracts = int(premium_budget / (put_price_per_share * 100))
                n_contracts = max(n_contracts, 1)
                actual_premium = n_contracts * put_price_per_share * 100

                active_put = {
                    "entry_date": date,
                    "entry_price": price,
                    "strike": strike,
                    "expiry_date": date + timedelta(days=put_duration_days),
                    "premium": actual_premium,
                    "n_contracts": n_contracts,
                    "entry_vol": implied_vol,
                }
                put_val -= actual_premium
                logger.debug(f"  PUT BUY: {date.date()} | SPY=${price:.0f} | "
                             f"K=${strike:.0f} | {n_contracts}x | premium=${actual_premium:.0f}")

        # Mark-to-market active put
        if active_put is not None:
            days_left = (active_put["expiry_date"] - date).days
            T_left = max(days_left / 252, 0)

            if days_left <= 0 or regime_ok == 1:
                # Exit: put expires or regime recovered
                intrinsic = max(active_put["strike"] - price, 0)
                proceeds = active_put["n_contracts"] * intrinsic * 100
                pnl = proceeds - active_put["premium"]

                put_trades.append({
                    "entry_date": active_put["entry_date"],
                    "exit_date": date,
                    "entry_spy": active_put["entry_price"],
                    "exit_spy": price,
                    "strike": active_put["strike"],
                    "premium": active_put["premium"],
                    "proceeds": proceeds,
                    "pnl": pnl,
                    "pnl_pct": pnl / active_put["premium"] * 100 if active_put["premium"] > 0 else 0,
                    "reason": "expiry" if days_left <= 0 else "regime_flip",
                    "spy_move_pct": (price / active_put["entry_price"] - 1) * 100,
                })
                put_val += proceeds
                logger.debug(f"  PUT EXIT: {date.date()} | proceeds=${proceeds:.0f} | "
                             f"P&L=${pnl:+.0f}")
                active_put = None
            else:
                # Mark to market (not realized)
                implied_vol = vol * 1.15
                current_put_price = black_scholes_put(price, active_put["strike"],
                                                       T_left, risk_free_rate, implied_vol)
                mtm_value = active_put["n_contracts"] * current_put_price * 100
                # put_val already had premium subtracted, MTM is unrealized
                pass

        # During good regime: market return
        if regime_ok == 1:
            put_ret = row["ret"] if not np.isnan(row["ret"]) else 0
            put_val *= (1 + put_ret)

        put_equity.append(put_val)

        # ── Strategy 3: Cash + SH ──
        sh_val = sh_equity[-1]

        if regime_ok == 1:
            sh_ret = row["ret"] if not np.isnan(row["ret"]) else 0
            sh_val *= (1 + sh_ret)
            sh_active = False
        else:
            # 50% of portfolio in SH during bad regime
            if not np.isnan(row["sh_ret"]):
                sh_val *= (1 + 0.5 * row["sh_ret"])
            sh_active = True

        sh_equity.append(sh_val)

    # ── Compute results ─────────────────────────────────────────────────
    dates = [spy["date"].iloc[0]] + spy["date"].tolist()[1:]

    results = {}
    for name, eq in [("Cash Only", cash_equity), ("Cash + Puts", put_equity), ("Cash + SH", sh_equity)]:
        eq_arr = np.array(eq)
        rets = np.diff(eq_arr) / eq_arr[:-1]
        total_ret = (eq_arr[-1] / eq_arr[0] - 1) * 100
        n_years = len(eq) / 252
        cagr = ((eq_arr[-1] / eq_arr[0]) ** (1/max(n_years, 0.1)) - 1) * 100
        sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        peak = np.maximum.accumulate(eq_arr)
        dd = (eq_arr - peak) / peak
        max_dd = dd.min() * 100

        results[name] = {
            "equity": eq,
            "total_return": total_ret,
            "cagr": cagr,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "final": eq_arr[-1],
        }

    # ── Print comparison ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SPY PUT HEDGE — BACKTEST COMPARISON")
    print("=" * 65)
    print(f"  {'Metric':<22} {'Cash Only':>12} {'Cash+Puts':>12} {'Cash+SH':>12}")
    print(f"  {'-'*58}")
    for metric, key, fmt in [
        ("Total Return", "total_return", ".1f"),
        ("CAGR", "cagr", ".1f"),
        ("Sharpe Ratio", "sharpe", ".2f"),
        ("Max Drawdown", "max_dd", ".1f"),
        ("Final Capital", "final", ",.0f"),
    ]:
        vals = []
        for strat in ["Cash Only", "Cash + Puts", "Cash + SH"]:
            v = results[strat][key]
            if "Capital" in metric:
                vals.append(f"${v:{fmt}}")
            elif "Sharpe" not in metric:
                vals.append(f"{v:{fmt}}%")
            else:
                vals.append(f"{v:{fmt}}")
        print(f"  {metric:<22} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")
    print("=" * 65)

    # Put trade summary
    if put_trades:
        pt_df = pd.DataFrame(put_trades)
        winners = pt_df[pt_df["pnl"] > 0]
        losers = pt_df[pt_df["pnl"] <= 0]
        total_premium = pt_df["premium"].sum()
        total_proceeds = pt_df["proceeds"].sum()

        print(f"\n  PUT TRADE DETAILS:")
        print(f"  Total put trades:     {len(pt_df)}")
        print(f"  Winners:              {len(winners)} ({len(winners)/len(pt_df)*100:.0f}%)")
        print(f"  Total premium spent:  ${total_premium:,.0f}")
        print(f"  Total proceeds:       ${total_proceeds:,.0f}")
        print(f"  Net P&L from puts:    ${total_proceeds - total_premium:+,.0f}")
        print(f"  Avg put P&L:          {pt_df['pnl_pct'].mean():+.1f}%")
        print(f"  Best put trade:       {pt_df['pnl_pct'].max():+.1f}%")
        print(f"  Worst put trade:      {pt_df['pnl_pct'].min():+.1f}%")
        print(f"  Avg SPY move (bad):   {pt_df['spy_move_pct'].mean():+.1f}%")
        print()

        print(f"  {'Entry':<12} {'Exit':<12} {'SPY Move':>9} {'Premium':>9} {'Proceeds':>10} {'P&L':>9} {'Reason'}")
        print(f"  {'-'*72}")
        for _, t in pt_df.iterrows():
            print(f"  {str(t['entry_date'].date()):<12} {str(t['exit_date'].date()):<12} "
                  f"{t['spy_move_pct']:>+8.1f}% ${t['premium']:>7,.0f} ${t['proceeds']:>9,.0f} "
                  f"${t['pnl']:>+8,.0f} {t['reason']}")
    print()

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor="#0d1117",
                              gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("Regime Hedge Comparison: Cash vs Puts vs SH",
                 color="white", fontsize=14, fontweight="bold")

    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="gray")
        ax.spines[:].set_color("#333")

    ax1, ax2 = axes

    # Equity curves
    ax1.plot(dates, results["Cash Only"]["equity"], color="#4a9eff", linewidth=1.5,
             label=f"Cash Only (+{results['Cash Only']['total_return']:.1f}%)")
    ax1.plot(dates, results["Cash + Puts"]["equity"], color="#ffd700", linewidth=1.5,
             label=f"Cash + Puts (+{results['Cash + Puts']['total_return']:.1f}%)")
    ax1.plot(dates, results["Cash + SH"]["equity"], color="#00e676", linewidth=1.5,
             label=f"Cash + SH (+{results['Cash + SH']['total_return']:.1f}%)")
    ax1.axhline(initial_capital, color="#555", linewidth=0.8, linestyle="--")

    # Shade unfavorable periods
    for i in range(1, len(spy)):
        if spy.iloc[i]["regime_ok"] == 0:
            ax1.axvspan(spy.iloc[i]["date"], spy.iloc[i]["date"] + timedelta(days=1),
                        alpha=0.06, color="red")

    ax1.set_ylabel("Portfolio Value ($)", color="gray")
    ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Drawdowns
    for name, color in [("Cash Only", "#4a9eff"), ("Cash + Puts", "#ffd700"), ("Cash + SH", "#00e676")]:
        eq = np.array(results[name]["equity"])
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak * 100
        ax2.fill_between(dates, dd, 0, alpha=0.3, color=color, label=name)

    ax2.set_ylabel("Drawdown %", color="gray")
    ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    plt.tight_layout()
    Path("research").mkdir(exist_ok=True)
    save_path = "research/spy_put_hedge_backtest.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    logger.info(f"Chart saved to {save_path}")

    return {
        "results": results,
        "put_trades": pd.DataFrame(put_trades) if put_trades else pd.DataFrame(),
    }


# ═══════════════════════════════════════════════════════════════════════
# Sensitivity Analysis
# ═══════════════════════════════════════════════════════════════════════

def run_sensitivity(spy_df, regime_df):
    """Tests different put parameters to find optimal settings."""
    print("\n" + "=" * 65)
    print("  SENSITIVITY ANALYSIS — Varying Put Parameters")
    print("=" * 65)
    print(f"  {'Allocation':>11} {'OTM %':>7} {'Days':>6} {'Return':>9} {'Sharpe':>8} {'Max DD':>8}")
    print(f"  {'-'*55}")

    best_sharpe = -999
    best_params = {}

    for alloc in [0.01, 0.02, 0.03, 0.05]:
        for otm in [0.03, 0.05, 0.07, 0.10]:
            for days in [21, 30, 45]:
                result = backtest_put_hedge(
                    spy_df, regime_df,
                    put_allocation_pct=alloc,
                    otm_pct=otm,
                    put_duration_days=days,
                )
                r = result["results"]["Cash + Puts"]
                print(f"  {alloc*100:>10.1f}% {otm*100:>6.0f}% {days:>5}d "
                      f"{r['total_return']:>+8.1f}% {r['sharpe']:>7.2f} {r['max_dd']:>7.1f}%")

                if r["sharpe"] > best_sharpe:
                    best_sharpe = r["sharpe"]
                    best_params = {"alloc": alloc, "otm": otm, "days": days}

    print(f"\n  Best Sharpe: {best_sharpe:.2f}")
    print(f"  Optimal: {best_params['alloc']*100:.1f}% allocation, "
          f"{best_params['otm']*100:.0f}% OTM, {best_params['days']}d expiry")
    print("=" * 65)

    return best_params


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensitivity", action="store_true",
                        help="Run sensitivity analysis on put parameters")
    parser.add_argument("--alloc", type=float, default=0.02,
                        help="Put allocation as fraction (default: 0.02)")
    parser.add_argument("--otm", type=float, default=0.05,
                        help="OTM percentage as fraction (default: 0.05)")
    parser.add_argument("--days", type=int, default=30,
                        help="Put duration in days (default: 30)")
    args = parser.parse_args()

    # Download SPY data
    logger.info("Downloading SPY data...")
    spy_raw = yf.download("SPY", start="2018-01-01", auto_adjust=True, progress=False)
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_raw.columns = [c[0] for c in spy_raw.columns]
    spy_raw.index = pd.to_datetime(spy_raw.index).tz_localize(None)
    spy_raw.index.name = "date"
    spy_df = spy_raw[["Close", "High", "Low"]].rename(
        columns={"Close": "close", "High": "high", "Low": "low"}
    ).reset_index()

    # Compute regime
    logger.info("Computing regime periods...")
    regime_df = get_regime_periods(spy_df)

    unfav_days = (regime_df["regime_ok"] == 0).sum()
    total_days = len(regime_df)
    logger.info(f"Unfavorable days: {unfav_days}/{total_days} ({unfav_days/total_days*100:.0f}%)")

    if args.sensitivity:
        best = run_sensitivity(spy_df, regime_df)
        print(f"\nRunning final backtest with optimal parameters...")
        backtest_put_hedge(spy_df, regime_df,
                           put_allocation_pct=best["alloc"],
                           otm_pct=best["otm"],
                           put_duration_days=best["days"])
    else:
        backtest_put_hedge(spy_df, regime_df,
                           put_allocation_pct=args.alloc,
                           otm_pct=args.otm,
                           put_duration_days=args.days)

    print("Done. Chart saved to research/spy_put_hedge_backtest.png")
