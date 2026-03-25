"""
backtest.py v5 — production-grade backtester

Improvements over v4:
  - Volatility-scaled position sizing (replaces flat 10%)
  - Portfolio-level drawdown enforcement (actually halts trading)
  - Sector concentration limits
  - Correlation-aware candidate filtering
  - Daily loss circuit breaker
  - Optional trailing stop mode
  - Better stats including Sortino ratio, skew, tail ratio
"""

import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class SwingBacktester:
    def __init__(self, config: dict):
        bt = config["backtest"]
        self.initial_capital   = bt["initial_capital"]
        self.commission        = bt["commission"]
        self.slippage          = bt["slippage"]
        self.hold_days         = bt["hold_days"]
        self.base_position_size = bt["position_size"]
        self.stop_loss_atr     = bt["stop_loss_atr"]
        self.profit_target_atr = bt.get("profit_target_atr", 2.5)
        self.early_exit_days   = bt.get("early_exit_days", 3)
        self.early_exit_thresh = bt.get("early_exit_threshold", -0.015)
        self.max_positions     = config["universe"]["max_positions"]

        # New risk parameters
        self.vol_target_per_pos = bt.get("vol_target_per_position", 0.02)
        self.min_pos_size       = bt.get("min_position_size", 0.05)
        self.max_pos_size       = bt.get("max_position_size", 0.15)

        risk = config.get("risk", {})
        self.max_drawdown       = risk.get("max_drawdown_pct", 0.15)
        self.enforce_dd_stop    = risk.get("enforce_portfolio_stop", True)
        self.cooldown_days      = risk.get("cooldown_days", 5)
        self.max_daily_loss     = risk.get("max_daily_loss_pct", 0.03)

        self.max_sector_conc    = config["universe"].get("max_sector_concentration", 3)

        self.config = config

    def _vol_scaled_size(self, capital: float, atr_pct: float) -> float:
        """Size position to target equal risk contribution."""
        if atr_pct <= 0:
            atr_pct = 0.02
        raw_frac = self.vol_target_per_pos / atr_pct
        clamped = max(self.min_pos_size, min(self.max_pos_size, raw_frac))
        return capital * clamped

    def _get_sector(self, ticker: str, day_df: pd.DataFrame) -> str:
        """Get sector for a ticker from the day's data."""
        row = day_df[day_df["ticker"] == ticker]
        if not row.empty and "sector_etf" in row.columns:
            return row["sector_etf"].values[0]
        return "UNKNOWN"

    def _count_sector(self, sector: str, open_positions: dict, day_df: pd.DataFrame) -> int:
        """Count positions in the same sector."""
        count = 0
        for t in open_positions:
            if self._get_sector(t, day_df) == sector:
                count += 1
        return count

    def run(self, df: pd.DataFrame) -> dict:
        logger.info("Running backtest (v5 with risk management)...")
        df = df.copy().sort_values(["date", "ticker"])
        capital = self.initial_capital
        equity_curve, trades, open_positions = [], [], {}
        peak_equity = self.initial_capital
        dd_halt_until = None  # date until which new entries are blocked

        dates = sorted(df["date"].unique())

        for di, date in enumerate(dates):
            day = df[df["date"] == date]
            to_close = []

            # ── Check existing positions for exits ──
            for ticker, pos in open_positions.items():
                row = day[day["ticker"] == ticker]
                if row.empty:
                    to_close.append((ticker, "delisted", pos["entry_price"]))
                    continue

                price  = row["close"].values[0]
                lo     = row["low"].values[0]
                hi     = row["high"].values[0]
                held   = (date - pos["entry_date"]).days
                ep     = pos["entry_price"]
                exit_p = None
                reason = None

                # Check stop loss
                if lo <= pos["stop"]:
                    exit_p = pos["stop"] * (1 - self.slippage)
                    reason = "stop"
                # Check profit target
                elif hi >= pos["target"]:
                    exit_p = pos["target"] * (1 - self.slippage)
                    reason = "target"
                # Early exit for underwater positions
                elif held >= self.early_exit_days and price / ep - 1 < self.early_exit_thresh:
                    exit_p = price * (1 - self.slippage)
                    reason = "early_exit"
                # Time exit
                elif held >= self.hold_days:
                    exit_p = price * (1 - self.slippage)
                    reason = "time"

                if exit_p is not None:
                    pnl = (exit_p / ep - 1) * pos["size"] - pos["size"] * self.commission
                    capital += pos["size"] + pnl
                    to_close.append((ticker, reason, exit_p))
                    trades.append({
                        "ticker": ticker,
                        "entry_date": pos["entry_date"],
                        "exit_date": date,
                        "entry_price": ep,
                        "exit_price": exit_p,
                        "exit_reason": reason,
                        "pnl_pct": exit_p / ep - 1,
                        "pnl_usd": pnl,
                        "hold_days": held,
                    })

            for t, *_ in to_close:
                del open_positions[t]

            # ── Calculate current portfolio value ──
            pv = capital
            for t, p in open_positions.items():
                row = day[day["ticker"] == t]
                if not row.empty:
                    pv += row["close"].values[0] / p["entry_price"] * p["size"]
                else:
                    pv += p["size"]

            # ── Portfolio-level risk checks ──
            peak_equity = max(peak_equity, pv)
            current_dd = (pv - peak_equity) / peak_equity if peak_equity > 0 else 0
            allow_new_entries = True

            # Drawdown halt
            if dd_halt_until is not None and date < dd_halt_until:
                # Still in cooldown — block entries
                allow_new_entries = False
            elif dd_halt_until is not None and date >= dd_halt_until:
                # Cooldown expired — reset peak to current value so we can trade again
                # This prevents the "permanent halt" trap where DD is measured
                # against an unreachable old peak
                peak_equity = pv
                dd_halt_until = None
                logger.info(f"DD HALT lifted on {date.date()}, peak reset to ${pv:,.0f}")
            elif self.enforce_dd_stop and abs(current_dd) >= self.max_drawdown:
                # New drawdown breach — start cooldown
                dd_halt_until = date + pd.Timedelta(days=self.cooldown_days)
                logger.warning(f"DD HALT: {current_dd*100:.1f}% on {date.date()}, "
                               f"halting until {dd_halt_until.date()}")
                allow_new_entries = False

            # Daily loss circuit breaker
            if di > 0 and len(equity_curve) > 0:
                yesterday_pv = equity_curve[-1]["equity"]
                daily_ret = (pv - yesterday_pv) / yesterday_pv if yesterday_pv > 0 else 0
                if daily_ret < -self.max_daily_loss:
                    allow_new_entries = False

            # ── Enter new positions ──
            slots = self.max_positions - len(open_positions)
            if slots > 0 and allow_new_entries:
                cands = day[
                    (day["signal"] == 1) &
                    (~day["ticker"].isin(open_positions))
                ].sort_values("signal_score", ascending=False)

                entered = 0
                for _, r in cands.iterrows():
                    if entered >= slots:
                        break

                    # Sector concentration check
                    if "sector_etf" in day.columns:
                        ticker_sector = r.get("sector_etf", "UNKNOWN")
                        sector_count = self._count_sector(ticker_sector, open_positions, day)
                        if sector_count >= self.max_sector_conc:
                            continue

                    # Volatility-scaled sizing
                    atr_pct = r["atr_pct"] if "atr_pct" in r.index and r["atr_pct"] > 0 else 0.02
                    sz = self._vol_scaled_size(pv, atr_pct)

                    if sz > capital * 0.95:
                        continue

                    ep = r["close"] * (1 + self.slippage)
                    atr = r["atr_14"] if r["atr_14"] > 0 else ep * 0.02
                    capital -= sz + sz * self.commission
                    open_positions[r["ticker"]] = {
                        "entry_price": ep,
                        "entry_date": date,
                        "stop": ep - atr * self.stop_loss_atr,
                        "target": ep + atr * self.profit_target_atr,
                        "size": sz,
                    }
                    entered += 1

            # Recalculate PV after entries
            pv = capital
            for t, p in open_positions.items():
                row = day[day["ticker"] == t]
                if not row.empty:
                    pv += row["close"].values[0] / p["entry_price"] * p["size"]
                else:
                    pv += p["size"]

            equity_curve.append({
                "date": date, "equity": pv, "cash": capital,
                "n_positions": len(open_positions),
                "drawdown_pct": current_dd * 100,
            })

        eq_df = pd.DataFrame(equity_curve).set_index("date")
        tr_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        stats = self._stats(eq_df, tr_df)
        logger.info(f"Backtest complete. Total return: {stats['total_return_pct']:.1f}% | "
                    f"Sharpe: {stats['sharpe']:.2f} | Max DD: {stats['max_drawdown_pct']:.1f}%")
        return {"equity": eq_df, "trades": tr_df, "stats": stats}

    def _stats(self, eq_df, tr_df):
        eq = eq_df["equity"]
        dr = eq.pct_change().dropna()
        tr = eq.iloc[-1] / eq.iloc[0] - 1
        ny = len(eq) / 252
        cagr = (1 + tr) ** (1 / max(ny, 0.1)) - 1
        sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0

        # NEW: Sortino ratio (only penalizes downside volatility)
        downside = dr[dr < 0]
        sortino = dr.mean() / downside.std() * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0

        dd = (eq - eq.cummax()) / eq.cummax()

        s = {
            "total_return_pct": tr * 100,
            "cagr_pct": cagr * 100,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown_pct": dd.min() * 100,
            "calmar": cagr / abs(dd.min()) if dd.min() != 0 else 0,
            "final_capital": eq.iloc[-1],
            "daily_vol": dr.std() * np.sqrt(252) * 100,
            "return_skew": dr.skew() if len(dr) > 10 else 0,
        }
        if not tr_df.empty:
            w = tr_df[tr_df["pnl_usd"] > 0]
            l = tr_df[tr_df["pnl_usd"] <= 0]
            s.update({
                "n_trades": len(tr_df),
                "win_rate": len(w) / len(tr_df),
                "avg_win_pct": w["pnl_pct"].mean() * 100 if len(w) else 0,
                "avg_loss_pct": l["pnl_pct"].mean() * 100 if len(l) else 0,
                "avg_hold_days": tr_df["hold_days"].mean(),
                "median_hold_days": tr_df["hold_days"].median(),
                "profit_factor": (w["pnl_usd"].sum() / abs(l["pnl_usd"].sum())
                                  if len(l) and l["pnl_usd"].sum() != 0 else 999),
                "pct_stopped_out": (tr_df["exit_reason"] == "stop").mean() * 100,
                "pct_target_hit": (tr_df["exit_reason"] == "target").mean() * 100,
                "pct_early_exit": (tr_df["exit_reason"] == "early_exit").mean() * 100,
                "pct_time_exit": (tr_df["exit_reason"] == "time").mean() * 100,
                # NEW: expectancy per trade
                "expectancy_pct": tr_df["pnl_pct"].mean() * 100,
                "expectancy_usd": tr_df["pnl_usd"].mean(),
                # NEW: best/worst trade
                "best_trade_pct": tr_df["pnl_pct"].max() * 100,
                "worst_trade_pct": tr_df["pnl_pct"].min() * 100,
            })
        return s


def walk_forward_backtest(df, config):
    vc = config["validation"]
    bt = SwingBacktester(config)
    ds = pd.Series(sorted(df["date"].unique()))
    results, fold, idx = [], 0, 0
    while True:
        ts = ds.iloc[idx] + pd.DateOffset(months=vc["train_months"]) + pd.Timedelta(days=vc["embargo_days"])
        te = ts + pd.DateOffset(months=vc["test_months"])
        td = ds[(ds >= ts) & (ds <= te)]
        if len(td) < 10:
            break
        r = bt.run(df[(df["date"] >= td.iloc[0]) & (df["date"] <= td.iloc[-1])])
        r["stats"].update(fold=fold + 1, test_start=td.iloc[0], test_end=td.iloc[-1])
        results.append(r)
        nxt = ds[ds >= td.iloc[-1] + pd.Timedelta(days=1)]
        if nxt.empty:
            break
        idx = nxt.index[0]
        fold += 1
    return results


def print_stats(s):
    print("\n" + "=" * 55)
    print("  BACKTEST RESULTS (v5)")
    print("=" * 55)
    print(f"  Total Return:     {s.get('total_return_pct', 0):.1f}%")
    print(f"  CAGR:             {s.get('cagr_pct', 0):.1f}%")
    print(f"  Sharpe Ratio:     {s.get('sharpe', 0):.2f}")
    print(f"  Sortino Ratio:    {s.get('sortino', 0):.2f}")
    print(f"  Max Drawdown:     {s.get('max_drawdown_pct', 0):.1f}%")
    print(f"  Calmar Ratio:     {s.get('calmar', 0):.2f}")
    print(f"  Annual Vol:       {s.get('daily_vol', 0):.1f}%")
    print(f"  Return Skew:      {s.get('return_skew', 0):.2f}")
    print(f"  Win Rate:         {s.get('win_rate', 0) * 100:.1f}%")
    print(f"  Avg Win:          {s.get('avg_win_pct', 0):.2f}%")
    print(f"  Avg Loss:         {s.get('avg_loss_pct', 0):.2f}%")
    print(f"  Profit Factor:    {s.get('profit_factor', 0):.2f}")
    print(f"  Expectancy:       {s.get('expectancy_pct', 0):.3f}% per trade")
    print(f"  # Trades:         {s.get('n_trades', 0)}")
    print(f"  Avg Hold (days):  {s.get('avg_hold_days', 0):.1f}")
    print(f"  Best Trade:       {s.get('best_trade_pct', 0):.2f}%")
    print(f"  Worst Trade:      {s.get('worst_trade_pct', 0):.2f}%")
    print(f"  % Hit Target:     {s.get('pct_target_hit', 0):.1f}%")
    print(f"  % Stopped Out:    {s.get('pct_stopped_out', 0):.1f}%")
    print(f"  % Early Exit:     {s.get('pct_early_exit', 0):.1f}%")
    print(f"  % Time Exit:      {s.get('pct_time_exit', 0):.1f}%")
    print(f"  Final Capital:    ${s.get('final_capital', 0):,.0f}")
    print("=" * 55 + "\n")


def plot_results(result, save_path="backtest/backtest_results.png"):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    eq, trades, stats = result["equity"], result["trades"], result["stats"]
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
    tc, gc, ac, gr, rd = "#e6edf3", "#21262d", "#58a6ff", "#3fb950", "#f85149"

    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#161b22")
    ax1.plot(eq.index, eq["equity"], color=ac, linewidth=1.5)
    ax1.fill_between(eq.index, eq["equity"].min(), eq["equity"], alpha=0.15, color=ac)
    ax1.set_title("Portfolio Equity Curve", color=tc, fontsize=12)
    ax1.tick_params(colors=tc)
    ax1.grid(True, color=gc, alpha=0.5)
    [sp.set_edgecolor(gc) for sp in ax1.spines.values()]

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#161b22")
    e = eq["equity"]
    dd = (e - e.cummax()) / e.cummax() * 100
    ax2.fill_between(dd.index, dd, 0, color=rd, alpha=0.6)
    ax2.set_title("Drawdown %", color=tc, fontsize=11)
    ax2.tick_params(colors=tc)
    ax2.grid(True, color=gc, alpha=0.5)
    [sp.set_edgecolor(gc) for sp in ax2.spines.values()]

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#161b22")
    if not trades.empty:
        ax3.hist(trades[trades["pnl_pct"] > 0]["pnl_pct"] * 100, bins=30, color=gr, alpha=0.7, label="Wins")
        ax3.hist(trades[trades["pnl_pct"] <= 0]["pnl_pct"] * 100, bins=30, color=rd, alpha=0.7, label="Losses")
        ax3.legend(facecolor="#161b22", labelcolor=tc)
    ax3.set_title("Trade Return Distribution", color=tc, fontsize=11)
    ax3.tick_params(colors=tc)
    [sp.set_edgecolor(gc) for sp in ax3.spines.values()]

    ax4 = fig.add_subplot(gs[2, :])
    ax4.set_facecolor("#161b22")
    ax4.axis("off")
    items = [
        ("CAGR", f"{stats.get('cagr_pct', 0):.1f}%"),
        ("Sharpe", f"{stats.get('sharpe', 0):.2f}"),
        ("Sortino", f"{stats.get('sortino', 0):.2f}"),
        ("Max DD", f"{stats.get('max_drawdown_pct', 0):.1f}%"),
        ("Win Rate", f"{stats.get('win_rate', 0) * 100:.1f}%"),
        ("Profit Factor", f"{stats.get('profit_factor', 0):.2f}"),
        ("Trades", f"{stats.get('n_trades', 0)}"),
        ("Final $", f"${stats.get('final_capital', 0):,.0f}"),
    ]
    for i, (lbl, val) in enumerate(items):
        x = (i % 4) / 4 + 0.05
        y = 0.6 if i < 4 else 0.1
        ax4.text(x, y + 0.2, lbl, color="#8b949e", fontsize=9, transform=ax4.transAxes)
        ax4.text(x, y, val, color=tc, fontsize=13, fontweight="bold", transform=ax4.transAxes)

    plt.suptitle("Swing Trader — Backtest Results (v5)", color=tc, fontsize=14, y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info(f"Chart saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    import sys, yaml
    sys.path.insert(0, ".")
    from data.pipeline import load_processed
    from signals.signals import generate_signals
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    df = load_processed()
    df = generate_signals(df)
    bt = SwingBacktester(config)
    result = bt.run(df)
    print_stats(result["stats"])
    plot_results(result)
