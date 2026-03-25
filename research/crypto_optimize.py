"""
research/crypto_optimize.py
-----------------------------
Parameter sweep for the crypto strategy.
Tests different trail stops, breakout periods, RSI thresholds, etc.
Finds the best configuration before paper trading.

Usage:
    python research/crypto_optimize.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
from itertools import product


# ═══════════════════════════════════════════════════════════════════════
# Indicators
# ═══════════════════════════════════════════════════════════════════════

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


# ═══════════════════════════════════════════════════════════════════════
# Backtest with configurable params
# ═══════════════════════════════════════════════════════════════════════

def backtest(data, breakout_bars=20, rsi_oversold=30, rsi_overbought=70,
             trail_mult=1.5, min_rvol=1.3, max_hold_bars=96,
             strategies=("breakout", "rsi")):
    """Run backtest with given parameters. Returns list of trades."""

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    close = data["Close"].values
    high = data["High"].values
    low = data["Low"].values
    volume = data["Volume"].values

    rsi = compute_rsi(pd.Series(close), 14).values
    atr = compute_atr(pd.Series(high), pd.Series(low), pd.Series(close), 14).values
    avg_vol = pd.Series(volume).rolling(20).mean().values

    trades = []
    in_pos = False
    entry_price = entry_bar = 0
    strategy = ""
    highest = trail_stop = trail_amt = 0

    start = max(breakout_bars + 1, 22)

    for i in range(start, len(close)):
        p = close[i]
        h = high[i]
        lo = low[i]
        r = rsi[i]
        a = atr[i]
        v = volume[i]
        av = avg_vol[i]

        if np.isnan(r) or np.isnan(a) or a <= 0:
            continue

        rvol = v / av if av > 0 else 1.0

        if in_pos:
            if h > highest:
                highest = h
                trail_stop = highest - trail_amt

            exit_p = None
            exit_r = None

            if lo <= trail_stop:
                exit_p = trail_stop
                exit_r = "trail"
            elif strategy == "rsi" and r >= rsi_overbought:
                exit_p = p
                exit_r = "rsi_ob"
            elif (i - entry_bar) >= max_hold_bars:
                exit_p = p
                exit_r = "time"

            if exit_p:
                pnl = ((exit_p - entry_price) / entry_price) * 100
                trades.append({
                    "strategy": strategy,
                    "pnl_pct": round(pnl, 3),
                    "exit_reason": exit_r,
                    "hold_bars": i - entry_bar,
                })
                in_pos = False
                continue

        if not in_pos:
            # Breakout
            if "breakout" in strategies:
                lbh = max(high[i - breakout_bars:i])
                if h > lbh and rvol >= min_rvol:
                    entry_price = p
                    entry_bar = i
                    strategy = "breakout"
                    trail_amt = a * trail_mult
                    highest = p
                    trail_stop = p - trail_amt
                    in_pos = True
                    continue

            # RSI
            if "rsi" in strategies:
                if r <= rsi_oversold:
                    entry_price = p
                    entry_bar = i
                    strategy = "rsi"
                    trail_amt = a * trail_mult
                    highest = p
                    trail_stop = p - trail_amt
                    in_pos = True
                    continue

    return trades


def score_trades(trades):
    """Score a set of trades."""
    if not trades:
        return {"trades": 0, "wr": 0, "avg_pnl": 0, "pf": 0, "total_pnl": 0}

    pnls = [t["pnl_pct"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    wr = len(wins) / len(pnls) * 100 if pnls else 0
    avg_pnl = np.mean(pnls) if pnls else 0
    total_pnl = sum(pnls)
    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    pf = gross_win / gross_loss

    return {
        "trades": len(trades),
        "wr": round(wr, 1),
        "avg_pnl": round(avg_pnl, 3),
        "pf": round(pf, 2),
        "total_pnl": round(total_pnl, 2),
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  CRYPTO PARAMETER OPTIMIZER")
    print("=" * 70)

    # Fetch data
    all_data = {}
    for ticker in ["BTC-USD", "ETH-USD"]:
        print(f"\n  Fetching {ticker}...")
        data = yf.download(ticker, period="60d", interval="15m",
                           progress=False, auto_adjust=True)
        if data is not None and not data.empty:
            all_data[ticker] = data
            print(f"  Got {len(data)} bars")

    if not all_data:
        print("  No data. Exiting.")
        return

    # Parameter grid
    param_grid = {
        "breakout_bars": [10, 20, 30, 40],
        "rsi_oversold":  [20, 25, 30],
        "rsi_overbought": [70, 75, 80],
        "trail_mult":    [1.5, 2.0, 2.5, 3.0, 4.0],
        "min_rvol":      [1.0, 1.3, 1.5, 2.0],
    }

    # Also test strategies independently
    strategy_configs = [
        ("both", ("breakout", "rsi")),
        ("breakout_only", ("breakout",)),
        ("rsi_only", ("rsi",)),
    ]

    print(f"\n  Testing {len(param_grid['breakout_bars'])} x "
          f"{len(param_grid['rsi_oversold'])} x "
          f"{len(param_grid['rsi_overbought'])} x "
          f"{len(param_grid['trail_mult'])} x "
          f"{len(param_grid['min_rvol'])} x "
          f"{len(strategy_configs)} = "
          f"{len(param_grid['breakout_bars']) * len(param_grid['rsi_oversold']) * len(param_grid['rsi_overbought']) * len(param_grid['trail_mult']) * len(param_grid['min_rvol']) * len(strategy_configs)} "
          f"combinations...")

    results = []
    total_combos = 0

    for strat_name, strat_val in strategy_configs:
        for bb in param_grid["breakout_bars"]:
            for rsi_lo in param_grid["rsi_oversold"]:
                for rsi_hi in param_grid["rsi_overbought"]:
                    for tm in param_grid["trail_mult"]:
                        for rv in param_grid["min_rvol"]:
                            total_combos += 1

                            all_trades = []
                            for ticker, data in all_data.items():
                                trades = backtest(
                                    data, breakout_bars=bb,
                                    rsi_oversold=rsi_lo,
                                    rsi_overbought=rsi_hi,
                                    trail_mult=tm, min_rvol=rv,
                                    strategies=strat_val,
                                )
                                all_trades.extend(trades)

                            scores = score_trades(all_trades)

                            # Only keep configs with enough trades
                            if scores["trades"] >= 20:
                                results.append({
                                    "strategy": strat_name,
                                    "breakout_bars": bb,
                                    "rsi_lo": rsi_lo,
                                    "rsi_hi": rsi_hi,
                                    "trail_mult": tm,
                                    "min_rvol": rv,
                                    **scores,
                                })

    print(f"  Tested {total_combos} combinations")
    print(f"  {len(results)} configs had 20+ trades")

    if not results:
        print("  No viable configs found.")
        return

    rdf = pd.DataFrame(results)

    # Sort by profit factor (most reliable metric)
    rdf = rdf.sort_values("pf", ascending=False)

    # Top 10 profitable configs
    profitable = rdf[rdf["pf"] > 1.0]

    print(f"\n  {len(profitable)} configs are PROFITABLE (PF > 1.0)")

    if len(profitable) > 0:
        print(f"\n  TOP 10 CONFIGURATIONS (by profit factor)")
        print(f"  {'=' * 65}")
        print(f"  {'Strategy':<15} {'BB':>3} {'RSI':>7} {'Trail':>5} {'RVol':>4} "
              f"{'Trades':>6} {'WR':>5} {'AvgPnL':>7} {'PF':>5} {'TotalPnL':>9}")
        print(f"  {'-' * 65}")

        for _, row in profitable.head(10).iterrows():
            print(f"  {row['strategy']:<15} {row['breakout_bars']:>3} "
                  f"{row['rsi_lo']:>3}/{row['rsi_hi']:<3} "
                  f"{row['trail_mult']:>5.1f} {row['min_rvol']:>4.1f} "
                  f"{row['trades']:>6} {row['wr']:>4.1f}% "
                  f"{row['avg_pnl']:>+6.3f}% {row['pf']:>5.2f} "
                  f"{row['total_pnl']:>+8.2f}%")

        # Best config
        best = profitable.iloc[0]
        print(f"\n  RECOMMENDED CONFIG:")
        print(f"  {'=' * 65}")
        print(f"  Strategy:      {best['strategy']}")
        print(f"  Breakout bars: {int(best['breakout_bars'])}")
        print(f"  RSI:           {int(best['rsi_lo'])}/{int(best['rsi_hi'])}")
        print(f"  Trail mult:    {best['trail_mult']}x ATR")
        print(f"  Min rvol:      {best['min_rvol']}x")
        print(f"  Win rate:      {best['wr']}%")
        print(f"  Profit factor: {best['pf']}")
        print(f"  Avg P&L:       {best['avg_pnl']:+.3f}%")
        print(f"  Total trades:  {int(best['trades'])} over 60 days")

    else:
        print("\n  No profitable configs found with these parameters.")
        print("  The current market regime may not suit these strategies.")
        print("  Consider:")
        print("    - Different entry signals (MACD crossover, Bollinger bands)")
        print("    - Longer timeframes (1h, 4h candles)")
        print("    - Different assets (SOL, AVAX)")

    # Also show worst to understand what NOT to do
    worst = rdf.tail(5)
    print(f"\n  WORST 5 CONFIGURATIONS (avoid these)")
    print(f"  {'-' * 65}")
    for _, row in worst.iterrows():
        print(f"  {row['strategy']:<15} BB={int(row['breakout_bars'])} "
              f"RSI={int(row['rsi_lo'])}/{int(row['rsi_hi'])} "
              f"trail={row['trail_mult']}x rvol={row['min_rvol']}x "
              f"=> PF={row['pf']:.2f} WR={row['wr']:.0f}%")

    # Save full results
    rdf.to_csv("data/crypto_optimize_results.csv", index=False)
    print(f"\n  Full results saved to data/crypto_optimize_results.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
