"""
research/crypto_backtest.py
-----------------------------
Backtests the crypto paper trading strategy on historical data.

Tests both strategies:
  1. Momentum breakout (20-bar high on volume)
  2. RSI mean reversion (buy < 30, sell > 70)

Uses 15-minute candles on BTC-USD and ETH-USD.

Usage:
    cd swing_trader_v2
    venv\Scripts\activate
    python research\crypto_backtest.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta


# ═══════════════════════════════════════════════════════════════════════
# Config — matches the paper trader exactly
# ═══════════════════════════════════════════════════════════════════════

TICKERS = ["BTC-USD", "ETH-USD"]
INTERVAL = "15m"
BREAKOUT_BARS = 20
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
ATR_PERIOD = 14
TRAIL_ATR_MULT = 1.5
MIN_RVOL = 1.3
STARTING_CAPITAL = 500
POSITION_PCT = 0.50
MAX_HOLD_BARS = 96   # 24 hours / 15 min = 96 bars


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
# Backtest Engine
# ═══════════════════════════════════════════════════════════════════════

def backtest_ticker(ticker, data):
    """Backtest both strategies on a single ticker."""
    
    if data is None or len(data) < 50:
        print(f"  Not enough data for {ticker}")
        return []

    # Flatten MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    close = data["Close"].values
    high = data["High"].values
    low = data["Low"].values
    volume = data["Volume"].values

    # Pre-compute indicators
    rsi = compute_rsi(pd.Series(close), RSI_PERIOD).values
    atr = compute_atr(
        pd.Series(high), pd.Series(low), pd.Series(close), ATR_PERIOD
    ).values

    # Avg volume (rolling 20)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(20).mean().values

    trades = []
    in_position = False
    entry_price = 0
    entry_bar = 0
    entry_strategy = ""
    highest_since_entry = 0
    trail_stop = 0
    trail_amount = 0

    start_bar = max(BREAKOUT_BARS + 1, RSI_PERIOD + 1, ATR_PERIOD + 1, 21)

    for i in range(start_bar, len(close)):
        current_price = close[i]
        current_high = high[i]
        current_low = low[i]
        current_rsi = rsi[i]
        current_atr = atr[i]
        current_vol = volume[i]
        current_avg_vol = avg_vol[i]

        if np.isnan(current_rsi) or np.isnan(current_atr) or current_atr <= 0:
            continue

        rvol = current_vol / current_avg_vol if current_avg_vol > 0 else 1.0

        # ── In a position: check exits ────────────────────────────
        if in_position:
            # Update trailing stop
            if current_high > highest_since_entry:
                highest_since_entry = current_high
                trail_stop = highest_since_entry - trail_amount

            exit_price = None
            exit_reason = None

            # Trailing stop
            if current_low <= trail_stop:
                exit_price = trail_stop
                exit_reason = "trail"

            # RSI overbought (mean reversion exit)
            elif entry_strategy == "rsi_mean_reversion" and current_rsi >= RSI_OVERBOUGHT:
                exit_price = current_price
                exit_reason = "rsi_overbought"

            # Time exit (24 hours)
            elif (i - entry_bar) >= MAX_HOLD_BARS:
                exit_price = current_price
                exit_reason = "time_exit"

            if exit_price is not None:
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                pnl_usd = (exit_price - entry_price) * (STARTING_CAPITAL * POSITION_PCT / entry_price)
                hold_bars = i - entry_bar
                hold_hours = round(hold_bars * 15 / 60, 1)

                trades.append({
                    "ticker": ticker,
                    "strategy": entry_strategy,
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "exit_reason": exit_reason,
                    "pnl_pct": round(pnl_pct, 2),
                    "pnl_usd": round(pnl_usd, 2),
                    "hold_hours": hold_hours,
                    "rsi_at_entry": round(rsi[entry_bar], 1),
                    "rvol_at_entry": round(rvol, 2),
                    "bar_index": i,
                })

                in_position = False
                continue

        # ── Not in a position: check entries ──────────────────────
        if not in_position:
            # Strategy 1: Momentum breakout
            lookback_high = max(high[i - BREAKOUT_BARS:i])
            if current_high > lookback_high and rvol >= MIN_RVOL:
                entry_price = current_price
                entry_bar = i
                entry_strategy = "momentum_breakout"
                trail_amount = current_atr * TRAIL_ATR_MULT
                highest_since_entry = current_price
                trail_stop = current_price - trail_amount
                in_position = True
                continue

            # Strategy 2: RSI mean reversion
            if current_rsi <= RSI_OVERSOLD:
                entry_price = current_price
                entry_bar = i
                entry_strategy = "rsi_mean_reversion"
                trail_amount = current_atr * TRAIL_ATR_MULT
                highest_since_entry = current_price
                trail_stop = current_price - trail_amount
                in_position = True
                continue

    return trades


def print_results(all_trades):
    """Print formatted backtest results."""
    if not all_trades:
        print("\n  No trades generated. Strategy may need adjustment.")
        return

    df = pd.DataFrame(all_trades)
    
    print("\n" + "=" * 70)
    print("  CRYPTO BACKTEST RESULTS")
    print("=" * 70)

    # Overall
    total = len(df)
    wins = len(df[df["pnl_pct"] > 0])
    losses = len(df[df["pnl_pct"] <= 0])
    win_rate = wins / total * 100 if total > 0 else 0
    avg_pnl = df["pnl_pct"].mean()
    total_pnl = df["pnl_usd"].sum()
    avg_win = df[df["pnl_pct"] > 0]["pnl_pct"].mean() if wins > 0 else 0
    avg_loss = df[df["pnl_pct"] <= 0]["pnl_pct"].mean() if losses > 0 else 0
    profit_factor = abs(df[df["pnl_pct"] > 0]["pnl_usd"].sum() / df[df["pnl_pct"] <= 0]["pnl_usd"].sum()) if losses > 0 and df[df["pnl_pct"] <= 0]["pnl_usd"].sum() != 0 else 999
    avg_hold = df["hold_hours"].mean()
    max_win = df["pnl_pct"].max()
    max_loss = df["pnl_pct"].min()

    print(f"\n  OVERALL ({total} trades)")
    print(f"  {'─' * 40}")
    print(f"  Win rate:       {win_rate:.1f}% ({wins}W / {losses}L)")
    print(f"  Avg P&L:        {avg_pnl:+.2f}%")
    print(f"  Total P&L:      ${total_pnl:+.2f} (on ${STARTING_CAPITAL} capital)")
    print(f"  Avg win:        {avg_win:+.2f}%")
    print(f"  Avg loss:       {avg_loss:+.2f}%")
    print(f"  Profit factor:  {profit_factor:.2f}")
    print(f"  Avg hold time:  {avg_hold:.1f} hours")
    print(f"  Best trade:     {max_win:+.2f}%")
    print(f"  Worst trade:    {max_loss:+.2f}%")

    # By strategy
    for strategy in df["strategy"].unique():
        sdf = df[df["strategy"] == strategy]
        st = len(sdf)
        sw = len(sdf[sdf["pnl_pct"] > 0])
        sl = len(sdf[sdf["pnl_pct"] <= 0])
        swr = sw / st * 100 if st > 0 else 0
        savg = sdf["pnl_pct"].mean()
        stotal = sdf["pnl_usd"].sum()
        savg_w = sdf[sdf["pnl_pct"] > 0]["pnl_pct"].mean() if sw > 0 else 0
        savg_l = sdf[sdf["pnl_pct"] <= 0]["pnl_pct"].mean() if sl > 0 else 0
        spf = abs(sdf[sdf["pnl_pct"] > 0]["pnl_usd"].sum() / sdf[sdf["pnl_pct"] <= 0]["pnl_usd"].sum()) if sl > 0 and sdf[sdf["pnl_pct"] <= 0]["pnl_usd"].sum() != 0 else 999

        label = strategy.replace("_", " ").title()
        print(f"\n  {label.upper()} ({st} trades)")
        print(f"  {'─' * 40}")
        print(f"  Win rate:       {swr:.1f}% ({sw}W / {sl}L)")
        print(f"  Avg P&L:        {savg:+.2f}%")
        print(f"  Total P&L:      ${stotal:+.2f}")
        print(f"  Avg win:        {savg_w:+.2f}%")
        print(f"  Avg loss:       {savg_l:+.2f}%")
        print(f"  Profit factor:  {spf:.2f}")

    # By ticker
    for ticker in df["ticker"].unique():
        tdf = df[df["ticker"] == ticker]
        tt = len(tdf)
        tw = len(tdf[tdf["pnl_pct"] > 0])
        twr = tw / tt * 100 if tt > 0 else 0
        tavg = tdf["pnl_pct"].mean()
        ttotal = tdf["pnl_usd"].sum()

        print(f"\n  {ticker} ({tt} trades)")
        print(f"  {'─' * 40}")
        print(f"  Win rate:       {twr:.1f}% ({tw}W / {tt - tw}L)")
        print(f"  Avg P&L:        {tavg:+.2f}%")
        print(f"  Total P&L:      ${ttotal:+.2f}")

    # By exit reason
    print(f"\n  EXIT REASONS")
    print(f"  {'─' * 40}")
    for reason in df["exit_reason"].unique():
        rdf = df[df["exit_reason"] == reason]
        rc = len(rdf)
        rwr = len(rdf[rdf["pnl_pct"] > 0]) / rc * 100 if rc > 0 else 0
        ravg = rdf["pnl_pct"].mean()
        print(f"  {reason:<18} {rc:>4} trades  {rwr:>5.1f}% WR  {ravg:>+6.2f}% avg")

    # Parameter sensitivity hint
    print(f"\n  PARAMETER SENSITIVITY")
    print(f"  {'─' * 40}")
    print(f"  Current: breakout={BREAKOUT_BARS} bars, RSI={RSI_OVERSOLD}/{RSI_OVERBOUGHT}, trail={TRAIL_ATR_MULT}x ATR")
    print(f"  If results are poor, try:")
    print(f"    - Breakout bars: 10, 15, 30, 40")
    print(f"    - RSI thresholds: 25/75, 20/80")
    print(f"    - Trail ATR mult: 1.0, 2.0, 2.5")
    print(f"    - Min rvol: 1.0, 1.5, 2.0")

    print("\n" + "=" * 70)

    # Save to CSV
    output_file = "data/crypto_backtest_results.csv"
    df.to_csv(output_file, index=False)
    print(f"  Results saved to {output_file}")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  CRYPTO STRATEGY BACKTEST")
    print(f"  Tickers: {', '.join(TICKERS)}")
    print(f"  Interval: {INTERVAL}")
    print(f"  Strategies: Momentum breakout + RSI mean reversion")
    print(f"  Trail stop: {TRAIL_ATR_MULT}x ATR")
    print("=" * 70)

    # yfinance limits 15m data to ~60 days
    # Pull maximum available
    all_trades = []

    for ticker in TICKERS:
        print(f"\n  Fetching {ticker} data...")
        try:
            data = yf.download(
                ticker, period="60d", interval=INTERVAL,
                progress=False, auto_adjust=True,
            )
            if data is not None and not data.empty:
                bars = len(data)
                days = bars * 15 / 60 / 24
                print(f"  Got {bars} bars ({days:.0f} days)")
                trades = backtest_ticker(ticker, data)
                all_trades.extend(trades)
                print(f"  {len(trades)} trades found")
            else:
                print(f"  No data returned for {ticker}")
        except Exception as e:
            print(f"  Error fetching {ticker}: {e}")

    print_results(all_trades)


if __name__ == "__main__":
    main()
