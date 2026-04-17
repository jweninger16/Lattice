"""
Emergency signal generator — bypasses yfinance entirely.
Scores yesterday's cached enriched features with v2 model to produce today's candidates,
then pulls LIVE prices from IBKR for actionable sizing.
"""
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

import sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.predict import generate_ml_signals
from live.regime import get_todays_vix_context

PARQUET = ROOT / "data/processed/price_features_enriched.parquet"

# Account constraints
ACCOUNT_BALANCE = 1853.53
MAX_POSITION_USD = 600.0   # Cap one position at ~32% of account
MAX_RISK_USD = 60.0        # 3.2% risk per trade
TOP_N = 3


def get_live_prices(tickers):
    """Pull current bid/last from IBKR for the given tickers."""
    try:
        import asyncio
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
        from ib_insync import IB, Stock
        ib = IB()
        ib.connect("127.0.0.1", 7496, clientId=99, timeout=8)  # different client id
        prices = {}
        contracts = [Stock(t, "SMART", "USD") for t in tickers]
        ib.qualifyContracts(*contracts)
        tickers_data = ib.reqTickers(*contracts)
        for tk, td in zip(tickers, tickers_data):
            last = td.last if td.last and td.last > 0 else td.close
            prices[tk] = last if last and last > 0 else None
        ib.disconnect()
        return prices
    except Exception as e:
        print(f"IBKR price fetch failed: {e}")
        return {}


def main():
    print("=" * 60)
    print("  EMERGENCY SIGNAL GENERATOR")
    print("=" * 60)

    df = pd.read_parquet(PARQUET)
    latest = df["date"].max()
    print(f"Cache: {len(df):,} rows through {latest.date()}")

    v1 = ROOT / "models/lgbm_model.pkl"
    v2 = ROOT / "models/lgbm_model_v2.pkl"
    v1_backup = ROOT / "models/lgbm_model_v1_backup.pkl"

    swap_in_v2 = v2.exists()
    if swap_in_v2:
        import shutil
        if not v1_backup.exists() or v1.stat().st_mtime > v1_backup.stat().st_mtime:
            shutil.copy2(v1, v1_backup)
        shutil.copy2(v2, v1)
        print("Using v2 model")

    try:
        scored = generate_ml_signals(df, top_pct=0.10, apply_regime_gate=False)
    finally:
        if swap_in_v2 and v1_backup.exists():
            import shutil
            shutil.copy2(v1_backup, v1)

    today = scored[scored["date"] == latest].copy()
    signals = today[today["signal"] == 1].sort_values("signal_score", ascending=False)

    # Regime check
    try:
        vix = get_todays_vix_context()
        pct_above = float(today["pct_above_sma50"].iloc[0]) if "pct_above_sma50" in today.columns else 0.5
        regime_ok = pct_above >= vix["threshold"]
        print(f"VIX={vix.get('vix_current', '?'):.1f} | Breadth {pct_above:.0%} vs {vix['threshold']:.0%} => "
              f"{'TRADE' if regime_ok else 'SIT OUT'}")
    except Exception:
        regime_ok = True

    if not regime_ok:
        print("\nRegime unfavorable. Sit out.")
        return

    # Pull live prices for top 10 candidates
    top_tickers = signals["ticker"].head(10).tolist()
    print(f"\nPulling live IBKR prices for top {len(top_tickers)}: {', '.join(top_tickers)}")
    live = get_live_prices(top_tickers)

    print()
    print("=" * 60)
    print(f"  TOP {TOP_N} ACTIONABLE CANDIDATES (live prices)")
    print(f"  Account ${ACCOUNT_BALANCE:.2f} | Cap ${MAX_POSITION_USD:.0f}/pos | Risk ${MAX_RISK_USD:.0f}/trade")
    print("=" * 60)

    shown = 0
    for _, row in signals.iterrows():
        if shown >= TOP_N:
            break
        t = row["ticker"]
        live_px = live.get(t)
        cache_px = row["close"]
        px = live_px if (live_px and live_px > 0) else cache_px
        atr = row.get("atr_14", 0) or 0
        score = row["signal_score"]
        if atr <= 0 or px <= 0:
            continue

        gap_pct = ((px - cache_px) / cache_px * 100) if live_px else 0.0
        stop = px - 2.0 * atr
        target = px + 4.0 * atr
        risk_per_sh = px - stop

        # Dual cap: position notional AND risk
        shares_by_notional = int(MAX_POSITION_USD / px)
        shares_by_risk = int(MAX_RISK_USD / risk_per_sh) if risk_per_sh > 0 else 0
        shares = min(shares_by_notional, shares_by_risk)
        if shares < 1:
            shares = 1  # at least 1 share if we're taking it at all
        cost = shares * px
        risk = shares * risk_per_sh
        reward = shares * (target - px)

        print(f"\n  #{shown+1}  {t}   ML score = {score:.3f}")
        live_str = f"LIVE ${live_px:.2f}  (cache ${cache_px:.2f}, gap {gap_pct:+.1f}%)" if live_px else f"CACHE ${cache_px:.2f} (no live quote)"
        print(f"       {live_str}")
        print(f"       ATR14 = ${atr:.2f}")
        print(f"       BUY {shares} sh @ MKT  =>  cost ${cost:.0f}")
        print(f"       STOP  ${stop:.2f}   ({-2.0*atr:.2f})   risk ${risk:.0f}")
        print(f"       TARGET ${target:.2f}  (+{4.0*atr:.2f})  reward ${reward:.0f}  (2R)")
        shown += 1

    print()
    print("=" * 60)
    print("  MANUAL ENTRY INSTRUCTIONS")
    print("=" * 60)
    print("  In TWS: place 3 orders as ONE bracket:")
    print("    1. Parent: MKT BUY <shares>  (transmit=False)")
    print("    2. Child:  STP  SELL <shares> @ stop_price   (GTC, transmit=False)")
    print("    3. Child:  LMT  SELL <shares> @ target_price (GTC, transmit=True)")
    print("  Or simpler: buy, then immediately set OCO stop+target.")
    print()
    print("  Rule: only enter between 9:31 and 10:00 ET. After 10am, skip.")


if __name__ == "__main__":
    main()
