"""
live/daily.py
-------------
The daily runner. Run this every weekday morning before market open.

Improvements:
  - Enriches live data with sector + earnings features (fixes missing features)
  - Checks portfolio drawdown before allowing new entries
  - Uses volatility-scaled position sizing
  - Better error recovery (continues even if enrichment fails)
  - Tracks cumulative P&L from position database
  - Shows risk status in briefing
"""

import sys
import yaml
import pandas as pd
import yfinance as yf
from datetime import datetime, date, timedelta
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")


def run_daily():
    from data.universe import build_universe
    from data.pipeline import add_technical_features, add_cross_sectional_features
    from models.predict import generate_ml_signals
    from live.positions import (load_positions, get_open_positions,
                                 update_positions_with_prices, print_positions,
                                 portfolio_value, get_performance_summary)
    from live.alerts import send_morning_brief, format_morning_brief
    from utils.risk import compute_volatility_scaled_size, check_portfolio_drawdown

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    today = date.today()
    logger.info(f"=== Daily Runner: {today} ===")

    # ── Check if monthly retrain is due ─────────────────────────────────
    try:
        from live.retrain import should_retrain_today
        if should_retrain_today():
            logger.info("Monthly retrain due — running in background...")
            import threading
            from live.retrain import run_retrain
            t = threading.Thread(target=run_retrain, daemon=True)
            t.start()
    except Exception as e:
        logger.warning(f"Retrain check failed: {e}")

    # ── 1. Download fresh data (last 120 days for feature calculation) ──
    logger.info("Downloading fresh market data...")
    universe = build_universe(config)

    raw = yf.download(
        universe,
        period="120d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if raw.empty:
        logger.error("No data downloaded. Markets may be closed.")
        return

    # Reshape to long format
    raw.columns.names = ["Field", "Ticker"]
    stacked = raw.stack(level="Ticker", future_stack=True).reset_index()
    stacked.columns = [c.lower() for c in stacked.columns]
    stacked["date"] = pd.to_datetime(stacked["date"])
    stacked = stacked.dropna(subset=["close", "volume"])
    stacked = stacked[stacked["close"] > 0]

    # ── 2. Build features ───────────────────────────────────────────────
    logger.info("Building technical features...")
    df = add_technical_features(stacked)
    df = add_cross_sectional_features(df)

    # ── 2b. Enrich with sector + earnings features ──────────────────────
    # This is critical for the v2 model — without these features, the model
    # is missing 12 of its 40 inputs and signal quality degrades significantly.
    start_str = df["date"].min().strftime("%Y-%m-%d")
    end_str   = df["date"].max().strftime("%Y-%m-%d")

    try:
        from data.sectors import add_sector_features
        logger.info("Adding sector momentum features...")
        df = add_sector_features(df, start_str, end_str)
    except Exception as e:
        logger.warning(f"Sector enrichment failed (continuing without): {e}")

    try:
        from data.earnings import fetch_earnings_dates, add_earnings_features
        logger.info("Adding earnings calendar features...")
        tickers = df["ticker"].unique().tolist()
        # Use cache for speed — earnings dates don't change intraday
        earnings_map = fetch_earnings_dates(tickers, use_cache=True)
        df = add_earnings_features(df, earnings_map)
    except Exception as e:
        logger.warning(f"Earnings enrichment failed (continuing without): {e}")

    # Log feature coverage
    enriched_cols = ["sector_momentum_21d", "spy_above_sma50",
                     "earnings_soon", "in_leading_sector"]
    present = [c for c in enriched_cols if c in df.columns]
    logger.info(f"Enriched features available: {len(present)}/{len(enriched_cols)} — {present}")

    # ── 3. Score with ML model ──────────────────────────────────────────
    logger.info("Scoring with ML model...")
    df = generate_ml_signals(df, top_pct=0.10)

    # Today's data only
    latest_date = df["date"].max()
    today_df    = df[df["date"] == latest_date].copy()
    regime_ok   = bool(today_df["regime_ok"].iloc[0]) if "regime_ok" in today_df.columns else True
    pct_above   = float(today_df["pct_above_sma50"].iloc[0] * 100) if "pct_above_sma50" in today_df.columns else 50.0

    # Get dynamic VIX-adjusted threshold
    try:
        from live.regime import get_todays_vix_context
        vix_ctx = get_todays_vix_context()
        dynamic_threshold = vix_ctx["threshold"]

        pct_above_raw = float(today_df["pct_above_sma50"].iloc[0]) if "pct_above_sma50" in today_df.columns else 0.5
        regime_ok = pct_above_raw >= dynamic_threshold

        vix_str = f"VIX={vix_ctx['vix_current']:.1f} ({vix_ctx['vix_regime']})" if vix_ctx["vix_current"] else ""
    except Exception as e:
        logger.warning(f"VIX context failed: {e}")
        vix_ctx = {"vix_current": None, "vix_regime": "UNKNOWN", "threshold": 0.50}
        vix_str = ""

    logger.info(f"Date: {latest_date.date()} | Regime: {'OK' if regime_ok else 'UNFAVORABLE'} | "
                f"{pct_above:.0f}% above SMA50 | {vix_str}")

    # ── 4. Check open positions ─────────────────────────────────────────
    open_positions = get_open_positions()
    current_prices = {}
    for p in open_positions:
        row = today_df[today_df["ticker"] == p["ticker"]]
        if not row.empty:
            current_prices[p["ticker"]] = {
                "close": row["close"].values[0],
                "low":   row["low"].values[0],
                "high":  row["high"].values[0],
                "atr_14": row["atr_14"].values[0] if "atr_14" in row.columns else 0,
            }

    positions_with_actions = update_positions_with_prices(current_prices)

    # ── 4b. Portfolio state ────────────────────────────────────────────
    from live.positions import (initialize_portfolio, get_cash_balance,
                                 record_snapshot, get_portfolio_summary)

    # Initialize portfolio on first run
    initialize_portfolio(config["backtest"]["initial_capital"])

    # Calculate current value from real tracked state
    cash = get_cash_balance()
    if cash <= 0:
        # Fallback if not yet tracking
        cash = config["backtest"]["initial_capital"] - sum(p["size_usd"] for p in open_positions)

    open_pos_value = sum(
        p.get("current_price", p["entry_price"]) / p["entry_price"] * p["size_usd"]
        for p in positions_with_actions
    )
    est_portfolio = cash + open_pos_value

    # Record daily snapshot
    record_snapshot(cash, open_pos_value, len(open_positions))

    dd_status = check_portfolio_drawdown(
        [config["backtest"]["initial_capital"], est_portfolio],
        config,
    )

    # ── 5. Print today's briefing ───────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  SWING TRADER — {today.strftime('%A %B %d, %Y').replace(' 0', ' ')}")
    print("=" * 55)

    regime_str = "FAVORABLE" if regime_ok else "UNFAVORABLE - NO NEW TRADES"
    print(f"\n  Market Regime: {regime_str} ({pct_above:.0f}% above SMA50)")
    if vix_str:
        print(f"  {vix_str} | Threshold: {vix_ctx.get('threshold', 0.50):.0%}")

    if dd_status["halted"]:
        print(f"\n  ⚠ PORTFOLIO STOP ACTIVE: DD {dd_status['current_dd_pct']:.1f}% "
              f"exceeds {dd_status['max_dd_limit']*100:.0f}% limit")

    print(f"  Portfolio: ~${est_portfolio:,.0f} | "
          f"DD: {dd_status['current_dd_pct']:.1f}%")

    print(f"\n  OPEN POSITIONS ({len(open_positions)}):")
    if open_positions:
        print_positions(positions_with_actions)
    else:
        print("  None")

    # Exits
    exits = [p for p in positions_with_actions if p.get("action", "HOLD") != "HOLD"]
    slots_freed = len(exits)

    if exits:
        print(f"\n  ACTION — SELL TODAY:")
        for p in exits:
            ret = (p.get("current_price", p["entry_price"]) / p["entry_price"] - 1) * 100
            reason = p.get("action", "").replace("SELL_", "").lower()
            print(f"    {p['ticker']:<6} @ ~${p.get('current_price', 0):.2f} ({reason}) {ret:+.1f}%")

    # New signals
    slots_available = config["universe"]["max_positions"] - len(open_positions) + slots_freed
    signals_today = today_df[today_df["signal"] == 1].sort_values("signal_score", ascending=False)
    allow_entries = regime_ok and not dd_status["halted"]

    if allow_entries and slots_available > 0 and len(signals_today) > 0:
        new_buys = signals_today.head(slots_available)

        print(f"\n  ACTION — BUY TODAY ({slots_available} slot(s)):")
        for _, row in new_buys.iterrows():
            atr_pct = row.get("atr_pct", 0.02)
            buy_size = compute_volatility_scaled_size(atr_pct, est_portfolio, config)

            atr = row.get("atr_14", row["close"] * 0.02)
            stop  = row["close"] - atr * config["backtest"]["stop_loss_atr"]
            tgt   = row["close"] + atr * config["backtest"]["profit_target_atr"]
            exit_dt = (date.today() + timedelta(days=config["backtest"]["hold_days"] + 2)).strftime("%b %d").replace(" 0", " ")
            print(f"    {row['ticker']:<6} @ ~${row['close']:.2f} | "
                  f"size ~${buy_size:,.0f} | stop=${stop:.2f} | target=${tgt:.2f} | exit ~{exit_dt}")
    elif dd_status["halted"]:
        print(f"\n  No new trades — portfolio drawdown limit hit")
    elif not regime_ok:
        print(f"\n  No new trades — market regime unfavorable")
    elif slots_available == 0:
        print(f"\n  No slots available — all {config['universe']['max_positions']} positions filled")
    else:
        print(f"\n  No signals today")

    print("\n" + "=" * 55 + "\n")

    # ── 5b. SH Hedge logic ─────────────────────────────────────────────
    sh_positions = [p for p in open_positions if p.get("ticker") == "SH"]
    sh_held = len(sh_positions) > 0

    sh_action = None
    if not regime_ok and not sh_held:
        sh_action = "BUY"
    elif regime_ok and sh_held:
        sh_action = "SELL"

    if sh_action == "BUY":
        print(f"\n  ACTION — HEDGE: BUY SH (regime unfavorable)")
        print(f"    Allocate ~50% of available cash to SH at market open")
    elif sh_action == "SELL":
        print(f"\n  ACTION — HEDGE: SELL SH (regime now favorable)")
        print(f"    Close SH position at market open")
    elif not regime_ok and sh_held:
        print("\n  HEDGE: Holding SH (regime still unfavorable)")

    # ── 6. Send SMS ─────────────────────────────────────────────────────
    try:
        message = format_morning_brief(
            regime_ok=regime_ok,
            pct_above_sma50=pct_above,
            positions=positions_with_actions,
            signals=[{"ticker": r["ticker"], "price": r["close"], "score": r["signal_score"]}
                     for _, r in signals_today.head(6).iterrows()],
            portfolio_value=est_portfolio,
            initial_capital=config["backtest"]["initial_capital"],
            slots_used=len(open_positions) - slots_freed,
            max_slots=config["universe"]["max_positions"],
            sh_action=sh_action,
        )
        send_morning_brief(today_df, pd.DataFrame(positions_with_actions),
                           portfolio_value=est_portfolio,
                           initial_capital=config["backtest"]["initial_capital"],
                           sh_action=sh_action)
        logger.info("SMS sent successfully")
    except Exception as e:
        logger.warning(f"SMS failed (is .env configured?): {e}")
        logger.info("Tip: configure .env file to enable SMS alerts")

    logger.info("Daily run complete.")


if __name__ == "__main__":
    run_daily()
