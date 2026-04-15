"""
live/daily.py
-------------
The daily runner. Run this every weekday morning before market open.

Improvements:
  - Enriches live data with sector + earnings features (fixes missing features)
  - Uses real equity curve from DB for drawdown (not 2-element hack)
  - Risk filters: sector concentration + correlation checks on new signals
  - VIX-aware regime is the SOLE regime gate (no dual regime conflict)
  - Reports ALL open positions with action items (HOLD included)
  - SH hedge tied to regime transitions with clear messaging
  - Volatility-scaled position sizing
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
                                 portfolio_value, get_performance_summary,
                                 get_equity_curve)
    from live.alerts import send_morning_brief, format_morning_brief
    from utils.risk import (compute_volatility_scaled_size, check_portfolio_drawdown,
                            compute_correlation_matrix, filter_candidates_by_risk)

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
    # apply_regime_gate=False: we apply VIX-aware regime below, not the
    # static 50% gate baked into signals.py
    logger.info("Scoring with ML model...")
    df = generate_ml_signals(df, top_pct=0.10, apply_regime_gate=False)

    # Today's data only
    latest_date = df["date"].max()
    today_df    = df[df["date"] == latest_date].copy()
    pct_above   = float(today_df["pct_above_sma50"].iloc[0] * 100) if "pct_above_sma50" in today_df.columns else 50.0

    # ── 3b. Single VIX-aware regime gate (sole authority) ───────────────
    try:
        from live.regime import get_todays_vix_context
        vix_ctx = get_todays_vix_context()
        dynamic_threshold = vix_ctx["threshold"]

        pct_above_raw = float(today_df["pct_above_sma50"].iloc[0]) if "pct_above_sma50" in today_df.columns else 0.5
        regime_ok = pct_above_raw >= dynamic_threshold

        vix_str = f"VIX={vix_ctx['vix_current']:.1f} ({vix_ctx['vix_regime']})" if vix_ctx["vix_current"] else ""
    except Exception as e:
        logger.warning(f"VIX context failed, falling back to static 50%: {e}")
        vix_ctx = {"vix_current": None, "vix_regime": "UNKNOWN", "threshold": 0.50}
        vix_str = ""
        regime_ok = bool(today_df["regime_ok"].iloc[0]) if "regime_ok" in today_df.columns else True

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

    # ── 4c. Real equity-curve drawdown (not 2-element hack) ────────────
    equity_df = get_equity_curve()
    if not equity_df.empty and len(equity_df) >= 2:
        equity_list = equity_df["total_equity"].tolist()
    else:
        equity_list = [config["backtest"]["initial_capital"], est_portfolio]

    dd_status = check_portfolio_drawdown(equity_list, config)

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
          f"DD: {dd_status['current_dd_pct']:.1f}% | "
          f"Peak: ${dd_status.get('peak', est_portfolio):,.0f}")

    # ── 5a. ALL open positions with actions ─────────────────────────────
    # Always show every position — HOLD, SELL, everything
    non_sh_positions = [p for p in positions_with_actions if p.get("ticker") != "SH"]
    sh_positions_list = [p for p in positions_with_actions if p.get("ticker") == "SH"]

    print(f"\n  OPEN POSITIONS ({len(non_sh_positions)} stocks"
          f"{' + SH hedge' if sh_positions_list else ''}):")

    if non_sh_positions:
        exits = [p for p in non_sh_positions if p.get("action", "HOLD") != "HOLD"]
        holds = [p for p in non_sh_positions if p.get("action", "HOLD") == "HOLD"]

        if exits:
            print(f"\n    SELL TODAY:")
            for p in exits:
                ret = (p.get("current_price", p["entry_price"]) / p["entry_price"] - 1) * 100
                reason = p.get("action", "").replace("SELL_", "").lower()
                days_held = (date.today() - datetime.strptime(p["entry_date"], "%Y-%m-%d").date()).days if p.get("entry_date") else "?"
                print(f"      {p['ticker']:<6} @ ~${p.get('current_price', 0):.2f} "
                      f"({reason}) {ret:+.1f}% | held {days_held}d")

        if holds:
            print(f"\n    HOLD ({len(holds)}):")
            for p in holds:
                ret = (p.get("current_price", p["entry_price"]) / p["entry_price"] - 1) * 100
                days_held = (date.today() - datetime.strptime(p["entry_date"], "%Y-%m-%d").date()).days if p.get("entry_date") else "?"
                planned = p.get("planned_exit", p.get("exit_date", "TBD"))
                stop_dist = ((p.get("current_price", p["entry_price"]) - p["stop_price"]) / p.get("current_price", p["entry_price"]) * 100) if p.get("stop_price") else 0
                print(f"      {p['ticker']:<6} @ ~${p.get('current_price', 0):.2f} "
                      f"{ret:+.1f}% | held {days_held}d | "
                      f"stop=${p.get('stop_price', 0):.2f} ({stop_dist:.1f}% away) | "
                      f"exit ~{planned}")
    else:
        print("    None")

    # ── 5b. SH Hedge logic ─────────────────────────────────────────────
    sh_held = len(sh_positions_list) > 0

    sh_action = None
    if not regime_ok and not sh_held:
        sh_action = "BUY"
    elif regime_ok and sh_held:
        sh_action = "SELL"

    if sh_positions_list:
        p = sh_positions_list[0]
        ret = (p.get("current_price", p["entry_price"]) / p["entry_price"] - 1) * 100
        print(f"\n    SH HEDGE: @ ~${p.get('current_price', 0):.2f} {ret:+.1f}%", end="")
        if sh_action == "SELL":
            print(" → SELL (regime now favorable)")
        else:
            print(" → HOLD (regime still unfavorable)")

    if sh_action == "BUY":
        print(f"\n  ACTION — HEDGE: BUY SH (regime unfavorable)")
        print(f"    Allocate ~50% of available cash to SH at market open")
    elif sh_action == "SELL" and not sh_positions_list:
        # Edge case: sh_action=SELL but no SH in DB (manual tracking)
        print(f"\n  ACTION — HEDGE: SELL SH (regime now favorable)")
        print(f"    Close SH position at market open")

    # ── 5c. New signals (with sector/correlation filters) ──────────────
    filtered_signals = pd.DataFrame()  # default empty; populated below if signals pass
    slots_freed = len([p for p in non_sh_positions if p.get("action", "HOLD") != "HOLD"])
    slots_available = config["universe"]["max_positions"] - len(non_sh_positions) + slots_freed
    signals_today = today_df[today_df["signal"] == 1].sort_values("signal_score", ascending=False)

    # Remove tickers we already hold
    held_tickers = [p["ticker"] for p in open_positions]
    signals_today = signals_today[~signals_today["ticker"].isin(held_tickers)]

    allow_entries = regime_ok and not dd_status["halted"]

    if allow_entries and slots_available > 0 and len(signals_today) > 0:
        # ── Apply risk filters: sector concentration + correlation ──
        # Build sector map from today's data
        sector_map = {}
        if "sector" in today_df.columns:
            sector_map = dict(zip(today_df["ticker"], today_df["sector"]))

        # Compute correlation matrix from recent returns
        try:
            corr_matrix = compute_correlation_matrix(df)
        except Exception as e:
            logger.warning(f"Correlation matrix failed: {e}")
            corr_matrix = pd.DataFrame()

        filtered_signals = filter_candidates_by_risk(
            candidates=signals_today,
            open_positions=non_sh_positions,
            correlation_matrix=corr_matrix,
            sector_map=sector_map,
            config=config,
        )

        if len(filtered_signals) > 0:
            new_buys = filtered_signals.head(slots_available)

            print(f"\n  ACTION — BUY TODAY ({min(slots_available, len(new_buys))} of {slots_available} slot(s)):")
            for _, row in new_buys.iterrows():
                atr_pct = row.get("atr_pct", 0.02)
                buy_size = compute_volatility_scaled_size(atr_pct, est_portfolio, config)

                atr = row.get("atr_14", row["close"] * 0.02)
                stop  = row["close"] - atr * config["backtest"]["stop_loss_atr"]
                tgt   = row["close"] + atr * config["backtest"]["profit_target_atr"]
                exit_dt = (date.today() + timedelta(days=config["backtest"]["hold_days"] + 2)).strftime("%b %d").replace(" 0", " ")
                score_str = f"score={row.get('ml_score', 0):.2f}" if "ml_score" in row.index else ""
                print(f"    {row['ticker']:<6} @ ~${row['close']:.2f} | "
                      f"size ~${buy_size:,.0f} | stop=${stop:.2f} | "
                      f"target=${tgt:.2f} | exit ~{exit_dt} | {score_str}")
        else:
            n_before = len(signals_today)
            print(f"\n  No new trades — {n_before} ML signals rejected by risk filters "
                  f"(sector/correlation)")
    elif dd_status["halted"]:
        print(f"\n  No new trades — portfolio drawdown limit hit")
    elif not regime_ok:
        print(f"\n  No new trades — market regime unfavorable")
    elif slots_available == 0:
        print(f"\n  No slots available — all {config['universe']['max_positions']} positions filled")
    else:
        print(f"\n  No ML signals today (score floor or percentile not met)")

    print("\n" + "=" * 55 + "\n")

    # ── 6. Send alert ───────────────────────────────────────────────────
    try:
        # Build signal list for the alert (post-filter)
        alert_signals = []
        if allow_entries and slots_available > 0 and len(filtered_signals) > 0:
            buy_df = filtered_signals.head(slots_available)
            for _, r in buy_df.iterrows():
                alert_signals.append({"ticker": r["ticker"], "price": r["close"],
                                      "score": r.get("signal_score", 0)})

        message = format_morning_brief(
            regime_ok=regime_ok,
            pct_above_sma50=pct_above,
            positions=positions_with_actions,
            signals=alert_signals,
            portfolio_value=est_portfolio,
            initial_capital=config["backtest"]["initial_capital"],
            slots_used=len(non_sh_positions) - slots_freed,
            max_slots=config["universe"]["max_positions"],
            sh_action=sh_action,
            vix_context=vix_ctx,
        )
        send_morning_brief(today_df, pd.DataFrame(positions_with_actions),
                           portfolio_value=est_portfolio,
                           initial_capital=config["backtest"]["initial_capital"],
                           sh_action=sh_action)
        logger.info("Alert sent successfully")
    except Exception as e:
        logger.warning(f"Alert failed (is .env configured?): {e}")
        logger.info("Tip: configure .env file to enable alerts")

    logger.info("Daily run complete.")


if __name__ == "__main__":
    run_daily()
