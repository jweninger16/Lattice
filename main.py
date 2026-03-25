"""
main.py — Swing Trader entry point (v2)

Commands:
  python main.py pipeline         Download data and build features
  python main.py signals          Show today's signals
  python main.py backtest         Run full backtest
  python main.py walkforward      Walk-forward validation
  python main.py train            Train LightGBM model
  python main.py enrich           Add sector + earnings features
  python main.py train_v2         Train with enriched features
  python main.py backtest_ml      Backtest with ML signals
  python main.py backtest_ml_v2   Backtest enriched v2 model
  python main.py backtest_inverse Compare ML vs ML+SH hedge
  python main.py backtest_dynamic Fixed vs dynamic VIX regime
  python main.py daily            Run daily morning briefing
  python main.py monitor          Intraday stop/target monitor
  python main.py monitor --once   Single check and exit
  python main.py dashboard        Launch web dashboard
  python main.py scheduler        Start daily scheduler
  python main.py retrain          Force model retrain
  python main.py retrain_log      Show retrain history
  python main.py vix              Show current VIX context
  python main.py risk             Show portfolio risk status
  python main.py portfolio        Show full portfolio status
  python main.py history          Show closed trade history
  python main.py all              Run pipeline + signals + backtest
"""

import sys
import yaml
from loguru import logger
from pathlib import Path

Path("logs").mkdir(exist_ok=True)
logger.add("logs/swing_trader.log", rotation="1 week", retention="1 month", level="INFO")


def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


def cmd_pipeline():
    from data.universe import build_universe
    from data.pipeline import run_pipeline
    config = load_config()
    universe = build_universe(config)
    print(f"\nUniverse: {len(universe)} stocks")
    df = run_pipeline(config, universe)
    print(f"Data shape: {df.shape}")


def cmd_signals():
    from data.pipeline import load_processed
    from signals.signals import generate_signals, get_todays_signals, signal_summary
    df = load_processed()
    df = generate_signals(df)
    summary = signal_summary(df)
    print("\n=== Signal Summary ===")
    for k, v in summary.items():
        if v is not None:
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("\n=== Today's Top Signals ===")
    print(get_todays_signals(df).to_string(index=False))


def cmd_backtest():
    from data.pipeline import load_processed
    from signals.signals import generate_signals
    from backtest.backtest import SwingBacktester, print_stats, plot_results
    config = load_config()
    df = load_processed()
    df = generate_signals(df)
    bt = SwingBacktester(config)
    result = bt.run(df)
    print_stats(result["stats"])
    plot_results(result)
    print("Chart saved to backtest/backtest_results.png")


def cmd_walkforward():
    from data.pipeline import load_processed
    from signals.signals import generate_signals
    from backtest.backtest import walk_forward_backtest
    config = load_config()
    df = load_processed()
    df = generate_signals(df)
    folds = walk_forward_backtest(df, config)
    print(f"\n=== Walk-Forward Results ({len(folds)} folds) ===")
    for fold in folds:
        s = fold["stats"]
        print(f"  Fold {s['fold']} [{s['test_start'].date()} -> {s['test_end'].date()}]: "
              f"Return={s.get('total_return_pct',0):.1f}% | "
              f"Sharpe={s.get('sharpe',0):.2f} | "
              f"MaxDD={s.get('max_drawdown_pct',0):.1f}%")


def cmd_enrich():
    from data.pipeline import load_processed, save_processed
    from data.sectors import add_sector_features
    from data.earnings import fetch_earnings_dates, add_earnings_features

    print("\nLoading processed data...")
    df = load_processed()
    start = df["date"].min().strftime("%Y-%m-%d")
    end   = df["date"].max().strftime("%Y-%m-%d")

    print("Adding sector momentum features (SPY, QQQ, 11 sector ETFs)...")
    df = add_sector_features(df, start, end)

    print("Fetching earnings calendar (this may take 5-10 minutes first time)...")
    tickers = df["ticker"].unique().tolist()
    earnings_map = fetch_earnings_dates(tickers, use_cache=True)
    df = add_earnings_features(df, earnings_map)

    print(f"Saving enriched dataset ({df.shape[0]:,} rows, {df.shape[1]} columns)...")
    save_processed(df, "price_features_enriched.parquet")
    print("Saved to data/processed/price_features_enriched.parquet")
    print("\nNext step: run 'python main.py train_v2' to train with new features")


def cmd_train():
    from data.pipeline import load_processed
    from signals.signals import compute_market_regime
    from models.train import build_features, build_target, walk_forward_train
    from models.train import evaluate_ml_signal, print_feature_importance
    config = load_config()

    print("\nLoading data and building ML features...")
    df = load_processed()
    regime = compute_market_regime(df)
    df = df.merge(regime, on="date", how="left")
    df, feature_cols = build_features(df)
    df = build_target(df)

    print(f"Training LightGBM on {len(df):,} rows, {len(feature_cols)} features...")
    print("This will take 1-3 minutes...\n")
    results = walk_forward_train(df, config, feature_cols)

    evaluate_ml_signal(results["oos_predictions"])
    print_feature_importance(results["model"], results["feature_cols"])

    print("\nFold-by-fold results:")
    print(results["fold_stats"].to_string(index=False))
    print("\nModel saved to models/lgbm_model.pkl")


def cmd_train_v2():
    from data.pipeline import load_processed
    from signals.signals import compute_market_regime
    from models.train import build_features, build_target, walk_forward_train
    from models.train import evaluate_ml_signal, print_feature_importance
    config = load_config()

    print("\nLoading enriched dataset...")
    try:
        df = load_processed("price_features_enriched.parquet")
    except FileNotFoundError:
        print("Enriched dataset not found. Run 'python main.py enrich' first.")
        return

    regime = compute_market_regime(df)
    df = df.merge(regime, on="date", how="left")
    df, feature_cols = build_features(df)
    df = build_target(df)

    print(f"Features: {len(feature_cols)}")
    print(f"Training on {len(df):,} rows...")
    print("This will take 3-5 minutes...\n")

    results = walk_forward_train(df, config, feature_cols)
    evaluate_ml_signal(results["oos_predictions"])
    print_feature_importance(results["model"], results["feature_cols"])

    print("\nFold-by-fold results:")
    print(results["fold_stats"].to_string(index=False))

    import pickle
    with open("models/lgbm_model_v2.pkl", "wb") as f:
        pickle.dump({"model": results["model"], "features": feature_cols}, f)
    print("\nModel v2 saved to models/lgbm_model_v2.pkl")


def cmd_backtest_ml():
    from data.pipeline import load_processed
    from models.predict import generate_ml_signals
    from backtest.backtest import SwingBacktester, print_stats, plot_results
    config = load_config()

    print("\nLoading data and scoring with ML model...")
    df = load_processed()
    df = generate_ml_signals(df, top_pct=0.10)

    print(f"Running ML backtest...")
    bt = SwingBacktester(config)
    result = bt.run(df)
    print_stats(result["stats"])
    plot_results(result, save_path="backtest/backtest_ml_results.png")
    print("Chart saved to backtest/backtest_ml_results.png")


def cmd_backtest_ml_v2():
    from data.pipeline import load_processed
    from models.predict import generate_ml_signals
    from backtest.backtest import SwingBacktester, print_stats, plot_results
    import pickle, shutil
    config = load_config()

    print("\nLoading enriched data and scoring with v2 model...")
    df = load_processed("price_features_enriched.parquet")

    v1_path = Path("models/lgbm_model.pkl")
    v2_path = Path("models/lgbm_model_v2.pkl")
    backup  = Path("models/lgbm_model_v1_backup.pkl")

    if v2_path.exists():
        shutil.copy(v1_path, backup)
        shutil.copy(v2_path, v1_path)

    try:
        df = generate_ml_signals(df, top_pct=0.10)
        bt = SwingBacktester(config)
        result = bt.run(df)
        print("\n=== V2 MODEL (with sector + earnings features) ===")
        print_stats(result["stats"])
        plot_results(result, save_path="backtest/backtest_ml_v2_results.png")
        print("Chart saved to backtest/backtest_ml_v2_results.png")
    finally:
        if backup.exists():
            shutil.copy(backup, v1_path)

    print("\nV1 model restored. V2 results saved above.")


def cmd_backtest_inverse():
    from data.pipeline import load_processed
    from models.predict import generate_ml_signals
    from backtest.backtest_inverse import run_inverse_backtest
    config = load_config()
    print("\nLoading data and scoring with ML model...")
    if Path("data/processed/price_features_enriched.parquet").exists():
        df = load_processed("price_features_enriched.parquet")
    else:
        df = load_processed()
    df = generate_ml_signals(df, top_pct=0.10)
    run_inverse_backtest(df, config, sh_allocation=0.50)


def cmd_backtest_dynamic():
    from data.pipeline import load_processed
    from models.predict import generate_ml_signals
    from backtest.backtest_dynamic_regime import run_dynamic_regime_backtest
    config = load_config()
    print("\nLoading data...")
    if Path("data/processed/price_features_enriched.parquet").exists():
        df = load_processed("price_features_enriched.parquet")
    else:
        df = load_processed()
    df = generate_ml_signals(df, top_pct=0.10)
    run_dynamic_regime_backtest(df, config)


def cmd_daily():
    from live.daily import run_daily
    run_daily()


def cmd_monitor():
    """Run intraday position monitor."""
    from live.monitor import run_monitor, run_once
    if "--once" in sys.argv:
        run_once()
    else:
        run_monitor()


def cmd_dashboard():
    """Launch web dashboard."""
    from live.dashboard import run_dashboard
    run_dashboard()


def cmd_ticker():
    """Launch desktop ticker overlay."""
    from live.ticker import run_ticker
    run_ticker()


def cmd_broker():
    """IBKR broker integration. Usage: python main.py broker [status|sync|execute|auto]"""
    from live.broker import run_broker
    sub = sys.argv[2] if len(sys.argv) > 2 else "status"
    run_broker(sub)


def cmd_orb():
    """ORB day trader. Usage: python main.py orb [--live] [--size 500]"""
    from live.orb_trader import run_orb
    run_orb()


def cmd_scheduler():
    from live.scheduler import job
    import schedule, time
    print("Scheduler running. Daily brief at 9:00 AM weekdays. Ctrl+C to stop.")
    schedule.every().day.at("09:00").do(job)
    job()
    while True:
        schedule.run_pending()
        time.sleep(60)


def cmd_retrain():
    from live.retrain import run_retrain
    force = "--force" in sys.argv
    print("\nStarting retrain pipeline (this takes ~20 minutes)...")
    results = run_retrain(force=force)
    print(f"\nRetrain complete:")
    for k, v in results.items():
        print(f"  {k}: {v}")


def cmd_retrain_log():
    import pandas as pd
    log_path = Path("logs/retrain_history.csv")
    if not log_path.exists():
        print("No retrain history yet. Run 'python main.py retrain' first.")
        return
    log = pd.read_csv(log_path)
    print("\n=== RETRAIN HISTORY ===")
    print(log.to_string(index=False))


def cmd_vix():
    from live.regime import get_todays_vix_context
    ctx = get_todays_vix_context()
    print(f"\nVIX Context:")
    for k, v in ctx.items():
        if v is not None:
            print(f"  {k}: {v}")


def cmd_risk():
    """Show portfolio risk status."""
    from live.positions import get_open_positions, get_performance_summary
    from utils.risk import check_portfolio_drawdown
    config = load_config()

    positions = get_open_positions()
    summary = get_performance_summary()

    print("\n=== RISK STATUS ===")
    print(f"  Open positions: {len(positions)}/{config['universe']['max_positions']}")

    if summary:
        print(f"  Total trades: {summary['total_trades']}")
        print(f"  Win rate: {summary['win_rate']:.1f}%")
        print(f"  Total P&L: ${summary['total_pnl']:.0f}")
        print(f"  Profit factor: {summary['profit_factor']:.2f}")
    else:
        print("  No closed trades yet")


def cmd_history():
    """Show closed trade history."""
    from live.positions import get_trade_history
    trades = get_trade_history()
    if trades.empty:
        print("No closed trades yet.")
        return
    cols = ["ticker", "entry_date", "exit_date", "entry_price", "exit_price",
            "exit_reason", "pnl_pct", "pnl_usd"]
    available = [c for c in cols if c in trades.columns]
    print("\n=== TRADE HISTORY ===")
    print(trades[available].to_string(index=False))


def cmd_portfolio():
    """Show full portfolio status."""
    from live.positions import (get_portfolio_summary, get_open_positions,
                                 print_positions, get_equity_curve,
                                 initialize_portfolio)
    config = load_config()
    initialize_portfolio(config["backtest"]["initial_capital"])

    summary = get_portfolio_summary()
    print("\n" + "=" * 50)
    print("  PORTFOLIO STATUS")
    print("=" * 50)
    if summary:
        print(f"  Cash:            ${summary['cash']:,.0f}")
        print(f"  Invested:        ${summary['invested']:,.0f}")
        print(f"  Total Equity:    ${summary['total_equity']:,.0f}")
        print(f"  Initial Capital: ${summary['initial_equity']:,.0f}")
        print(f"  Total Return:    {summary['total_return_pct']:+.1f}%")
        print(f"  Cumulative P&L:  ${summary['cumulative_pnl']:+,.0f}")
        print(f"  Today's P&L:     ${summary['daily_pnl']:+,.0f}")
        print(f"  Snapshots:       {summary['n_snapshots']}")
    else:
        print("  No portfolio data yet. Run: python main.py daily")

    positions = get_open_positions()
    print(f"\n  Open Positions ({len(positions)}/{config['universe']['max_positions']}):")
    if positions:
        print_positions(positions)
    else:
        print("  None")

    eq = get_equity_curve()
    if len(eq) > 1:
        print(f"\n  Equity Curve ({len(eq)} snapshots):")
        print(f"  {'Date':<12} {'Equity':>10} {'Daily P&L':>10}")
        for _, row in eq.tail(10).iterrows():
            print(f"  {str(row['date'].date()):<12} ${row['total_equity']:>9,.0f} "
                  f"${row['daily_pnl']:>+9,.0f}")
    print("=" * 50)


def cmd_all():
    cmd_pipeline()
    cmd_signals()
    cmd_backtest()


def cmd_buy():
    """Manually log a buy trade. Usage: python main.py buy TICKER PRICE SHARES"""
    from live.positions import add_position, get_cash_balance, record_snapshot
    if len(sys.argv) < 5:
        print("Usage: python main.py buy TICKER PRICE SHARES")
        print("Example: python main.py buy SH 16.50 60")
        return

    ticker = sys.argv[2].upper()
    price = float(sys.argv[3])
    shares = float(sys.argv[4])
    size_usd = price * shares

    # For hedge positions like SH, use wide stop/target since they're manual
    is_hedge = ticker in ("SH", "SDS", "SPXU", "PSQ", "DOG", "SQQQ")
    if is_hedge:
        stop_price = price * 0.50    # 50% wide stop (manual management)
        target_price = price * 2.0   # 100% target (manual management)
    else:
        stop_price = price * 0.93    # 7% default stop
        target_price = price * 1.15  # 15% default target

    add_position(
        ticker=ticker,
        entry_price=price,
        size_usd=size_usd,
        stop_price=stop_price,
        target_price=target_price,
        exit_date="manual",
    )

    cash = get_cash_balance()
    print(f"\n  BUY logged: {ticker}")
    print(f"  Price:  ${price:.2f}")
    print(f"  Shares: {shares:.0f}")
    print(f"  Value:  ${size_usd:,.2f}")
    if is_hedge:
        print(f"  Type:   HEDGE (manual stop/target)")
    print(f"  Cash remaining: ${cash:,.2f}")


def cmd_sell():
    """Manually log a sell trade. Usage: python main.py sell TICKER PRICE"""
    from live.positions import close_position, get_open_positions
    if len(sys.argv) < 4:
        print("Usage: python main.py sell TICKER PRICE")
        print("Example: python main.py sell SH 17.00")
        return

    ticker = sys.argv[2].upper()
    price = float(sys.argv[3])

    # Check position exists
    positions = get_open_positions()
    found = any(p["ticker"] == ticker for p in positions)
    if not found:
        print(f"  No open position found for {ticker}")
        return

    close_position(ticker=ticker, exit_price=price, reason="manual_sell")
    print(f"\n  SELL logged: {ticker} @ ${price:.2f}")


COMMANDS = {
    "pipeline":         cmd_pipeline,
    "signals":          cmd_signals,
    "backtest":         cmd_backtest,
    "walkforward":      cmd_walkforward,
    "train":            cmd_train,
    "enrich":           cmd_enrich,
    "train_v2":         cmd_train_v2,
    "backtest_ml":      cmd_backtest_ml,
    "backtest_ml_v2":   cmd_backtest_ml_v2,
    "backtest_inverse": cmd_backtest_inverse,
    "backtest_dynamic": cmd_backtest_dynamic,
    "daily":            cmd_daily,
    "monitor":          cmd_monitor,
    "dashboard":        cmd_dashboard,
    "ticker":           cmd_ticker,
    "broker":           cmd_broker,
    "orb":              cmd_orb,
    "scheduler":        cmd_scheduler,
    "retrain":          cmd_retrain,
    "retrain_log":      cmd_retrain_log,
    "vix":              cmd_vix,
    "risk":             cmd_risk,
    "portfolio":        cmd_portfolio,
    "history":          cmd_history,
    "buy":              cmd_buy,
    "sell":             cmd_sell,
    "all":              cmd_all,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    logger.info(f"Running: {cmd}")
    COMMANDS[cmd]()
