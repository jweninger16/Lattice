"""
live/broker.py - IBKR broker integration
"""

import sys
import yaml
import time
import asyncio
from datetime import datetime, date, timedelta
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")

# Python 3.14 fix: create event loop before importing ib_insync
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

DEFAULT_PORT = 7496
DEFAULT_HOST = "127.0.0.1"
CLIENT_ID = 10


def get_ib_connection(host=DEFAULT_HOST, port=DEFAULT_PORT, client_id=CLIENT_ID):
    """Connects to TWS/IB Gateway."""
    try:
        from ib_insync import IB
        ib = IB()
        ib.connect(host, port, clientId=client_id)
        logger.info(f"Connected to IBKR at {host}:{port}")
        return ib
    except Exception as e:
        logger.error(f"Failed to connect to IBKR: {e}")
        logger.error("Make sure TWS is running and API is enabled.")
        return None


def place_buy_order(ib, ticker, quantity, stop_price=None, target_price=None):
    """Places a buy order with optional bracket."""
    from ib_insync import Stock, MarketOrder, LimitOrder, StopOrder
    contract = Stock(ticker, "SMART", "USD")
    ib.qualifyContracts(contract)

    if stop_price and target_price:
        parent = MarketOrder("BUY", quantity)
        parent.transmit = False
        parent_trade = ib.placeOrder(contract, parent)
        ib.sleep(0.5)

        stop = StopOrder("SELL", quantity, stop_price)
        stop.parentId = parent_trade.order.orderId
        stop.transmit = False
        ib.placeOrder(contract, stop)

        target = LimitOrder("SELL", quantity, target_price)
        target.parentId = parent_trade.order.orderId
        target.transmit = True
        ib.placeOrder(contract, target)
        ib.sleep(1)

        logger.info(f"BRACKET: BUY {quantity} {ticker} | stop=${stop_price:.2f} | target=${target_price:.2f}")
        return {"ticker": ticker, "action": "BUY", "quantity": quantity,
                "order_id": parent_trade.order.orderId,
                "status": parent_trade.orderStatus.status}
    else:
        order = MarketOrder("BUY", quantity)
        trade = ib.placeOrder(contract, order)
        ib.sleep(1)
        logger.info(f"MARKET: BUY {quantity} {ticker}")
        return {"ticker": ticker, "action": "BUY", "quantity": quantity,
                "order_id": trade.order.orderId, "status": trade.orderStatus.status}


def place_sell_order(ib, ticker, quantity):
    """Places a market sell order."""
    from ib_insync import Stock, MarketOrder
    contract = Stock(ticker, "SMART", "USD")
    ib.qualifyContracts(contract)
    order = MarketOrder("SELL", quantity)
    trade = ib.placeOrder(contract, order)
    ib.sleep(1)
    logger.info(f"MARKET: SELL {quantity} {ticker}")
    return {"ticker": ticker, "action": "SELL", "quantity": quantity,
            "order_id": trade.order.orderId, "status": trade.orderStatus.status}


def get_ibkr_positions(ib):
    """Returns current IBKR positions."""
    result = []
    for pos in ib.positions():
        if pos.position != 0:
            result.append({"ticker": pos.contract.symbol, "quantity": int(pos.position),
                           "avg_cost": pos.avgCost})
    return result


def get_ibkr_account_summary(ib):
    """Returns account cash and equity."""
    summary = {}
    try:
        for av in ib.accountSummary():
            if av.tag == "TotalCashValue":
                summary["cash"] = float(av.value)
            elif av.tag == "NetLiquidation":
                summary["equity"] = float(av.value)
            elif av.tag == "GrossPositionValue":
                summary["invested"] = float(av.value)
    except Exception as e:
        logger.warning(f"Account summary failed: {e}")
    return summary


def execute_daily_signals(confirm=True):
    """Runs signal pipeline and executes trades via IBKR."""
    from live.positions import (get_open_positions, add_position, close_position,
                                 get_cash_balance, record_snapshot)
    from live.alerts import send_discord
    from utils.risk import compute_volatility_scaled_size, check_portfolio_drawdown

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    ib = get_ib_connection()
    if not ib:
        return

    try:
        account = get_ibkr_account_summary(ib)
        cash = account.get("cash", 0)
        equity = account.get("equity", 0)
        logger.info(f"IBKR Account: equity=${equity:,.2f}, cash=${cash:,.2f}")

        # Run signal pipeline
        from data.universe import build_universe
        from data.pipeline import add_technical_features, add_cross_sectional_features
        from models.predict import generate_ml_signals
        import yfinance as yf
        import pandas as pd

        logger.info("Running signal pipeline...")
        universe = build_universe(config)

        raw = yf.download(universe, period="90d", auto_adjust=True,
                           progress=False, threads=True)
        if raw.empty:
            logger.error("No market data available")
            return

        if isinstance(raw.columns, pd.MultiIndex):
            frames = []
            for ticker in universe:
                try:
                    t_data = raw.xs(ticker, level=1, axis=1).copy()
                    t_data.columns = [c.lower() for c in t_data.columns]
                    t_data["ticker"] = ticker
                    t_data["date"] = t_data.index
                    frames.append(t_data)
                except (KeyError, Exception):
                    continue
            df = pd.concat(frames, ignore_index=True)
        else:
            df = raw.reset_index()
            df.columns = [c.lower() for c in df.columns]

        df = add_technical_features(df)
        df = add_cross_sectional_features(df)

        try:
            from data.enrich import enrich_features
            df = enrich_features(df, universe)
        except Exception:
            pass

        df = generate_ml_signals(df)

        today_df = df[df["date"] == df["date"].max()]
        regime_ok = today_df["regime_ok"].any() if "regime_ok" in today_df.columns else False

        local_positions = get_open_positions()
        ibkr_positions = get_ibkr_positions(ib)
        open_tickers = [p["ticker"] for p in local_positions]
        non_hedge = [p for p in local_positions if p["ticker"] not in ("SH", "SDS", "SPXU")]
        slots_available = config["universe"]["max_positions"] - len(non_hedge)

        # Check for exits
        exits_executed = []
        for pos in local_positions:
            ticker = pos["ticker"]
            if ticker in ("SH", "SDS", "SPXU"):
                continue

            row = today_df[today_df["ticker"] == ticker]
            if row.empty:
                continue

            price = float(row.iloc[0]["close"])
            entry = pos["entry_price"]
            try:
                held_days = (date.today() - datetime.strptime(str(pos["entry_date"]), "%Y-%m-%d").date()).days
            except Exception:
                held_days = 0

            should_exit = False
            reason = ""

            if price <= pos["stop_price"]:
                should_exit, reason = True, "stop"
            elif price >= pos["target_price"]:
                should_exit, reason = True, "target"
            elif held_days >= config["backtest"]["hold_days"]:
                should_exit, reason = True, "time"
            elif (held_days >= config["backtest"]["early_exit_days"] and
                  price / entry - 1 < config["backtest"]["early_exit_threshold"]):
                should_exit, reason = True, "early_exit"

            if should_exit:
                ibkr_pos = next((p for p in ibkr_positions if p["ticker"] == ticker), None)
                qty = ibkr_pos["quantity"] if ibkr_pos else max(1, int(pos["size_usd"] / price))

                if qty > 0:
                    if confirm:
                        print(f"\n  SELL {ticker}: {qty} shares @ ~${price:.2f} ({reason})")
                        resp = input("  Execute? (y/n): ").strip().lower()
                        if resp != "y":
                            continue

                    place_sell_order(ib, ticker, qty)
                    close_position(ticker, price, reason)
                    exits_executed.append({"ticker": ticker, "reason": reason, "price": price})
                    slots_available += 1

                    try:
                        send_discord(f"SOLD {ticker}: {qty} shares @ ${price:.2f} ({reason})")
                    except Exception:
                        pass

        # Enter new positions
        entries_executed = []
        signals_today = today_df[today_df["signal"] == 1].sort_values("signal_score", ascending=False)
        allow_entries = regime_ok

        try:
            from live.positions import get_equity_curve
            eq = get_equity_curve()
            if not eq.empty:
                dd_status = check_portfolio_drawdown(eq["total_equity"].tolist(), config)
                if dd_status.get("halted"):
                    allow_entries = False
                    logger.warning("Entries blocked by drawdown limit")
        except Exception:
            pass

        if allow_entries and slots_available > 0 and len(signals_today) > 0:
            new_buys = signals_today.head(slots_available)

            for _, row in new_buys.iterrows():
                ticker = row["ticker"]
                if ticker in open_tickers:
                    continue

                price = float(row["close"])
                atr_pct = float(row.get("atr_pct", 0.02))
                atr = float(row.get("atr_14", price * 0.02))

                buy_size = compute_volatility_scaled_size(atr_pct, equity, config)
                quantity = max(1, int(buy_size / price))

                stop = round(price - atr * config["backtest"]["stop_loss_atr"], 2)
                target = round(price + atr * config["backtest"]["profit_target_atr"], 2)

                if quantity * price > cash * 0.95:
                    quantity = max(1, int(cash * 0.95 / price))

                if confirm:
                    print(f"\n  BUY {ticker}: {quantity} shares @ ~${price:.2f}")
                    print(f"    Size: ${quantity * price:,.2f} | Stop: ${stop:.2f} | Target: ${target:.2f}")
                    resp = input("  Execute? (y/n): ").strip().lower()
                    if resp != "y":
                        continue

                place_buy_order(ib, ticker, quantity, stop_price=stop, target_price=target)

                exit_date = (date.today() + timedelta(
                    days=config["backtest"]["hold_days"] + 2)).strftime("%Y-%m-%d")
                add_position(ticker, price, quantity * price, stop, target, exit_date)

                entries_executed.append({"ticker": ticker, "quantity": quantity,
                                         "price": price, "stop": stop, "target": target})
                cash -= quantity * price

                try:
                    send_discord(
                        f"BUY {ticker}: {quantity} shares @ ${price:.2f} | "
                        f"stop=${stop:.2f} | target=${target:.2f}")
                except Exception:
                    pass

        # SH hedge disabled - managed manually on Robinhood

        # Summary
        print("\n" + "=" * 55)
        print("  EXECUTION SUMMARY")
        print("=" * 55)
        print(f"  Regime:        {'FAVORABLE' if regime_ok else 'UNFAVORABLE'}")
        print(f"  Exits:         {len(exits_executed)}")
        for e in exits_executed:
            print(f"    SOLD {e['ticker']} @ ${e['price']:.2f} ({e['reason']})")
        print(f"  Entries:       {len(entries_executed)}")
        for e in entries_executed:
            print(f"    BUY {e['ticker']}: {e['quantity']} @ ${e['price']:.2f}")
        print(f"  IBKR Equity:   ${equity:,.2f}")
        print(f"  IBKR Cash:     ${cash:,.2f}")
        print("=" * 55)

    finally:
        ib.disconnect()
        logger.info("Disconnected from IBKR")


def show_status():
    """Shows IBKR connection status and positions."""
    ib = get_ib_connection()
    if not ib:
        return
    try:
        account = get_ibkr_account_summary(ib)
        positions = get_ibkr_positions(ib)

        print("\n" + "=" * 55)
        print("  IBKR ACCOUNT STATUS")
        print("=" * 55)
        print(f"  Equity:    ${account.get('equity', 0):,.2f}")
        print(f"  Cash:      ${account.get('cash', 0):,.2f}")
        print(f"  Invested:  ${account.get('invested', 0):,.2f}")
        print(f"\n  Positions ({len(positions)}):")
        if positions:
            for p in positions:
                print(f"    {p['ticker']:<6} {p['quantity']:>5} shares @ ${p['avg_cost']:.2f}")
        else:
            print("    None")
        print("=" * 55)
    finally:
        ib.disconnect()


def sync_positions():
    """Syncs IBKR positions to local database."""
    from live.positions import get_open_positions, add_position, close_position

    ib = get_ib_connection()
    if not ib:
        return
    try:
        ibkr_positions = get_ibkr_positions(ib)
        local_positions = get_open_positions()

        ibkr_tickers = {p["ticker"] for p in ibkr_positions}
        local_tickers = {p["ticker"] for p in local_positions}

        for pos in ibkr_positions:
            if pos["ticker"] not in local_tickers:
                logger.info(f"Adding {pos['ticker']} to local DB")
                price = pos["avg_cost"]
                size = pos["quantity"] * price
                is_hedge = pos["ticker"] in ("SH", "SDS", "SPXU", "PSQ", "DOG")
                add_position(pos["ticker"], price, size,
                             price * (0.5 if is_hedge else 0.93),
                             price * (2.0 if is_hedge else 1.15), "synced")

        for pos in local_positions:
            if pos["ticker"] not in ibkr_tickers and pos["ticker"] not in ("SH", "SDS", "SPXU"):
                logger.info(f"Closing {pos['ticker']} in local DB (not in IBKR)")
                close_position(pos["ticker"], pos["entry_price"], "external_close")

        print(f"\n  Sync complete: {len(ibkr_positions)} IBKR / {len(local_positions)} local")
    finally:
        ib.disconnect()


def run_broker(command="status"):
    if command == "status":
        show_status()
    elif command == "sync":
        sync_positions()
    elif command == "execute":
        execute_daily_signals(confirm=True)
    elif command == "auto":
        execute_daily_signals(confirm=False)
    else:
        print(f"Unknown: {command}")
        print("Usage: python main.py broker [status|sync|execute|auto]")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    run_broker(cmd)
