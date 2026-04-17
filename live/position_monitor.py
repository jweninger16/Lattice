"""
live/position_monitor.py
------------------------
Queries IBKR for current swing positions + live prices, posts a summary
to Discord. Can run once or loop every 30 min.

Usage:
    python live/position_monitor.py           # single check
    python live/position_monitor.py --loop    # repeat every 30 min
    python live/position_monitor.py --paper   # connect to 7497 instead of 7496
"""
import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import sys
import time
import asyncio
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from zoneinfo import ZoneInfo
from loguru import logger
from live.alerts import send_discord

ET = ZoneInfo("America/New_York")
LOOP_SECONDS = 1800   # 30 min
CLIENT_ID = 97        # avoid clash with swing=10, orb=20, emergency=99


def _ensure_loop():
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


def get_snapshot(paper: bool = False):
    """
    Pulls positions, live prices, and account NLV from IBKR.
    Returns (positions_list, account_dict).
    """
    _ensure_loop()
    from ib_insync import IB

    port = 7497 if paper else 7496
    ib = IB()
    ib.connect("127.0.0.1", port, clientId=CLIENT_ID, timeout=10)

    try:
        # Position list + live quotes
        positions = []
        raw_positions = [p for p in ib.positions() if p.position != 0]

        # Batch request market data
        md_handles = []
        for p in raw_positions:
            md = ib.reqMktData(p.contract, "", False, False)
            md_handles.append((p, md))
        ib.sleep(3)  # let quotes arrive

        for p, md in md_handles:
            last = md.last if (md.last and md.last > 0) else md.close
            if not last or last <= 0:
                last = p.avgCost  # fallback, no P&L info
            qty = int(p.position)
            avg = p.avgCost
            unreal = (last - avg) * qty
            positions.append({
                "ticker": p.contract.symbol,
                "qty": qty,
                "avg_cost": avg,
                "last": last,
                "unreal": unreal,
            })

        # Cancel the market data subs so we don't leak them
        for _, md in md_handles:
            try:
                ib.cancelMktData(md.contract)
            except Exception:
                pass

        # Account summary
        account = {}
        try:
            for v in ib.accountSummary():
                if v.tag == "NetLiquidation":
                    account["nlv"] = float(v.value)
                elif v.tag == "AvailableFunds":
                    account["cash"] = float(v.value)
        except Exception as e:
            logger.warning(f"Account summary failed: {e}")

        # Open orders (for visibility on bracket legs, including other clients')
        open_orders = []
        try:
            ib.reqAllOpenOrders()
            ib.sleep(1)
            seen = set()
            for trade in ib.trades():
                order = trade.order
                if order.orderId in seen:
                    continue
                seen.add(order.orderId)
                status = trade.orderStatus.status
                if status not in ("Submitted", "PreSubmitted", "PendingSubmit"):
                    continue
                open_orders.append({
                    "ticker": trade.contract.symbol,
                    "action": order.action,
                    "type": order.orderType,
                    "qty": float(order.totalQuantity),
                    "stop": getattr(order, "auxPrice", 0),
                    "limit": getattr(order, "lmtPrice", 0),
                    "status": status,
                })
        except Exception as e:
            logger.warning(f"Open orders fetch failed: {e}")

        return positions, account, open_orders

    finally:
        ib.disconnect()


def format_message(positions, account, open_orders):
    """Compact plain-text summary for Discord code block."""
    now_et = datetime.now(ET).strftime("%a %H:%M ET")
    lines = [f"SWING MONITOR — {now_et}"]

    nlv = account.get("nlv")
    cash = account.get("cash")
    if nlv is not None:
        header = f"NLV ${nlv:,.2f}"
        if cash is not None:
            header += f"  |  Cash ${cash:,.2f}"
        lines.append(header)

    lines.append("")

    if not positions:
        lines.append("No open positions.")
    else:
        lines.append("POSITIONS:")
        total = 0.0
        for p in positions:
            pct = ((p["last"] / p["avg_cost"] - 1) * 100) if p["avg_cost"] else 0.0
            sign = "+" if p["unreal"] >= 0 else ""
            lines.append(
                f"  {p['ticker']:5s} {p['qty']:>3d} sh  "
                f"entry ${p['avg_cost']:.2f} → ${p['last']:.2f}  "
                f"{pct:+.1f}%  ({sign}${p['unreal']:.2f})"
            )
            total += p["unreal"]
        sign = "+" if total >= 0 else ""
        lines.append("")
        lines.append(f"Unrealized P&L: {sign}${total:.2f}")

    # Brackets — group stop/target per ticker
    brackets = {}
    for o in open_orders:
        if o["action"] != "SELL":
            continue
        brackets.setdefault(o["ticker"], {})
        if o["type"] == "STP" and o["stop"]:
            brackets[o["ticker"]]["stop"] = o["stop"]
        elif o["type"] == "LMT" and o["limit"]:
            brackets[o["ticker"]]["target"] = o["limit"]

    if brackets:
        lines.append("")
        lines.append("BRACKETS:")
        for tk, b in brackets.items():
            stop = f"${b['stop']:.2f}" if "stop" in b else "—"
            tgt = f"${b['target']:.2f}" if "target" in b else "—"
            lines.append(f"  {tk:5s}  stop {stop}  target {tgt}")

    return "\n".join(lines)


def run_once(paper: bool = False, send: bool = True) -> str:
    try:
        positions, account, open_orders = get_snapshot(paper=paper)
        msg = format_message(positions, account, open_orders)
    except Exception as e:
        msg = f"SWING MONITOR ERROR\n\nConnection / query failed: {e}"
        logger.error(msg)

    print(msg)
    print()
    if send:
        try:
            ok = send_discord(msg)
            if not ok:
                logger.warning("Discord send returned false (webhook missing?)")
        except Exception as e:
            logger.warning(f"Discord send raised: {e}")
    return msg


def main():
    loop = "--loop" in sys.argv
    paper = "--paper" in sys.argv
    no_send = "--no-send" in sys.argv

    mode = "PAPER" if paper else "LIVE"
    print(f"Swing position monitor | {mode} | send={'off' if no_send else 'on'} | loop={loop}")

    while True:
        run_once(paper=paper, send=not no_send)
        if not loop:
            break
        print(f"sleeping {LOOP_SECONDS}s ...")
        time.sleep(LOOP_SECONDS)


if __name__ == "__main__":
    main()
