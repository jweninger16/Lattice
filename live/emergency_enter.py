"""
Emergency bracket entry — places today's hand-picked bracket orders on IBKR.
Runs a dry-run by default; pass --confirm to actually transmit.
Waits until 9:31 ET entry window if invoked early.
"""
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

import sys
import time
import asyncio
from datetime import datetime, time as dtime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from zoneinfo import ZoneInfo

# ── Hand-picked plan (from emergency_signals.py output 2026-04-17) ───
PLAN = [
    {
        "ticker": "NOW",
        "shares": 5,
        "stop": 86.01,
        "target": 120.73,
    },
    {
        "ticker": "DOCU",
        "shares": 12,
        "stop": 42.15,
        "target": 56.64,
    },
]

HOST = "127.0.0.1"
PORT = 7496       # LIVE
CLIENT_ID = 99    # avoid clash with swing=10, ORB=20
ENTRY_OPEN = dtime(9, 31)
ENTRY_CLOSE = dtime(10, 0)
ET = ZoneInfo("America/New_York")


def now_et():
    return datetime.now(ET)


def wait_for_entry_window():
    while True:
        n = now_et()
        t = n.time()
        if t < ENTRY_OPEN:
            remaining = (datetime.combine(n.date(), ENTRY_OPEN, tzinfo=ET) - n).total_seconds()
            print(f"  [{n.strftime('%H:%M:%S')}]  waiting {remaining:.0f}s until 9:31 ET")
            time.sleep(min(30, max(2, remaining / 2)))
        elif t > ENTRY_CLOSE:
            print(f"  [{n.strftime('%H:%M:%S')}]  past 10:00 ET — entry window closed. Aborting.")
            return False
        else:
            return True


def place_bracket(ib, Stock, MarketOrder, LimitOrder, StopOrder, item):
    t = item["ticker"]
    shares = item["shares"]
    stop_px = item["stop"]
    tgt_px = item["target"]

    contract = Stock(t, "SMART", "USD")
    ib.qualifyContracts(contract)

    # Pre-allocate order IDs so child parentId is correct
    parent_id = ib.client.getReqId()
    stop_id = ib.client.getReqId()
    tgt_id = ib.client.getReqId()

    parent = MarketOrder("BUY", shares)
    parent.orderId = parent_id
    parent.transmit = False
    parent.tif = "DAY"

    stop = StopOrder("SELL", shares, stop_px)
    stop.orderId = stop_id
    stop.parentId = parent_id
    stop.transmit = False
    stop.tif = "GTC"

    tgt = LimitOrder("SELL", shares, tgt_px)
    tgt.orderId = tgt_id
    tgt.parentId = parent_id
    tgt.transmit = True   # triggers the whole bracket
    tgt.tif = "GTC"

    print(f"  [{t}] placing parent MKT BUY {shares} (id={parent_id}) ...")
    ib.placeOrder(contract, parent)
    ib.sleep(1)
    print(f"  [{t}] placing child STP SELL @ ${stop_px:.2f} (id={stop_id}, parent={parent_id}) ...")
    ib.placeOrder(contract, stop)
    ib.sleep(1)
    print(f"  [{t}] placing child LMT SELL @ ${tgt_px:.2f} (id={tgt_id}, parent={parent_id}) — TRANSMIT")
    trade = ib.placeOrder(contract, tgt)
    ib.sleep(2)

    # Fetch all our trades back for status check
    trades = [t_ for t_ in ib.trades() if t_.order.orderId in (parent_id, stop_id, tgt_id)]
    for tr in trades:
        print(f"      order {tr.order.orderId}: {tr.orderStatus.status}")
    return trades


def main():
    confirm = "--confirm" in sys.argv
    skip_wait = "--now" in sys.argv   # bypass 9:31 wait for test

    print("=" * 60)
    print("  EMERGENCY BRACKET ENTRY")
    print("=" * 60)
    print(f"  Mode: {'LIVE TRANSMIT' if confirm else 'DRY RUN (pass --confirm to transmit)'}")
    print(f"  Port: {PORT}  ClientId: {CLIENT_ID}")
    print()
    print("  Plan:")
    total = 0
    for p in PLAN:
        cost = p["shares"] * ((p["stop"] + p["target"]) / 2)  # rough notional
        print(f"    {p['ticker']:5s}  BUY {p['shares']:3d} sh  STOP ${p['stop']:.2f}  TARGET ${p['target']:.2f}")
        total += cost
    print(f"  Approx notional: ~${total:.0f}")
    print()

    if not confirm:
        print("  DRY RUN complete. Pass --confirm to transmit to IBKR.")
        return

    # Connect
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    from ib_insync import IB, Stock, MarketOrder, LimitOrder, StopOrder

    ib = IB()
    print(f"  Connecting to TWS {HOST}:{PORT} (clientId={CLIENT_ID}) ...")
    ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=10)
    print(f"  Connected. Account: {ib.managedAccounts()}")

    try:
        if not skip_wait:
            if not wait_for_entry_window():
                return

        print()
        print(f"  ENTRY WINDOW OPEN — {now_et().strftime('%H:%M:%S ET')}")
        print()

        for item in PLAN:
            try:
                place_bracket(ib, Stock, MarketOrder, LimitOrder, StopOrder, item)
                print()
            except Exception as e:
                print(f"  [{item['ticker']}] FAILED: {e}")
                print()

        # Final status dump
        print()
        print("=" * 60)
        print("  FINAL ORDER STATUS")
        print("=" * 60)
        ib.sleep(3)
        for tr in ib.trades():
            if tr.contract.symbol in [p["ticker"] for p in PLAN]:
                print(f"  {tr.contract.symbol:5s}  orderId={tr.order.orderId:>5d}  "
                      f"{tr.order.action:4s} {tr.order.orderType:4s} "
                      f"qty={tr.order.totalQuantity:>3d}  status={tr.orderStatus.status}")
    finally:
        ib.disconnect()
        print()
        print("  Disconnected. Brackets are GTC — they persist on IBKR servers.")


if __name__ == "__main__":
    main()
