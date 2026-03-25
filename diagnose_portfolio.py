"""
Diagnostic: inspect the positions database to find the portfolio tracking issue.
Run: python diagnose_portfolio.py
"""
import sqlite3
from pathlib import Path

DB_PATH = Path("live/positions.db")

if not DB_PATH.exists():
    print(f"Database not found at {DB_PATH}")
    exit(1)

conn = sqlite3.connect(str(DB_PATH))
conn.row_factory = sqlite3.Row

print("=" * 70)
print("  PORTFOLIO DIAGNOSTIC")
print("=" * 70)

# 1. Initial state
print("\n1. INITIAL PORTFOLIO STATE:")
row = conn.execute(
    "SELECT * FROM portfolio_state WHERE event_type='initial' LIMIT 1"
).fetchone()
if row:
    print(f"   Date: {row['date']}")
    print(f"   Cash: ${row['cash_balance']:,.2f}")
    print(f"   Total equity: ${row['total_equity']:,.2f}")
else:
    print("   NO INITIAL STATE FOUND — this is the problem")

# 2. Latest snapshot
print("\n2. LATEST SNAPSHOT:")
row = conn.execute(
    "SELECT * FROM portfolio_state ORDER BY id DESC LIMIT 1"
).fetchone()
if row:
    print(f"   Date: {row['date']}")
    print(f"   Cash: ${row['cash_balance']:,.2f}")
    print(f"   Invested: ${row['invested_value']:,.2f}")
    print(f"   Total equity: ${row['total_equity']:,.2f}")
    print(f"   Cumulative P&L: ${row['cumulative_pnl']:,.2f}")
    print(f"   Event: {row['event_type']}")

# 3. All snapshots (recent)
print("\n3. RECENT SNAPSHOTS (last 10):")
rows = conn.execute(
    "SELECT id, date, cash_balance, invested_value, total_equity, "
    "cumulative_pnl, event_type, notes FROM portfolio_state "
    "ORDER BY id DESC LIMIT 10"
).fetchall()
print(f"   {'ID':>4} {'Date':<12} {'Cash':>10} {'Invested':>10} "
      f"{'Total':>10} {'Cum P&L':>10} {'Event':<12} Notes")
print(f"   {'-'*90}")
for r in reversed(rows):
    notes = (r['notes'] or '')[:25]
    print(f"   {r['id']:>4} {r['date']:<12} ${r['cash_balance']:>9,.2f} "
          f"${r['invested_value']:>9,.2f} ${r['total_equity']:>9,.2f} "
          f"${r['cumulative_pnl']:>9,.2f} {r['event_type']:<12} {notes}")

# 4. Open positions
print("\n4. OPEN POSITIONS:")
rows = conn.execute(
    "SELECT * FROM positions WHERE status='open'"
).fetchall()
if rows:
    for r in rows:
        print(f"   {r['ticker']:<6} entry=${r['entry_price']:.2f} "
              f"size=${r['size_usd']:.2f} stop=${r['stop_price']:.2f} "
              f"target=${r['target_price']:.2f} date={r['entry_date']}")
else:
    print("   None")

# 5. Closed positions
print("\n5. CLOSED POSITIONS (last 10):")
rows = conn.execute(
    "SELECT * FROM positions WHERE status='closed' ORDER BY id DESC LIMIT 10"
).fetchall()
if rows:
    for r in rows:
        pnl = r['pnl_pct'] or 0
        print(f"   {r['ticker']:<6} entry=${r['entry_price']:.2f} "
              f"exit=${r['exit_price'] or 0:.2f} P&L={pnl:+.1f}% "
              f"reason={r['exit_reason'] or 'N/A'} "
              f"date={r['entry_date']}→{r['exit_date'] or '?'}")
else:
    print("   None")

# 6. Cash transactions
print("\n6. CASH TRANSACTIONS:")
rows = conn.execute(
    "SELECT * FROM cash_transactions ORDER BY id"
).fetchall()
if rows:
    for r in rows:
        print(f"   {r['date']} {r['transaction_type']:<12} "
              f"${r['amount']:>+10,.2f}  {r['notes'] or ''}")
else:
    print("   None")

# 7. The math
print("\n7. THE MATH:")
initial = conn.execute(
    "SELECT total_equity FROM portfolio_state WHERE event_type='initial' LIMIT 1"
).fetchone()
latest = conn.execute(
    "SELECT total_equity FROM portfolio_state ORDER BY id DESC LIMIT 1"
).fetchone()
if initial and latest:
    init_eq = initial['total_equity']
    curr_eq = latest['total_equity']
    dd = (curr_eq / init_eq - 1) * 100
    print(f"   Initial equity: ${init_eq:,.2f}")
    print(f"   Current equity: ${curr_eq:,.2f}")
    print(f"   Drawdown: {dd:+.1f}%")
    print(f"   DD limit: -20% → {'HALTED' if dd < -20 else 'OK'}")

conn.close()

print("\n" + "=" * 70)
print("  Copy and paste this output back to me and I'll fix it.")
print("=" * 70)
