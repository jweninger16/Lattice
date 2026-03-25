"""
Fix portfolio tracking to match actual capital.

Actual situation as of March 18, 2026:
  - $1,000 cash in IBKR
  - $1,000 incoming to IBKR (funds hold)
  - $500 SH position in Robinhood
  - Total: $2,500

This script:
  1. Updates the initial portfolio state to $1,500
     (the $1,000 cash + $500 SH that existed when tracking started)
  2. Fixes all snapshots so cumulative P&L is calculated correctly
  3. Does NOT count the incoming $1,000 yet — run record_deposit
     when it clears

Run: python fix_portfolio.py
"""
import sqlite3
from pathlib import Path

DB_PATH = Path("live/positions.db")

if not DB_PATH.exists():
    print(f"Database not found at {DB_PATH}")
    exit(1)

conn = sqlite3.connect(str(DB_PATH))
conn.row_factory = sqlite3.Row

print("=" * 60)
print("  PORTFOLIO FIX")
print("=" * 60)

# Show current state
row = conn.execute(
    "SELECT * FROM portfolio_state WHERE event_type='initial' LIMIT 1"
).fetchone()
print(f"\n  BEFORE:")
print(f"    Initial equity in DB: ${row['cash_balance']:,.2f}")

latest = conn.execute(
    "SELECT * FROM portfolio_state ORDER BY id DESC LIMIT 1"
).fetchone()
print(f"    Current equity: ${latest['total_equity']:,.2f}")

old_dd = (latest['total_equity'] / 20000 - 1) * 100
print(f"    Config initial_capital: $20,000")
print(f"    Calculated DD (config vs current): {old_dd:+.1f}% → HALTED")

# Fix 1: Update initial state to $1,500
# You started with $1,000 cash and bought $500 SH shortly after,
# so effective starting capital was $1,500 once SH was purchased.
# But the initial snapshot should reflect what you had at the START
# before the SH buy: $1,000 cash.
# The real fix is making the config match the DB.
#
# Actually, let's set initial to $1,000 (what you actually deposited)
# and make sure the config matches.

ACTUAL_STARTING_CAPITAL = 1000.0

print(f"\n  FIX:")
print(f"    Setting initial equity to ${ACTUAL_STARTING_CAPITAL:,.2f}")
print(f"    (matches your actual IBKR deposit)")

# Update initial state
conn.execute("""
    UPDATE portfolio_state
    SET cash_balance = ?, total_equity = ?
    WHERE event_type = 'initial'
""", (ACTUAL_STARTING_CAPITAL, ACTUAL_STARTING_CAPITAL))

# Recalculate cumulative P&L for all snapshots
rows = conn.execute(
    "SELECT id, total_equity FROM portfolio_state ORDER BY id"
).fetchall()

for r in rows:
    cum_pnl = r['total_equity'] - ACTUAL_STARTING_CAPITAL
    conn.execute(
        "UPDATE portfolio_state SET cumulative_pnl = ? WHERE id = ?",
        (cum_pnl, r['id'])
    )

conn.commit()

# Verify
print(f"\n  AFTER:")
row = conn.execute(
    "SELECT * FROM portfolio_state WHERE event_type='initial' LIMIT 1"
).fetchone()
print(f"    Initial equity: ${row['total_equity']:,.2f}")

latest = conn.execute(
    "SELECT * FROM portfolio_state ORDER BY id DESC LIMIT 1"
).fetchone()
print(f"    Current equity: ${latest['total_equity']:,.2f}")
print(f"    Cumulative P&L: ${latest['cumulative_pnl']:,.2f}")

new_dd = (latest['total_equity'] / ACTUAL_STARTING_CAPITAL - 1) * 100
print(f"    Drawdown vs initial: {new_dd:+.1f}% → {'HALTED' if new_dd < -20 else 'OK'}")

conn.close()

print(f"""
  NEXT STEPS:
  1. Update config/config.yaml:
     Change: initial_capital: 20000
     To:     initial_capital: 1000

  2. When the $1,000 transfer clears in IBKR, run:
     python -c "from live.positions import record_deposit; record_deposit(1000, 'IBKR transfer')"

  3. Run the daily briefing to verify:
     python main.py daily
""")
print("=" * 60)
