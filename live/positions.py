"""
live/positions.py
-----------------
Tracks open positions, portfolio state, and trade history.

SQLite-backed with persistent cash balance tracking.

Key improvements:
  - Tracks actual cash balance across sessions (not just assumed $20K)
  - Records daily equity snapshots for equity curve
  - Knows your real P&L, not estimated
  - Supports deposit/withdrawal tracking
  - All state survives restarts
"""

import sqlite3
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, date
from loguru import logger


DB_PATH = Path("live/positions.db")
LEGACY_JSON = Path("live/positions.json")


def _get_db():
    """Returns a connection to the positions database."""
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Positions table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            entry_price REAL NOT NULL,
            size_usd REAL NOT NULL,
            stop_price REAL NOT NULL,
            target_price REAL NOT NULL,
            entry_date TEXT NOT NULL,
            exit_date TEXT,
            planned_exit TEXT,
            status TEXT DEFAULT 'open',
            exit_price REAL,
            exit_reason TEXT,
            pnl_pct REAL,
            pnl_usd REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Portfolio state table — tracks cash, deposits, withdrawals
    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            cash_balance REAL NOT NULL,
            invested_value REAL DEFAULT 0,
            total_equity REAL NOT NULL,
            daily_pnl REAL DEFAULT 0,
            cumulative_pnl REAL DEFAULT 0,
            n_positions INTEGER DEFAULT 0,
            event_type TEXT DEFAULT 'snapshot',
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Cash transactions table — deposits, withdrawals
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cash_transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            amount REAL NOT NULL,
            transaction_type TEXT NOT NULL,
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    return conn


def _migrate_from_json():
    """One-time migration from old JSON format to SQLite."""
    if not LEGACY_JSON.exists():
        return
    try:
        with open(LEGACY_JSON) as f:
            positions = json.load(f)
        if not positions:
            return

        conn = _get_db()
        for p in positions:
            conn.execute("""
                INSERT INTO positions (ticker, entry_price, size_usd, stop_price,
                    target_price, entry_date, planned_exit, status, exit_price,
                    exit_reason, pnl_pct, pnl_usd)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                p.get("ticker"), p.get("entry_price", 0), p.get("size_usd", 0),
                p.get("stop_price", 0), p.get("target_price", 0),
                p.get("entry_date", ""), p.get("exit_date", ""),
                p.get("status", "open"), p.get("exit_price"),
                p.get("exit_reason"), p.get("pnl_pct"), p.get("pnl_usd"),
            ))
        conn.commit()
        conn.close()

        LEGACY_JSON.rename(LEGACY_JSON.with_suffix(".json.bak"))
        logger.info(f"Migrated {len(positions)} positions from JSON to SQLite")
    except Exception as e:
        logger.warning(f"JSON migration failed (OK if first run): {e}")


# Run migration on import
_migrate_from_json()


# ═══════════════════════════════════════════════════════════════════════
# Portfolio State
# ═══════════════════════════════════════════════════════════════════════

def initialize_portfolio(starting_capital: float):
    """
    Sets up initial portfolio state. Call once when starting.
    Skips if already initialized.
    """
    conn = _get_db()
    existing = conn.execute("SELECT COUNT(*) FROM portfolio_state").fetchone()[0]
    if existing > 0:
        logger.debug("Portfolio already initialized")
        conn.close()
        return

    conn.execute("""
        INSERT INTO portfolio_state (date, cash_balance, invested_value, total_equity,
            daily_pnl, cumulative_pnl, n_positions, event_type, notes)
        VALUES (?, ?, 0, ?, 0, 0, 0, 'initial', 'Portfolio initialized')
    """, (str(date.today()), starting_capital, starting_capital))
    conn.commit()
    conn.close()
    logger.info(f"Portfolio initialized with ${starting_capital:,.0f}")


def get_cash_balance() -> float:
    """Returns current cash balance."""
    conn = _get_db()
    row = conn.execute(
        "SELECT cash_balance FROM portfolio_state ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if row:
        return float(row["cash_balance"])
    return 0.0


def get_latest_equity() -> dict:
    """Returns the most recent portfolio snapshot."""
    conn = _get_db()
    row = conn.execute(
        "SELECT * FROM portfolio_state ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if row:
        return dict(row)
    return {}


def record_snapshot(cash: float, invested_value: float, n_positions: int,
                    notes: str = ""):
    """Records a daily portfolio snapshot."""
    total = cash + invested_value

    # Get previous snapshot for daily P&L
    prev = get_latest_equity()
    prev_equity = prev.get("total_equity", total) if prev else total
    daily_pnl = total - prev_equity

    # Get initial equity for cumulative P&L
    conn = _get_db()
    initial = conn.execute(
        "SELECT total_equity FROM portfolio_state WHERE event_type='initial' LIMIT 1"
    ).fetchone()
    initial_equity = float(initial["total_equity"]) if initial else total
    cumulative_pnl = total - initial_equity

    conn.execute("""
        INSERT INTO portfolio_state (date, cash_balance, invested_value, total_equity,
            daily_pnl, cumulative_pnl, n_positions, event_type, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'snapshot', ?)
    """, (str(date.today()), cash, invested_value, total,
          daily_pnl, cumulative_pnl, n_positions, notes))
    conn.commit()
    conn.close()

    logger.info(f"Portfolio snapshot: ${total:,.0f} (cash=${cash:,.0f}, "
                f"invested=${invested_value:,.0f}, daily={daily_pnl:+,.0f})")


def record_deposit(amount: float, notes: str = ""):
    """Records a cash deposit."""
    conn = _get_db()
    conn.execute("""
        INSERT INTO cash_transactions (date, amount, transaction_type, notes)
        VALUES (?, ?, 'deposit', ?)
    """, (str(date.today()), amount, notes))

    # Update latest cash balance
    current_cash = get_cash_balance()
    new_cash = current_cash + amount
    record_snapshot(new_cash, 0, 0, f"Deposit: ${amount:,.0f}")
    conn.close()
    logger.info(f"Deposit recorded: +${amount:,.0f}")


def record_withdrawal(amount: float, notes: str = ""):
    """Records a cash withdrawal."""
    conn = _get_db()
    conn.execute("""
        INSERT INTO cash_transactions (date, amount, transaction_type, notes)
        VALUES (?, ?, 'withdrawal', ?)
    """, (str(date.today()), -amount, notes))

    current_cash = get_cash_balance()
    new_cash = current_cash - amount
    record_snapshot(new_cash, 0, 0, f"Withdrawal: ${amount:,.0f}")
    conn.close()
    logger.info(f"Withdrawal recorded: -${amount:,.0f}")


def get_equity_curve() -> pd.DataFrame:
    """Returns equity curve as a DataFrame."""
    conn = _get_db()
    rows = conn.execute("""
        SELECT date, cash_balance, invested_value, total_equity,
               daily_pnl, cumulative_pnl, n_positions
        FROM portfolio_state
        ORDER BY id
    """).fetchall()
    conn.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_portfolio_summary() -> dict:
    """Returns complete portfolio summary."""
    latest = get_latest_equity()
    if not latest:
        return {}

    conn = _get_db()
    initial = conn.execute(
        "SELECT total_equity FROM portfolio_state WHERE event_type='initial' LIMIT 1"
    ).fetchone()
    initial_equity = float(initial["total_equity"]) if initial else 0

    n_snapshots = conn.execute("SELECT COUNT(*) FROM portfolio_state").fetchone()[0]
    conn.close()

    total_equity = latest.get("total_equity", 0)
    total_return = ((total_equity / initial_equity) - 1) * 100 if initial_equity > 0 else 0

    return {
        "cash": latest.get("cash_balance", 0),
        "invested": latest.get("invested_value", 0),
        "total_equity": total_equity,
        "initial_equity": initial_equity,
        "total_return_pct": total_return,
        "cumulative_pnl": latest.get("cumulative_pnl", 0),
        "daily_pnl": latest.get("daily_pnl", 0),
        "n_positions": latest.get("n_positions", 0),
        "n_snapshots": n_snapshots,
    }


# ═══════════════════════════════════════════════════════════════════════
# Position Management
# ═══════════════════════════════════════════════════════════════════════

def load_positions() -> list:
    """Loads all positions from database."""
    conn = _get_db()
    rows = conn.execute("SELECT * FROM positions ORDER BY entry_date DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_positions(positions: list):
    """Compatibility shim."""
    pass


def add_position(ticker: str, entry_price: float, size_usd: float,
                 stop_price: float, target_price: float, exit_date: str):
    """Records a new position and deducts from cash."""
    conn = _get_db()
    conn.execute("""
        INSERT INTO positions (ticker, entry_price, size_usd, stop_price,
            target_price, entry_date, planned_exit, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'open')
    """, (ticker, entry_price, size_usd, stop_price, target_price,
          str(date.today()), exit_date))
    conn.commit()
    conn.close()

    # Update cash balance
    current_cash = get_cash_balance()
    if current_cash > 0:
        new_cash = current_cash - size_usd
        # Don't record a full snapshot here — daily runner will do that
        conn = _get_db()
        conn.execute("""
            INSERT INTO portfolio_state (date, cash_balance, invested_value,
                total_equity, event_type, notes)
            VALUES (?, ?, ?, ?, 'trade_entry', ?)
        """, (str(date.today()), new_cash, size_usd,
              new_cash + size_usd, f"BUY {ticker} ${size_usd:,.0f}"))
        conn.commit()
        conn.close()

    logger.info(f"Position added: {ticker} @ ${entry_price:.2f} | "
                f"stop=${stop_price:.2f} | target=${target_price:.2f} | "
                f"size=${size_usd:,.0f}")


def close_position(ticker: str, exit_price: float, reason: str):
    """Marks a position as closed and returns cash + P&L."""
    conn = _get_db()
    row = conn.execute(
        "SELECT * FROM positions WHERE ticker=? AND status='open' LIMIT 1",
        (ticker,)
    ).fetchone()

    if row:
        pnl_pct = (exit_price / row["entry_price"] - 1) * 100
        pnl_usd = (exit_price / row["entry_price"] - 1) * row["size_usd"]
        proceeds = row["size_usd"] + pnl_usd

        conn.execute("""
            UPDATE positions SET status='closed', exit_price=?, exit_date=?,
                exit_reason=?, pnl_pct=?, pnl_usd=?, updated_at=CURRENT_TIMESTAMP
            WHERE id=?
        """, (exit_price, str(date.today()), reason, pnl_pct, pnl_usd, row["id"]))
        conn.commit()

        # Return proceeds to cash
        current_cash = get_cash_balance()
        new_cash = current_cash + proceeds
        conn.execute("""
            INSERT INTO portfolio_state (date, cash_balance, invested_value,
                total_equity, event_type, notes)
            VALUES (?, ?, 0, ?, 'trade_exit', ?)
        """, (str(date.today()), new_cash, new_cash,
              f"SELL {ticker} {reason} P&L={pnl_pct:+.1f}%"))
        conn.commit()

        logger.info(f"Position closed: {ticker} @ ${exit_price:.2f} | "
                    f"P&L: {pnl_pct:+.1f}% (${pnl_usd:+.0f}) | "
                    f"Cash now: ${new_cash:,.0f}")
    conn.close()


def get_open_positions() -> list:
    """Returns only open positions."""
    conn = _get_db()
    rows = conn.execute(
        "SELECT * FROM positions WHERE status='open' ORDER BY entry_date"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_positions_with_prices(current_prices: dict) -> list:
    """
    Updates open positions with current market prices.
    Determines action needed for each position.
    """
    positions = get_open_positions()
    today = str(date.today())

    for p in positions:
        ticker = p["ticker"]
        if ticker not in current_prices:
            p["current_price"] = p["entry_price"]
            p["action"] = "HOLD"
            continue

        prices = current_prices[ticker]
        p["current_price"] = prices["close"]
        p["unrealized_pct"] = (prices["close"] / p["entry_price"] - 1) * 100

        if prices["low"] <= p["stop_price"]:
            p["action"] = "SELL_STOP"
        elif prices["high"] >= p["target_price"]:
            p["action"] = "SELL_TARGET"
        elif today >= p.get("planned_exit", p.get("exit_date", "9999-12-31")):
            p["action"] = "SELL_TIME"
        else:
            p["action"] = "HOLD"

    return positions


def print_positions(positions: list):
    """Pretty prints current positions."""
    if not positions:
        print("  No open positions")
        return
    for p in positions:
        ret = p.get("unrealized_pct", 0)
        action = p.get("action", "HOLD")
        flag = " ← ACTION NEEDED" if action != "HOLD" else ""
        planned = p.get("planned_exit", p.get("exit_date", "TBD"))
        print(f"  {p['ticker']:<6} entry=${p['entry_price']:.2f} | "
              f"now=${p.get('current_price', p['entry_price']):.2f} | "
              f"{ret:+.1f}% | exit {planned} | {action}{flag}")


def portfolio_value(positions: list, cash: float) -> float:
    """Calculates total portfolio value."""
    position_value = sum(
        p.get("current_price", p["entry_price"]) / p["entry_price"] * p["size_usd"]
        for p in positions
    )
    return cash + position_value


def get_trade_history() -> pd.DataFrame:
    """Returns all closed trades as a DataFrame."""
    conn = _get_db()
    rows = conn.execute(
        "SELECT * FROM positions WHERE status='closed' ORDER BY exit_date DESC"
    ).fetchall()
    conn.close()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


def get_performance_summary() -> dict:
    """Returns aggregate performance stats for closed trades."""
    trades = get_trade_history()
    if trades.empty:
        return {}

    winners = trades[trades["pnl_usd"] > 0]
    losers = trades[trades["pnl_usd"] <= 0]

    return {
        "total_trades": len(trades),
        "win_rate": len(winners) / len(trades) * 100,
        "total_pnl": trades["pnl_usd"].sum(),
        "avg_pnl_pct": trades["pnl_pct"].mean(),
        "best_trade": trades["pnl_pct"].max(),
        "worst_trade": trades["pnl_pct"].min(),
        "profit_factor": (
            winners["pnl_usd"].sum() / abs(losers["pnl_usd"].sum())
            if len(losers) > 0 and losers["pnl_usd"].sum() != 0 else 999
        ),
    }


if __name__ == "__main__":
    print("=== Portfolio Status ===")

    summary = get_portfolio_summary()
    if summary:
        print(f"  Cash:          ${summary['cash']:,.0f}")
        print(f"  Invested:      ${summary['invested']:,.0f}")
        print(f"  Total Equity:  ${summary['total_equity']:,.0f}")
        print(f"  Total Return:  {summary['total_return_pct']:+.1f}%")
        print(f"  Daily P&L:     ${summary['daily_pnl']:+,.0f}")
    else:
        print("  No portfolio data. Run: python main.py daily")

    positions = get_open_positions()
    print(f"\n=== Open Positions ({len(positions)}) ===")
    if positions:
        print_positions(positions)
    else:
        print("  None")

    perf = get_performance_summary()
    if perf:
        print(f"\n=== Trade History ===")
        print(f"  Trades: {perf['total_trades']} | Win: {perf['win_rate']:.0f}% | "
              f"P&L: ${perf['total_pnl']:+,.0f}")
