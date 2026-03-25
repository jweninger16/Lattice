"""
lattice/stats_sync.py
----------------------
Syncs user trading stats via the git repo.
Each user's bot writes stats to lattice/stats/username.json.
On startup, git pull grabs everyone's stats.
After trades, git push shares yours.
"""

import json
import subprocess
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("lattice.stats")

PROJECT_ROOT = Path(__file__).parent.parent
STATS_DIR = PROJECT_ROOT / "lattice" / "stats"
ACCOUNT_FILE = PROJECT_ROOT / "live" / "gap_scanner_account.json"
USERS_FILE = PROJECT_ROOT / "lattice" / "users.json"


def get_username():
    """Get the current user's username from users.json."""
    try:
        if USERS_FILE.exists():
            with open(USERS_FILE) as f:
                users = json.load(f)
            if users:
                # Return the first (and usually only) user
                return list(users.keys())[0]
    except Exception:
        pass
    return None


def write_stats(extra_trade=None):
    """Write current user's stats to the shared stats directory."""
    username = get_username()
    if not username:
        return

    STATS_DIR.mkdir(parents=True, exist_ok=True)

    # Read account data
    account = {}
    if ACCOUNT_FILE.exists():
        with open(ACCOUNT_FILE) as f:
            account = json.load(f)

    # Read user info
    user_info = {}
    if USERS_FILE.exists():
        with open(USERS_FILE) as f:
            users = json.load(f)
        user_info = users.get(username, {})

    # Build stats summary
    stats = {
        "username": username,
        "name": user_info.get("name", username),
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "balance": account.get("balance", 0),
        "starting_capital": account.get("starting_capital", 0),
        "wins": account.get("wins", 0),
        "losses": account.get("losses", 0),
        "total_trades": account.get("total_trades", 0),
        "total_pnl_usd": account.get("total_pnl_usd", 0),
        "peak_balance": account.get("peak_balance", 0),
        "max_drawdown_pct": account.get("max_drawdown_pct", 0),
        "recent_trades": account.get("trade_history", [])[-10:],
        "equity_curve": [],
    }

    # Build mini equity curve from trade history
    history = account.get("trade_history", [])
    if history:
        starting = account.get("starting_capital", 2000)
        curve = [{"date": history[0].get("date", ""), "balance": starting}]
        for t in history:
            curve.append({
                "date": t.get("date", ""),
                "balance": t.get("balance_after", starting),
            })
        stats["equity_curve"] = curve[-20:]  # Last 20 points

    # Write to file
    stats_file = STATS_DIR / f"{username}.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Stats written to {stats_file}")
    return stats_file


def push_stats():
    """Push stats to git repo."""
    git_dir = PROJECT_ROOT / ".git"
    if not git_dir.exists():
        return

    try:
        subprocess.run(
            ["git", "add", "lattice/stats/"],
            cwd=str(PROJECT_ROOT),
            capture_output=True, timeout=10,
        )
        subprocess.run(
            ["git", "commit", "-m", "update stats", "--allow-empty"],
            cwd=str(PROJECT_ROOT),
            capture_output=True, timeout=10,
        )
        subprocess.run(
            ["git", "push"],
            cwd=str(PROJECT_ROOT),
            capture_output=True, timeout=15,
        )
        logger.info("Stats pushed to git")
    except Exception as e:
        logger.warning(f"Failed to push stats: {e}")


def read_all_stats():
    """Read all users' stats from the stats directory."""
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    all_stats = []

    for f in sorted(STATS_DIR.glob("*.json")):
        try:
            with open(f) as fh:
                stats = json.load(fh)
            all_stats.append(stats)
        except Exception:
            continue

    # Sort by balance descending
    all_stats.sort(key=lambda s: s.get("balance", 0), reverse=True)
    return all_stats


def sync_and_write():
    """Full sync: write own stats, push to git."""
    write_stats()
    push_stats()
