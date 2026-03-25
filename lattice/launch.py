"""
lattice/launch.py
------------------
Desktop launcher for Lattice.
Auto-updates via git pull, starts the API server, opens the browser.
"""

import os
import sys
import time
import webbrowser
import subprocess
import threading
import socket
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
LOCAL_API = "http://localhost:8080"
HOSTED_URL = "https://jweninger16.github.io/Lattice"


def check_dependencies():
    try:
        import fastapi
        import uvicorn
    except ImportError:
        print("  Installing dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "fastapi", "uvicorn", "--quiet"
        ])


def is_server_running():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(("127.0.0.1", 8080))
    sock.close()
    return result == 0


def auto_update():
    """Pull latest code from GitHub if this is a git repo."""
    git_dir = PROJECT_ROOT / ".git"
    if not git_dir.exists():
        print("  Not a git repo - skipping auto-update")
        return

    print("  Checking for updates...")
    try:
        result = subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = result.stdout.strip()
        if "Already up to date" in output:
            print("  Already up to date.")
        elif "Updating" in output:
            print("  Updated! New changes pulled.")
            # Re-install deps in case requirements changed
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r",
                 str(PROJECT_ROOT / "requirements.txt"), "--quiet"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
            )
        else:
            print(f"  Git: {output}")
    except FileNotFoundError:
        print("  Git not installed - skipping auto-update")
    except subprocess.TimeoutExpired:
        print("  Update check timed out - continuing with current version")
    except Exception as e:
        print(f"  Update check failed: {e}")


def open_browser_delayed():
    time.sleep(2)
    print(f"  Opening {HOSTED_URL}")
    webbrowser.open(HOSTED_URL)


def main():
    os.chdir(str(PROJECT_ROOT))

    print()
    print("  ============================================")
    print("   LATTICE - Automated Trading Engine")
    print("  ============================================")
    print()

    # Check if already running
    if is_server_running():
        print("  Server already running!")
        webbrowser.open(HOSTED_URL)
        return

    # Auto-update from GitHub
    auto_update()

    # Install deps if needed
    check_dependencies()

    # Write and push stats on startup
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from lattice.stats_sync import sync_and_write
        sync_and_write()
        print("  Stats synced.")
    except Exception:
        pass

    # Open browser after server starts
    threading.Thread(target=open_browser_delayed, daemon=True).start()

    print(f"  API server: {LOCAL_API}")
    print(f"  Frontend:   {HOSTED_URL}")
    print("  Close this window to stop Lattice.")
    print()

    import uvicorn
    sys.path.insert(0, str(PROJECT_ROOT))
    from lattice.server import app
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")


if __name__ == "__main__":
    main()
