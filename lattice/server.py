"""
lattice/server.py
------------------
Backend server for Lattice trading dashboard.
Connects the React UI to the existing gap scanner bot.

Runs on localhost:8080 — open in browser to use.

Usage:
    cd C:\\Users\\jww9t\\OneDrive\\Desktop\\swing_trader_v2\\swing_trader_v2
    python lattice/server.py
"""

import os
import sys
import json
import csv
import asyncio
import hashlib
import secrets
import subprocess
import threading
import time
from pathlib import Path
from datetime import datetime, date
from collections import deque

from fastapi import FastAPI, WebSocket, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ── Paths (relative to the project root) ─────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
ACCOUNT_FILE = PROJECT_ROOT / "live" / "gap_scanner_account.json"
TRADE_LOG = PROJECT_ROOT / "data" / "trade_log.csv"
USERS_FILE = PROJECT_ROOT / "lattice" / "users.json"
SETTINGS_FILE = PROJECT_ROOT / "lattice" / "lattice_settings.json"

# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(title="Lattice", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Bot Manager ──────────────────────────────────────────────────────
class BotManager:
    """Manages the gap scanner bot process."""

    def __init__(self):
        self.process = None
        self.running = False
        self.log_buffer = deque(maxlen=500)
        self.clients = []  # WebSocket clients
        self._lock = threading.Lock()

    def start(self, live=True):
        if self.running:
            return {"status": "already_running"}

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "live" / "gap_scanner.py"),
        ]
        if live:
            cmd.append("--live")
        else:
            cmd.append("--dry-run")

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(PROJECT_ROOT),
        )
        self.running = True
        self.log_buffer.clear()

        # Start log reader thread
        thread = threading.Thread(target=self._read_output, daemon=True)
        thread.start()

        return {"status": "started"}

    def stop(self):
        if not self.running or not self.process:
            return {"status": "not_running"}

        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()

        self.running = False
        self.process = None
        return {"status": "stopped"}

    def _read_output(self):
        """Reads bot stdout and broadcasts to WebSocket clients."""
        try:
            for line in self.process.stdout:
                line = line.strip()
                if not line:
                    continue

                entry = {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "msg": line,
                }
                self.log_buffer.append(entry)

                # Broadcast to all connected clients
                for client in self.clients[:]:
                    try:
                        asyncio.run(client.send_json(entry))
                    except Exception:
                        self.clients.remove(client)
        except Exception:
            pass
        finally:
            self.running = False

    def get_recent_logs(self, n=50):
        return list(self.log_buffer)[-n:]


bot_manager = BotManager()


# ── User Auth ────────────────────────────────────────────────────────
class UserStore:
    """Simple JSON-based user storage."""

    def __init__(self):
        self.users_file = USERS_FILE
        self.users_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.users_file.exists():
            self._save({})

    def _load(self):
        with open(self.users_file) as f:
            return json.load(f)

    def _save(self, data):
        with open(self.users_file, "w") as f:
            json.dump(data, f, indent=2)

    def _hash_password(self, password, salt=None):
        if salt is None:
            salt = secrets.token_hex(16)
        hashed = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
        return salt, hashed

    def create_user(self, username, password, name, settings=None):
        users = self._load()
        if username in users:
            return None, "Username already exists"

        salt, hashed = self._hash_password(password)
        token = secrets.token_hex(32)

        users[username] = {
            "name": name,
            "salt": salt,
            "password_hash": hashed,
            "token": token,
            "settings": settings or {},
            "created": str(datetime.now()),
        }
        self._save(users)
        return token, None

    def login(self, username, password):
        users = self._load()
        if username not in users:
            return None, "User not found"

        user = users[username]
        _, hashed = self._hash_password(password, user["salt"])
        if hashed != user["password_hash"]:
            return None, "Wrong password"

        # Refresh token
        token = secrets.token_hex(32)
        users[username]["token"] = token
        self._save(users)
        return token, None

    def verify_token(self, token):
        users = self._load()
        for username, data in users.items():
            if data.get("token") == token:
                return username, data
        return None, None

    def get_settings(self, username):
        users = self._load()
        if username in users:
            return users[username].get("settings", {})
        return {}

    def update_settings(self, username, settings):
        users = self._load()
        if username in users:
            users[username]["settings"] = settings
            self._save(users)


user_store = UserStore()


# ── Data Readers ─────────────────────────────────────────────────────
def read_account():
    """Reads account state from gap_scanner_account.json."""
    if not ACCOUNT_FILE.exists():
        return {
            "balance": 0,
            "starting_capital": 0,
            "wins": 0,
            "losses": 0,
            "total_trades": 0,
            "total_pnl_usd": 0,
            "peak_balance": 0,
            "max_drawdown_pct": 0,
            "trade_history": [],
        }
    with open(ACCOUNT_FILE) as f:
        return json.load(f)


def read_trade_log():
    """Reads the full trade log CSV for training/display."""
    if not TRADE_LOG.exists():
        return []
    trades = []
    with open(TRADE_LOG, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in ["entry_price", "exit_price", "pnl_usd", "pnl_pct",
                        "score", "gap_pct", "rvol", "si_pct",
                        "call_vol_oi_ratio", "account_after"]:
                if key in row and row[key]:
                    try:
                        row[key] = float(row[key])
                    except (ValueError, TypeError):
                        pass
            trades.append(row)
    return trades


def get_equity_curve():
    """Builds equity curve from trade history."""
    account = read_account()
    history = account.get("trade_history", [])
    if not history:
        return [{"date": str(date.today()),
                 "balance": account.get("balance", 0)}]

    curve = []
    starting = account.get("starting_capital", 2000)
    curve.append({"date": history[0].get("date", ""),
                  "balance": starting})

    for trade in history:
        curve.append({
            "date": trade.get("date", ""),
            "balance": trade.get("balance_after", starting),
        })
    return curve


# ── Request Models ───────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str


class CreateAccountRequest(BaseModel):
    username: str
    password: str
    name: str
    account_type: str = "live"
    commission_plan: str = "lite"
    account_mode: str = "cash"


class BotControlRequest(BaseModel):
    action: str  # "start" or "stop"
    live: bool = True


# ── API Routes ───────────────────────────────────────────────────────

@app.post("/api/auth/login")
async def login(req: LoginRequest):
    token, error = user_store.login(req.username, req.password)
    if error:
        raise HTTPException(status_code=401, detail=error)
    users = user_store._load()
    name = users[req.username]["name"]
    return {"token": token, "username": req.username, "name": name}


@app.post("/api/auth/register")
async def register(req: CreateAccountRequest):
    settings = {
        "account_type": req.account_type,
        "commission_plan": req.commission_plan,
        "account_mode": req.account_mode,
        "port": 7496 if req.account_type == "live" else 7497,
    }
    token, error = user_store.create_user(
        req.username, req.password, req.name, settings)
    if error:
        raise HTTPException(status_code=400, detail=error)
    return {"token": token, "username": req.username, "name": req.name}


@app.get("/api/account")
async def get_account():
    account = read_account()
    return {
        "balance": account.get("balance", 0),
        "starting_capital": account.get("starting_capital", 0),
        "wins": account.get("wins", 0),
        "losses": account.get("losses", 0),
        "total_trades": account.get("total_trades", 0),
        "total_pnl_usd": account.get("total_pnl_usd", 0),
        "peak_balance": account.get("peak_balance", 0),
        "max_drawdown_pct": account.get("max_drawdown_pct", 0),
    }


@app.get("/api/trades")
async def get_trades():
    return read_trade_log()


@app.get("/api/trades/recent")
async def get_recent_trades():
    account = read_account()
    return account.get("trade_history", [])[-20:]


@app.get("/api/equity")
async def get_equity():
    return get_equity_curve()


@app.get("/api/settings")
async def get_settings():
    """Read current bot settings."""
    defaults = {
        "position_pct": 0.33,
        "max_trades_per_day": 5,
        "trail_atr_mult": 0.20,
        "entry_mode": "smart",
        "min_gap_pct": 1.5,
        "max_gap_pct": 8.0,
        "min_rvol": 1.5,
        "last_entry_hour": 15,
        "last_entry_minute": 0,
    }
    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE) as f:
            saved = json.load(f)
        defaults.update(saved)
    return defaults


class SettingsUpdate(BaseModel):
    position_pct: float = None
    max_trades_per_day: int = None
    trail_atr_mult: float = None
    entry_mode: str = None
    min_gap_pct: float = None
    max_gap_pct: float = None
    min_rvol: float = None
    last_entry_hour: int = None
    last_entry_minute: int = None


@app.post("/api/settings")
async def update_settings(req: SettingsUpdate):
    """Update bot settings. Takes effect on next bot start."""
    current = {}
    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE) as f:
            current = json.load(f)

    # Only update fields that were provided
    updates = req.dict(exclude_none=True)
    current.update(updates)

    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(current, f, indent=2)

    return {"status": "saved", "settings": current}


@app.get("/api/bot/status")
async def bot_status():
    return {
        "running": bot_manager.running,
        "logs": bot_manager.get_recent_logs(20),
    }


# ── Admin / Stats ───────────────────────────────────────────────────
@app.get("/api/admin/users")
async def admin_users():
    """Get all users' stats (synced via git)."""
    try:
        from lattice.stats_sync import read_all_stats
        return read_all_stats()
    except Exception as e:
        return []


@app.post("/api/admin/sync")
async def admin_sync():
    """Write own stats and push to git."""
    try:
        from lattice.stats_sync import sync_and_write
        sync_and_write()
        return {"status": "synced"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/api/bot/control")
async def bot_control(req: BotControlRequest):
    if req.action == "start":
        result = bot_manager.start(live=req.live)
    elif req.action == "stop":
        result = bot_manager.stop()
    else:
        raise HTTPException(status_code=400, detail="Invalid action")
    return result


@app.websocket("/api/ws/logs")
async def websocket_logs(websocket: WebSocket):
    """Streams bot logs in real time."""
    await websocket.accept()
    bot_manager.clients.append(websocket)

    # Send recent logs on connect
    for log in bot_manager.get_recent_logs(50):
        await websocket.send_json(log)

    try:
        while True:
            # Keep connection alive, receive any client messages
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except Exception:
        pass
    finally:
        if websocket in bot_manager.clients:
            bot_manager.clients.remove(websocket)


# ── Serve Frontend ───────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent / "frontend"


@app.get("/")
async def serve_index():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return JSONResponse({
        "message": "Lattice API is running",
        "docs": "/docs",
        "note": "Frontend not built yet — run the React build first",
    })


# Serve static files if frontend is built
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)),
              name="static")


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  LATTICE — Automated Trading Engine")
    print("=" * 50)
    print(f"  Server: http://localhost:8080")
    print(f"  API docs: http://localhost:8080/docs")
    print(f"  Project: {PROJECT_ROOT}")
    print(f"  Account: {ACCOUNT_FILE}")
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8080)
