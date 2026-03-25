"""
live/ticker.py
--------------
Desktop ticker overlay — a small always-on-top window showing
portfolio status, positions, and market data at a glance.

Requires: pip install tkinter (usually built-in with Python on Windows)

Usage:
    python live/ticker.py
    python main.py ticker

Features:
  - Always-on-top compact window
  - Draggable to any screen position
  - Auto-refreshes every 30s during market hours, 5min after
  - Shows: P&L, positions, VIX, regime status, SPY change
  - Color-coded: green = profit, red = loss
  - Click to expand/collapse details
"""

import sys
import time
import threading
from datetime import datetime, date
from pathlib import Path
from loguru import logger

sys.path.insert(0, ".")

try:
    import tkinter as tk
    from tkinter import font as tkfont
except ImportError:
    print("tkinter not available. On Windows it's included with Python.")
    sys.exit(1)


class TickerOverlay:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Swing Trader")
        self.root.attributes("-topmost", True)
        self.root.overrideredirect(True)  # No title bar
        self.root.configure(bg="#0a0e18")

        # Position in bottom-right corner
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        self.width = 320
        self.height = 180
        x = screen_w - self.width - 20
        y = screen_h - self.height - 60  # above taskbar
        self.root.geometry(f"{self.width}x{self.height}+{x}+{y}")

        # Make draggable
        self._drag_data = {"x": 0, "y": 0}
        self.root.bind("<Button-1>", self._on_drag_start)
        self.root.bind("<B1-Motion>", self._on_drag)

        # Fonts
        self.font_mono = tkfont.Font(family="Consolas", size=9)
        self.font_mono_sm = tkfont.Font(family="Consolas", size=7)
        self.font_mono_lg = tkfont.Font(family="Consolas", size=14, weight="bold")
        self.font_title = tkfont.Font(family="Consolas", size=8, weight="bold")

        # Colors
        self.bg = "#0a0e18"
        self.bg_card = "#0f1520"
        self.green = "#00e676"
        self.red = "#ff5252"
        self.yellow = "#ffd740"
        self.muted = "#4a5568"
        self.text = "#c9d1d9"
        self.purple = "#b388ff"
        self.border = "#1a2035"

        self._build_ui()
        self._start_refresh()

    def _on_drag_start(self, event):
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def _on_drag(self, event):
        x = self.root.winfo_x() + event.x - self._drag_data["x"]
        y = self.root.winfo_y() + event.y - self._drag_data["y"]
        self.root.geometry(f"+{x}+{y}")

    def _build_ui(self):
        # Main frame with border
        self.frame = tk.Frame(self.root, bg=self.bg, highlightbackground=self.border,
                               highlightthickness=1)
        self.frame.pack(fill="both", expand=True)

        # Header row
        header = tk.Frame(self.frame, bg=self.bg)
        header.pack(fill="x", padx=6, pady=(4, 0))

        self.lbl_title = tk.Label(header, text="SWING TRADER", font=self.font_title,
                                   fg=self.text, bg=self.bg)
        self.lbl_title.pack(side="left")

        self.lbl_regime = tk.Label(header, text=" ● LOADING ", font=self.font_mono_sm,
                                    fg=self.yellow, bg=self.bg)
        self.lbl_regime.pack(side="right")

        # Close button
        close_btn = tk.Label(header, text=" ✕ ", font=self.font_mono_sm,
                              fg=self.muted, bg=self.bg, cursor="hand2")
        close_btn.pack(side="right", padx=(0, 2))
        close_btn.bind("<Button-1>", lambda e: self.root.destroy())

        # Separator
        tk.Frame(self.frame, bg=self.border, height=1).pack(fill="x", padx=6, pady=2)

        # Equity row
        eq_row = tk.Frame(self.frame, bg=self.bg)
        eq_row.pack(fill="x", padx=6, pady=(2, 0))

        self.lbl_equity = tk.Label(eq_row, text="$0", font=self.font_mono_lg,
                                    fg=self.green, bg=self.bg)
        self.lbl_equity.pack(side="left")

        self.lbl_pnl = tk.Label(eq_row, text="+$0 today", font=self.font_mono,
                                 fg=self.muted, bg=self.bg)
        self.lbl_pnl.pack(side="right")

        # Return row
        ret_row = tk.Frame(self.frame, bg=self.bg)
        ret_row.pack(fill="x", padx=6, pady=(0, 2))

        self.lbl_return = tk.Label(ret_row, text="+0.0% total", font=self.font_mono_sm,
                                    fg=self.muted, bg=self.bg)
        self.lbl_return.pack(side="left")

        self.lbl_cash = tk.Label(ret_row, text="Cash: $0", font=self.font_mono_sm,
                                  fg=self.muted, bg=self.bg)
        self.lbl_cash.pack(side="right")

        # Separator
        tk.Frame(self.frame, bg=self.border, height=1).pack(fill="x", padx=6, pady=2)

        # Positions area
        self.positions_frame = tk.Frame(self.frame, bg=self.bg)
        self.positions_frame.pack(fill="x", padx=6, pady=(0, 2))

        self.lbl_no_pos = tk.Label(self.positions_frame, text="No positions",
                                    font=self.font_mono_sm, fg=self.muted, bg=self.bg)
        self.lbl_no_pos.pack()

        # Market pulse row
        tk.Frame(self.frame, bg=self.border, height=1).pack(fill="x", padx=6, pady=1)
        pulse_row = tk.Frame(self.frame, bg=self.bg)
        pulse_row.pack(fill="x", padx=6, pady=(1, 2))

        self.lbl_spy = tk.Label(pulse_row, text="SPY —", font=self.font_mono_sm,
                                 fg=self.muted, bg=self.bg)
        self.lbl_spy.pack(side="left")

        self.lbl_vix = tk.Label(pulse_row, text="VIX —", font=self.font_mono_sm,
                                 fg=self.muted, bg=self.bg)
        self.lbl_vix.pack(side="left", padx=(10, 0))

        self.lbl_time = tk.Label(pulse_row, text="--:--", font=self.font_mono_sm,
                                  fg=self.muted, bg=self.bg)
        self.lbl_time.pack(side="right")

        # Bottom status
        status_row = tk.Frame(self.frame, bg=self.bg)
        status_row.pack(fill="x", padx=6, pady=(0, 3))

        self.lbl_status = tk.Label(status_row, text="Connecting...", font=self.font_mono_sm,
                                    fg=self.muted, bg=self.bg)
        self.lbl_status.pack(side="left")

    def _start_refresh(self):
        """Start background refresh thread."""
        def refresh_loop():
            while True:
                try:
                    self._refresh_data()
                except Exception as e:
                    logger.warning(f"Ticker refresh error: {e}")

                # 30s during market hours, 5min otherwise
                if self._is_market_hours():
                    time.sleep(30)
                else:
                    time.sleep(300)

        t = threading.Thread(target=refresh_loop, daemon=True)
        t.start()

        # Update clock every second
        self._update_clock()

    def _is_market_hours(self):
        now = datetime.now()
        # Approximate ET (adjust if needed for your timezone)
        h, m = now.hour, now.minute
        day = now.weekday()
        if day >= 5:
            return False
        # This is CST — market is 8:30-3:00 CST
        mins = h * 60 + m
        return 510 <= mins <= 900  # 8:30 AM - 3:00 PM CST

    def _update_clock(self):
        self.lbl_time.configure(text=datetime.now().strftime("%H:%M:%S"))
        self.root.after(1000, self._update_clock)

    def _refresh_data(self):
        """Fetch data and update UI."""
        import requests
        try:
            resp = requests.get("http://localhost:5050/api/status", timeout=5)
            data = resp.json()
            self.root.after(0, self._update_ui, data)

            resp2 = requests.get("http://localhost:5050/api/pulse", timeout=5)
            pulse = resp2.json()
            self.root.after(0, self._update_pulse, pulse)
        except Exception:
            self.root.after(0, lambda: self.lbl_status.configure(
                text="Dashboard offline — start with: python main.py dashboard",
                fg=self.red))

    def _update_ui(self, data):
        p = data["portfolio"]

        # Equity
        equity = p.get("total_equity", 0)
        ret_pct = p.get("total_return_pct", 0)
        daily_pnl = p.get("daily_pnl", 0)
        color = self.green if ret_pct >= 0 else self.red

        self.lbl_equity.configure(text=f"${equity:,.0f}", fg=color)
        self.lbl_return.configure(text=f"{ret_pct:+.1f}% total")

        pnl_color = self.green if daily_pnl >= 0 else self.red
        self.lbl_pnl.configure(text=f"{'+' if daily_pnl >= 0 else ''}${daily_pnl:,.0f} today",
                                fg=pnl_color)
        self.lbl_cash.configure(text=f"Cash: ${p.get('cash', 0):,.0f}")

        # Regime
        regime = data.get("regime", {})
        if regime.get("regime_ok"):
            self.lbl_regime.configure(text=" ● FAVORABLE ", fg=self.green)
        else:
            self.lbl_regime.configure(text=" ● UNFAVORABLE ", fg=self.red)

        # Positions
        for w in self.positions_frame.winfo_children():
            w.destroy()

        positions = data.get("positions", [])
        if positions:
            for pos in positions[:5]:
                row = tk.Frame(self.positions_frame, bg=self.bg)
                row.pack(fill="x")

                ticker = pos["ticker"]
                pnl = pos.get("unrealized_pct", 0)
                pnl_color = self.green if pnl >= 0 else self.red
                is_hedge = pos.get("is_hedge", False)

                tag = " [H]" if is_hedge else ""
                tk.Label(row, text=f"{ticker}{tag}", font=self.font_mono_sm,
                         fg=self.purple if is_hedge else self.text, bg=self.bg,
                         width=8, anchor="w").pack(side="left")
                tk.Label(row, text=f"${pos.get('current_price', 0):.2f}",
                         font=self.font_mono_sm, fg=self.muted, bg=self.bg,
                         width=8).pack(side="left")
                tk.Label(row, text=f"{pnl:+.1f}%", font=self.font_mono_sm,
                         fg=pnl_color, bg=self.bg, width=7, anchor="e").pack(side="right")
        else:
            tk.Label(self.positions_frame, text="No positions — cash",
                     font=self.font_mono_sm, fg=self.muted, bg=self.bg).pack()

        self.lbl_status.configure(text=f"Updated {datetime.now().strftime('%H:%M:%S')}",
                                   fg=self.muted)

    def _update_pulse(self, pulse):
        spy = pulse.get("spy", {})
        vix = pulse.get("vix", {})

        if spy.get("price"):
            ch = spy.get("change_pct", 0)
            color = self.green if ch >= 0 else self.red
            self.lbl_spy.configure(text=f"SPY {spy['price']:.0f} {ch:+.1f}%", fg=color)

        if vix.get("price"):
            ch = vix.get("change_pct", 0)
            color = self.red if ch > 0 else self.green
            self.lbl_vix.configure(text=f"VIX {vix['price']:.1f} {ch:+.1f}%", fg=color)

    def run(self):
        self.root.mainloop()


def run_ticker():
    logger.info("Starting desktop ticker overlay...")
    logger.info("NOTE: Dashboard must be running (python main.py dashboard) for data")
    app = TickerOverlay()
    app.run()


if __name__ == "__main__":
    run_ticker()
