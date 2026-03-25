"""
live/alerts.py
--------------
Alert system with Discord (primary) and Gmail SMS gateway (fallback).

Sends notifications via:
  1. Discord webhook (most reliable, free)
  2. Gmail SMTP to carrier gateway (fallback for carriers that accept it)
"""

import os
import json
import requests as req
from pathlib import Path
from datetime import datetime
from loguru import logger


def send_discord(message: str) -> bool:
    """Sends a message to Discord via webhook."""
    webhook_url = os.getenv("DISCORD_WEBHOOK")
    if not webhook_url:
        return False

    try:
        # Discord has a 2000 char limit per message
        # Split long messages if needed
        chunks = []
        if len(message) <= 1900:
            chunks = [message]
        else:
            lines = message.split("\n")
            chunk = ""
            for line in lines:
                if len(chunk) + len(line) + 1 > 1900:
                    chunks.append(chunk)
                    chunk = line
                else:
                    chunk += ("\n" + line if chunk else line)
            if chunk:
                chunks.append(chunk)

        for chunk in chunks:
            payload = {
                "content": f"```\n{chunk}\n```"
            }
            resp = req.post(webhook_url, json=payload, timeout=10)
            if resp.status_code not in (200, 204):
                logger.warning(f"Discord webhook returned {resp.status_code}: {resp.text[:100]}")
                return False

        logger.info("Alert sent via Discord")
        return True
    except Exception as e:
        logger.warning(f"Discord send failed: {e}")
        return False


def send_gmail_sms(message: str) -> bool:
    """Sends SMS via Gmail SMTP to carrier email gateways."""
    gmail = os.getenv("GMAIL_ADDRESS")
    app_pw = os.getenv("GMAIL_APP_PASSWORD")

    recipients = []
    tmobile = os.getenv("TMOBILE_SMS")
    verizon = os.getenv("VERIZON_SMS")
    if tmobile:
        recipients.append(tmobile)
    if verizon:
        recipients.append(verizon)

    if not recipients or not gmail or not app_pw:
        return False

    try:
        import smtplib
        from email.mime.text import MIMEText

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail, app_pw)
            for recipient in recipients:
                m = MIMEText(message)
                m["From"] = gmail
                m["To"] = recipient
                m["Subject"] = ""
                server.sendmail(gmail, recipient, m.as_string())
                logger.info(f"SMS sent via Gmail to {recipient}")
        return True
    except Exception as e:
        logger.warning(f"Gmail SMS failed: {e}")
        return False


def send_sms(message: str, to_number: str = None, from_number: str = None):
    """
    Sends alert via best available method:
      1. Discord webhook (primary — always reliable)
      2. Gmail SMS gateway (fallback — works for some carriers)

    The function is still called send_sms for backward compatibility
    but it now sends to Discord first.
    """
    discord_sent = False
    gmail_sent = False

    # Try Discord first (most reliable)
    discord_sent = send_discord(message)

    # Also try Gmail gateway (for carriers that accept it, like Verizon)
    gmail_sent = send_gmail_sms(message)

    if not discord_sent and not gmail_sent:
        raise ValueError("No notification method succeeded. Check .env configuration.")

    return True


def format_morning_brief(
    regime_ok: bool,
    pct_above_sma50: float,
    positions: list,
    signals: list,
    portfolio_value: float,
    initial_capital: float,
    slots_used: int,
    max_slots: int,
    sh_action: str = None,
    vix_context: dict = None,
) -> str:
    """Formats the morning briefing message."""
    today = datetime.now().strftime("%a %b %d").replace(" 0", " ")
    total_return = (portfolio_value / initial_capital - 1) * 100

    lines = [f"SWING TRADER {today}"]
    lines.append("")

    # Regime
    vix_str = ""
    if vix_context and vix_context.get("vix_current"):
        vix_str = f" | VIX {vix_context['vix_current']:.0f} ({vix_context['vix_regime']})"
    if regime_ok:
        lines.append(f"Regime: FAVORABLE ({pct_above_sma50:.0f}% > SMA50{vix_str})")
        if sh_action == "SELL":
            lines.append(f"HEDGE: SELL SH (regime recovered)")
            lines.append("")
    else:
        lines.append(f"Regime: UNFAVORABLE ({pct_above_sma50:.0f}% > SMA50{vix_str})")
        if sh_action == "BUY":
            lines.append(f"HEDGE: BUY SH (~50% of cash)")
        elif sh_action is None:
            lines.append(f"HEDGE: Holding SH")
        lines.append(f"Positions: {slots_used}/{max_slots} held")
        lines.append(f"Portfolio: ${portfolio_value:,.0f} ({total_return:+.1f}%)")
        return "\n".join(lines)

    lines.append("")

    # Exits needed
    exits = [p for p in positions if p.get("action") in ("SELL_STOP", "SELL_TIME", "SELL_TARGET")]
    if exits:
        for p in exits:
            reason = {"SELL_STOP": "stop hit", "SELL_TIME": "time exit", "SELL_TARGET": "target hit"}[p["action"]]
            ret = (p["current_price"] / p["entry_price"] - 1) * 100
            lines.append(f"SELL: {p['ticker']} ({reason} {ret:+.1f}%)")
        lines.append("")

    # New buys
    slots_available = max_slots - slots_used + len(exits)
    buys = signals[:slots_available]
    if buys and regime_ok:
        position_size = (portfolio_value * 0.10)
        for s in buys:
            lines.append(f"BUY: {s['ticker']} ${s['price']:.0f} (~${position_size:,.0f})")
        lines.append("")
    elif regime_ok and not buys:
        lines.append("No new signals today")
        lines.append("")

    # Holds
    holds = [p for p in positions if p.get("action") == "HOLD"]
    if holds:
        for p in holds:
            ret = (p["current_price"] / p["entry_price"] - 1) * 100
            lines.append(f"HOLD: {p['ticker']} {ret:+.1f}% (exit {p['exit_date']})")
        lines.append("")

    # Summary
    lines.append(f"Slots: {slots_used}/{max_slots}")
    lines.append(f"Portfolio: ${portfolio_value:,.0f} ({total_return:+.1f}%)")

    return "\n".join(lines)


def send_morning_brief(signals_df, positions_df, portfolio_value: float,
                       initial_capital: float = 20000.0, sh_action: str = None):
    """
    Main function called by the daily runner.
    Formats and sends the morning briefing.
    """
    import pandas as pd

    # Regime info
    regime_ok = bool(signals_df["regime_ok"].iloc[0]) if "regime_ok" in signals_df.columns else True
    pct_above = float(signals_df["pct_above_sma50"].iloc[0] * 100) if "pct_above_sma50" in signals_df.columns else 50.0

    # Format positions
    positions = []
    if positions_df is not None and len(positions_df) > 0:
        for _, row in positions_df.iterrows():
            positions.append({
                "ticker":        row["ticker"],
                "entry_price":   row["entry_price"],
                "current_price": row.get("current_price", row["entry_price"]),
                "entry_date":    row["entry_date"],
                "exit_date":     row.get("exit_date", "TBD"),
                "action":        row.get("action", "HOLD"),
            })

    # Format signals
    signals = []
    today_signals = signals_df[signals_df["signal"] == 1].sort_values("signal_score", ascending=False)
    for _, row in today_signals.head(6).iterrows():
        signals.append({
            "ticker": row["ticker"],
            "price":  row["close"],
            "score":  row["signal_score"],
        })

    slots_used = len([p for p in positions if p["action"] == "HOLD"])
    max_slots  = 6

    message = format_morning_brief(
        regime_ok=regime_ok,
        pct_above_sma50=pct_above,
        positions=positions,
        signals=signals,
        portfolio_value=portfolio_value,
        initial_capital=initial_capital,
        slots_used=slots_used,
        max_slots=max_slots,
        sh_action=sh_action,
    )

    logger.info(f"Morning brief:\n{message}")
    send_sms(message)
    return message


if __name__ == "__main__":
    # Quick test
    print("Testing Discord notification...")
    try:
        send_discord("SWING TRADER — Test Alert\n\nIf you see this, Discord notifications are working!")
        print("Discord: sent!")
    except Exception as e:
        print(f"Discord failed: {e}")

    print("\nTesting Gmail SMS...")
    try:
        send_gmail_sms("Swing Trader test")
        print("Gmail SMS: sent!")
    except Exception as e:
        print(f"Gmail SMS failed: {e}")
