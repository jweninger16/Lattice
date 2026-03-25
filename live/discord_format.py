"""
live/discord_format.py
-----------------------
Discord messages for the trading bots.
Short, plain English, no jargon.
"""


def _why_we_liked_it(extra):
    """Turns the technical data into a plain English reason."""
    if not extra:
        return ""

    reasons = []

    # Gap
    gap = extra.get("Gap", "")
    if gap:
        pct = gap.replace("+", "").replace("%", "").strip()
        try:
            g = float(pct)
            if g >= 3:
                reasons.append(f"opened up {gap} from yesterday")
            else:
                reasons.append(f"opened up {gap}")
        except ValueError:
            pass

    # Volume
    vol = extra.get("Volume", "")
    if vol:
        try:
            mult = float(vol.split("x")[0])
            if mult >= 4:
                reasons.append(f"huge volume ({vol})")
            elif mult >= 2:
                reasons.append(f"heavy volume ({vol})")
            else:
                reasons.append(f"above-average volume")
        except (ValueError, IndexError):
            pass

    # Short interest
    si = extra.get("Short int", "")
    if si:
        reasons.append(f"shorts could get squeezed ({si})")

    # Options flow
    opts = extra.get("Options", "")
    if opts and "UNUSUAL" in opts.upper():
        reasons.append("unusual options activity (smart money buying calls)")
    elif opts and "active" in opts.lower():
        reasons.append("active options interest")

    # Entry type
    entry = extra.get("Entry", "")
    if "vwap" in entry.lower():
        reasons.append("pulled back to a good entry price")
    elif "pm" in entry.lower() or "breakout" in entry.lower():
        reasons.append("broke above its pre-market high")

    if not reasons:
        return ""

    return "Why: " + ", ".join(reasons)


def fmt_entry(bot_name, ticker, direction, qty, price, stop, target,
              position_size=None, account_balance=None, extra=None):
    """
    Example:
        Bought 7 shares of CF @ $126.51
        Why: opened up +2.1%, heavy volume (2.7x), shorts could get squeezed (8%)
    """
    why = _why_we_liked_it(extra)

    lines = []
    lines.append(f"Bought {qty} shares of {ticker} @ ${price:.2f}")
    if why:
        lines.append(why)

    return "\n".join(lines)


def fmt_exit(bot_name, ticker, direction, qty, entry, exit_price,
             reason, account_balance=None, record=None):
    """
    Example (win):
        Sold CF @ $128.30 — made $12.53
        Bought at $126.51, profit locked in by trailing stop
        Account: $2,017 (3W / 1L)

    Example (loss):
        Sold CF @ $124.71 — lost $12.60
        Bought at $126.51, stopped out
        Account: $1,993 (2W / 2L)
    """
    if direction == "long":
        pnl_usd = (exit_price - entry) * qty
    else:
        pnl_usd = (entry - exit_price) * qty

    is_win = pnl_usd > 0

    # Plain English reason
    reason_text = {
        "trail": "profit locked in by trailing stop",
        "stop": "stopped out to limit the loss",
        "eod": "closed at end of day",
        "target": "hit the profit target",
    }.get(reason, "trade closed")

    lines = []
    if is_win:
        lines.append(f"Sold {ticker} @ ${exit_price:.2f} — "
                     f"made ${abs(pnl_usd):.2f}")
    else:
        lines.append(f"Sold {ticker} @ ${exit_price:.2f} — "
                     f"lost ${abs(pnl_usd):.2f}")

    lines.append(f"Bought at ${entry:.2f}, {reason_text}")

    if account_balance is not None and record:
        wins = record.get("wins", 0)
        losses = record.get("losses", 0)
        lines.append(f"Account: ${account_balance:,.2f} "
                     f"({wins}W / {losses}L)")
    elif account_balance is not None:
        lines.append(f"Account: ${account_balance:,.2f}")

    return "\n".join(lines)


def fmt_day_complete(bot_name, traded, ticker=None, pnl_pct=0,
                     account_balance=None, record=None, extra=None):
    """
    Example:
        Day's done — traded CF, made +0.98%
        Account: $2,005 (2W / 1L)

    Or:
        Day's done — no good setups today
        Account: $2,005
    """
    lines = []

    if traded and ticker:
        if pnl_pct > 0:
            lines.append(f"Day's done — traded {ticker}, "
                         f"made {pnl_pct:+.2f}%")
        elif pnl_pct < 0:
            lines.append(f"Day's done — traded {ticker}, "
                         f"lost {abs(pnl_pct):.2f}%")
        else:
            lines.append(f"Day's done — traded {ticker}, "
                         f"broke even")
    else:
        reason = extra.get("reason", "no good setups") if extra else "no good setups"
        lines.append(f"Day's done — {reason}")

    if account_balance is not None and record:
        wins = record.get("wins", 0)
        losses = record.get("losses", 0)
        trades_str = ""
        if extra and extra.get("Trades"):
            trades_str = f" | {extra['Trades']} trades today"
        lines.append(f"Account: ${account_balance:,.2f} "
                     f"({wins}W / {losses}L){trades_str}")
    elif account_balance is not None:
        lines.append(f"Account: ${account_balance:,.2f}")

    return "\n".join(lines)


def fmt_scan_results(bot_name, candidates, target_ticker, position_size=None):
    """
    Example:
        Found 5 stocks in play this morning
        Picking CF — up 2.1% on 2.7x volume
    """
    count = len(candidates)
    lines = []
    lines.append(f"Found {count} stock{'s' if count != 1 else ''} "
                 f"in play this morning")

    top = candidates[candidates["ticker"] == target_ticker]
    if not top.empty:
        row = top.iloc[0]
        lines.append(f"Picking {target_ticker} — "
                     f"up {row['gap_pct']:+.1f}% on "
                     f"{row['rvol']:.1f}x volume")

    return "\n".join(lines)


def fmt_gap_skip(bot_name, ticker, gap_pct, late_orb=False):
    if late_orb:
        return (f"{ticker} gapped {gap_pct:+.1f}% — too big for normal entry, "
                f"watching for a late setup at 10:30")
    else:
        return f"{ticker} gapped {gap_pct:+.1f}% — sitting this one out"


def fmt_error(bot_name, message):
    return f"Something went wrong:\n{message}"


def fmt_abort(bot_name, ticker, reason):
    return f"Skipping {ticker} — {reason}"
