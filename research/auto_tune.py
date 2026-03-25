"""
research/auto_tune.py
-----------------------
Analyzes the trade log and recommends scoring weight adjustments.

Run after every 20 trades to see what's working and what's not.
After 50+ trades, it can generate optimized weights automatically.

Three modes:
  1. Report — shows which factors predict wins vs losses
  2. Recommend — suggests specific weight changes
  3. Apply — writes optimized weights to a config file the bot reads

Usage:
    python research/auto_tune.py                # Report mode
    python research/auto_tune.py --recommend    # Show recommendations
    python research/auto_tune.py --apply        # Write new weights
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, ".")

TRADE_LOG = Path("data/trade_log.csv")
WEIGHTS_FILE = Path("data/scoring_weights.json")

# Default weights (what the bot starts with)
DEFAULT_WEIGHTS = {
    "rvol": 0.35,
    "gap_pct": 0.20,
    "mom_5d": 0.10,
    "above_sma50": 0.10,
    "atr_pct": 0.10,
    "si_bonus_high": 3.0,      # SI > 10%
    "si_bonus_mid": 1.5,       # SI 5-10%
    "float_bonus_low": 2.0,    # Float < 50M
    "float_bonus_mid": 0.5,    # Float 50-200M
    "gap_sweet_spot": 1.0,     # Gap 1.5-3%
    "options_very_unusual": 3.0,
    "options_unusual": 1.5,
    "options_active": 0.5,
    "trail_atr_mult": 0.20,
}


def load_trades():
    if not TRADE_LOG.exists():
        print("No trade log found. Run the bot to generate trades.")
        return pd.DataFrame()
    df = pd.read_csv(TRADE_LOG)
    print(f"Loaded {len(df)} trades from {TRADE_LOG}")
    return df


def report(df):
    """Shows which factors correlate with winning trades."""
    if len(df) < 10:
        print(f"\nOnly {len(df)} trades — need at least 10 for basic analysis.")
        print("Keep trading, the data will build up.")
        return

    print(f"\n{'='*70}")
    print(f"  TRADE LOG ANALYSIS — {len(df)} trades")
    print(f"{'='*70}")

    wins = df[df["win"] == 1]
    losses = df[df["win"] == 0]
    print(f"\n  Overall: {len(wins)}W / {len(losses)}L "
          f"({len(wins)/len(df)*100:.0f}% win rate)")
    print(f"  Avg P&L: ${df['pnl_usd'].mean():+.2f}/trade")
    print(f"  Total P&L: ${df['pnl_usd'].sum():+.2f}")

    # Factor comparison: wins vs losses
    print(f"\n  FACTOR COMPARISON (wins vs losses):")
    print(f"  {'Factor':<25} {'Winners':>10} {'Losers':>10} {'Edge':>10}")
    print(f"  {'-'*55}")

    factors = [
        ("gap_pct", "Gap %"),
        ("rvol", "Rel. volume"),
        ("atr_pct", "ATR %"),
        ("si_pct", "Short interest %"),
        ("call_vol_oi_ratio", "Options vol/OI"),
        ("mom_5d", "5-day momentum"),
        ("score", "Composite score"),
        ("minutes_after_open", "Minutes after open"),
        ("trade_duration_min", "Trade duration"),
        ("max_unrealized_pct", "Max unrealized %"),
    ]

    for col, label in factors:
        if col in df.columns:
            w_mean = wins[col].mean() if len(wins) > 0 else 0
            l_mean = losses[col].mean() if len(losses) > 0 else 0
            edge = w_mean - l_mean
            print(f"  {label:<25} {w_mean:>10.2f} {l_mean:>10.2f} "
                  f"{edge:>+10.2f}")

    # Entry signal breakdown
    if "entry_signal" in df.columns:
        print(f"\n  BY ENTRY SIGNAL:")
        for signal in df["entry_signal"].unique():
            subset = df[df["entry_signal"] == signal]
            if len(subset) >= 2:
                wr = subset["win"].mean() * 100
                avg = subset["pnl_usd"].mean()
                print(f"    {signal:<20} {len(subset):>3} trades, "
                      f"{wr:.0f}% win, ${avg:+.2f}/trade")

    # Exit reason breakdown
    if "exit_reason" in df.columns:
        print(f"\n  BY EXIT REASON:")
        for reason in df["exit_reason"].unique():
            subset = df[df["exit_reason"] == reason]
            if len(subset) >= 2:
                avg = subset["pnl_usd"].mean()
                print(f"    {reason:<20} {len(subset):>3} trades, "
                      f"${avg:+.2f}/trade")

    # Day of week
    if "day_of_week" in df.columns:
        print(f"\n  BY DAY OF WEEK:")
        days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        for d in range(5):
            subset = df[df["day_of_week"] == d]
            if len(subset) >= 2:
                wr = subset["win"].mean() * 100
                avg = subset["pnl_usd"].mean()
                print(f"    {days[d]:<10} {len(subset):>3} trades, "
                      f"{wr:.0f}% win, ${avg:+.2f}/trade")

    # SPY correlation
    if "spy_change_pct" in df.columns and len(df) >= 10:
        spy_up = df[df["spy_change_pct"] > 0]
        spy_down = df[df["spy_change_pct"] <= 0]
        if len(spy_up) >= 3 and len(spy_down) >= 3:
            print(f"\n  MARKET CONTEXT:")
            print(f"    SPY up days:   {len(spy_up)} trades, "
                  f"{spy_up['win'].mean()*100:.0f}% win, "
                  f"${spy_up['pnl_usd'].mean():+.2f}/trade")
            print(f"    SPY down days: {len(spy_down)} trades, "
                  f"{spy_down['win'].mean()*100:.0f}% win, "
                  f"${spy_down['pnl_usd'].mean():+.2f}/trade")


def recommend(df):
    """Generates specific weight change recommendations."""
    if len(df) < 20:
        print(f"\nNeed at least 20 trades for recommendations. "
              f"You have {len(df)}.")
        return None

    print(f"\n{'='*70}")
    print(f"  SCORING WEIGHT RECOMMENDATIONS")
    print(f"{'='*70}")

    recommendations = {}
    current = DEFAULT_WEIGHTS.copy()

    # Load custom weights if they exist
    if WEIGHTS_FILE.exists():
        with open(WEIGHTS_FILE) as f:
            current.update(json.load(f))

    wins = df[df["win"] == 1]
    losses = df[df["win"] == 0]

    # Analyze each continuous factor
    factor_map = {
        "rvol": "rvol",
        "gap_pct": "gap_pct",
        "mom_5d": "mom_5d",
        "atr_pct": "atr_pct",
    }

    for weight_key, col in factor_map.items():
        if col not in df.columns:
            continue
        w_mean = wins[col].mean() if len(wins) > 0 else 0
        l_mean = losses[col].mean() if len(losses) > 0 else 0

        # If winners have higher values, increase the weight
        if w_mean > l_mean * 1.2:
            new_weight = min(current[weight_key] * 1.25, 0.50)
            if new_weight != current[weight_key]:
                recommendations[weight_key] = {
                    "old": current[weight_key],
                    "new": round(new_weight, 3),
                    "reason": f"winners avg {w_mean:.2f} vs losers {l_mean:.2f}"
                }
        elif l_mean > w_mean * 1.2:
            new_weight = max(current[weight_key] * 0.75, 0.05)
            if new_weight != current[weight_key]:
                recommendations[weight_key] = {
                    "old": current[weight_key],
                    "new": round(new_weight, 3),
                    "reason": f"losers had higher values ({l_mean:.2f} vs {w_mean:.2f})"
                }

    # Analyze SI bonus
    if "si_pct" in df.columns:
        high_si = df[df["si_pct"] >= 10]
        if len(high_si) >= 3:
            si_wr = high_si["win"].mean() * 100
            if si_wr > 70:
                recommendations["si_bonus_high"] = {
                    "old": current["si_bonus_high"],
                    "new": min(current["si_bonus_high"] + 0.5, 5.0),
                    "reason": f"high SI win rate: {si_wr:.0f}% — increase bonus"
                }
            elif si_wr < 45:
                recommendations["si_bonus_high"] = {
                    "old": current["si_bonus_high"],
                    "new": max(current["si_bonus_high"] - 1.0, 0.5),
                    "reason": f"high SI win rate only {si_wr:.0f}% — reduce bonus"
                }

    # Analyze trailing stop distance
    if "max_unrealized_pct" in df.columns and "pnl_pct" in df.columns:
        # If avg max unrealized is much higher than avg realized,
        # the trail might be too tight (giving back too much)
        avg_max = df["max_unrealized_pct"].mean()
        avg_realized = df["pnl_pct"].mean()
        if avg_max > 0 and avg_realized > 0:
            capture_rate = avg_realized / avg_max
            if capture_rate < 0.3:
                recommendations["trail_atr_mult"] = {
                    "old": current["trail_atr_mult"],
                    "new": round(current["trail_atr_mult"] * 0.8, 2),
                    "reason": (f"only capturing {capture_rate:.0%} of max move "
                               f"— tighten trail")
                }
            elif capture_rate > 0.7:
                recommendations["trail_atr_mult"] = {
                    "old": current["trail_atr_mult"],
                    "new": round(current["trail_atr_mult"] * 1.2, 2),
                    "reason": (f"capturing {capture_rate:.0%} of max move "
                               f"— trail working well, could loosen slightly")
                }

    # Entry signal analysis
    if "entry_signal" in df.columns:
        for signal in ["vwap_pullback", "pm_breakout", "timeout"]:
            subset = df[df["entry_signal"] == signal]
            if len(subset) >= 5:
                wr = subset["win"].mean() * 100
                avg = subset["pnl_usd"].mean()
                if wr < 40:
                    recommendations[f"entry_{signal}"] = {
                        "old": "enabled",
                        "new": "consider disabling",
                        "reason": f"only {wr:.0f}% win rate on {len(subset)} trades"
                    }

    # Print recommendations
    if recommendations:
        print(f"\n  {'Parameter':<25} {'Current':>10} {'Suggested':>10} "
              f"{'Reason'}")
        print(f"  {'-'*80}")
        for key, rec in recommendations.items():
            print(f"  {key:<25} {str(rec['old']):>10} "
                  f"{str(rec['new']):>10}  {rec['reason']}")
    else:
        print("\n  No changes recommended — current weights look good.")

    return recommendations


def apply_weights(recommendations):
    """Writes optimized weights to config file."""
    if not recommendations:
        print("No recommendations to apply.")
        return

    current = DEFAULT_WEIGHTS.copy()
    if WEIGHTS_FILE.exists():
        with open(WEIGHTS_FILE) as f:
            current.update(json.load(f))

    # Apply numeric recommendations
    for key, rec in recommendations.items():
        if isinstance(rec["new"], (int, float)):
            current[key] = rec["new"]

    WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(WEIGHTS_FILE, "w") as f:
        json.dump(current, f, indent=2)

    print(f"\nWeights saved to {WEIGHTS_FILE}")
    print(f"The bot will load these on next startup.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze trade log and tune scoring")
    parser.add_argument("--recommend", action="store_true",
                        help="Show weight recommendations")
    parser.add_argument("--apply", action="store_true",
                        help="Apply recommended weights")
    args = parser.parse_args()

    df = load_trades()
    if df.empty:
        return

    report(df)

    if args.recommend or args.apply:
        recs = recommend(df)
        if args.apply and recs:
            apply_weights(recs)


if __name__ == "__main__":
    main()
