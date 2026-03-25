"""
utils/risk.py
-------------
Portfolio-level risk management utilities.

Improvements over original:
  - Enforces max drawdown stop (was configured but never checked)
  - Correlation-aware position screening
  - Volatility-scaled position sizing (replaces flat 10%)
  - Sector concentration limits
  - Daily loss circuit breaker
"""

import pandas as pd
import numpy as np
from loguru import logger


def compute_volatility_scaled_size(
    atr_pct: float,
    capital: float,
    config: dict,
) -> float:
    """
    Sizes positions so each contributes roughly equal daily risk.

    Instead of flat 10% per position, we target a fixed daily vol
    contribution. A stock with 1% daily ATR gets ~2x the allocation
    of one with 2% daily ATR.

    Returns position size in USD.
    """
    bt = config["backtest"]
    vol_target = bt.get("vol_target_per_position", 0.02)
    min_size = bt.get("min_position_size", 0.05)
    max_size = bt.get("max_position_size", 0.15)

    if atr_pct <= 0:
        atr_pct = 0.02  # fallback

    # Target: position_size * atr_pct = vol_target * capital
    # => position_size = vol_target / atr_pct
    raw_fraction = vol_target / atr_pct
    clamped = max(min_size, min(max_size, raw_fraction))
    size_usd = capital * clamped

    return size_usd


def check_portfolio_drawdown(
    equity_curve: list,
    config: dict,
) -> dict:
    """
    Checks if portfolio has hit max drawdown limit.
    Returns dict with status and details.
    """
    risk = config.get("risk", {})
    max_dd = risk.get("max_drawdown_pct", 0.15)
    enforce = risk.get("enforce_portfolio_stop", True)

    if not equity_curve or len(equity_curve) < 2:
        return {"halted": False, "current_dd": 0, "peak": equity_curve[-1] if equity_curve else 0}

    peak = max(equity_curve)
    current = equity_curve[-1]
    current_dd = (current - peak) / peak if peak > 0 else 0

    halted = enforce and (abs(current_dd) >= max_dd)

    if halted:
        logger.warning(
            f"PORTFOLIO STOP: drawdown {current_dd*100:.1f}% exceeds "
            f"limit {max_dd*100:.1f}%. Halting new entries."
        )

    return {
        "halted": halted,
        "current_dd": current_dd,
        "current_dd_pct": current_dd * 100,
        "peak": peak,
        "current": current,
        "max_dd_limit": max_dd,
    }


def check_daily_loss(
    today_equity: float,
    yesterday_equity: float,
    config: dict,
) -> bool:
    """
    Returns True if daily loss exceeds limit (circuit breaker).
    """
    risk = config.get("risk", {})
    max_daily = risk.get("max_daily_loss_pct", 0.03)

    if yesterday_equity <= 0:
        return False

    daily_return = (today_equity - yesterday_equity) / yesterday_equity
    if daily_return < -max_daily:
        logger.warning(
            f"DAILY LOSS BREAKER: {daily_return*100:.1f}% exceeds "
            f"limit {max_daily*100:.1f}%"
        )
        return True
    return False


def check_sector_concentration(
    ticker: str,
    sector_map: dict,
    open_positions: list,
    config: dict,
) -> bool:
    """
    Returns True if adding this ticker would exceed sector concentration limit.
    """
    max_sector = config["universe"].get("max_sector_concentration", 3)
    ticker_sector = sector_map.get(ticker, "UNKNOWN")

    same_sector = sum(
        1 for p in open_positions
        if sector_map.get(p.get("ticker", ""), "X") == ticker_sector
    )

    if same_sector >= max_sector:
        logger.debug(
            f"Sector limit: {ticker} ({ticker_sector}) rejected — "
            f"already {same_sector}/{max_sector} in sector"
        )
        return True
    return False


def check_correlation(
    ticker: str,
    open_tickers: list,
    correlation_matrix: pd.DataFrame,
    config: dict,
) -> bool:
    """
    Returns True if ticker is too correlated with existing positions.
    Uses pre-computed 63-day rolling correlation matrix.
    """
    min_distance = config["universe"].get("min_correlation_distance", 0.3)
    max_corr = 1.0 - min_distance  # 0.7 by default

    if ticker not in correlation_matrix.columns:
        return False  # can't check, allow

    for held in open_tickers:
        if held not in correlation_matrix.columns:
            continue
        corr = correlation_matrix.loc[ticker, held]
        if abs(corr) > max_corr:
            logger.debug(
                f"Correlation filter: {ticker} rejected — "
                f"corr={corr:.2f} with {held} (limit={max_corr:.2f})"
            )
            return True
    return False


def compute_correlation_matrix(df: pd.DataFrame, lookback: int = 63) -> pd.DataFrame:
    """
    Computes cross-sectional correlation matrix using recent returns.
    Returns a ticker x ticker DataFrame.
    """
    latest_date = df["date"].max()
    cutoff = latest_date - pd.Timedelta(days=lookback * 2)  # calendar days buffer
    recent = df[df["date"] >= cutoff].copy()

    # Pivot to wide format: dates x tickers
    pivot = recent.pivot_table(index="date", columns="ticker", values="ret_1d")

    # Need at least 30 observations
    valid_cols = pivot.columns[pivot.notna().sum() >= 30]
    pivot = pivot[valid_cols]

    corr = pivot.corr()
    return corr


def filter_candidates_by_risk(
    candidates: pd.DataFrame,
    open_positions: list,
    correlation_matrix: pd.DataFrame,
    sector_map: dict,
    config: dict,
) -> pd.DataFrame:
    """
    Applies all risk filters to candidate signals:
    1. Sector concentration
    2. Correlation with existing positions
    Returns filtered DataFrame.
    """
    if candidates.empty:
        return candidates

    open_tickers = [p.get("ticker", "") for p in open_positions]
    keep = []

    for _, row in candidates.iterrows():
        ticker = row["ticker"]

        # Sector check
        if check_sector_concentration(ticker, sector_map, open_positions, config):
            continue

        # Correlation check
        if check_correlation(ticker, open_tickers, correlation_matrix, config):
            continue

        keep.append(row)

    if keep:
        filtered = pd.DataFrame(keep)
        removed = len(candidates) - len(filtered)
        if removed > 0:
            logger.info(f"Risk filters removed {removed}/{len(candidates)} candidates")
        return filtered
    else:
        return pd.DataFrame()
