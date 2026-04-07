"""
data/market_intel.py
---------------------
Pre-market intelligence orchestrator.

Coordinates all data sources (pre-market scanner, Finnhub, FRED, CBOE,
short interest) and produces a PreMarketBrief that the ORB trader uses
to narrow its watchlist, score candidates, and adjust stance.

Every external call is isolated — one failure never blocks the others.
If all APIs fail, the system falls back to current behavior (full
85-ticker universe, pure vol_ratio ranking).

Usage:
    from data.market_intel import MarketIntelCollector
    collector = MarketIntelCollector()
    brief = collector.collect(tickers)

    # Standalone test:
    python data/market_intel.py
"""

import os
from dataclasses import dataclass, field
from datetime import datetime, date
from loguru import logger

from data.premarket_scanner import (
    scan_premarket_gaps,
    load_short_interest,
    rank_candidates,
)

try:
    from data.finnhub_client import FinnhubClient
except ImportError:
    FinnhubClient = None

try:
    from data.fred_client import FREDClient
except ImportError:
    FREDClient = None

try:
    from data.cboe_pcr import fetch_put_call_ratio
except ImportError:
    fetch_put_call_ratio = None

import pandas as pd


@dataclass
class PreMarketBrief:
    """Complete pre-market intelligence package."""

    # Market context
    vix_term: dict | None = None          # {vix, vix3m, term_slope, regime, signal}
    put_call_ratio: dict | None = None    # {pcr_equity, pcr_total, signal, date}
    economic_events: list = field(default_factory=list)
    is_event_day: bool = False

    # Per-ticker intelligence
    gap_rankings: pd.DataFrame | None = None
    earnings_today: set = field(default_factory=set)
    high_si_tickers: set = field(default_factory=set)
    news_tickers: set = field(default_factory=set)

    # Recommendations
    watchlist: list = field(default_factory=list)
    skip_tickers: set = field(default_factory=set)
    market_stance: str = "normal"       # "aggressive" / "normal" / "cautious"

    timestamp: datetime = field(default_factory=datetime.now)

    def summary(self) -> str:
        """One-line summary for logging."""
        parts = [f"stance={self.market_stance}"]
        if self.vix_term:
            parts.append(f"VIX={self.vix_term['vix']:.1f}({self.vix_term['regime']})")
        if self.put_call_ratio and self.put_call_ratio.get("pcr_equity"):
            parts.append(f"PCR={self.put_call_ratio['pcr_equity']:.2f}")
        if self.is_event_day:
            parts.append("EVENT_DAY")
        parts.append(f"watchlist={len(self.watchlist)}")
        parts.append(f"skip={len(self.skip_tickers)}")
        return " | ".join(parts)


def determine_market_stance(
    vix_term: dict | None,
    pcr: dict | None,
    events: list,
) -> str:
    """
    Determine overall market stance for the day.
    Returns: "aggressive", "normal", or "cautious"
    """
    caution_signals = 0

    # VIX term structure backwardation = near-term fear
    if vix_term and vix_term.get("term_slope", 0) < -0.05:
        caution_signals += 1

    # Extreme put/call ratio = panic
    if pcr and pcr.get("pcr_equity") and pcr["pcr_equity"] > 1.2:
        caution_signals += 1

    # High-impact economic event today
    high_impact = [e for e in events if e.get("impact") in ("high", 3)]
    if high_impact:
        caution_signals += 1

    if caution_signals >= 2:
        return "cautious"
    elif caution_signals == 0:
        # Check for aggressive conditions
        if pcr and pcr.get("pcr_equity") and 0.5 < pcr["pcr_equity"] < 0.8:
            if vix_term and vix_term.get("term_slope", 0) > 0.05:
                return "aggressive"
    return "normal"


class MarketIntelCollector:
    """Orchestrates all pre-market data collection."""

    def __init__(self):
        self.finnhub = FinnhubClient() if FinnhubClient and os.getenv("FINNHUB_API_KEY") else None
        self.fred = FREDClient() if FREDClient and os.getenv("FRED_API_KEY") else None
        self._log_sources()

    def _log_sources(self):
        sources = []
        sources.append("yfinance(premarket)")
        sources.append("short_interest(cached)")
        if self.finnhub and self.finnhub.available:
            sources.append("finnhub")
        if self.fred and self.fred.available:
            sources.append("fred")
        if fetch_put_call_ratio:
            sources.append("cboe")
        logger.info(f"Market intel sources: {', '.join(sources)}")

    def collect(self, tickers: list[str], max_watchlist: int = 30) -> PreMarketBrief:
        """
        Master method — collects all pre-market intelligence.
        Called by MultiORBTrader during 9:00-9:25 AM window.

        Each data source is isolated in try/except — one failure
        never blocks the others.
        """
        logger.info(f"Collecting pre-market intelligence for {len(tickers)} tickers...")
        start_time = datetime.now()

        # ── Phase 1: Market context (cached, low API cost) ────────
        vix_term = self._get_vix_term()
        pcr = self._get_pcr()
        events = self._get_economic_events()
        earnings = self._get_earnings_today(tickers)
        si_data = self._get_short_interest()

        # ── Phase 2: Pre-market gaps (real-time) ──────────────────
        gaps = self._scan_premarket(tickers)

        # ── Phase 3: News catalyst check (Finnhub, per-ticker) ────
        # Only check tickers that are gapping (top 20 by abs gap)
        news_tickers = set()
        if gaps is not None and not gaps.empty:
            top_gap_tickers = gaps.head(20)["ticker"].tolist()
            news_tickers = self._get_news_tickers(top_gap_tickers)

        # ── Phase 4: Score and rank ───────────────────────────────
        ranked = None
        if gaps is not None and not gaps.empty:
            ranked = rank_candidates(
                gaps,
                si_data=si_data,
                earnings_today=earnings,
                news_tickers=news_tickers,
                max_watchlist=max_watchlist,
            )

        # ── Phase 5: Build recommendations ────────────────────────
        is_event_day = bool([e for e in events if e.get("impact") in ("high", 3)])
        stance = determine_market_stance(vix_term, pcr, events)

        # Watchlist: top N non-skipped tickers by intel score
        watchlist = []
        skip_tickers = set()
        if ranked is not None and not ranked.empty:
            skip_tickers = set(ranked[ranked["skip_reason"] != ""]["ticker"])
            eligible = ranked[ranked["skip_reason"] == ""]
            watchlist = eligible.head(max_watchlist)["ticker"].tolist()

        # High SI tickers
        high_si = set()
        if si_data:
            high_si = {t for t, d in si_data.items()
                       if (d.get("short_pct_float") or 0) > 0.10}

        brief = PreMarketBrief(
            vix_term=vix_term,
            put_call_ratio=pcr,
            economic_events=events,
            is_event_day=is_event_day,
            gap_rankings=ranked,
            earnings_today=earnings,
            high_si_tickers=high_si,
            news_tickers=news_tickers,
            watchlist=watchlist,
            skip_tickers=skip_tickers,
            market_stance=stance,
            timestamp=datetime.now(),
        )

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Pre-market brief ready ({elapsed:.1f}s): {brief.summary()}")

        # ── Phase 6: Discord summary ──────────────────────────────
        self._send_discord_brief(brief)

        return brief

    # ── Data Source Methods (each isolated) ───────────────────────

    def _get_vix_term(self) -> dict | None:
        if not self.fred:
            # Fallback: try yfinance VIX
            return self._vix_fallback()
        try:
            return self.fred.get_vix_term_structure()
        except Exception as e:
            logger.warning(f"FRED VIX failed: {e}")
            return self._vix_fallback()

    def _vix_fallback(self) -> dict | None:
        """Fetch spot VIX via yfinance as fallback."""
        try:
            import yfinance as yf
            vix = yf.Ticker("^VIX").fast_info.get("lastPrice")
            if vix:
                return {
                    "vix": round(vix, 2),
                    "vix3m": None,
                    "term_slope": None,
                    "regime": "unknown",
                    "signal": "neutral",
                }
        except Exception:
            pass
        return None

    def _get_pcr(self) -> dict | None:
        if not fetch_put_call_ratio:
            return None
        try:
            return fetch_put_call_ratio()
        except Exception as e:
            logger.warning(f"CBOE PCR failed: {e}")
            return None

    def _get_economic_events(self) -> list:
        if not self.finnhub:
            return []
        try:
            return self.finnhub.get_economic_events_today()
        except Exception as e:
            logger.warning(f"Finnhub economic calendar failed: {e}")
            return []

    def _get_earnings_today(self, tickers: list[str]) -> set:
        if not self.finnhub:
            return set()
        try:
            return self.finnhub.get_earnings_today(tickers)
        except Exception as e:
            logger.warning(f"Finnhub earnings calendar failed: {e}")
            return set()

    def _get_short_interest(self) -> dict:
        try:
            return load_short_interest()
        except Exception as e:
            logger.warning(f"Short interest load failed: {e}")
            return {}

    def _scan_premarket(self, tickers: list[str]):
        try:
            return scan_premarket_gaps(tickers)
        except Exception as e:
            logger.warning(f"Pre-market scan failed: {e}")
            return pd.DataFrame()

    def _get_news_tickers(self, tickers: list[str]) -> set:
        if not self.finnhub:
            return set()
        try:
            return self.finnhub.get_news_tickers(tickers)
        except Exception as e:
            logger.warning(f"Finnhub news check failed: {e}")
            return set()

    # ── Discord Alert ─────────────────────────────────────────────

    def _send_discord_brief(self, brief: PreMarketBrief):
        """Send pre-market intelligence summary to Discord."""
        try:
            from live.alerts import send_discord
        except ImportError:
            return

        lines = []
        lines.append(f"PRE-MARKET INTEL ({brief.timestamp.strftime('%H:%M')} ET)")
        lines.append(f"Stance: {brief.market_stance.upper()}")

        # Market context
        ctx = []
        if brief.vix_term:
            v = brief.vix_term
            ctx.append(f"VIX {v['vix']:.1f} ({v['regime']})")
        if brief.put_call_ratio and brief.put_call_ratio.get("pcr_equity"):
            ctx.append(f"P/C {brief.put_call_ratio['pcr_equity']:.2f}")
        if ctx:
            lines.append("  " + " | ".join(ctx))

        # Event warnings
        if brief.is_event_day:
            high = [e.get("event", "?") for e in brief.economic_events
                    if e.get("impact") in ("high", 3)]
            lines.append(f"  EVENT DAY: {', '.join(high[:3])}")

        # Top gap candidates
        if brief.gap_rankings is not None and not brief.gap_rankings.empty:
            eligible = brief.gap_rankings[brief.gap_rankings["skip_reason"] == ""]
            top = eligible.head(5)
            lines.append(f"\nTop candidates ({len(brief.watchlist)} watching):")
            for _, r in top.iterrows():
                lines.append(f"  {r['ticker']:<6} gap={r['gap_pct']:+.2f}% "
                             f"pm_vol={r['pm_vol_ratio']:.1f}x "
                             f"score={r['intel_score']:.2f}")

        # Skips
        if brief.skip_tickers:
            lines.append(f"\nSkipping: {', '.join(sorted(brief.skip_tickers)[:10])}")

        try:
            send_discord("\n".join(lines))
        except Exception:
            pass


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from dotenv import load_dotenv
    load_dotenv()

    universe_path = "data/orb_universe.csv"
    try:
        tickers = pd.read_csv(universe_path)["ticker"].tolist()
    except Exception:
        print(f"Could not load {universe_path}")
        sys.exit(1)

    print(f"\nMarket Intel Collector Test ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print("=" * 70)

    collector = MarketIntelCollector()
    brief = collector.collect(tickers)

    print(f"\n{'='*70}")
    print(f"  BRIEF SUMMARY")
    print(f"{'='*70}")
    print(f"  {brief.summary()}")
    print(f"  Watchlist ({len(brief.watchlist)}): {brief.watchlist[:15]}")
    print(f"  Skip ({len(brief.skip_tickers)}): {brief.skip_tickers}")
    print(f"  Earnings today: {brief.earnings_today}")
    print(f"  High SI: {brief.high_si_tickers}")
    print(f"  News catalysts: {brief.news_tickers}")
    if brief.economic_events:
        print(f"  Economic events: {[e.get('event') for e in brief.economic_events[:5]]}")
