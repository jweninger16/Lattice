"""
data/finnhub_client.py
-----------------------
Finnhub API client for economic calendar, earnings calendar,
company news, and social sentiment.

Free tier: 60 calls/minute, no credit card required.
Get API key at: https://finnhub.io/register

Usage:
    from data.finnhub_client import FinnhubClient
    client = FinnhubClient()
    events = client.get_economic_calendar_today()

    # Standalone test:
    python data/finnhub_client.py
"""

import os
import time
import pickle
import requests
from pathlib import Path
from datetime import datetime, date, timedelta
from loguru import logger


CACHE_DIR = Path("data/market_intel_cache")


class FinnhubClient:
    BASE_URL = "https://finnhub.io/api/v1"
    MAX_CALLS_PER_MINUTE = 55  # Stay under 60 limit

    def __init__(self):
        self.api_key = os.getenv("FINNHUB_API_KEY")
        self._calls = []  # Timestamps of recent calls

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _rate_limit(self):
        """Enforce rate limit by sleeping if needed."""
        now = time.time()
        # Remove calls older than 60 seconds
        self._calls = [t for t in self._calls if now - t < 60]
        if len(self._calls) >= self.MAX_CALLS_PER_MINUTE:
            sleep_time = 60 - (now - self._calls[0]) + 0.5
            if sleep_time > 0:
                logger.debug(f"Finnhub rate limit: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
        self._calls.append(time.time())

    def _get(self, endpoint: str, params: dict = None) -> dict | list | None:
        """Make a GET request with rate limiting and error handling."""
        if not self.api_key:
            return None

        self._rate_limit()
        params = params or {}
        params["token"] = self.api_key

        try:
            resp = requests.get(
                f"{self.BASE_URL}/{endpoint}",
                params=params,
                timeout=10,
            )
            if resp.status_code == 429:
                logger.warning("Finnhub rate limit hit (429). Waiting 60s...")
                time.sleep(60)
                return self._get(endpoint, {k: v for k, v in params.items() if k != "token"})
            if resp.status_code != 200:
                logger.warning(f"Finnhub {endpoint}: HTTP {resp.status_code}")
                return None
            return resp.json()
        except requests.Timeout:
            logger.warning(f"Finnhub {endpoint}: timeout")
            return None
        except Exception as e:
            logger.warning(f"Finnhub {endpoint}: {e}")
            return None

    # ── Economic Calendar ─────────────────────────────────────────

    def get_economic_calendar(self, from_date: str = None, to_date: str = None) -> list[dict]:
        """
        Get high-impact US economic events.
        Returns list of {event, date, time, impact, country, actual, estimate}.
        """
        today = date.today()
        from_date = from_date or today.strftime("%Y-%m-%d")
        to_date = to_date or (today + timedelta(days=7)).strftime("%Y-%m-%d")

        data = self._get("calendar/economic", {"from": from_date, "to": to_date})
        if not data or "economicCalendar" not in data:
            return []

        events = data["economicCalendar"]
        # Filter to US, high/medium impact
        us_events = [
            e for e in events
            if e.get("country") == "US"
            and e.get("impact") in ("high", "medium", 3, 2)
        ]
        return us_events

    def get_economic_events_today(self) -> list[dict]:
        """Get today's high-impact US economic events."""
        today = date.today().strftime("%Y-%m-%d")
        cached = self._load_cache("economic_events", ttl_hours=12)
        if cached is not None and cached.get("date") == today:
            return cached.get("events", [])

        events = self.get_economic_calendar(today, today)
        self._save_cache("economic_events", {"date": today, "events": events})
        return events

    # ── Earnings Calendar ─────────────────────────────────────────

    def get_earnings_calendar(self, from_date: str = None, to_date: str = None) -> dict[str, str]:
        """
        Get earnings reports in date range.
        Returns {ticker: date_str} for all companies reporting.
        """
        today = date.today()
        from_date = from_date or today.strftime("%Y-%m-%d")
        to_date = to_date or (today + timedelta(days=7)).strftime("%Y-%m-%d")

        data = self._get("calendar/earnings", {"from": from_date, "to": to_date})
        if not data or "earningsCalendar" not in data:
            return {}

        result = {}
        for e in data["earningsCalendar"]:
            ticker = e.get("symbol")
            dt = e.get("date")
            if ticker and dt:
                result[ticker] = dt
        return result

    def get_earnings_today(self, tickers: list[str] = None) -> set[str]:
        """Get tickers reporting earnings today."""
        today = date.today().strftime("%Y-%m-%d")
        cached = self._load_cache("earnings_today", ttl_hours=12)
        if cached is not None and cached.get("date") == today:
            reporting = cached.get("tickers", set())
        else:
            calendar = self.get_earnings_calendar(today, today)
            reporting = set(calendar.keys())
            self._save_cache("earnings_today", {"date": today, "tickers": reporting})

        if tickers:
            return reporting & set(tickers)
        return reporting

    # ── Company News ──────────────────────────────────────────────

    def get_company_news(self, ticker: str, days_back: int = 1) -> list[dict]:
        """
        Get recent news for a ticker.
        Returns list of {headline, summary, source, datetime}.
        """
        today = date.today()
        from_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
        to_date = today.strftime("%Y-%m-%d")

        data = self._get("company-news", {
            "symbol": ticker,
            "from": from_date,
            "to": to_date,
        })
        return data if isinstance(data, list) else []

    def get_news_tickers(self, tickers: list[str], max_tickers: int = 20) -> set[str]:
        """
        Check which tickers have recent news (overnight catalyst).
        Only checks top max_tickers to conserve API calls.
        Returns set of tickers with news.
        """
        has_news = set()
        for ticker in tickers[:max_tickers]:
            news = self.get_company_news(ticker, days_back=1)
            if news and len(news) > 0:
                has_news.add(ticker)
        logger.info(f"News catalyst check: {len(has_news)}/{min(len(tickers), max_tickers)} "
                    f"tickers have overnight news")
        return has_news

    # ── Social Sentiment ──────────────────────────────────────────

    def get_social_sentiment(self, ticker: str) -> dict:
        """
        Get Reddit/Twitter sentiment for a ticker.
        Returns {reddit_mention, twitter_mention, score} or empty dict.
        """
        data = self._get("stock/social-sentiment", {"symbol": ticker})
        if not data:
            return {}
        # Aggregate recent mentions
        reddit = data.get("reddit", [])
        twitter = data.get("twitter", [])
        return {
            "reddit_mentions": sum(r.get("mention", 0) for r in reddit[-24:]),
            "twitter_mentions": sum(t.get("mention", 0) for t in twitter[-24:]),
            "reddit_score": sum(r.get("positiveScore", 0) - r.get("negativeScore", 0)
                               for r in reddit[-24:]),
        }

    # ── Cache Helpers ─────────────────────────────────────────────

    def _load_cache(self, name: str, ttl_hours: int = 12):
        cache_file = CACHE_DIR / f"{name}.pkl"
        if not cache_file.exists():
            return None
        try:
            age = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime))
            if age.total_seconds() / 3600 > ttl_hours:
                return None
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _save_cache(self, name: str, data):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(CACHE_DIR / f"{name}.pkl", "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.debug(f"Cache write failed for {name}: {e}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    client = FinnhubClient()
    if not client.available:
        print("FINNHUB_API_KEY not set in .env")
        print("Get a free key at: https://finnhub.io/register")
        exit(1)

    print(f"\nFinnhub API Test ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print("=" * 60)

    # Economic calendar
    events = client.get_economic_events_today()
    print(f"\nEconomic events today: {len(events)}")
    for e in events[:5]:
        print(f"  {e.get('event', '?'):<40} impact={e.get('impact', '?')}")

    # Earnings today
    earnings = client.get_earnings_today()
    print(f"\nEarnings today: {len(earnings)} companies")
    if earnings:
        print(f"  Sample: {list(earnings)[:10]}")

    # News for a sample ticker
    print("\nRecent AAPL news:")
    news = client.get_company_news("AAPL", days_back=1)
    for n in news[:3]:
        print(f"  {n.get('headline', '?')[:70]}")

    print(f"\nTotal API calls this session: {len(client._calls)}")
