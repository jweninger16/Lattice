"""
live/sentiment.py
------------------
Lightweight social sentiment data for gap scanner candidates.

Fetches StockTwits message count and sentiment (bullish/bearish ratio)
for each ticker. Data is logged silently — NOT used in scoring until
the auto-tuner confirms it has predictive value.

No API key required for basic StockTwits data.
"""

import logging
import time
import requests
from datetime import datetime, timedelta

logger = logging.getLogger("sentiment")

# StockTwits public API (no auth needed for basic stream)
ST_BASE = "https://api.stocktwits.com/api/2"


def get_stocktwits_sentiment(ticker, timeout=5):
    """
    Fetch StockTwits data for a single ticker.
    
    Returns dict with:
        - st_messages: number of recent messages (up to 30)
        - st_bullish: count of bullish-tagged messages
        - st_bearish: count of bearish-tagged messages
        - st_bull_ratio: bullish / total sentiment-tagged messages
        - st_watchers: number of people watching this ticker
    
    Returns empty dict on failure (never blocks trading).
    """
    try:
        url = f"{ST_BASE}/streams/symbol/{ticker}.json"
        resp = requests.get(url, timeout=timeout, headers={
            "User-Agent": "Lattice/1.0"
        })
        
        if resp.status_code == 429:
            # Rate limited — back off silently
            logger.debug(f"StockTwits rate limited on {ticker}")
            return {}
        
        if resp.status_code != 200:
            return {}
        
        data = resp.json()
        
        # Extract symbol info
        symbol = data.get("symbol", {})
        watchers = symbol.get("watchlist_count", 0)
        
        # Count messages and sentiment
        messages = data.get("messages", [])
        msg_count = len(messages)
        
        bullish = 0
        bearish = 0
        for msg in messages:
            sentiment = msg.get("entities", {}).get("sentiment", {})
            if sentiment:
                if sentiment.get("basic") == "Bullish":
                    bullish += 1
                elif sentiment.get("basic") == "Bearish":
                    bearish += 1
        
        total_sentiment = bullish + bearish
        bull_ratio = bullish / total_sentiment if total_sentiment > 0 else 0.5
        
        return {
            "st_messages": msg_count,
            "st_bullish": bullish,
            "st_bearish": bearish,
            "st_bull_ratio": round(bull_ratio, 2),
            "st_watchers": watchers,
        }
    
    except requests.Timeout:
        logger.debug(f"StockTwits timeout on {ticker}")
        return {}
    except Exception as e:
        logger.debug(f"StockTwits error on {ticker}: {e}")
        return {}


def get_batch_sentiment(tickers, delay=0.5):
    """
    Fetch StockTwits data for a list of tickers.
    
    Args:
        tickers: list of ticker strings
        delay: seconds between requests (rate limiting)
    
    Returns dict of {ticker: sentiment_data}
    """
    results = {}
    
    for ticker in tickers:
        data = get_stocktwits_sentiment(ticker)
        if data:
            results[ticker] = data
        time.sleep(delay)  # Be nice to their API
    
    fetched = len([r for r in results.values() if r])
    logger.info(f"  StockTwits sentiment: fetched {fetched}/{len(tickers)} tickers")
    
    return results
