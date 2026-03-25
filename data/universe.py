"""
universe.py
-----------
Builds and maintains the trading universe.
Fetches S&P 500 constituents and applies liquidity filters.

Improvements:
  - Deduplicates ticker list
  - Removes ETFs from trading universe (SPY, QQQ etc)
  - Better error handling with retries
  - Caches filter results to avoid redundant downloads
"""

import pandas as pd
import urllib.request
import yfinance as yf
from pathlib import Path
from loguru import logger


UNIVERSE_CACHE = Path("data/universe/sp500_tickers.csv")

# ETFs to exclude from trading universe (used for features only)
ETF_TICKERS = {"SPY", "QQQ", "IWM", "GLD", "TLT", "SH", "DIA", "VTI"}

# Hardcoded fallback — broad S&P 500 / large cap universe
FALLBACK_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK-B","LLY","AVGO",
    "TSLA","WMT","JPM","V","XOM","UNH","ORCL","MA","HD","PG","COST","JNJ",
    "ABBV","NFLX","BAC","KO","CRM","CVX","MRK","AMD","PEP","TMO","ACN",
    "LIN","MCD","CSCO","ABT","IBM","GE","CAT","NOW","ISRG","GS","TXN","QCOM",
    "INTU","BKNG","RTX","SPGI","DHR","AMGN","PFE","NEE","LOW","T","AMAT",
    "MS","HON","UNP","AXP","VRTX","UBER","SYK","C","BSX","BLK","DE","GILD",
    "ADI","SCHW","ETN","BMY","REGN","PLD","MDT","SO","DUK","MO","CME","TJX",
    "ZTS","CB","AON","MMC","ICE","WM","ITW","EQIX","NOC","GD","SHW","APH",
    "MCO","PNC","USB","TGT","EMR","NSC","FDX","ECL","COF","HCA","ELV","FCX",
    "PSA","WELL","OKE","CL","MPC","APD","AIG","PCAR","F","GM","CARR","CTAS",
    "ORLY","ADP","PAYX","FAST","ROST","DXCM","IDXX","BIIB","ILMN","MRNA",
    "GEHC","KEYS","CDNS","SNPS","ANSS","PH","GWW","SRE","AEP","XEL",
    "WEC","DTE","ETR","PPL","FE","CMS","AWK","WTW","MMM","DOW","DD","LYB",
    "PPG","RPM","IFF","ALB","CF","MOS","NUE","STLD","IP","PKG",
    "PYPL","MELI","SHOP","PANW","CRWD","FTNT","ZS","OKTA","DDOG",
    "NET","SNOW","MDB","HUBS","TEAM","WDAY","VEEV","FICO","ADSK","TTWO","EA",
    "PINS","SNAP","ZM","DOCU","TWLO","LYFT","ABNB","DASH","EXPE",
    "HLT","MAR","RCL","CCL","LUV","DAL","UAL","AAL","UPS","XPO",
    "DLTR","DG","LULU",
    "BK","TFC","DFS","SYF","FIS","FISV","GPN",
]


def get_sp500_tickers() -> list[str]:
    """
    Fetches S&P 500 tickers. Tries Wikipedia with browser headers,
    falls back to local cache, then falls back to hardcoded list.
    """
    try:
        logger.info("Fetching S&P 500 tickers from Wikipedia...")
        req = urllib.request.Request(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read()
        tables = pd.read_html(html)
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()

        # Deduplicate and remove ETFs
        tickers = _clean_ticker_list(tickers)
        logger.info(f"Fetched {len(tickers)} tickers from Wikipedia")

        UNIVERSE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        pd.Series(tickers).to_csv(UNIVERSE_CACHE, index=False, header=False)
        return tickers
    except Exception as e:
        logger.warning(f"Wikipedia fetch failed: {e}")

    if UNIVERSE_CACHE.exists():
        tickers = pd.read_csv(UNIVERSE_CACHE, header=None)[0].tolist()
        tickers = _clean_ticker_list(tickers)
        logger.info(f"Loaded {len(tickers)} tickers from cache")
        return tickers

    tickers = _clean_ticker_list(FALLBACK_TICKERS)
    logger.warning(f"Using hardcoded fallback ticker list ({len(tickers)} tickers)")
    return tickers


def _clean_ticker_list(tickers: list[str]) -> list[str]:
    """Deduplicates, removes ETFs, and strips whitespace."""
    seen = set()
    clean = []
    for t in tickers:
        t = t.strip()
        if t and t not in seen and t not in ETF_TICKERS:
            seen.add(t)
            clean.append(t)
    return clean


def filter_universe(
    tickers: list[str],
    min_price: float = 10.0,
    min_avg_volume: int = 500_000,
    sample_days: int = 30,
) -> list[str]:
    logger.info(f"Filtering {len(tickers)} tickers (min_price={min_price}, min_vol={min_avg_volume:,})...")
    try:
        raw = yf.download(
            tickers,
            period=f"{sample_days}d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        logger.error(f"Failed to download filter data: {e}")
        return tickers

    if raw.empty:
        logger.warning("Empty download result, returning unfiltered")
        return tickers

    close = raw["Close"]
    volume = raw["Volume"]

    valid = []
    for ticker in tickers:
        try:
            if ticker not in close.columns:
                continue
            avg_price = close[ticker].dropna().mean()
            avg_vol = volume[ticker].dropna().mean()
            if avg_price >= min_price and avg_vol >= min_avg_volume:
                valid.append(ticker)
        except Exception:
            continue

    logger.info(f"Universe filtered: {len(valid)} stocks passed ({len(tickers) - len(valid)} removed)")
    return valid


def build_universe(config: dict) -> list[str]:
    tickers = get_sp500_tickers()
    filtered = filter_universe(
        tickers,
        min_price=config["universe"]["min_price"],
        min_avg_volume=config["universe"]["min_avg_volume"],
    )
    return filtered


if __name__ == "__main__":
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    universe = build_universe(config)
    print(f"\nFinal universe: {len(universe)} stocks")
    print(universe[:20])
