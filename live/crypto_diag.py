import sys
sys.path.insert(0, ".")
from live.crypto_paper import CryptoPaperBot, CryptoConfig
import yfinance as yf
import pandas as pd

b = CryptoPaperBot()
for ticker in ["BTC-USD", "ETH-USD"]:
    data = b.fetch_data(ticker)
    if data is not None:
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        a = b.analyze(ticker, data)
        if a:
            # How close to breakout?
            high_20 = float(data["High"].iloc[-(CryptoConfig.BREAKOUT_BARS+1):-1].max())
            current = a["current_price"]
            pct_from_breakout = ((current - high_20) / high_20) * 100

            print(f"{ticker}:")
            print(f"  Price:     ${current:,.2f}")
            print(f"  20-bar hi: ${high_20:,.2f} ({pct_from_breakout:+.2f}% away)")
            print(f"  RSI:       {a['rsi']}")
            print(f"  RVol:      {a['rvol']}x")
            print(f"  Breakout:  {a['breakout_signal']}")
            print(f"  Volume[-1]: {float(data['Volume'].iloc[-1]):,.0f}")
            print(f"  Volume avg: {float(data['Volume'].iloc[-20:].mean()):,.0f}")
            print()
