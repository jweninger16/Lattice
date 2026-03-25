# Swing Trader — v2

A systematic swing trading system targeting US equities (S&P 500).
Built for a ~$20K account with walk-forward validated backtesting.

---

## What's New in v2

### Risk Management (previously missing)
- **Portfolio drawdown enforcement** — halts new entries when DD exceeds 15% (was configured but never checked in v1)
- **Daily loss circuit breaker** — stops new entries if portfolio drops >3% in one day
- **Sector concentration limits** — max 3 positions per sector to prevent correlated blowups
- **Volatility-scaled position sizing** — replaces flat 10% allocation; a 30% vol stock gets smaller size than a 15% vol stock
- **Cooldown period** — 5-day pause after hitting max drawdown before resuming

### Better Signals
- **Regime smoothing** — requires 2 consecutive days to change regime state (avoids whipsawing)
- **Earnings avoidance** — won't enter positions within 5 days of earnings announcements
- **Gap filter** — skips stocks that gapped up >3% (avoids buying euphoria)
- **Improved signal scoring** — multi-factor score includes momentum acceleration, Bollinger squeeze, sector strength

### Better Data Pipeline
- **Overnight gap detection** — open vs previous close as a feature
- **Bollinger Band width** — identifies volatility squeeze setups
- **Intraday range** — (high - low) / close as a feature
- **On-balance volume slope** — 20-day OBV trend
- **Bad data detection** — removes >50% spike-and-revert anomalies

### Better ML Model
- **Lower learning rate** (0.03 vs 0.05) with more boosting rounds for better generalization
- **Stronger regularization** — higher L1/L2 penalties and explicit max_depth
- **Per-date median imputation** — avoids information leakage from global median fill
- **Quintile monotonicity check** — verifies model predictions rank correctly
- **Cross-fold variance tracking** — reports AUC mean ± std

### Infrastructure
- **SQLite position tracker** — replaces JSON file (crash-safe, concurrent access, queryable)
- **Trade history command** — `python main.py history` shows all closed trades
- **Risk status command** — `python main.py risk` shows portfolio risk metrics
- **Removed unused dependencies** — no more vectorbt, streamlit, plotly in requirements

---

## Strategy

**Signal:** Relative Strength + Volume Confirmation + ML Scoring
- LightGBM ranks all S&P 500 stocks daily
- Top 20% by ML score, gated by market regime filter
- Avoids earnings, gap-ups, and high-correlation clusters

**Execution:**
- Hold period: 5 trading days
- Stop loss: 2.0x ATR from entry
- Profit target: 2.5x ATR from entry
- Early exit: positions down >1.5% after 3 days
- Max 6 concurrent positions, vol-scaled sizing
- Max 3 positions per sector

---

## Setup (Windows)

### Prerequisites
- Python 3.11 or higher — [download here](https://python.org)
- Internet connection (for data download)

### Install
```
Double-click setup.bat
```
Or manually:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

Activate environment first:
```bash
venv\Scripts\activate
```

### Step 1: Download Data (run once, takes ~10-20 min)
```bash
python main.py pipeline
```

### Step 2: Train ML Model
```bash
python main.py train
```

### Step 3: Run Backtest
```bash
python main.py backtest_ml
```

### Step 4: Enrich Data (sector + earnings)
```bash
python main.py enrich
python main.py train_v2
python main.py backtest_ml_v2
```

### Step 5: Run Daily Briefing
```bash
python main.py daily
```

### Other Commands
```bash
python main.py signals          # Today's top signals
python main.py walkforward      # Walk-forward validation
python main.py vix              # Current VIX context
python main.py risk             # Portfolio risk status
python main.py history          # Closed trade history
python main.py retrain          # Force model retrain
python main.py scheduler        # Auto-run daily at 9am
```

---

## Project Structure

```
swing_trader/
├── config/
│   └── config.yaml              ← All parameters (v8, with risk controls)
├── data/
│   ├── universe.py              ← S&P 500 tickers (deduped, no ETFs)
│   ├── pipeline.py              ← Downloads + features (gap, BB, OBV)
│   ├── sectors.py               ← Sector momentum features
│   └── earnings.py              ← Earnings calendar features
├── signals/
│   └── signals.py               ← Signal generation (v6, smoothed regime)
├── backtest/
│   ├── backtest.py              ← Backtester (v5, vol-scaled, DD enforcement)
│   ├── backtest_inverse.py      ← ML vs ML+SH hedge comparison
│   └── backtest_dynamic_regime.py ← Fixed vs dynamic VIX threshold
├── models/
│   ├── train.py                 ← LightGBM training (better hyperparams)
│   └── predict.py               ← ML scoring for live + backtest
├── live/
│   ├── daily.py                 ← Daily runner (with risk checks)
│   ├── positions.py             ← Position tracker (SQLite)
│   ├── alerts.py                ← SMS alerts via Gmail
│   ├── regime.py                ← Dynamic VIX threshold
│   ├── retrain.py               ← Monthly auto-retrain
│   └── scheduler.py             ← Windows Task Scheduler wrapper
├── utils/
│   └── risk.py                  ← Portfolio risk management (NEW)
├── main.py                      ← Entry point (all commands)
├── requirements.txt
└── setup.bat
```

---

## Key Differences from v1

| Area | v1 | v2 |
|---|---|---|
| Position sizing | Flat 10% | Vol-scaled (2% risk budget) |
| Portfolio stop | Config only | Enforced with cooldown |
| Sector limits | None | Max 3 per sector |
| Position storage | JSON file | SQLite database |
| Regime filter | Instant flip | 2-day smoothing |
| Signal filter | 6 factors | 9 factors + earnings |
| ML regularization | Light | Strong (L1=L2=0.3, depth=6) |
| Pipeline features | 15 | 19 (gap, BB, range, OBV) |
| Backtest stats | 12 metrics | 18 metrics (Sortino, skew, etc) |

---

## Important Disclaimers

This is educational software. Past backtest performance does not guarantee future returns.
Always paper trade before using real capital. You are responsible for your own trading decisions.
