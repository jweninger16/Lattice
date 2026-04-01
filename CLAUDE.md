# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Systematic swing trading system for S&P 500 equities. ~$20K account. LightGBM ML model + rule-based signals, walk-forward validated backtesting, live trading with SQLite position tracking. Windows-first (Python 3.11+).

## Commands

All commands run from the project root with the venv activated.

```bash
# Setup
setup.bat                            # First-time Windows setup (creates venv, installs deps)
venv\Scripts\activate                # Activate venv in subsequent sessions

# Data pipeline (run first, ~10-20 min)
python main.py pipeline              # Download OHLCV + build 19 technical features
python main.py enrich                # Add sector momentum + earnings features (5-10 min)

# Training
python main.py train                 # Train LightGBM (walk-forward, base features)
python main.py train_v2              # Train with enriched features (requires enrich first)

# Backtesting
python main.py backtest              # Rule-based signal backtest
python main.py backtest_ml           # ML-scored backtest
python main.py backtest_ml_v2        # ML + sector/earnings backtest
python main.py walkforward           # Walk-forward validation (25 folds)

# Live trading
python main.py daily                 # Morning briefing + signals
python main.py monitor               # Intraday stop/target check loop (30-min)
python main.py monitor --once        # Single check and exit
python main.py buy TICK PRICE SHARES # Log a manual buy
python main.py sell TICK PRICE       # Log a manual sell
python main.py portfolio             # Full portfolio status
python main.py risk                  # Portfolio risk metrics
python main.py history               # Closed trade history
python main.py vix                   # Current VIX context

# Automation
python main.py scheduler             # Run daily briefing at 9:00 AM (blocking)
python main.py retrain               # Force model retrain (~20 min)
python main.py retrain --force       # Retrain even if recent model exists
```

There is no test suite. No linter is configured.

## Architecture

### Data Flow

```
yfinance OHLCV → data/pipeline.py (19 features) → data/processed/*.parquet
                                                  ↓
                              data/sectors.py + data/earnings.py → enriched parquet
                                                  ↓
                              models/train.py (walk-forward LightGBM) → models/lgbm_model.pkl
                                                  ↓
                              models/predict.py (scoring) → signal DataFrame with ml_signal column
                                                  ↓
                              backtest/backtest.py (simulation) OR live/daily.py (live signals)
```

### Key Design Decisions

- **Two signal paths**: Rule-based (`signals/signals.py`) and ML-based (`models/predict.py`). Both can be backtested independently. ML signals are used for live trading.
- **Walk-forward training**: 24-month rolling train windows, 3-month test periods, 5-day embargo. Model is only deployed if AUC > 0.55. Configured in `config.yaml` under `validation:`.
- **Market regime gating**: Signals are filtered by VIX-based regime (bull/bear). Regime requires 2 consecutive days to flip (smoothing to prevent whipsaws). See `signals/signals.py:compute_market_regime` and `live/regime.py`.
- **Vol-scaled position sizing**: Position size is inversely proportional to ATR, not flat allocation. Configured via `backtest.vol_target_per_position` and bounded by `min_position_size`/`max_position_size` in config.
- **Risk enforcement in both backtest and live**: Portfolio drawdown halt (with cooldown), daily loss circuit breaker, sector concentration limits, and correlation checks are applied in `backtest/backtest.py` and checked in `live/daily.py` via `utils/risk.py`.
- **SQLite position tracking** (`live/positions.db`): Tables for positions, portfolio state snapshots, and cash transactions. All live state lives here — not in memory or flat files.
- **Lazy imports in main.py**: Each CLI command imports its dependencies at call time to keep startup fast. Follow this pattern when adding commands.

### Configuration

`config/config.yaml` is the single source for all tunable parameters: universe filters, feature lookbacks, backtest settings (hold days, ATR multiples, sizing bounds), validation windows, and risk limits. Loaded via `main.py:load_config()`.

Alert credentials (Discord webhook, Gmail) go in `.env` (see `.env.example`). Loaded by `live/alerts.py` via `python-dotenv`.

### Enrichment Pipeline

Base pipeline (`pipeline`) produces `price_features.parquet`. The `enrich` command layers on sector ETF momentum (11 sectors + SPY/QQQ via `data/sectors.py`) and earnings proximity features (`data/earnings.py`), saving to `price_features_enriched.parquet`. The `train_v2`/`backtest_ml_v2` commands use the enriched data; base `train`/`backtest_ml` use the non-enriched version.

### Model Versioning

- `models/lgbm_model.pkl` — base model (trained by `train`)
- `models/lgbm_model_v2.pkl` — enriched model (trained by `train_v2`)
- `backtest_ml_v2` temporarily swaps v2 into the v1 path for scoring, then restores v1 after

### research/ Directory

Contains 13+ experimental backtest scripts (gap scanner, ORB, entry methods, etc.). These are one-off explorations with duplicated logic — not part of the core system.
