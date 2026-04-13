# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Wall Street Oracle v5.0 — an autonomous trading bot for Binance USDT perpetual futures (demo-fapi by default). Hybrid rule-based + ML architecture. Zero human interaction once running. Scans every 15 minutes synced to Binance server time.

## Commands

```bash
# Setup
cp .env.example .env           # Then fill in API keys
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Run
python main.py                  # Bot (foreground, blocks)
python dashboard.py             # Dashboard on http://localhost:5000 (separate terminal)

# Docker
docker-compose up -d            # Bot + Dashboard containers
curl http://localhost:8080/health
```

No test suite, Makefile, or linter configuration exists yet.

## Architecture

**Signal pipeline (every 15 minutes):**

```
main.py orchestrator
  → strategy/rules.py      Multi-timeframe rules (4H trend + 1H momentum + 15m entry)
  → ml_model/predictor.py   LightGBM filter (must agree >65% probability)
  → risk/manager.py         ATR position sizing, drawdown kill-switch, correlation check
  → execution/engine.py     Direct signed Binance API orders (SL/TP/trailing)
```

**Data flow:**

```
data/binance_ws.py (3 WebSocket streams: ticker, aggTrade, user data)
data/binance_rest.py (REST klines, funding rates, bulk downloads from data.binance.vision)
  → indicators/technical.py (pure functions: EMA, RSI, MACD, BB, ATR, ADX, VWAP, SMC)
  → ml_model/features.py (40+ features engineered from OHLCV)
```

**Background threads in main.py:**
- Tick collector (30s) → saves to Supabase via `data/storage.py`
- Position monitor (10s) → moves SL to break-even after TP1 hit
- Nightly ML retrain (2 AM UTC) → walk-forward validation before going live
- Health check HTTP server on port 8080

## Key Design Decisions

- **Dual-signal approval**: Both rules engine (confidence >= MIN_CONFIDENCE) AND ML classifier (probability >= ML_MIN_PROBABILITY) must agree before trading.
- **Three kill-switches**: Daily loss limit (-2%), equity drawdown (-8% from peak), circuit breaker (extreme ATR).
- **Demo-first**: `USE_LIVE=false` by default. All API calls go to `demo-fapi.binance.com`.
- **Completed candles only**: MACD crossovers use `iloc[-2]`/`iloc[-3]` (not current forming candle) to avoid phantom crosses.
- **ATR-based sizing**: Risk exactly `RISK_PER_TRADE_PCT` (0.5%) of equity per trade. SL distance = ATR * 1.6.
- **Free stack only**: No paid data providers. Uses Binance free REST/WebSocket + data.binance.vision bulk downloads + Supabase free tier.

## Module Responsibilities

| Module | Role |
|---|---|
| `config.py` | All settings from `.env`. Every magic number lives here. |
| `utils/helpers.py` | `signed_request()` — HMAC-SHA256 signed Binance API calls with synced timestamps |
| `data/storage.py` | Supabase connection pool. 6 tables: market_ticks, signals_log, equity_curve, trade_journal, ml_feature_importance, ohlcv_1m |
| `strategy/regime.py` | GMM with 4 regimes (Trending/Ranging/High-Volatility/Low-Liquidity). Fitted on BTC 1H data. |
| `ml_model/trainer.py` | LightGBM binary classifier. Saves model as pickle + metadata JSON to `ml_models/`. Feature importance logged to DB. |
| `backtester/engine.py` | Walk-forward backtester with realistic slippage (0.02%), exact Binance fees (maker 0.02% / taker 0.04%), funding rate drag. |
| `backtester/monte_carlo.py` | 10,000-run shuffle simulation for tail risk. |
| `monitoring/performance.py` | Rolling Sharpe, drawdown tracking, `/health` and `/metrics` HTTP endpoints. |

## Configuration

All config in `config.py`, loaded from `.env`. Key groups:
- **Trading**: `MAX_OPEN_POSITIONS`, `MAX_DAILY_TRADES`, `TRADE_COOLDOWN_MINUTES`, `LEVERAGE`
- **Risk**: `RISK_PER_TRADE_PCT`, `DAILY_LOSS_LIMIT_PCT`, `MAX_DRAWDOWN_PCT`, `MAX_CORRELATED_EXPOSURE_PCT`
- **Strategy**: `MIN_CONFIDENCE`, `ML_MIN_PROBABILITY`, `ADX_MIN_THRESHOLD`, `ENABLE_ML_LAYER`, `ENABLE_TRAILING_STOP`
- **ML**: `ML_RETRAIN_HOUR`, `ML_LOOKBACK_DAYS`, `ML_WALK_FORWARD_TRAIN/TEST`
- **Correlation groups**: Pre-defined in `config.py` as `CORRELATION_GROUPS` dict (BTC_GROUP, ETH_GROUP, DEFI_GROUP, L1_GROUP, MEME_GROUP)

## Important Patterns

- `signed_request(method, path, params)` in `utils/helpers.py` handles all authenticated Binance calls with automatic time offset correction.
- `from __future__ import annotations` is used in `data/binance_ws.py` and `strategy/rules.py` for Python 3.9 compatibility with union type hints.
- The backtester's `_generate_signals_and_simulate()` mirrors the exact logic in `strategy/rules.py` — changes to signal logic must be reflected in both.
- `data/binance_rest.py` has built-in rate limiting (50ms between requests) to stay within Binance's 1200 req/min limit.
