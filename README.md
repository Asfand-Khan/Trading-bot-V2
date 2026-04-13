# Wall Street Oracle v5.0 — Elite Autonomous Scalper

Professional-grade autonomous trading bot for Binance USDT perpetual futures.
Hybrid Rule + ML architecture with dynamic risk management and walk-forward validation.

## Architecture

```
Tbot/
├── main.py                    # Master orchestrator
├── config.py                  # Central configuration (all from .env)
├── dashboard.py               # Flask live performance dashboard
├── utils/
│   ├── logging_utils.py       # Structured logging + audit trail
│   ├── email_alerts.py        # Email notifications
│   └── helpers.py             # Signed requests, time sync, utilities
├── data/
│   ├── binance_ws.py          # WebSocket: ticker, aggTrade, user data streams
│   ├── binance_rest.py        # REST API: klines, funding, bulk downloads
│   ├── storage.py             # Supabase PostgreSQL storage layer
│   └── data_quality.py        # Gap detection, outlier removal, validation
├── indicators/
│   └── technical.py           # All indicators: EMA, RSI, MACD, BB, ATR, ADX, VWAP, SMC
├── strategy/
│   ├── rules.py               # Multi-timeframe rule-based signal engine
│   └── regime.py              # GMM-based market regime detection
├── ml_model/
│   ├── features.py            # Feature engineering (40+ features)
│   ├── trainer.py             # LightGBM nightly retrainer + walk-forward
│   └── predictor.py           # Live ML prediction wrapper
├── risk/
│   └── manager.py             # Dynamic position sizing, drawdown, correlation
├── execution/
│   └── engine.py              # Order execution + position monitor
├── backtester/
│   ├── data_loader.py         # Historical data loader
│   ├── engine.py              # Walk-forward backtester with realistic costs
│   └── monte_carlo.py         # Monte Carlo simulation (10,000 runs)
├── monitoring/
│   └── performance.py         # Live metrics, health check, equity tracking
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Quick Start

### 1. Setup Environment
```bash
cp .env.example .env
# Edit .env with your Binance Demo API keys and Supabase URL
```

### 2. Install Dependencies
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 3. Run the Bot
```bash
python main.py
```

### 4. Run the Dashboard (separate terminal)
```bash
python dashboard.py
# Open http://127.0.0.1:5000
```

### 5. Docker (optional)
```bash
docker-compose up -d
# Bot runs on port 8080 (health), Dashboard on port 5000
```

## How It Works

### Signal Flow
1. Every 15 minutes (synced to Binance server time), the bot scans the watchlist
2. **Rules Engine**: Multi-timeframe analysis (4H trend + 1H momentum + 15m entry)
3. **ML Filter**: LightGBM classifier must agree with >65% probability
4. **Risk Check**: Position sizing, drawdown limits, correlation check, circuit breaker
5. **Execution**: Direct signed API with SL/TP/trailing stop placement

### ML Retraining
- Runs nightly at 2 AM UTC (configurable via `ML_RETRAIN_HOUR`)
- Trains on 180 days of 15-minute data across top 10 symbols
- LightGBM binary classifier predicting next-candle direction
- Feature importance logged to database
- Walk-forward validation before model goes live

### Risk Management
- **Position Sizing**: Risk exactly 0.5% of equity per trade (ATR-based)
- **Daily Loss Limit**: Pauses 24h if daily loss exceeds -2%
- **Drawdown Kill-Switch**: Pauses if drawdown from peak exceeds -8%
- **Portfolio Correlation**: Never >30% exposure to correlated asset group
- **Circuit Breaker**: Pauses on extreme volatility (>3% ATR on 15m)

### Monitoring
- Health check: `http://localhost:8080/health`
- Metrics API: `http://localhost:8080/metrics`
- Dashboard: `http://localhost:5000`
- Daily P&L email report at midnight UTC
- All signals (taken and skipped) logged with full feature values

## Cost
**$0/month**. Uses only:
- Binance free public API + WebSocket streams
- Supabase free tier (500MB)
- Free open-source Python libraries
