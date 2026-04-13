# Skill: Tune Strategy Parameters

When the user wants to adjust strategy parameters, risk settings, or ML thresholds:

## Parameter Categories

### Risk Parameters (in `.env`)
- `RISK_PER_TRADE_PCT` — % of equity risked per trade (default 0.5). Range: 0.1-2.0
- `DAILY_LOSS_LIMIT_PCT` — Daily loss limit before 24h pause (default 2.0). Range: 1.0-5.0
- `MAX_DRAWDOWN_PCT` — Equity peak drawdown kill-switch (default 8.0). Range: 5.0-15.0
- `MAX_OPEN_POSITIONS` — Concurrent positions allowed (default 3). Range: 1-5
- `LEVERAGE` — Leverage multiplier (default 10). Range: 1-20

### Signal Parameters (in `.env`)
- `MIN_CONFIDENCE` — Minimum rule-based score to trade (default 70). Range: 60-90
- `ML_MIN_PROBABILITY` — ML classifier minimum confidence (default 0.65). Range: 0.55-0.80
- `ADX_MIN_THRESHOLD` — Minimum ADX for trend strength (default 25). Range: 20-35
- `VOLUME_CONFIRMATION_RATIO` — Volume vs SMA ratio (default 0.8). Range: 0.5-1.2

### Entry/Exit Parameters (in `config.py`)
- `ATR_STOP_MULTIPLIER` — SL distance in ATR units (default 1.6). Range: 1.0-2.5
- `TP1_RR_RATIO` — TP1 risk:reward ratio (default 2.0). Range: 1.5-3.0
- `TP2_RR_RATIO` — TP2 risk:reward ratio (default 3.5). Range: 2.5-5.0

## Process
1. Change the parameter in `.env` (runtime) or `config.py` (requires restart)
2. After tuning, run a backtest to validate: the backtester in `backtester/engine.py` uses the same config values
3. Monitor the first 10-20 trades after any parameter change
4. If `PERFORMANCE_DEVIATION_THRESHOLD` (15%) triggers, the bot auto-pauses — review the changes

## Important
- Never change `ATR_STOP_MULTIPLIER` below 1.0 (stops too tight = constant stop-outs)
- Never set `ML_MIN_PROBABILITY` below 0.55 (random noise territory)
- After changing strategy params, the backtester must be re-run to validate
- Changes to `.env` take effect on next bot restart
