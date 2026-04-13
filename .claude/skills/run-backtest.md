# Skill: Run a Backtest

When the user asks to backtest a strategy, asset, or parameter change:

## Quick Single-Asset Backtest
```python
from backtester.data_loader import load_backtest_data
from backtester.engine import BacktestEngine

df = load_backtest_data('BTCUSDT', interval='15m', days=180)
engine = BacktestEngine()
result = engine.run_backtest(df, symbol='BTCUSDT')
print(f"Trades: {result['total_trades']}")
print(f"Win Rate: {result['win_rate']:.1f}%")
print(f"Sharpe: {result['sharpe']:.2f}")
print(f"Max DD: {result['max_drawdown_pct']:.2f}%")
print(f"Profit Factor: {result['profit_factor']:.2f}")
print(f"Net PnL: {result['net_pnl']:.2f}")
```

## Walk-Forward Validation
```python
result = engine.run_walk_forward(df, symbol='BTCUSDT', train_days=90, test_days=30)
# Returns out-of-sample metrics only (no overfitting)
```

## Monte Carlo Simulation
```python
from backtester.monte_carlo import run_monte_carlo
mc = run_monte_carlo(result['trades'], n_runs=10000)
print(f"Prob of profit: {mc['prob_profit_pct']:.1f}%")
print(f"Worst-case DD (95th): {mc['max_dd_p95']:.1f}%")
```

## Multi-Asset Backtest
```python
from backtester.data_loader import load_multi_symbol_data
data = load_multi_symbol_data(['BTCUSDT', 'ETHUSDT', 'SOLUSDT'], interval='15m', days=365)
for symbol, df in data.items():
    result = engine.run_backtest(df, symbol=symbol)
    print(f"{symbol}: PF={result['profit_factor']:.2f} Sharpe={result['sharpe']:.2f}")
```

## Target Metrics (professional grade)
- Profit Factor > 1.6
- Sharpe > 1.8
- Max Drawdown < 12%
- Win Rate > 45% (with 2:1 R:R)
- Total Trades > 100 (statistical significance)

## Important
- Backtest uses exact Binance fees (taker 0.04%, maker 0.02%)
- Includes slippage model (configurable via `SLIPPAGE_TICKS_MIN/MAX` in config)
- Includes funding rate drag (fetched from Binance history)
- Uses free Binance data only (data.binance.vision bulk downloads + REST API)
