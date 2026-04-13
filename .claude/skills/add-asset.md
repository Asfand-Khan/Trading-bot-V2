# Skill: Add New Trading Asset

When the user wants to add a new asset to the watchlist:

1. **Add to `config.py`** → `DEFAULT_WATCHLIST` list. Must be a valid Binance USDT-M Perpetual symbol (e.g., `TRUMPUSDT`, `WIFUSDT`).
2. **Add correlation group** → `CORRELATION_GROUPS` in `config.py`. Assign the asset to the correct group or create a new one. This prevents over-exposure to correlated assets.
3. **Verify on Binance**: Check the symbol exists on demo-fapi by running:
   ```python
   from data.binance_rest import fetch_klines
   df = fetch_klines('NEWSYMBOLUSDT', '15m', limit=10)
   print(len(df))  # Should return 10 rows
   ```
4. **Backtest first**: The dynamic watchlist filter in `main.py` → `run_watchlist_filter()` will automatically backtest the new asset and drop it if unprofitable. No manual intervention needed.

Notes:
- Keep watchlist under 30 assets to stay within API rate limits
- The WebSocket ticker stream in `data/binance_ws.py` auto-subscribes to all watchlist assets
- AggTrade stream only subscribes to first 10 assets (for order flow data)
