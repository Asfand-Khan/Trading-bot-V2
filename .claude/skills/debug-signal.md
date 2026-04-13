# Skill: Debug Why a Signal Was Skipped or Taken

When the user asks why a specific asset didn't generate a signal, or wants to understand a signal:

## Diagnostic Steps

1. **Check the audit log** — every signal (taken or skipped) is logged with reason:
   ```bash
   # Check CSV log
   tail -50 signals_log.csv | grep ASSETNAME
   
   # Check terminal output for [SKIP] or [AUDIT] lines
   # Format: [SKIP] BTCUSDT: Reason here
   ```

2. **Common skip reasons and what they mean**:
   - `"ADX X.X < 25.0"` — Market not trending. The 1H ADX is below threshold. Wait for directional move.
   - `"No 15m MACD+SMC synergy"` — No fresh MACD crossover OR SMC zone not aligned. This is the most common skip.
   - `"Volume X% < 80% threshold"` — Last completed candle had low volume. Fakeout protection.
   - `"Below VWAP / Above VWAP"` — Price wrong side of VWAP relative to signal direction. Trap protection.
   - `"Confidence X < MIN_CONFIDENCE"` — Signal passed basic filters but total score too low.
   - `"ML rejected: X%"` — Rules approved but LightGBM classifier disagreed.
   - `"Daily trade limit"` / `"Cooldown"` / `"Max positions"` — Risk manager blocked it.
   - `"PAUSED: ..."` — Bot is in risk pause (daily loss or drawdown kill-switch hit).
   - `"Circuit breaker: ATR X% too high"` — Extreme volatility detected.

3. **To see all feature values for a specific scan**, search the terminal output for:
   ```
   [AUDIT] BTCUSDT | Action=EXECUTED | Reason=... | Features={...}
   ```

4. **To manually check an asset right now**:
   ```python
   from strategy.rules import analyze_asset
   result = analyze_asset('BTCUSDT')
   print(result)  # None = skipped, dict = signal generated
   ```

## Key Files
- `strategy/rules.py` → `analyze_asset()` — all signal logic lives here
- `utils/logging_utils.py` → `log_decision_audit()` — writes audit trail
- `signals_log.csv` — persistent log of all signals
