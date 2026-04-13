# Skill: Add New Technical Indicator

When the user asks to add a new technical indicator:

1. **Add the pure function** to `indicators/technical.py` — must be a stateless function taking pandas Series/DataFrame and returning Series.
2. **Add it to `compute_all_indicators()`** in the same file so it's auto-computed on every OHLCV DataFrame.
3. **Add the feature** to `ml_model/features.py`:
   - Add the derived feature in `engineer_features()`
   - Add the column name to `FEATURE_COLUMNS` list
4. **If it affects signal logic**, update `strategy/rules.py` → `analyze_asset()` and mirror the same logic in `backtester/engine.py` → `_generate_signals_and_simulate()`.
5. **Retrain ML**: The new feature will be picked up automatically on next nightly retrain. For immediate use, run `ml_model/trainer.py` manually.
6. **Test**: Run `python main.py`, verify the indicator appears in scan logs.

Critical rules:
- Indicators must handle NaN gracefully (use `.fillna(0)` or `min_periods`)
- Never use `datetime.now()` — use `get_synced_now()` from `utils/helpers.py`
- Keep backtester in sync with rules engine — both must use the same signal logic
