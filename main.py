"""
main.py — Wall Street Oracle v5.0 Elite Autonomous Scalper
Master orchestrator: ties together all modules.
Zero human interaction once running.
"""

import os
import sys
import time
import logging
import threading
from datetime import datetime, date, timedelta, timezone

# ====================== SETUP LOGGING FIRST ======================
from utils.logging_utils import setup_logging, logger, log_signal_csv, log_skip_csv, log_decision_audit

setup_logging()

# ====================== IMPORTS ======================
from config import (
    DEFAULT_WATCHLIST, BINANCE_TEST_API, BINANCE_TEST_SECRET,
    SCAN_INTERVAL_MINUTES, HIGH_IMPACT_DATES, ENABLE_ML_LAYER,
    ML_RETRAIN_HOUR, BASE_URL
)

from utils.helpers import signed_request, is_api_error, sync_binance_time, get_synced_timestamp, get_synced_now
from utils.email_alerts import send_signal_email, send_daily_report, send_alert

from data.storage import init_db, save_signal, save_ticks_batch
from data.binance_ws import (
    start_ticker_stream, start_agg_trade_stream,
    start_user_data_stream, get_live_price, reset_order_flow
)
from data.binance_rest import fetch_klines
from data.data_quality import check_staleness

from strategy.rules import analyze_asset
from strategy.regime import RegimeDetector

from ml_model.predictor import ml_filter, retrain_if_needed, get_trainer
from ml_model.features import extract_live_features

from risk.manager import RiskManager
from execution.engine import ExecutionEngine

from monitoring.performance import (
    PerformanceTracker, set_global_tracker, start_health_server
)

from backtester.engine import BacktestEngine
from backtester.data_loader import load_backtest_data


# ====================== GLOBAL STATE ======================
WATCHLIST = list(DEFAULT_WATCHLIST)
risk_manager = RiskManager()
execution_engine = ExecutionEngine(risk_manager)
performance_tracker = PerformanceTracker(risk_manager)
regime_detector = RegimeDetector()
set_global_tracker(performance_tracker)

# Track the last scan minute to prevent duplicate scans
_last_scan_minute = -1
_last_retrain_date = None
_last_daily_report_date = None


# ====================== STARTUP VERIFICATION ======================
def verify_connection():
    """Verify API keys, sync time, fetch balance."""
    if not BINANCE_TEST_API or not BINANCE_TEST_SECRET:
        logger.error("FATAL: API keys not configured in .env")
        sys.exit(1)

    # Sync time
    sync_binance_time()

    # Verify keys
    resp = signed_request('GET', '/fapi/v2/balance')
    if is_api_error(resp):
        logger.error(f"FATAL: API verification failed: {resp}")
        sys.exit(1)

    usdt_balance = 0.0
    if isinstance(resp, list):
        usdt_balance = next((float(a['balance']) for a in resp if a['asset'] == 'USDT'), 0)

    risk_manager.update_equity(usdt_balance)
    logger.info(f"Connection verified! Equity: ${usdt_balance:,.2f}")
    return usdt_balance


# ====================== DYNAMIC WATCHLIST BACKTESTER ======================
def run_watchlist_filter():
    """Backtest watchlist and drop underperformers (like v4.0 but using Binance data)."""
    global WATCHLIST
    logger.info("Running dynamic watchlist filter (30-day backtest)...")

    backtester = BacktestEngine()
    new_watchlist = []

    for ticker in DEFAULT_WATCHLIST:
        try:
            df = fetch_klines(ticker, '15m', limit=1500)  # ~15 days
            if len(df) < 200:
                logger.info(f"  {ticker}: Insufficient data. Retaining.")
                new_watchlist.append(ticker)
                continue

            result = backtester.run_backtest(df, ticker)
            total = result.get('total_trades', 0)
            net_pnl = result.get('net_pnl', 0)
            win_rate = result.get('win_rate', 0)

            if total < 3:
                logger.info(f"  {ticker}: Only {total} trades. Retaining.")
                new_watchlist.append(ticker)
            elif net_pnl > 0:
                logger.info(f"  {ticker}: PASSED | PnL={net_pnl:.2f} | WR={win_rate:.1f}% | Trades={total}")
                new_watchlist.append(ticker)
            else:
                logger.warning(f"  {ticker}: DROPPED | PnL={net_pnl:.2f} | WR={win_rate:.1f}% | Trades={total}")

            time.sleep(0.2)  # Rate limit
        except Exception as e:
            logger.error(f"  {ticker}: Error — {e}. Retaining.")
            new_watchlist.append(ticker)

    WATCHLIST = new_watchlist
    logger.info(f"Watchlist finalized: {len(WATCHLIST)} assets")


# ====================== 30-SECOND TICK COLLECTOR ======================
def tick_collector_loop():
    """Collect ticks every 30s and save to DB."""
    logger.info("Tick collector started (30s intervals)")
    while True:
        try:
            timestamp = get_synced_now()
            ticks = []
            for ticker in WATCHLIST:
                price = get_live_price(ticker)
                if price is not None:
                    ticks.append((str(timestamp), ticker, float(price)))

            if ticks:
                save_ticks_batch(ticks)
                if len(ticks) <= 5:
                    debug = " | ".join(f"{t}: ${p:.2f}" for _, t, p in ticks)
                    logger.debug(f"[TICK] {debug}")
        except Exception as e:
            logger.error(f"Tick collector error: {e}")

        time.sleep(30)


# ====================== MAIN SCAN LOOP ======================
def run_scan():
    """Run full analysis scan across watchlist."""
    logger.info(f"{'='*60}")
    logger.info(f"SCAN STARTED at {get_synced_now().strftime('%Y-%m-%d %H:%M:%S')} UTC (Binance-synced)")
    logger.info(f"{'='*60}")

    # Check high-impact news dates (use Binance-synced date)
    today = get_synced_now().strftime("%Y-%m-%d")
    if today in HIGH_IMPACT_DATES:
        logger.info("HIGH IMPACT NEWS DAY — skipping all signals")
        return

    # Detect current regime using BTC as market proxy
    current_regime = 'UNKNOWN'
    try:
        btc_data = fetch_klines('BTCUSDT', '1h', limit=600)
        if len(btc_data) > 100:
            from indicators.technical import compute_all_indicators
            btc_data = compute_all_indicators(btc_data)
            regime_id = regime_detector.predict(btc_data)
            current_regime = regime_detector.get_regime_name(regime_id)
            logger.info(f"Market Regime: {current_regime}")
    except Exception as e:
        logger.warning(f"Regime detection failed: {e}")

    # Refresh equity
    risk_manager.fetch_account_equity()

    signals_found = 0
    for ticker in WATCHLIST:
        try:
            # Rule-based analysis
            signal_data = analyze_asset(ticker)

            if signal_data is None:
                continue

            # Attach regime
            signal_data['regime'] = current_regime

            # ML filter
            if ENABLE_ML_LAYER:
                signal_data = ml_filter(signal_data)
                if not signal_data.get('ml_approved', True):
                    log_decision_audit(
                        ticker, 'ML_REJECTED',
                        f"ML prob {signal_data.get('ml_probability', 0):.1%}",
                        signal_data.get('features', {})
                    )
                    log_skip_csv(ticker, f"ML rejected: {signal_data.get('ml_probability', 0):.1%}")
                    continue

            # Risk pre-trade checks
            approved, reason = risk_manager.pre_trade_check(ticker, signal_data)
            if not approved:
                log_decision_audit(ticker, 'RISK_REJECTED', reason,
                                   signal_data.get('features', {}))
                log_skip_csv(ticker, f"Risk rejected: {reason}")
                continue

            # Calculate position size
            position_size = risk_manager.calculate_position_size(
                entry_price=signal_data['price'],
                stop_loss=signal_data['stop_loss_raw'],
                atr=signal_data['atr'],
                symbol=ticker
            )
            signal_data['position_size'] = f"${position_size['notional']:,.2f} ({position_size['quantity']:.6f})"

            if position_size['quantity'] <= 0:
                log_skip_csv(ticker, "Position size too small")
                continue

            # Execute trade
            exec_result = execution_engine.execute_trade(signal_data, position_size)

            if exec_result.get('success'):
                # Log everywhere
                signal_data['status'] = 'EXECUTED'
                log_signal_csv(signal_data)
                save_signal(signal_data)
                send_signal_email(signal_data)
                risk_manager.record_trade(ticker)

                log_decision_audit(
                    ticker, 'EXECUTED',
                    signal_data.get('why', ''),
                    signal_data.get('features', {})
                )

                signals_found += 1
                logger.info(f"SIGNAL EXECUTED: {ticker} {signal_data['signal']} "
                             f"| Confidence: {signal_data['confidence']}% "
                             f"| ML: {signal_data.get('ml_probability', 'N/A')}")
            else:
                log_skip_csv(ticker, f"Execution failed: {exec_result.get('reason', 'unknown')}")

        except Exception as e:
            import traceback
            logger.error(f"Scan error for {ticker}: {e}\n{traceback.format_exc()}")

    # Save performance snapshot
    performance_tracker.save_snapshot(current_regime)
    logger.info(f"Scan complete: {signals_found} signals executed from {len(WATCHLIST)} assets")


# ====================== NIGHTLY TASKS ======================
def nightly_tasks():
    """Run nightly: ML retrain, regime refit, daily report."""
    global _last_retrain_date, _last_daily_report_date

    while True:
        try:
            now = get_synced_now()  # Binance-synced time

            # ML Retrain at configured hour
            if (now.hour == ML_RETRAIN_HOUR and
                    _last_retrain_date != now.date()):
                _last_retrain_date = now.date()
                logger.info("=== NIGHTLY ML RETRAIN ===")
                try:
                    metrics = retrain_if_needed()
                    if metrics:
                        logger.info(f"Retrain complete: {metrics}")
                except Exception as e:
                    logger.error(f"ML retrain error: {e}")

                # Refit regime detector
                try:
                    btc_data = fetch_klines('BTCUSDT', '1h', limit=1000)
                    if len(btc_data) > 200:
                        from indicators.technical import compute_all_indicators
                        btc_data = compute_all_indicators(btc_data)
                        regime_detector.fit(btc_data)
                except Exception as e:
                    logger.error(f"Regime refit error: {e}")

            # Daily report at midnight UTC
            if (now.hour == 0 and now.minute < 5 and
                    _last_daily_report_date != now.date()):
                _last_daily_report_date = now.date()
                try:
                    report = performance_tracker.generate_daily_report()
                    send_daily_report(report)
                except Exception as e:
                    logger.error(f"Daily report error: {e}")

        except Exception as e:
            logger.error(f"Nightly tasks error: {e}")

        time.sleep(60)  # Check every minute


# ====================== MAIN ENTRY POINT ======================
def main():
    global _last_scan_minute

    logger.info("=" * 70)
    logger.info("  WALL STREET ORACLE v5.0 — Elite Autonomous Scalper")
    logger.info("  Hybrid Rule + ML | Dynamic Risk | Walk-Forward Validated")
    logger.info("=" * 70)

    # 1. Verify connection
    equity = verify_connection()

    # 2. Initialize database
    init_db()

    # 3. Start WebSocket streams
    logger.info("Starting WebSocket streams...")
    ws_thread = threading.Thread(
        target=start_ticker_stream, args=(WATCHLIST,), daemon=True
    )
    ws_thread.start()
    time.sleep(2)  # Let WS connect

    # Aggregated trades for order flow
    start_agg_trade_stream(WATCHLIST[:10])  # Top 10 for flow data

    # User Data Stream for real-time fills
    start_user_data_stream()

    # 4. Start background threads
    # Tick collector
    threading.Thread(target=tick_collector_loop, daemon=True).start()

    # Position monitor
    threading.Thread(
        target=execution_engine.position_monitor_loop, daemon=True
    ).start()

    # Health check server
    start_health_server()

    # Nightly tasks (ML retrain, daily report)
    threading.Thread(target=nightly_tasks, daemon=True).start()

    # 5. Initial regime fit
    logger.info("Fitting regime detector...")
    try:
        btc_data = fetch_klines('BTCUSDT', '1h', limit=600)
        if len(btc_data) > 200:
            from indicators.technical import compute_all_indicators
            btc_data = compute_all_indicators(btc_data)
            regime_detector.fit(btc_data)
    except Exception as e:
        logger.warning(f"Initial regime fit failed: {e}")

    # 6. Run watchlist filter
    run_watchlist_filter()

    # 7. Initial scan
    logger.info("Running initial scan...")
    threading.Thread(target=run_scan).start()

    # 8. Main loop — scan on 15-minute Binance time boundaries
    logger.info("Entering main loop (15-min Binance-synced scans)...")
    while True:
        try:
            synced_ts = get_synced_timestamp()
            synced_now = datetime.fromtimestamp(synced_ts / 1000.0)
            current_minute = synced_now.minute
            current_second = synced_now.second

            is_boundary = (current_minute % SCAN_INTERVAL_MINUTES == 0)

            if is_boundary and current_second >= 2 and current_minute != _last_scan_minute:
                logger.info(f"Binance time triggered scan ({current_minute:02d}:{current_second:02d})")
                _last_scan_minute = current_minute
                threading.Thread(target=run_scan).start()

            # Periodic time re-sync every 15 min (at :07 to avoid scan overlap)
            if current_minute % 15 == 7 and current_second < 2:
                sync_binance_time()

            # Reset order flow counters every 15 min
            if is_boundary and current_second < 2:
                for ticker in WATCHLIST:
                    reset_order_flow(ticker)

        except Exception as e:
            logger.error(f"Main loop error: {e}")

        time.sleep(1)


if __name__ == "__main__":
    main()
