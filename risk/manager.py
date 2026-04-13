"""
risk/manager.py — Dynamic risk management engine.
- Volatility-based position sizing (ATR)
- Daily loss limit with 24h pause
- Equity curve drawdown kill-switch
- Portfolio correlation heat check
- Circuit breaker for extreme volatility
"""

import logging
import threading
import time
from datetime import datetime, date, timedelta, timezone

from utils.helpers import get_synced_now
from config import (
    RISK_PER_TRADE_PCT, DAILY_LOSS_LIMIT_PCT, MAX_DRAWDOWN_PCT,
    MAX_CORRELATED_EXPOSURE_PCT, MAX_OPEN_POSITIONS, MAX_DAILY_TRADES,
    TRADE_COOLDOWN_MINUTES, LEVERAGE, CIRCUIT_BREAKER_ATR_MULTIPLE,
    CORRELATION_GROUPS, PERFORMANCE_DEVIATION_THRESHOLD
)
from utils.helpers import signed_request, is_api_error

logger = logging.getLogger("oracle.risk")


class RiskManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.last_trade_time = None
        self.daily_trade_count = 0
        self.daily_trade_date = get_synced_now().date()
        self.asset_cooldowns = {}  # {ticker: last_signal_datetime}
        self.daily_pnl = 0.0
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.is_paused = False
        self.pause_reason = ""
        self.pause_until = None

        # Performance tracking for backtest-to-live guard
        self.live_trades = []  # Recent trade results

    def update_equity(self, equity: float):
        with self._lock:
            self.current_equity = equity
            if equity > self.peak_equity:
                self.peak_equity = equity

    def get_drawdown_pct(self) -> float:
        with self._lock:
            if self.peak_equity <= 0:
                return 0.0
            return (self.peak_equity - self.current_equity) / self.peak_equity * 100

    def fetch_account_equity(self) -> float:
        """Fetch current equity from Binance."""
        try:
            resp = signed_request('GET', '/fapi/v2/balance')
            if is_api_error(resp):
                return self.current_equity
            if isinstance(resp, list):
                usdt = next((float(a['balance']) for a in resp if a['asset'] == 'USDT'), 0)
                self.update_equity(usdt)
                return usdt
        except Exception as e:
            logger.error(f"Failed to fetch equity: {e}")
        return self.current_equity

    def calculate_position_size(self, entry_price: float, stop_loss: float,
                                 atr: float, symbol: str) -> dict:
        """
        Calculate position size based on ATR and risk per trade.
        Risk exactly RISK_PER_TRADE_PCT of equity per trade.
        """
        equity = self.current_equity or self.fetch_account_equity()
        if equity <= 0:
            return {'quantity': 0, 'margin': 0, 'notional': 0, 'risk_usdt': 0}

        risk_usdt = equity * (RISK_PER_TRADE_PCT / 100)
        risk_per_unit = abs(entry_price - stop_loss)

        if risk_per_unit <= 0:
            risk_per_unit = atr * 1.6  # Fallback

        # Quantity = risk_in_dollars / risk_per_unit
        quantity = risk_usdt / risk_per_unit
        notional = quantity * entry_price
        margin = notional / LEVERAGE

        logger.info(
            f"Position Sizing: equity=${equity:,.2f}, risk=${risk_usdt:,.2f} "
            f"({RISK_PER_TRADE_PCT}%), qty={quantity:.6f}, "
            f"notional=${notional:,.2f}, margin=${margin:,.2f}"
        )

        return {
            'quantity': quantity,
            'margin': margin,
            'notional': notional,
            'risk_usdt': risk_usdt,
        }

    def pre_trade_check(self, ticker: str, signal_data: dict) -> tuple:
        """
        Run all pre-trade risk checks.
        Returns (approved: bool, reason: str).
        """
        with self._lock:
            # Reset daily counter at midnight
            if get_synced_now().date() != self.daily_trade_date:
                self.daily_trade_count = 0
                self.daily_trade_date = get_synced_now().date()
                self.daily_pnl = 0.0

            # Check if paused
            if self.is_paused:
                if self.pause_until and get_synced_now() > self.pause_until:
                    self.is_paused = False
                    self.pause_reason = ""
                    self.pause_until = None
                    logger.info("Risk pause expired — resuming trading")
                else:
                    return False, f"PAUSED: {self.pause_reason}"

            # Daily trade limit
            if self.daily_trade_count >= MAX_DAILY_TRADES:
                return False, f"Daily trade limit ({self.daily_trade_count}/{MAX_DAILY_TRADES})"

            # Global cooldown
            if self.last_trade_time:
                elapsed = (get_synced_now() - self.last_trade_time).total_seconds() / 60
                if elapsed < TRADE_COOLDOWN_MINUTES:
                    remaining = TRADE_COOLDOWN_MINUTES - int(elapsed)
                    return False, f"Cooldown ({remaining}min remaining)"

            # Asset cooldown
            if ticker in self.asset_cooldowns:
                elapsed = (get_synced_now() - self.asset_cooldowns[ticker]).total_seconds() / 60
                if elapsed < TRADE_COOLDOWN_MINUTES:
                    return False, f"Asset cooldown ({int(elapsed)}min ago)"

        # Max open positions check
        try:
            positions = signed_request('GET', '/fapi/v2/positionRisk')
            if isinstance(positions, list):
                active = [p for p in positions if float(p.get('positionAmt', 0)) != 0]
                if len(active) >= MAX_OPEN_POSITIONS:
                    return False, f"Max positions ({len(active)}/{MAX_OPEN_POSITIONS})"

                # Portfolio correlation check
                corr_ok, corr_reason = self._check_correlation(
                    ticker, signal_data.get('signal', ''), active
                )
                if not corr_ok:
                    return False, corr_reason
        except Exception as e:
            logger.warning(f"Position check failed: {e}")

        # Daily loss limit
        dd = self.get_drawdown_pct()
        if dd >= MAX_DRAWDOWN_PCT:
            self._pause(f"Drawdown kill-switch ({dd:.1f}% >= {MAX_DRAWDOWN_PCT}%)",
                        hours=24)
            return False, f"Drawdown kill-switch ({dd:.1f}%)"

        # Circuit breaker: check if current ATR is extreme
        atr_val = signal_data.get('atr', 0)
        price = signal_data.get('price', 0)
        if atr_val > 0 and price > 0:
            atr_pct = atr_val / price * 100
            if atr_pct > 3.0:  # >3% ATR on 15m is extreme
                return False, f"Circuit breaker: ATR {atr_pct:.2f}% too high"

        return True, "All checks passed"

    def _check_correlation(self, ticker: str, signal: str,
                           active_positions: list) -> tuple:
        """Check portfolio correlation exposure."""
        # Find which group the new ticker belongs to
        ticker_group = None
        for group_name, symbols in CORRELATION_GROUPS.items():
            if ticker in symbols:
                ticker_group = group_name
                break

        if ticker_group is None:
            return True, "OK"

        # Count existing positions in same group
        group_symbols = CORRELATION_GROUPS[ticker_group]
        group_positions = [p for p in active_positions
                           if p['symbol'] in group_symbols
                           and float(p.get('positionAmt', 0)) != 0]

        # Calculate group exposure as % of total positions
        total_active = len(active_positions)
        group_active = len(group_positions) + 1  # +1 for proposed trade

        if total_active > 0:
            group_pct = (group_active / MAX_OPEN_POSITIONS) * 100
            if group_pct > MAX_CORRELATED_EXPOSURE_PCT:
                return False, (f"Correlated exposure too high: {ticker_group} "
                              f"({group_active}/{MAX_OPEN_POSITIONS} = {group_pct:.0f}%)")

        return True, "OK"

    def record_trade(self, ticker: str):
        """Record that a trade was executed."""
        with self._lock:
            self.last_trade_time = get_synced_now()
            self.daily_trade_count += 1
            self.asset_cooldowns[ticker] = get_synced_now()

    def record_trade_result(self, pnl_pct: float):
        """Record trade result for performance tracking."""
        with self._lock:
            self.daily_pnl += pnl_pct
            self.live_trades.append({
                'timestamp': get_synced_now(),
                'pnl_pct': pnl_pct
            })
            # Keep only last 100 trades
            if len(self.live_trades) > 100:
                self.live_trades = self.live_trades[-100:]

            # Check daily loss limit
            if self.daily_pnl <= -DAILY_LOSS_LIMIT_PCT:
                self._pause(
                    f"Daily loss limit ({self.daily_pnl:.2f}% <= -{DAILY_LOSS_LIMIT_PCT}%)",
                    hours=24
                )

    def _pause(self, reason: str, hours: float = 24):
        self.is_paused = True
        self.pause_reason = reason
        self.pause_until = get_synced_now() + timedelta(hours=hours)
        logger.warning(f"RISK PAUSE: {reason} — resuming at {self.pause_until}")

        # Send alert
        try:
            from utils.email_alerts import send_alert
            send_alert("Risk Pause Activated", reason)
        except Exception:
            pass

    def check_backtest_consistency(self, backtest_sharpe: float,
                                     live_sharpe: float) -> bool:
        """Auto-pause if live performance deviates >15% from backtest."""
        if backtest_sharpe <= 0:
            return True
        deviation = abs(live_sharpe - backtest_sharpe) / backtest_sharpe * 100
        if deviation > PERFORMANCE_DEVIATION_THRESHOLD:
            self._pause(
                f"Backtest-to-live deviation {deviation:.1f}% > {PERFORMANCE_DEVIATION_THRESHOLD}%",
                hours=24
            )
            return False
        return True

    def get_status(self) -> dict:
        return {
            'is_paused': self.is_paused,
            'pause_reason': self.pause_reason,
            'daily_trades': self.daily_trade_count,
            'daily_pnl': self.daily_pnl,
            'drawdown_pct': self.get_drawdown_pct(),
            'equity': self.current_equity,
            'peak_equity': self.peak_equity,
        }
