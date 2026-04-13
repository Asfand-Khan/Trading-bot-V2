"""
monitoring/performance.py — Live performance tracking.
- Rolling Sharpe, drawdown, win-rate, profit factor
- Equity curve monitoring with alerts
- Backtest-to-live consistency guard
- Health check endpoint
"""

import logging
import numpy as np
import threading
import time
from datetime import datetime, date, timezone

from utils.helpers import get_synced_now
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

from config import HEALTH_CHECK_PORT
from data.storage import save_equity_snapshot, get_equity_curve, get_recent_trades

logger = logging.getLogger("oracle.monitoring")


class PerformanceTracker:
    def __init__(self, risk_manager=None):
        self.risk_manager = risk_manager
        self._lock = threading.Lock()
        self.daily_returns = []  # [(date, return_pct)]
        self.trade_results = []  # [{pnl_pct, ...}]
        self.snapshots = []  # Periodic equity snapshots

    def record_trade(self, trade_result: dict):
        with self._lock:
            self.trade_results.append(trade_result)
            self.daily_returns.append((get_synced_now().date(), trade_result.get('pnl_pct', 0)))

    def get_rolling_sharpe(self, window: int = 30) -> float:
        """Calculate rolling Sharpe ratio from recent trades."""
        with self._lock:
            if len(self.trade_results) < 5:
                return 0.0
            recent = self.trade_results[-window:]
            returns = [t.get('pnl_pct', 0) for t in recent]

        arr = np.array(returns)
        if arr.std() == 0:
            return 0.0
        # Annualize: assume ~6 trades per day
        return float((arr.mean() / arr.std()) * np.sqrt(252 * 6))

    def get_win_rate(self, window: int = 50) -> float:
        with self._lock:
            if not self.trade_results:
                return 0.0
            recent = self.trade_results[-window:]
            wins = sum(1 for t in recent if t.get('pnl_pct', 0) > 0)
            return wins / len(recent) * 100

    def get_profit_factor(self, window: int = 50) -> float:
        with self._lock:
            if not self.trade_results:
                return 0.0
            recent = self.trade_results[-window:]
            gross_profit = sum(t['pnl_pct'] for t in recent if t.get('pnl_pct', 0) > 0)
            gross_loss = abs(sum(t['pnl_pct'] for t in recent if t.get('pnl_pct', 0) < 0))
            if gross_loss == 0:
                return float('inf') if gross_profit > 0 else 0
            return gross_profit / gross_loss

    def get_live_metrics(self) -> dict:
        """Get current live performance metrics."""
        equity = 0.0
        drawdown = 0.0
        if self.risk_manager:
            equity = self.risk_manager.current_equity
            drawdown = self.risk_manager.get_drawdown_pct()

        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'equity': equity,
            'drawdown_pct': drawdown,
            'sharpe_30d': self.get_rolling_sharpe(30),
            'win_rate': self.get_win_rate(),
            'profit_factor': self.get_profit_factor(),
            'total_trades': len(self.trade_results),
            'daily_trades': self.risk_manager.daily_trade_count if self.risk_manager else 0,
            'is_paused': self.risk_manager.is_paused if self.risk_manager else False,
            'pause_reason': self.risk_manager.pause_reason if self.risk_manager else '',
        }

    def save_snapshot(self, regime: str = ''):
        """Save equity snapshot to database."""
        metrics = self.get_live_metrics()
        try:
            save_equity_snapshot(
                equity=metrics['equity'],
                daily_pnl=self.risk_manager.daily_pnl if self.risk_manager else 0,
                drawdown_pct=metrics['drawdown_pct'],
                sharpe=metrics['sharpe_30d'],
                open_positions=metrics['daily_trades'],
                regime=regime
            )
        except Exception as e:
            logger.error(f"Snapshot save error: {e}")

    def generate_daily_report(self) -> dict:
        """Generate daily P&L report for email."""
        metrics = self.get_live_metrics()
        metrics['date'] = get_synced_now().date().isoformat()

        with self._lock:
            today_trades = [t for d, t in zip(
                [r[0] for r in self.daily_returns],
                [r[1] for r in self.daily_returns]
            ) if d == get_synced_now().date()]
            metrics['daily_pnl'] = sum(today_trades) if today_trades else 0
            metrics['daily_pnl_pct'] = metrics['daily_pnl']
            metrics['trades_today'] = len(today_trades)
            metrics['open_positions'] = 0

        return metrics


# ==================== HEALTH CHECK HTTP SERVER ====================
_performance_tracker = None


def set_global_tracker(tracker: PerformanceTracker):
    global _performance_tracker
    _performance_tracker = tracker


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            data = {'status': 'ok', 'timestamp': datetime.now(timezone.utc).isoformat()}
            if _performance_tracker:
                data.update(_performance_tracker.get_live_metrics())
            self.wfile.write(json.dumps(data).encode())
        elif self.path == '/metrics':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            data = {}
            if _performance_tracker:
                data = _performance_tracker.get_live_metrics()
            self.wfile.write(json.dumps(data).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress request logs


def start_health_server(port: int = None):
    port = port or HEALTH_CHECK_PORT
    server = HTTPServer(('0.0.0.0', port), HealthHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    logger.info(f"Health check server started on port {port}")
    return server
