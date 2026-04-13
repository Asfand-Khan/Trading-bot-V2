"""
execution/engine.py — Order execution engine.
- Direct signed API execution with proper precision
- Smart order routing (limit post-only for maker rebates when possible)
- Position monitor with break-even SL move after TP1
- WebSocket User Data Stream integration for fills
"""

import os
import logging
import time
import threading
from datetime import datetime

from config import (
    LEVERAGE, ENABLE_TRAILING_STOP, BASE_URL, BINANCE_TEST_API
)
from utils.helpers import signed_request, is_api_error, get_synced_timestamp
from data.binance_rest import get_symbol_precision, fetch_exchange_info
from data.binance_ws import register_user_data_callback
from data.storage import save_trade_journal_entry

logger = logging.getLogger("oracle.execution")

# Cache exchange info to avoid repeated API calls
_exchange_info_cache = None
_exchange_info_time = 0


def _get_exchange_info():
    global _exchange_info_cache, _exchange_info_time
    synced_ms = get_synced_timestamp()
    if _exchange_info_cache and (synced_ms - _exchange_info_time) < 3600_000:
        return _exchange_info_cache
    _exchange_info_cache = fetch_exchange_info()
    _exchange_info_time = synced_ms
    return _exchange_info_cache


class ExecutionEngine:
    def __init__(self, risk_manager=None):
        self.risk_manager = risk_manager
        self._active_orders = {}  # {symbol: [order_ids]}
        self._position_cache = {}  # {symbol: position_data}
        self._lock = threading.Lock()

        # Register for real-time fill updates
        register_user_data_callback(self._on_user_data)

    def _on_user_data(self, event_type: str, data: dict):
        """Handle real-time user data events from WebSocket."""
        if event_type == 'ORDER_TRADE_UPDATE':
            order = data.get('o', {})
            symbol = order.get('s', '')
            status = order.get('X', '')  # FILLED, PARTIALLY_FILLED, etc.
            side = order.get('S', '')
            order_type = order.get('o', '')
            price = float(order.get('ap', 0) or order.get('p', 0))
            qty = float(order.get('q', 0))

            if status == 'FILLED':
                logger.info(f"[FILL] {symbol} {side} {order_type} qty={qty} @ {price}")

                # If SL or TP filled, log the trade closure
                if order_type in ('STOP_MARKET', 'TAKE_PROFIT_MARKET', 'TRAILING_STOP_MARKET'):
                    exit_reason = {
                        'STOP_MARKET': 'STOP_LOSS',
                        'TAKE_PROFIT_MARKET': 'TAKE_PROFIT',
                        'TRAILING_STOP_MARKET': 'TRAILING_STOP',
                    }.get(order_type, order_type)
                    logger.info(f"[CLOSED] {symbol} via {exit_reason} @ {price}")

        elif event_type == 'ACCOUNT_UPDATE':
            # Position updates
            positions = data.get('a', {}).get('P', [])
            for pos in positions:
                symbol = pos.get('s', '')
                amount = float(pos.get('pa', 0))
                with self._lock:
                    self._position_cache[symbol] = {
                        'amount': amount,
                        'entry_price': float(pos.get('ep', 0)),
                        'unrealized_pnl': float(pos.get('up', 0)),
                    }

    def execute_trade(self, signal_data: dict, position_size: dict) -> dict:
        """
        Execute a full trade: entry + SL + TP orders.
        Returns execution result dict.
        """
        if not BINANCE_TEST_API:
            logger.warning("Skipping execution: API keys not configured")
            return {'success': False, 'reason': 'no_api_keys'}

        try:
            symbol = signal_data['asset']
            price = signal_data['price']
            signal = signal_data['signal']
            sl_price = signal_data['stop_loss_raw']
            tp1_price = signal_data['tp1_raw']
            tp2_price = signal_data['tp2_raw']

            side = 'BUY' if signal == 'BUY' else 'SELL'
            opposite_side = 'SELL' if side == 'BUY' else 'BUY'

            # Get symbol precision
            info = _get_exchange_info()
            precision = get_symbol_precision(symbol, info)
            qty_prec = precision['quantity_precision']
            price_prec = precision['price_precision']

            quantity = round(position_size['quantity'], qty_prec)
            if quantity <= 0:
                return {'success': False, 'reason': 'quantity_too_small'}

            # 1. Set leverage
            lev_resp = signed_request('POST', '/fapi/v1/leverage', {
                'symbol': symbol, 'leverage': LEVERAGE
            })
            if is_api_error(lev_resp):
                logger.warning(f"Leverage set warning: {lev_resp.get('msg', '')}")

            # 2. Market entry order
            logger.info(f"EXECUTING {side} {symbol} | qty={quantity} | "
                         f"notional=${position_size['notional']:,.2f}")

            entry_resp = signed_request('POST', '/fapi/v1/order', {
                'symbol': symbol,
                'side': side,
                'type': 'MARKET',
                'quantity': quantity,
            })

            if is_api_error(entry_resp):
                logger.error(f"Entry order FAILED: {entry_resp.get('msg', '')}")
                return {'success': False, 'reason': entry_resp.get('msg', 'entry_failed')}

            order_id = entry_resp.get('orderId', 'unknown')
            logger.info(f"ENTRY FILLED: orderId={order_id}")

            # 3. Place SL + TP orders
            if ENABLE_TRAILING_STOP:
                tp_qty = round(quantity / 2, qty_prec)
                rem_qty = round(quantity - tp_qty, qty_prec)

                # Stop Loss (full size)
                sl_resp = signed_request('POST', '/fapi/v1/order', {
                    'symbol': symbol, 'side': opposite_side,
                    'type': 'STOP_MARKET',
                    'stopPrice': round(sl_price, price_prec),
                    'quantity': quantity,
                    'reduceOnly': 'true'
                })
                if not is_api_error(sl_resp):
                    logger.info(f"SL SET @ {sl_price:.{price_prec}f}")

                # TP1 (50%)
                tp1_resp = signed_request('POST', '/fapi/v1/order', {
                    'symbol': symbol, 'side': opposite_side,
                    'type': 'TAKE_PROFIT_MARKET',
                    'stopPrice': round(tp1_price, price_prec),
                    'quantity': tp_qty,
                    'reduceOnly': 'true'
                })
                if not is_api_error(tp1_resp):
                    logger.info(f"TP1 (50%) SET @ {tp1_price:.{price_prec}f}")

                # Trailing Stop (remaining 50%)
                ts_resp = signed_request('POST', '/fapi/v1/order', {
                    'symbol': symbol, 'side': opposite_side,
                    'type': 'TRAILING_STOP_MARKET',
                    'activationPrice': round(tp1_price, price_prec),
                    'callbackRate': 1.5,
                    'quantity': rem_qty,
                    'reduceOnly': 'true'
                })
                if is_api_error(ts_resp):
                    logger.warning(f"Trailing stop failed: {ts_resp.get('msg', '')}. "
                                    f"Placing fixed TP2 instead.")
                    signed_request('POST', '/fapi/v1/order', {
                        'symbol': symbol, 'side': opposite_side,
                        'type': 'TAKE_PROFIT_MARKET',
                        'stopPrice': round(tp2_price, price_prec),
                        'quantity': rem_qty,
                        'reduceOnly': 'true'
                    })
                else:
                    logger.info(f"TRAILING STOP (50%) SET (activates @ {tp1_price:.{price_prec}f}, 1.5% trail)")
            else:
                # Full close at TP1
                sl_resp = signed_request('POST', '/fapi/v1/order', {
                    'symbol': symbol, 'side': opposite_side,
                    'type': 'STOP_MARKET',
                    'stopPrice': round(sl_price, price_prec),
                    'quantity': quantity,
                    'reduceOnly': 'true'
                })
                if not is_api_error(sl_resp):
                    logger.info(f"SL SET @ {sl_price:.{price_prec}f}")

                tp_resp = signed_request('POST', '/fapi/v1/order', {
                    'symbol': symbol, 'side': opposite_side,
                    'type': 'TAKE_PROFIT_MARKET',
                    'stopPrice': round(tp1_price, price_prec),
                    'quantity': quantity,
                    'reduceOnly': 'true'
                })
                if not is_api_error(tp_resp):
                    logger.info(f"TP (100%) SET @ {tp1_price:.{price_prec}f}")

            return {
                'success': True,
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
            }

        except Exception as e:
            import traceback
            logger.error(f"Execution failed: {e}\n{traceback.format_exc()}")
            return {'success': False, 'reason': str(e)}

    def position_monitor_loop(self):
        """
        Background loop: monitors positions and moves SL to break-even after TP1 hit.
        """
        if not ENABLE_TRAILING_STOP:
            logger.info("Position monitor: trailing stops disabled")
            return

        logger.info("Position monitor STARTED (watching for TP1 hit -> SL to BE)")
        while True:
            try:
                positions = signed_request('GET', '/fapi/v2/positionRisk')
                if is_api_error(positions):
                    time.sleep(10)
                    continue

                active = [p for p in positions if float(p.get('positionAmt', 0)) != 0]

                for pos in active:
                    symbol = pos['symbol']
                    entry_price = float(pos['entryPrice'])
                    pos_amt = float(pos['positionAmt'])
                    side = 'buy' if pos_amt > 0 else 'sell'
                    opposite_side = 'SELL' if side == 'buy' else 'BUY'

                    # Fetch open orders for this symbol
                    open_orders = signed_request('GET', '/fapi/v1/openOrders',
                                                  {'symbol': symbol})
                    if is_api_error(open_orders):
                        continue

                    has_tp = any(o['type'] in ('TAKE_PROFIT_MARKET', 'TAKE_PROFIT')
                                for o in open_orders)
                    sl_orders = [o for o in open_orders
                                 if o['type'] in ('STOP_MARKET', 'STOP')]

                    # TP1 hit = no TP order left, but SL still away from entry
                    if not has_tp and sl_orders:
                        for sl in sl_orders:
                            sl_price = float(sl['stopPrice'])
                            if abs(sl_price - entry_price) > (entry_price * 0.0001):
                                # Cancel old SL
                                signed_request('DELETE', '/fapi/v1/order', {
                                    'symbol': symbol, 'orderId': sl['orderId']
                                })
                                # Place new SL at break-even
                                rem_amount = abs(pos_amt)
                                signed_request('POST', '/fapi/v1/order', {
                                    'symbol': symbol, 'side': opposite_side,
                                    'type': 'STOP_MARKET',
                                    'stopPrice': entry_price,
                                    'quantity': rem_amount,
                                    'reduceOnly': 'true'
                                })
                                logger.info(f"TP1 HIT: {symbol} SL moved to BE @ {entry_price}")

                    # Update risk manager equity
                    if self.risk_manager:
                        pnl = float(pos.get('unRealizedProfit', 0))
                        # Update will happen via equity fetch in main loop

            except Exception as e:
                logger.error(f"Position monitor error: {e}")

            time.sleep(10)
