"""
data/binance_ws.py — Binance WebSocket streams for real-time data.
- Price ticker stream for all watchlist assets
- Aggregated trade stream for order-flow imbalance proxy
- User Data Stream for real-time order fills & position updates
"""
from __future__ import annotations

import json
import time
import logging
import threading
import requests
import websocket

from config import BINANCE_WS_URL, BASE_URL, BINANCE_TEST_API

logger = logging.getLogger("oracle.ws")

# Global price store
live_prices = {}
# Order flow: {symbol: {'buy_vol': float, 'sell_vol': float, 'last_reset': float}}
order_flow = {}
_order_flow_lock = threading.Lock()

# User data stream callbacks
_user_data_callbacks = []


def get_live_price(symbol: str) -> float | None:
    return live_prices.get(symbol)


def get_order_flow_imbalance(symbol: str) -> float:
    """Returns buy_vol / (buy_vol + sell_vol) ratio. >0.5 = buy pressure."""
    with _order_flow_lock:
        flow = order_flow.get(symbol, {})
        buy_v = flow.get('buy_vol', 0)
        sell_v = flow.get('sell_vol', 0)
        total = buy_v + sell_v
        if total == 0:
            return 0.5
        return buy_v / total


def reset_order_flow(symbol: str):
    with _order_flow_lock:
        try:
            from utils.helpers import get_synced_timestamp
            ts = get_synced_timestamp()
        except Exception:
            ts = int(time.time() * 1000)
        order_flow[symbol] = {'buy_vol': 0, 'sell_vol': 0, 'last_reset': ts}


def register_user_data_callback(callback):
    _user_data_callbacks.append(callback)


# ==================== TICKER STREAM ====================
def _on_ticker_message(ws, message):
    try:
        data = json.loads(message)
        if 'data' in data:
            data = data['data']
        if 'c' in data and 's' in data:
            live_prices[data['s']] = float(data['c'])
    except Exception:
        pass


def _on_ticker_error(ws, error):
    logger.warning(f"Ticker WS error: {error}")


def _on_ticker_close(ws, *args):
    logger.info("Ticker WS closed — reconnecting in 5s...")
    time.sleep(5)
    start_ticker_stream(list(live_prices.keys()))


def _on_ticker_open(ws):
    logger.info("Ticker WS connected")
    subs = [f"{s.lower()}@ticker" for s in live_prices.keys()]
    if subs:
        ws.send(json.dumps({"method": "SUBSCRIBE", "params": subs, "id": 1}))


def start_ticker_stream(symbols: list):
    for s in symbols:
        live_prices.setdefault(s, None)
    ws = websocket.WebSocketApp(
        BINANCE_WS_URL,
        on_open=_on_ticker_open,
        on_message=_on_ticker_message,
        on_error=_on_ticker_error,
        on_close=_on_ticker_close
    )
    ws.run_forever(ping_interval=20, ping_timeout=10)


# ==================== AGGREGATED TRADES STREAM ====================
def _on_agg_trade_message(ws, message):
    try:
        data = json.loads(message)
        if 'data' in data:
            data = data['data']
        if 'e' in data and data['e'] == 'aggTrade':
            symbol = data['s']
            qty = float(data['q'])
            is_buyer_maker = data['m']  # True = sell aggressor, False = buy aggressor
            with _order_flow_lock:
                if symbol not in order_flow:
                    order_flow[symbol] = {'buy_vol': 0, 'sell_vol': 0, 'last_reset': int(time.time() * 1000)}
                if is_buyer_maker:
                    order_flow[symbol]['sell_vol'] += qty
                else:
                    order_flow[symbol]['buy_vol'] += qty
    except Exception:
        pass


def _on_agg_trade_open(ws):
    logger.info("AggTrade WS connected")


def _on_agg_trade_close(ws, *args):
    logger.info("AggTrade WS closed — reconnecting in 5s...")
    time.sleep(5)


def start_agg_trade_stream(symbols: list):
    streams = "/".join([f"{s.lower()}@aggTrade" for s in symbols])
    url = f"{BINANCE_WS_URL}/{streams}"

    def _run():
        while True:
            ws = websocket.WebSocketApp(
                url,
                on_open=_on_agg_trade_open,
                on_message=_on_agg_trade_message,
                on_error=_on_ticker_error,
                on_close=_on_agg_trade_close
            )
            ws.run_forever(ping_interval=20, ping_timeout=10)
            time.sleep(5)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


# ==================== USER DATA STREAM ====================
_listen_key = None
_listen_key_lock = threading.Lock()


def _get_listen_key() -> str | None:
    if not BINANCE_TEST_API:
        return None
    try:
        resp = requests.post(
            f"{BASE_URL}/fapi/v1/listenKey",
            headers={"X-MBX-APIKEY": BINANCE_TEST_API},
            timeout=10
        )
        data = resp.json()
        return data.get('listenKey')
    except Exception as e:
        logger.error(f"Failed to get listen key: {e}")
        return None


def _keepalive_listen_key():
    while True:
        time.sleep(30 * 60)  # Every 30 min
        try:
            with _listen_key_lock:
                if _listen_key:
                    requests.put(
                        f"{BASE_URL}/fapi/v1/listenKey",
                        headers={"X-MBX-APIKEY": BINANCE_TEST_API},
                        timeout=10
                    )
                    logger.debug("Listen key keepalive sent")
        except Exception as e:
            logger.warning(f"Listen key keepalive failed: {e}")


def _on_user_data_message(ws, message):
    try:
        data = json.loads(message)
        event_type = data.get('e', '')
        for cb in _user_data_callbacks:
            try:
                cb(event_type, data)
            except Exception as e:
                logger.error(f"User data callback error: {e}")
    except Exception:
        pass


def start_user_data_stream():
    global _listen_key
    _listen_key = _get_listen_key()
    if not _listen_key:
        logger.warning("User Data Stream disabled (no listen key)")
        return

    # Start keepalive thread
    threading.Thread(target=_keepalive_listen_key, daemon=True).start()

    def _run():
        while True:
            url = f"{BINANCE_WS_URL}/{_listen_key}"
            ws = websocket.WebSocketApp(
                url,
                on_message=_on_user_data_message,
                on_error=lambda ws, e: logger.warning(f"UserData WS error: {e}"),
                on_close=lambda ws, *a: logger.info("UserData WS closed — reconnecting..."),
                on_open=lambda ws: logger.info("UserData WS connected — real-time fills active")
            )
            ws.run_forever(ping_interval=20, ping_timeout=10)
            time.sleep(5)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t
