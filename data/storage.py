"""
data/storage.py — Supabase/PostgreSQL storage layer.
Stores ticks, signals, OHLCV snapshots, trade journal, equity curve.
"""

import logging
import psycopg2
from psycopg2 import pool
from datetime import datetime

from config import SUPABASE_DB_URL

logger = logging.getLogger("oracle.storage")

db_pool = None


def init_db():
    global db_pool
    if not SUPABASE_DB_URL or "[YOUR-PASSWORD]" in SUPABASE_DB_URL:
        logger.warning("No valid SUPABASE_DB_URL — database disabled")
        return False
    try:
        db_pool = psycopg2.pool.SimpleConnectionPool(1, 10, SUPABASE_DB_URL)
        conn = db_pool.getconn()
        cur = conn.cursor()

        # Core tables
        cur.execute("""
            CREATE TABLE IF NOT EXISTS market_ticks (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                asset VARCHAR(20),
                price NUMERIC
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS signals_log (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                asset VARCHAR(20),
                signal VARCHAR(10),
                price NUMERIC,
                stop_loss VARCHAR(20),
                tp1 VARCHAR(20),
                tp2 VARCHAR(20),
                confidence INTEGER,
                ml_probability NUMERIC,
                regime VARCHAR(30),
                why TEXT,
                status VARCHAR(20) DEFAULT 'PENDING'
            );
        """)

        # New tables for v5.0
        cur.execute("""
            CREATE TABLE IF NOT EXISTS equity_curve (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                equity NUMERIC,
                daily_pnl NUMERIC,
                drawdown_pct NUMERIC,
                sharpe_30d NUMERIC,
                open_positions INTEGER,
                regime VARCHAR(30)
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS trade_journal (
                id SERIAL PRIMARY KEY,
                trade_id VARCHAR(50),
                timestamp_open TIMESTAMPTZ,
                timestamp_close TIMESTAMPTZ,
                asset VARCHAR(20),
                side VARCHAR(10),
                entry_price NUMERIC,
                exit_price NUMERIC,
                quantity NUMERIC,
                leverage INTEGER,
                pnl_usdt NUMERIC,
                pnl_pct NUMERIC,
                fees_usdt NUMERIC,
                funding_paid NUMERIC,
                net_pnl NUMERIC,
                duration_minutes INTEGER,
                exit_reason VARCHAR(30)
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ml_feature_importance (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                feature_name VARCHAR(50),
                importance NUMERIC,
                model_version VARCHAR(20)
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_1m (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ,
                asset VARCHAR(20),
                open NUMERIC,
                high NUMERIC,
                low NUMERIC,
                close NUMERIC,
                volume NUMERIC,
                UNIQUE(timestamp, asset)
            );
        """)

        # Add columns if they don't exist (migration-safe)
        for col, typ in [('ml_probability', 'NUMERIC'), ('regime', 'VARCHAR(30)')]:
            try:
                cur.execute(f"ALTER TABLE signals_log ADD COLUMN IF NOT EXISTS {col} {typ};")
            except Exception:
                pass

        conn.commit()
        cur.close()
        db_pool.putconn(conn)
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Database init failed: {e}")
        return False


def _execute(query: str, params: tuple = None, fetch: bool = False):
    if not db_pool:
        return None
    conn = None
    try:
        conn = db_pool.getconn()
        cur = conn.cursor()
        cur.execute(query, params)
        if fetch:
            result = cur.fetchall()
        else:
            result = None
        conn.commit()
        cur.close()
        return result
    except Exception as e:
        logger.error(f"DB query error: {e}")
        if conn:
            conn.rollback()
        return None
    finally:
        if conn:
            db_pool.putconn(conn)


def save_signal(signal_data: dict):
    _execute("""
        INSERT INTO signals_log (asset, signal, price, stop_loss, tp1, tp2,
                                 confidence, ml_probability, regime, why, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        str(signal_data.get('asset')),
        str(signal_data.get('signal')),
        float(signal_data.get('price', 0)),
        str(signal_data.get('stop_loss', '')),
        str(signal_data.get('tp1', '')),
        str(signal_data.get('tp2', '')),
        int(signal_data.get('confidence', 0)),
        float(signal_data.get('ml_probability', 0)),
        str(signal_data.get('regime', '')),
        str(signal_data.get('why', '')),
        str(signal_data.get('status', 'EXECUTED'))
    ))


def save_tick(asset: str, price: float):
    _execute(
        "INSERT INTO market_ticks (asset, price) VALUES (%s, %s)",
        (asset, price)
    )


def save_ticks_batch(ticks: list):
    """ticks: list of (timestamp_str, asset, price)"""
    if not db_pool or not ticks:
        return
    conn = None
    try:
        conn = db_pool.getconn()
        cur = conn.cursor()
        cur.executemany(
            "INSERT INTO market_ticks (timestamp, asset, price) VALUES (%s, %s, %s)",
            ticks
        )
        conn.commit()
        cur.close()
    except Exception as e:
        logger.error(f"Batch tick insert error: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            db_pool.putconn(conn)


def save_equity_snapshot(equity: float, daily_pnl: float, drawdown_pct: float,
                         sharpe: float, open_positions: int, regime: str):
    _execute("""
        INSERT INTO equity_curve (equity, daily_pnl, drawdown_pct, sharpe_30d, open_positions, regime)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (equity, daily_pnl, drawdown_pct, sharpe, open_positions, regime))


def save_trade_journal_entry(trade: dict):
    _execute("""
        INSERT INTO trade_journal (trade_id, timestamp_open, timestamp_close, asset, side,
            entry_price, exit_price, quantity, leverage, pnl_usdt, pnl_pct, fees_usdt,
            funding_paid, net_pnl, duration_minutes, exit_reason)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        trade.get('trade_id'), trade.get('timestamp_open'), trade.get('timestamp_close'),
        trade.get('asset'), trade.get('side'), trade.get('entry_price'),
        trade.get('exit_price'), trade.get('quantity'), trade.get('leverage'),
        trade.get('pnl_usdt'), trade.get('pnl_pct'), trade.get('fees_usdt'),
        trade.get('funding_paid'), trade.get('net_pnl'), trade.get('duration_minutes'),
        trade.get('exit_reason')
    ))


def save_feature_importance(features: dict, model_version: str):
    if not db_pool:
        return
    conn = None
    try:
        conn = db_pool.getconn()
        cur = conn.cursor()
        for name, importance in features.items():
            cur.execute("""
                INSERT INTO ml_feature_importance (feature_name, importance, model_version)
                VALUES (%s, %s, %s)
            """, (name, float(importance), model_version))
        conn.commit()
        cur.close()
    except Exception as e:
        logger.error(f"Feature importance save error: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            db_pool.putconn(conn)


def get_equity_curve(days: int = 30) -> list:
    result = _execute("""
        SELECT timestamp, equity, daily_pnl, drawdown_pct, sharpe_30d
        FROM equity_curve
        WHERE timestamp > NOW() - INTERVAL '%s days'
        ORDER BY timestamp ASC
    """, (days,), fetch=True)
    return result or []


def get_recent_trades(days: int = 30) -> list:
    result = _execute("""
        SELECT * FROM trade_journal
        WHERE timestamp_close > NOW() - INTERVAL '%s days'
        ORDER BY timestamp_close DESC
    """, (days,), fetch=True)
    return result or []
