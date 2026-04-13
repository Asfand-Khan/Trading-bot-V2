"""
strategy/rules.py — Core rule-based signal engine.
Multi-timeframe institutional strategy preserved from v4.0, now modular.
Returns structured signal dict or None with full audit trail.
"""
from __future__ import annotations

import logging
import pandas as pd

from indicators.technical import (
    compute_all_indicators, detect_smc_zones, ema, atr as calc_atr, adx as calc_adx
)
from data.binance_rest import fetch_klines, fetch_funding_rate
from data.binance_ws import get_live_price, get_order_flow_imbalance
from config import (
    MIN_CONFIDENCE, ADX_MIN_THRESHOLD, VOLUME_CONFIRMATION_RATIO,
    ATR_STOP_MULTIPLIER, TP1_RR_RATIO, TP2_RR_RATIO
)

logger = logging.getLogger("oracle.rules")


def _safe_float(val) -> float:
    if hasattr(val, 'iloc'):
        return float(val.iloc[0])
    elif hasattr(val, 'item'):
        return float(val.item())
    return float(val)


def analyze_asset(ticker: str) -> dict | None:
    """
    Full multi-timeframe analysis for a single asset.
    Returns signal_data dict with all features for audit, or None.
    """
    try:
        # ---------- Current Price ----------
        current_price = get_live_price(ticker)
        if current_price is None:
            # Fallback to REST
            df_tmp = fetch_klines(ticker, '1m', limit=2)
            if df_tmp.empty:
                logger.info(f"[SKIP] {ticker}: No price data")
                return None
            current_price = float(df_tmp['Close'].iloc[-1])

        # ---------- 4H Data (trend context) ----------
        df_4h = fetch_klines(ticker, '4h', limit=210)
        if len(df_4h) < 200:
            logger.info(f"[SKIP] {ticker}: Insufficient 4H data ({len(df_4h)} bars)")
            return None
        df_4h['EMA200'] = ema(df_4h['Close'], 200)

        # ---------- 1H Data (momentum context) ----------
        df_1h = fetch_klines(ticker, '1h', limit=210)
        if len(df_1h) < 200:
            logger.info(f"[SKIP] {ticker}: Insufficient 1H data ({len(df_1h)} bars)")
            return None
        df_1h['EMA200'] = ema(df_1h['Close'], 200)
        adx_1h, _, _ = calc_adx(df_1h['High'], df_1h['Low'], df_1h['Close'])
        df_1h['ADX'] = adx_1h

        # ---------- 15m Data (entry timeframe) ----------
        df_15m = fetch_klines(ticker, '15m', limit=300)
        if len(df_15m) < 100:
            logger.info(f"[SKIP] {ticker}: Insufficient 15m data ({len(df_15m)} bars)")
            return None
        df_15m = compute_all_indicators(df_15m)

        # ---------- Extract scalar values ----------
        close_4h = _safe_float(df_4h['Close'].iloc[-1])
        ema200_4h = _safe_float(df_4h['EMA200'].iloc[-1])
        close_1h = _safe_float(df_1h['Close'].iloc[-1])
        ema200_1h = _safe_float(df_1h['EMA200'].iloc[-1])
        adx_1h_val = _safe_float(df_1h['ADX'].iloc[-1]) if pd.notna(df_1h['ADX'].iloc[-1]) else 0

        # ADX filter
        if adx_1h_val < ADX_MIN_THRESHOLD:
            return _skip(ticker, f"ADX {adx_1h_val:.1f} < {ADX_MIN_THRESHOLD}", current_price)

        # 15m scalars
        ema50_15m = _safe_float(df_15m['EMA50'].iloc[-1])
        ema200_15m = _safe_float(df_15m['EMA200'].iloc[-1])
        rsi_15m = _safe_float(df_15m['RSI'].iloc[-1])
        macd_15m = _safe_float(df_15m['MACD'].iloc[-1])
        macd_sig_15m = _safe_float(df_15m['MACD_Signal'].iloc[-1])
        bb_upper = _safe_float(df_15m['BB_Upper'].iloc[-1])
        bb_lower = _safe_float(df_15m['BB_Lower'].iloc[-1])
        bb_mid = _safe_float(df_15m['BB_Mid'].iloc[-1])
        vwap_15m = _safe_float(df_15m['VWAP'].iloc[-1])
        atr_val = _safe_float(df_15m['ATR'].iloc[-1])
        if pd.isna(atr_val) or atr_val == 0:
            atr_val = current_price * 0.005

        # Volume: use last COMPLETED candle (iloc[-2])
        current_vol = _safe_float(df_15m['Volume'].iloc[-2]) if len(df_15m) > 2 else _safe_float(df_15m['Volume'].iloc[-1])
        sma_vol = _safe_float(df_15m['Volume_SMA'].iloc[-2]) if len(df_15m) > 2 else _safe_float(df_15m['Volume_SMA'].iloc[-1])

        # MACD crossover on COMPLETED candles
        macd_c = _safe_float(df_15m['MACD'].iloc[-2]) if len(df_15m) > 2 else macd_15m
        macd_s_c = _safe_float(df_15m['MACD_Signal'].iloc[-2]) if len(df_15m) > 2 else macd_sig_15m
        macd_prev = _safe_float(df_15m['MACD'].iloc[-3]) if len(df_15m) > 3 else macd_c
        macd_s_prev = _safe_float(df_15m['MACD_Signal'].iloc[-3]) if len(df_15m) > 3 else macd_s_c
        fresh_buy_cross = macd_c > macd_s_c and macd_prev <= macd_s_prev
        fresh_sell_cross = macd_c < macd_s_c and macd_prev >= macd_s_prev

        # Funding rate
        funding_rate = fetch_funding_rate(ticker)

        # SMC zones
        smc = detect_smc_zones(df_15m)

        # Order flow imbalance
        ofi = get_order_flow_imbalance(ticker)

        # ---------- Volume Filter ----------
        vol_ratio = (current_vol / sma_vol) if sma_vol > 0 else 1.0
        if vol_ratio < VOLUME_CONFIRMATION_RATIO and sma_vol > 0:
            return _skip(ticker, f"Volume {vol_ratio:.0%} < {VOLUME_CONFIRMATION_RATIO:.0%} threshold",
                         current_price)

        # ---------- Signal Direction ----------
        signal = None
        direction = ""

        if (current_price > ema50_15m > ema200_15m and fresh_buy_cross
                and rsi_15m < 65 and smc['smc_buy']):
            if current_price > vwap_15m:
                signal = "BUY"
                direction = "bullish"
            else:
                return _skip(ticker, f"Below VWAP {vwap_15m:.4f} — bearish trap risk",
                             current_price)
        elif (current_price < ema50_15m < ema200_15m and fresh_sell_cross
              and rsi_15m > 35 and smc['smc_sell']):
            if current_price < vwap_15m:
                signal = "SELL"
                direction = "bearish"
            else:
                return _skip(ticker, f"Above VWAP {vwap_15m:.4f} — bullish trap risk",
                             current_price)

        if not signal:
            return _skip(ticker, "No 15m MACD+SMC synergy", current_price)

        # ---------- Confluence Score (0-100) ----------
        score = 0
        why_parts = []

        # Macro alignment (20pts)
        if signal == "BUY" and close_4h > ema200_4h:
            score += 20
            why_parts.append("4H bullish trend")
        elif signal == "SELL" and close_4h < ema200_4h:
            score += 20
            why_parts.append("4H bearish trend")

        # 1H momentum (20pts)
        if signal == "BUY" and close_1h > ema200_1h:
            score += 20
            why_parts.append("1H momentum aligned")
        elif signal == "SELL" and close_1h < ema200_1h:
            score += 20
            why_parts.append("1H momentum aligned")

        # 15m setup (15pts base for passing all filters)
        score += 15
        why_parts.append("Fresh MACD cross + SMC liquidity")

        # RSI sweet spot (15pts)
        if 35 < rsi_15m < 65:
            score += 15

        # MACD acceleration (10pts)
        if signal == "BUY" and macd_15m > 0:
            score += 10
        elif signal == "SELL" and macd_15m < 0:
            score += 10

        # BB position (10pts)
        if signal == "BUY" and current_price < bb_mid:
            score += 10
        elif signal == "SELL" and current_price > bb_mid:
            score += 10

        # Funding squeeze (15pts)
        why_append = ""
        if signal == "BUY" and funding_rate < -0.0001:
            score += 15
            why_append = " + SQUEEZE (negative funding)"
        elif signal == "SELL" and funding_rate > 0.0001:
            score += 15
            why_append = " + SQUEEZE (positive funding)"

        # Volume surge bonus (5pts)
        if vol_ratio >= 1.5:
            score += 5
            why_append += f" + VOLUME SURGE ({vol_ratio:.0%}x)"

        # Order flow alignment bonus (5pts) — new in v5.0
        if signal == "BUY" and ofi > 0.55:
            score += 5
            why_append += f" + BUY FLOW ({ofi:.0%})"
        elif signal == "SELL" and ofi < 0.45:
            score += 5
            why_append += f" + SELL FLOW ({1 - ofi:.0%})"

        score = min(score, 100)

        if score < MIN_CONFIDENCE:
            return _skip(ticker, f"Confidence {score} < {MIN_CONFIDENCE}", current_price)

        # ---------- Risk Levels ----------
        stop_loss = current_price - (ATR_STOP_MULTIPLIER * atr_val) if signal == "BUY" else current_price + (ATR_STOP_MULTIPLIER * atr_val)
        risk = abs(current_price - stop_loss)
        tp1 = current_price + (TP1_RR_RATIO * risk) if signal == "BUY" else current_price - (TP1_RR_RATIO * risk)
        tp2 = current_price + (TP2_RR_RATIO * risk) if signal == "BUY" else current_price - (TP2_RR_RATIO * risk)
        entry_min = current_price - (0.2 * atr_val)
        entry_max = current_price + (0.2 * atr_val)
        dec = 5 if current_price < 2.0 else 2

        why_str = ", ".join(why_parts) + why_append + f". Score: {score}/100"

        # ---------- Full feature dict for ML / audit ----------
        features = {
            'price': current_price, 'ema50_15m': ema50_15m, 'ema200_15m': ema200_15m,
            'ema200_4h': ema200_4h, 'ema200_1h': ema200_1h,
            'rsi': rsi_15m, 'macd': macd_15m, 'macd_signal': macd_sig_15m,
            'adx_1h': adx_1h_val, 'atr': atr_val, 'bb_upper': bb_upper,
            'bb_lower': bb_lower, 'bb_mid': bb_mid, 'vwap': vwap_15m,
            'vol_ratio': vol_ratio, 'funding_rate': funding_rate,
            'order_flow_imbalance': ofi, 'smc_buy': smc['smc_buy'],
            'smc_sell': smc['smc_sell'], 'fresh_buy_cross': fresh_buy_cross,
            'fresh_sell_cross': fresh_sell_cross, 'close_4h': close_4h,
            'close_1h': close_1h, 'direction': direction,
        }

        signal_data = {
            'asset': ticker,
            'price': current_price,
            'signal': signal,
            'timeframe': '15m MTF (4H/1H/15m)',
            'confidence': score,
            'why': why_str,
            'entry': f"${min(entry_min, entry_max):,.{dec}f} - ${max(entry_min, entry_max):,.{dec}f}",
            'stop_loss': f"${stop_loss:,.{dec}f}",
            'tp1': f"${tp1:,.{dec}f}",
            'tp2': f"${tp2:,.{dec}f}",
            'stop_loss_raw': stop_loss,
            'tp1_raw': tp1,
            'tp2_raw': tp2,
            'atr': atr_val,
            'funding_rate': funding_rate,
            'features': features,
            'df_15m': df_15m,
        }
        return signal_data

    except Exception as e:
        import traceback
        logger.error(f"Error analyzing {ticker}: {e}\n{traceback.format_exc()}")
        return None


def _skip(ticker: str, reason: str, price: float = 0) -> None:
    logger.info(f"[SKIP] {ticker}: {reason}")
    return None
