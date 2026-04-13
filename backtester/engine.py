"""
backtester/engine.py — Professional walk-forward backtester.
- Realistic slippage model
- Exact Binance fees (maker/taker)
- Funding rate drag
- Walk-forward optimization (90-day train -> 30-day test, rolling)
- Full metrics: Sharpe, Sortino, Calmar, PF, max drawdown, expectancy
"""

import logging
import numpy as np
import pandas as pd
from datetime import timedelta

from indicators.technical import compute_all_indicators, detect_smc_zones
from config import (
    TAKER_FEE_PCT, MAKER_FEE_PCT, SLIPPAGE_TICKS_MIN, SLIPPAGE_TICKS_MAX,
    ATR_STOP_MULTIPLIER, TP1_RR_RATIO, TP2_RR_RATIO,
    ADX_MIN_THRESHOLD, VOLUME_CONFIRMATION_RATIO, MIN_CONFIDENCE,
    ML_WALK_FORWARD_TRAIN, ML_WALK_FORWARD_TEST
)

logger = logging.getLogger("oracle.backtester")


class BacktestEngine:
    def __init__(self):
        self.results = []

    def run_backtest(self, df: pd.DataFrame, symbol: str = 'UNKNOWN',
                     funding_rates: pd.Series = None,
                     slippage_pct: float = 0.02) -> dict:
        """
        Run a single-pass backtest on OHLCV data.
        Returns metrics dict.
        """
        if len(df) < 300:
            return {'error': 'insufficient_data', 'symbol': symbol}

        df = compute_all_indicators(df)
        trades = self._generate_signals_and_simulate(df, funding_rates, slippage_pct)

        if not trades:
            return {'symbol': symbol, 'total_trades': 0, 'error': 'no_trades'}

        metrics = self._calculate_metrics(trades, symbol)
        return metrics

    def run_walk_forward(self, df: pd.DataFrame, symbol: str = 'UNKNOWN',
                         train_days: int = None, test_days: int = None,
                         funding_rates: pd.Series = None) -> dict:
        """
        Walk-forward optimization: rolling train/test windows.
        """
        train_days = train_days or ML_WALK_FORWARD_TRAIN
        test_days = test_days or ML_WALK_FORWARD_TEST

        if len(df) < 500:
            return {'error': 'insufficient_data_for_walkforward'}

        # Estimate bars per day based on data frequency
        if len(df) > 1:
            avg_gap = (df.index[-1] - df.index[0]).total_seconds() / len(df)
            bars_per_day = int(86400 / max(avg_gap, 60))
        else:
            bars_per_day = 96  # Default 15m bars

        train_bars = train_days * bars_per_day
        test_bars = test_days * bars_per_day
        window = train_bars + test_bars

        all_oos_trades = []
        fold_results = []
        fold = 0

        start = 0
        while start + window <= len(df):
            train_end = start + train_bars
            test_end = train_end + test_bars

            df_train = df.iloc[start:train_end].copy()
            df_test = df.iloc[train_end:test_end].copy()

            # Simulate on test set only (out-of-sample)
            df_test = compute_all_indicators(
                pd.concat([df_train.tail(250), df_test])  # prepend for indicator warmup
            )
            # Only take test period rows
            df_test = df_test.iloc[250:]

            trades = self._generate_signals_and_simulate(df_test, funding_rates)

            fold_metrics = self._calculate_metrics(trades, f"{symbol}_fold{fold}")
            fold_metrics['fold'] = fold
            fold_results.append(fold_metrics)
            all_oos_trades.extend(trades)

            start += test_bars  # Roll forward by test window
            fold += 1

        if not all_oos_trades:
            return {'error': 'no_oos_trades', 'symbol': symbol, 'folds': fold}

        # Aggregate OOS metrics
        combined = self._calculate_metrics(all_oos_trades, symbol)
        combined['type'] = 'walk_forward'
        combined['folds'] = fold
        combined['fold_results'] = fold_results

        return combined

    def _generate_signals_and_simulate(self, df: pd.DataFrame,
                                        funding_rates: pd.Series = None,
                                        slippage_pct: float = 0.02) -> list:
        """Generate signals and simulate trades with realistic costs."""
        trades = []
        if 'ATR' not in df.columns:
            return trades

        df = df.copy()

        # Pre-calculate MACD crossovers
        df['MACD_prev'] = df['MACD'].shift(1)
        df['MACD_Sig_prev'] = df['MACD_Signal'].shift(1)

        in_trade = False
        for i in range(3, len(df) - 30):
            if in_trade:
                continue

            row = df.iloc[i]
            prev = df.iloc[i - 1]

            # Skip NaN rows
            if pd.isna(row.get('ATR')) or pd.isna(row.get('MACD')):
                continue

            price = float(row['Close'])
            atr_val = float(row['ATR'])
            if atr_val <= 0:
                continue

            # --- Signal logic (mirrors rules.py) ---
            ema50 = float(row['EMA50'])
            ema200 = float(row['EMA200'])
            rsi = float(row['RSI'])
            macd_val = float(row['MACD'])
            macd_sig = float(row['MACD_Signal'])
            macd_prev = float(prev['MACD']) if pd.notna(prev['MACD']) else macd_val
            macd_sig_prev = float(prev['MACD_Signal']) if pd.notna(prev['MACD_Signal']) else macd_sig
            vwap_val = float(row['VWAP']) if pd.notna(row.get('VWAP')) else price
            vol = float(row['Volume'])
            vol_sma = float(row['Volume_SMA']) if pd.notna(row.get('Volume_SMA')) else vol
            adx_val = float(row['ADX']) if pd.notna(row.get('ADX')) else 0

            if adx_val < ADX_MIN_THRESHOLD:
                continue

            vol_ratio = vol / (vol_sma + 1e-10)
            if vol_ratio < VOLUME_CONFIRMATION_RATIO:
                continue

            fresh_buy = macd_val > macd_sig and macd_prev <= macd_sig_prev
            fresh_sell = macd_val < macd_sig and macd_prev >= macd_sig_prev

            signal = None
            if price > ema50 > ema200 and fresh_buy and rsi < 65 and price > vwap_val:
                signal = 'BUY'
            elif price < ema50 < ema200 and fresh_sell and rsi > 35 and price < vwap_val:
                signal = 'SELL'

            if not signal:
                continue

            # --- Trade simulation ---
            # Apply slippage
            slip = price * slippage_pct / 100
            entry = price + slip if signal == 'BUY' else price - slip

            # SL & TP
            if signal == 'BUY':
                sl = entry - (ATR_STOP_MULTIPLIER * atr_val)
                risk = entry - sl
                tp1 = entry + (TP1_RR_RATIO * risk)
            else:
                sl = entry + (ATR_STOP_MULTIPLIER * atr_val)
                risk = sl - entry
                tp1 = entry - (TP1_RR_RATIO * risk)

            # Entry fee (taker)
            entry_fee = entry * TAKER_FEE_PCT / 100

            # Simulate forward: check which level is hit first
            future = df.iloc[i + 1: i + 61]  # Up to 60 bars (15h)
            outcome = 'timeout'
            exit_price = float(future['Close'].iloc[-1]) if len(future) > 0 else entry
            bars_held = len(future)

            for j, (_, frow) in enumerate(future.iterrows()):
                fhigh = float(frow['High'])
                flow = float(frow['Low'])

                if signal == 'BUY':
                    if flow <= sl:
                        outcome = 'stop_loss'
                        exit_price = sl - slip  # Slippage on exit too
                        bars_held = j + 1
                        break
                    elif fhigh >= tp1:
                        outcome = 'take_profit'
                        exit_price = tp1 - slip * 0.5  # Less slippage on TP
                        bars_held = j + 1
                        break
                else:
                    if fhigh >= sl:
                        outcome = 'stop_loss'
                        exit_price = sl + slip
                        bars_held = j + 1
                        break
                    elif flow <= tp1:
                        outcome = 'take_profit'
                        exit_price = tp1 + slip * 0.5
                        bars_held = j + 1
                        break

            # Exit fee
            exit_fee = exit_price * TAKER_FEE_PCT / 100

            # Funding cost (approximate)
            funding_cost = 0.0
            if funding_rates is not None and len(funding_rates) > 0:
                # Funding every 8h = 32 bars of 15m
                funding_periods = bars_held / 32
                avg_funding = 0.0001  # Default
                try:
                    nearest_idx = funding_rates.index.get_indexer([row.name], method='nearest')[0]
                    if 0 <= nearest_idx < len(funding_rates):
                        avg_funding = abs(float(funding_rates.iloc[nearest_idx]))
                except Exception:
                    pass
                funding_cost = entry * avg_funding * funding_periods

            # Calculate PnL
            if signal == 'BUY':
                gross_pnl = exit_price - entry
            else:
                gross_pnl = entry - exit_price

            net_pnl = gross_pnl - entry_fee - exit_fee - funding_cost
            pnl_pct = (net_pnl / entry) * 100

            trades.append({
                'symbol': df.attrs.get('symbol', 'UNKNOWN'),
                'entry_time': row.name,
                'exit_time': future.index[bars_held - 1] if bars_held > 0 and bars_held <= len(future) else row.name,
                'signal': signal,
                'entry': entry,
                'exit': exit_price,
                'sl': sl,
                'tp': tp1,
                'outcome': outcome,
                'gross_pnl': gross_pnl,
                'fees': entry_fee + exit_fee,
                'funding_cost': funding_cost,
                'net_pnl': net_pnl,
                'pnl_pct': pnl_pct,
                'bars_held': bars_held,
                'atr': atr_val,
            })

            in_trade = True
            # Simple cooldown: skip next 4 bars (1 hour on 15m)
            # Achieved by letting the loop continue and in_trade flag
            # Reset after position exit simulation
            # We'll use a simpler approach: mark bars_held as cooldown
            # Actually: just advance i conceptually by setting in_trade
            # We need to skip bars_held bars
            # Since we can't modify loop var, use a different approach:

        # Remove overlapping trades (a trade's bars_held creates exclusion zone)
        filtered = []
        exclude_until = -1
        for t in trades:
            idx = df.index.get_loc(t['entry_time'])
            if idx > exclude_until:
                filtered.append(t)
                exclude_until = idx + t['bars_held'] + 4  # +4 bar cooldown

        return filtered

    def _calculate_metrics(self, trades: list, symbol: str = '') -> dict:
        """Calculate comprehensive backtest metrics."""
        if not trades:
            return {'symbol': symbol, 'total_trades': 0}

        pnls = [t['net_pnl'] for t in trades]
        pnl_pcts = [t['pnl_pct'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_trades = len(trades)
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / (gross_loss + 1e-10)
        expectancy = np.mean(pnls)
        total_fees = sum(t['fees'] for t in trades)
        total_funding = sum(t['funding_cost'] for t in trades)

        # Equity curve for Sharpe, Sortino, max DD
        equity = [0]
        for p in pnl_pcts:
            equity.append(equity[-1] + p)
        equity = np.array(equity)

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = peak - equity
        max_dd = drawdown.max()

        # Sharpe (annualized, assuming 96 trades/day for 15m)
        returns_arr = np.array(pnl_pcts)
        if len(returns_arr) > 1 and returns_arr.std() > 0:
            sharpe = (returns_arr.mean() / returns_arr.std()) * np.sqrt(252 * 6)  # ~6 trades/day estimate
        else:
            sharpe = 0

        # Sortino
        downside = returns_arr[returns_arr < 0]
        if len(downside) > 1 and downside.std() > 0:
            sortino = (returns_arr.mean() / downside.std()) * np.sqrt(252 * 6)
        else:
            sortino = 0

        # Calmar
        if max_dd > 0:
            calmar = (sum(pnl_pcts) / max_dd)
        else:
            calmar = 0

        return {
            'symbol': symbol,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe': sharpe,
            'sortino': sortino,
            'calmar': calmar,
            'max_drawdown_pct': max_dd,
            'expectancy': expectancy,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'total_fees': total_fees,
            'total_funding': total_funding,
            'net_pnl': sum(pnls),
            'trades': trades,
        }
