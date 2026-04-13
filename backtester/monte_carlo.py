"""
backtester/monte_carlo.py — Monte Carlo simulation (10,000 runs).
Shuffles trade sequence to estimate tail risk and confidence intervals.
"""

import logging
import numpy as np

from config import MONTE_CARLO_RUNS

logger = logging.getLogger("oracle.monte_carlo")


def run_monte_carlo(trades: list, n_runs: int = None,
                    initial_equity: float = 10000) -> dict:
    """
    Run Monte Carlo simulation by randomly shuffling trade order.
    Returns distribution of outcomes.
    """
    n_runs = n_runs or MONTE_CARLO_RUNS

    if not trades or len(trades) < 10:
        return {'error': 'insufficient_trades', 'n_trades': len(trades)}

    pnl_pcts = np.array([t['pnl_pct'] for t in trades])
    n_trades = len(pnl_pcts)

    final_equities = []
    max_drawdowns = []
    sharpe_ratios = []

    for _ in range(n_runs):
        # Shuffle trade order
        shuffled = np.random.permutation(pnl_pcts)

        # Build equity curve
        equity = initial_equity
        peak = equity
        max_dd = 0.0
        equity_curve = [equity]

        for pnl in shuffled:
            equity *= (1 + pnl / 100)
            equity_curve.append(equity)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd

        final_equities.append(equity)
        max_drawdowns.append(max_dd)

        # Sharpe on this run
        returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
        if returns.std() > 0:
            sharpe_ratios.append(returns.mean() / returns.std() * np.sqrt(252 * 6))
        else:
            sharpe_ratios.append(0)

    final_equities = np.array(final_equities)
    max_drawdowns = np.array(max_drawdowns)
    sharpe_ratios = np.array(sharpe_ratios)

    # Probability of ruin (equity drops below 50% of initial)
    ruin_threshold = initial_equity * 0.5
    prob_ruin = (final_equities < ruin_threshold).mean() * 100

    # Probability of profit
    prob_profit = (final_equities > initial_equity).mean() * 100

    results = {
        'n_runs': n_runs,
        'n_trades': n_trades,
        'initial_equity': initial_equity,

        # Final equity distribution
        'equity_mean': float(np.mean(final_equities)),
        'equity_median': float(np.median(final_equities)),
        'equity_p5': float(np.percentile(final_equities, 5)),
        'equity_p25': float(np.percentile(final_equities, 25)),
        'equity_p75': float(np.percentile(final_equities, 75)),
        'equity_p95': float(np.percentile(final_equities, 95)),
        'equity_std': float(np.std(final_equities)),

        # Max drawdown distribution
        'max_dd_mean': float(np.mean(max_drawdowns)),
        'max_dd_median': float(np.median(max_drawdowns)),
        'max_dd_p95': float(np.percentile(max_drawdowns, 95)),
        'max_dd_worst': float(np.max(max_drawdowns)),

        # Sharpe distribution
        'sharpe_mean': float(np.mean(sharpe_ratios)),
        'sharpe_median': float(np.median(sharpe_ratios)),
        'sharpe_p5': float(np.percentile(sharpe_ratios, 5)),

        # Risk metrics
        'prob_profit_pct': float(prob_profit),
        'prob_ruin_pct': float(prob_ruin),
    }

    logger.info(
        f"Monte Carlo ({n_runs} runs, {n_trades} trades): "
        f"Median equity=${results['equity_median']:,.0f}, "
        f"P5 DD={results['max_dd_p95']:.1f}%, "
        f"Prob profit={prob_profit:.1f}%, "
        f"Sharpe median={results['sharpe_median']:.2f}"
    )

    return results
