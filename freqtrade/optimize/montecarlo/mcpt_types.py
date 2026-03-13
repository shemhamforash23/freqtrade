"""
MCPT result types and utility functions.

This module provides data structures for Monte Carlo Permutation Test results
and core utility functions for p-value calculation and metric computation.
"""

import math
from dataclasses import dataclass

import numpy as np
from pandas import DataFrame


@dataclass
class MCPTResult:
    """Result of a single MCPT run."""

    method_name: str
    metric_name: str
    metric_real: float
    metric_permuted: list[float]
    p_value: float
    n_permutations: int
    count_better: int


@dataclass
class MCPTSummary:
    """Aggregated MCPT results across metrics/windows."""

    results: list[MCPTResult]
    aggregated_p_value: float | None
    aggregation_method: str | None
    interpretation: str


@dataclass
class MCPTWorkerResult:
    """Result from a single MCPT worker process."""

    run_index: int
    metric_value: float
    metric_name: str
    n_trades: int
    profit_total: float
    elapsed: float


def calculate_p_value(
    metric_real: float,
    metric_permuted: list[float],
    alternative: str = "greater",
) -> tuple[float, int]:
    """
    Calculate p-value using standard MCPT formula.

    The p-value represents the probability of observing a metric value
    as extreme as the real value under the null hypothesis (random trading).

    :param metric_real: Real metric value from the original strategy
    :param metric_permuted: List of metric values from permuted runs
    :param alternative: 'greater' (test if real > permuted) or 'less'
    :return: (p_value, count_better) tuple
    """
    n = len(metric_permuted)

    if alternative == "greater":
        count_better = sum(1 for p in metric_permuted if p >= metric_real)
    elif alternative == "less":
        count_better = sum(1 for p in metric_permuted if p <= metric_real)
    else:
        raise ValueError(f"Invalid alternative: {alternative}")

    # Standard MCPT formula: (count_better + 1) / (n + 1)
    # The +1 accounts for the original observation
    p_value = (count_better + 1) / (n + 1)

    return p_value, count_better


def calculate_mcpt_metric(trades: DataFrame, metric: str, start_balance: float) -> float:  # noqa: C901
    """
    Calculate metric from trades DataFrame for MCPT.

    Shared helper function used by both sequential and parallel MCPT execution
    to ensure consistent metric calculation.

    :param trades: DataFrame with trade results (must have 'profit_abs' column)
    :param metric: Metric name to calculate
    :param start_balance: Starting balance for relative profit calculation
    :return: Calculated metric value
    """
    if len(trades) == 0:
        return 0.0

    profit_abs: np.ndarray = np.asarray(trades["profit_abs"].values, dtype=np.float64)

    if metric in ("profit_total",):
        # Relative profit (same as in generate_strategy_stats)
        total_abs = float(profit_abs.sum())
        return total_abs / start_balance if start_balance > 0 else 0.0

    elif metric == "profit_total_abs":
        return float(profit_abs.sum())

    elif metric == "profit_factor":
        wins = profit_abs[profit_abs > 0].sum()
        losses = abs(profit_abs[profit_abs < 0].sum())
        return wins / losses if losses > 0 else float("inf")

    elif metric == "max_drawdown":
        # Calculate equity curve and max drawdown
        equity_curve = profit_abs.cumsum()
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = running_max - equity_curve
        return float(drawdown.max())

    elif metric == "sharpe":
        # Simplified Sharpe: mean / std
        if len(profit_abs) < 2:
            return 0.0
        std = profit_abs.std()
        return float(profit_abs.mean() / std) if std > 0 else 0.0

    elif metric == "sortino":
        # Simplified Sortino: mean / downside_std
        if len(profit_abs) < 2:
            return 0.0
        downside = profit_abs[profit_abs < 0]
        downside_std = downside.std() if len(downside) > 0 else 0.0
        return float(profit_abs.mean() / downside_std) if downside_std > 0 else 0.0

    elif metric == "calmar":
        # Mirrors freqtrade's calculate_calmar(): CAGR-like return / max relative drawdown
        if start_balance <= 0:
            return 0.0
        total_profit = float(profit_abs.sum()) / start_balance

        # Derive time period from trade close dates when available
        days_period = 0
        if "close_date" in trades.columns and len(trades) >= 2:
            min_date = trades["close_date"].min()
            max_date = trades["close_date"].max()
            if min_date != max_date:
                days_period = max(1, (max_date - min_date).days)

        if days_period == 0:
            return 0.0

        expected_returns_mean = total_profit / days_period * 100

        # Max relative drawdown from account equity curve
        equity = profit_abs.cumsum() + start_balance
        running_max = np.maximum.accumulate(equity)
        rel_drawdown = (
            float(((running_max - equity) / running_max).max()) if running_max.max() > 0 else 0.0
        )

        if rel_drawdown != 0:
            return float(expected_returns_mean / rel_drawdown * math.sqrt(365))
        else:
            return -100.0  # convention from freqtrade: no drawdown is flagged as not optimal

    else:
        # Default: return relative profit
        total_abs = float(profit_abs.sum())
        return total_abs / start_balance if start_balance > 0 else 0.0
