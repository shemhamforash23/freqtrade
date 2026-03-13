"""
Monte Carlo driver class for MCPT.

Coordinates permutation testing process using permuters and validators.
Supports multiple methods running sequentially.
"""

import logging
from typing import Any

import numpy as np
from pandas import DataFrame

from freqtrade.enums.mcptstate import MCPTState
from freqtrade.optimize.montecarlo.mc_progress import MCProgress
from freqtrade.optimize.montecarlo.mcpt_types import (
    MCPTResult,
    calculate_p_value,
)
from freqtrade.optimize.montecarlo.permuters.bar_permute import BarPermutePermuter
from freqtrade.optimize.montecarlo.permuters.base_permuter import IPermuter
from freqtrade.optimize.montecarlo.permuters.trade_shuffle import TradeShufflePermuter


logger = logging.getLogger(__name__)

# Process this many permutations per vectorized batch (balances memory vs. loop overhead)
_BATCH_CHUNK = 512


class MonteCarlo:
    """
    Monte Carlo driver for permutation testing.

    Supports multiple MCPT methods running sequentially:
    - trade_shuffle: Fast P&L shuffling (GPU-accelerated when CUDA available)
    - bar_permute: Block bootstrap permutation of OHLCV bars
    """

    def __init__(self, config: dict[str, Any], method_config: dict[str, Any] | None = None):
        """
        Initialize MonteCarlo with configuration.

        :param config: Freqtrade configuration dictionary
        :param method_config: Specific method configuration (from methods array)
        """
        self.config = config
        self.mcpt_config = config.get("mcpt", {})
        self.method_config = method_config or {}
        self.progress = MCProgress()

        # Load permuter based on method_config if provided
        if method_config:
            self.permuter = self._load_permuter_for_method(method_config)
        else:
            self.permuter = None

    @classmethod
    def get_enabled_methods(cls, config: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Get list of enabled MCPT methods from configuration.

        :param config: Freqtrade configuration dictionary
        :return: List of enabled method configurations
        """
        mcpt_config = config.get("mcpt", {})
        methods = mcpt_config.get("methods", [])
        global_seed = mcpt_config.get("seed")

        enabled_methods = []
        for method in methods:
            if method.get("enabled", True):
                # Merge global seed if not overridden
                if global_seed is not None and "seed" not in method:
                    method = {**method, "seed": global_seed}
                enabled_methods.append(method)

        return enabled_methods

    @staticmethod
    def method_requires_backtest_rerun(method_name: str) -> bool:
        """Check if a method requires backtest re-run."""
        return method_name in ("bar_permute", "block_bootstrap")

    def _load_permuter_for_method(self, method_config: dict[str, Any]) -> IPermuter:
        """
        Load permuter for a specific method configuration.

        :param method_config: Method configuration dict
        :return: Configured permuter instance
        """
        method_name = method_config.get("name", "trade_shuffle")
        scope = method_config.get("scope", "global")
        block_size = method_config.get("block_size", 20)

        if method_name == "trade_shuffle":
            return TradeShufflePermuter(scope=scope)
        elif method_name == "bar_permute":
            return BarPermutePermuter(block_size=block_size)
        else:
            raise ValueError(f"Unsupported MCPT method: {method_name}")

    def validate(
        self,
        trades: DataFrame,
        backtest_stats: dict[str, Any],
        method_config: dict[str, Any] | None = None,
    ) -> MCPTResult:
        """
        Run MCPT validation on backtest results.

        :param trades: DataFrame with backtest trades
        :param backtest_stats: Dictionary with backtest statistics
        :param method_config: Method configuration (uses self.method_config if None)
        :return: MCPTResult with p-value and details
        """
        cfg = method_config or self.method_config
        method_name = cfg.get("name", "trade_shuffle")
        metric = cfg.get("metric", "profit_total")
        n_runs = cfg.get("runs", 1000)
        seed = cfg.get("seed", self.mcpt_config.get("seed"))

        logger.info(f"MCPT [{method_name}]: Starting validation - {n_runs} runs, metric={metric}")

        # Warn about uninformative combinations
        if method_name == "trade_shuffle" and metric in ("profit_total", "profit_total_abs"):
            logger.warning(
                f"MCPT: trade_shuffle with metric={metric} is uninformative! "
                "Trade shuffle preserves total profit (only changes order), "
                "so p-value will always be ~1.0. "
                "Consider using 'profit_factor', 'max_drawdown', or 'sharpe' instead."
            )

        # Load permuter if not already loaded
        if self.permuter is None or method_config:
            self.permuter = self._load_permuter_for_method(cfg)

        # Extract real metric value from backtest stats
        metric_real = self._extract_metric(backtest_stats, metric)

        # Run permutation tests
        permuted_metrics = self._run_permutations(trades, metric, n_runs, seed)

        # Calculate p-value
        p_value, count_better = calculate_p_value(metric_real, permuted_metrics)

        result = MCPTResult(
            method_name=method_name,
            metric_name=metric,
            metric_real=metric_real,
            metric_permuted=permuted_metrics,
            p_value=p_value,
            n_permutations=n_runs,
            count_better=count_better,
        )

        self._log_result(result)

        return result

    def _extract_metric(self, stats: dict[str, Any], metric: str) -> float:
        """
        Extract metric value from backtest statistics.

        :param stats: Backtest statistics dictionary
        :param metric: Metric name
        :return: Metric value
        """
        metric_mapping = {
            "profit_total": "profit_total",
            "profit_total_abs": "profit_total_abs",
            "sharpe": "sharpe",
            "sortino": "sortino",
            "calmar": "calmar",
            "profit_factor": "profit_factor",
            "max_drawdown": "max_drawdown_abs",
        }

        stat_key = metric_mapping.get(metric, metric)
        value = stats.get(stat_key, 0.0)

        return float(value) if value is not None else 0.0

    def _run_permutations(
        self,
        trades: DataFrame,
        metric: str,
        n_runs: int,
        seed: int | None = None,
    ) -> list[float]:
        """
        Run permutation tests and collect metric values.

        Uses a vectorized fast path for TradeShufflePermuter (global scope):
        generates all N shuffled arrays at once and computes metrics directly
        on numpy arrays, avoiding N DataFrame copies and N dict lookups.

        :param trades: Original trades DataFrame
        :param metric: Metric to calculate
        :param n_runs: Number of permutation runs
        :param seed: Random seed for reproducibility
        :return: List of metric values from permuted data
        """
        self.progress.init_step(MCPTState.PERMUTATION_TESTS, n_runs)
        permuted_metrics: list[float] = []

        # Fast path: TradeShufflePermuter global scope — skip DataFrame overhead entirely
        if (
            isinstance(self.permuter, TradeShufflePermuter)
            and self.permuter.scope == "global"
            and not self.permuter._use_jax
        ):
            profit_arr: np.ndarray = np.asarray(trades["profit_abs"].values, dtype=np.float64)
            n = len(profit_arr)
            rng = np.random.default_rng(seed)
            # All N shuffled arrays in one vectorized call: (n_runs, n_trades)
            indices_matrix = np.tile(np.arange(n), (n_runs, 1))
            rng.permuted(indices_matrix, axis=1, out=indices_matrix)
            shuffled_batch = profit_arr[indices_matrix]
            # Chunked vectorized metric computation — no Python loop over rows
            for start in range(0, n_runs, _BATCH_CHUNK):
                chunk = shuffled_batch[start : start + _BATCH_CHUNK]
                permuted_metrics.extend(self._calculate_batch_metrics(chunk, metric))
                for _ in range(len(chunk)):
                    self.progress.increment()
        else:
            for permuted_trades in self.permuter.permute_batch(trades, n_runs, base_seed=seed):
                permuted_metrics.append(self._calculate_trade_metric(permuted_trades, metric))
                self.progress.increment()

        return permuted_metrics

    def _calculate_trade_metric(self, trades: DataFrame, metric: str) -> float:
        """
        Calculate metric from trades DataFrame.

        :param trades: Trades DataFrame
        :param metric: Metric name
        :return: Metric value
        """
        if len(trades) == 0:
            return 0.0

        profit_abs = np.asarray(trades["profit_abs"].values, dtype=np.float64)

        if metric in ("profit_total", "profit_total_abs"):
            return float(profit_abs.sum())

        elif metric == "profit_factor":
            wins = profit_abs[profit_abs > 0].sum()
            losses = abs(profit_abs[profit_abs < 0].sum())
            return wins / losses if losses > 0 else float("inf")

        elif metric == "max_drawdown":
            # Calculate equity curve and max drawdown
            equity_curve = profit_abs.cumsum()
            running_max = self._running_max(equity_curve)
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

        else:
            # Default: return total profit
            return float(profit_abs.sum())

    def _calculate_batch_metrics(self, batch: np.ndarray, metric: str) -> list[float]:
        """
        Compute metric for every row of a 2-D batch in one vectorized pass.

        This eliminates the Python loop over individual permutation rows inside
        the fast path of _run_permutations. Peak memory is bounded by chunk size.

        :param batch: (n, n_trades) float64 array of per-permutation profit_abs rows
        :param metric: Metric name
        :return: List of n metric values
        """
        if metric in ("profit_total", "profit_total_abs"):
            return batch.sum(axis=1).tolist()

        elif metric == "profit_factor":
            pos = np.where(batch > 0, batch, 0.0).sum(axis=1)
            neg = np.where(batch < 0, -batch, 0.0).sum(axis=1)
            return np.where(neg > 0, pos / neg, np.inf).tolist()

        elif metric == "sharpe":
            means = batch.mean(axis=1)
            stds = batch.std(axis=1)
            return np.where(stds > 0, means / stds, 0.0).tolist()

        elif metric == "sortino":
            means = batch.mean(axis=1)
            neg_only = np.where(batch < 0, batch, np.nan)
            downside_std = np.nanstd(neg_only, axis=1)
            return np.where(downside_std > 0, means / downside_std, 0.0).tolist()

        elif metric == "max_drawdown":
            equity = np.cumsum(batch, axis=1)
            running_max = np.maximum.accumulate(equity, axis=1)
            return (running_max - equity).max(axis=1).tolist()

        else:
            return batch.sum(axis=1).tolist()

    def _calculate_array_metric(self, profit_abs: np.ndarray, metric: str) -> float:
        """
        Calculate metric directly from a profit_abs ndarray.

        Mirrors _calculate_trade_metric but operates on a raw numpy array,
        avoiding DataFrame construction and column lookup overhead in the hot loop.

        :param profit_abs: 1-D array of per-trade P&L values
        :param metric: Metric name
        :return: Metric value
        """
        if len(profit_abs) == 0:
            return 0.0

        if metric in ("profit_total", "profit_total_abs"):
            return float(profit_abs.sum())

        elif metric == "profit_factor":
            wins = profit_abs[profit_abs > 0].sum()
            losses = abs(profit_abs[profit_abs < 0].sum())
            return wins / losses if losses > 0 else float("inf")

        elif metric == "max_drawdown":
            equity_curve = profit_abs.cumsum()
            running_max = self._running_max(equity_curve)
            drawdown = running_max - equity_curve
            return float(drawdown.max())

        elif metric == "sharpe":
            if len(profit_abs) < 2:
                return 0.0
            std = profit_abs.std()
            return float(profit_abs.mean() / std) if std > 0 else 0.0

        elif metric == "sortino":
            if len(profit_abs) < 2:
                return 0.0
            downside = profit_abs[profit_abs < 0]
            downside_std = downside.std() if len(downside) > 0 else 0.0
            return float(profit_abs.mean() / downside_std) if downside_std > 0 else 0.0

        else:
            return float(profit_abs.sum())

    def _running_max(self, arr: np.ndarray) -> np.ndarray:
        """Calculate running maximum of array."""
        return np.maximum.accumulate(arr)

    def _log_result(self, result: MCPTResult) -> None:
        """
        Log MCPT result with interpretation.

        :param result: MCPTResult to log
        """
        if result.p_value < 0.01:
            significance = "Very Strong (p < 0.01)"
            symbol = "✓"
        elif result.p_value < 0.05:
            significance = "Significant (p < 0.05)"
            symbol = "✓"
        elif result.p_value < 0.10:
            significance = "Marginal (p < 0.10)"
            symbol = "~"
        else:
            significance = "Not Significant (p >= 0.10)"
            symbol = "✗"

        logger.info(
            f"MCPT [{result.method_name}]: {symbol} {significance}\n"
            f"  Metric: {result.metric_name} = {result.metric_real:.4f}\n"
            f"  p-value: {result.p_value:.4f}\n"
            f"  Permutations: {result.n_permutations} "
            f"(better: {result.count_better})"
        )
