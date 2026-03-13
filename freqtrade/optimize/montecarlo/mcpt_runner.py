"""
MCPT Runner for parallel permutation testing.

This class is sent to worker processes via joblib.Parallel.
Follows the same pattern as HyperOptimizer for cross-platform compatibility.
"""

import logging
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from joblib import delayed, dump, load, wrap_non_picklable_objects
from pandas import DataFrame

from freqtrade.data import history
from freqtrade.data.converter import trim_dataframes
from freqtrade.optimize.montecarlo.mcpt_types import MCPTWorkerResult, calculate_mcpt_metric
from freqtrade.persistence import CustomDataWrapper, PairLocks, Trade
from freqtrade.util.dry_run_wallet import get_dry_run_wallet


logger = logging.getLogger(__name__)


class MCPTRunner:
    """
    MCPT Runner class for parallel backtest execution.

    This class encapsulates everything needed to run a permuted backtest
    and can be serialized/sent to worker processes via joblib.

    Follows the HyperOptimizer pattern for cross-platform compatibility.
    """

    def __init__(
        self,
        config: dict[str, Any],
        timerange,
        required_startup: int,
        timeframe: str,
        timeframe_detail: str | None = None,
    ) -> None:
        """
        Initialize MCPTRunner.

        Note: Backtesting instance is created lazily in worker processes
        to avoid serialization issues.

        :param config: Freqtrade configuration
        :param timerange: TimeRange for backtesting
        :param required_startup: Required startup candles
        :param timeframe: Main timeframe
        :param timeframe_detail: Detail timeframe (optional)
        """
        self.config = config
        self.timerange = timerange
        self.required_startup = required_startup
        self.timeframe = timeframe
        self.timeframe_detail = timeframe_detail

        # Backtesting instance - created lazily in worker
        self._backtesting = None
        self._strategy = None

        # Temporary directory for data exchange
        self._temp_dir = Path(tempfile.gettempdir()) / "freqtrade_mcpt"
        self._temp_dir.mkdir(exist_ok=True)

    def _get_backtesting(self):
        """
        Get or create Backtesting instance.

        Creates instance lazily to avoid serialization issues.
        Each worker process gets its own instance.
        """
        if self._backtesting is None:
            # Import here to avoid circular imports
            from freqtrade.optimize.backtesting import Backtesting

            self._backtesting = Backtesting(self.config)
            self._backtesting._set_strategy(self._backtesting.strategylist[0])
            self._strategy = self._backtesting.strategy

            # Close exchange - not needed for backtesting
            self._backtesting.exchange.close()
            self._backtesting.exchange._api = None
            self._backtesting.exchange._api_async = None

        return self._backtesting, self._strategy

    @delayed
    @wrap_non_picklable_objects
    def run_permuted_backtest(
        self,
        run_index: int,
        data_file: str,
        detail_file: str | None,
        metric: str,
    ) -> MCPTWorkerResult | None:
        """
        Run a single backtest with permuted data.

        This method is decorated for joblib parallel execution.
        It loads permuted data from pickle files and runs backtest.

        :param run_index: Index of this permutation run
        :param data_file: Path to pickle file with permuted OHLCV data
        :param detail_file: Path to pickle file with permuted detail data (optional)
        :param metric: Metric to extract from results
        :return: Metric value or None if failed
        """
        try:
            t_start = time.time()

            # Reset global state in this worker process
            PairLocks.reset_locks()
            Trade.reset_trades()
            CustomDataWrapper.reset_custom_data()

            # Get/create backtesting instance
            bt, strategy = self._get_backtesting()
            bt.dataprovider.clear_cache()

            # Load permuted data from pickle file
            permuted_data = load(data_file, mmap_mode="r")

            # Load detail data if provided
            permuted_detail = None
            if detail_file:
                permuted_detail = load(detail_file, mmap_mode="r")

            # Calculate indicators for permuted data
            processed = {}
            for pair, df in permuted_data.items():
                # Convert from memmap to regular DataFrame if needed
                if hasattr(df, "copy"):
                    df = df.copy()
                processed[pair] = strategy.advise_indicators(df, {"pair": pair})

            # Trim startup period
            processed_trimmed = trim_dataframes(processed, self.timerange, self.required_startup)

            # Get timerange from trimmed data
            min_date, max_date = history.get_timerange(processed_trimmed)

            # Set detail data if provided
            saved_detail = bt.detail_data
            if permuted_detail:
                bt.detail_data = dict(permuted_detail)  # Convert from memmap

            try:
                # Run backtest
                results = bt.backtest(processed_trimmed, min_date, max_date)
            finally:
                # Restore detail data
                bt.detail_data = saved_detail

            # Extract metric
            trades = results.get("results", [])
            elapsed = time.time() - t_start
            if len(trades) == 0:
                return MCPTWorkerResult(
                    run_index=run_index,
                    metric_value=0.0,
                    metric_name=metric,
                    n_trades=0,
                    profit_total=0.0,
                    elapsed=elapsed,
                )

            # Calculate metric using shared helper
            start_balance = get_dry_run_wallet(bt.config)
            metric_value = calculate_mcpt_metric(trades, metric, start_balance)

            # Collect metadata for logging in main process
            n_trades = len(trades)
            profit_total = trades["profit_abs"].sum() / start_balance if start_balance > 0 else 0.0
            elapsed = time.time() - t_start

            return MCPTWorkerResult(
                run_index=run_index,
                metric_value=metric_value,
                metric_name=metric,
                n_trades=n_trades,
                profit_total=profit_total,
                elapsed=elapsed,
            )

        except Exception as e:
            logger.error(f"MCPT Worker [{run_index + 1}] failed: {e}")
            traceback.print_exc()
            return None

    def save_permuted_data(
        self,
        run_index: int,
        permuted_data: dict[str, DataFrame],
        permuted_detail: dict[str, DataFrame] | None = None,
    ) -> tuple[str, str | None]:
        """
        Save permuted data to temporary pickle files.

        :param run_index: Index of this permutation run
        :param permuted_data: Permuted OHLCV data
        :param permuted_detail: Permuted detail data (optional)
        :return: Tuple of (data_file_path, detail_file_path or None)
        """
        data_file = self._temp_dir / f"mcpt_data_{run_index}.pkl"
        dump(permuted_data, data_file)

        detail_file = None
        if permuted_detail:
            detail_file = self._temp_dir / f"mcpt_detail_{run_index}.pkl"
            dump(permuted_detail, detail_file)

        return str(data_file), str(detail_file) if detail_file else None

    def cleanup_temp_files(self, run_indices: list[int]) -> None:
        """
        Clean up temporary pickle files after processing.

        :param run_indices: List of run indices to clean up
        """
        for idx in run_indices:
            data_file = self._temp_dir / f"mcpt_data_{idx}.pkl"
            detail_file = self._temp_dir / f"mcpt_detail_{idx}.pkl"

            if data_file.exists():
                data_file.unlink()
            if detail_file.exists():
                detail_file.unlink()
