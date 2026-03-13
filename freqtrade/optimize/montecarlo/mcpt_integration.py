"""
MCPT Integration for Backtesting.

This module contains the integration layer between Backtesting and MCPT.
Extracted from backtesting.py to keep that file focused on core backtesting logic.
"""

import logging
import multiprocessing
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from joblib import Parallel, cpu_count
from pandas import DataFrame

from freqtrade.data import history
from freqtrade.data.converter import trim_dataframes
from freqtrade.enums.mcptstate import MCPTState
from freqtrade.optimize.montecarlo.mcpt_runner import MCPTRunner
from freqtrade.optimize.montecarlo.mcpt_types import MCPTResult, MCPTWorkerResult, calculate_p_value
from freqtrade.optimize.montecarlo.montecarlo import MonteCarlo
from freqtrade.util.dry_run_wallet import get_dry_run_wallet


if TYPE_CHECKING:
    from freqtrade.optimize.backtesting import Backtesting

logger = logging.getLogger(__name__)


class MCPTIntegration:
    """
    Monte Carlo Permutation Test integration for Backtesting.

    This class encapsulates all MCPT-related functionality that was previously
    in the Backtesting class. It handles:
    - Running MCPT validation on backtest results
    - Trade shuffle method (fast, post-backtest)
    - Bar permute method (slow, requires backtest re-run)
    - Sequential and parallel execution
    - Result storage
    """

    def __init__(self, backtesting: "Backtesting") -> None:
        """
        Initialize MCPTIntegration.

        :param backtesting: Backtesting instance to integrate with
        """
        self.bt = backtesting
        self.config = backtesting.config

    def run(self, data: dict[str, DataFrame] | None = None) -> None:
        """
        Run Monte Carlo Permutation Test validation on backtest results.

        Runs all enabled MCPT methods sequentially as configured in config["mcpt"]["methods"].

        :param data: Original OHLCV data (required for bar_permute method)
        """
        logger.info("Starting MCPT validation...")

        # Get list of enabled methods
        enabled_methods = MonteCarlo.get_enabled_methods(self.config)

        if not enabled_methods:
            logger.warning("MCPT: No methods enabled, skipping validation")
            return

        logger.info(
            f"MCPT: Running {len(enabled_methods)} method(s): {[m['name'] for m in enabled_methods]}"
        )

        # Run each method sequentially
        for method_config in enabled_methods:
            method_name = method_config.get("name", "trade_shuffle")

            # Check if method requires backtest re-run
            if MonteCarlo.method_requires_backtest_rerun(method_name):
                if data is None:
                    logger.error(f"MCPT: {method_name} requires OHLCV data, skipping")
                    continue
                self._run_bar_permute_method(method_config, data)
            else:
                self._run_trade_shuffle_method(method_config)

    def _run_trade_shuffle_method(self, method_config: dict) -> None:
        """
        Run MCPT with trade shuffle method (fast, post-backtest).

        :param method_config: Method configuration dict
        """
        mc = MonteCarlo(self.config, method_config)

        # Run MCPT for each strategy
        for strategy_name, content in self.bt.all_bt_content.items():
            trades_df = content["results"]

            if len(trades_df) < 10:
                logger.warning(
                    f"MCPT: Skipping {strategy_name} - not enough trades ({len(trades_df)} < 10 required)"
                )
                continue

            # Get strategy stats from results
            strategy_stats = self.bt.results["strategy"].get(strategy_name, {})

            # Run MCPT validation
            mcpt_result = mc.validate(trades_df, strategy_stats, method_config)

            self._store_result(strategy_name, mcpt_result)

    def _run_bar_permute_method(self, method_config: dict, data: dict[str, DataFrame]) -> None:
        """
        Run MCPT with bar permute method (slow, requires backtest re-run).

        :param method_config: Method configuration dict
        :param data: OHLCV data for permutation
        """
        # Create MonteCarlo instance for this method
        mc = MonteCarlo(self.config, method_config)

        method_name = method_config.get("name", "bar_permute")
        n_runs = method_config.get("runs", 100)
        metric = method_config.get("metric", "profit_total")
        seed = method_config.get("seed", self.config.get("mcpt", {}).get("seed"))
        block_size = method_config.get("block_size", 20)
        n_jobs = method_config.get("jobs", -1)  # -1 = all CPUs

        # Check if we have detail data for multi-timeframe permutation
        has_detail = bool(self.bt.detail_data)
        main_tf = self.bt.timeframe
        detail_tf = self.bt.timeframe_detail if has_detail else None

        logger.info(f"MCPT Bar Permute: {n_runs} runs, metric={metric}, block_size={block_size}")
        logger.debug(f"Data: {len(data)} pairs, seed={seed}")
        if has_detail:
            logger.debug(f"Multi-timeframe mode: {main_tf} + {detail_tf} (synchronized)")
        else:
            logger.debug(f"Single timeframe mode: {main_tf}")

        # Determine parallelization
        cpus = cpu_count()
        effective_jobs = n_jobs if n_jobs > 0 else cpus
        effective_jobs = min(effective_jobs, n_runs)  # Don't use more jobs than runs
        use_parallel = effective_jobs > 1

        if use_parallel:
            logger.info(f"Parallel mode: {effective_jobs} workers (of {cpus} CPUs available)")
        else:
            logger.info("Sequential mode (single worker)")

        mc.progress.init_step(MCPTState.PERMUTATION_TESTS, n_runs)
        start_time = time.time()

        # Run for each strategy
        for strat in self.bt.strategylist:
            strategy_name = strat.get_strategy_name()
            logger.info(f"MCPT: Starting permutation tests for {strategy_name}")

            # Get original metric value
            strategy_stats = self.bt.results["strategy"].get(strategy_name, {})
            metric_real = mc._extract_metric(strategy_stats, metric)
            logger.info(f"MCPT: Original {metric} = {metric_real:.4f}")

            # Process in batches to avoid OOM with large run counts
            # Only batch if runs >= 30, otherwise process all in one batch
            MIN_BATCH_SIZE = 30
            if n_runs < MIN_BATCH_SIZE:
                batch_size = n_runs  # All in one batch
            else:
                # Batch size = max(30, workers * 2) for efficient parallelism
                batch_size = max(MIN_BATCH_SIZE, effective_jobs * 2)
            permuted_metrics: list[float] = []

            # Use prefetching: generate next batch while processing current
            def generate_batch(start_idx: int, count: int) -> list:
                """Generate a batch of permutations."""
                batch_seed = seed + start_idx if seed is not None else None
                if has_detail:
                    return list(
                        mc.permuter.permute_batch_multi_timeframe(
                            data,
                            self.bt.detail_data,
                            main_tf,
                            detail_tf,
                            count,
                            base_seed=batch_seed,
                        )
                    )
                else:
                    return [
                        (perm_data, {})
                        for perm_data in mc.permuter.permute_batch(
                            data, count, base_seed=batch_seed
                        )
                    ]

            # Calculate batch boundaries
            batches = []
            for batch_start in range(0, n_runs, batch_size):
                batch_end = min(batch_start + batch_size, n_runs)
                batches.append((batch_start, batch_end - batch_start))

            n_batches = len(batches)
            logger.info(
                f"MCPT: Processing {n_runs} runs in {n_batches} batches (size={batch_size})"
            )

            # Use single thread for prefetching (to limit memory to 2 batches max)
            with ThreadPoolExecutor(max_workers=1) as prefetch_executor:
                # Generate first batch synchronously
                batch_start, batch_n = batches[0]
                logger.info(
                    f"MCPT: Generating batch 1/{n_batches} ({batch_start + 1}-{batch_start + batch_n})..."
                )
                t_gen = time.time()
                current_batch = generate_batch(batch_start, batch_n)
                logger.info(
                    f"MCPT: Generated {len(current_batch)} permutations in {time.time() - t_gen:.1f}s"
                )

                for i, (batch_start, batch_n) in enumerate(batches):
                    # Start prefetching next batch (if exists)
                    next_batch_future = None
                    if i + 1 < n_batches:
                        next_start, next_n = batches[i + 1]
                        next_batch_future = prefetch_executor.submit(
                            generate_batch, next_start, next_n
                        )

                    # Process current batch
                    if use_parallel:
                        batch_metrics = self._run_parallel(
                            strat, current_batch, metric, effective_jobs, mc
                        )
                    else:
                        batch_metrics = self._run_sequential(strat, current_batch, metric, mc)

                    permuted_metrics.extend(batch_metrics)

                    # Free current batch memory
                    del current_batch

                    logger.info(f"MCPT: Completed {len(permuted_metrics)}/{n_runs} runs")

                    # Get prefetched next batch (if exists)
                    if next_batch_future is not None:
                        t_wait = time.time()
                        current_batch = next_batch_future.result()
                        wait_time = time.time() - t_wait
                        if wait_time > 0.1:
                            logger.debug(f"MCPT: Waited {wait_time:.1f}s for next batch")
                        # else: batch was ready (prefetched successfully)

            if len(permuted_metrics) < 10:
                logger.warning(f"MCPT: Not enough successful permutations for {strategy_name}")
                continue

            # Calculate p-value

            p_value, count_better = calculate_p_value(metric_real, permuted_metrics)

            mcpt_result = MCPTResult(
                method_name=method_name,
                metric_name=metric,
                metric_real=metric_real,
                metric_permuted=permuted_metrics,
                p_value=p_value,
                n_permutations=len(permuted_metrics),
                count_better=count_better,
            )

            self._store_result(strategy_name, mcpt_result)
            mc._log_result(mcpt_result)

            # Log summary
            total_time = time.time() - start_time
            logger.info(
                f"MCPT {strategy_name} completed: {len(permuted_metrics)}/{n_runs} runs "
                f"in {total_time:.1f}s (avg {total_time / len(permuted_metrics):.1f}s/run)"
            )

    def _run_sequential(
        self,
        strategy,
        permutations_list: list,
        metric: str,
        mc,
    ) -> list[float]:
        """Run MCPT permutations sequentially."""
        permuted_metrics = []

        for i, (permuted_data, permuted_detail) in enumerate(permutations_list):
            perm_start = time.time()

            try:
                permuted_results = self._run_permuted_backtest(
                    strategy, permuted_data, permuted_detail
                )
                permuted_metric = mc._extract_metric(permuted_results, metric)
                permuted_metrics.append(permuted_metric)

                perm_time = time.time() - perm_start
                logger.info(
                    f"MCPT [{i + 1}/{len(permutations_list)}]: {metric}={permuted_metric:.4f} (took {perm_time:.1f}s)"
                )
            except Exception as e:
                logger.warning(f"MCPT permutation {i + 1} failed: {e}")
                continue

            mc.progress.increment()

        return permuted_metrics

    def _run_parallel(
        self,
        strategy,
        permutations_list: list,
        metric: str,
        n_jobs: int,
        mc,
    ) -> list[float]:
        """
        Run MCPT permutations in parallel using joblib.

        Uses joblib.Parallel for cross-platform compatibility (Windows, Linux, macOS).
        Follows the same pattern as HyperOpt for reliable parallel execution.

        Falls back to sequential execution if parallel fails.
        """
        n_runs = len(permutations_list)
        logger.info(f"MCPT: Running {n_runs} permutations in parallel ({n_jobs} workers)...")

        t_start = time.time()

        try:
            # Create MCPTRunner - this will be serialized and sent to workers
            runner = MCPTRunner(
                config=self.config,
                timerange=self.bt.timerange,
                required_startup=self.bt.required_startup,
                timeframe=self.bt.timeframe,
                timeframe_detail=self.bt.timeframe_detail,
            )

            # Save all permuted data to pickle files
            logger.info("MCPT: Saving permuted data to temporary files...")
            t_save = time.time()
            task_files = []
            for i, (permuted_data, permuted_detail) in enumerate(permutations_list):
                data_file, detail_file = runner.save_permuted_data(
                    i, permuted_data, permuted_detail
                )
                task_files.append((i, data_file, detail_file, metric))
            logger.info(f"MCPT: Saved {len(task_files)} files in {time.time() - t_save:.1f}s")

            # Run in parallel using joblib
            logger.info(f"MCPT: Starting parallel execution with {n_jobs} workers...")
            with Parallel(n_jobs=n_jobs) as parallel:
                results = parallel(
                    runner.run_permuted_backtest(i, data_file, detail_file, m)
                    for i, data_file, detail_file, m in task_files
                )

            # Cleanup temporary files
            logger.debug("MCPT: Cleaning up temporary files...")
            runner.cleanup_temp_files(list(range(n_runs)))

        except Exception as e:
            logger.warning(f"Parallel execution failed: {e}")

            traceback.print_exc()
            logger.info("Falling back to sequential execution...")
            return self._run_sequential(strategy, permutations_list, metric, mc)

        # Filter successful results and log worker details
        permuted_metrics = []
        for i, result in enumerate(results):
            if result is not None:
                if isinstance(result, MCPTWorkerResult):
                    # Log worker result in main process
                    logger.info(
                        f"MCPT Worker [{result.run_index + 1}]: {result.metric_name}={result.metric_value:.4f} | "
                        f"{result.profit_total * 100:+.2f}% | {result.n_trades} trades | {result.elapsed:.1f}s"
                    )
                    permuted_metrics.append(result.metric_value)
                else:
                    # Fallback for raw float (shouldn't happen normally)
                    permuted_metrics.append(result)
                mc.progress.increment()
            else:
                logger.warning(f"MCPT permutation {i + 1} returned None")

        elapsed = time.time() - t_start
        avg_time = elapsed / len(permuted_metrics) if permuted_metrics else 0

        logger.info(
            f"MCPT: Parallel execution completed in {elapsed:.1f}s "
            f"({len(permuted_metrics)}/{n_runs} successful, avg {avg_time:.1f}s/run)"
        )

        return permuted_metrics

    def _run_permuted_backtest(
        self,
        strategy,
        permuted_data: dict[str, DataFrame],
        permuted_detail: dict[str, DataFrame] | None = None,
    ) -> dict[str, float]:
        """
        Run a single backtest with permuted data.

        :param strategy: Strategy to test
        :param permuted_data: Permuted OHLCV data (main timeframe)
        :param permuted_detail: Permuted detail data (optional, for multi-timeframe)
        :return: Dictionary with backtest metrics
        """
        logger.info("  -> Calculating indicators...")
        t0 = time.time()
        processed = self._prepare_data_for_backtest(permuted_data, strategy)
        logger.info(f"  -> Indicators done in {time.time() - t0:.1f}s")

        # Trim startup period from processed data (same as backtest_one_strategy)
        processed_trimmed = trim_dataframes(processed, self.bt.timerange, self.bt.required_startup)

        # Get timerange from TRIMMED data (not raw permuted data!)
        min_date, max_date = history.get_timerange(processed_trimmed)

        # Debug: check data integrity (use trimmed data for accurate counts)
        total_rows = sum(len(df) for df in processed_trimmed.values())
        logger.info(
            f"  -> Data: {len(processed_trimmed)} pairs, {total_rows} total rows (after trim)"
        )
        logger.info(f"  -> Timerange: {min_date} to {max_date} ({(max_date - min_date).days} days)")

        # Temporarily replace detail_data with permuted detail if provided
        saved_detail_data = None
        if permuted_detail:
            saved_detail_data = self.bt.detail_data
            self.bt.detail_data = permuted_detail
            logger.info(f"  -> Using permuted detail data ({len(permuted_detail)} pairs)")

        # Run backtest
        logger.info("  -> Running backtest on permuted data...")
        t0 = time.time()
        try:
            results = self.bt.backtest(processed, min_date, max_date)
        finally:
            # Restore original detail_data
            if saved_detail_data is not None:
                self.bt.detail_data = saved_detail_data

        logger.info(f"  -> Backtest done in {time.time() - t0:.1f}s")
        logger.info(f"  -> Results: {len(results.get('results', []))} trades generated")

        # Extract metrics from results
        trades = results.get("results", [])
        if len(trades) == 0:
            logger.info("  -> No trades generated")
            return {"profit_total": 0.0, "profit_total_abs": 0.0, "max_drawdown_abs": 0.0}

        # Get starting balance for relative profit calculation
        start_balance = get_dry_run_wallet(self.config)
        profit_abs = trades["profit_abs"].sum() if len(trades) > 0 else 0.0
        # profit_total should be relative (same as in generate_strategy_stats)
        profit_total = profit_abs / start_balance if start_balance > 0 else 0.0

        logger.info(
            f"  -> {len(trades)} trades, profit_abs={profit_abs:.2f}, profit_total={profit_total:.4f} ({profit_total * 100:.2f}%)"
        )

        return {
            "profit_total": profit_total,  # Relative profit (same as original)
            "profit_total_abs": profit_abs,  # Absolute profit in stake currency
            "max_drawdown_abs": self._calculate_drawdown(trades),
        }

    def _prepare_data_for_backtest(
        self, data: dict[str, DataFrame], strategy
    ) -> dict[str, DataFrame]:
        """Prepare permuted data for backtest by calculating indicators."""
        processed = {}

        # Determine number of workers (leave one core free for system)
        max_workers = max(1, multiprocessing.cpu_count() - 1)

        logger.info(f"  -> Calculating indicators (parallel, {max_workers} workers)...")
        t_start = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_pair = {
                executor.submit(strategy.advise_indicators, df.copy(), {"pair": pair}): pair
                for pair, df in data.items()
            }

            # Process results as they complete
            for future in as_completed(future_to_pair):
                pair = future_to_pair[future]
                try:
                    processed[pair] = future.result()
                except Exception as exc:
                    logger.error(f"  -> Indicators for {pair} generated an exception: {exc}")
                    # Better to fail hard or skip pair?
                    # For now, re-raise to fail the run
                    raise exc

        logger.info(f"  -> All indicators calculated in {time.time() - t_start:.1f}s")
        return processed

    def _calculate_drawdown(self, trades: DataFrame) -> float:
        """Calculate max drawdown from trades."""
        if len(trades) == 0:
            return 0.0
        equity = trades["profit_abs"].cumsum()
        running_max = equity.expanding().max()
        drawdown = running_max - equity
        return float(drawdown.max())

    def _store_result(self, strategy_name: str, mcpt_result: Any) -> None:
        """
        Store MCPT result in backtest results.

        Results are stored as a list per strategy to support multiple methods.
        """
        if "mcpt" not in self.bt.results:
            self.bt.results["mcpt"] = {}

        if strategy_name not in self.bt.results["mcpt"]:
            self.bt.results["mcpt"][strategy_name] = []

        # Append result for this method
        self.bt.results["mcpt"][strategy_name].append(
            {
                "method_name": mcpt_result.method_name,
                "metric_name": mcpt_result.metric_name,
                "metric_real": mcpt_result.metric_real,
                "p_value": mcpt_result.p_value,
                "n_permutations": mcpt_result.n_permutations,
                "count_better": mcpt_result.count_better,
                "significant": mcpt_result.p_value < 0.05,
            }
        )

        logger.info(
            f"MCPT [{mcpt_result.method_name}] {strategy_name}: "
            f"p-value={mcpt_result.p_value:.4f} "
            f"({'significant' if mcpt_result.p_value < 0.05 else 'not significant'})"
        )
