"""
Trade Shuffle MCPT benchmarks.

Tests:
  TS-1  TradeShufflePermuter.permute()          — single permutation
  TS-2  MonteCarlo._run_permutations()          — full N-iteration loop
  TS-3  MonteCarlo._calculate_trade_metric()    — isolated per metric

Synthetic trades are used (fixed seed) since we benchmark the algorithm,
not the data loading. Distributions are realistic: win_rate ~55%, avg win $3.5, avg loss -$2.8.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from freqtrade.optimize.montecarlo.benchmarks._utils import (
    BenchResult,
    print_result,
    print_section,
    timeit_fn,
)
from freqtrade.optimize.montecarlo.montecarlo import MonteCarlo
from freqtrade.optimize.montecarlo.permuters.trade_shuffle import TradeShufflePermuter


# ---------------------------------------------------------------------------
# Parametric scenarios
# ---------------------------------------------------------------------------

# (n_trades, n_runs, metric)
TS2_SCENARIOS: list[tuple[int, int, str]] = [
    (50, 100, "sharpe"),
    (50, 100, "profit_factor"),
    (200, 500, "sharpe"),
    (200, 500, "profit_factor"),
    (200, 500, "max_drawdown"),
    (200, 1000, "sharpe"),
    (500, 1000, "sharpe"),
    (500, 5000, "sharpe"),
]

# Metrics for isolated TS-3 benchmark (fixed n_trades=200)
TS3_METRICS: list[str] = ["sharpe", "sortino", "profit_factor", "max_drawdown"]


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def make_synthetic_trades(n_trades: int, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic trades DataFrame with realistic P&L distribution.

    win_rate ~55%, avg win ~$3.5, avg loss ~-$2.8 → profit_factor ~1.45

    :param n_trades: Number of synthetic trades
    :param seed: Random seed for reproducibility
    :return: DataFrame with columns: profit_abs, pair, direction
    """
    rng = np.random.default_rng(seed)
    n_wins = int(n_trades * 0.55)
    n_losses = n_trades - n_wins

    wins = rng.exponential(3.5, n_wins)
    losses = -rng.exponential(2.8, n_losses)
    profit_abs = np.concatenate([wins, losses])
    rng.shuffle(profit_abs)

    pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
    pair_col = [pairs[i % len(pairs)] for i in range(n_trades)]
    direction_col = [1 if p > 0 else -1 for p in profit_abs]

    return pd.DataFrame(
        {
            "profit_abs": profit_abs,
            "pair": pair_col,
            "direction": direction_col,
        }
    )


def _make_mc_instance(scope: str = "global") -> MonteCarlo:
    """Create minimal MonteCarlo instance for benchmarking."""
    config: dict = {"mcpt": {}}
    method_config: dict = {"name": "trade_shuffle", "scope": scope}
    return MonteCarlo(config, method_config)


# ---------------------------------------------------------------------------
# TS-1: Single permutation
# ---------------------------------------------------------------------------


def run_ts1_benchmarks(n_warmup: int = 3, n_reps: int = 10) -> dict[str, BenchResult]:
    """Benchmark TradeShufflePermuter.permute() — single shuffle."""
    print_section("TS-1: Single permutation (TradeShufflePermuter.permute)")

    results: dict[str, BenchResult] = {}
    permuter = TradeShufflePermuter(scope="global")

    for n_trades in (50, 200, 500):
        trades = make_synthetic_trades(n_trades)
        key = f"ts1__n_trades={n_trades}"
        label = f"permute()  n_trades={n_trades}"

        result = timeit_fn(
            permuter.permute,
            trades,
            seed=42,
            n_warmup=n_warmup,
            n_reps=n_reps,
            label=label,
            component="TS-1",
            params={"n_trades": n_trades},
        )
        results[key] = result
        print_result(result)

    return results


# ---------------------------------------------------------------------------
# TS-2: Full permutation loop
# ---------------------------------------------------------------------------


def run_ts2_benchmarks(n_warmup: int = 3, n_reps: int = 10) -> dict[str, BenchResult]:
    """Benchmark MonteCarlo._run_permutations() — full N-run loop."""
    print_section("TS-2: Full permutation loop (MonteCarlo._run_permutations)")

    results: dict[str, BenchResult] = {}
    mc = _make_mc_instance()

    for n_trades, n_runs, metric in TS2_SCENARIOS:
        trades = make_synthetic_trades(n_trades)
        key = f"ts2__n_trades={n_trades}__n_runs={n_runs}__metric={metric}"
        label = f"_run_permutations  n={n_trades} x {n_runs:>4} runs  {metric}"

        result = timeit_fn(
            mc._run_permutations,
            trades,
            metric,
            n_runs,
            42,  # seed
            n_warmup=n_warmup,
            n_reps=n_reps,
            throughput_count=n_runs,
            label=label,
            component="TS-2",
            params={"n_trades": n_trades, "n_runs": n_runs, "metric": metric},
        )
        results[key] = result
        print_result(result)

    return results


# ---------------------------------------------------------------------------
# TS-3: Isolated metric calculation
# ---------------------------------------------------------------------------


def run_ts3_benchmarks(n_warmup: int = 3, n_reps: int = 50) -> dict[str, BenchResult]:
    """Benchmark MonteCarlo._calculate_trade_metric() — single metric call."""
    print_section("TS-3: Isolated metric calculation (_calculate_trade_metric)")

    results: dict[str, BenchResult] = {}
    mc = _make_mc_instance()
    trades = make_synthetic_trades(200)

    for metric in TS3_METRICS:
        key = f"ts3__metric={metric}"
        label = f"_calculate_trade_metric  metric={metric}  n_trades=200"

        result = timeit_fn(
            mc._calculate_trade_metric,
            trades,
            metric,
            n_warmup=n_warmup,
            n_reps=n_reps,
            label=label,
            component="TS-3",
            params={"metric": metric, "n_trades": 200},
        )
        results[key] = result
        print_result(result)

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_all(n_warmup: int = 3, n_reps: int = 10) -> dict[str, BenchResult]:
    """Run all Trade Shuffle benchmarks and return combined results."""
    results: dict[str, BenchResult] = {}
    results.update(run_ts1_benchmarks(n_warmup=n_warmup, n_reps=n_reps))
    results.update(run_ts2_benchmarks(n_warmup=n_warmup, n_reps=n_reps))
    results.update(run_ts3_benchmarks(n_warmup=n_warmup, n_reps=min(n_reps * 5, 50)))
    return results


if __name__ == "__main__":
    run_all()
