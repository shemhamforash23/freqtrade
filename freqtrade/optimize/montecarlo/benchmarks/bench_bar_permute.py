"""
Bar Permute MCPT benchmarks using real Binance OHLCV data.

Tests:
  BP-1  BarPermutePermuter._permute_df_with_indices()  — price reconstruction loop (micro)
  BP-2  BarPermutePermuter.permute()                   — single full permutation (1+ pairs)
  BP-3  BarPermutePermuter.permute_batch()             — N permutations batch (macro)
  MTF-1 BarPermutePermuter.permute_multi_timeframe()   — single MTF permutation (5m+1m)
  MTF-2 BarPermutePermuter.permute_batch_multi_timeframe() — N MTF permutations batch
"""

from __future__ import annotations

import numpy as np

from freqtrade.optimize.montecarlo.benchmarks._utils import (
    BenchResult,
    print_result,
    print_section,
    timeit_fn,
)
from freqtrade.optimize.montecarlo.benchmarks.data_loader import (
    BENCHMARK_PAIRS_5M,
    BENCHMARK_PAIRS_MTF,
    load_mtf_pairs,
    load_multi_pair,
    load_ohlcv,
)
from freqtrade.optimize.montecarlo.permuters.bar_permute import BarPermutePermuter


# ---------------------------------------------------------------------------
# Parametric scenarios
# ---------------------------------------------------------------------------

# n_bars for BP-1 micro benchmark (isolated price reconstruction)
BP1_N_BARS: list[int] = [5_000, 20_000, 50_000]

# (pairs, timeframe, n_bars, n_runs) for BP-2 / BP-3 macro benchmark
BP_MACRO_SCENARIOS: list[tuple[list[str], str, int, int]] = [
    (["BTC_USDT"], "5m", 5_000, 10),
    (["BTC_USDT"], "5m", 20_000, 20),
    (BENCHMARK_PAIRS_5M[:5], "5m", 5_000, 10),
    (BENCHMARK_PAIRS_5M[:5], "5m", 20_000, 10),
]

BLOCK_SIZE = 20

# (pairs, main_tf, detail_tf, tf_ratio, n_main_bars, n_runs)
MTF_SCENARIOS: list[tuple[list[str], str, str, int, int, int]] = [
    (BENCHMARK_PAIRS_MTF[:1], "5m", "1m", 5, 5_000, 1),
    (BENCHMARK_PAIRS_MTF[:1], "5m", "1m", 5, 20_000, 1),
    (BENCHMARK_PAIRS_MTF, "5m", "1m", 5, 5_000, 10),
    (BENCHMARK_PAIRS_MTF, "5m", "1m", 5, 20_000, 5),
]


# ---------------------------------------------------------------------------
# BP-1: Isolated price reconstruction loop
# ---------------------------------------------------------------------------


def run_bp1_benchmarks(n_warmup: int = 3, n_reps: int = 10) -> dict[str, BenchResult]:
    """
    Benchmark BarPermutePermuter._permute_df_with_indices() in isolation.

    Pre-generates block_indices outside the timed section.
    This isolates exactly the Python loop that we plan to replace with np.cumsum.
    """
    print_section("BP-1: Price reconstruction loop (_permute_df_with_indices)")

    results: dict[str, BenchResult] = {}
    permuter = BarPermutePermuter(block_size=BLOCK_SIZE)
    rng = np.random.default_rng(42)

    # Load once, slice to different sizes
    full_df = load_ohlcv("BTC_USDT", "5m")
    available = len(full_df)
    print(f"  BTC_USDT 5m: {available:,} bars available")

    for n_bars in BP1_N_BARS:
        actual_n = min(n_bars, available)
        df = full_df.tail(actual_n).reset_index(drop=True)

        # Pre-generate indices (NOT part of the timed section)
        block_indices = permuter._generate_block_indices(len(df), rng)

        key = f"bp1__n_bars={actual_n}"
        label = f"_permute_df_with_indices  n_bars={actual_n:>6,}"

        result = timeit_fn(
            permuter._permute_df_with_indices,
            df,
            block_indices,
            n_warmup=n_warmup,
            n_reps=n_reps,
            label=label,
            component="BP-1",
            params={"n_bars": actual_n, "block_size": BLOCK_SIZE},
        )
        results[key] = result
        print_result(result)

    return results


# ---------------------------------------------------------------------------
# BP-2: Single full permutation (permute() call)
# ---------------------------------------------------------------------------


def run_bp2_benchmarks(n_warmup: int = 3, n_reps: int = 10) -> dict[str, BenchResult]:
    """
    Benchmark BarPermutePermuter.permute() — one full permutation including
    block index generation + price reconstruction for N pairs.
    """
    print_section("BP-2: Single full permutation (BarPermutePermuter.permute)")

    results: dict[str, BenchResult] = {}
    permuter = BarPermutePermuter(block_size=BLOCK_SIZE)

    for pairs, timeframe, n_bars, _ in BP_MACRO_SCENARIOS:
        data = load_multi_pair(pairs, timeframe, n_bars)
        if not data:
            continue
        actual_pairs = len(data)
        actual_n = len(next(iter(data.values())))

        key = f"bp2__pairs={actual_pairs}__n_bars={actual_n}"
        label = f"permute()  {actual_pairs} pair(s) x {actual_n:>6,} bars"

        result = timeit_fn(
            permuter.permute,
            data,
            42,  # seed
            n_warmup=n_warmup,
            n_reps=n_reps,
            label=label,
            component="BP-2",
            params={"n_pairs": actual_pairs, "n_bars": actual_n, "block_size": BLOCK_SIZE},
        )
        results[key] = result
        print_result(result)

    return results


# ---------------------------------------------------------------------------
# BP-3: Batch permutations (permute_batch() generator consumed fully)
# ---------------------------------------------------------------------------


def run_bp3_benchmarks(n_warmup: int = 1, n_reps: int = 5) -> dict[str, BenchResult]:
    """
    Benchmark BarPermutePermuter.permute_batch() — consume full generator
    of N permutations. Measures total wall time for a realistic MCPT run.

    Fewer reps since each rep is (n_runs x n_pairs x n_bars) work.
    """
    print_section("BP-3: Batch permutations (BarPermutePermuter.permute_batch)")

    results: dict[str, BenchResult] = {}
    permuter = BarPermutePermuter(block_size=BLOCK_SIZE)

    for pairs, timeframe, n_bars, n_runs in BP_MACRO_SCENARIOS:
        data = load_multi_pair(pairs, timeframe, n_bars)
        if not data:
            continue
        actual_pairs = len(data)
        actual_n = len(next(iter(data.values())))

        # Wrap in a function that fully consumes the generator
        def _run_batch(d=data, nr=n_runs) -> None:
            list(permuter.permute_batch(d, nr, base_seed=42))

        key = f"bp3__pairs={actual_pairs}__n_bars={actual_n}__n_runs={n_runs}"
        label = f"permute_batch()  {actual_pairs} pair(s) x {actual_n:>6,} bars x {n_runs} runs"

        result = timeit_fn(
            _run_batch,
            n_warmup=n_warmup,
            n_reps=n_reps,
            throughput_count=n_runs,
            label=label,
            component="BP-3",
            params={
                "n_pairs": actual_pairs,
                "n_bars": actual_n,
                "n_runs": n_runs,
                "block_size": BLOCK_SIZE,
            },
        )
        results[key] = result
        print_result(result)

    return results


# ---------------------------------------------------------------------------
# MTF-1: Single multi-timeframe permutation
# ---------------------------------------------------------------------------


def run_mtf1_benchmarks(n_warmup: int = 3, n_reps: int = 10) -> dict[str, BenchResult]:
    """
    Benchmark BarPermutePermuter.permute_multi_timeframe() — one synchronised
    5m+1m permutation across N pairs. Exercises _block_starts_to_indices() and
    the double-timeframe price reconstruction.
    """
    print_section("MTF-1: Single multi-timeframe permutation (5m + 1m)")

    results: dict[str, BenchResult] = {}
    permuter = BarPermutePermuter(block_size=BLOCK_SIZE)

    for pairs, main_tf, detail_tf, tf_ratio, n_main_bars, _ in MTF_SCENARIOS:
        main_data, detail_data = load_mtf_pairs(pairs, main_tf, detail_tf, n_main_bars, tf_ratio)
        if not main_data:
            continue
        actual_pairs = len(main_data)
        actual_n = len(next(iter(main_data.values())))
        actual_detail = len(next(iter(detail_data.values())))

        key = f"mtf1__pairs={actual_pairs}__n_main={actual_n}"
        label = (
            f"permute_mtf()  {actual_pairs}p  "
            f"{main_tf}:{actual_n:>6,} + {detail_tf}:{actual_detail:>7,} bars"
        )

        result = timeit_fn(
            permuter.permute_multi_timeframe,
            main_data,
            detail_data,
            main_tf,
            detail_tf,
            42,
            n_warmup=n_warmup,
            n_reps=n_reps,
            label=label,
            component="MTF-1",
            params={
                "n_pairs": actual_pairs,
                "n_main_bars": actual_n,
                "n_detail_bars": actual_detail,
                "main_tf": main_tf,
                "detail_tf": detail_tf,
                "block_size": BLOCK_SIZE,
            },
        )
        results[key] = result
        print_result(result)

    return results


# ---------------------------------------------------------------------------
# MTF-2: Batch multi-timeframe permutations
# ---------------------------------------------------------------------------


def run_mtf2_benchmarks(n_warmup: int = 1, n_reps: int = 5) -> dict[str, BenchResult]:
    """
    Benchmark BarPermutePermuter.permute_batch_multi_timeframe() — consume
    full generator of N synchronised 5m+1m permutations.
    """
    print_section("MTF-2: Batch multi-timeframe permutations (5m + 1m)")

    results: dict[str, BenchResult] = {}
    permuter = BarPermutePermuter(block_size=BLOCK_SIZE)

    for pairs, main_tf, detail_tf, tf_ratio, n_main_bars, n_runs in MTF_SCENARIOS:
        if n_runs < 2:
            continue  # skip single-run scenarios for batch benchmark
        main_data, detail_data = load_mtf_pairs(pairs, main_tf, detail_tf, n_main_bars, tf_ratio)
        if not main_data:
            continue
        actual_pairs = len(main_data)
        actual_n = len(next(iter(main_data.values())))
        actual_detail = len(next(iter(detail_data.values())))

        def _run_batch(
            md=main_data,
            dd=detail_data,
            nr=n_runs,
            mtf=main_tf,
            dtf=detail_tf,
        ) -> None:
            list(permuter.permute_batch_multi_timeframe(md, dd, mtf, dtf, nr, base_seed=42))

        key = f"mtf2__pairs={actual_pairs}__n_main={actual_n}__n_runs={n_runs}"
        label = (
            f"permute_batch_mtf()  {actual_pairs}p  "
            f"{main_tf}:{actual_n:>6,}+{detail_tf}:{actual_detail:>7,} x{n_runs} runs"
        )

        result = timeit_fn(
            _run_batch,
            n_warmup=n_warmup,
            n_reps=n_reps,
            throughput_count=n_runs,
            label=label,
            component="MTF-2",
            params={
                "n_pairs": actual_pairs,
                "n_main_bars": actual_n,
                "n_detail_bars": actual_detail,
                "n_runs": n_runs,
                "main_tf": main_tf,
                "detail_tf": detail_tf,
                "block_size": BLOCK_SIZE,
            },
        )
        results[key] = result
        print_result(result)

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_all(n_warmup: int = 3, n_reps: int = 10) -> dict[str, BenchResult]:
    """Run all Bar Permute benchmarks and return combined results."""
    results: dict[str, BenchResult] = {}
    results.update(run_bp1_benchmarks(n_warmup=n_warmup, n_reps=n_reps))
    results.update(run_bp2_benchmarks(n_warmup=n_warmup, n_reps=n_reps))
    results.update(run_bp3_benchmarks(n_warmup=1, n_reps=max(n_reps // 2, 3)))
    results.update(run_mtf1_benchmarks(n_warmup=n_warmup, n_reps=n_reps))
    results.update(run_mtf2_benchmarks(n_warmup=1, n_reps=max(n_reps // 2, 3)))
    return results


if __name__ == "__main__":
    run_all()
