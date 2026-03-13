"""
Shared timing and reporting utilities for MCPT benchmarks.
"""

from __future__ import annotations

import platform
import subprocess
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class BenchResult:
    """Result of a single benchmark run."""

    label: str
    component: str
    mean_s: float
    std_s: float
    min_s: float
    max_s: float
    throughput: float | None  # ops/sec, e.g. permutations/sec
    peak_memory_mb: float
    n_warmup: int
    n_reps: int
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def timeit_fn(
    fn: Callable,
    *args: Any,
    n_warmup: int = 3,
    n_reps: int = 10,
    throughput_count: int | None = None,
    label: str = "",
    component: str = "",
    params: dict[str, Any] | None = None,
    **kwargs: Any,
) -> BenchResult:
    """
    Benchmark a callable with warmup and multiple repetitions.

    Memory is measured on a single isolated run after warmup.
    Timing is measured across n_reps runs.

    :param fn: Function to benchmark
    :param args: Positional arguments for fn
    :param n_warmup: Number of warmup runs (not timed)
    :param n_reps: Number of timed repetitions
    :param throughput_count: If set, compute throughput = throughput_count / mean_s
    :param label: Human-readable label for the result
    :param component: Component ID (e.g. 'TS-2', 'BP-1')
    :param params: Dict of parameters for this benchmark case
    :param kwargs: Keyword arguments for fn
    :return: BenchResult
    """
    import numpy as np

    # Warmup
    for _ in range(n_warmup):
        fn(*args, **kwargs)

    # Memory measurement (1 run after warmup)
    tracemalloc.start()
    fn(*args, **kwargs)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Timing
    times: list[float] = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)

    times_arr = np.array(times)
    mean_s = float(times_arr.mean())
    throughput = (
        (throughput_count / mean_s) if (throughput_count is not None and mean_s > 0) else None
    )

    return BenchResult(
        label=label,
        component=component,
        mean_s=mean_s,
        std_s=float(times_arr.std()),
        min_s=float(times_arr.min()),
        max_s=float(times_arr.max()),
        throughput=throughput,
        peak_memory_mb=peak_bytes / 1024 / 1024,
        n_warmup=n_warmup,
        n_reps=n_reps,
        params=params or {},
    )


def get_meta() -> dict[str, Any]:
    """Collect environment metadata for the benchmark report."""
    import numpy as np
    import pandas as pd

    git_sha = "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_sha = result.stdout.strip()
    except Exception:  # git not available in this environment
        git_sha = "unknown"

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": git_sha,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def print_result(result: BenchResult) -> None:
    """Print a single benchmark result to stdout."""
    tput = f"  {result.throughput:>8.1f} ops/s" if result.throughput is not None else ""
    print(
        f"  [{result.component:>4}] {result.label:<55} "
        f"{result.mean_s * 1000:>8.2f} ms ± {result.std_s * 1000:.2f} ms"
        f"{tput}"
        f"  mem={result.peak_memory_mb:.2f} MB"
    )


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 75}")
    print(f"  {title}")
    print(f"{'=' * 75}")
