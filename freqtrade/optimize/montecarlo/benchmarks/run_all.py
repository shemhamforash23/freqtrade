"""
MCPT Benchmark orchestrator.

Runs all Trade Shuffle and Bar Permute benchmarks in single-threaded mode,
then saves results as a JSON baseline file for before/after comparison.

Usage (from freqtrade/ repo root):
    python -m freqtrade.optimize.montecarlo.benchmarks.run_all
    python -m freqtrade.optimize.montecarlo.benchmarks.run_all --quick
    python -m freqtrade.optimize.montecarlo.benchmarks.run_all --output results/baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from freqtrade.optimize.montecarlo.benchmarks import bench_bar_permute, bench_trade_shuffle
from freqtrade.optimize.montecarlo.benchmarks._utils import BenchResult, get_meta, print_section


def _results_to_json(results: dict[str, BenchResult], meta: dict) -> dict:
    """Serialize benchmark results to a JSON-compatible dict."""
    return {
        "meta": meta,
        "benchmarks": {key: r.to_dict() for key, r in results.items()},
    }


def _default_output_path(meta: dict) -> Path:
    """Build a timestamped output filename from git SHA and timestamp."""
    sha = meta.get("git_sha", "unknown")
    ts = meta.get("timestamp", "").replace(":", "").replace("-", "").replace("T", "_")
    return Path(f"mcpt_baseline_{sha}_{ts}.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MCPT benchmarks and save baseline JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output JSON path (default: mcpt_baseline_<sha>_<ts>.json in cwd)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer reps/warmup for fast verification (n_warmup=1, n_reps=3)",
    )
    parser.add_argument(
        "--ts-only",
        action="store_true",
        help="Run only Trade Shuffle benchmarks",
    )
    parser.add_argument(
        "--bp-only",
        action="store_true",
        help="Run only Bar Permute benchmarks",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    n_warmup = 1 if args.quick else 3
    n_reps = 3 if args.quick else 10

    print_section("MCPT Benchmark Suite — single-threaded baseline")
    meta = get_meta()

    print(f"\n  Platform : {meta['platform']}")
    print(f"  Machine  : {meta['machine']}")
    print(f"  Python   : {meta['python']}")
    print(f"  NumPy    : {meta['numpy']}")
    print(f"  Git SHA  : {meta['git_sha']}")
    print(f"  Mode     : {'QUICK' if args.quick else 'FULL'} (n_reps={n_reps})")

    total_start = time.perf_counter()
    all_results: dict[str, BenchResult] = {}

    run_ts = not args.bp_only
    run_bp = not args.ts_only

    if run_ts:
        ts_results = bench_trade_shuffle.run_all(n_warmup=n_warmup, n_reps=n_reps)
        all_results.update(ts_results)

    if run_bp:
        bp_results = bench_bar_permute.run_all(n_warmup=n_warmup, n_reps=n_reps)
        all_results.update(bp_results)

    total_elapsed = time.perf_counter() - total_start

    print_section(f"Completed {len(all_results)} benchmarks in {total_elapsed:.1f}s")

    # Determine output path
    output_path = args.output if args.output else _default_output_path(meta)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize and save
    payload = _results_to_json(all_results, meta)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n  Baseline saved → {output_path.resolve()}")
    print(f"  Total benchmarks : {len(all_results)}")
    print(f"  Total wall time  : {total_elapsed:.1f}s")

    # Print key summary table
    print_section("Key results summary")
    print(f"  {'Benchmark':<55} {'mean':>10}  {'std':>8}  {'throughput':>14}")
    print(f"  {'-' * 55} {'-' * 10}  {'-' * 8}  {'-' * 14}")
    for key, r in all_results.items():
        tput = f"{r.throughput:>11.1f} /s" if r.throughput is not None else f"{'':>14}"
        print(f"  {r.label:<55} {r.mean_s * 1000:>8.2f} ms  {r.std_s * 1000:>6.2f} ms  {tput}")


if __name__ == "__main__":
    main(sys.argv[1:])
