"""
Compare two MCPT baseline JSON files and print a speedup table.

Usage:
    python -m freqtrade.optimize.montecarlo.benchmarks.compare baseline.json optimized.json
    python -m freqtrade.optimize.montecarlo.benchmarks.compare before.json after.json --min-speedup 1.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _speedup_color(speedup: float) -> str:
    if speedup >= 3.0:
        return _GREEN
    if speedup >= 1.5:
        return _YELLOW
    return _RED


def load_baseline(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def print_meta_diff(before: dict, after: dict) -> None:
    bm = before.get("meta", {})
    am = after.get("meta", {})
    print(f"\n  {'':30} {'BEFORE':>30}  {'AFTER':>30}")
    print(f"  {'-' * 93}")
    for key in ("git_sha", "timestamp", "python", "numpy"):
        print(f"  {key:<30} {str(bm.get(key, '?')):>30}  {str(am.get(key, '?')):>30}")


def compare(
    before_path: Path,
    after_path: Path,
    min_speedup: float = 1.0,
) -> None:
    before = load_baseline(before_path)
    after = load_baseline(after_path)

    print(f"\n{_BOLD}MCPT Benchmark Comparison{_RESET}")
    print(f"  Before : {before_path}")
    print(f"  After  : {after_path}")

    print_meta_diff(before, after)

    b_benchmarks: dict = before.get("benchmarks", {})
    a_benchmarks: dict = after.get("benchmarks", {})

    common_keys = sorted(set(b_benchmarks) & set(a_benchmarks))
    only_before = sorted(set(b_benchmarks) - set(a_benchmarks))
    only_after = sorted(set(a_benchmarks) - set(b_benchmarks))

    # Header
    col_label = 55
    print(
        f"\n  {'Benchmark':<{col_label}} {'Before':>10}  {'After':>10}  {'Speedup':>8}  {'Status'}"
    )
    print(f"  {'-' * col_label} {'-' * 10}  {'-' * 10}  {'-' * 8}  {'-' * 10}")

    improved = 0
    regressed = 0
    neutral = 0

    for key in common_keys:
        bv = b_benchmarks[key]
        av = a_benchmarks[key]

        b_mean = bv.get("mean_s", 0.0)
        a_mean = av.get("mean_s", 0.0)

        if b_mean <= 0 or a_mean <= 0:
            continue

        speedup = b_mean / a_mean
        label = bv.get("label", key)[:col_label]

        if speedup < min_speedup:
            continue

        color = _speedup_color(speedup)

        if speedup >= 1.05:
            status = f"{color}FASTER{_RESET}"
            improved += 1
        elif speedup <= 0.95:
            status = f"{_RED}SLOWER{_RESET}"
            regressed += 1
        else:
            status = "~same"
            neutral += 1

        print(
            f"  {label:<{col_label}} "
            f"{b_mean * 1000:>8.2f}ms  "
            f"{a_mean * 1000:>8.2f}ms  "
            f"{color}{speedup:>7.2f}x{_RESET}  "
            f"{status}"
        )

    # Summary
    print(f"\n  {'=' * 93}")
    print(f"  Common benchmarks : {len(common_keys)}")
    print(f"  Faster            : {_GREEN}{improved}{_RESET}")
    print(f"  Slower            : {_RED}{regressed}{_RESET}")
    print(f"  ~Same             : {neutral}")

    if only_before:
        print(f"\n  Benchmarks only in BEFORE — removed ({len(only_before)}):")
        print(f"  {'Benchmark':<{col_label}} {'Before':>10}")
        print(f"  {'-' * col_label} {'-' * 10}")
        for k in only_before:
            bv = b_benchmarks[k]
            b_mean = bv.get("mean_s", 0.0)
            label = bv.get("label", k)[:col_label]
            print(f"  {label:<{col_label}} {b_mean * 1000:>8.2f}ms  (no longer measured)")

    if only_after:
        print(f"\n  Benchmarks only in AFTER — new ({len(only_after)}):")
        print(f"  {'Benchmark':<{col_label}} {'After':>10}  {'Throughput':>14}")
        print(f"  {'-' * col_label} {'-' * 10}  {'-' * 14}")
        for k in sorted(only_after):
            av = a_benchmarks[k]
            a_mean = av.get("mean_s", 0.0)
            a_std = av.get("std_s", 0.0)
            throughput = av.get("throughput")
            label = av.get("label", k)[:col_label]
            tput_str = f"{throughput:>12.1f} /s" if throughput else ""
            print(
                f"  {label:<{col_label}} {a_mean * 1000:>8.2f}ms ± {a_std * 1000:.2f}ms  {tput_str}"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two MCPT benchmark JSON baselines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("before", type=Path, help="Path to baseline (before) JSON")
    parser.add_argument("after", type=Path, help="Path to optimized (after) JSON")
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=1.0,
        help="Only show rows with speedup >= this value (default: 1.0 = show all)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    compare(args.before, args.after, min_speedup=args.min_speedup)


if __name__ == "__main__":
    main(sys.argv[1:])
