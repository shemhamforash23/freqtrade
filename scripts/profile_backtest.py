#!/usr/bin/env python3
"""
Backtesting performance profiler — LLM-readable text output.

Usage (run from freqtrade/ directory):
    python scripts/profile_backtest.py \\
        --config user_data/config/config_freqai_guide.json \\
        --strategy SampleStrategy \\
        --timerange 20240101-20240115 \\
        [--top 50]

Output:
    1. PHASE TIMERS  — wall-clock time per major phase + call counts
    2. CALL COUNTS   — most-called functions (ncalls desc)
    3. TOP FUNCTIONS — sorted by cumulative time (cProfile table)
"""

import cProfile
import io
import pstats
import sys
import time
from functools import wraps


# ---------------------------------------------------------------------------
# Phase timing infrastructure
# ---------------------------------------------------------------------------

_phase_times: dict[str, float] = {}
_phase_calls: dict[str, int] = {}


def _timed(label: str):
    """Wrap a method to record cumulative wall-clock time under *label*."""

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            _phase_times[label] = _phase_times.get(label, 0.0) + elapsed
            _phase_calls[label] = _phase_calls.get(label, 0) + 1
            return result

        return wrapper

    return decorator


def _patch_timers() -> None:
    """Monkey-patch key backtesting methods before any instantiation."""
    from freqtrade.optimize.backtesting import Backtesting
    from freqtrade.strategy.interface import IStrategy
    from freqtrade.wallets import Wallets

    Backtesting._get_ohlcv_as_lists = _timed("_get_ohlcv_as_lists")(Backtesting._get_ohlcv_as_lists)
    Backtesting.backtest = _timed("backtest")(Backtesting.backtest)
    Backtesting.get_detail_data = _timed("get_detail_data")(Backtesting.get_detail_data)
    IStrategy.advise_all_indicators = _timed("advise_all_indicators")(
        IStrategy.advise_all_indicators
    )
    IStrategy.ft_advise_signals = _timed("ft_advise_signals")(IStrategy.ft_advise_signals)
    Wallets.update = _timed("wallets_update")(Wallets.update)


# ---------------------------------------------------------------------------
# Report printers
# ---------------------------------------------------------------------------

SEP = "=" * 72


def _print_phase_report(total: float) -> None:
    print(f"\n{SEP}")
    print("PHASE TIMERS  (wall-clock, cumulative)")
    print(SEP)
    print(f"{'Phase':<35} {'Time':>9}  {'  %':>6}  {'Calls':>8}")
    print("-" * 72)
    for label, t in sorted(_phase_times.items(), key=lambda x: -x[1]):
        pct = t / total * 100 if total > 0 else 0.0
        calls = _phase_calls.get(label, 0)
        print(f"{label:<35} {t:>9.3f}s  {pct:>5.1f}%  {calls:>8,}")
    other = max(0.0, total - sum(_phase_times.values()))
    print("-" * 72)
    if other > 0.01:
        print(f"{'(untracked / overhead)':<35} {other:>9.3f}s  {other / total * 100:>5.1f}%")
    print(f"{'TOTAL':<35} {total:>9.3f}s")


def _print_cprofile_report(profiler: cProfile.Profile, top_n: int) -> None:
    # --- Top N by cumulative time ---
    print(f"\n{SEP}")
    print(f"TOP {top_n} FUNCTIONS  (sorted by cumulative time)")
    print(SEP)
    buf = io.StringIO()
    ps = pstats.Stats(profiler, stream=buf)
    ps.sort_stats("cumulative")
    ps.print_stats(top_n)
    _emit_pstats(buf.getvalue())

    # --- Top 30 by call count (reveals hot inner loops) ---
    print(f"\n{SEP}")
    print("TOP 30 FUNCTIONS  (sorted by call count)")
    print(SEP)
    buf2 = io.StringIO()
    ps2 = pstats.Stats(profiler, stream=buf2)
    ps2.sort_stats("calls")
    ps2.print_stats(30)
    _emit_pstats(buf2.getvalue())

    # --- Top 30 by tottime (own time, no children) ---
    print(f"\n{SEP}")
    print("TOP 30 FUNCTIONS  (sorted by own time / tottime)")
    print(SEP)
    buf3 = io.StringIO()
    ps3 = pstats.Stats(profiler, stream=buf3)
    ps3.sort_stats("tottime")
    ps3.print_stats(30)
    _emit_pstats(buf3.getvalue())


def _emit_pstats(raw: str) -> None:
    """Skip pstats boilerplate, keep only the data table."""
    lines = raw.split("\n")
    for i, line in enumerate(lines):
        if "ncalls" in line:
            print("\n".join(lines[i:]))
            return
    print(raw)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _pop_own_args(argv: list[str]) -> tuple[int, list[str]]:
    """Extract --top N from argv before passing the rest to freqtrade."""
    top_n = 50
    rest: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--top" and i + 1 < len(argv):
            top_n = int(argv[i + 1])
            i += 2
        else:
            rest.append(argv[i])
            i += 1
    return top_n, rest


def main() -> None:
    top_n, ft_args = _pop_own_args(sys.argv[1:])

    # Must patch BEFORE freqtrade imports load classes
    _patch_timers()

    from freqtrade.commands import Arguments
    from freqtrade.commands.optimize_commands import setup_optimize_configuration
    from freqtrade.enums import RunMode

    parsed = Arguments(["backtesting"] + ft_args).get_parsed_arg()
    config = setup_optimize_configuration(parsed, RunMode.BACKTEST)

    from freqtrade.optimize.backtesting import Backtesting

    profiler = cProfile.Profile()
    t_start = time.perf_counter()

    profiler.enable()
    backtesting = Backtesting(config)
    backtesting.start()
    profiler.disable()

    total = time.perf_counter() - t_start

    # Flush freqtrade logs so they don't interleave with our report
    sys.stdout.flush()

    _print_phase_report(total)
    _print_cprofile_report(profiler, top_n)

    print(f"\n{SEP}")
    print(f"TOTAL WALL TIME: {total:.3f}s")
    print(SEP)


if __name__ == "__main__":
    main()
