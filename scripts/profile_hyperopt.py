#!/usr/bin/env python3
"""
Hyperopt performance profiler — LLM-readable text output.

Forces -j 1 for single-process accurate profiling.
Run from freqtrade/ directory:

    python scripts/profile_hyperopt.py \\
        --config ../strategies/a-lot-of-strategies/config/config_base.json \\
        --config ../strategies/a-lot-of-strategies/pairlist/pairlist_spot.json \\
        --config ../strategies/a-lot-of-strategies/pairlist/blacklist.json \\
        --config ../strategies/a-lot-of-strategies/config/config_stake.json \\
        --user-data-dir user_data \\
        --config ../strategies/a-lot-of-strategies/strategies/e0v1e/config.json \\
        --strategy E0V1E \\
        --strategy-path ../strategies/a-lot-of-strategies/strategies/e0v1e \\
        --hyperopt-path ../strategies/a-lot-of-strategies/hyperopt_loss \\
        --timeframe-detail 1m \\
        --epochs 50 \\
        --spaces sell risk dynamic_stake time_stop \\
        --hyperopt-loss SharpeHyperOptLoss \\
        --timerange 20250901-20251205 \\
        --random-state 12345 \\
        --top 40 \\
        2>/dev/null > /tmp/ho_profile_baseline.txt

Output:
    1. PHASE TIMERS  — wall-clock per phase + call counts
    2. TOP N FUNCTIONS — cProfile sorted by cumulative / calls / tottime
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
    """Wrap a callable to record cumulative wall-clock time under *label*."""

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


def _patch_class_methods() -> None:
    """Patch class methods early — before any freqtrade instantiation."""
    from freqtrade.optimize.backtesting import Backtesting
    from freqtrade.optimize.hyperopt.hyperopt_optimizer import HyperOptimizer
    from freqtrade.strategy.interface import IStrategy
    from freqtrade.wallets import Wallets

    # Per-epoch hot path
    HyperOptimizer.generate_optimizer = _timed("generate_optimizer")(
        HyperOptimizer.generate_optimizer
    )
    Backtesting.backtest = _timed("backtest")(Backtesting.backtest)
    Backtesting._get_ohlcv_as_lists = _timed("_get_ohlcv_as_lists")(Backtesting._get_ohlcv_as_lists)
    Backtesting.get_detail_data = _timed("get_detail_data")(Backtesting.get_detail_data)

    # Startup / data-load phases
    IStrategy.advise_all_indicators = _timed("advise_all_indicators")(
        IStrategy.advise_all_indicators
    )
    IStrategy.ft_advise_signals = _timed("ft_advise_signals")(IStrategy.ft_advise_signals)
    Wallets.update = _timed("wallets_update")(Wallets.update)


def _patch_module_fns() -> None:
    """Patch module-level functions — must be called AFTER all freqtrade imports.

    Functions imported via 'from X import f' bind the name in the caller's
    module __dict__. We replace those bindings so per-epoch calls go through
    the timer wrappers.
    """
    import freqtrade.optimize.hyperopt.hyperopt_optimizer as hoo
    import freqtrade.optimize.optimize_reports.optimize_reports as orr

    # generate_strategy_stats is imported into hyperopt_optimizer's globals
    hoo.generate_strategy_stats = _timed("generate_strategy_stats")(hoo.generate_strategy_stats)

    # generate_pair_metrics / generate_tag_metrics are called from within
    # generate_strategy_stats via optimize_reports module globals
    orr.generate_pair_metrics = _timed("generate_pair_metrics")(orr.generate_pair_metrics)
    orr.generate_tag_metrics = _timed("generate_tag_metrics")(orr.generate_tag_metrics)


# ---------------------------------------------------------------------------
# Report printers
# ---------------------------------------------------------------------------

SEP = "=" * 72


def _print_phase_report(total: float, epochs: int) -> None:
    print(f"\n{SEP}")
    print("PHASE TIMERS  (wall-clock, cumulative)")
    print(SEP)
    print(f"{'Phase':<35} {'Time':>9}  {'  %':>6}  {'Calls':>8}  {'ms/call':>9}")
    print("-" * 72)
    for label, t in sorted(_phase_times.items(), key=lambda x: -x[1]):
        pct = t / total * 100 if total > 0 else 0.0
        calls = _phase_calls.get(label, 0)
        ms_per_call = (t / calls * 1000) if calls else 0.0
        print(f"{label:<35} {t:>9.3f}s  {pct:>5.1f}%  {calls:>8,}  {ms_per_call:>8.1f}ms")
    other = max(0.0, total - sum(_phase_times.values()))
    print("-" * 72)
    if other > 0.01:
        print(f"{'(untracked / overhead)':<35} {other:>9.3f}s  {other / total * 100:>5.1f}%")
    print(f"{'TOTAL':<35} {total:>9.3f}s")
    if epochs > 0:
        gen_opt_total = _phase_times.get("generate_optimizer", 0.0)
        calls = _phase_calls.get("generate_optimizer", 0)
        if calls:
            print(f"\nEffective epochs evaluated: {calls}")
            print(f"Avg time per epoch (generate_optimizer): {gen_opt_total / calls * 1000:.1f}ms")


def _print_cprofile_report(profiler: cProfile.Profile, top_n: int) -> None:
    print(f"\n{SEP}")
    print(f"TOP {top_n} FUNCTIONS  (sorted by cumulative time)")
    print(SEP)
    buf = io.StringIO()
    ps = pstats.Stats(profiler, stream=buf)
    ps.sort_stats("cumulative")
    ps.print_stats(top_n)
    _emit_pstats(buf.getvalue())

    print(f"\n{SEP}")
    print("TOP 30 FUNCTIONS  (sorted by call count)")
    print(SEP)
    buf2 = io.StringIO()
    ps2 = pstats.Stats(profiler, stream=buf2)
    ps2.sort_stats("calls")
    ps2.print_stats(30)
    _emit_pstats(buf2.getvalue())

    print(f"\n{SEP}")
    print("TOP 30 FUNCTIONS  (sorted by own time / tottime)")
    print(SEP)
    buf3 = io.StringIO()
    ps3 = pstats.Stats(profiler, stream=buf3)
    ps3.sort_stats("tottime")
    ps3.print_stats(30)
    _emit_pstats(buf3.getvalue())


def _emit_pstats(raw: str) -> None:
    lines = raw.split("\n")
    for i, line in enumerate(lines):
        if "ncalls" in line:
            print("\n".join(lines[i:]))
            return
    print(raw)


# ---------------------------------------------------------------------------
# Argument helpers
# ---------------------------------------------------------------------------


def _pop_own_args(argv: list[str]) -> tuple[int, list[str]]:
    """Extract --top N from argv; return (top_n, remaining_ft_args)."""
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


def _get_epochs(argv: list[str]) -> int:
    """Return --epochs value from argv (for display only)."""
    for i, a in enumerate(argv):
        if a in ("--epochs", "-e") and i + 1 < len(argv):
            try:
                return int(argv[i + 1])
            except ValueError:
                pass
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    top_n, ft_args = _pop_own_args(sys.argv[1:])
    epochs = _get_epochs(ft_args)

    # Step 1: patch class methods before any freqtrade instantiation
    _patch_class_methods()

    # Step 2: standard freqtrade imports (modules now cached in sys.modules)
    from freqtrade.commands import Arguments
    from freqtrade.commands.optimize_commands import setup_optimize_configuration
    from freqtrade.enums import RunMode

    # Step 3: patch module-level functions (all imports done)
    _patch_module_fns()

    # Step 4: parse args and build config
    parsed = Arguments(["hyperopt"] + ft_args).get_parsed_arg()
    config = setup_optimize_configuration(parsed, RunMode.HYPEROPT)

    # Force single-process: accurate profiling with cProfile
    config["hyperopt_jobs"] = 1

    print(
        "[profile_hyperopt] Forced hyperopt_jobs=1 for single-process profiling",
        file=sys.stderr,
    )

    from freqtrade.optimize.hyperopt import Hyperopt

    profiler = cProfile.Profile()
    t_start = time.perf_counter()

    profiler.enable()
    hyperopt = Hyperopt(config)
    hyperopt.start()
    profiler.disable()

    total = time.perf_counter() - t_start

    sys.stdout.flush()

    _print_phase_report(total, epochs)
    _print_cprofile_report(profiler, top_n)

    print(f"\n{SEP}")
    print(f"TOTAL WALL TIME: {total:.3f}s")
    print(SEP)


if __name__ == "__main__":
    main()
