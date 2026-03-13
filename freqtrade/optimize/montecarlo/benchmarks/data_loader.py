"""
OHLCV data loader for MCPT benchmarks.

Loads real Binance data from feather files stored in user_data/data/binance/.
"""

from pathlib import Path

import pandas as pd


# Path to binance data relative to this file:
# benchmarks/ -> montecarlo/ -> optimize/ -> freqtrade/ (pkg) -> freqtrade/ (repo) -> user_data/
_REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
DATA_DIR = _REPO_ROOT / "user_data" / "data" / "binance"

# Pairs confirmed available in 5m feather format
BENCHMARK_PAIRS_5M = [
    "BTC_USDT",
    "ETH_USDT",
    "SOL_USDT",
    "BNB_USDT",
    "XRP_USDT",
    "ADA_USDT",
    "DOGE_USDT",
    "DOT_USDT",
]

# Pairs with both 1m and 5m data confirmed (for MTF benchmarks)
BENCHMARK_PAIRS_MTF = ["BTC_USDT", "SOL_USDT"]


def load_ohlcv(pair: str, timeframe: str = "5m", n_bars: int | None = None) -> pd.DataFrame:
    """
    Load OHLCV data for a single pair from feather file.

    :param pair: Pair name with underscore separator, e.g. 'BTC_USDT'
    :param timeframe: Timeframe string, e.g. '5m', '15m', '1h'
    :param n_bars: If set, return only the last n_bars rows
    :return: DataFrame with columns: date, open, high, low, close, volume
    """
    filename = f"{pair}-{timeframe}.feather"
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}\nAvailable pairs: check {DATA_DIR}")

    df = pd.read_feather(path)

    if n_bars is not None:
        actual = min(n_bars, len(df))
        df = df.tail(actual).reset_index(drop=True)

    return df


def load_multi_pair(
    pairs: list[str],
    timeframe: str = "5m",
    n_bars: int | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load OHLCV data for multiple pairs.

    Silently skips pairs whose files are not found.

    :param pairs: List of pair names with underscore separator
    :param timeframe: Timeframe string
    :param n_bars: If set, return only the last n_bars rows per pair
    :return: Dict mapping pair name to DataFrame
    """
    result: dict[str, pd.DataFrame] = {}

    for pair in pairs:
        try:
            result[pair] = load_ohlcv(pair, timeframe, n_bars)
        except FileNotFoundError:
            pass

    return result


def load_mtf_pairs(
    pairs: list[str],
    main_tf: str,
    detail_tf: str,
    n_main_bars: int,
    tf_ratio: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    Load synchronized main and detail timeframe data for MTF benchmarks.

    :param pairs: List of pair names
    :param main_tf: Main timeframe string, e.g. '5m'
    :param detail_tf: Detail timeframe string, e.g. '1m'
    :param n_main_bars: Number of bars to load for the main timeframe
    :param tf_ratio: Ratio of detail bars per main bar (e.g. 5 for 5m/1m)
    :return: Tuple of (main_data, detail_data) dicts
    """
    n_detail_bars = n_main_bars * tf_ratio
    main_data = load_multi_pair(pairs, main_tf, n_main_bars)
    detail_data = load_multi_pair(list(main_data.keys()), detail_tf, n_detail_bars)
    common = set(main_data) & set(detail_data)
    return (
        {p: main_data[p] for p in common},
        {p: detail_data[p] for p in common},
    )


def get_bar_count(pair: str, timeframe: str = "5m") -> int:
    """Return number of available bars for a pair/timeframe without loading data."""
    filename = f"{pair}-{timeframe}.feather"
    path = DATA_DIR / filename
    if not path.exists():
        return 0
    df = pd.read_feather(path, columns=["open"])
    return len(df)
